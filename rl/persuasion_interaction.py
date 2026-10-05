"""verl multi-turn interaction that plays the real Sender <-> Receiver game inside the rollout.

Registered in rl/config/interaction_config.yaml as "old_bailey_persuasion". After each Sender
<argument> it queries the Receiver (the served 35B, via rl.receiver_client) and injects the
CoT-stripped reply, wrapped in <receiver_response> so the reward function can strip it before the
monitors read the Sender's <argument>s, as the next round's context.

verl multi_turn must be max_assistant_turns=4, max_user_turns=3, one above the 3 game rounds: the
receiver re-entry gate `current_turns < max_assistant_turns` is evaluated at current_turns=3 after
arg3. The interaction terminates on its 3rd call, so the Sender still emits exactly 3 arguments:
    arg1 -> recv1  (injected as round-2 context)
    arg2 -> recv2  (injected as round-3 context)
    arg3 -> recv3  (not injected; its belief is the reward)

Each call returns the Receiver's parsed P(guilty) as turn_score. verl records it (the
user_turn_rewards.append in _async_rollout_a_request) before the terminate check, and it reaches
rl/reward_function.py via extra_info["rollout_reward_scores"]["user_turn_rewards"], whose last
value is the final mu, with no re-query. A failed or empty receiver call gives a plain round-advance
and a -1.0 turn_score (the reward function then re-queries). Every round asks for a <thinking> block before the <argument>
(checked by the format bonus), so the CoT is in every dumped round. The blocking HTTP call runs in
a thread executor; verl masks the injected user tokens out of the loss.
"""
import asyncio
import json
import re
from uuid import uuid4

from verl.interactions.base import BaseInteraction

from agents.agent_quality import _last_closed_block
from evaluation.belief import parse_belief
from rl import cognitive_models, rollout_seed
from rl.receiver_client import receiver_chat
from rl.reward_function import _strip_thinking

# Sender <argument> from the latest assistant message: closed block, then open-tag-to-boundary,
# then thinking-stripped fallback (mirror reward_function._extract_arguments for a single turn).
_ARG_RE = re.compile(r"<argument>(.*?)</argument>", re.IGNORECASE | re.DOTALL)
_ARG_OPEN_RE = re.compile(
    r"<argument>(.*?)(?=<argument>|</?think(?:ing)?>|<\|im_end\|>|\Z)",
    re.IGNORECASE | re.DOTALL,
)
# Strip <think>/<thinking> CoT: closed pairs and an unterminated open tag to end-of-text, so a
# sender turn truncated mid-<thinking> (8192-tok cap) never leaks its reasoning to the receiver.
_THINK_RE = re.compile(r"<think(?:ing)?>.*?(?:</think(?:ing)?>|\Z)", re.IGNORECASE | re.DOTALL)

# Name the open tags <thinking> / <argument> (they tell the policy the format) but never embed a
# closed `<x>...</x>` example: a closed <argument> literal survives
# reward_function._strip_receiver_turns and was extracted as a phantom "..." argument (n_args 3 -> 5),
# and a closed <thinking> pair would phantom-satisfy the turn-format check.
# Invariant: both templates must end with "wrapped in a single <argument> block."; the
# reward_function._ADVANCE_RE terminator splits the trajectory into rounds on it. Changing the
# phrase requires that regex to match old and new wording, or re-scoring stored rollouts breaks.
_RECEIVER_TURN = (
    "Round {next_round} of {total}. The Receiver responded to your argument:\n"
    "<receiver_response>\n{receiver}\n</receiver_response>\n"
    "First explain your reasoning in a <thinking> block, then present your next argument to "
    "further influence the Receiver's belief, wrapped in a single <argument> block."
)
_FALLBACK_ADVANCE = (
    "Round {next_round} of {total}. The Receiver has noted your argument. First explain your "
    "reasoning in a <thinking> block, then present your next argument to further influence the "
    "Receiver's belief, wrapped in a single <argument> block."
)


def _msg_field(m, key):
    return m.get(key) if isinstance(m, dict) else getattr(m, key, None)


def _last_argument(messages) -> str:
    """The Sender's <argument> from the latest assistant message ('' if none parseable).

    CoT is stripped first (reward_function._strip_thinking), so an <argument> token inside
    <thinking> cannot leak the sender's reasoning to the receiver. An unterminated <thinking>
    followed by a closed <argument> keeps that argument."""
    content = ""
    for m in reversed(messages):
        if _msg_field(m, "role") == "assistant":
            content = _msg_field(m, "content") or ""
            break
    if not content:
        return ""
    text = content.replace("[argument]", "<argument>").replace("[/argument]", "</argument>")
    text = _strip_thinking(text)
    for rx in (_ARG_RE, _ARG_OPEN_RE):
        blocks = [b.strip() for b in rx.findall(text) if b.strip()]
        if blocks:
            return blocks[-1]
    return text.strip()


def _receiver_context(resp: str) -> str:
    """The receiver's reply reduced to its <belief> + <argument> blocks for the sender's next
    round; CoT is stripped and the <action> verdict deliberately dropped."""
    if not resp:
        return ""
    text = _THINK_RE.sub("", resp)
    parts = []
    belief = _last_closed_block(text, "belief")
    if belief is not None:
        parts.append(f"<belief>\n{belief}\n</belief>")
    argument = _last_closed_block(text, "argument")
    if argument is not None:
        parts.append(f"<argument>\n{argument}\n</argument>")
    return "\n".join(parts).strip()

class OldBaileyPersuasionInteraction(BaseInteraction):
    def __init__(self, config: dict):
        super().__init__(config)
        self._instances = {}

    def _new_state(self, kwargs):
        payload = kwargs.get("payload")
        kw = json.loads(payload) if payload else {}
        return {
            "turn": 0,
            "total_rounds": int(kwargs.get("total_rounds", 3)),
            "spec": kw.get("cognitive_model", {"type": "bayesian", "params": {}}),
            "public": kw.get("public", {}),
            "prior": float(kw.get("prior", 0.5)),
            "game_id": kw.get("game_id", 0),
            # Mixture of receivers: stamped by the 10H patch beside payload in interaction_kwargs.
            # None (validation, eval, feature off) -> receiver_client.resolve_backend fallback.
            "receiver_backend": kwargs.get("receiver_backend"),
            # Opt-in seed from the rollout-seed verl patch; each receiver round derives its own.
            "rollout_sampling_seed": (
                int(kwargs["rollout_sampling_seed"])
                if kwargs.get("rollout_sampling_seed") is not None else None
            ),
            "arguments": [],
            "prior_responses": [],
        }

    async def start_interaction(self, instance_id: str = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instances[instance_id] = self._new_state(kwargs)
        return instance_id

    async def generate_response(self, instance_id: str, messages, **kwargs):
        """Return (should_terminate, response_text, turn_score, metadata).

        turn_score is the Receiver's parsed P(guilty) for this round (-1.0 on failure); the reward
        function reads the last one as the final mu.
        """
        state = self._instances.get(instance_id)
        if state is None:  # start_interaction missed (defensive)
            state = self._instances[instance_id] = self._new_state(kwargs)
        state["turn"] += 1
        total = state["total_rounds"]

        # Keep arguments aligned with the round index (append even an empty parse).
        state["arguments"].append(_last_argument(messages))

        spec, public = state["spec"], state["public"]
        args = list(state["arguments"])
        prior = list(state["prior_responses"])  # responses to rounds strictly before this one

        receiver_seed = None
        if state["rollout_sampling_seed"] is not None:
            receiver_seed = rollout_seed.derive_receiver_turn_seed(
                rollout_seed=state["rollout_sampling_seed"], turn_index=state["turn"] - 1
            )

        receiver_seed_kwargs = {"seed": receiver_seed} if receiver_seed is not None else {}

        def _query():
            prompt = cognitive_models.build_receiver_prompt(spec, public, args, prior)
            # Local server, or the hosted arm named by this game's 10H receiver_backend stamp.
            return receiver_chat(
                prompt, game_id=state.get("game_id"),
                backend=state.get("receiver_backend"), max_tokens=8192, **receiver_seed_kwargs
            )

        mu, receiver_reply = -1.0, ""
        try:
            resp = await asyncio.get_running_loop().run_in_executor(None, _query)
        except Exception as e:  # noqa: BLE001 - a single rollout must never crash training
            print(f"[interaction] receiver call failed (game {state.get('game_id')}): {e}")
            resp = ""
        if resp:
            state["prior_responses"].append(resp)
            receiver_reply = _receiver_context(resp)
            b = parse_belief(resp)
            if b is not None:
                mu = float(b)

        if state["turn"] >= total:
            return True, "", mu, {}
        next_round = state["turn"] + 1
        content = (_RECEIVER_TURN.format(next_round=next_round, total=total, receiver=receiver_reply)
                   if receiver_reply
                   else _FALLBACK_ADVANCE.format(next_round=next_round, total=total))
        return False, content, mu, {}

    async def calculate_score(self, instance_id: str, **kwargs) -> float:
        # Reward is produced by rl.reward_function.belief_reward from the turn-score channel.
        return 0.0

    async def finalize_interaction(self, instance_id: str, **kwargs) -> None:
        self._instances.pop(instance_id, None)
