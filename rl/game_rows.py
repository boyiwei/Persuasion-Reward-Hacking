"""One Old Bailey game template -> one GRPO parquet row, plus the seeded train/validation split.

The importable half of the dataset builder: `datasets/old_bailey/split_rl_sft.py` (the CLI) takes
every row-level decision from here, and `evaluation/rl_rollout.py` reuses the same constants, so a
replayed rollout sees byte-identical prompts and the same prior.

Distributions differ only in the per-row Receiver spec and the prior: bayesian (neutral juror,
prior 0.5) and stubborn (presumption of innocence + resist-updating disposition, prior 0.1).

Each row:
  prompt      : the Sender's initial chat from AgentQuality over the resolved sender prompt
                (rl/sender_prompts.py; <thinking> scratchpad then <argument>), with evidence cleaned
                of annotations (clean_evidence). The prompt is baked in, so each spec writes its own
                data root (scripts/rl_train_sender.slurm derives it from --print data-suffix).
  extra_info  : {index, split, n_pros_favoring, interaction_kwargs, payload}. interaction_kwargs
                carries a slim JSON receiver payload for rl/persuasion_interaction.py (no annotated
                evidence); the top-level `payload` adds the annotated evidences for the reward and
                monitors. JSON keeps the parquet schema homogeneous.
  agent_name  : "tool_agent" (verl ToolAgentLoop / multi-turn).
  data_source : "old_bailey_persuasion".
"""
import json
import random

from agents.agent_quality import AgentQuality
from agents.rollout import TranscriptConfig
from rl.evidence import clean_evidence, parse_strength

_TOTAL_ROUNDS = 3
_DEFAULT_PRIOR = 0.5
# Per-distribution prior P(guilty) (default 0.5). It is both the reward baseline (score = mu - prior)
# and, via _PRIOR_BELIEF_BY_DIST, the public.prior_belief prose sender and receiver read.
_PRIOR_BY_DIST = {"stubborn": 0.1}
_STUBBORN_PRIOR_BELIEF = (
    "Prior: strong presumption of innocence — only about a 10% prior probability of guilt "
    "before any evidence is examined."
)
# Per-distribution override of the template's public.prior_belief prose. None => keep neutral text.
_PRIOR_BELIEF_BY_DIST = {"stubborn": _STUBBORN_PRIOR_BELIEF}
_INTERACTION_NAME = "old_bailey_persuasion"


def _evidence_sides(game: dict) -> tuple:
    """(n_pros, n_def, n_total). An item favors prosecution iff Prosecution Strength > Defense
    Strength (strict; ties lean to neither), the rule used across the pipeline (rl/monitors.py)."""
    info = game.get("params", {}).get("private", {}).get("information") or []
    n_pros = n_def = 0
    for ev in info:
        if not ev:
            continue
        p, d = parse_strength(ev, "prosecution"), parse_strength(ev, "defense")
        if p > d:
            n_pros += 1
        elif d > p:
            n_def += 1
    return n_pros, n_def, len(info)


def _n_pros_favoring(game: dict) -> int:
    """Number of prosecution-favoring items, the per-game difficulty tag in extra_info."""
    return _evidence_sides(game)[0]


def _sender_prompt(agent: AgentQuality, public: dict, cleaned_evidence: list) -> list:
    transcript = TranscriptConfig(
        params={"public": dict(public), "private": {"information": cleaned_evidence}},
    )
    return agent.construct_messages(transcript)


def _row(game: dict, spec: dict, agent: AgentQuality, split: str,
         prior: float = _DEFAULT_PRIOR, prior_belief: str = None) -> dict:
    game_id = game["id"]
    # Copy: the CLI reuses the same `games` list for every distribution. The override reaches both
    # the sender prompt and the receiver context.
    public = dict(game["params"]["public"])
    if prior_belief is not None:
        public["prior_belief"] = prior_belief
    evidences = game["params"]["private"]["information"]
    cleaned = [clean_evidence(ev) for ev in evidences]
    # Reward/monitor payload; annotated evidence never enters the rollout context.
    payload = {
        "cognitive_model": spec,
        "public": public,
        "evidences": evidences,
        "prior": prior,
        "game_id": game_id,
    }
    # Slim payload the interaction needs to query the Receiver live during the rollout.
    recv_payload = {
        "cognitive_model": spec,
        "public": public,
        "prior": prior,
        "game_id": game_id,
    }
    return {
        "data_source": "old_bailey_persuasion",
        "agent_name": "tool_agent",
        "prompt": _sender_prompt(agent, public, cleaned),
        "reward_model": {"style": "rule", "ground_truth": ""},
        "extra_info": {
            "index": game_id,
            "split": split,
            # Difficulty tag for later bucketing; verl passes extra_info through untouched.
            "n_pros_favoring": _n_pros_favoring(game),
            "interaction_kwargs": {
                "name": _INTERACTION_NAME,
                "total_rounds": _TOTAL_ROUNDS,
                "payload": json.dumps(recv_payload),
            },
            "payload": json.dumps(payload),
        },
    }


def _spec_for(distribution: str) -> dict:
    """Per-row Receiver spec for a distribution; an unknown name raises so a typo never picks
    another juror."""
    if distribution == "bayesian":
        return {"type": "bayesian", "params": {}}
    if distribution == "stubborn":
        return {"type": "stubborn", "params": {}}
    raise ValueError(f"unknown receiver distribution {distribution!r} "
                     f"(supported: 'bayesian', 'stubborn')")


def _seeded_val_idx(n_games: int, val_size: int, seed: int) -> set:
    """Validation positions: the first `val_size` of a seeded shuffle of range(n_games). The one
    source for the split (the CLI's build() and its SFT holdout both call it)."""
    rng = random.Random(seed)
    order = list(range(n_games))
    rng.shuffle(order)
    return set(order[:val_size])
