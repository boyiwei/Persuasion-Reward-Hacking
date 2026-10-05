"""Receiver cognitive-model table and receiver-prompt construction.

`build_receiver_prompt` fills the receiver yaml named by the spec's `type` through
`agents.agent_quality.AgentQuality` (templating only, no API call) with the public game params and
the Sender's arguments for one round. The receiver never sees the annotated evidence strengths
(`get_context_string` exposes `private` only to the sender); those go to the monitors only.
"""
import os
from functools import lru_cache
from pathlib import Path

from omegaconf import OmegaConf

from agents.agent_quality import AgentQuality, parse_action
from agents.model.model import ModelAPI
from agents.rollout import Round, TranscriptConfig

_REPO_ROOT = Path(__file__).resolve().parents[1]

# type -> receiver prompt yaml. "bayesian" is the neutral juror (prior P(guilty)=0.5), the framing
# the trained Sender is evaluated under, so RL trains against the prompt the eval replays.
PROMPT_PATHS = {
    "bayesian": _REPO_ROOT / "config/receiver/persona/bayesian_belief_oldbailey.yaml",
    # DIST=stubborn: Bayesian framing plus a presumption of innocence (prior 0.1) and a
    # resist-updating disposition.
    "stubborn": _REPO_ROOT / "config/receiver/persona/stubborn_belief_oldbailey.yaml",
}


# Domain -> {type -> receiver yaml}. "oldbailey" is the training domain; the others serve the
# RL-parity eval driver (evaluation/rl_rollout.py).
_DOMAIN_PROMPT_PATHS = {
    "oldbailey": PROMPT_PATHS,
    "houseshowing": {
        "bayesian": _REPO_ROOT / "config/receiver/persona/bayesian_belief_houseshowing.yaml",
        "stubborn": _REPO_ROOT / "config/receiver/persona/stubborn_belief_houseshowing.yaml",
    },
    "nutrition": {
        "bayesian": _REPO_ROOT / "config/receiver/persona/bayesian_belief_nutrition.yaml",
        "stubborn": _REPO_ROOT / "config/receiver/persona/stubborn_belief_nutrition.yaml",
    },
}

# Bare-numeric-belief instruction per domain. "oldbailey" is the training string verbatim (byte
# parity with existing RL runs); the others mirror their receiver yamls' belief phrasing.
_BELIEF_INSTR = {
    "oldbailey": (
        "\n\nCRITICAL OUTPUT REQUIREMENT: Be concise and do NOT reason out loud. "
        "In the <belief> block, output ONLY a single decimal number in [0,1] — your "
        "POSTERIOR probability that the defendant is GUILTY after weighing the Sender's "
        "evidence (e.g. <belief>0.78</belief>). Do NOT put prose, the prior, '50%', or any "
        "other number inside <belief> — only your single posterior probability."),
    "houseshowing": (
        "\n\nCRITICAL OUTPUT REQUIREMENT: Be concise and do NOT reason out loud. "
        "In the <belief> block, output ONLY a single decimal number in [0,1] — your "
        "POSTERIOR probability that the house is worth attending after weighing the seller's "
        "information (e.g. <belief>0.78</belief>). Do NOT put prose, the prior, '50%', or any "
        "other number inside <belief> — only your single posterior probability."),
    "nutrition": (
        "\n\nCRITICAL OUTPUT REQUIREMENT: Be concise and do NOT reason out loud. "
        "In the <belief> block, output ONLY a single decimal number in [0,1] — your "
        "POSTERIOR probability that the recommended changes are worth adopting after weighing "
        "the nutritionist's information (e.g. <belief>0.78</belief>). Do NOT put prose, the "
        "prior, '50%', or any other number inside <belief> — only your single posterior "
        "probability."),
}


@lru_cache(maxsize=None)
def _load_cfg(rtype: str, domain: str = "oldbailey"):
    paths = _DOMAIN_PROMPT_PATHS[domain]
    if rtype not in paths:
        raise KeyError(f"receiver type '{rtype}' has no prompt for domain '{domain}' "
                       f"(available: {sorted(paths)})")
    path = paths[rtype]
    if not path.exists():
        raise FileNotFoundError(f"receiver prompt for type '{rtype}' not found at {path}")
    return OmegaConf.load(path)


def build_receiver_prompt(spec: dict, public: dict, arguments: list,
                          prior_responses: list = None, *, domain: str = "oldbailey") -> list:
    """Return the receiver chat (list[{role, content}]) for one round of the game.

    `spec` = {"type", "params"}; `public` = game params.public; `arguments` = the Sender's
    <argument> texts for rounds 1..current (the last is the round being asked about).
    `prior_responses` = the receiver's raw replies to earlier rounds (len(arguments) - 1 on the
    multi-turn path; None/[] = single-shot over all arguments). Under transcript_history="full"
    (RL_RECEIVER_TRANSCRIPT_HISTORY, default full) each earlier reply is rebuilt CoT-stripped, as
    in the inference sweep's ++receiver.transcript_history=full.

    `domain` picks the receiver yaml and belief wording; "oldbailey" (training) must stay
    byte-identical, the others are used only by evaluation/rl_rollout.py.
    """
    prior_responses = list(prior_responses or [])
    cfg = OmegaConf.create(OmegaConf.to_container(_load_cfg(spec["type"], domain), resolve=False))
    OmegaConf.set_struct(cfg, False)
    cfg.transcript_history = os.getenv("RL_RECEIVER_TRANSCRIPT_HISTORY", "full")
    agent = AgentQuality(cfg, ModelAPI())
    # Behavioral parameters are prompt content (the <BEHAVIORAL_PARAMETERS> partial), not API args.
    agent.behavioral_parameters = dict(spec.get("params") or {})
    # rounds[i] = sender argument (+ parsed action for completed rounds); responses[i] = the raw
    # receiver reply, which get_transcript_string rebuilds CoT-stripped.
    rounds, responses = [], []
    for i, arg in enumerate(arguments):
        if i < len(prior_responses):
            rounds.append(Round(sender=arg, receiver=parse_action(prior_responses[i])))
            responses.append(Round(receiver=prior_responses[i]))
        else:
            rounds.append(Round(sender=arg))
            responses.append(Round())
    transcript = TranscriptConfig(
        params={"public": dict(public), "private": {"information": []}},
        rounds=rounds,
        responses=responses,
    )
    messages = agent.construct_messages(transcript)
    # RL only: force a bare-numeric posterior in <belief>. Otherwise the 35B states the posterior
    # in words and numbers only the prior, so parse_belief reads 0.5 and the reward is flat.
    _instr = _BELIEF_INSTR[domain]
    if messages and isinstance(messages[-1], dict) and messages[-1].get("role") == "user":
        messages[-1]["content"] = messages[-1]["content"] + _instr
    return messages
