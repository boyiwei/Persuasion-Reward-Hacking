#!/usr/bin/env python3
"""RL-parity rollout driver: plays the persuasion game exactly as the verl GRPO rollout does.

GRPO runs a true multi-turn chat (assistant = the sender's full raw output incl. <thinking>,
user = the receiver's reply + round-advance instruction), unlike the agents/ pipeline's flattened
transcript. This driver replays it by importing the training code:

  * round-1 sender chat   -> AgentQuality.construct_messages over the --sender-prompt prompt
                             (rl/sender_prompts.py; same call rl/game_rows.py bakes into parquets)
  * round-advance turns   -> rl.persuasion_interaction._RECEIVER_TURN / _FALLBACK_ADVANCE
  * argument extraction   -> rl.persuasion_interaction._last_argument (CoT stripped first)
  * receiver reply squash -> rl.persuasion_interaction._receiver_context (<belief>+<argument>)
  * receiver prompt       -> rl.cognitive_models.build_receiver_prompt(..., domain=...)
  * receiver transport    -> --receiver <id> from config/receiver/model/: qwen3.5-35B served locally
                             via rl.receiver_client (RECEIVER_HOST/PORT/MODEL_ID, RL_RECV_*),
                             DeepSeek-V4-Flash via the API gateway. Resolves to the low-level
                             --receiver-api/--receiver-name pair.

Sender sampling is verl's (temperature=1.0, top_p=1.0, max_tokens=8192, enable_thinking=False).
The receiver answers every argument incl. the last (training's reward), so `responses` holds 3
receiver replies. Output is the standard result JSON (`rounds`, raw `responses`, `complete`) read
by evaluation/old_bailey|house_showing|nutrition; `params.private.information` is cleaned evidence.

Examples:
  # Old Bailey val split bound to the checkpoint's parquet (omit --val-parquet: seed-42 shuffle)
  RECEIVER_HOST=127.0.0.1 RECEIVER_PORT=20001 RECEIVER_MODEL_ID=qwen3.5-35B \
    python evaluation/rl_rollout.py --domain old-bailey --profile stubborn --split val \
      --val-parquet datasets/old_bailey/_generated/rl/stubborn/rl_validation.parquet \
      --sender-host 127.0.0.1 --sender-port 20000 --sender-name qwen3-8B-rl-stubborn \
      --out experiments/results/old-bailey/qwen3-8B-rl-stubborn/stubborn_oldbailey_rlrollout_val.json

  # hosted juror: run where the API gateway (GATEWAY_URL) is reachable; sender served on a GPU node
  # bound to 0.0.0.0 (scripts/serve_sender_xnode.slurm + scripts/hosted_juror_login_driver.sh).
  # Audit judges stay on RECEIVER_MODEL_ID (separate processes via rl.receiver_client).
  GATEWAY_URL=... GATEWAY_API_KEY=... python evaluation/rl_rollout.py --receiver DeepSeek-V4-Flash \
      --domain old-bailey --profile stubborn \
      --sender-host $NODE --sender-port 20000 --sender-name qwen3-4B-... --out ...

  # served juror (what every in-repo SLURM script selects)
  RECEIVER_HOST=127.0.0.1 RECEIVER_PORT=20001 RECEIVER_MODEL_ID=qwen3.5-35B \
    python evaluation/rl_rollout.py --receiver-api sglang --domain old-bailey ... --out ...
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from time import sleep as time_sleep
from time import time
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from openai import OpenAI  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

from agents.agent_quality import AgentQuality, parse_action  # noqa: E402
from agents.model.model import ModelAPI  # noqa: E402
from agents.rollout import TranscriptConfig  # noqa: E402

try:
    # Needs verl (repo .venv). Imported rather than copied so this tracks the training rollout.
    from rl.persuasion_interaction import (  # noqa: E402
        _FALLBACK_ADVANCE, _RECEIVER_TURN, _last_argument, _receiver_context,
    )
except ImportError as e:  # pragma: no cover - environment guard
    raise SystemExit(
        f"evaluation/rl_rollout.py needs the repo .venv (verl importable): {e}\n"
        "Run with ./.venv/bin/python (or INFER_PYTHON in the slurm drivers).") from e

from evaluation import receiver_models  # noqa: E402
from evaluation.belief import parse_belief  # noqa: E402
from rl import cognitive_models  # noqa: E402
from rl.evidence import clean_evidence  # noqa: E402
from rl.game_rows import _STUBBORN_PRIOR_BELIEF  # noqa: E402
# Mixture-aware: with RECEIVER_MIX_MODEL set it routes per game by hash (RECEIVER_MIX_FRAC/SEED);
# per-receiver evals use FRAC=0 (local) / FRAC=1 (API) with RECEIVER_MIX_MODEL still set.
# Unset => local.
from rl.receiver_client import receiver_chat  # noqa: E402
from rl.receiver_client import receiver_sampling  # noqa: E402
# Plain local call + its (host, port), so --receiver-probe checks the served endpoint without
# mixture routing.
from rl.receiver_client import chat as local_receiver_chat  # noqa: E402
from rl.receiver_client import local_endpoint  # noqa: E402
from rl.sender_prompts import (  # noqa: E402
    PromptSpecError, parse_spec, resolve_sender_prompt,
)

# Sender sampling = verl GRPO rollout defaults (rl/config/grpo_persuasion.yaml rollout section;
# enable_thinking=False from patch_verl/train slurm).
_SENDER_SAMPLING = dict(temperature=1.0, top_p=1.0, max_tokens=8192)
_SENDER_EXTRA_BODY = {"chat_template_kwargs": {"enable_thinking": False},
                      "top_k": -1, "min_p": 0.0, "repetition_penalty": 1.0}

# Hosted receiver through the API gateway, used only by --receiver-api gateway (the juror). The
# fabrication monitor (rl/monitors.py) and 42-way strategy audit (rl/strategy_audit/audit.py) are
# separate processes on rl.receiver_client.chat, so they stay on RECEIVER_MODEL_ID.
# The gateway's chat-completions URL has no default: it is read from this env var at call time.
_GATEWAY_URL_ENV = "GATEWAY_URL"
# Default gateway model and default juror of a bare invocation (every in-repo SLURM script passes
# --receiver-api sglang). Must match config/receiver/model/hosted_deepseek.yaml (checked by
# `python -m evaluation.receiver_models --selftest`).
_DEFAULT_HOSTED_RECEIVER = "DeepSeek-V4-Flash"
# Applied in main() rather than as the argparse default, so main() can reject a --receiver that
# contradicts an explicit --receiver-api.
_DEFAULT_RECEIVER_API = "gateway"

_HOSTED_TIMEOUT = 600
_HOSTED_ATTEMPTS = 6
_HOSTED_BACKOFF = 2.0     # seconds, doubling per attempt
_HOSTED_BACKOFF_CAP = 60.0


class ReceiverCallFailed(RuntimeError):
    """A hosted juror turn failed. Raised (rl.receiver_client.chat returns "" instead) so play_game
    marks the game incomplete and --skip replays it."""


def _require_gateway_key(role: str) -> str:
    """Return GATEWAY_API_KEY, or raise with the fix instead of sending `Bearer None` (a bare 401).

    Read at call time. Non-interactive drivers are where it goes missing when ~/.bashrc exports it
    only for interactive shells."""
    key = os.getenv("GATEWAY_API_KEY")
    if not key:
        raise ReceiverCallFailed(
            f"GATEWAY_API_KEY is unset or empty, so the hosted {role} cannot authenticate. "
            f"Export it before running (if ~/.bashrc guards it behind an interactive-shell check, "
            f"a batch/driver shell needs: "
            f"eval \"$(grep -m1 '^export GATEWAY_API_KEY=' ~/.bashrc)\").")
    return key


def _require_gateway_url(role: str) -> str:
    """Return GATEWAY_URL, the API gateway's chat-completions URL, or raise with the fix.

    There is no default URL. Read at call time, like the key."""
    url = os.getenv(_GATEWAY_URL_ENV)
    if not url:
        raise ReceiverCallFailed(
            f"{_GATEWAY_URL_ENV} is unset or empty, so the hosted {role} has no endpoint. "
            f"Export it before running, set to the API gateway's OpenAI-compatible "
            f"chat-completions URL.")
    return url


def _hosted_receiver_sampling() -> dict:
    """Hosted juror sampling: temperature, top_p (RL_RECV_* apply) and max_tokens=8192.

    Drops rl.receiver_client's top_k=20/min_p/presence_penalty=1.5/repetition_penalty: they are
    Qwen3.5-35B-A3B model-card values, and forwarding them to another model is a second
    uncontrolled change. chat_template_kwargs is sglang-only. Temperature stays because the
    reports' replicate/noise claims assume a sampling juror; max_tokens matches the served juror."""
    s = receiver_sampling()
    return {"temperature": s["temperature"], "top_p": s["top_p"], "max_tokens": 8192}


def _gateway_receiver_chat(model: str, messages: list) -> str:
    """One juror turn through the API gateway; raises ReceiverCallFailed on hard failure.

    Exponential backoff with jitter (rl.receiver_client's 4 unslept retries burn out in ms against a
    rate limit). Non-retryable 4xx fails fast, so a bad model id dies on game 1."""
    import requests  # noqa: PLC0415 - only the hosted branch needs it

    url = _require_gateway_url("receiver")
    body = {"model": model, "tier": os.getenv("GATEWAY_TIER", "base"), "messages": messages,
            **_hosted_receiver_sampling()}
    headers = {"Authorization": f"Bearer {_require_gateway_key('receiver')}",
               "Content-Type": "application/json"}
    delay, last = _HOSTED_BACKOFF, None
    for attempt in range(_HOSTED_ATTEMPTS):
        try:
            r = requests.post(url, headers=headers, json=body, timeout=_HOSTED_TIMEOUT)
        except Exception as e:  # noqa: BLE001 - timeouts/connection resets are retryable
            last = e
        else:
            if r.status_code < 300:
                msg = r.json()["choices"][0]["message"]
                # some hosted reasoning models leave content empty and fill reasoning_content
                return ((msg.get("content") or msg.get("reasoning_content") or "")).strip()
            if 400 <= r.status_code < 500 and r.status_code not in (408, 409, 429):
                raise ReceiverCallFailed(
                    f"non-retryable HTTP {r.status_code} from the gateway "
                    f"(model={model!r}): {r.text[:400]}")
            last = f"HTTP {r.status_code}: {r.text[:200]}"
            try:                                  # honor Retry-After when the gateway sends it
                delay = max(delay, float(r.headers.get("Retry-After", 0)))
            except (TypeError, ValueError):
                pass
        if attempt < _HOSTED_ATTEMPTS - 1:
            time_sleep(min(delay, _HOSTED_BACKOFF_CAP) * (0.5 + random.random()))
            delay = min(delay * 2, _HOSTED_BACKOFF_CAP)
    raise ReceiverCallFailed(f"{_HOSTED_ATTEMPTS} attempts exhausted (model={model!r}): {last}")

# profile -> receiver cognitive-model spec type (params={}; disposition and prior come from the
# receiver yaml and the prior_belief prose, as in training).
_PROFILE_SPEC = {"neutral": "bayesian", "stubborn": "stubborn"}


def _experiment_prior(stem: str) -> str:
    """prior_belief prose from config/experiment/<stem>.yaml (as the agents/ path injects it)."""
    return OmegaConf.load(REPO_ROOT / f"config/experiment/{stem}.yaml").prior_belief


# Domain registry. Old Bailey loads the annotated full set (the split is defined over it) and
# cleans it with clean_evidence like rl/game_rows.py; house-showing/nutrition load pre-built clean
# templates ([FEATURE]/[FACT] format that clean_evidence cannot derive). `prior_belief`: profile ->
# prose override (None = template's neutral prose); Old Bailey uses the rl/game_rows constants,
# the others their config/experiment yamls.
DOMAINS = {
    "old-bailey": dict(
        template="datasets/old_bailey/processed/full/old_bailey_revised_independent_full.json",
        cm_domain="oldbailey",
        has_split=True,
        clean=lambda info: [clean_evidence(ev) for ev in info],
        prior_belief={"neutral": None,
                      "stubborn": _STUBBORN_PRIOR_BELIEF},
    ),
    "house-showing": dict(
        template="datasets/house_showing/processed/clean/house_showing_clean.json",
        cm_domain="houseshowing",
        has_split=False,
        clean=lambda info: list(info),
        prior_belief={"neutral": None,
                      "stubborn": lambda: _experiment_prior("houseshowing_stubborn")},
    ),
    "nutrition": dict(
        template="datasets/nutrition/processed/clean/nutrition_clean.json",
        cm_domain="nutrition",
        has_split=False,
        clean=lambda info: list(info),
        prior_belief={"neutral": None,
                      "stubborn": lambda: _experiment_prior("nutrition_stubborn")},
    ),
}


def _val_ids_from_parquet(path: str) -> set:
    """Held-out val game-ids from rl_validation.parquet (`extra_info.index` == `game["id"]`).

    Drift-proof: binds the eval to the games the checkpoint actually held out. Keeps only
    `split == "validation"` rows when present, so a train parquet fails with an empty set."""
    import pandas as pd
    if not os.path.exists(path):
        raise FileNotFoundError(f"--val-parquet not found: {path}")
    df = pd.read_parquet(path)
    if "extra_info" not in df.columns:
        raise ValueError(f"{path} has no 'extra_info' column (not an rl_*.parquet?)")
    rows = [r for r in df["extra_info"] if r is not None]
    has_split = any(isinstance(r, dict) and "split" in r for r in rows)
    if has_split:
        rows = [r for r in rows if r.get("split") == "validation"]
    ids = {r["index"] for r in rows if "index" in r}
    if not ids:
        raise ValueError(f"{path} yielded no validation game-ids "
                         f"(is it rl_validation.parquet, not rl_train.parquet?)")
    return ids


def _select_split(games: list, split: str, seed: int = 42, val_size: int = 100,
                  val_ids: Optional[set] = None) -> list:
    """Pick the train/val side of the RL split.

    Prefer `val_ids` (from `_val_ids_from_parquet`). Without it, replicate the seeded shuffle of
    datasets/old_bailey/split_rl_sft.py (seed 42, val = first 100 of the zero-evidence-filtered
    list; keep in lockstep), which matches the trainer only for the canonical 1221-game build."""
    if val_ids is not None:
        if split == "val":
            return [g for g in games if g["id"] in val_ids]
        if split == "train":
            return [g for g in games if g["id"] not in val_ids]
        return list(games)
    order = list(range(len(games)))
    random.Random(seed).shuffle(order)
    val_idx = set(order[:val_size])
    if split == "val":
        return [g for i, g in enumerate(games) if i in val_idx]
    if split == "train":
        return [g for i, g in enumerate(games) if i not in val_idx]
    return list(games)


def _resolve_prior(dom: dict, profile: str):
    prior = dom["prior_belief"][profile]
    return prior() if callable(prior) else prior


def _sender_messages(agent: AgentQuality, public: dict, cleaned: list, num_rounds: int) -> list:
    # Without take_turn, construct_messages assumes total_steps=3; set it so <ROUNDS_STATEMENT>
    # matches --num-rounds.
    agent.total_steps = int(num_rounds)
    transcript = TranscriptConfig(params={"public": dict(public),
                                          "private": {"information": list(cleaned)}})
    return agent.construct_messages(transcript)


def _steer_hint(game_id, k: int, seed: int) -> tuple:
    """SFT-rollout diversity hint: k seeded legit strategies, appended to the round-1 user message.

    Deterministic in (seed, game_id), so resume-safe. Generation-only: the SFT builder rebuilds the
    prompt from the hint-free sft_holdout.parquet. Returns (hint_sentence, [slugs])."""
    from rl.strategy_audit.taxonomy import LEGIT, by_slug, slugs_by_legitimacy
    legit = slugs_by_legitimacy(LEGIT)
    # string seed: stable across processes/runs (str seeding hashes via sha512, not PYTHONHASHSEED)
    picked = random.Random(f"steer:{seed}:{game_id}").sample(legit, min(k, len(legit)))
    names = ", ".join(by_slug()[s]["name"] for s in picked)
    hint = ("\n\nFor variety, where the evidence genuinely supports it, favor these ALLOWED "
            f"strategies in this case: {names}.")
    return hint, picked


def play_game(game: dict, dom: dict, agent: AgentQuality, sender_call,
              spec: dict, prior_prose, num_rounds: int, steer: Optional[tuple] = None,
              receiver_call=None, receiver_tag: Optional[str] = None) -> dict:
    """Play one RL-faithful game; returns the game dict with rounds/responses/complete.

    `sender_call(msgs) -> str` and `receiver_call(msgs, game_id=None) -> str` are the endpoints
    built in main(); receiver_call=None uses rl.receiver_client.receiver_chat (incl.
    RECEIVER_MIX_MODEL routing by game_id). `receiver_tag` is recorded on the game.
    `steer=(k, seed)` appends `_steer_hint` to the round-1 user message."""
    if receiver_call is None:
        def receiver_call(msgs, game_id=None):
            return receiver_chat(msgs, game_id=game_id, max_tokens=8192)
    public = dict(game["params"]["public"])
    if prior_prose is not None:
        public["prior_belief"] = prior_prose
    cleaned = dom["clean"](game["params"]["private"]["information"])
    msgs = _sender_messages(agent, public, cleaned, num_rounds)
    steer_slugs = None
    if steer is not None:
        hint, steer_slugs = _steer_hint(game["id"], steer[0], steer[1])
        msgs[-1] = {"role": "user", "content": msgs[-1]["content"] + hint}

    args, prior_responses = [], []
    rounds, responses = [], []
    sender_failed = False  # a sender call exhausted its retries -> game is incomplete
    # Only the hosted receiver sets this; rl.receiver_client.chat never raises.
    receiver_failed = False
    for r in range(1, num_rounds + 1):
        resp = ""
        got_resp = False
        for _att in range(4):  # retry transient server hiccups (as in training/gen_base)
            try:
                resp = sender_call(msgs) or ""
                got_resp = True
                break
            except Exception as e:  # noqa: BLE001
                if _att == 3:
                    print(f"  [warn] sender call failed (game {game['id']} r{r}): {e}")
        if not got_resp:
            sender_failed = True
        msgs.append({"role": "assistant", "content": resp})
        args.append(_last_argument([{"role": "assistant", "content": resp}]))

        # Receiver answers every argument incl. the last (its belief is the training reward; that
        # reply never goes back to the sender).
        try:
            recv_resp = receiver_call(
                cognitive_models.build_receiver_prompt(
                    spec, public, list(args), list(prior_responses), domain=dom["cm_domain"]),
                game_id=game["id"])
        except Exception as e:  # noqa: BLE001 - hosted transport raises; sglang one never does
            print(f"  [warn] receiver call failed (game {game['id']} r{r}): {e}")
            recv_resp, receiver_failed = "", True
        if recv_resp:
            # Training appends only non-empty replies (rl/persuasion_interaction); a failed call
            # advances via _FALLBACK_ADVANCE.
            prior_responses.append(recv_resp)

        rounds.append({"sender": args[-1], "receiver": parse_action(recv_resp)})
        responses.append({"sender": resp, "receiver": recv_resp})

        if r < num_rounds:
            recv_ctx = _receiver_context(recv_resp) if recv_resp else ""
            content = (_RECEIVER_TURN.format(next_round=r + 1, total=num_rounds, receiver=recv_ctx)
                       if recv_ctx
                       else _FALLBACK_ADVANCE.format(next_round=r + 1, total=num_rounds))
            msgs.append({"role": "user", "content": content})

    out = dict(game)
    out["params"] = {"public": public, "private": {"information": list(cleaned)}}
    out["rounds"] = rounds
    out["responses"] = responses
    # Exact sender-side chat (system, user1, a1_raw, u2_advance, ...); the SFT builder reads its
    # assistant and advance turns from here.
    out["sender_messages"] = msgs
    if steer_slugs is not None:
        out["strategy_hint"] = steer_slugs
    # Which juror produced this game ("sglang:qwen3.5-35B" / "gateway:DeepSeek-V4-Flash").
    if receiver_tag is not None:
        out["receiver_model"] = receiver_tag
    # A sender failure on any round, or a hosted receiver failure, leaves the game incomplete:
    # --skip retries it and the evaluators exclude it from the scored mean.
    out["complete"] = not (sender_failed or receiver_failed)
    out["finished_time"] = time()
    return out


def _final_belief(game: dict):
    for entry in reversed(game.get("responses") or []):
        if entry.get("receiver"):
            return parse_belief(entry["receiver"])
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--domain", required=True, choices=sorted(DOMAINS))
    ap.add_argument("--profile", required=True, choices=sorted(_PROFILE_SPEC))
    ap.add_argument("--sender-host", default="127.0.0.1")
    ap.add_argument("--sender-port", default=None,
                    help="port of the served SGLang sender (required unless --sender-api gateway)")
    ap.add_argument("--sender-name", required=True, help="served-model-name of the sender policy")
    ap.add_argument("--sender-prompt", default="base", metavar="SPEC",
                    help="which sender prompt to play: base (default) | strategies (the "
                         "42-technique ALLOWED/FORBIDDEN guide) | single_strategy:<slug> (one "
                         "allowed technique); rendered for --domain by rl/sender_prompts.py. It "
                         "is the ONE seam between prompt arms: receiver prompt/_instr, samplings, "
                         "multi-turn chat and split stay byte-identical, so two runs differing "
                         "only here isolate the sender-prompt effect. The resolved spec is "
                         "recorded on every game as `sender_prompt`.")
    ap.add_argument("--sender-api", choices=["sglang", "gateway"], default="sglang",
                    help="sender endpoint: 'sglang' (default; the served policy at verl sampling) "
                         "or 'gateway' (hosted sender via the API gateway, GATEWAY_URL and "
                         "GATEWAY_API_KEY; no local sender server, so run it from a host that can "
                         "reach the gateway)")
    ap.add_argument("--receiver", default=None, metavar="ID",
                    help="which juror plays the games, by model id from config/receiver/model/ "
                         "(`python -m evaluation.receiver_models --list`): qwen3.5-35B is served "
                         "locally with SGLang, DeepSeek-V4-Flash is queried through the API "
                         "gateway from a host that can reach it. This is the one documented way to "
                         "pick a juror; it resolves to the --receiver-api/--receiver-name pair "
                         "below, which stay the low-level seam. Giving both is an error when they "
                         "disagree.")
    ap.add_argument("--receiver-api", choices=["sglang", "gateway"], default=None,
                    help=f"juror endpoint (the seam --receiver resolves to). Default "
                         f"'{_DEFAULT_RECEIVER_API}': the hosted juror "
                         f"({_DEFAULT_HOSTED_RECEIVER}) via the API gateway, GATEWAY_URL and "
                         f"GATEWAY_API_KEY. A job on a host that cannot reach the gateway (such "
                         f"as a compute node without outbound access) must either pass "
                         f"--receiver-api sglang or serve its sender cross-node and drive the "
                         f"games from a host that can (scripts/serve_sender_xnode.slurm + a "
                         f"login driver). 'sglang' uses rl.receiver_client at the training "
                         f"transport/sampling via RECEIVER_HOST/RECEIVER_PORT/RECEIVER_MODEL_ID. "
                         f"This switches the receiver only: the fabrication and 42-way strategy "
                         f"audits are separate processes that go through rl.receiver_client, so "
                         f"they stay on RECEIVER_MODEL_ID and are unaffected by this flag.")
    ap.add_argument("--receiver-name", default=None,
                    help=f"gateway: the gateway model id, exact case (default "
                         f"{_DEFAULT_HOSTED_RECEIVER}; the gateway rejects the lowercase form). "
                         f"sglang: optional override of RECEIVER_MODEL_ID, which "
                         f"rl/receiver_client.py freezes at import time. It is left as None here "
                         f"so the hosted default can never leak in as a served-model name.")
    ap.add_argument("--receiver-probe", action=argparse.BooleanOptionalAction, default=True,
                    help="issue one trivial juror call before playing any game and exit non-zero "
                         "if it fails (default on). On gateway it catches a bad model id / "
                         "missing key or URL / unreachable gateway for a fraction of a cent; on "
                         "sglang it catches an endpoint with nothing served, where "
                         "rl.receiver_client returns '' per call and every game would otherwise be "
                         "written complete with empty juror turns. Either way it costs one call "
                         "instead of a run's worth of sender GPU-hours or paid sender calls.")
    ap.add_argument("--out", required=True, help="result JSON path (standard result schema)")
    ap.add_argument("--num-rounds", type=int, default=3)
    ap.add_argument("--end-idx", type=int, default=None,
                    help="cap on games AFTER split selection (default: all)")
    ap.add_argument("--max-workers", type=int, default=16)
    ap.add_argument("--save-frequency", type=int, default=10)
    ap.add_argument("--skip", action=argparse.BooleanOptionalAction, default=True,
                    help="resume: skip games already complete in --out (default on)")
    ap.add_argument("--split", choices=["train", "val", "all"], default="val",
                    help="old-bailey only: which side of the RL split (default val)")
    ap.add_argument("--val-parquet", default=None,
                    help="old-bailey only: read the val/train membership from this checkpoint's "
                         "rl_validation.parquet (extra_info.index) instead of re-deriving the "
                         "seed-42 shuffle — binds the eval to the exact games the checkpoint held "
                         "out (drift-proof). Omit to fall back to the seeded shuffle.")
    ap.add_argument("--ids-file", default=None,
                    help="restrict to these game ids: a JSON file holding a bare id list or a dict "
                         "with an id list under --ids-key (e.g. the sft_split.json written by "
                         "split_rl_sft.py --sft-holdout). Applied AFTER --split/--val-parquet "
                         "and BEFORE --end-idx; --skip resume is id-keyed and unaffected.")
    ap.add_argument("--ids-key", default="sft_train_ids",
                    help="key of the id list when --ids-file holds a dict (default sft_train_ids)")
    ap.add_argument("--steer-strategies", type=int, default=0, metavar="K",
                    help="diversity steering for SFT-rollout generation: append a per-game seeded "
                         "'favor these ALLOWED strategies' hint (K legit slugs from the audit "
                         "taxonomy) to the round-1 sender message. Generation-only; recorded as "
                         "game['strategy_hint']. 0 = off (default).")
    ap.add_argument("--steer-seed", type=int, default=0,
                    help="seed for --steer-strategies (vary per pass for different strategy mixes)")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the round-1 sender chat + receiver prompt for the first game, no HTTP")
    args = ap.parse_args(argv)

    # Resolve --receiver into --receiver-api/--receiver-name and refuse a disagreeing pair, which
    # would otherwise play one model while the logs name another.
    if args.receiver is not None:
        try:
            wanted = receiver_models.rollout_args(args.receiver)
        except receiver_models.ReceiverModelError as e:  # an id typo is a usage error
            ap.error(str(e))
        wanted = dict(zip(wanted[::2], wanted[1::2]))
        wanted_api = wanted["--receiver-api"]
        wanted_name = wanted.get("--receiver-name")
        if args.receiver_api is not None and args.receiver_api != wanted_api:
            ap.error(f"--receiver {args.receiver} is reached with --receiver-api {wanted_api}, "
                     f"not {args.receiver_api}")
        if (wanted_name is not None and args.receiver_name is not None
                and args.receiver_name != wanted_name):
            ap.error(f"--receiver {args.receiver} is the gateway model {wanted_name!r}, "
                     f"not {args.receiver_name!r}")
        args.receiver_api = wanted_api
        if wanted_name is not None:
            args.receiver_name = wanted_name
    if args.receiver_api is None:
        args.receiver_api = _DEFAULT_RECEIVER_API

    dom = DOMAINS[args.domain]
    spec = {"type": _PROFILE_SPEC[args.profile], "params": {}}
    prior_prose = _resolve_prior(dom, args.profile)
    # Parse the sender prompt spec before any GPU or paid call.
    try:
        prompt_spec = parse_spec(args.sender_prompt)
    except PromptSpecError as e:  # a spec typo is a usage error, not a crash
        ap.error(str(e))
    sender_prompt = str(prompt_spec)

    raw = json.load(open(REPO_ROOT / dom["template"]))
    # Drop zero-evidence games first, as datasets/old_bailey/split_rl_sft.py does (split indices
    # are over the filtered list).
    games = [g for g in raw if (g.get("params", {}).get("private") or {}).get("information")]
    if dom["has_split"]:
        val_ids = _val_ids_from_parquet(args.val_parquet) if args.val_parquet else None
        if val_ids is not None:
            print(f"[rl_rollout] split from parquet: {len(val_ids)} held-out val ids "
                  f"({args.val_parquet})")
        games = _select_split(games, args.split, val_ids=val_ids)
    elif args.val_parquet:
        print(f"[rl_rollout] note: --val-parquet is old-bailey-only; ignoring for {args.domain}")
    elif args.split != "val":
        print(f"[rl_rollout] note: --split is old-bailey-only; ignoring for {args.domain}")
    if args.ids_file:
        ids = json.load(open(args.ids_file))
        if isinstance(ids, dict):
            ids = ids[args.ids_key]
        ids = set(ids)
        games = [g for g in games if g["id"] in ids]
        print(f"[rl_rollout] --ids-file: {len(games)} games selected (of {len(ids)} ids in "
              f"{args.ids_file})")
        if not games:
            raise SystemExit("[rl_rollout] --ids-file selected 0 games (wrong --split side, or "
                             "ids not in this template?)")
    if args.end_idx is not None:
        games = games[:args.end_idx]
    steer = (args.steer_strategies, args.steer_seed) if args.steer_strategies > 0 else None
    if steer:
        print(f"[rl_rollout] diversity steering ON: {args.steer_strategies} legit slugs/game "
              f"(steer-seed {args.steer_seed})")

    agent = AgentQuality(resolve_sender_prompt(args.domain, prompt_spec), ModelAPI())

    if args.dry_run:
        g = games[0]
        public = dict(g["params"]["public"])
        if prior_prose is not None:
            public["prior_belief"] = prior_prose
        cleaned = dom["clean"](g["params"]["private"]["information"])
        sender_msgs = _sender_messages(agent, public, cleaned, args.num_rounds)
        if steer:
            hint, hint_slugs = _steer_hint(g["id"], steer[0], steer[1])
            sender_msgs[-1] = {"role": "user", "content": sender_msgs[-1]["content"] + hint}
            print(f"[rl_rollout] dry-run strategy_hint: {hint_slugs}")
        receiver_msgs = cognitive_models.build_receiver_prompt(
            spec, public, ["<argument placeholder>"], [], domain=dom["cm_domain"])
        print(json.dumps({"game_id": g["id"], "n_games": len(games),
                          "sender_messages": sender_msgs,
                          "receiver_messages": receiver_msgs}, indent=2, ensure_ascii=False))
        return

    # Resume overlay: reuse completed games from a previous --out (id-keyed).
    done = {}
    if args.skip and os.path.exists(args.out):
        try:
            done = {g["id"]: g for g in json.load(open(args.out)) if g.get("complete")}
            print(f"[rl_rollout] resume: {len(done)} complete games loaded from {args.out}")
        except Exception as e:  # noqa: BLE001
            print(f"[rl_rollout] resume load failed ({e}); starting fresh")

    # sender_call(msgs) -> str: served policy or hosted sender.
    if args.sender_api == "gateway":
        # Hosted sender (cf. agents/model/openai_llm.GatewayChatModel). temp/top_p mirror verl; no
        # max_tokens (gpt-5.x reasoning models reject it) and no SGLang-only extra_body.
        import requests  # noqa: PLC0415 - only the gateway branch needs it

        def sender_call(msgs):
            r = requests.post(
                _require_gateway_url("sender"),
                headers={"Authorization": f"Bearer {_require_gateway_key('sender')}",
                         "Content-Type": "application/json"},
                json={"model": args.sender_name, "tier": os.getenv("GATEWAY_TIER", "base"),
                      "messages": msgs, "temperature": 1.0, "top_p": 1.0},
                timeout=600,
            )
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
    else:
        if not args.sender_port:
            ap.error("--sender-port is required with --sender-api sglang")
        sender_client = OpenAI(base_url=f"http://{args.sender_host}:{args.sender_port}/v1",
                               api_key="None")

        def sender_call(msgs):
            return sender_client.chat.completions.create(
                model=args.sender_name, messages=msgs,
                extra_body=_SENDER_EXTRA_BODY, **_SENDER_SAMPLING,
            ).choices[0].message.content

    # receiver_call(msgs) -> str: juror only. Built here instead of repointing rl.receiver_client,
    # which also carries the fabrication monitor and strategy audit; repointing it would silently
    # swap both audit judges too.
    if args.receiver_api == "gateway":
        # Defaulted here, not in argparse, so the hosted id never reaches the sglang branch as a
        # served-model name.
        hosted_receiver = args.receiver_name or _DEFAULT_HOSTED_RECEIVER
        receiver_tag = f"gateway:{hosted_receiver}"

        def receiver_call(msgs, game_id=None):  # game_id unused: all games go to the hosted juror
            return _gateway_receiver_chat(hosted_receiver, msgs)

        if args.receiver_probe:
            # A failed probe is a config error (e.g. an unreachable gateway or a missing key): exit
            # via ap.error naming the fixes rather than with a traceback.
            try:
                probe = receiver_call([{"role": "user", "content": "Reply with exactly: OK"}])
            except Exception as e:  # noqa: BLE001 - any probe failure is a config error here
                ap.error(
                    f"hosted juror probe failed for {hosted_receiver!r}: {e}\n"
                    f"  The hosted juror is the default (--receiver-api gateway) and needs a "
                    f"host that can reach the API gateway at GATEWAY_URL.\n"
                    f"  -> on a host that cannot reach it, pass --receiver-api sglang (with the "
                    f"juror served locally via RECEIVER_HOST/RECEIVER_PORT/RECEIVER_MODEL_ID);\n"
                    f"  -> on a host that can, export GATEWAY_URL and GATEWAY_API_KEY "
                    f"(eval \"$(grep -m1 '^export GATEWAY_API_KEY=' ~/.bashrc)\");\n"
                    f"  -> or skip the check with --no-receiver-probe (games will then fail "
                    f"one by one instead of up front).")
            print(f"[rl_rollout] receiver probe OK ({hosted_receiver}): {probe[:60]!r}")
    else:
        _rid = args.receiver_name or os.getenv("RECEIVER_MODEL_ID", "qwen3.5-35B")
        _rhost, _rport = local_endpoint()
        receiver_tag = f"sglang:{_rid}"

        def receiver_call(msgs, game_id=None):
            return receiver_chat(msgs, game_id=game_id, model_id=args.receiver_name,
                                 max_tokens=8192)

        if args.receiver_probe:
            # rl.receiver_client.chat returns "" on any error, so a wrong endpoint would write every
            # game complete with empty juror turns. Probes the served endpoint via chat(), not
            # receiver_call, to bypass RECEIVER_MIX_* routing.
            if not (local_receiver_chat([{"role": "user", "content": "Reply with exactly: OK"}],
                                        model_id=args.receiver_name, max_tokens=64) or "").strip():
                ap.error(
                    f"served juror probe got an EMPTY reply for {_rid!r} at {_rhost}:{_rport}.\n"
                    f"  rl.receiver_client returns '' for every failure, so this is what an "
                    f"unreachable endpoint, a dead server or a wrong served-model name all look "
                    f"like.\n"
                    f"  -> serve the juror and point RECEIVER_HOST/RECEIVER_PORT/"
                    f"RECEIVER_MODEL_ID at it (scripts/*_rl_eval.slurm co-serve it beside the "
                    f"sender; scripts/serve_receiver_xnode.slurm serves it on its own node);\n"
                    f"  -> or skip the check with --no-receiver-probe (every game would then be "
                    f"played against a juror that never answers).")
            print(f"[rl_rollout] receiver probe OK ({_rid} @ {_rhost}:{_rport})")
    experiment_data = [done.get(g["id"], g) for g in games]
    # Stamp the sender prompt on every game, --skip carry-overs included. Done here, not in
    # play_game (whose signature the SFT dataset builder mirrors); play_game copies the dict.
    for g in experiment_data:
        g["sender_prompt"] = sender_prompt
    todo = [i for i, g in enumerate(experiment_data) if not g.get("complete")]
    print(f"[rl_rollout] {args.domain}/{args.profile}: {len(games)} games in scope, "
          f"{len(todo)} to play (rounds={args.num_rounds}, sender={args.sender_name}, "
          f"prompt={sender_prompt}, receiver={receiver_tag})")
    # With --receiver-api gateway, RECEIVER_MODEL_ID governs only the later audit judges; logging it
    # makes an accidental judge swap visible.
    print(f"[rl_rollout] audit-judge env: RECEIVER_MODEL_ID="
          f"{os.getenv('RECEIVER_MODEL_ID', 'qwen3.5-35B')} @ "
          f"{os.getenv('RECEIVER_HOST', '127.0.0.1')}:{os.getenv('RECEIVER_PORT', '30001')} "
          f"(used by the fabrication + strategy audits, NOT by this rollout's juror)")

    save_lock = threading.Lock()
    completed = 0

    def _save():
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        try:
            with open(args.out, "w") as f:
                json.dump(experiment_data, f, indent=4)
        except Exception as e:  # noqa: BLE001
            print(f"[rl_rollout] checkpoint save failed: {e}")
            with open(args.out + ".backup", "w") as f:
                json.dump(experiment_data, f, indent=4)

    def work(i):
        return i, play_game(experiment_data[i], dom, agent, sender_call,
                            spec, prior_prose, args.num_rounds, steer=steer,
                            receiver_call=receiver_call, receiver_tag=receiver_tag)

    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = [ex.submit(work, i) for i in todo]
        for fut in as_completed(futures):
            try:
                i, result = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] game failed: {e}")
                continue
            with save_lock:
                experiment_data[i] = result
                completed += 1
                if completed % args.save_frequency == 0:
                    _save()
                    beliefs = [b for b in (_final_belief(g) for g in experiment_data
                                           if g.get("complete")) if b is not None]
                    mean_b = sum(beliefs) / len(beliefs) if beliefs else float("nan")
                    print(f"  ... {completed}/{len(todo)} games "
                          f"(mean final belief {mean_b:.3f}, n={len(beliefs)})", flush=True)
    with save_lock:
        _save()
    n_complete = sum(1 for g in experiment_data if g.get("complete"))
    print(f"[rl_rollout] done: {n_complete}/{len(experiment_data)} complete -> {args.out}")


if __name__ == "__main__":
    main()
