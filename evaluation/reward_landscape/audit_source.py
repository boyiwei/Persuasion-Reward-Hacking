#!/usr/bin/env python3
"""Where a cell's files live, and the guard on the strategy audit's provenance.

Owns the experiment's naming rule in both directions, built on `rl.sender_prompts`:

    forward   (sender dir, juror profile, prompt spec) -> the rollout result path
    backward  a result stem                            -> the manifest / audit key

`result_stem(domain, spec)` is a result file's sender-prompt token and `arm_key(spec)` the arm in a
key (`base` | `strat` | `only_<slug>`), so build_manifest.py and the analyzer agree on cells and a
prompt spec (which may contain ':') never reaches a path.

`load_audit_for_stem` is the only way the analysis opens a strategy audit. It refuses an artifact
missing POSTFIX_KEYS: that is another instrument (binary judge `false_information` instead of the
genuine-fabrication indicator, grounded slugs judged at a smaller budget).

Roots are resolved on call, not at import, so RESULTS_ROOT / AUDIT_DIR and --results-root /
--audit-dir mean the same wherever they are set.
"""
import functools
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

# One domain and one served juror: the correlation is with that juror's belief, so neither is a
# knob.
DOMAIN = "old-bailey"
JUROR = "qwen3.5-35B"

# Provenance keys this experiment's instrument writes; their absence marks another instrument.
# All are checked so a partially-stamped artifact also fails.
POSTFIX_KEYS = ("judge_budget_policy", "judge_budget_grounded_slugs", "false_information_source")

_STEM_MID = "_oldbailey_rlrollout_val_sender-"
_RECV_SUFFIX = f"__recv_{JUROR}"


# Roots
def default_results_root() -> str:
    """The one root every input and output of the package lives under ($RESULTS_ROOT)."""
    return os.getenv("RESULTS_ROOT") or str(_REPO / "experiments" / "results")


def old_bailey_dir(results_root=None) -> str:
    """Where the rollout driver writes `<sender dir>/<result>.json`."""
    return os.path.join(results_root or default_results_root(), "old-bailey")


def default_audit_dir(results_root=None) -> str:
    """The `by_result/` directory reaudit_all.slurm publishes ($AUDIT_DIR)."""
    return os.getenv("AUDIT_DIR") or os.path.join(
        old_bailey_dir(results_root), "_reward_landscape_audit", "final", "by_result")


def default_out_dir(results_root=None) -> str:
    """Where the manifest, the stats json and the figure are written."""
    return os.path.join(results_root or default_results_root(), "reward_landscape")


# The naming rule
@functools.lru_cache(maxsize=None)
def _arm_of_stem() -> dict:
    """result-stem token -> arm key, for every prompt arm of the domain."""
    from rl.sender_prompts import all_specs, arm_key, result_stem
    return {result_stem(DOMAIN, spec): arm_key(spec) for spec in all_specs(DOMAIN)}


def result_name(profile: str, spec) -> str:
    """The result file name of one cell (the name the rollout driver writes)."""
    from rl.sender_prompts import result_stem
    return f"{profile}{_STEM_MID}{result_stem(DOMAIN, spec)}{_RECV_SUFFIX}.json"


def result_path(sender_dir: str, profile: str, spec, results_root=None) -> str:
    """The full path of one cell's rollout transcript."""
    return os.path.join(old_bailey_dir(results_root), sender_dir, result_name(profile, spec))


def key_for_stem(stem: str) -> str:
    """`<OB>/<sender dir>/<profile>_..._sender-<stem>__recv_<juror>` -> `<sender>__<profile>__<arm>`.

    The inverse of `result_path`.
    """
    sender_dir = os.path.basename(os.path.dirname(stem))
    fname = os.path.basename(stem)
    profile, _, rest = fname.partition(_STEM_MID)
    if not rest or not rest.endswith(_RECV_SUFFIX):
        raise ValueError(f"unrecognised result stem: {stem}")
    sender_stem = rest[: -len(_RECV_SUFFIX)]
    arm = _arm_of_stem().get(sender_stem)
    if arm is None:
        raise ValueError(f"unrecognised sender prompt {sender_stem!r} in stem {stem}")
    return f"{sender_dir}__{profile}__{arm}"


def key_for_result(result_path_: str) -> str:
    """Same as `key_for_stem` for a path that still carries its `.json` suffix."""
    return key_for_stem(result_path_[:-5] if result_path_.endswith(".json") else result_path_)


# Reading an audit
def audit_path_for_stem(stem: str, audit_dir=None) -> str:
    return os.path.join(audit_dir or default_audit_dir(),
                        key_for_stem(stem) + ".strategy_audit.json")


def fabrication_path_for_stem(stem: str, audit_dir=None) -> str:
    return os.path.join(audit_dir or default_audit_dir(),
                        key_for_stem(stem) + ".fabrication.json")


def load_audit_for_stem(stem: str, audit_dir=None):
    """The strategy audit of a cell, or None if absent. Raises on another instrument's artifact."""
    p = audit_path_for_stem(stem, audit_dir)
    if not os.path.exists(p):
        return None
    with open(p) as fh:
        d = json.load(fh)
    missing = [k for k in POSTFIX_KEYS if k not in d]
    if missing:
        raise RuntimeError(
            f"{p} is a strategy audit from another instrument (missing {missing}). Its "
            "false_information column is a binary judge verdict, not the genuine-fabrication "
            "indicator, and its grounded slugs were judged at 256 tokens. Re-run the audit "
            "rather than reading it.")
    return d
