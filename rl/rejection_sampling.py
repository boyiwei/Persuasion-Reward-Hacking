"""Pure selection logic for rejection sampling of penalty-free GRPO rollouts.

Opt-in via REJECTION_SAMPLING=1 (scripts/rl_train_sender.slurm). Per step, prompts (GRPO group =
one `uid`) with fewer than RS_TARGET_CLEAN positives among their scored rollout.n rollouts get up
to RS_MAX_GEN_ROUNDS extra full-n rounds, batched across prompts. Then exactly n rows are kept per
prompt, positives first; negatives fill the rest (a counted fallback if the target is unreachable).
The trainer loop is patch 10G in rl/patch_verl.sh, which imports this module lazily.

Positive row: penalty_total == 0, parse_failure == 0, format_reward == 1.0, monitor_failure == 0
(reward extras from rl/reward_function.py). `format_reward` is the raw 0..1 round-format fraction
(not scaled by FORMAT_REWARD_COEFF), zeroed on a forged round terminator, so empty arguments
(trivially penalty-free) and terminator forgers are rejected. `monitor_failure` flags an active
penalty monitor that returned None; without it a judge outage would read as penalty_total == 0 and
unaudited rollouts would be selected.

Do not import rl.reward_function: verl loads it by path as sys.modules["custom_module"], and a
dotted import would create a second module instance with separate state.
"""
import os
from dataclasses import dataclass

import numpy as np

POSITIVE_EPS = 1e-9
# Reward-extra keys the predicate reads; rl/reward_function.py::belief_reward always emits them.
PREDICATE_KEYS = ("penalty_total", "parse_failure", "format_reward", "monitor_failure")


@dataclass(frozen=True)
class RSConfig:
    enabled: bool
    target_clean: int       # RS_TARGET_CLEAN as configured (run-name tag uses this value)
    max_gen_rounds: int     # RS_MAX_GEN_ROUNDS: extra regeneration rounds after the initial one
    n_rollout: int          # rollout.n == kept group size K == regen chunk size
    target_effective: int   # min(target_clean, n_rollout), the applied target (SMOKE clamps here)
    budget: int             # per-prompt total rollout budget = n_rollout * (1 + max_gen_rounds)


def config_from_env(n_rollout, env=None) -> RSConfig:
    """Parse the RS knobs from the environment (or a dict, for tests).

    REJECTION_SAMPLING must be exactly "1" or "0"/unset: the launcher's submit guard, _rs name tag
    and wandb metadata gate on "1", so a lenient parse could enable RS untagged (checkpoint-clobber
    risk). Sub-knobs are validated only when enabled."""
    env = os.environ if env is None else env
    n_rollout = int(n_rollout)
    if n_rollout < 1:
        raise ValueError(f"rejection sampling: n_rollout must be >= 1 (got {n_rollout})")
    raw = env.get("REJECTION_SAMPLING", "0")
    if raw in ("0", ""):
        return RSConfig(False, 0, 0, n_rollout, 0, n_rollout)
    if raw != "1":
        raise ValueError(
            f"REJECTION_SAMPLING must be '1' (on) or '0'/unset (off), got {raw!r} -- the launcher "
            "gates on exactly '1', so any other spelling would enable rejection sampling while "
            "bypassing its submit-time guards and the _rs run-name tag")
    try:
        target = int(env.get("RS_TARGET_CLEAN", "8"))
    except ValueError as e:
        raise ValueError(f"RS_TARGET_CLEAN must be a positive integer "
                         f"(got {env.get('RS_TARGET_CLEAN')!r})") from e
    if target < 1:
        raise ValueError(f"RS_TARGET_CLEAN must be >= 1 (got {target})")
    try:
        rounds = int(env.get("RS_MAX_GEN_ROUNDS", "2"))
    except ValueError as e:
        raise ValueError(f"RS_MAX_GEN_ROUNDS must be a non-negative integer "
                         f"(got {env.get('RS_MAX_GEN_ROUNDS')!r})") from e
    if rounds < 0:
        raise ValueError(f"RS_MAX_GEN_ROUNDS must be >= 0 (got {rounds})")
    return RSConfig(True, target, rounds, n_rollout,
                    min(target, n_rollout), n_rollout * (1 + rounds))


def positive_mask(reward_extra):
    """Row-wise positive predicate over {reward-extra key -> per-sample sequence}.

    Returns a bool ndarray, or None if a key is missing, uncoercible or misaligned (the caller then
    skips rejection sampling)."""
    cols = []
    for key in PREDICATE_KEYS:
        if key not in reward_extra:
            return None
        try:
            col = np.asarray(reward_extra[key], dtype=float)
        except (TypeError, ValueError):
            return None
        if col.ndim != 1:
            return None
        cols.append(col)
    penalty_total, parse_failure, format_reward, monitor_failure = cols
    if len({len(c) for c in cols}) != 1:
        return None
    return ((np.abs(penalty_total) <= POSITIVE_EPS)
            & (parse_failure < 0.5)
            & (format_reward >= 1.0 - 1e-6)
            & (monitor_failure < 0.5))


def _group_rows(pool_uids):
    """{uid -> [row indices]} in first-appearance order; rows need not be contiguous."""
    groups = {}
    for i, uid in enumerate(pool_uids):
        groups.setdefault(uid, []).append(i)
    return groups


def step_snapshot(pool_uids, mask, cfg: RSConfig) -> dict:
    """Pre-regen stats of a scored pool: overall clean fraction + groups below the target."""
    groups = _group_rows(pool_uids)
    mask = np.asarray(mask, dtype=bool)
    below = sum(1 for rows in groups.values() if int(mask[rows].sum()) < cfg.target_effective)
    return {
        "clean_frac": float(mask.mean()) if len(mask) else 0.0,
        "prompts_below_target": float(below),
        "n_prompts": float(len(groups)),
    }


def prompts_needing_regen(prompt_uids, pool_uids, mask, cfg: RSConfig) -> dict:
    """{prompt position -> regen chunk size} for prompts below target and still under budget.

    Positions index the pre-repeat prompt rows (gen_batch / the stashed prompt remainder). Chunks
    are n rows; the min() guards a partial-budget config. An empty dict stops regeneration."""
    groups = _group_rows(pool_uids)
    mask = np.asarray(mask, dtype=bool)
    plan = {}
    for pos, uid in enumerate(prompt_uids):
        rows = groups.get(uid, [])
        attempts = len(rows)
        positives = int(mask[rows].sum()) if rows else 0
        if positives >= cfg.target_effective or attempts >= cfg.budget:
            continue
        chunk = min(cfg.n_rollout, cfg.budget - attempts)
        if chunk > 0:
            plan[pos] = chunk
    return plan


def select_kept(pool_uids, mask, cfg: RSConfig, scores=None):
    """Keep k = cfg.n_rollout rows per uid: positives first, then negatives, in generation order.

    RNG-free (rollouts are iid, so order carries no signal). Indices are sorted, so a pool that
    never regenerated selects the identity. Returns (kept_row_indices, stats); optional
    pool-aligned `scores` feed the kept-score means by predicate (0.0 for an empty side, not NaN).
    """
    groups = _group_rows(pool_uids)
    mask = np.asarray(mask, dtype=bool)
    k = cfg.n_rollout
    kept, kept_pos_rows, kept_neg_rows = [], [], []
    below_target = 0
    fallback_fill = 0
    underfilled = 0
    for rows in groups.values():
        pos = [r for r in rows if mask[r]]
        neg = [r for r in rows if not mask[r]]
        kp = pos[:k]
        kn = neg[:max(0, k - len(kp))]
        if len(kp) + len(kn) < k:  # fewer pool rows than k: keep what exists
            underfilled += 1
        if len(kp) < cfg.target_effective:
            below_target += 1
            fallback_fill += len(kn)
        kept.extend(kp + kn)
        kept_pos_rows.extend(kp)
        kept_neg_rows.extend(kn)

    def _mean_score(rows):
        if scores is None or not rows:
            return 0.0
        try:
            return float(np.mean(np.asarray(scores, dtype=float)[rows]))
        except (TypeError, ValueError):
            return 0.0

    stats = {
        "kept_total": float(len(kept)),
        "kept_positives": float(len(kept_pos_rows)),
        "kept_negatives": float(len(kept_neg_rows)),
        "prompts_below_target": float(below_target),
        "fallback_fill_count": float(fallback_fill),
        "underfilled_groups": float(underfilled),
        "kept_score_mean_pos": _mean_score(kept_pos_rows),
        "kept_score_mean_neg": _mean_score(kept_neg_rows),
    }
    return np.sort(np.asarray(kept, dtype=np.int64)), stats


def step_metrics(init: dict, sel: dict, cfg: RSConfig, *, gen_rounds, extra_rollouts,
                 pad_rollouts, chunks_dropped) -> dict:
    """Per-step rejection/ wandb scalars, added by the trainer patch to fit()'s `metrics`."""
    kept_total = max(1.0, float(sel["kept_total"]))
    return {
        "rejection/initial_clean_frac": float(init["clean_frac"]),
        "rejection/prompts_below_target_initial": float(init["prompts_below_target"]),
        "rejection/gen_rounds_used": float(gen_rounds),
        "rejection/extra_rollouts": float(extra_rollouts),
        "rejection/pad_rollouts": float(pad_rollouts),
        "rejection/chunks_dropped": float(chunks_dropped),
        "rejection/prompts_below_target_final": float(sel["prompts_below_target"]),
        "rejection/fallback_fill_count": float(sel["fallback_fill_count"]),
        "rejection/kept_positive_frac": float(sel["kept_positives"]) / kept_total,
        "rejection/kept_negatives": float(sel["kept_negatives"]),
        "rejection/kept_score_mean_pos": float(sel["kept_score_mean_pos"]),
        "rejection/kept_score_mean_neg": float(sel["kept_score_mean_neg"]),
        "rejection/target_clean_effective": float(cfg.target_effective),
    }
