"""Pure scalar logic for balancing auxiliary CE against the GRPO gradient.

The torch/verl integration is in rl/patch_verl.sh; this module is testable without a GPU or verl.

``AUX_CE_BALANCE_MODE=off`` keeps the fixed-coefficient update. In ``norm_ratio`` mode
``AUX_CE_COEFF`` is a cap:

    coeff = min(cap, target_ratio * policy_norm / (aux_norm + epsilon)).

A zero policy gradient gives a zero coefficient, so the aux term never updates the policy alone.
"""

from __future__ import annotations

import argparse
import math
import os
from dataclasses import dataclass
from typing import Mapping


MODES = ("off", "norm_ratio")
DEFAULT_TARGET_GRAD_RATIO = 0.25
DEFAULT_EPSILON = 1e-12
TORCH_CLIP_EPSILON = 1e-6


@dataclass(frozen=True)
class BalanceConfig:
    mode: str
    target_ratio: float
    coeff_cap: float
    epsilon: float = DEFAULT_EPSILON


def _finite_positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} must be finite and > 0, got {value!r}")
    return value


def _finite_norm(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise FloatingPointError(f"{name} must be finite and >= 0, got {value!r}")
    return value


def config_from_mapping(env: Mapping[str, str], *, coeff_cap: float) -> BalanceConfig:
    """Parse balancing knobs from ``env``; ``coeff_cap`` is the validated ``AUX_CE_COEFF``.

    The target is ignored in ``off`` mode, so a stale target in the environment cannot break it.
    """

    mode = str(env.get("AUX_CE_BALANCE_MODE", "off")).strip()
    if mode not in MODES:
        raise ValueError(f"AUX_CE_BALANCE_MODE must be one of {MODES}, got {mode!r}")
    cap = _finite_positive("AUX_CE_COEFF", coeff_cap)
    target = DEFAULT_TARGET_GRAD_RATIO
    if mode == "norm_ratio":
        try:
            target = float(
                env.get("AUX_CE_TARGET_GRAD_RATIO", str(DEFAULT_TARGET_GRAD_RATIO))
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "AUX_CE_TARGET_GRAD_RATIO must be a finite number > 0"
            ) from exc
        target = _finite_positive("AUX_CE_TARGET_GRAD_RATIO", target)
    return BalanceConfig(mode=mode, target_ratio=target, coeff_cap=cap)


def config_from_env(*, coeff_cap: float) -> BalanceConfig:
    return config_from_mapping(os.environ, coeff_cap=coeff_cap)


def effective_aux_coefficient(
    policy_norm: float, aux_norm: float, config: BalanceConfig
) -> float:
    """Return the coefficient for one optimizer update. Raises outside ``norm_ratio`` mode."""

    if config.mode != "norm_ratio":
        raise ValueError(
            f"effective_aux_coefficient requires norm_ratio, got {config.mode!r}"
        )
    policy = _finite_norm("policy_grad_norm", policy_norm)
    aux = _finite_norm("aux_grad_norm", aux_norm)
    if policy == 0.0:
        return 0.0
    coeff = min(config.coeff_cap, config.target_ratio * policy / (aux + config.epsilon))
    if not math.isfinite(coeff) or coeff < 0.0 or coeff > config.coeff_cap:
        raise FloatingPointError(
            f"invalid adaptive aux coefficient {coeff!r} from policy={policy}, aux={aux}"
        )
    return coeff


def gradient_diagnostics(
    *,
    policy_norm: float,
    aux_norm: float,
    coefficient: float,
    combined_norm: float,
    grad_clip: float,
    epsilon: float = DEFAULT_EPSILON,
) -> dict[str, float]:
    """Derive buffer-free telemetry from three measured norms.

    The policy/aux cosine comes from ``||p + c*a||^2 = ||p||^2 + c^2||a||^2 + 2*c<p,a>`` and is
    reported as 0 when undefined (zero coefficient or norm) to keep NaNs out of metric reducers.
    """

    policy = _finite_norm("policy_grad_norm", policy_norm)
    aux = _finite_norm("aux_grad_norm", aux_norm)
    combined = _finite_norm("combined_grad_norm", combined_norm)
    coeff = float(coefficient)
    if not math.isfinite(coeff) or coeff < 0.0:
        raise FloatingPointError(
            f"aux coefficient must be finite and >= 0, got {coeff!r}"
        )
    clip = _finite_positive("grad_clip", grad_clip)
    weighted = coeff * aux
    ratio = weighted / (policy + epsilon) if policy > 0.0 else 0.0
    cosine = 0.0
    if coeff > 0.0 and policy > 0.0 and aux > 0.0:
        dot = (combined * combined - policy * policy - weighted * weighted) / (
            2.0 * coeff
        )
        cosine = max(-1.0, min(1.0, dot / (policy * aux)))
    clip_factor = min(1.0, clip / (combined + TORCH_CLIP_EPSILON))
    return {
        "aux_grad_norm_raw": aux,
        "aux_grad_norm_weighted": weighted,
        "policy_grad_norm_raw": policy,
        "aux_grad_ratio": ratio,
        "aux_ce_coeff_effective": coeff,
        "aux_policy_grad_cosine": cosine,
        "combined_grad_norm_preclip": combined,
        "final_grad_clip_factor": clip_factor,
    }


def _selftest() -> None:
    off = config_from_mapping({}, coeff_cap=0.5)
    assert off == BalanceConfig("off", DEFAULT_TARGET_GRAD_RATIO, 0.5)
    # Irrelevant target garbage cannot perturb the legacy path.
    assert (
        config_from_mapping({"AUX_CE_TARGET_GRAD_RATIO": "garbage"}, coeff_cap=0.5)
        == off
    )

    cfg = config_from_mapping(
        {"AUX_CE_BALANCE_MODE": "norm_ratio", "AUX_CE_TARGET_GRAD_RATIO": "0.25"},
        coeff_cap=0.5,
    )
    assert math.isclose(effective_aux_coefficient(2.0, 20.0, cfg), 0.025)
    assert effective_aux_coefficient(2.0, 1e-30, cfg) == 0.5  # cap binds
    assert effective_aux_coefficient(0.0, 20.0, cfg) == 0.0  # aux cannot act alone

    for env in (
        {"AUX_CE_BALANCE_MODE": "ratio"},
        {"AUX_CE_BALANCE_MODE": "norm_ratio", "AUX_CE_TARGET_GRAD_RATIO": "0"},
        {"AUX_CE_BALANCE_MODE": "norm_ratio", "AUX_CE_TARGET_GRAD_RATIO": "nan"},
    ):
        try:
            config_from_mapping(env, coeff_cap=0.5)
            raise AssertionError(f"bad config must fail: {env}")
        except ValueError:
            pass
    for bad in (float("nan"), float("inf"), -1.0):
        try:
            effective_aux_coefficient(1.0, bad, cfg)
            raise AssertionError(f"bad aux norm must fail: {bad}")
        except FloatingPointError:
            pass

    # Known parallel, orthogonal and anti-parallel norm triangles.
    parallel = gradient_diagnostics(
        policy_norm=3.0,
        aux_norm=4.0,
        coefficient=0.5,
        combined_norm=5.0,
        grad_clip=10.0,
    )
    assert abs(parallel["aux_policy_grad_cosine"] - 1.0) < 1e-12
    assert abs(parallel["aux_grad_ratio"] - 2.0 / 3.0) < 1e-12
    orthogonal = gradient_diagnostics(
        policy_norm=3.0,
        aux_norm=4.0,
        coefficient=0.5,
        combined_norm=math.sqrt(13.0),
        grad_clip=1.0,
    )
    assert abs(orthogonal["aux_policy_grad_cosine"]) < 1e-12
    assert (
        abs(
            orthogonal["final_grad_clip_factor"]
            - 1.0 / (math.sqrt(13.0) + TORCH_CLIP_EPSILON)
        )
        < 1e-12
    )
    antiparallel = gradient_diagnostics(
        policy_norm=3.0,
        aux_norm=4.0,
        coefficient=0.5,
        combined_norm=1.0,
        grad_clip=10.0,
    )
    assert abs(antiparallel["aux_policy_grad_cosine"] + 1.0) < 1e-12
    zero = gradient_diagnostics(
        policy_norm=0.0, aux_norm=4.0, coefficient=0.0, combined_norm=0.0, grad_clip=1.0
    )
    assert zero["aux_grad_ratio"] == zero["aux_policy_grad_cosine"] == 0.0
    assert zero["final_grad_clip_factor"] == 1.0
    print("aux_grad_balance selftest: OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if not args.selftest:
        parser.error("pass --selftest")
    _selftest()
