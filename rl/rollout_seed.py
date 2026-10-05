"""Stable common-random-number seeds for stochastic GRPO rollouts.

``data.seed`` picks the prompts in a step but not SGLang sampling. These helpers hash
(experiment seed, step, game id, rollout index) into a request seed, then derive a seed per
sender and receiver turn, independent of process scheduling and Python's ``hash()``.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Sequence


MAX_SAMPLING_SEED = 2**31 - 1
_PERSON = b"pgrseed1"


def _nonnegative_int(name: str, value: int) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must be a non-negative integer, got {value!r}"
        ) from exc
    if parsed < 0 or str(parsed) != str(value).strip():
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}")
    return parsed


def _derive(*parts: object) -> int:
    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.blake2s(payload, digest_size=8, person=_PERSON).digest()
    return int.from_bytes(digest, "big") % MAX_SAMPLING_SEED


def derive_rollout_seed(
    *, base_seed: int, global_step: int, game_id: object, rollout_index: int
) -> int:
    """Derive the sampling seed for one repeated rollout request."""

    base = _nonnegative_int("base_seed", base_seed)
    step = _nonnegative_int("global_step", global_step)
    offset = _nonnegative_int("rollout_index", rollout_index)
    if game_id is None or str(game_id) == "":
        raise ValueError("game_id must be non-empty")
    return _derive("request", base, step, game_id, offset)


def derive_turn_seed(*, rollout_seed: int, turn_index: int) -> int:
    """Derive an independent SGLang ``sampling_seed`` for one assistant turn."""

    request_seed = _nonnegative_int("rollout_seed", rollout_seed)
    turn = _nonnegative_int("turn_index", turn_index)
    return _derive("turn", request_seed, turn)


def derive_receiver_turn_seed(*, rollout_seed: int, turn_index: int) -> int:
    """Derive an independent OpenAI-compatible seed for one receiver turn."""

    request_seed = _nonnegative_int("rollout_seed", rollout_seed)
    turn = _nonnegative_int("turn_index", turn_index)
    return _derive("receiver-turn", request_seed, turn)


def derive_engine_seed(*, base_seed: int, worker_rank: int) -> int:
    """Pin otherwise-random SGLang engine initialization per stable worker rank."""

    base = _nonnegative_int("base_seed", base_seed)
    rank = _nonnegative_int("worker_rank", worker_rank)
    return _derive("engine", base, rank)


def interleaved_rollout_seeds(
    game_ids: Sequence[object], *, repeat_times: int, base_seed: int, global_step: int
) -> list[int]:
    """Match ``DataProto.repeat(..., interleave=True)`` exactly."""

    repeats = _nonnegative_int("repeat_times", repeat_times)
    if repeats <= 0:
        raise ValueError("repeat_times must be > 0")
    normalized = [str(game_id) for game_id in game_ids]
    if len(set(normalized)) != len(normalized):
        raise ValueError("game ids within one training batch must be unique")
    seeds = [
        derive_rollout_seed(
            base_seed=base_seed,
            global_step=global_step,
            game_id=game_id,
            rollout_index=rollout_index,
        )
        for game_id in normalized
        for rollout_index in range(repeats)
    ]
    if len(set(seeds)) != len(seeds):
        raise RuntimeError("deterministic rollout seed collision within one batch")
    return seeds


def seed_digest(seeds: Sequence[int]) -> str:
    """Compact provenance fingerprint for a resolved step seed vector."""

    text = ",".join(str(_nonnegative_int("seed", seed)) for seed in seeds)
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _selftest() -> None:
    kwargs = {"repeat_times": 4, "base_seed": 2026, "global_step": 3}
    first = interleaved_rollout_seeds([17, 29], **kwargs)
    assert first == interleaved_rollout_seeds([17, 29], **kwargs)
    assert len(first) == len(set(first)) == 8
    assert first != interleaved_rollout_seeds([17, 29], **{**kwargs, "global_step": 4})
    assert first != interleaved_rollout_seeds([17, 30], **kwargs)
    assert first[0] != derive_turn_seed(rollout_seed=first[0], turn_index=0)
    assert derive_turn_seed(rollout_seed=first[0], turn_index=0) != derive_turn_seed(
        rollout_seed=first[0], turn_index=1
    )
    assert derive_receiver_turn_seed(
        rollout_seed=first[0], turn_index=0
    ) != derive_turn_seed(rollout_seed=first[0], turn_index=0)
    assert derive_receiver_turn_seed(
        rollout_seed=first[0], turn_index=0
    ) != derive_receiver_turn_seed(rollout_seed=first[0], turn_index=1)
    assert derive_engine_seed(base_seed=2026, worker_rank=0) != derive_engine_seed(
        base_seed=2026, worker_rank=1
    )
    assert len(seed_digest(first)) == 64
    for bad in (-1, 1.5, "01", True):
        try:
            derive_rollout_seed(
                base_seed=bad, global_step=1, game_id=1, rollout_index=0
            )
            raise AssertionError(f"invalid base seed accepted: {bad!r}")
        except ValueError:
            pass
    print("rollout_seed selftest: OK")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if not args.selftest:
        parser.error("pass --selftest")
    _selftest()
