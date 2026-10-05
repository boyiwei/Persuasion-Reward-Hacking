"""Static check for the generated deterministic SGLang rollout-seed patch."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


def _require(source: str, needles: tuple[str, ...], path: Path) -> None:
    missing = [needle for needle in needles if needle not in source]
    if missing:
        raise AssertionError(f"{path}: missing rollout-seed fragments: {missing}")


def check(*, trainer: Path, rollout: Path, schema: Path) -> None:
    trainer_source = trainer.read_text()
    rollout_source = rollout.read_text()
    schema_source = schema.read_text()
    for path, source in (
        (trainer, trainer_source),
        (rollout, rollout_source),
        (schema, schema_source),
    ):
        ast.parse(source, filename=str(path))

    _require(
        trainer_source,
        (
            "[persuasion-gym] (14T)",
            "interleaved_rollout_seeds(",
            'gen_batch_output.non_tensor_batch["rollout_sampling_seed"]',
            "[persuasion-gym] (14TF)",
            'batch.non_tensor_batch["extra_info"] = np.asarray(',
            "seed_digest(_rollout_seeds)",
            '"ROLLOUT_SEED_PROVENANCE_DIR"',
            "os.replace(_rollout_seed_tmp, _rollout_seed_path)",
        ),
        trainer,
    )
    _require(
        rollout_source,
        (
            "[persuasion-gym] (14S)",
            'args["random_seed"] = _rollout_seed_14s.derive_engine_seed(',
            "worker_rank=0",
            'args["enable_deterministic_inference"] = True',
            'prompts.non_tensor_batch.get("rollout_sampling_seed")',
            "rollout_sampling_seed=(",
            "[persuasion-gym] (14SR)",
            "_interaction_kwargs = dict(",
            '_interaction_kwargs["rollout_sampling_seed"] = int(',
            'sampling_params["sampling_seed"] = _rollout_seed_14s.derive_turn_seed(',
        ),
        rollout,
    )
    _require(
        schema_source,
        ("[persuasion-gym] (14R)", "rollout_sampling_seed: Optional[int] = None"),
        schema,
    )

    trainer_seed = trainer_source.index("interleaved_rollout_seeds(")
    trainer_repeat = trainer_source.index("gen_batch_output = gen_batch.repeat(")
    trainer_attach = trainer_source.index(
        'gen_batch_output.non_tensor_batch["rollout_sampling_seed"]'
    )
    if not trainer_repeat < trainer_seed < trainer_attach:
        raise AssertionError(
            f"{trainer}: rollout seed attachment moved away from repeated training rows"
        )

    fallback_stamp = trainer_source.index("[persuasion-gym] (14TF)")
    fallback_repeat = trainer_source.rfind(
        "batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)",
        0,
        fallback_stamp,
    )
    fallback_union = trainer_source.index(
        "batch = batch.union(gen_batch_output)", fallback_stamp
    )
    if not 0 <= fallback_repeat < fallback_stamp < fallback_union:
        raise AssertionError(
            f"{trainer}: fallback receiver seed must be stamped after repeat and before union"
        )

    interaction_copy = rollout_source.index("_interaction_kwargs = dict(")
    interaction_stamp = rollout_source.index(
        '_interaction_kwargs["rollout_sampling_seed"] = int('
    )
    request_construct = rollout_source.index(
        "req = AsyncRolloutRequest(", interaction_stamp
    )
    if not interaction_copy < interaction_stamp < request_construct:
        raise AssertionError(
            f"{rollout}: receiver seed must stamp a copied interaction dict before request construction"
        )
    deterministic_flag = rollout_source.index(
        'args["enable_deterministic_inference"] = True'
    )
    engine_constructor = rollout_source.index("AsyncEngine(**args)")
    if deterministic_flag > engine_constructor:
        raise AssertionError(
            f"{rollout}: deterministic-inference flag must be set before engine construction"
        )
    turn_seed = rollout_source.index('sampling_params["sampling_seed"]')
    engine_call = rollout_source.index(
        "output = await self._handle_engine_call(", turn_seed
    )
    if turn_seed > engine_call:
        raise AssertionError(
            f"{rollout}: turn seed must be set before the SGLang engine call"
        )
    print("deterministic rollout-seed generated-patch check: OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--rollout", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    args = parser.parse_args()
    check(trainer=args.trainer, rollout=args.rollout, schema=args.schema)


if __name__ == "__main__":
    main()
