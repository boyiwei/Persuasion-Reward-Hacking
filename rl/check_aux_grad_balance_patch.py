"""Static smoke check for the generated Verl aux-gradient balancing patch.

Run this against a pristine temporary Verl checkout after rl/patch_verl.sh:

    python -m rl.check_aux_grad_balance_patch \
      --actor /tmp/verl/verl/workers/actor/dp_actor.py \
      --trainer /tmp/verl/verl/trainer/ppo/ray_trainer.py \
      --launcher scripts/rl_train_sender.slurm
"""

from __future__ import annotations

import argparse
import ast
from pathlib import Path


ACTOR_REQUIRED = (
    "[persuasion-gym] (13T)",
    '_aux_balance_mode == "off"',
    "(_coefficient * _aux_ce * (1.0 / self.gradient_accumulation)).backward()",
    'if _aux_balance_mode == "norm_ratio":',
    '_aux_balance_mode == "norm_ratio"',
    "_pre_aux_rng = _aux_capture_rng()",
    '_aux_raw_grad_norm = _aux_grad_norm_no_clip("aux_grad_norm", _soft=True)',
    "_aux_restore_rng(_pre_aux_rng)",
    '_policy_grad_norm = _aux_grad_norm_no_clip("policy_grad_norm")',
    "_effective_coeff = _aux_sync_coefficient(_effective_coeff)",
    "torch.distributed.ReduceOp.MIN",
    "torch.distributed.ReduceOp.MAX",
    "rank-divergent aux_valid",
    "_aux_restore_rng(_aux_rng)",
    "_aux_restore_rng(_post_policy_rng)",
    '_combined_grad_norm = _aux_grad_norm_no_clip("combined_grad_norm")',
    'f"actor/{_key}"',
    # A non-finite raw aux gradient skips the aux term for that step (within a budget, logged)
    # instead of killing the run; these three pins catch a re-patch that reverts to the fatal guard.
    "AUX_CE_NONFINITE_BUDGET",
    "_aux_nonfinite_steps",
    '"actor/aux_skipped"',
)
TRAINER_REQUIRED = (
    "[persuasion-gym] (10I)",
    'batch.meta_info["aux_ce_coeff"]',
    'batch.meta_info["aux_ce_terms"]',
    'batch.meta_info["aux_ce_balance_mode"]',
    'batch.meta_info["aux_ce_target_grad_ratio"]',
    'batch.meta_info["aux_ce_balance_epsilon"]',
    "_aux_balance_mode_10i = os.environ.get(",
    'if _aux_balance_mode_10i == "norm_ratio":',
    'if self._persuasion_aux_balance_mode == "norm_ratio":',
)
LAUNCHER_REQUIRED = (
    'if [[ "$AUX_CE_BALANCE_MODE" == norm_ratio ]]; then',
    '"$PY" -m rl.check_aux_grad_balance_patch',
    '--actor "$_AUX_DP" --trainer "$_AUX_RT"',
    'AUXCE_SUFFIX="_auxce${AUX_CE_COEFF//./p}"',
    "_AUXCE_META_BALANCE_MODE=off; _AUXCE_META_TARGET_RATIO=0.0",
)


def _require(source: str, needles: tuple[str, ...], path: Path) -> None:
    missing = [needle for needle in needles if needle not in source]
    if missing:
        raise AssertionError(f"{path}: missing generated patch fragments: {missing}")


def check(actor_path: Path, trainer_path: Path) -> None:
    actor = actor_path.read_text()
    trainer = trainer_path.read_text()
    actor_tree = ast.parse(actor, filename=str(actor_path))
    trainer_tree = ast.parse(trainer, filename=str(trainer_path))
    _require(actor, ACTOR_REQUIRED, actor_path)
    _require(trainer, TRAINER_REQUIRED, trainer_path)
    # Fixed mode must not import the adaptive module, so fixed-coeff runs from older checkouts
    # still work against a shared, newly patched verl.
    adaptive_config = actor.index('if _aux_balance_mode == "norm_ratio":')
    adaptive_import = actor.index(
        "import rl.aux_grad_balance as _aux_gb", adaptive_config
    )
    aux_select = actor.index("select_keys.extend(_aux_keys)", adaptive_import)
    if not adaptive_config < adaptive_import < aux_select:
        raise AssertionError(
            f"{actor_path}: adaptive import escaped its norm-ratio gate"
        )
    if "import rl.aux_grad_balance as _aux_gb" in actor[:adaptive_config]:
        raise AssertionError(f"{actor_path}: fixed mode imports the adaptive helper")
    adaptive_imports = [
        node
        for node in ast.walk(actor_tree)
        if isinstance(node, ast.Import)
        and any(
            alias.name == "rl.aux_grad_balance" and alias.asname == "_aux_gb"
            for alias in node.names
        )
    ]
    if len(adaptive_imports) != 1:
        raise AssertionError(
            f"{actor_path}: expected one adaptive helper import, got {len(adaptive_imports)}"
        )
    adaptive_import_node = adaptive_imports[0]
    enclosing_guards = [
        node
        for node in ast.walk(actor_tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "_aux_balance_mode"
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Eq)
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "norm_ratio"
        and adaptive_import_node in tuple(ast.walk(node))
    ]
    if not enclosing_guards:
        raise AssertionError(
            f"{actor_path}: adaptive import is not nested under the norm-ratio AST guard"
        )
    trainer_imports = [
        node
        for node in ast.walk(trainer_tree)
        if isinstance(node, ast.Import)
        and any(
            alias.name == "rl.aux_grad_balance" and alias.asname == "_aux_balance_10i"
            for alias in node.names
        )
    ]
    if len(trainer_imports) != 1:
        raise AssertionError(
            f"{trainer_path}: expected one adaptive helper import, got {len(trainer_imports)}"
        )
    trainer_import_node = trainer_imports[0]
    trainer_import_guards = [
        node
        for node in ast.walk(trainer_tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "_aux_balance_mode_10i"
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Eq)
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "norm_ratio"
        and trainer_import_node in tuple(ast.walk(node))
    ]
    if not trainer_import_guards:
        raise AssertionError(
            f"{trainer_path}: adaptive import is not nested under the norm-ratio AST guard"
        )
    metadata_guards = [
        node
        for node in ast.walk(trainer_tree)
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Attribute)
        and node.test.left.attr == "_persuasion_aux_balance_mode"
        and len(node.test.ops) == 1
        and isinstance(node.test.ops[0], ast.Eq)
        and len(node.test.comparators) == 1
        and isinstance(node.test.comparators[0], ast.Constant)
        and node.test.comparators[0].value == "norm_ratio"
    ]
    if len(metadata_guards) != 1:
        raise AssertionError(
            f"{trainer_path}: expected one adaptive metadata guard, got {len(metadata_guards)}"
        )
    adaptive_keys = {
        "aux_ce_balance_mode",
        "aux_ce_target_grad_ratio",
        "aux_ce_balance_epsilon",
    }
    guarded_nodes = tuple(ast.walk(metadata_guards[0]))
    assignments = [
        (node, target.slice.value)
        for node in ast.walk(trainer_tree)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Subscript)
        and isinstance(target.slice, ast.Constant)
        and target.slice.value in adaptive_keys
    ]
    if {key for _, key in assignments} != adaptive_keys or any(
        node not in guarded_nodes for node, _ in assignments
    ):
        raise AssertionError(
            f"{trainer_path}: adaptive metadata escaped its norm-ratio guard"
        )

    # Required order: snapshot RNG, measure/discard aux, restore RNG (so the diagnostic pass does
    # not perturb policy sampling), measure policy, sync, replay, measure combined, optimizer step.
    ordered = (
        "_pre_aux_rng = _aux_capture_rng()",
        '_aux_raw_grad_norm = _aux_grad_norm_no_clip("aux_grad_norm", _soft=True)',
        "_aux_restore_rng(_pre_aux_rng)",
        '_policy_grad_norm = _aux_grad_norm_no_clip("policy_grad_norm")',
        "_effective_coeff = _aux_sync_coefficient(_effective_coeff)",
        "_aux_restore_rng(_aux_rng)",
        '_combined_grad_norm = _aux_grad_norm_no_clip("combined_grad_norm")',
        "grad_norm = self._optimizer_step()",
    )
    positions = [actor.index(fragment) for fragment in ordered]
    discard = actor.index("self.actor_optimizer.zero_grad()", positions[1])
    expected = [positions[0], positions[1], discard, *positions[2:]]
    if expected != sorted(expected):
        raise AssertionError(
            f"{actor_path}: adaptive operation order drifted: {expected}"
        )

    # Legacy mode stays at the fixed coefficient inside the policy micro-batch loop.
    off = actor.index('if _aux_on and _aux_balance_mode == "off":')
    policy = actor.index('response_mask = model_inputs["response_mask"]', off)
    legacy_call = actor.index("model_inputs, _aux_coeff, _strict=False", off)
    if not off < legacy_call < policy:
        raise AssertionError(
            f"{actor_path}: legacy aux backward moved out of the policy loop"
        )
    print("aux gradient generated-patch check: OK")


def check_launcher(path: Path) -> None:
    source = path.read_text()
    _require(source, LAUNCHER_REQUIRED, path)
    if source.count('"$PY" -m rl.check_aux_grad_balance_patch') != 1:
        raise AssertionError(
            f"{path}: adaptive structural checker must appear exactly once"
        )
    checker = source.index('"$PY" -m rl.check_aux_grad_balance_patch')
    adaptive_guard = source.rfind(
        'if [[ "$AUX_CE_BALANCE_MODE" == norm_ratio ]]; then', 0, checker
    )
    adaptive_end = source.index("\n  fi", checker)
    sidecar_gate = source.index("# Sidecar <-> run parity", adaptive_end)
    if not 0 <= adaptive_guard < checker < adaptive_end < sidecar_gate:
        raise AssertionError(
            f"{path}: adaptive structural checker is not inside the pre-sidecar norm-ratio gate"
        )
    print("aux gradient launcher preflight check: OK")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actor", type=Path, required=True)
    parser.add_argument("--trainer", type=Path, required=True)
    parser.add_argument("--launcher", type=Path)
    args = parser.parse_args()
    check(args.actor, args.trainer)
    if args.launcher is not None:
        check_launcher(args.launcher)


if __name__ == "__main__":
    main()
