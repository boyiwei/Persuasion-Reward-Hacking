"""Resolve the training algorithm ALGO=base[+penalty][+aux_loss] into environment variables.

`base` (the outcome reward) is always selected; `penalty` and `aux_loss` add to it and combine
freely. Each component's values live in rl/config/algorithm/<id>.yaml. A component can only emit
variables that scripts/rl_train_sender.slurm, rl/reward_function.py, rl/aux_ce.py and the verl
patches already read: the environment is the only channel into the Ray workers, and the reward
function is loaded by file path.

    python -m rl.arm_config --describe  --algo base+penalty --prompt strategies --repo $PWD
    python -m rl.arm_config --print-env --algo base+aux_loss --prompt strategies --repo $PWD
    python -m rl.arm_config --selftest

Join components with '+', not ',': `sbatch --export` splits on commas, so ALGO=base,penalty
reaches the job as ALGO=base.

A variable already set (non-empty) in the environment wins and --describe reports it as OVERRIDE.
That is how a coefficient sweep runs: ALGO=base+penalty PENALTY_COEFFICIENT=0.5

Component file keys (any other key is an error):

    id, order               the file stem; registry position (selections are sorted into it)
    implicit                always selected, ALGO may omit it (base)
    doc                     one sentence, printed by --describe
    env / env_off           variables emitted when selected / not selected; env_off covers those
                            the launcher reads unconditionally (unset aborts under `set -u`)
    gate                    variables whose non-off value means the component is active
    requires                prompt: [<mode>, ...] and/or env: {VAR: value}
    conflicts               env: {VAR: value}
    expect_name_tag[_absent]  run-name tag the launcher derives when selected / not selected

Values may use only {repo} and {size} (the policy model without its `qwen3-` prefix).
"""
from __future__ import annotations

import argparse
import functools
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional

import yaml

from rl.sender_prompts import PromptSpecError, parse_spec

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _REPO_ROOT / "rl" / "config" / "algorithm"
_LAUNCHER = _REPO_ROOT / "scripts" / "rl_train_sender.slurm"

SCHEMA = ("id", "order", "implicit", "doc", "env", "env_off", "gate", "requires", "conflicts",
          "expect_name_tag", "expect_name_tag_absent")
_REQUIRES_KEYS = ("prompt", "env")
_CONFLICTS_KEYS = ("env",)
_PLACEHOLDERS = ("repo", "size")
_BRACED = re.compile(r"\{([^{}]*)\}")


class ArmConfigError(ValueError):
    """An unknown component, a contradictory environment, or an unusable combination."""


@dataclass(frozen=True)
class Group:
    """One algorithm component, as read from rl/config/algorithm/<id>.yaml."""

    id: str
    order: int
    doc: str
    implicit: bool
    env: dict
    env_off: dict
    gate: tuple
    requires: dict
    conflicts: dict
    expect_name_tag: str
    expect_name_tag_absent: str


@dataclass(frozen=True)
class Resolution:
    """What one (ALGO, PROMPT, policy, repo, environment) resolves to."""

    algo: str                 # the selection, registry-ordered and '+'-joined
    prompt: str               # the sender prompt spec, canonical
    policy_model: str
    repo: str
    selected: tuple           # component ids, registry order
    env: dict                 # variable -> value, ordered
    source: dict              # variable -> the component that owns it
    overridden: dict          # variable -> the environment value that wins over it
    warnings: tuple


# ---------------------------------------------------------------- the component registry
def _as_str_map(raw, where: str) -> dict:
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ArmConfigError(f"{where} must be a mapping, got {type(raw).__name__}")
    out = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            raise ArmConfigError(f"{where} has a non-string variable name {key!r}")
        if not isinstance(value, str):
            # YAML 1.1 reads bare off/on/yes/no as booleans, so an unquoted "off" would arrive as
            # False and be exported as the string "False".
            raise ArmConfigError(f"{where}.{key} must be a quoted string, got "
                                 f"{type(value).__name__} {value!r}")
        out[key] = value
    return out


def _read_group(path: Path) -> Group:
    with open(path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise ArmConfigError(f"{path.name} does not hold a mapping")
    unknown = [k for k in cfg if k not in SCHEMA]
    if unknown:
        raise ArmConfigError(f"{path.name} has keys outside the schema: {', '.join(sorted(unknown))} "
                             f"(known: {', '.join(SCHEMA)})")
    gid = cfg.get("id")
    if gid != path.stem:
        raise ArmConfigError(f"{path.name} declares id {gid!r}")
    if not isinstance(cfg.get("order"), int):
        raise ArmConfigError(f"{path.name} needs an integer `order`")
    doc = cfg.get("doc")
    if not isinstance(doc, str) or not doc.strip():
        raise ArmConfigError(f"{path.name} needs a one-sentence `doc`")
    env = _as_str_map(cfg.get("env"), f"{path.name} env")
    env_off = _as_str_map(cfg.get("env_off"), f"{path.name} env_off")
    gate = tuple(cfg.get("gate") or ())
    for var in gate:
        if var not in env or var not in env_off:
            raise ArmConfigError(f"{path.name} gates on {var}, which needs a value in both env "
                                 f"and env_off (the off value is what 'inactive' means)")
    requires = dict(cfg.get("requires") or {})
    bad = [k for k in requires if k not in _REQUIRES_KEYS]
    if bad:
        raise ArmConfigError(f"{path.name} requires: unknown key(s) {', '.join(sorted(bad))} "
                             f"(known: {', '.join(_REQUIRES_KEYS)})")
    if "prompt" in requires:
        requires["prompt"] = [str(m) for m in requires["prompt"]]
    if "env" in requires:
        requires["env"] = _as_str_map(requires["env"], f"{path.name} requires.env")
    conflicts = dict(cfg.get("conflicts") or {})
    bad = [k for k in conflicts if k not in _CONFLICTS_KEYS]
    if bad:
        raise ArmConfigError(f"{path.name} conflicts: unknown key(s) {', '.join(sorted(bad))} "
                             f"(known: {', '.join(_CONFLICTS_KEYS)})")
    if "env" in conflicts:
        conflicts["env"] = _as_str_map(conflicts["env"], f"{path.name} conflicts.env")
    return Group(id=gid, order=cfg["order"], doc=" ".join(doc.split()),
                 implicit=bool(cfg.get("implicit", False)), env=env, env_off=env_off, gate=gate,
                 requires=requires, conflicts=conflicts,
                 expect_name_tag=str(cfg.get("expect_name_tag", "")),
                 expect_name_tag_absent=str(cfg.get("expect_name_tag_absent", "")))


@functools.lru_cache(maxsize=None)
def load_groups() -> dict:
    """Every algorithm component, keyed by id, in registry order."""
    groups = [_read_group(p) for p in sorted(_CONFIG_DIR.glob("*.yaml"))]
    if not groups:
        raise ArmConfigError(f"no algorithm components under {_CONFIG_DIR}")
    groups.sort(key=lambda g: (g.order, g.id))
    return {g.id: g for g in groups}


# ---------------------------------------------------------------- resolution
def _expand(value: str, *, repo: str, size: str, where: str) -> str:
    def one(match):
        key = match.group(1)
        if key not in _PLACEHOLDERS:
            raise ArmConfigError(f"{where} uses the placeholder {{{key}}}; only "
                                 f"{', '.join('{%s}' % p for p in _PLACEHOLDERS)} are legal")
        return {"repo": repo, "size": size}[key]

    out = _BRACED.sub(one, value)
    if "{" in out or "}" in out:
        raise ArmConfigError(f"{where} has an unbalanced brace: {value!r}")
    return out


def _is_set(environ: Mapping, var: str) -> bool:
    """An exported-but-empty value counts as unset: that is how `--export=ALL` carries an off knob."""
    return (environ.get(var) or "") != ""


def _same(a: str, b: str) -> bool:
    """Equal as the launcher and the runtime compare: numerically when both are numbers."""
    try:
        return float(a) == float(b)
    except ValueError:
        return a == b


def _parse_algo(algo: str, groups: dict) -> list:
    text = (algo or "").strip()
    if "," in text:
        raise ArmConfigError(f"ALGO components are joined with '+', not ',' (got {text!r}). "
                             "`sbatch --export` is itself comma-separated, so a comma would reach "
                             "the job as the first component alone.")
    names = [t.strip() for t in text.split("+")] if text else []
    picked = []
    for name in names:
        if not name:
            raise ArmConfigError(f"ALGO has an empty component (got {text!r})")
        if name not in groups:
            raise ArmConfigError(f"unknown algorithm component {name!r}; known: "
                                 f"{', '.join(groups)}")
        if name in picked:
            raise ArmConfigError(f"ALGO names {name!r} twice (got {text!r})")
        picked.append(name)
    for gid, group in groups.items():
        if group.implicit and gid not in picked:
            picked.append(gid)
    return [gid for gid in groups if gid in picked]


def resolve(algo: str, prompt: str, *, policy_model: str, repo: str,
            environ: Mapping) -> Resolution:
    """Turn a selection into the variables it sets, honouring values already in `environ`."""
    groups = load_groups()
    spec = parse_spec(prompt)
    selected = _parse_algo(algo, groups)
    size = policy_model.removeprefix("qwen3-")

    env: dict = {}
    source: dict = {}
    for gid in selected:
        for var, raw in groups[gid].env.items():
            value = _expand(raw, repo=repo, size=size, where=f"{gid}.env.{var}")
            if var in env and env[var] != value:
                raise ArmConfigError(
                    f"{source[var]} sets {var}={env[var]!r} and {gid} sets {var}={value!r}; "
                    "two selected components cannot disagree about one variable")
            env.setdefault(var, value)
            source.setdefault(var, gid)
    for gid, group in groups.items():
        if gid in selected:
            continue
        for var, raw in group.env_off.items():
            if var in source:
                continue    # a selected component owns it
            value = _expand(raw, repo=repo, size=size, where=f"{gid}.env_off.{var}")
            if var in env and env[var] != value:
                raise ArmConfigError(
                    f"{source[var]} and {gid} disagree about the off value of {var}")
            env.setdefault(var, value)
            source.setdefault(var, f"{gid} (off)")

    overridden = {var: environ[var] for var in env if _is_set(environ, var)}

    warnings = []
    for gid, group in groups.items():
        if gid in selected:
            continue
        for var in group.gate:
            if not _is_set(environ, var):
                continue
            if _same(environ[var], group.env_off[var]):
                continue
            warnings.append(
                f"{var}={environ[var]!r} is set in the environment but ALGO does not name "
                f"{gid!r}: the environment configures a component the selection leaves out, and "
                f"custom_metadata.algo records only the components ALGO names. "
                f"Add {gid} to ALGO, or clear {var}. (--strict-env makes this fatal.)")

    return Resolution(algo="+".join(selected), prompt=str(spec), policy_model=policy_model,
                      repo=repo, selected=tuple(selected), env=env, source=source,
                      overridden=overridden, warnings=tuple(warnings))


def effective(res: Resolution, environ: Mapping, var: str) -> str:
    """The value the job will actually run with: the environment wins over the component."""
    if _is_set(environ, var):
        return environ[var]
    return res.env.get(var, "")


def validate(res: Resolution, environ: Mapping, *, strict: bool = False) -> None:
    """Raise ArmConfigError when the selection and the environment cannot describe one run."""
    groups = load_groups()
    mode = parse_spec(res.prompt).mode
    for gid in res.selected:
        group = groups[gid]
        modes = group.requires.get("prompt")
        if modes and mode not in modes:
            raise ArmConfigError(
                f"{gid} requires PROMPT={' or '.join(modes)} (got {res.prompt!r}): its probe "
                "items bake that sender prompt into their contexts, and nothing downstream "
                "compares the two.")
        for var, want in group.requires.get("env", {}).items():
            got = effective(res, environ, var)
            if not _same(got, want):
                raise ArmConfigError(f"{gid} requires {var}={want} (got {got!r})")
        for var, clash in group.conflicts.get("env", {}).items():
            got = effective(res, environ, var)
            if _same(got, clash):
                raise ArmConfigError(f"{gid} cannot run with {var}={clash}")
        for var in group.gate:
            if _is_set(environ, var) and _same(environ[var], group.env_off[var]):
                raise ArmConfigError(
                    f"ALGO names {gid!r} but {var}={environ[var]!r} in the environment switches it "
                    f"off; drop {gid} from ALGO, or clear {var}")
    if strict and res.warnings:
        raise ArmConfigError(" ".join(res.warnings))


# ---------------------------------------------------------------- output
def emit(res: Resolution, environ: Mapping) -> str:
    """`export VAR=value` lines for every variable the environment does not already set."""
    lines = [f"export {var}={shlex.quote(value)}"
             for var, value in res.env.items() if not _is_set(environ, var)]
    return "".join(line + "\n" for line in lines)


def describe(res: Resolution) -> str:
    groups = load_groups()
    width = max((len(v) for v in res.env), default=0)
    out = [f"algorithm : {res.algo}",
           f"prompt    : {res.prompt}",
           f"policy    : {res.policy_model}",
           f"repo      : {res.repo}",
           "",
           "components"]
    for gid, group in groups.items():
        state = "on " if gid in res.selected else "off"
        out.append(f"  [{state}] {gid:<9} {group.doc}")
    out += ["", "environment"]
    for var, value in res.env.items():
        if var in res.overridden:
            note = f"OVERRIDE (environment; {res.source[var]} sets {value})"
            value = res.overridden[var]
        else:
            note = res.source[var]
        out.append(f"  {var:<{width}} = {value:<8} <- {note}")
    return "".join(line.rstrip() + "\n" for line in out)


# ---------------------------------------------------------------- self-test
_ARM_4B_BASE = {
    "FORMAT_REWARD_COEFF": "0.1",
    "RL_MONITOR_ENABLE": "1",
    "PENALTY_TERMS": "",
    "PENALTY_COEFFICIENT": "",
    "AUX_CE": "0",
    "AUX_CE_DATA": "/repo/datasets/old_bailey/_generated/aux_ce_probe/variants/final/4B/sidecar.jsonl.gz",
    "AUX_CE_COEFF": "0.5",
    "AUX_CE_SAMPLING": "random",
    "AUX_CE_KIND_BALANCE": "0",
    "AUX_CE_BALANCE_MODE": "off",
    "AUX_CE_TARGET_GRAD_RATIO": "0.25",
    "AUX_CE_MAX_PROMPT": "8192",
    "AUX_CE_ROW_FRAC": "1.0",
    "AUX_CE_SEED": "2026",
}
_PENALTY_ON = {"PENALTY_TERMS": "illegal_full11", "PENALTY_COEFFICIENT": "0.3",
               "RL_MONITOR_SAMPLE_RATE": "1.0"}
_AUX_ON = {"AUX_CE": "1", "AUX_CE_KIND_BALANCE": "1", "AUX_CE_BALANCE_MODE": "norm_ratio"}


def _expect(algo: str, prompt: str, size: str = "4B", extra: Optional[dict] = None) -> dict:
    want = dict(_ARM_4B_BASE)
    want["AUX_CE_DATA"] = want["AUX_CE_DATA"].replace("/final/4B/", f"/final/{size}/")
    want.update(extra or {})
    got = resolve(algo, prompt, policy_model=f"qwen3-{size}", repo="/repo", environ={})
    validate(got, {})
    assert got.env == want, f"{algo} / {size}: {got.env} != {want}"
    assert got.algo == algo, f"{algo}: canonical spelling is {got.algo!r}"
    return got.env


def _raises(fn, needle: str, what: str) -> None:
    try:
        fn()
    except (ArmConfigError, PromptSpecError) as exc:
        assert needle in str(exc), f"{what}: message {str(exc)!r} lacks {needle!r}"
        return
    raise AssertionError(f"{what} was accepted")


def _roundtrip(text: str) -> dict:
    proc = subprocess.run(["bash", "-c", 'set -u; eval "$1"; env', "_", text],
                          capture_output=True, text=True, check=True,
                          env={"PATH": os.environ.get("PATH", "/usr/bin:/bin")})
    got = {}
    for line in proc.stdout.splitlines():
        key, sep, value = line.partition("=")
        if sep:
            got[key] = value
    return got


def _check_name_tags() -> None:
    """Each component's expect_name_tag, rebuilt from its own values and matched to the launcher."""
    groups = load_groups()
    base, penalty, aux = groups["base"], groups["penalty"], groups["aux_loss"]
    p = lambda v: v.replace(".", "p")   # noqa: E731 -- the launcher's ${VAR//./p}
    assert base.expect_name_tag == "_fmt" + p(base.env["FORMAT_REWARD_COEFF"])
    assert penalty.expect_name_tag == (
        "pen" + p(penalty.env["PENALTY_COEFFICIENT"]) + "_" + penalty.env["PENALTY_TERMS"])
    assert aux.expect_name_tag == (
        "_auxce" + p(aux.env["AUX_CE_COEFF"]) + "_" + aux.env["AUX_CE_SAMPLING"] + "_kindbal"
        + "_normratio" + p(aux.env["AUX_CE_TARGET_GRAD_RATIO"]))
    assert aux.env["AUX_CE_KIND_BALANCE"] == "1" and aux.env["AUX_CE_BALANCE_MODE"] == "norm_ratio"
    text = _LAUNCHER.read_text(encoding="utf-8")
    for anchor in ('FMT_SUFFIX="_fmt${FORMAT_REWARD_COEFF//./p}"',
                   'FAKEPEN_TAG=fakepen0',
                   'FAKEPEN_TAG="pen${EFF_COEFF//./p}_illegal_full11"',
                   'AUXCE_SUFFIX="_auxce${AUX_CE_COEFF//./p}"',
                   'AUXCE_SUFFIX="${AUXCE_SUFFIX}_${AUX_CE_SAMPLING}"',
                   'AUXKIND_SUFFIX="_kindbal"',
                   'AUXBAL_SUFFIX="_normratio${AUX_CE_TARGET_GRAD_RATIO//./p}"'):
        assert anchor in text, f"{_LAUNCHER.name} does not derive the run name with: {anchor}"
    assert penalty.expect_name_tag_absent == "fakepen0"


def _selftest() -> int:
    groups = load_groups()
    assert list(groups) == ["base", "penalty", "aux_loss"], list(groups)

    # the paper's arm families, variable for variable
    _expect("base", "base")
    _expect("base+penalty", "strategies", extra=_PENALTY_ON)
    for size in ("4B", "8B", "14B"):
        _expect("base+aux_loss", "strategies", size=size, extra=_AUX_ON)
    _expect("base+penalty+aux_loss", "strategies", extra={**_PENALTY_ON, **_AUX_ON})
    # base is implicit, and the selection is sorted into registry order
    assert resolve("aux_loss+penalty", "strategies", policy_model="qwen3-4B", repo="/repo",
                   environ={}).algo == "base+penalty+aux_loss"

    # validation
    def res(algo, prompt="strategies", env=None, policy="qwen3-4B"):
        got = resolve(algo, prompt, policy_model=policy, repo="/repo", environ=env or {})
        validate(got, env or {})
        return got

    _raises(lambda: res("base+aux_loss", "base"), "requires PROMPT=strategies", "aux_loss + base prompt")
    _raises(lambda: res("base+aux_loss", "single_strategy:anchoring"), "requires PROMPT=strategies",
            "aux_loss + single_strategy")
    _raises(lambda: res("base,penalty"), "'+', not ','", "comma-separated ALGO")
    _raises(lambda: res("penalty+penalty"), "twice", "duplicate component")
    _raises(lambda: res("nope"), "unknown algorithm component", "unknown component")
    _raises(lambda: res("base+"), "empty component", "trailing separator")
    _raises(lambda: res("base", "single_strategy:not_a_slug"), "unknown technique slug", "bad slug")
    _raises(lambda: res("base+aux_loss", env={"REJECTION_SAMPLING": "1"}),
            "cannot run with REJECTION_SAMPLING=1", "aux_loss + rejection sampling")
    _raises(lambda: res("base+aux_loss", env={"AUX_CE": "0"}), "switches it off",
            "aux_loss switched off in the environment")
    _raises(lambda: res("base+penalty", env={"RL_MONITOR_ENABLE": "0"}),
            "requires RL_MONITOR_ENABLE=1", "penalty without monitors")
    _raises(lambda: res("base+penalty", env={"RL_MONITOR_SAMPLE_RATE": "0.5"}),
            "requires RL_MONITOR_SAMPLE_RATE=1.0", "penalty at a throttled sample rate")
    # a spelling that is numerically 1.0 is the same rate
    res("base+penalty", env={"RL_MONITOR_SAMPLE_RATE": "1"})
    _raises(lambda: _expand("{nope}", repo="/r", size="4B", where="x"), "only {repo}, {size}",
            "unknown placeholder")
    _raises(lambda: _expand("{repo", repo="/r", size="4B", where="x"), "unbalanced brace",
            "unbalanced brace")

    # an explicit environment value wins and is reported, it is not emitted again
    over = res("base+penalty", env={"PENALTY_COEFFICIENT": "0.5"})
    assert over.overridden == {"PENALTY_COEFFICIENT": "0.5"}, over.overridden
    assert "PENALTY_COEFFICIENT" not in emit(over, {"PENALTY_COEFFICIENT": "0.5"})
    assert "OVERRIDE (environment; penalty sets 0.3)" in describe(over)
    assert effective(over, {"PENALTY_COEFFICIENT": "0.5"}, "PENALTY_COEFFICIENT") == "0.5"
    # an exported-but-empty value counts as unset
    assert resolve("base+penalty", "strategies", policy_model="qwen3-4B", repo="/repo",
                   environ={"PENALTY_TERMS": ""}).overridden == {}

    # a non-selected component's gate variable left in the environment warns, and --strict-env fails
    leak_env = {"PENALTY_TERMS": "illegal_full11", "PENALTY_COEFFICIENT": "0.3"}
    leak = resolve("base", "strategies", policy_model="qwen3-4B", repo="/repo", environ=leak_env)
    validate(leak, leak_env)
    assert len(leak.warnings) == 2 and "PENALTY_TERMS" in leak.warnings[0], leak.warnings
    _raises(lambda: validate(leak, leak_env, strict=True), "ALGO does not name", "--strict-env leak")

    # the emitted lines survive `set -u` and reproduce every value
    full = res("base+penalty+aux_loss", policy="qwen3-8B")
    text = emit(full, {})
    assert all(line.startswith("export ") for line in text.splitlines()), text
    got = _roundtrip(text)
    for var, value in full.env.items():
        assert got.get(var) == value, f"round trip: {var}={got.get(var)!r} != {value!r}"
    assert full.env["AUX_CE_DATA"].endswith("/final/8B/sidecar.jsonl.gz")

    _check_name_tags()
    print(f"[arm_config] selftest OK: {len(groups)} components "
          f"({', '.join(groups)}), {len(full.env)} variables resolved, "
          f"{len(full.selected)}-component arm round-trips through bash")
    return 0


# ---------------------------------------------------------------- CLI
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m rl.arm_config",
        description="Resolve ALGO=base[+penalty][+aux_loss] into the training environment.",
        epilog="Components are joined with '+', never a comma: `sbatch --export` is itself "
               "comma-separated, so ALGO=base,penalty reaches the job as ALGO=base.")
    ap.add_argument("--algo", default="base",
                    help="'+'-joined component ids, e.g. base+penalty+aux_loss (default: base)")
    ap.add_argument("--prompt", default="base",
                    help="sender prompt spec: base | strategies | single_strategy:<slug>")
    ap.add_argument("--policy-model", default="qwen3-4B",
                    help="policy model; its size fills {size} in a component value")
    ap.add_argument("--repo", default=str(_REPO_ROOT), help="repository root, fills {repo}")
    ap.add_argument("--strict-env", action="store_true",
                    help="make a stale knob of a non-selected component fatal instead of a warning")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--print-env", action="store_true",
                      help="write `export VAR=value` lines to stdout, diagnostics to stderr")
    mode.add_argument("--describe", action="store_true",
                      help="print the selected components and every variable with its source")
    mode.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()
    try:
        res = resolve(args.algo, args.prompt, policy_model=args.policy_model, repo=args.repo,
                      environ=os.environ)
        validate(res, os.environ, strict=args.strict_env)
    except (ArmConfigError, PromptSpecError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    for warning in res.warnings:
        print(f"WARNING: {warning}", file=sys.stderr)
    sys.stdout.write(emit(res, os.environ) if args.print_env else describe(res))
    return 0


if __name__ == "__main__":
    sys.exit(main())
