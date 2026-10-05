#!/usr/bin/env python3
"""The evaluation juror, selected by model id.

config/receiver/model/*.yaml names every juror an evaluation can play against:

    qwen3.5-35B         served locally with SGLang: the study's main juror, the juror the SLURM
                        launchers serve, and the audit judge whichever juror played the games
    DeepSeek-V4-Flash   queried through the API gateway, from a host that reaches it

Each yaml holds the id (exact case; the gateway rejects lowercase), transport, role and
`rollout_args` (the evaluation/rl_rollout.py flags that select it). Index on `id`, never the file
stem. `default: true` marks the main juror, not rl_rollout.py's default (its --receiver-api falls
back to the hosted transport; every caller here names a juror). rollout_args(id) is the one
id -> --receiver-api / --receiver-name mapping, used by rl_rollout.py --receiver and the launchers'
transport guard. The `api:` block is descriptive; --selftest pins it to rl_rollout.py. A
`serve.path` may name ${MODELS_DIR} (default <repo>/models), expanded when the yaml is loaded.

    python -m evaluation.receiver_models --list
    python -m evaluation.receiver_models --receiver DeepSeek-V4-Flash --print-rollout-args
    python -m evaluation.receiver_models --receiver qwen3.5-35B --print-transport
    python -m evaluation.receiver_models --selftest

--print-transport prints `unknown` and exits 3 for an id no yaml names (so a launcher can tell a
configured juror from a local model named by RECEIVER_MODEL_PATH); an invalid yaml exits 2.
"""
from __future__ import annotations

import argparse
import functools
import os
import sys
from pathlib import Path

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_CONFIG_DIR = _REPO_ROOT / "config" / "receiver" / "model"

#: exit code of --print-transport for an id no yaml names
UNKNOWN_EXIT = 3

_TOP_KEYS = {"id", "default", "transport", "role", "doc", "serve", "api", "rollout_args"}
_REQUIRED_KEYS = ("id", "transport", "role", "doc", "rollout_args")
_TRANSPORTS = ("sglang", "gateway")
_SERVE_KEYS = {"path", "tp"}
_API_KEYS = {"url_env", "key_env", "tier_env", "login_node_only", "sampling"}
# ordered: this is the argv order rollout_args() emits
_ROLLOUT_ARGS = (("receiver_api", "--receiver-api"), ("receiver_name", "--receiver-name"))


class ReceiverModelError(ValueError):
    """An unknown juror id, or a yaml that does not match the schema above."""


def _models_dir() -> str:
    """Model checkpoint root: MODELS_DIR, default <repo>/models."""
    return os.environ.get("MODELS_DIR") or str(_REPO_ROOT / "models")


def _expand_path(path) -> str:
    """Expand ${MODELS_DIR} (defaulted when unset) and any other env var in a yaml path."""
    return os.path.expandvars(str(path).replace("${MODELS_DIR}", _models_dir()))


def _check(cfg: dict, path: Path) -> dict:
    """Validate one yaml against the schema (an unknown key is a typo, not a comment)."""
    if not isinstance(cfg, dict):
        raise ReceiverModelError(f"{path.name}: expected a mapping")
    extra = set(cfg) - _TOP_KEYS
    if extra:
        raise ReceiverModelError(f"{path.name}: unknown key(s) {sorted(extra)}; "
                                 f"known: {sorted(_TOP_KEYS)}")
    for key in _REQUIRED_KEYS:
        if not cfg.get(key):
            raise ReceiverModelError(f"{path.name}: missing '{key}'")
    if cfg["transport"] not in _TRANSPORTS:
        raise ReceiverModelError(f"{path.name}: transport {cfg['transport']!r} is not one of "
                                 f"{list(_TRANSPORTS)}")
    if "serve" in cfg and set(cfg["serve"]) - _SERVE_KEYS:
        raise ReceiverModelError(f"{path.name}: unknown serve key(s) "
                                 f"{sorted(set(cfg['serve']) - _SERVE_KEYS)}")
    if "api" in cfg and set(cfg["api"]) - _API_KEYS:
        raise ReceiverModelError(f"{path.name}: unknown api key(s) "
                                 f"{sorted(set(cfg['api']) - _API_KEYS)}")
    known = {k for k, _ in _ROLLOUT_ARGS}
    extra = set(cfg["rollout_args"]) - known
    if extra:
        raise ReceiverModelError(f"{path.name}: unknown rollout_args key(s) {sorted(extra)}; "
                                 f"known: {sorted(known)}")
    if cfg["rollout_args"].get("receiver_api") != cfg["transport"]:
        raise ReceiverModelError(f"{path.name}: rollout_args.receiver_api "
                                 f"{cfg['rollout_args'].get('receiver_api')!r} does not match "
                                 f"transport {cfg['transport']!r}")
    return cfg


@functools.lru_cache(maxsize=None)
def load_all() -> dict:
    """{id: config} for every juror under config/receiver/model/, keyed by EXACT-case id."""
    out = {}
    for path in sorted(_CONFIG_DIR.glob("*.yaml")):
        with open(path, encoding="utf-8") as f:
            cfg = _check(yaml.safe_load(f), path)
        if cfg["id"] in out:
            raise ReceiverModelError(f"{path.name}: duplicate juror id {cfg['id']!r}")
        if "path" in cfg.get("serve", {}):
            cfg["serve"]["path"] = _expand_path(cfg["serve"]["path"])
        out[cfg["id"]] = cfg
    if not out:
        raise ReceiverModelError(f"no juror yaml under {_CONFIG_DIR}")
    defaults = [rid for rid, cfg in out.items() if cfg.get("default")]
    if len(defaults) != 1:
        raise ReceiverModelError(f"exactly one juror must carry `default: true`, got {defaults}")
    return out


def default_id() -> str:
    """The id of the juror marked `default: true`: the study's main juror."""
    return next(rid for rid, cfg in load_all().items() if cfg.get("default"))


def load(receiver_id: str) -> dict:
    """The config of `receiver_id` (exact case)."""
    jurors = load_all()
    if receiver_id not in jurors:
        raise ReceiverModelError(f"unknown juror id {receiver_id!r}; known (exact case): "
                                 f"{', '.join(jurors)}")
    return jurors[receiver_id]


def transport(receiver_id: str) -> str:
    """'sglang' (served locally) or 'gateway' (queried through the API gateway)."""
    return load(receiver_id)["transport"]


def rollout_args(receiver_id: str) -> list:
    """The evaluation/rl_rollout.py flags that select this juror, as an argv list."""
    cfg = load(receiver_id)["rollout_args"]
    argv = []
    for key, flag in _ROLLOUT_ARGS:
        if key in cfg:
            argv += [flag, str(cfg[key])]
    return argv


def describe(receiver_id: str) -> str:
    """A human-readable block: what this juror is, how it is reached, how to select it."""
    cfg = load(receiver_id)
    lines = [f"id:         {cfg['id']}" + ("  (default)" if cfg.get("default") else ""),
             f"transport:  {cfg['transport']}",
             f"role:       {cfg['role']}",
             f"rollout:    evaluation/rl_rollout.py {' '.join(rollout_args(receiver_id))}"]
    if "serve" in cfg:
        lines.append(f"serve:      {cfg['serve']['path']} (tp={cfg['serve']['tp']})")
    if "api" in cfg:
        api = cfg["api"]
        lines.append(f"api:        ${api['url_env']}")
        lines.append(f"            key {api['key_env']}, tier {api['tier_env']}"
                     + (", login node only" if api.get("login_node_only") else ""))
        lines.append(f"            sampling: {' '.join(str(api['sampling']).split())}")
    lines.append("doc:        " + "\n            ".join(str(cfg["doc"]).strip().splitlines()))
    return "\n".join(lines)


# ---------------------------------------------------------------- self-test
def _selftest() -> int:
    """Pin the yamls to the code they describe (no network, no key)."""
    import importlib
    import inspect

    jurors = load_all()
    assert len(jurors) >= 2, f"expected both jurors, got {list(jurors)}"
    for rid, cfg in jurors.items():
        assert cfg["id"] == rid, rid                      # round-trips with exact case
        assert load(rid) is cfg, rid
        assert transport(rid) in _TRANSPORTS, rid
        if rid.lower() != rid:                            # a lowercased id must not resolve
            try:
                load(rid.lower())
            except ReceiverModelError:
                pass
            else:
                raise AssertionError(f"{rid.lower()!r} resolved; the id is case-sensitive")

    local = load(default_id())
    assert local["transport"] == "sglang", local["transport"]
    assert rollout_args(local["id"]) == ["--receiver-api", "sglang"], rollout_args(local["id"])
    assert "serve" in local and local["serve"]["tp"] >= 1, local.get("serve")
    # serve.path is fully expanded (${MODELS_DIR} and any other variable) at load time.
    assert "$" not in local["serve"]["path"], local["serve"]["path"]

    hosted = [cfg for cfg in jurors.values() if cfg["transport"] == "gateway"]
    assert len(hosted) == 1, f"expected one hosted juror, got {len(hosted)}"
    hosted = hosted[0]
    assert rollout_args(hosted["id"]) == ["--receiver-api", "gateway",
                                          "--receiver-name", hosted["id"]], hosted["id"]

    # Imported here, not at module scope: the CLI the launchers call must not pay for verl.
    rollout = importlib.import_module("evaluation.rl_rollout")
    api = hosted["api"]
    assert api["url_env"] == rollout._GATEWAY_URL_ENV, (api["url_env"], rollout._GATEWAY_URL_ENV)
    assert hosted["id"] == rollout._DEFAULT_HOSTED_RECEIVER, rollout._DEFAULT_HOSTED_RECEIVER
    # key_env and url_env: the variables the driver actually reads, checked by reading them back,
    # and an unset one must fail with an error that names it.
    sentinel = "receiver-models-selftest"
    for var, require in ((api["key_env"], rollout._require_gateway_key),
                         (api["url_env"], rollout._require_gateway_url)):
        previous = os.environ.get(var)
        try:
            os.environ[var] = sentinel
            assert require("receiver") == sentinel, var
            os.environ.pop(var)
            try:
                require("receiver")
            except rollout.ReceiverCallFailed as e:
                assert var in str(e), (var, str(e))
            else:
                raise AssertionError(f"{require.__name__} accepted an unset {var}")
        finally:
            if previous is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = previous
    assert api["tier_env"] in inspect.getsource(rollout._gateway_receiver_chat), api["tier_env"]
    # sampling: the description says temperature/top_p/max_tokens 8192 and nothing else.
    sampling = rollout._hosted_receiver_sampling()
    assert set(sampling) == {"temperature", "top_p", "max_tokens"}, sorted(sampling)
    assert sampling["max_tokens"] == 8192, sampling["max_tokens"]
    for token in ("temperature", "top_p", "8192"):
        assert token in str(api["sampling"]), token

    print(f"[receiver_models] selftest OK: {len(jurors)} jurors "
          f"({', '.join(jurors)}), default {default_id()}; "
          f"the hosted yaml matches evaluation/rl_rollout.py")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description="The evaluation juror, selected by model id (config/receiver/model/).")
    ap.add_argument("--receiver", metavar="ID",
                    help="juror id, EXACT case (see --list)")
    ap.add_argument("--print-transport", action="store_true",
                    help="print 'sglang' or 'gateway'; an id no yaml names prints 'unknown' and "
                         f"exits {UNKNOWN_EXIT}, an invalid yaml exits 2")
    ap.add_argument("--print-rollout-args", action="store_true",
                    help="print the evaluation/rl_rollout.py flags that select this juror")
    ap.add_argument("--describe", action="store_true", help="print what this juror is")
    ap.add_argument("--list", action="store_true", help="list every configured juror")
    ap.add_argument("--selftest", action="store_true",
                    help="check the yamls against the code they describe and exit")
    args = ap.parse_args(argv)

    try:
        if args.selftest:
            return _selftest()
        if args.list:
            for rid, cfg in load_all().items():
                print(f"{rid}\t{cfg['transport']}\t{cfg['role']}"
                      + ("\t(default)" if cfg.get("default") else ""))
            return 0
        if not (args.print_transport or args.print_rollout_args or args.describe):
            ap.print_usage(sys.stderr)
            return 2
        if not args.receiver:
            ap.error("--receiver is required with "
                     "--print-transport / --print-rollout-args / --describe")
        if args.print_transport:
            # Load first so only an unknown id exits `unknown`; a broken yaml raises to the ERROR
            # path below, which launcher guards stop on.
            jurors = load_all()
            if args.receiver not in jurors:
                # Not an error: a caller may name a model it serves itself.
                print("unknown")
                return UNKNOWN_EXIT
            print(jurors[args.receiver]["transport"])
            return 0
        if args.print_rollout_args:
            print(" ".join(rollout_args(args.receiver)))
            return 0
        print(describe(args.receiver))
        return 0
    except ReceiverModelError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
