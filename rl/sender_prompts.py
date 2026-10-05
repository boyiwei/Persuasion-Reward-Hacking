"""The sender prompt, selected by one spec string.

    base                     the plain role prompt
    strategies               role prompt + the 42-technique ALLOWED / FORBIDDEN guide
    single_strategy:<slug>   role prompt + one system line allowing exactly one technique

One file per mode under rl/config/prompt/ says which sender yaml each game domain renders and how
the mode is named on disk (GRPO data root, run name, wandb tag, result stem, manifest arm key).
`resolve_sender_prompt(domain, spec)` returns the config `agents.agent_quality.AgentQuality` takes.

`single_strategy` has no yaml of its own: the base template gets one extra system line
(`allowance_line` in rl/config/prompt/single_strategy.yaml, with {name} / {definition} verbatim
from rl.strategy_audit.taxonomy for the domain) and is then parsed like any other sender yaml.

    python -m rl.sender_prompts --selftest
    python -m rl.sender_prompts --show --domain old-bailey --prompt single_strategy:framing
    python -m rl.sender_prompts --check strategies            # exit 2 on an unknown spec
    python -m rl.sender_prompts --print name-suffix --prompt single_strategy:framing
"""
from __future__ import annotations

import argparse
import functools
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_DIR = _REPO_ROOT / "rl" / "config" / "prompt"

MODES = ("base", "strategies", "single_strategy")
DOMAINS = ("old-bailey", "house-showing", "nutrition")

# The system -> user message boundary of a sender yaml. The single-strategy line goes right before
# it, as the last line of the system `content: |` block (8-space indent).
_BOUNDARY = "\n\n    - role: user"
_LINE_INDENT = " " * 8


class PromptSpecError(ValueError):
    """An unknown mode, a missing / unknown technique slug, or an unknown domain."""


@dataclass(frozen=True)
class PromptSpec:
    mode: str
    slug: Optional[str] = None

    def __str__(self) -> str:
        return self.mode if self.slug is None else f"{self.mode}:{self.slug}"


@functools.lru_cache(maxsize=None)
def mode_config(mode: str) -> dict:
    """rl/config/prompt/<mode>.yaml as a dict."""
    if mode not in MODES:
        raise PromptSpecError(f"unknown sender prompt mode {mode!r}; known: {', '.join(MODES)}")
    with open(_CONFIG_DIR / f"{mode}.yaml", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg.get("id") != mode:
        raise PromptSpecError(f"{mode}.yaml declares id {cfg.get('id')!r}")
    return cfg


def _technique_slugs() -> list:
    from rl.strategy_audit.taxonomy import slugs
    return slugs()


def parse_spec(text) -> PromptSpec:
    """'base' | 'strategies' | 'single_strategy:<slug>' -> PromptSpec (a PromptSpec passes through)."""
    if isinstance(text, PromptSpec):
        return text
    mode, sep, slug = str(text).strip().partition(":")
    cfg = mode_config(mode)
    if cfg.get("argument") == "slug":
        if not slug:
            raise PromptSpecError(f"{mode} needs a technique: {mode}:<slug> "
                                  f"(slugs: rl/strategy_audit/taxonomy.py)")
        if slug not in _technique_slugs():
            raise PromptSpecError(f"unknown technique slug {slug!r}; known: "
                                  f"{', '.join(_technique_slugs())}")
        return PromptSpec(mode, slug)
    if sep:
        raise PromptSpecError(f"{mode} takes no argument, got {text!r}")
    return PromptSpec(mode)


def _check_domain(domain: str) -> None:
    if domain not in DOMAINS:
        raise PromptSpecError(f"unknown domain {domain!r}; known: {', '.join(DOMAINS)}")


def template_path(domain: str, spec) -> Path:
    """The sender yaml this (domain, spec) renders from."""
    _check_domain(domain)
    return _REPO_ROOT / mode_config(parse_spec(spec).mode)["templates"][domain]


def allowance_line(domain: str, slug: str) -> str:
    """The single-strategy system line for `slug`, worded for `domain`."""
    from rl.strategy_audit.taxonomy import by_slug
    entry = by_slug(domain)[slug]
    return mode_config("single_strategy")["allowance_line"].format(
        name=entry["name"], definition=entry["definition"])


def render_yaml_text(domain: str, spec) -> str:
    """The sender yaml text for (domain, spec)."""
    spec = parse_spec(spec)
    text = template_path(domain, spec).read_text(encoding="utf-8")
    if spec.slug is None:
        return text
    if text.count(_BOUNDARY) != 1:
        raise PromptSpecError(f"expected exactly one system->user boundary in "
                              f"{template_path(domain, spec).name}, found {text.count(_BOUNDARY)}")
    return text.replace(
        _BOUNDARY, f"\n\n{_LINE_INDENT}{allowance_line(domain, spec.slug)}{_BOUNDARY}", 1)


def resolve_sender_prompt(domain: str, spec):
    """The sender config for AgentQuality(cfg, ModelAPI())."""
    from omegaconf import OmegaConf
    spec = parse_spec(spec)
    if spec.slug is None:
        return OmegaConf.load(template_path(domain, spec))
    return OmegaConf.create(render_yaml_text(domain, spec))


def system_message(domain: str, spec) -> str:
    """The rendered system message (the yaml's system content, verbatim)."""
    messages = resolve_sender_prompt(domain, spec).prompts.messages
    assert messages[0].role == "system", f"first message of {spec} is {messages[0].role!r}"
    return messages[0].content


def _named(spec, key: str) -> str:
    spec = parse_spec(spec)
    return mode_config(spec.mode)[key].format(slug=spec.slug)


def data_suffix(spec) -> str:
    """Suffix of the GRPO parquet root (rl[_sftsplit]<suffix>)."""
    return _named(spec, "data_suffix")


def name_suffix(spec) -> str:
    """Suffix the prompt contributes to the run / checkpoint name."""
    return _named(spec, "name_suffix")


def wandb_tag(spec) -> str:
    return _named(spec, "wandb_tag")


def arm_key(spec) -> str:
    """Key of this prompt arm in result manifests: base | strat | only_<slug>."""
    return _named(spec, "arm_key")


def result_stem(domain: str, spec) -> str:
    """The sender-prompt token of a result file stem (`..._sender-<stem>__recv_...`)."""
    spec = parse_spec(spec)
    stem = template_path(domain, spec).stem
    return stem if spec.slug is None else f"{stem}_only_{spec.slug}"


def all_specs(domain: str) -> list:
    """Every spec a domain can render: base, strategies, one single_strategy per technique."""
    _check_domain(domain)
    return ([PromptSpec("base"), PromptSpec("strategies")]
            + [PromptSpec("single_strategy", s) for s in _technique_slugs()])


# ---------------------------------------------------------------- self-test
def _selftest() -> int:
    n_rendered = 0
    for domain in DOMAINS:
        base_text = render_yaml_text(domain, "base")
        assert base_text.count(_BOUNDARY) == 1, f"{domain}: base template boundary"
        base_cfg = resolve_sender_prompt(domain, "base")
        for spec in all_specs(domain):
            cfg = resolve_sender_prompt(domain, spec)
            system = cfg.prompts.messages[0].content
            n_rendered += 1
            if spec.slug is not None:
                line = allowance_line(domain, spec.slug)
                assert "\n" not in line, f"{spec}: allowance line spans lines"
                # only the system message differs from base: a blank line, then the one line
                assert system == base_cfg.prompts.messages[0].content + "\n" + line + "\n", \
                    str(spec)
                assert cfg.prompts.messages[1] == base_cfg.prompts.messages[1], str(spec)
                assert cfg.language_model.max_tokens == base_cfg.language_model.max_tokens
            assert ":" not in data_suffix(spec) + name_suffix(spec) + result_stem(domain, spec), \
                f"{spec}: a name or path token carries ':'"

    assert [data_suffix(s) for s in ("base", "strategies", "single_strategy:framing")] == \
        ["", "_strategies", "_only_framing"]
    assert [name_suffix(s) for s in ("base", "strategies", "single_strategy:framing")] == \
        ["", "_strategies", "_only_framing"]
    assert [arm_key(s) for s in ("base", "strategies", "single_strategy:framing")] == \
        ["base", "strat", "only_framing"]
    assert result_stem("old-bailey", "strategies") == "initial_base_oldbailey_strategies"
    assert result_stem("old-bailey", "single_strategy:framing") == \
        "initial_base_oldbailey_only_framing"
    assert str(parse_spec("single_strategy:framing")) == "single_strategy:framing"
    for bad in ("not_a_mode", "single_strategy", "single_strategy:not_a_technique", "base:framing",
                ""):
        try:
            parse_spec(bad)
        except PromptSpecError:
            continue
        raise AssertionError(f"parse_spec accepted {bad!r}")
    print(f"[sender_prompts] selftest OK: {n_rendered} system prompts rendered, "
          f"{len(_technique_slugs())} techniques x {len(DOMAINS)} domains")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Sender prompt resolver "
                                             "(base | strategies | single_strategy:<slug>)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--check", metavar="SPEC", help="validate a spec; exit 2 if it is unknown")
    ap.add_argument("--show", action="store_true", help="print the rendered sender yaml")
    ap.add_argument("--print", dest="print_", choices=["data-suffix", "name-suffix", "wandb-tag",
                                                       "result-stem", "arm-key", "template"])
    ap.add_argument("--prompt", default="base")
    ap.add_argument("--domain", default="old-bailey", choices=DOMAINS)
    args = ap.parse_args(argv)
    try:
        if args.selftest:
            return _selftest()
        if args.check is not None:
            print(parse_spec(args.check))
            return 0
        spec = parse_spec(args.prompt)
        if args.show:
            sys.stdout.write(render_yaml_text(args.domain, spec))
            return 0
        if args.print_:
            print({"data-suffix": lambda: data_suffix(spec),
                   "name-suffix": lambda: name_suffix(spec),
                   "wandb-tag": lambda: wandb_tag(spec),
                   "arm-key": lambda: arm_key(spec),
                   "result-stem": lambda: result_stem(args.domain, spec),
                   "template": lambda: str(template_path(args.domain, spec)
                                           .relative_to(_REPO_ROOT))}[args.print_]())
            return 0
    except PromptSpecError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2
    ap.print_usage(sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
