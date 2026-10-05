"""Exact per-step stratified receiver assignment for the mixture-of-receivers feature.

Pure stdlib logic. patch_verl.sh (10H) calls stamp_rows() on each
dataloader batch, before gen_batch derivation and repeat, so every optimizer step trains an exact
number of games against each configured receiver engine.

Modes:
  2-way (RECEIVER_MIX_MODEL + RECEIVER_MIX_FRAC): assign_receivers(), labels "api" / "local",
    split at int(frac*n + 0.5).
  N-way (RECEIVER_MIX_SPEC): assign_receivers_nway(), an ordered "label:weight,..." spec whose
    labels are hosted model ids; the reserved label "local" is the co-served model. E.g.
    'local:4,DeepSeek-V4-Flash:4,gpt-5-mini:4,grok-4-1-fast-reasoning:4' gives exactly 4 games
    each at the 16-game batch and 1 each at the 4-game smoke batch (see allocate()).

Assignment is per game and pre-repeat, so all n rollouts of a game (including 10G rejection-sampling
regen chunks) share one receiver and a belief offset between receiver engines cancels out of the
group-normalized GRPO advantage. Seeded by random.Random(f"{seed}:{global_step}"), so process- and
resume-stable. Validation batches skip 10H and fall back to rl.receiver_client.resolve_backend().

Selftest: `python -m rl.receiver_mix --selftest` (no GPU, no network).
"""
import hashlib
import random
import re

API = "api"      # 2-way label for the single hosted RECEIVER_MIX_MODEL
LOCAL = "local"  # reserved label, both modes: the co-served SGLang model

# Labels may only contain characters that survive a run name, a metric key and an env-var suffix.
_LABEL_RE = re.compile(r"^[A-Za-z0-9._/-]+$")


def slug(label: str) -> str:
    """Metric/run-name slug for a label: 'DeepSeek-V4-Flash' -> 'deepseek-v4-flash'.

    Byte-identical to the launcher's bash pipeline
        tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9\\n' '-' | sed 's/-\\{2,\\}/-/g; s/^-//; s/-$//'
    so existing run names and eval result stems are reproduced."""
    return re.sub(r"-{2,}", "-", re.sub(r"[^a-z0-9]", "-", label.lower())).strip("-")


def short(label: str) -> str:
    """Run-name token: lowercase alphanumerics, first 8 chars ('gpt-5-mini' -> 'gpt5mini').

    Not injective (grok-4-1-fast-reasoning and grok-4-1-fast-non-reasoning both give 'grok41fa'),
    so name_suffix() also carries a hash of the canonical spec."""
    return re.sub(r"[^a-z0-9]", "", label.lower())[:8]


def canonical(spec) -> str:
    """The spec re-rendered as 'label:weight,...' in spec order. This is what gets hashed and
    logged, so equivalent spellings ('+' separators, default weights, padding) agree."""
    return ",".join(f"{lab}:{w}" for lab, w in _arms(spec))


def parse_spec(spec: str) -> list:
    """'local:4,DeepSeek-V4-Flash:4,+gpt-5-mini' -> [('local',4), ('DeepSeek-V4-Flash',4),
    ('gpt-5-mini',1)].  '' -> [].

    The separator is ',' or '+'. `sbatch --export` splits on commas, so an inline comma spec is
    truncated to its first arm and trains 100% local under a `_mix4-` name. Export the variable
    in the shell and pass a bare --export=ALL, or use '+'.

    Order matters: it fixes the sha1 buckets for unstamped rows
    (rl.receiver_client.resolve_backend) and every derived name.

    Raises ValueError on: a label outside [A-Za-z0-9._/-]; a duplicate label or slug() (two arms
    would merge into one metric); a weight that is not a positive integer ('1:3' already gives
    25/75); the reserved label 'api'; a spec that is the single arm 'local' (the --export
    truncation signature; write RECEIVER_MIX_SPEC='' for off)."""
    if spec is None:
        return []
    spec = spec.strip()
    if not spec:
        return []
    arms, seen, seen_slugs = [], set(), {}
    for tok in (t.strip() for t in re.split(r"[,+]", spec)):
        if not tok:
            continue
        label, sep, wtxt = tok.partition(":")
        label, wtxt = label.strip(), wtxt.strip()
        if not label or not _LABEL_RE.match(label):
            raise ValueError(
                f"[receiver-mix] bad label {label!r} in RECEIVER_MIX_SPEC={spec!r} "
                "(allowed: letters, digits, dot, underscore, slash, dash)")
        if label == API:
            raise ValueError(
                f"[receiver-mix] {API!r} is the legacy 2-way sentinel, not a model id -- name the "
                "actual hosted model in RECEIVER_MIX_SPEC (or use RECEIVER_MIX_MODEL for the "
                "legacy 2-way path)")
        if label in seen:
            raise ValueError(f"[receiver-mix] duplicate label {label!r} in RECEIVER_MIX_SPEC")
        s = slug(label)
        if not s:
            raise ValueError(f"[receiver-mix] label {label!r} slugs to the empty string")
        if s in seen_slugs:
            raise ValueError(
                f"[receiver-mix] labels {seen_slugs[s]!r} and {label!r} share the slug {s!r} -- "
                "their per-arm reward-extra keys would collide and silently merge two arms")
        if sep and wtxt == "":
            raise ValueError(f"[receiver-mix] label {label!r} has a ':' but no weight")
        if not sep:
            w = 1
        else:
            if not re.fullmatch(r"[0-9]+", wtxt):
                raise ValueError(
                    f"[receiver-mix] weight {wtxt!r} for label {label!r} must be a positive "
                    "integer (fractional weights are not supported; use e.g. '1:3' for 25/75)")
            w = int(wtxt)
            if w <= 0:
                raise ValueError(f"[receiver-mix] weight for label {label!r} must be > 0 (got {w})")
        seen.add(label)
        seen_slugs[s] = label
        arms.append((label, w))
    if not arms:
        return []
    if len(arms) == 1 and arms[0][0] == LOCAL:
        raise ValueError(
            "[receiver-mix] RECEIVER_MIX_SPEC parsed to the single arm 'local'. That is almost "
            "certainly `sbatch --export` comma-truncation (--export is itself comma-separated): "
            "export RECEIVER_MIX_SPEC in your shell and pass a bare --export=ALL, or use '+' as "
            "the arm separator. If you really meant the mixture OFF, set RECEIVER_MIX_SPEC=''.")
    return arms


def _arms(spec) -> list:
    """Accept either a raw spec string or an already-parsed [(label, weight)] list."""
    return parse_spec(spec) if isinstance(spec, (str, type(None))) else list(spec)


def labels_of(spec) -> list:
    """Configured labels, in canonical (spec) order."""
    return [lab for lab, _ in _arms(spec)]


def hosted_labels(spec) -> list:
    """Configured labels except 'local': the arms that need a gateway preflight, an API key and a
    per-arm timeout/semaphore."""
    return [lab for lab in labels_of(spec) if lab != LOCAL]


def name_suffix(spec) -> str:
    """Run-name tag for an N-way mixture, e.g. '_mix4-deepseek-gpt5mini-grok41fa-3f9c1a'.

    N counts all arms; the short tokens name only hosted arms (local is identified by the
    JUDGE_TAG part of the run name). '_mix<N>-' cannot collide with the 2-way '_mix-<slug><frac>'.
    The 6-hex sha1 of the canonical spec keeps names unique: short() is not injective, and
    mixtures differing only in weights or arm order (which changes unstamped-row buckets) must not
    share a checkpoint directory."""
    arms = _arms(spec)
    if not arms:
        return ""
    canon = canonical(arms)
    h = hashlib.sha1(canon.encode()).hexdigest()[:6]
    toks = "-".join(short(lab) for lab in hosted_labels(arms))
    return f"_mix{len(arms)}-{toks}-{h}" if toks else f"_mix{len(arms)}-{h}"


def allocate(n: int, weights: list, global_step: int, seed: int) -> list:
    """Largest-remainder (Hamilton) apportionment of n seats over positive integer weights.

    Integer arithmetic, so sum(result) == n exactly. Weights that already sum to n reproduce
    themselves: 'local:4,A:4,B:4,C:4' gives [4,4,4,4] at batch 16 and [1,1,1,1] at the 4-game
    smoke batch, so the smoke runs the production path. Remainder ties are broken by a per-step
    seeded shuffle (the sort is stable), so when the batch is smaller than the arm count the arms
    take turns instead of the late ones starving."""
    k = len(weights)
    if k == 0:
        return []
    if n <= 0:
        return [0] * k
    tot = sum(weights)
    if tot <= 0:
        raise ValueError(f"[receiver-mix] weights must sum to a positive value (got {weights})")
    base = [(n * w) // tot for w in weights]
    rem = [(n * w) % tot for w in weights]
    order = list(range(k))
    random.Random(f"{seed}:{global_step}:arms").shuffle(order)
    order.sort(key=lambda i: -rem[i])
    for i in order[: n - sum(base)]:
        base[i] += 1
    return base


def assign_receivers(game_ids: list, global_step: int, seed: int, frac: float) -> dict:
    """2-way path: {game_id -> "api"|"local"} for one batch.

    Seeded shuffle of the unique game ids; the first int(frac*n + 0.5) go to "api" (not round(),
    whose banker's rounding gives 5@0.5 -> 2 instead of 3). Duplicate ids collapse to one entry."""
    unique = sorted(set(game_ids))  # sorted: independent of batch row order
    rng = random.Random(f"{seed}:{global_step}")
    rng.shuffle(unique)
    frac = min(1.0, max(0.0, float(frac)))
    n_api = int(frac * len(unique) + 0.5)
    return {gid: (API if i < n_api else LOCAL) for i, gid in enumerate(unique)}


def assign_receivers_nway(game_ids: list, global_step: int, seed: int, spec) -> dict:
    """N-way path: {game_id -> label} for one batch.

    Same shuffle as assign_receivers() (sorted unique ids, stream f"{seed}:{global_step}"), then
    contiguous slices sized by allocate(), one per arm in spec order."""
    arms = _arms(spec)
    if not arms:
        raise ValueError("[receiver-mix] assign_receivers_nway called with an empty spec")
    unique = sorted(set(game_ids))
    rng = random.Random(f"{seed}:{global_step}")
    rng.shuffle(unique)
    counts = allocate(len(unique), [w for _, w in arms], global_step, seed)
    out, off = {}, 0
    for (label, _), c in zip(arms, counts):
        for gid in unique[off:off + c]:
            out[gid] = label
        off += c
    assert off == len(unique), (
        f"[receiver-mix] apportionment covered {off} of {len(unique)} games -- refusing to leave "
        "games unassigned")
    return out


def stamp_rows(interaction_kwargs_rows, extra_info_rows, global_step: int, seed: int,
               frac: float = 0.5, spec: str = "") -> dict:
    """Stamp receiver_backend onto every batch row in place and return the assignment.

    A non-empty `spec` selects the N-way path, otherwise the 2-way `frac` path. Games are keyed by
    extra_info["index"] (the game_id from rl/game_rows.py). The key is written beside the payload
    on (a) each row's interaction_kwargs, which verl passes to start_interaction() in
    rl/persuasion_interaction.py, and (b) each row's extra_info, which reaches
    rl/reward_function.belief_reward.

    verl's RLHFDataset hoists extra_info["interaction_kwargs"] to a top-level field as the same
    dict object (rl_dataset.py __getitem__), and the rollout reads only that copy. This asserts the
    aliasing per row, so a verl change that breaks it fails here instead of silently training 100%
    local under a `_mix` name."""
    n = len(extra_info_rows)
    assert len(interaction_kwargs_rows) == n, \
        f"[receiver-mix] row count mismatch: {len(interaction_kwargs_rows)} interaction_kwargs vs {n} extra_info"
    game_ids = [ei["index"] for ei in extra_info_rows]
    arms = _arms(spec)
    assignment = (assign_receivers_nway(game_ids, global_step, seed, arms) if arms
                  else assign_receivers(game_ids, global_step, seed, frac))
    for i in range(n):
        ik, ei = interaction_kwargs_rows[i], extra_info_rows[i]
        assert ik is ei.get("interaction_kwargs"), (
            "[receiver-mix] verl no longer aliases the top-level interaction_kwargs to "
            "extra_info['interaction_kwargs'] (row {}) -- the rollout would not see the "
            "assignment; refusing to train a mislabeled mixture run".format(i))
        backend = assignment[ei["index"]]
        ik["receiver_backend"] = backend
        ei["receiver_backend"] = backend
    return assignment


def step_metrics(assignment: dict, spec="") -> dict:
    """Per-step wandb scalars for fit()'s `metrics` dict (patch_verl 10H).

    Every configured label emits every step (0.0 when it drew no games), so no series blinks out.
    receiver_mix/n_api_games (all non-local games), n_local_games and n_arms are emitted in both
    modes so existing 2-way charts continue."""
    arms = _arms(spec)
    total = len(assignment)
    n_local = sum(1 for v in assignment.values() if v == LOCAL)
    out = {
        "receiver_mix/n_api_games": float(total - n_local),
        "receiver_mix/n_local_games": float(n_local),
        "receiver_mix/n_arms": float(len(arms) if arms else 2),
    }
    for label in labels_of(arms):
        out[f"receiver_mix/n_{slug(label)}_games"] = float(
            sum(1 for v in assignment.values() if v == label))
    return out


def format_counts(assignment: dict, spec="") -> str:
    """One-line job-log summary with raw labels: 'local=4 DeepSeek-V4-Flash=4 ... (n=16, arms=4)'."""
    arms = _arms(spec)
    order = labels_of(arms) if arms else [LOCAL, API]
    counts = {lab: sum(1 for v in assignment.values() if v == lab) for lab in order}
    for v in assignment.values():  # surface any label outside the configured set
        if v not in counts:
            counts[v] = sum(1 for x in assignment.values() if x == v)
    body = " ".join(f"{lab}={counts[lab]}" for lab in counts)
    return f"{body} (n={len(assignment)}, arms={len(arms) if arms else 2})"


# ---------------------------------------------------------------- selftest

_SPEC4 = "local:4,DeepSeek-V4-Flash:4,gpt-5-mini:4,grok-4-1-fast-reasoning:4"


def _selftest() -> None:  # noqa: C901 - a flat list of independent assertions reads better here
    import collections
    import os
    import subprocess
    import sys

    def ok(name):
        print(f"  ok  {name}")

    # 1. Exact counts at both batch sizes from one spec string.
    w4 = [4, 4, 4, 4]
    for step in range(60):
        assert allocate(16, w4, step, 2026) == [4, 4, 4, 4], step
        assert allocate(4, w4, step, 2026) == [1, 1, 1, 1], step
        assert allocate(16, [1, 1, 1, 1], step, 2026) == [4, 4, 4, 4], step
        assert allocate(16, [1, 3], step, 2026) == [4, 12], step
        assert allocate(8, [1, 1, 1, 1], step, 2026) == [2, 2, 2, 2], step
    ok("exact counts: 4/4/4/4 @16, 1/1/1/1 @4 (SMOKE), 4/12 for 1:3")

    # 2. Sum invariant for arbitrary n and weights.
    rng = random.Random(11)
    for _ in range(3000):
        k = rng.randint(1, 6)
        ws = [rng.randint(1, 9) for _ in range(k)]
        n = rng.randint(0, 40)
        got = allocate(n, ws, rng.randint(0, 999), 2026)
        assert len(got) == k and sum(got) == n and all(c >= 0 for c in got), (n, ws, got)
    ok("sum invariant: sum(allocate(n, w)) == n for all n, w")

    # 3. Cross-process determinism under PYTHONHASHSEED, so the rollout worker and the reward-side
    #    fallback agree on each game.
    snippet = (
        "import sys, json; sys.path.insert(0, %r);"
        "import rl.receiver_mix as m;"
        "print(json.dumps([[str(k), v] for k, v in "
        "sorted(m.assign_receivers_nway(list(range(16)), 5, 2026, %r).items(), key=lambda x: str(x[0]))]))"
        % (os.getcwd(), _SPEC4))
    outs = set()
    for i in range(6):
        env = dict(os.environ, PYTHONHASHSEED=str(i * 7 + 1))
        r = subprocess.run([sys.executable, "-c", snippet], capture_output=True, text=True, env=env)
        assert r.returncode == 0, r.stderr[-2000:]
        outs.add(r.stdout.strip())
    assert len(outs) == 1, f"assignment is NOT process-stable: {len(outs)} distinct results"
    ok("cross-process determinism under PYTHONHASHSEED salting (6 subprocesses)")

    # 4. Resume stability: (seed, global_step) fixes the assignment.
    a = assign_receivers_nway(list(range(16)), 42, 2026, _SPEC4)
    for _ in range(20):
        assert assign_receivers_nway(list(range(16)), 42, 2026, _SPEC4) == a
    assert assign_receivers_nway(list(range(16)), 43, 2026, _SPEC4) != a, \
        "different steps must give different assignments"
    ok("resume-stability: (seed, step) fully determines the assignment")

    # 5. Independent of batch row order.
    ids = list(range(16))
    for _ in range(30):
        shuffled = ids[:]
        random.Random(_).shuffle(shuffled)
        assert assign_receivers_nway(shuffled, 9, 2026, _SPEC4) == \
            assign_receivers_nway(ids, 9, 2026, _SPEC4)
    ok("order-independence: batch row order cannot change the assignment")

    # 6. No starvation when the batch is smaller than the arm count.
    seen = {i: 0 for i in range(4)}
    for step in range(400):
        for i, c in enumerate(allocate(2, w4, step, 2026)):
            seen[i] += c
    assert all(v > 0 for v in seen.values()), f"an arm was starved over 400 steps: {seen}"
    assert sum(seen.values()) == 800
    ok(f"no starvation at batch 2 over 400 steps: {list(seen.values())}")

    # 7. 2-way split: floor(+0.5) counts and the frac 0/1 extremes.
    g5 = assign_receivers(list(range(5)), 0, 2026, 0.5)
    assert sum(1 for v in g5.values() if v == API) == 3, "floor(+0.5) rule broken: 5@0.5 must be 3"
    assert set(assign_receivers(list(range(8)), 3, 2026, 0.0).values()) == {LOCAL}
    assert set(assign_receivers(list(range(8)), 3, 2026, 1.0).values()) == {API}
    ok("2-way split: 5@0.5 -> 3 api, frac 0/1")

    # 8. parse_spec equivalences and rejections.
    assert parse_spec("") == [] and parse_spec(None) == [] and parse_spec("   ") == []
    assert parse_spec("local:4,gpt-5-mini:4") == parse_spec("local:4+gpt-5-mini:4")
    assert parse_spec(" local : 4 , gpt-5-mini:4 ") == [("local", 4), ("gpt-5-mini", 4)]
    assert parse_spec("local,gpt-5-mini") == [("local", 1), ("gpt-5-mini", 1)]
    assert parse_spec(_SPEC4) == [("local", 4), ("DeepSeek-V4-Flash", 4),
                                  ("gpt-5-mini", 4), ("grok-4-1-fast-reasoning", 4)]
    for bad, why in [
        ("local", "the --export comma-truncation tripwire"),
        ("local:4,local:4", "duplicate label"),
        ("gpt-5-mini:4,gpt.5.mini:4", "slug collision"),
        ("local:4,api:4", "the reserved legacy sentinel"),
        ("local:0,gpt-5-mini:4", "zero weight"),
        ("local:-1,gpt-5-mini:4", "negative weight"),
        ("local:0.5,gpt-5-mini:4", "fractional weight"),
        ("local:,gpt-5-mini:4", "empty weight"),
        ("local:4,bad label:4", "space in label"),
        ("local:4,bad$label:4", "illegal character in label"),
    ]:
        try:
            parse_spec(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"parse_spec({bad!r}) should have raised -- {why}")
    # a lone non-local arm is legitimate (an all-hosted mixture)
    assert parse_spec("gpt-5-mini:4") == [("gpt-5-mini", 4)]
    ok("parse_spec: ','/'+' equivalence, default weight, and 10 rejections incl. the export trap")

    # 9. slug, short, name_suffix.
    for label, want in [("DeepSeek-V4-Flash", "deepseek-v4-flash"), ("gpt-5-mini", "gpt-5-mini"),
                        ("grok-4-1-fast-reasoning", "grok-4-1-fast-reasoning"),
                        ("local", "local"), ("qwen3.5-35B", "qwen3-5-35b"),
                        ("A--B", "a-b"), ("-lead-", "lead")]:
        assert slug(label) == want, (label, slug(label), want)
    for label, want in [("DeepSeek-V4-Flash", "deepseek"), ("gpt-5-mini", "gpt5mini"),
                        ("grok-4-1-fast-reasoning", "grok41fa"), ("local", "local")]:
        assert short(label) == want, (label, short(label), want)
    sfx = name_suffix(_SPEC4)
    assert sfx.startswith("_mix4-") and re.fullmatch(r"_mix4-[a-z0-9-]+", sfx), sfx
    assert name_suffix("") == ""
    # distinctness where short() alone collides, and across weights / arm order
    assert name_suffix(_SPEC4) != name_suffix(
        _SPEC4.replace("grok-4-1-fast-reasoning", "grok-4-1-fast-non-reasoning"))
    assert name_suffix(_SPEC4) != name_suffix(_SPEC4.replace("local:4", "local:8"))
    assert name_suffix("local:4,gpt-5-mini:4") != name_suffix("gpt-5-mini:4,local:4")
    # and never collides with the 2-way '_mix-<slug><frac>' form
    assert not name_suffix(_SPEC4).startswith("_mix-")
    assert canonical(" local:4 + gpt-5-mini ") == "local:4,gpt-5-mini:1"
    ok(f"slug/short/name_suffix: {sfx}  (collision-free vs weights, order, and legacy '_mix-')")

    # 10. step_metrics and format_counts in both modes.
    asg = assign_receivers_nway(list(range(16)), 1, 2026, _SPEC4)
    m = step_metrics(asg, _SPEC4)
    for label in labels_of(_SPEC4):
        assert m[f"receiver_mix/n_{slug(label)}_games"] == 4.0, (label, m)
    assert m["receiver_mix/n_local_games"] == 4.0 and m["receiver_mix/n_api_games"] == 12.0
    assert m["receiver_mix/n_arms"] == 4.0
    # The 'local' arm's per-label key is n_local_games, the always-emitted key (same value).
    assert m["receiver_mix/n_local_games"] == 4.0
    assert set(m) == {f"receiver_mix/n_{slug(x)}_games" for x in labels_of(_SPEC4)} | \
        {"receiver_mix/n_api_games", "receiver_mix/n_local_games", "receiver_mix/n_arms"}
    # an arm that draws no games still emits, and per-arm counts cover the batch
    m2 = step_metrics(assign_receivers_nway([0], 1, 2026, _SPEC4), _SPEC4)
    per_arm = {lab: m2[f"receiver_mix/n_{slug(lab)}_games"] for lab in labels_of(_SPEC4)}
    assert len(per_arm) == len(labels_of(_SPEC4)), per_arm
    assert sum(per_arm.values()) == 1.0 and sorted(per_arm.values()) == [0.0, 0.0, 0.0, 1.0], per_arm
    assert m2["receiver_mix/n_api_games"] + m2["receiver_mix/n_local_games"] == 1.0
    # 2-way mode: only n_api_games, n_local_games, n_arms
    ml = step_metrics(assign_receivers(list(range(16)), 7, 2026, 0.5), "")
    assert set(ml) == {"receiver_mix/n_api_games", "receiver_mix/n_local_games",
                       "receiver_mix/n_arms"}, ml
    assert ml["receiver_mix/n_api_games"] == 8.0 and ml["receiver_mix/n_local_games"] == 8.0
    fc = format_counts(asg, _SPEC4)
    assert "grok-4-1-fast-reasoning=4" in fc and "(n=16, arms=4)" in fc, fc
    assert format_counts(assign_receivers(list(range(4)), 1, 2026, 0.5), "") == \
        "local=2 api=2 (n=4, arms=2)"
    ok("step_metrics: every arm every step (incl. 0), legacy keys unchanged; format_counts")

    # 11. stamp_rows end to end, including the aliasing guard.
    def mkrows(n):
        eis = []
        for i in range(n):
            ik = {"name": "persuasion", "payload": "{}"}
            eis.append({"index": i, "interaction_kwargs": ik})
        return [ei["interaction_kwargs"] for ei in eis], eis

    iks, eis = mkrows(16)
    asg = stamp_rows(iks, eis, global_step=2, seed=2026, spec=_SPEC4)
    assert all(ik["receiver_backend"] == ei["receiver_backend"] for ik, ei in zip(iks, eis))
    assert collections.Counter(asg.values()) == \
        collections.Counter({lab: 4 for lab in labels_of(_SPEC4)}), collections.Counter(asg.values())
    assert set(asg.values()) <= set(labels_of(_SPEC4)), "an unconfigured label was stamped"
    iks, eis = mkrows(4)
    stamp_rows(iks, eis, global_step=2, seed=2026, frac=0.5)   # 2-way path via default spec=""
    assert set(e["receiver_backend"] for e in eis) == {API, LOCAL}
    iks, eis = mkrows(4)
    eis[2]["interaction_kwargs"] = dict(eis[2]["interaction_kwargs"])  # break the aliasing
    try:
        stamp_rows(iks, eis, global_step=2, seed=2026, spec=_SPEC4)
    except AssertionError:
        pass
    else:
        raise AssertionError("stamp_rows must fail LOUDLY when verl stops aliasing interaction_kwargs")
    ok("stamp_rows: 4/4/4/4 stamped on both dicts; aliasing guard still fires")

    print("\nreceiver_mix selftest: ALL PASSED")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
        print("usage: python -m rl.receiver_mix --selftest")
