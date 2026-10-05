"""Auxiliary conduct-probe CE loss for sender-GRPO — pure logic (no ray, no verl imports).

AUX_CE=1 adds to every GRPO optimizer step coeff * CE on the gold true/false answer token of in-role
probe prompts, over independent channels (TERMS_KNOWN): fabrication ("is this STATEMENT a true fact
about this case?"), the STRATEGY_TERMS slugs ("did your arguments use technique X?", gold = the 35B
strategy-audit verdict on that game's trajectory) and illegitimate_any ("were ALL techniques
legitimate?"). The coarse channel exists because a narrower one leaves the policy room to displace:
fabrication-only sped the deceptive block up and slowed the coercive one with no net conduct gain.

Wiring (rl/patch_verl.sh): 10I `attach()`es row-aligned aux_* tensors in ray_trainer.fit() right
before update_actor (row order final, rejection sampling done); 13 runs a teacher-forced aux forward
+ scaled backward per micro-batch in dp_actor.update_policy(), before the policy forward so the aux
logits graph is freed first. AUX_CE=0 -> no keys -> 13 is a no-op.

Scheduling is positional (schedule, term_plan): every DP rank issues the identical [aux fwd+bwd,
policy fwd+bwd] sequence, which FSDP2 collectives require; data-dependent scheduling would deadlock.
Adding terms changes which item fills a row, never how many rows run. AUX_CE_SAMPLING="random" (the
launcher's pin) draws from global per-(term, label) pools, exactly 1:1 per channel per step. "game"
(fabrication-only) looks items up by game id, so its label mix tracks game coverage. Items cycle
without replacement per pool key (_cycle); cursors live in trainer memory, so a resume starts a
fresh pass.

Sidecar (datasets/old_bailey/aux_loss chain): gzip JSONL of {game_id, item_id, term?, label,
messages, ...} plus ../_meta.json with the chat-template sha. A row without "term" is fabrication.

  .venv/bin/python -m rl.aux_ce --selftest
"""
import collections
import gzip
import hashlib
import json
import os
import random
import re
from dataclasses import dataclass, field
from pathlib import Path

RESP_WIDTH = 8      # the widest surface below is 7 tokens on Qwen3-4B/8B/14B
_ANS_OPEN, _ANS_CLOSE = "<answer>", "</answer>"
LABELS_ALL = ("true", "false")
RESP_TEXT = {lb: f"{_ANS_OPEN}{lb}{_ANS_CLOSE}" for lb in LABELS_ALL}
# char span of the bare label word inside RESP_TEXT ("<answer>" is 8 chars)
_LABEL_SPAN = {lb: (len(_ANS_OPEN), len(_ANS_OPEN) + len(lb)) for lb in LABELS_ALL}
assert _LABEL_SPAN["true"] == (8, 12) and _LABEL_SPAN["false"] == (8, 13), \
    "label spans drifted from the pre-multi-term literals"

DEFAULT_TERM = "fabrication"
# Strategy channels (taxonomy slugs). Gold = the 35B strategy-audit verdict on the game's own
# 3-round trajectory (1 -> "true" = the technique was used).
STRATEGY_TERMS = ("belief_coercion", "misrepresentation", "procedural_manipulation",
                  "authority_impersonation", "discouragement")
# Coarse channel: one binary over the whole forbidden set, rendered from the taxonomy's FORBIDDEN
# block. Not a taxonomy slug, so it sits outside STRATEGY_TERMS.
COARSE_TERM = "illegitimate_any"
# Append-only (STRATEGY_TERMS too): a term's index drives the positional slot schedule (term_plan),
# so reordering silently changes what an existing multi-term run trained on.
TERMS_KNOWN = (DEFAULT_TERM,) + STRATEGY_TERMS + (COARSE_TERM,)
# (parity-0 label, parity-1 label); parity 0 is always "violation present". fabrication keeps its
# pre-multi-term ("false", "true") mapping (T=1 bit-compat, see term_plan). COARSE_TERM shares it
# because it asks in the legitimacy direction, where "false" is the violation.
_INVERTED_TERMS = (DEFAULT_TERM, COARSE_TERM)
LABEL_ORDER = {t: (("false", "true") if t in _INVERTED_TERMS else ("true", "false"))
               for t in TERMS_KNOWN}
AUX_KEYS = ("aux_input_ids", "aux_attention_mask", "aux_position_ids",
            "aux_responses", "aux_response_mask", "aux_valid", "aux_term_id")
SAMPLING_MODES = ("game", "random")
_GLOBAL_GID = "*"   # cursor/perm key for the sampling="random" global pools (never a real gid)

# --- AUX_CE_KIND_BALANCE: per-batch balance over the sidecar's item kinds --------------------
# Every matched pair belongs to one block: "a" (claim_real TRUE, fabrication FALSE) or "b"
# (paraphrase_real TRUE, altered_real FALSE, the hard negatives). Plain "random" draws TRUE and
# FALSE from independent pools, so a pair can straddle blocks and lose its matched contrast. Kind
# balance assigns the block positionally, so all four kinds appear equally in every rank chunk.
KIND_BLOCK = {"claim_real": "a", "fabrication": "a",
              "paraphrase_real": "b", "altered_real": "b"}
BLOCKS = ("a", "b")


def term_labels(term: str) -> tuple:
    """(parity-0, parity-1) answer-surface labels for `term`."""
    if term not in LABEL_ORDER:
        raise ValueError(f"unknown aux-CE term {term!r} (known: {TERMS_KNOWN})")
    return LABEL_ORDER[term]


def parse_terms(raw, present=None) -> tuple:
    """AUX_CE_TERMS -> de-duplicated term tuple in TERMS_KNOWN order (so "a,b" and "b,a" give one
    schedule and run name). Empty -> the sidecar's `present` terms, else (DEFAULT_TERM,).
    Separators are comma, whitespace or '+'; use '+' under `sbatch --export=`, which splits on commas.
    """
    if raw is None or not str(raw).strip():
        names = set(present) if present is not None else {DEFAULT_TERM}
    else:
        listed = [t for t in re.split(r"[,\s+]+", str(raw).strip()) if t]
        if len(set(listed)) != len(listed):
            raise ValueError(f"AUX_CE_TERMS has duplicates: {listed}")
        names = set(listed)
    bad = names - set(TERMS_KNOWN)
    if bad:
        raise ValueError(f"unknown AUX_CE_TERMS entries {sorted(bad)} (known: {TERMS_KNOWN})")
    if not names:
        raise ValueError("AUX_CE_TERMS resolved to an empty term set")
    terms = tuple(t for t in TERMS_KNOWN if t in names)
    # verl's reduce_metrics (verl/utils/metric/utils.py) uses np.max/np.min for any key containing
    # 'max'/'min', which would turn the additive actor/aux_*__<term> series into a min/max.
    for t in terms:
        assert "max" not in t and "min" not in t, f"term {t!r} contains 'max'/'min'"
    return terms


def config_from_env() -> dict:
    """Strict parse of the AUX_CE_* env knobs (fail loud on any malformed value)."""
    on = os.environ.get("AUX_CE", "0")
    if on not in ("0", "1"):
        raise ValueError(f"AUX_CE must be '0' or '1', got {on!r}")
    cfg = {"on": on == "1",
           "data": os.environ.get("AUX_CE_DATA", ""),
           "coeff": float(os.environ.get("AUX_CE_COEFF", "0.5")),
           "max_prompt": int(os.environ.get("AUX_CE_MAX_PROMPT", "8192")),
           "frac": float(os.environ.get("AUX_CE_ROW_FRAC", "1.0")),
           "seed": int(os.environ.get("AUX_CE_SEED", "2026")),
           "sampling": os.environ.get("AUX_CE_SAMPLING", "random"),
           "soft": os.environ.get("AUX_CE_SOFT", "0"),
           "kind_balance": os.environ.get("AUX_CE_KIND_BALANCE", "1"),
           "terms": os.environ.get("AUX_CE_TERMS", "").strip()}
    if cfg["soft"] not in ("0", "1"):
        raise ValueError(f"AUX_CE_SOFT must be '0' or '1', got {cfg['soft']!r}")
    cfg["soft"] = cfg["soft"] == "1"
    if cfg["kind_balance"] not in ("0", "1"):
        raise ValueError(
            f"AUX_CE_KIND_BALANCE must be '0' or '1', got {cfg['kind_balance']!r}")
    cfg["kind_balance"] = cfg["kind_balance"] == "1"
    if cfg["sampling"] not in SAMPLING_MODES:
        raise ValueError(f"AUX_CE_SAMPLING must be one of {SAMPLING_MODES}, "
                         f"got {cfg['sampling']!r}")
    # Validated even at AUX_CE=0 so a typo fails early. Block assignment is positional inside a
    # rank chunk, which only "random" defines.
    if cfg["kind_balance"] and cfg["sampling"] != "random":
        raise ValueError("AUX_CE_KIND_BALANCE=1 requires AUX_CE_SAMPLING=random "
                         f"(got {cfg['sampling']!r})")
    # Also validated at AUX_CE=0.
    if cfg["terms"]:
        cfg["terms"] = parse_terms(cfg["terms"])
        if cfg["sampling"] == "game" and cfg["terms"] != (DEFAULT_TERM,):
            raise ValueError(f"AUX_CE_SAMPLING=game is fabrication-only (got {cfg['terms']}); "
                             f"multi-term needs AUX_CE_SAMPLING=random")
    else:
        cfg["terms"] = None      # resolve from the sidecar in load_state
    if cfg["on"]:
        if not cfg["data"] or not Path(cfg["data"]).is_file():
            raise ValueError(f"AUX_CE=1 but AUX_CE_DATA is not a file: {cfg['data']!r}")
        if not cfg["coeff"] > 0:
            raise ValueError(f"AUX_CE_COEFF must be > 0, got {cfg['coeff']}")
        if not (0.0 < cfg["frac"] <= 1.0):
            raise ValueError(f"AUX_CE_ROW_FRAC must be in (0, 1], got {cfg['frac']}")
    return cfg


def label_response(tokenizer, label: str):
    """-> (ids, attn, loss_mask), each RESP_WIDTH long. loss_mask marks the token(s) overlapping the
    bare true/false word by offset mapping, so a BPE merge gluing '>' onto the label stays masked."""
    enc = tokenizer(RESP_TEXT[label], add_special_tokens=False, return_offsets_mapping=True)
    ids, offs = list(enc["input_ids"]), list(enc["offset_mapping"])
    lo, hi = _LABEL_SPAN[label]
    mask = [1 if (s < hi and e > lo) else 0 for (s, e) in offs]
    if not (1 <= sum(mask) <= 3):
        raise ValueError(f"label-token span for {label!r} has {sum(mask)} tokens: {ids} {offs}")
    if len(ids) > RESP_WIDTH:
        raise ValueError(f"{RESP_TEXT[label]!r} tokenizes to {len(ids)} > RESP_WIDTH={RESP_WIDTH}")
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    n_pad = RESP_WIDTH - len(ids)
    return (ids + [pad_id] * n_pad,
            [1] * len(enc["input_ids"]) + [0] * n_pad,
            mask + [0] * n_pad)


@dataclass
class AuxCEState:
    items_by_game: dict            # gid -> {"true": [row...], "false": [row...]}
    coeff: float
    frac: float
    seed: int
    max_prompt: int
    tokenizer: object
    chat_kwargs: dict
    pad_id: int
    resp: dict                     # label -> (ids, attn, loss_mask)
    n_items: int
    sampling: str = "game"         # "game" | "random" (see the module docstring)
    soft: bool = False             # AUX_CE_SOFT: sample the target from p_hat (see attach)
    terms: tuple = (DEFAULT_TERM,)  # the ACTIVE channels, in TERMS_KNOWN order
    # label -> [row...], the fabrication pools (the aux_loss filter builder and the
    # aux_ce/pool_{true,false} metrics read this name)
    items_by_label: dict = field(default_factory=lambda: {"true": [], "false": []})
    # (term, label) -> [row...], what sampling="random" draws from; DEFAULT_TERM entries are the
    # same list objects as items_by_label.
    items_by_term_label: dict = field(default_factory=dict)
    kind_balance: bool = False     # AUX_CE_KIND_BALANCE: balance the 4 kinds per batch
    # (term, block, label) -> [row...], order-preserving views of items_by_term_label.
    items_by_block_label: dict = field(default_factory=dict)
    prompt_cache: dict = field(default_factory=dict)   # item_id -> token ids
    cursors: dict = field(default_factory=dict)        # _pool_key(...) -> (cursor, pass_idx)
    perms: dict = field(default_factory=dict)          # _pool_key(...) + (pass_idx,) -> perm


def _pool_key(term: str, gid, label: str) -> tuple:
    """Cursor/permutation key. _cycle seeds on it, so DEFAULT_TERM keeps the pre-multi-term
    (gid, label) key and fabrication-only runs stay bit-reproducible. Other terms prefix their
    name, which cannot collide with those keys or their seed strings."""
    return (gid, label) if term == DEFAULT_TERM else (term, gid, label)


def _block_pool_key(term: str, block: str, label: str) -> tuple:
    """Cursor/permutation key for a kind_balance block pool. The "kb" 4-tuple cannot collide with
    _pool_key's keys, so kind balance never perturbs the per-(term, label) RNG streams."""
    return ("kb", term, block, label)


def load_state(tokenizer, apply_chat_template_kwargs=None) -> AuxCEState:
    """Read AUX_CE_DATA (+ sibling _meta.json), verify chat-template parity, precompute the
    per-label response rows. Raises on every inconsistency — a silent aux no-op or a
    template-drifted prompt must never train."""
    cfg = config_from_env()
    assert cfg["on"], "load_state called with AUX_CE!=1"
    data_path = Path(cfg["data"])
    size = data_path.parent.name
    meta_path = data_path.parent.parent / "_meta.json"
    assert meta_path.is_file(), f"sidecar meta missing: {meta_path}"
    meta = json.loads(meta_path.read_text())
    m = meta["sizes"].get(size)
    assert m, f"_meta.json has no entry for size {size!r} (AUX_CE_DATA path layout changed?)"
    tpl_sha = hashlib.sha256(tokenizer.chat_template.encode()).hexdigest()
    assert tpl_sha == m["chat_template_sha"], (
        f"chat_template sha mismatch: policy tokenizer {tpl_sha[:16]} != sidecar "
        f"{m['chat_template_sha'][:16]} — the sidecar was token-checked under a different "
        f"template; rebuild it for this policy")
    assert m["max_prompt_tokens"] <= cfg["max_prompt"], (
        f"sidecar max prompt {m['max_prompt_tokens']} > AUX_CE_MAX_PROMPT {cfg['max_prompt']}")

    items_by_game, items_by_term_label, n_items = {}, {}, 0
    seen_ids, present = set(), set()
    with gzip.open(data_path, "rt") as fh:
        for line in fh:
            if not line.strip():
                continue
            r = json.loads(line)
            sv = r.get("schema_version")
            assert sv in (1, 2), f"unknown sidecar schema: {sv}"
            term = r.get("term", DEFAULT_TERM)      # absent -> fabrication (schema v1 back-compat)
            assert sv == 2 or "term" not in r, "schema_version 1 rows must not carry a 'term' field"
            assert term in TERMS_KNOWN, f"{r['item_id']}: unknown term {term!r}"
            assert r["label"] in term_labels(term), (
                f"{r['item_id']}: label {r['label']!r} is not a {term!r} label "
                f"(expected one of {term_labels(term)})")
            # _prompt_ids caches on item_id, so a collision would serve one term's prompt for
            # another term's label.
            assert r["item_id"] not in seen_ids, f"duplicate item_id {r['item_id']!r}"
            seen_ids.add(r["item_id"])
            present.add(term)
            # p_hat: soft-label target (vote share). Fabrication rows lack it (near-noiseless
            # instrument), so the hard label maps to 1.0/0.0 and soft mode leaves them unchanged.
            ph = r.get("p_hat")
            if ph is None:
                ph = 1.0 if r["label"] == "true" else 0.0
            assert 0.0 <= ph <= 1.0, f"{r['item_id']}: p_hat {ph} out of range"
            row = {"item_id": r["item_id"], "messages": r["messages"], "label": r["label"],
                   "term": term, "p_hat": float(ph), "kind": r.get("kind")}
            if term == DEFAULT_TERM:
                slot = items_by_game.setdefault(int(r["game_id"]), {"true": [], "false": []})
                slot[r["label"]].append(row)        # flattened game-sorted below (bit-compat)
            else:
                items_by_term_label.setdefault((term, r["label"]), []).append(row)
            n_items += 1
    assert n_items > 0, f"empty sidecar: {data_path}"

    kwargs = dict(apply_chat_template_kwargs or {})
    kwargs["enable_thinking"] = False   # sidecar token check + probe parity: thinking always off
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    # Flat per-label fabrication pools, game-sorted rather than file order: the seeded permutation
    # indexes into pool order.
    items_by_label = {"true": [], "false": []}
    for gid in sorted(items_by_game):
        for lb in ("true", "false"):
            items_by_label[lb].extend(items_by_game[gid][lb])
    if items_by_game:
        items_by_term_label[(DEFAULT_TERM, "true")] = items_by_label["true"]   # same list objects
        items_by_term_label[(DEFAULT_TERM, "false")] = items_by_label["false"]
    for k, pool in items_by_term_label.items():
        if k[0] != DEFAULT_TERM:        # make non-fabrication pool order independent of line order
            pool.sort(key=lambda r: r["item_id"])

    terms = cfg["terms"] or parse_terms("", present=present)
    if cfg["sampling"] == "game":
        assert terms == (DEFAULT_TERM,) and present == {DEFAULT_TERM}, (
            f"AUX_CE_SAMPLING=game is fabrication-only; sidecar holds {sorted(present)}")
    for t in terms:
        counts = {lb: len(items_by_term_label.get((t, lb), [])) for lb in term_labels(t)}
        assert all(counts.values()), (
            f"term {t!r} must hold BOTH labels in the sidecar, got {counts} ({data_path}); "
            f"drop it from AUX_CE_TERMS or rebuild the sidecar")
    extra = present - set(terms)
    if extra:
        print(f"[aux-ce] NOTE: sidecar also holds {sorted(extra)}, not selected by AUX_CE_TERMS")

    # Block pools for AUX_CE_KIND_BALANCE, filtered from the (term, label) pools so each keeps its
    # parent's order. Every gate is fatal: an unpaired sidecar (base_origin, coarse1k*, ill6*) must
    # not be sampled as if paired.
    items_by_block_label = {}
    if cfg["kind_balance"]:
        for t in terms:
            for lb in term_labels(t):
                pool = items_by_term_label.get((t, lb), [])
                for r in pool:
                    kind = r.get("kind")
                    if kind not in KIND_BLOCK:
                        raise ValueError(
                            f"AUX_CE_KIND_BALANCE=1 but item {r['item_id']!r} has kind "
                            f"{kind!r}, not one of {sorted(KIND_BLOCK)} — this is not a paired, "
                            f"kind-labelled sidecar; unset AUX_CE_KIND_BALANCE")
                    items_by_block_label.setdefault((t, KIND_BLOCK[kind], lb), []).append(r)
            for blk in BLOCKS:
                sizes = {lb: len(items_by_block_label.get((t, blk, lb), []))
                         for lb in term_labels(t)}
                if not all(sizes.values()):
                    raise ValueError(
                        f"AUX_CE_KIND_BALANCE=1: term {t!r} block {blk!r} is missing a label "
                        f"({sizes}) — every block must hold BOTH labels to be drawn as a pair")
                if len(set(sizes.values())) != 1:
                    raise ValueError(
                        f"AUX_CE_KIND_BALANCE=1: term {t!r} block {blk!r} is label-imbalanced "
                        f"({sizes}) — the block's rows are matched pairs, so its two labels must "
                        f"be equinumerous")

    n_true = len(items_by_label["true"])
    n_false = len(items_by_label["false"])
    state = AuxCEState(
        items_by_game=items_by_game, coeff=cfg["coeff"], frac=cfg["frac"], seed=cfg["seed"],
        max_prompt=cfg["max_prompt"], tokenizer=tokenizer, chat_kwargs=kwargs, pad_id=pad_id,
        resp={lb: label_response(tokenizer, lb)
              for lb in dict.fromkeys(x for t in terms for x in term_labels(t))},
        n_items=n_items, sampling=cfg["sampling"], soft=cfg["soft"], terms=terms,
        items_by_label=items_by_label, items_by_term_label=items_by_term_label,
        kind_balance=cfg["kind_balance"], items_by_block_label=items_by_block_label)
    pools = {f"{t}:{lb}": len(items_by_term_label.get((t, lb), []))
             for t in terms for lb in term_labels(t)}
    print(f"[aux-ce] loaded {n_items} items from {data_path} (coeff={cfg['coeff']}, "
          f"frac={cfg['frac']}, seed={cfg['seed']}, sampling={cfg['sampling']}, "
          f"terms={','.join(terms)})")
    print(f"[aux-ce]   fabrication: {n_true}T/{n_false}F over {len(items_by_game)} games; "
          f"pools={pools}")
    if cfg["kind_balance"]:
        kinds = collections.Counter(
            r["kind"] for pool in items_by_term_label.values() for r in pool)
        blocks = {f"{t}:{blk}:{lb}": len(v)
                  for (t, blk, lb), v in sorted(items_by_block_label.items())}
        print(f"[aux-ce]   kind_balance=ON kinds={dict(sorted(kinds.items()))} "
              f"block_pools={blocks}")
    return state


def schedule(n: int, world_size: int, frac: float) -> list:
    """Scheduled row indices: the FIRST round(frac*chunk) rows of each rank's contiguous
    dispatch chunk. Positional, so every rank issues the identical collective op sequence."""
    assert n % world_size == 0, f"batch rows {n} not divisible by world_size {world_size}"
    chunk = n // world_size
    k = max(1, round(frac * chunk))
    return [c * chunk + i for c in range(world_size) for i in range(k)]


def term_plan(n: int, world_size: int, frac: float, terms: tuple, global_step: int) -> list:
    """-> [(row_index, term_index, parity), ...] for the scheduled rows, identical in shape on
    every DP rank (FSDP2 needs the same op sequence on every rank).

    Each chunk's scheduled prefix is walked in adjacent pairs (parity 0 then 1); pair p gets term
    (p + global_step) % T. So per-term parity counts are exactly equal every step, term slot counts
    differ by at most one pair, exposure equalizes every T steps, and T == 1 reduces to the
    single-channel rule parity = (row % chunk) % 2. The alternative "term = terms[j % T], parity =
    (j // T) % 2" gave a permanent 24/20 label skew on 4 of 6 channels at chunk=64, T=6.
    """
    sched = schedule(n, world_size, frac)
    chunk = n // world_size
    T = len(terms)
    k = len(sched) // world_size
    if T > 1 and k % 2:
        k -= 1        # keep pairs whole; the odd tail row of each chunk goes unscheduled
        assert k >= 2, (f"AUX_CE_ROW_FRAC={frac} leaves {k + 1} scheduled row(s) per chunk of "
                        f"{chunk}, too few for a pos/neg pair with {T} terms")
    return [(c * chunk + j, (j // 2 + global_step) % T, j % 2)
            for c in range(world_size) for j in range(k)]


def _cycle(state: AuxCEState, key: tuple, pool: list):
    """Advance the without-replacement cursor for `key` over `pool` and return the item. The
    permutation is seeded by (AUX_CE_SEED, *key, pass_idx); exhausting a pool starts a new pass."""
    cur, pidx = state.cursors.get(key, (0, 0))
    if cur >= len(pool):
        cur, pidx = 0, pidx + 1
    pkey = key + (pidx,)
    if pkey not in state.perms:
        rng = random.Random(":".join([str(state.seed), *(str(k) for k in key), str(pidx)]))
        perm = list(range(len(pool)))
        rng.shuffle(perm)
        state.perms[pkey] = perm
        state.perms.pop(key + (pidx - 1,), None)   # keep memory bounded
    item = pool[state.perms[pkey][cur]]
    state.cursors[key] = (cur + 1, pidx)
    return item


def draw(state: AuxCEState, gid: int, label: str):
    """sampling="game": without-replacement cycling draw from (gid, label); falls back to the
    other label when the requested pool is empty; None when the game has no items at all
    (dummy row)."""
    pools = state.items_by_game.get(gid)
    if not pools:
        return None
    if not pools[label]:
        label = "true" if label == "false" else "false"
        if not pools[label]:
            return None
    return _cycle(state, _pool_key(DEFAULT_TERM, gid, label), pools[label])


def draw_term(state: AuxCEState, term: str, label: str):
    """sampling="random": without-replacement cycling draw from the whole (term, label) pool,
    ignoring which game the row belongs to. Never None — load_state asserts every active term
    holds both labels."""
    pool = state.items_by_term_label.get((term, label)) or []
    assert pool, (f"empty global pool for ({term!r}, {label!r}) — load_state should have refused "
                  f"this sidecar")
    return _cycle(state, _pool_key(term, _GLOBAL_GID, label), pool)


def block_plan(plan: list, world_size: int, terms: tuple, global_step: int) -> list:
    """Block index per scheduled row, for AUX_CE_KIND_BALANCE.

    Pair p (see term_plan) takes block (p // T + global_step) % 2: both rows of a pair share a
    block, blocks alternate across a term's pairs (all four kinds equal when a chunk holds an even
    number of pairs per term), and the odd-pair remainder rotates with global_step. Dividing by T
    keeps block independent of term; (p + global_step) % 2 would pin one channel to one block at
    T == 2. Positional, so every DP rank computes the same assignment.
    """
    n_terms = max(1, len(terms))
    assert world_size > 0 and len(plan) % world_size == 0, (
        f"term_plan returned {len(plan)} rows, not a multiple of world_size {world_size}")
    k = len(plan) // world_size          # scheduled rows per rank chunk (term_plan keeps k even)
    # `plan` is chunk-major, so idx % k is the row's position in its chunk and // 2 its pair index;
    # derived from term_plan's output so the two cannot drift.
    return [(((idx % k) // 2) // n_terms + global_step) % 2 for idx in range(len(plan))]


def draw_block(state: AuxCEState, term: str, block: str, label: str):
    """kind_balance draw: without-replacement cycling over the (term, block, label) pool."""
    pool = state.items_by_block_label.get((term, block, label)) or []
    assert pool, (f"empty block pool for ({term!r}, {block!r}, {label!r}) — load_state should "
                  f"have refused this sidecar")
    return _cycle(state, _block_pool_key(term, block, label), pool)


def draw_global(state: AuxCEState, label: str):
    """Back-compat alias: the fabrication channel's global draw. Seed string "<seed>:*:<label>:0",
    byte-identical to the pre-multi-term form."""
    return draw_term(state, DEFAULT_TERM, label)


def _target_label(state: AuxCEState, item: dict, global_step: int, row: int) -> str:
    """The answer this row is trained on: the item's majority label, or with AUX_CE_SOFT=1 a sample
    from p_hat (the share of the K judges who said "used").

    A hard label on a 3-2 split is unattainable, so CE floors at the label-noise entropy and keeps
    emitting gradient (the multi-channel loss competed with the policy gradient for all 100 steps,
    while the clean fabrication channel reached 0.000 by step 30). Sampling makes the expected
    gradient the soft-label gradient, which vanishes at q = p_hat, at no extra forward cost (a
    weighted two-response loss needs a second forward, too slow for the 8B arms under a 22h cap).
    Deterministic in (seed, step, row).
    """
    if not state.soft:
        return item["label"]
    p = item.get("p_hat")
    if p is None:                                   # no vote information -> hard, unchanged
        return item["label"]
    rng = random.Random(f"{state.seed}:soft:{global_step}:{row}:{item['item_id']}")
    return "true" if rng.random() < p else "false"


def _prompt_ids(state: AuxCEState, item: dict) -> list:
    ids = state.prompt_cache.get(item["item_id"])
    if ids is None:
        ids = state.tokenizer.apply_chat_template(
            list(item["messages"]), add_generation_prompt=True, tokenize=True,
            return_dict=False, **state.chat_kwargs)
        assert isinstance(ids, list) and (not ids or isinstance(ids[0], int)), type(ids)
        assert len(ids) <= state.max_prompt, (
            f"{item['item_id']}: composed prompt {len(ids)} > AUX_CE_MAX_PROMPT "
            f"{state.max_prompt} (sidecar cap drifted?)")
        state.prompt_cache[item["item_id"]] = ids
    return ids


def collate(rows: list, pad_id: int, term_ids=None):
    """rows: per batch row, None (unscheduled), "dummy", or (prompt_ids, (rids, rattn, rmask)).
    term_ids: per batch row, the index into state.terms, -1 for an unscheduled row. None -> 0
    (DEFAULT_TERM) for every scheduled row, which is what sampling="game" wants.
    -> dict of CPU tensors, each (n, W+RESP_WIDTH) / (n, RESP_WIDTH) / (n,)."""
    import torch
    n = len(rows)
    if term_ids is None:
        term_ids = [0 if r is not None else -1 for r in rows]
    assert len(term_ids) == n, (len(term_ids), n)
    assert max(term_ids) < 127, f"term id {max(term_ids)} does not fit int8"
    W = max([len(r[0]) for r in rows if isinstance(r, tuple)] or [1])
    T = W + RESP_WIDTH
    input_ids = torch.full((n, T), pad_id, dtype=torch.long)
    attn = torch.zeros((n, T), dtype=torch.long)
    responses = torch.full((n, RESP_WIDTH), pad_id, dtype=torch.long)
    loss_mask = torch.zeros((n, RESP_WIDTH), dtype=torch.long)
    valid = torch.zeros((n,), dtype=torch.int8)
    # -1 (not 0) on unscheduled rows is load-bearing: the dp_actor per-term scatter compares
    # against arange(T), so -1 matches no column and self-excludes without an aux_valid multiply.
    term_id = torch.tensor(term_ids, dtype=torch.int8)
    for i, r in enumerate(rows):
        if r is None:
            assert term_ids[i] == -1, f"row {i} unscheduled but term_id={term_ids[i]}"
            continue
        valid[i] = 1
        assert term_ids[i] >= 0, f"row {i} scheduled but term_id={term_ids[i]}"
        if r == "dummy":
            # 1 valid prompt token + 1 attended (zero-mask) response token: a real forward with
            # an exactly-zero loss, keeping per-rank collective counts data-independent.
            attn[i, W - 1] = 1
            attn[i, W] = 1
            continue
        pids, (rids, rattn, rmask) = r
        input_ids[i, W - len(pids):W] = torch.tensor(pids, dtype=torch.long)
        attn[i, W - len(pids):W] = 1
        input_ids[i, W:] = torch.tensor(rids, dtype=torch.long)
        attn[i, W:] = torch.tensor(rattn, dtype=torch.long)
        responses[i] = torch.tensor(rids, dtype=torch.long)
        loss_mask[i] = torch.tensor(rmask, dtype=torch.long)
    position_ids = torch.clamp(attn.cumsum(dim=-1) - 1, min=0)
    return {"aux_input_ids": input_ids, "aux_attention_mask": attn,
            "aux_position_ids": position_ids, "aux_responses": responses,
            "aux_response_mask": loss_mask, "aux_valid": valid, "aux_term_id": term_id}


def attach(batch, state: AuxCEState, global_step: int, world_size: int) -> dict:
    """Write the AUX_KEYS tensors into batch.batch (row-aligned) and return step metrics.
    `batch` needs only .batch (TensorDict) and .non_tensor_batch (dict of np arrays)."""
    n = int(batch.batch.batch_size[0])
    gids = [int(e["index"]) for e in batch.non_tensor_batch["extra_info"]]

    rows, term_ids = [None] * n, [-1] * n
    n_dummy = 0
    counts = {(t, lb): 0 for t in state.terms for lb in term_labels(t)}
    kind_counts = {k: 0 for k in sorted(KIND_BLOCK)}
    if state.sampling == "random":
        # Term and label come from the row's position in its rank chunk: identical on every rank,
        # exactly 1:1 per term every step, whatever the sidecar.
        plan = term_plan(n, world_size, state.frac, state.terms, global_step)
        sched = [i for i, _, _ in plan]
        # kind_balance also fixes each pair's block by position (see block_plan).
        blocks = (block_plan(plan, world_size, state.terms, global_step)
                  if state.kind_balance else None)
        for p, (i, ti, parity) in enumerate(plan):
            term = state.terms[ti]
            label = term_labels(term)[parity]
            if state.kind_balance:
                item = draw_block(state, term, BLOCKS[blocks[p]], label)
            else:
                item = draw_term(state, term, label)
            lbl = _target_label(state, item, global_step, i)
            rows[i] = (_prompt_ids(state, item), state.resp[lbl])
            term_ids[i] = ti
            counts[(term, lbl)] += 1
            if state.kind_balance:
                kind_counts[item["kind"]] += 1
    else:
        sched = schedule(n, world_size, state.frac)
        rank_in_game = {}
        for i in sched:
            gid = gids[i]
            r = rank_in_game.get(gid, 0)
            rank_in_game[gid] = r + 1
            label = "false" if r % 2 == 0 else "true"   # exact per-game alternation
            item = draw(state, gid, label)
            term_ids[i] = 0                             # game mode is fabrication-only
            if item is None:
                rows[i] = "dummy"
                n_dummy += 1
                continue
            rows[i] = (_prompt_ids(state, item), state.resp[item["label"]])
            counts[(DEFAULT_TERM, item["label"])] += 1

    n_true = counts.get((DEFAULT_TERM, "true"), 0)
    n_false = counts.get((DEFAULT_TERM, "false"), 0)
    tensors = collate(rows, state.pad_id, term_ids)
    for k, v in tensors.items():
        batch.batch[k] = v
    # Coverage only means something when items are looked up by game; -1 sentinel under "random".
    if state.sampling == "random":
        coverage = -1.0
    else:
        coverage = sum(1 for g in set(gids) if g in state.items_by_game) / max(1, len(set(gids)))
    # n_true/n_false/pool_* always count the fabrication channel, so the series means the same
    # thing across arms.
    metrics = {"aux_ce/n_scheduled": float(len(sched)), "aux_ce/n_true": float(n_true),
               "aux_ce/n_false": float(n_false), "aux_ce/n_dummy": float(n_dummy),
               "aux_ce/prompt_width": float(tensors["aux_input_ids"].shape[1] - RESP_WIDTH),
               "aux_ce/pool_true": float(len(state.items_by_label["true"])),
               "aux_ce/pool_false": float(len(state.items_by_label["false"])),
               "aux_ce/coverage": coverage,
               "aux_ce/n_terms": float(len(state.terms))}
    for t in state.terms:
        pos, neg = term_labels(t)
        metrics[f"aux_ce/n_{t}_pos"] = float(counts[(t, pos)])
        metrics[f"aux_ce/n_{t}_neg"] = float(counts[(t, neg)])
        metrics[f"aux_ce/pool_{t}_pos"] = float(len(state.items_by_term_label.get((t, pos), [])))
        metrics[f"aux_ce/pool_{t}_neg"] = float(len(state.items_by_term_label.get((t, neg), [])))
    if state.kind_balance:
        # Only under kind_balance, so a default run's key set is unchanged. kind_spread is 0 when
        # the balance held.
        for k, v in kind_counts.items():
            metrics[f"aux_ce/kind_{k}"] = float(v)
        metrics["aux_ce/kind_spread"] = float(max(kind_counts.values())
                                              - min(kind_counts.values()))
    if n_dummy:
        print(f"[aux-ce] step={global_step}: {n_dummy} dummy rows "
              f"(games without sidecar items: "
              f"{sorted({gids[i] for i in sched if rows[i] == 'dummy'})[:8]})")
    return metrics


# ---------------------------------------------------------------------------------------------
def _selftest():
    import torch  # (collate needs it; fail here, not mid-test)
    th_eq = torch.equal

    # 1. schedule: positional, per-chunk-equal, frac handling
    assert schedule(256, 4, 1.0) == list(range(256))
    s = schedule(256, 4, 0.25)
    assert len(s) == 64 and s[:3] == [0, 1, 2] and s[16] == 64 and s[32] == 128, s[:20]
    for w in (1, 2, 4):
        for frac in (1.0, 0.5, 0.25, 0.01):
            ss = schedule(64, w, frac)
            per_chunk = [sum(1 for i in ss if c * (64 // w) <= i < (c + 1) * (64 // w))
                         for c in range(w)]
            assert len(set(per_chunk)) == 1 and per_chunk[0] >= 1, (w, frac, per_chunk)

    # 1b. term_plan: rank-identical, per-term 1:1 every step, equal exposure every T steps, and
    #     the pre-multi-term rule at T=1 (bit-compat).
    for frac in (1.0, 0.5, 0.25, 0.1):
        for step in range(100):
            got = term_plan(256, 4, frac, (DEFAULT_TERM,), step)
            want = [(i, 0, (i % 64) % 2) for i in schedule(256, 4, frac)]
            assert got == want, (frac, step, got[:6], want[:6])
    T6 = TERMS_KNOWN
    plan0 = term_plan(256, 4, 1.0, T6, 0)
    assert len(plan0) == 256 and len({i for i, _, _ in plan0}) == 256
    # identical (term, parity) pattern in every rank chunk
    by_chunk = [[(ti, p) for i, ti, p in plan0 if i // 64 == c] for c in range(4)]
    assert all(b == by_chunk[0] for b in by_chunk), by_chunk[0][:8]
    for step in (0, 1, 5, 37):
        pl = term_plan(256, 4, 1.0, T6, step)
        for ti in range(len(T6)):
            npos = sum(1 for _, t, p in pl if t == ti and p == 0)
            nneg = sum(1 for _, t, p in pl if t == ti and p == 1)
            assert npos == nneg, (step, T6[ti], npos, nneg)   # exact 1:1 per term per step
        slots = [sum(1 for _, t, _ in pl if t == ti) for ti in range(len(T6))]
        assert max(slots) - min(slots) <= 2 * 4, (step, slots)  # <=1 pair per rank
    tot = [0] * len(T6)
    for step in range(len(T6)):
        for _, ti, _ in term_plan(256, 4, 1.0, T6, step):
            tot[ti] += 1
    assert len(set(tot)) == 1, tot                        # exactly equal exposure every T steps
    # whole-step label balance is unchanged from the fabrication-only run: 128/128
    assert sum(1 for _, _, p in plan0 if p == 0) == 128, plan0[:4]
    # parse_terms: normalization, separators, duplicates, unknowns
    assert parse_terms("") == (DEFAULT_TERM,)
    assert parse_terms(None, present={"discouragement", DEFAULT_TERM}) == \
        (DEFAULT_TERM, "discouragement")
    assert parse_terms("discouragement,fabrication") == (DEFAULT_TERM, "discouragement")
    assert parse_terms("fabrication+belief_coercion") == (DEFAULT_TERM, "belief_coercion")
    assert parse_terms("  fabrication   belief_coercion ") == (DEFAULT_TERM, "belief_coercion")
    for bad in ("fabrication,fabrication", "nope", "fabrication+nope"):
        try:
            parse_terms(bad)
            raise AssertionError(f"parse_terms({bad!r}) must fail")
        except ValueError:
            pass
    assert term_labels(DEFAULT_TERM) == ("false", "true")      # parity 0 = fabrication present
    assert term_labels("belief_coercion") == ("true", "false")  # parity 0 = technique used
    assert all("min" not in t and "max" not in t for t in TERMS_KNOWN), TERMS_KNOWN
    # _pool_key: fabrication keeps the pre-multi-term seed string byte-for-byte
    assert _pool_key(DEFAULT_TERM, _GLOBAL_GID, "true") == ("*", "true")
    assert _pool_key(DEFAULT_TERM, 3, "false") == (3, "false")
    assert ":".join(["7", *(str(k) for k in _pool_key(DEFAULT_TERM, "*", "true")), "0"]) == \
        "7:*:true:0"
    assert ":".join(["7", *(str(k) for k in _pool_key(DEFAULT_TERM, 3, "false")), "0"]) == \
        "7:3:false:0"
    assert _pool_key("belief_coercion", "*", "true") == ("belief_coercion", "*", "true")

    # 2. draw: without-replacement within a pass, deterministic, cross-label fallback, dummy
    def mkstate(pools):
        return AuxCEState(items_by_game=pools, coeff=0.5, frac=1.0, seed=7, max_prompt=64,
                          tokenizer=None, chat_kwargs={}, pad_id=0, resp={}, n_items=0)

    pools = {1: {"true": [{"item_id": f"t{i}", "messages": [], "label": "true"} for i in range(3)],
                 "false": [{"item_id": f"f{i}", "messages": [], "label": "false"} for i in range(5)]},
             2: {"true": [{"item_id": "only_t", "messages": [], "label": "true"}], "false": []}}
    st = mkstate(pools)
    got = [draw(st, 1, "false")["item_id"] for _ in range(5)]
    assert sorted(got) == [f"f{i}" for i in range(5)], got          # one full pass, no repeats
    got2 = [draw(st, 1, "false")["item_id"] for _ in range(5)]
    assert sorted(got2) == sorted(got) and got2 != got[:1] * 5      # next pass, reshuffled
    st2 = mkstate(pools)
    assert [draw(st2, 1, "false")["item_id"] for _ in range(5)] == got  # deterministic
    assert draw(st, 2, "false")["item_id"] == "only_t"              # empty-pool label fallback
    assert draw(st, 99, "false") is None                            # unknown game -> dummy

    # 2b. sampling="random": global pools, exact label balance, no dummies, no game lookup
    def mkrand(pools):
        st = mkstate(pools)
        st.sampling = "random"
        st.items_by_label = {lb: [it for g in sorted(pools) for it in pools[g][lb]]
                             for lb in ("true", "false")}
        st.items_by_term_label = {(DEFAULT_TERM, lb): st.items_by_label[lb]
                                  for lb in ("true", "false")}
        return st

    st = mkrand(pools)
    got = [draw_global(st, "false")["item_id"] for _ in range(5)]
    assert sorted(got) == [f"f{i}" for i in range(5)], got          # one full pass, no repeats
    got2 = [draw_global(st, "false")["item_id"] for _ in range(5)]
    assert sorted(got2) == sorted(got) and got2 != got              # next pass, reshuffled
    st_fresh = mkrand(pools)                                        # deterministic from a fresh state
    assert [draw_global(st_fresh, "false")["item_id"] for _ in range(5)] == got
    # game 2 contributes only a TRUE item, but the global FALSE pool is game 1's -> no fallback
    assert {draw_global(st, "true")["item_id"] for _ in range(4)} == {"t0", "t1", "t2", "only_t"}

    class _FakeTD(dict):
        def __init__(self, n):
            super().__init__()
            self.batch_size = (n,)

    class _FakeBatch:
        def __init__(self, gids):
            self.batch = _FakeTD(len(gids))
            self.non_tensor_batch = {"extra_info": [{"index": g} for g in gids]}

    # 32 rows / 4 ranks = chunk 8. Games 97-99 are absent from the sidecar; "random" must not
    # produce dummy rows for them.
    _RESP = {lb: ([9, 8, 7, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0, 0, 0]) for lb in ("true", "false")}

    def _primed(st):
        st.resp = _RESP
        for g in st.items_by_game.values():
            for lb in ("true", "false"):
                for it in g[lb]:
                    st.prompt_cache[it["item_id"]] = [1, 2, 3]   # skip tokenization
        return st

    gids32 = [g for g in (1, 2, 97, 98, 99, 1, 2, 97) for _ in range(4)]
    b = _FakeBatch(gids32)
    m = attach(b, _primed(mkrand(pools)), global_step=1, world_size=4)
    assert m["aux_ce/n_dummy"] == 0.0, m                             # no game lookup -> no dummies
    assert m["aux_ce/n_true"] == m["aux_ce/n_false"] == 16.0, m      # exact 1:1 over 32 rows
    assert m["aux_ce/coverage"] == -1.0, m                           # sentinel, not a fake 1.0
    assert m["aux_ce/pool_false"] == 5.0 and m["aux_ce/pool_true"] == 4.0, m
    assert b.batch["aux_valid"].tolist() == [1] * 32
    # per-rank identical FALSE counts -> identical collective sequence on every rank
    for w in (1, 2, 4):
        ck = 32 // w
        per_rank = [sum(1 for i in schedule(32, w, 1.0) if i // ck == c and (i % ck) % 2 == 0)
                    for c in range(w)]
        assert len(set(per_rank)) == 1 and per_rank[0] == ck // 2, (w, per_rank)

    # game mode is unchanged: rows of absent games still become dummies
    m2 = attach(_FakeBatch(gids32), _primed(mkstate(pools)), global_step=1, world_size=4)
    assert m2["aux_ce/n_dummy"] == 16.0, m2       # gids 97 (x2), 98, 99 -> 4 occurrences * 4 rows
    assert m2["aux_ce/coverage"] == 2 / 5, m2     # 2 of 5 distinct gids are in the sidecar

    # 2c. Multi-term attach: every active channel drawn from its own pool, exactly 1:1, no dummies
    def mkmulti(terms):
        st = mkstate({})
        st.sampling = "random"
        st.terms = terms
        st.resp = _RESP
        st.items_by_term_label = {}
        for t in terms:
            for lb in term_labels(t):
                pool = [{"item_id": f"{t}_{lb}_{i}", "messages": [], "label": lb}
                        for i in range(3)]
                st.items_by_term_label[(t, lb)] = pool
                for it in pool:
                    st.prompt_cache[it["item_id"]] = [1, 2, 3]
        st.items_by_label = {lb: st.items_by_term_label.get((DEFAULT_TERM, lb), [])
                             for lb in ("true", "false")}
        return st

    # RANKS*2*T rows -> exactly one pair per term per rank. Derived from len(TERMS_KNOWN) so
    # appending a term cannot silently break this case (it did at the 7th term).
    _NR, _T = 4, len(TERMS_KNOWN)
    _ROWS = _NR * 2 * _T
    b6 = _FakeBatch([1] * _ROWS)
    st6 = mkmulti(TERMS_KNOWN)
    m6 = attach(b6, st6, global_step=0, world_size=_NR)
    assert m6["aux_ce/n_terms"] == float(_T) and m6["aux_ce/n_scheduled"] == float(_ROWS), m6
    assert m6["aux_ce/n_dummy"] == 0.0, m6
    for t in TERMS_KNOWN:
        assert m6[f"aux_ce/n_{t}_pos"] == m6[f"aux_ce/n_{t}_neg"] == float(_NR), (t, m6)
    assert sum(m6[f"aux_ce/n_{t}_{s}"] for t in TERMS_KNOWN for s in ("pos", "neg")) == float(_ROWS)
    # n_true/n_false are the fabrication view only, so they stay at one pair per rank
    assert m6["aux_ce/n_true"] == m6["aux_ce/n_false"] == float(_NR), m6
    tid = b6.batch["aux_term_id"].tolist()
    assert set(tid) == set(range(_T)) and -1 not in tid, sorted(set(tid))
    assert b6.batch["aux_valid"].tolist() == [1] * _ROWS
    # each row's item really came from its own term's pool
    for i, ti, _ in term_plan(_ROWS, _NR, 1.0, TERMS_KNOWN, 0):
        assert tid[i] == ti
    # T=1 through the multi-term path == the pre-multi-term tensors, bit for bit
    b_new = _FakeBatch(gids32)
    attach(b_new, mkmulti((DEFAULT_TERM,)), global_step=0, world_size=4)
    b_old = _FakeBatch(gids32)
    st_old = mkmulti((DEFAULT_TERM,))
    rows_old = [None] * 32
    for i in schedule(32, 4, 1.0):
        it = draw_global(st_old, "false" if (i % 8) % 2 == 0 else "true")
        rows_old[i] = (_prompt_ids(st_old, it), _RESP[it["label"]])
    for k, v in collate(rows_old, 0).items():
        b_old.batch[k] = v
    for k in AUX_KEYS:
        assert th_eq(b_new.batch[k], b_old.batch[k]), f"T=1 tensor {k} differs from the single-term rule"

    # 2d. Soft targets: sampled from p_hat, deterministic, and hard mode untouched.
    st_soft = mkstate({})
    st_soft.soft = True
    it = {"item_id": "x1", "messages": [], "label": "true", "p_hat": 0.6}
    draws = [_target_label(st_soft, it, 7, r) for r in range(4000)]
    frac = sum(d == "true" for d in draws) / len(draws)
    assert abs(frac - 0.6) < 0.03, f"soft draw rate {frac} != p_hat 0.6"
    # deterministic in (seed, step, row)
    assert [_target_label(st_soft, it, 7, r) for r in range(50)] == draws[:50]
    # a different step gives a different realisation (else every step trains the same targets)
    assert [_target_label(st_soft, it, 8, r) for r in range(50)] != draws[:50]
    # p_hat at the extremes is exactly the hard label
    for p, want in ((1.0, "true"), (0.0, "false")):
        ex = {"item_id": "x2", "messages": [], "label": want, "p_hat": p}
        assert {_target_label(st_soft, ex, 3, r) for r in range(200)} == {want}
    # no p_hat (fabrication rows copied verbatim from the parent) -> hard, unchanged
    nop = {"item_id": "x3", "messages": [], "label": "false"}
    assert {_target_label(st_soft, nop, 3, r) for r in range(50)} == {"false"}
    # soft OFF: the majority label, whatever p_hat says
    st_hard = mkstate({})
    assert {_target_label(st_hard, it, 7, r) for r in range(200)} == {"true"}

    # 3. label_response masks on the real policy tokenizers (skip a size whose weights are absent)
    from transformers import AutoTokenizer
    models_dir = os.environ.get("MODELS_DIR") or str(Path(__file__).resolve().parents[1] / "models")
    paths = {"4B": f"{models_dir}/Qwen3-4B-Instruct-2507",
             "8B": f"{models_dir}/Qwen3-8B"}
    for size, path in paths.items():
        if not Path(path).is_dir():
            print(f"[selftest] WARN: {size} tokenizer missing at {path}; skipped")
            continue
        tok = AutoTokenizer.from_pretrained(path)
        for label in ("true", "false"):
            ids, attn, mask = label_response(tok, label)
            assert len(ids) == len(attn) == len(mask) == RESP_WIDTH
            masked = [i for i, m, a in zip(ids, mask, attn) if m and a]
            text = tok.decode(masked)
            assert label in text.lower(), (size, label, text)
            # the mask must not swallow the closing tag
            assert "</answer>" not in text, (size, label, text)
        print(f"[selftest] {size}: label spans OK "
              f"(true={tok.decode([i for i, m in zip(*label_response(tok, 'true')[::2]) if m])!r})")

    # 4. collate invariants (synthetic rows; no tokenizer needed)
    rows = [(list(range(1, 6)), ([9, 8, 7, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0],
                                 [0, 1, 0, 0, 0, 0, 0, 0])),
            "dummy", None,
            (list(range(1, 3)), ([9, 8, 7, 0, 0, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0],
                                 [0, 1, 0, 0, 0, 0, 0, 0]))]
    t = collate(rows, pad_id=0)
    import torch as th
    assert t["aux_input_ids"].shape == (4, 5 + RESP_WIDTH)
    assert t["aux_valid"].tolist() == [1, 1, 0, 1]
    assert t["aux_response_mask"][0].sum().item() == 1
    assert t["aux_response_mask"][1].sum().item() == 0               # dummy: zero loss
    assert t["aux_attention_mask"][1].sum().item() == 2              # but a real (tiny) forward
    assert t["aux_attention_mask"][2].sum().item() == 0              # unscheduled: untouched
    assert t["aux_input_ids"][3, 3:5].tolist() == [1, 2]             # left-padded prompt
    assert th.all(t["aux_position_ids"] >= 0)
    assert set(t) == set(AUX_KEYS)

    # 5. config_from_env strictness. Knobs are read at call time, so each case sets its own env.
    for _k in ("AUX_CE_SAMPLING", "AUX_CE_KIND_BALANCE", "AUX_CE_TERMS"):
        os.environ.pop(_k, None)
    os.environ["AUX_CE"] = "0"
    assert config_from_env()["on"] is False
    os.environ["AUX_CE"] = "yes"
    try:
        config_from_env()
        raise AssertionError("AUX_CE=yes must fail")
    except ValueError:
        pass
    finally:
        os.environ["AUX_CE"] = "0"
    assert config_from_env()["sampling"] == "random"         # the default sampler
    os.environ["AUX_CE_SAMPLING"] = "random"
    assert config_from_env()["sampling"] == "random"
    os.environ["AUX_CE_SAMPLING"] = "shuffled"
    try:
        config_from_env()
        raise AssertionError("AUX_CE_SAMPLING=shuffled must fail")
    except ValueError:
        pass
    finally:
        os.environ.pop("AUX_CE_SAMPLING", None)
    # AUX_CE_TERMS: unset -> None (resolve from sidecar); multi-term under game mode is refused.
    # Kind balance is pinned off so the ValueError can only come from the term check.
    assert config_from_env()["terms"] is None
    os.environ["AUX_CE_KIND_BALANCE"] = "0"
    os.environ["AUX_CE_TERMS"] = "fabrication+belief_coercion"
    os.environ["AUX_CE_SAMPLING"] = "game"
    try:
        config_from_env()
        raise AssertionError("multi-term + AUX_CE_SAMPLING=game must fail")
    except ValueError:
        pass
    os.environ["AUX_CE_SAMPLING"] = "random"
    assert config_from_env()["terms"] == (DEFAULT_TERM, "belief_coercion")
    os.environ["AUX_CE_TERMS"] = "fabrication"
    os.environ["AUX_CE_SAMPLING"] = "game"
    assert config_from_env()["terms"] == (DEFAULT_TERM,)     # fabrication-only stays legal
    os.environ["AUX_CE_TERMS"] = "belief_coersion"           # typo -> loud
    try:
        config_from_env()
        raise AssertionError("unknown AUX_CE_TERMS entry must fail")
    except ValueError:
        pass
    finally:
        os.environ.pop("AUX_CE_TERMS", None)
        os.environ.pop("AUX_CE_SAMPLING", None)
        os.environ.pop("AUX_CE_KIND_BALANCE", None)
    # 9. AUX_CE_KIND_BALANCE: block_plan + draw_block.
    import types
    # 9a. env parsing: on by default, strict values, and random-only.
    os.environ.pop("AUX_CE_KIND_BALANCE", None)
    os.environ.pop("AUX_CE_SAMPLING", None)
    assert config_from_env()["kind_balance"] is True         # the default, under the default sampler
    os.environ["AUX_CE_KIND_BALANCE"] = "0"
    assert config_from_env()["kind_balance"] is False
    os.environ["AUX_CE_KIND_BALANCE"] = "yes"
    try:
        config_from_env()
        raise AssertionError("AUX_CE_KIND_BALANCE must reject non-0/1")
    except ValueError:
        pass
    os.environ["AUX_CE_KIND_BALANCE"] = "1"
    os.environ["AUX_CE_SAMPLING"] = "game"
    try:
        config_from_env()
        raise AssertionError("kind_balance under sampling=game must fail")
    except ValueError:
        pass
    os.environ["AUX_CE_SAMPLING"] = "random"
    assert config_from_env()["kind_balance"] is True
    os.environ.pop("AUX_CE_KIND_BALANCE", None)
    os.environ.pop("AUX_CE_SAMPLING", None)

    # 9b. Both rows of a pair share a block; the pattern is identical in every rank chunk.
    pl1 = term_plan(256, 4, 1.0, (DEFAULT_TERM,), 0)
    bl1 = block_plan(pl1, 4, (DEFAULT_TERM,), 0)
    assert len(bl1) == len(pl1)
    for m in range(len(bl1) // 2):
        assert bl1[2 * m] == bl1[2 * m + 1], (m, bl1[2 * m: 2 * m + 2])
    per_chunk = [bl1[c * 64:(c + 1) * 64] for c in range(4)]
    assert all(b == per_chunk[0] for b in per_chunk)

    # 9c. Exact 4-way kind balance: block + label determines kind, so equal (block, label) counts
    #     are equal kind counts. 32 pairs per chunk -> 16 per block -> 16 of each kind.
    for step in range(8):
        pl = term_plan(256, 4, 1.0, (DEFAULT_TERM,), step)
        bl = block_plan(pl, 4, (DEFAULT_TERM,), step)
        cnt = collections.Counter((BLOCKS[b], term_labels(DEFAULT_TERM)[p])
                                  for (_i, _t, p), b in zip(pl, bl))
        assert len(cnt) == 4 and len(set(cnt.values())) == 1, (step, cnt)
        assert sum(cnt.values()) == 256

    # 9d. An odd number of pairs per chunk cannot balance within one step, so the remainder must
    #     rotate with global_step and even out over two steps.
    odd = [collections.Counter(block_plan(term_plan(64, 1, 6 / 64, (DEFAULT_TERM,), s), 1,
                                          (DEFAULT_TERM,), s)) for s in (0, 1)]
    assert odd[0] != odd[1] and odd[0][0] + odd[1][0] == odd[0][1] + odd[1][1], odd

    # 9e. Block must not be a function of term (the T=2 confound the // n_terms divide prevents).
    pl2 = term_plan(256, 4, 1.0, (DEFAULT_TERM, "belief_coercion"), 0)
    bl2 = block_plan(pl2, 4, (DEFAULT_TERM, "belief_coercion"), 0)
    seen = collections.defaultdict(set)
    for (_i, ti, _p), b in zip(pl2, bl2):
        seen[ti].add(b)
    assert all(v == {0, 1} for v in seen.values()), dict(seen)

    # 9f. draw_block returns only the requested block and cycles without replacement.
    pools = {(DEFAULT_TERM, "a", "true"): [{"item_id": f"a{i}", "kind": "claim_real"}
                                           for i in range(4)],
             (DEFAULT_TERM, "a", "false"): [{"item_id": f"f{i}", "kind": "fabrication"}
                                            for i in range(4)],
             (DEFAULT_TERM, "b", "true"): [{"item_id": f"p{i}", "kind": "paraphrase_real"}
                                           for i in range(3)],
             (DEFAULT_TERM, "b", "false"): [{"item_id": f"x{i}", "kind": "altered_real"}
                                            for i in range(3)]}
    st = types.SimpleNamespace(items_by_block_label=pools, cursors={}, perms={}, seed=2026)
    for (t, blk, lb), pool in pools.items():
        got = [draw_block(st, t, blk, lb)["item_id"] for _ in range(len(pool))]
        assert sorted(got) == sorted(r["item_id"] for r in pool), (blk, lb, got)
        assert all(KIND_BLOCK[draw_block(st, t, blk, lb)["kind"]] == blk for _ in range(5))
    # its keyspace is disjoint from the per-(term, label) pools', so their RNG streams are untouched
    assert not (set(st.cursors) & {_pool_key(DEFAULT_TERM, _GLOBAL_GID, "true"),
                                   _pool_key(DEFAULT_TERM, _GLOBAL_GID, "false")})

    print("[selftest] PASS — schedule, term_plan (T=1 bit-compat + per-term 1:1), draw, random "
          "sampling, multi-term attach, label spans, collate, env parsing, "
          "kind_balance (block_plan pairing/balance/rotation/decorrelation + draw_block)")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        sys.exit("usage: python -m rl.aux_ce --selftest")
