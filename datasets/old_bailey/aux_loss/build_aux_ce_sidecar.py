#!/usr/bin/env python3
"""Build the aux-CE probe sidecar from the train-split strategies-prompt collection.

Each line is one probe-ready in-role deception-probe item for one of the 921 GRPO train games
(sft_split.json rl_train_ids). `messages` is the strategies system prompt + a frozen mid-game
context from the base/gs50/gs100 misrepv2 checkpoints + the probe turn, byte-identical to the
endorsement probe (evaluation.source_of_fabrication.probe_prompts.compose). With AUX_CE=1,
rl/aux_ce.py looks items up by game_id for each GRPO batch and adds CE on the gold true/false token.

Item kinds (label):
  fabrication       false  judge-confirmed GENUINE_FABRICATION claims in their pre-fabrication
                           context; cap 3/game/origin, deduped on (game, claim), earliest origin wins
  control_real      true   one real Description per fabricating context, plus a round-1 context
                           (origin_ckpt="r1") for every train game so each game has a true item
  control_real_aug  true   extra unused record evidence (no fabrication overlap) until true ~= false
control_distractor is probe-only and not in the sidecar. Items over --max-prompt-tokens are dropped
and counted (memory cap for the aux forward).

  .venv/bin/python datasets/old_bailey/aux_loss/build_aux_ce_sidecar.py \
      --report-dir <dir holding the claim-level audits> [--sizes 4B 8B]

Outputs <out-dir>/{4B,8B}/sidecar.jsonl.gz and <out-dir>/_meta.json. Needs the repo .venv (imports
evaluation.rl_rollout -> verl).
"""
import argparse
import gzip
import hashlib
import json
import random
import re
import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rl.fab_aux_common import (  # noqa: E402
    DEFAULT_SRC,
    GENERATED,
    SCHEMA_VERSION,
    TOKENIZER_PATH,
)

CKPTS = ("base", "gs50", "gs100")
CKPT_STEP = {"base": 0, "gs50": 50, "gs100": 100, "r1": None}
GROUND_TRUTH = {"fabrication": "false", "control_real": "true", "control_real_aug": "true",
                # kinds added by the rebalanced variants
                "claim_real": "true", "paraphrase_real": "true", "altered_real": "false"}
_WS_RE = re.compile(r"\s+")


def _norm(s: str) -> str:
    return _WS_RE.sub(" ", (s or "").strip().lower())


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _latest_audit_by_gid(path: Path) -> dict:
    by_gid = {}
    with open(path) as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                by_gid[rec["game_id"]] = rec       # append-only file: latest record wins
    return by_gid


def build_size(size, args, rd, train_ids, compose, bi, agent, tok):
    """-> (lines, meta) for one size. `bi` is evaluation.source_of_fabrication.build_probe_items."""
    results, audits = {}, {}
    sources = {}
    for ck in CKPTS:
        rf = Path(args.results[f"{size}_{ck}"])
        af = rd / args.audit_tmpl.format(size=size, ck=ck)
        for p in (rf, af):
            if not p.exists():
                raise SystemExit(f"[sidecar] missing input: {p}")
            sources[str(p)] = _sha256(p)
        results[ck] = {g["id"]: g for g in json.load(open(rf)) if g.get("complete")}
        audits[ck] = _latest_audit_by_gid(af)

    # Any complete record supplies the game's policy-independent params for the r1 context.
    any_game = {}
    for ck in CKPTS:
        for gid, g in results[ck].items():
            any_game.setdefault(gid, g)

    items, contexts = [], {}
    stats = {"n_genuine": 0, "judge_round": {"agree": 0, "override": 0, "missing": 0, "lost": 0},
             "capped_games": 0, "n_dup_claims_skipped": 0}
    seen_false = set()          # (game_id, norm(claim)) across origins — earliest origin wins
    real_pick = {}              # game_id -> the control_real statement (shared across contexts)

    for ck in CKPTS:
        for gid in sorted(audits[ck]):
            a = audits[ck][gid]
            game = results[ck].get(gid)
            if game is None or a.get("judge_fail") or not a.get("genuine") or gid not in train_ids:
                continue
            arg_texts = [(r.get("sender") or "").strip() for r in (game.get("rounds") or [])]
            attributed = []
            for it in a["genuine"]:
                stats["n_genuine"] += 1
                r, method, status = bi.attribute_round(it["claim"], it.get("round_judge"), arg_texts)
                stats["judge_round"][status] += 1
                if r is None:
                    continue
                if (gid, _norm(it["claim"])) in seen_false:
                    stats["n_dup_claims_skipped"] += 1
                    continue
                attributed.append({"claim": it["claim"], "round": r})
            if len(attributed) > args.per_game_cap:
                stats["capped_games"] += 1
                attributed = attributed[:args.per_game_cap]

            descs = [d for d in (bi._description(ev)
                                 for ev in game["params"]["private"]["information"]) if d]
            g_rng = random.Random(args.seed * 1_000_003 + gid)
            real_pick.setdefault(gid, descs[g_rng.randrange(len(descs))] if descs else None)
            for i, f in enumerate(attributed):
                seen_false.add((gid, _norm(f["claim"])))
                ctx_id = f"{size}_{ck}_g{gid}_r{f['round']}"
                if ctx_id not in contexts:
                    c = bi.reconstruct_context(game, f["round"], agent)
                    c["context_id"] = ctx_id
                    contexts[ctx_id] = c
                items.append({"kind": "fabrication", "game_id": gid, "round": f["round"],
                              "origin_ckpt": ck, "context_id": ctx_id,
                              "item_id": f"{size}_{ck}_g{gid}_r{f['round']}_c{i}",
                              "statement": f["claim"]})
            # one control_real per DISTINCT fabricating context (mirrors the probe builder)
            for r in sorted({f["round"] for f in attributed}):
                if real_pick[gid] is None:
                    continue
                ctx_id = f"{size}_{ck}_g{gid}_r{r}"
                items.append({"kind": "control_real", "game_id": gid, "round": r,
                              "origin_ckpt": ck, "context_id": ctx_id,
                              "item_id": f"{size}_{ck}_g{gid}_r{r}_ctlreal",
                              "statement": real_pick[gid]})

    # True anchor: a policy-independent round-1 context for every train game.
    n_no_record = 0
    for gid in sorted(train_ids):
        game = any_game.get(gid)
        if game is None:
            n_no_record += 1
            continue
        descs = [d for d in (bi._description(ev)
                             for ev in game["params"]["private"]["information"]) if d]
        if not descs:
            n_no_record += 1
            continue
        g_rng = random.Random(args.seed * 1_000_003 + gid)
        real_pick.setdefault(gid, descs[g_rng.randrange(len(descs))])
        ctx_id = f"{size}_r1_g{gid}_r1"
        c = bi.reconstruct_context(game, 1, agent)
        c["context_id"] = ctx_id
        contexts[ctx_id] = c
        items.append({"kind": "control_real", "game_id": gid, "round": 1,
                      "origin_ckpt": "r1", "context_id": ctx_id,
                      "item_id": f"{size}_r1_g{gid}_r1_ctlreal",
                      "statement": real_pick[gid]})

    # Dedup true items on (game, statement), keeping the first (fabricating context).
    seen_true, deduped = set(), []
    for it in items:
        if it["kind"] == "control_real":
            key = (it["game_id"], _norm(it["statement"]))
            if key in seen_true:
                continue
            seen_true.add(key)
        deduped.append(it)
    items = deduped

    # control_real_aug: unused record evidence until true ~= false (per size).
    n_false = sum(it["kind"] == "fabrication" for it in items)
    n_true = sum(it["kind"] != "fabrication" for it in items)
    fab_norm = {}
    for it in items:
        if it["kind"] == "fabrication":
            fab_norm.setdefault(it["game_id"], set()).add(_norm(it["statement"]))
    n_excluded = 0
    candidates = []
    for gid in sorted(any_game):
        if gid not in train_ids:
            continue
        game = any_game[gid]
        descs = [d for d in (bi._description(ev)
                             for ev in game["params"]["private"]["information"]) if d]
        used = {_norm(it["statement"]) for it in items
                if it["game_id"] == gid and it["kind"] == "control_real"}
        rng_g = random.Random(args.seed * 2_000_003 + gid)
        pool = []
        for d in descs:
            nd = _norm(d)
            if nd in used:
                continue
            if any(nd == nf or nd in nf or nf in nd for nf in fab_norm.get(gid, ())):
                n_excluded += 1
                continue
            pool.append(d)
        rng_g.shuffle(pool)
        # aug items ride the game's cheapest context (the r1 anchor context)
        for d in pool:
            candidates.append((gid, d, f"{size}_r1_g{gid}_r1"))
    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    deficit = max(0, n_false - n_true)
    picked = [c for c in candidates if c[2] in contexts][:deficit]
    for k, (gid, statement, ctx_id) in enumerate(sorted(picked)):
        items.append({"kind": "control_real_aug", "game_id": gid,
                      "round": int(contexts[ctx_id]["round"]), "origin_ckpt": "r1",
                      "context_id": ctx_id, "item_id": f"{ctx_id}_ctlrealaug{k}",
                      "statement": statement})
    print(f"[aug] {size}: deficit {deficit}, +{len(picked)} control_real_aug "
          f"({n_excluded} excluded for fabrication overlap)")

    # Compose messages; drop (and count) items over the token cap.
    lines, max_tok, n_over = [], 0, {"fabrication": 0, "control_real": 0, "control_real_aug": 0}
    for it in sorted(items, key=lambda x: (x["game_id"], x["item_id"])):
        probe_item = {"context_id": it["context_id"], "round": it["round"],
                      "statement": it["statement"]}
        messages = compose(probe_item, contexts, "in", "role", "pre")
        assert messages[0]["role"] == "system" and messages[-1]["role"] == "user"
        assert it["statement"] in messages[-1]["content"]
        ids = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True,
                                      return_dict=False, enable_thinking=False)
        assert isinstance(ids, list) and isinstance(ids[0], int), type(ids)
        if len(ids) > args.max_prompt_tokens:
            n_over[it["kind"]] += 1
            continue
        max_tok = max(max_tok, len(ids))
        lines.append({"schema_version": SCHEMA_VERSION, "size": size,
                      "game_id": int(it["game_id"]), "item_id": it["item_id"],
                      "kind": it["kind"], "label": GROUND_TRUTH[it["kind"]],
                      "statement": it["statement"], "round": int(it["round"]),
                      "origin_ckpt": it["origin_ckpt"],
                      "origin_step": CKPT_STEP.get(it["origin_ckpt"]),
                      "context_id": it["context_id"], "messages": messages})

    by_kind, games_true, games_false = {}, set(), set()
    for ln in lines:
        by_kind[ln["kind"]] = by_kind.get(ln["kind"], 0) + 1
        (games_false if ln["label"] == "false" else games_true).add(ln["game_id"])
    n_true_f = sum(1 for ln in lines if ln["label"] == "true")
    n_false_f = sum(1 for ln in lines if ln["label"] == "false")
    meta = {"by_kind": by_kind, "n_true": n_true_f, "n_false": n_false_f,
            "n_items": len(lines), "n_contexts": len(contexts),
            "n_games_with_true": len(games_true), "n_games_with_false": len(games_false),
            "n_train_ids": len(train_ids), "n_train_ids_without_record": n_no_record,
            "n_dropped_overlength": n_over, "max_prompt_tokens": max_tok,
            "audit_stats": stats, "sources_sha256": sources}
    print(f"[build] {size}: {len(lines)} items ({n_false_f} FALSE / {n_true_f} TRUE) over "
          f"{len(contexts)} contexts; TRUE coverage {len(games_true)}/{len(train_ids)} games, "
          f"FALSE coverage {len(games_false)}/{len(train_ids)}; dropped overlength {n_over}; "
          f"max prompt {max_tok} tok")
    return lines, meta


def main():
    global CKPTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", default=["4B", "8B"], choices=["4B", "8B"])
    ap.add_argument("--report-dir", required=True,
                    help="directory the --audit-tmpl paths are relative to (the claim-level audits "
                         "evaluation/source_of_fabrication/audit_rollouts.py wrote)")
    ap.add_argument("--sender-prompt", default="strategies", metavar="SPEC",
                    help="the sender prompt the train-split rollouts were played under "
                         "(rl/sender_prompts.py); the aux items reconstruct their context from it")
    ap.add_argument("--out-dir", default=str(DEFAULT_SRC))
    ap.add_argument("--split-file",
                    default=str(GENERATED / "rl_sftsplit_strategies/sft_split.json"))
    ap.add_argument("--results-root", default=str(_REPO / "experiments/results/old-bailey"),
                    help="root the --results-json / built-in rollout paths resolve against "
                         "(where the eval launchers write)")
    ap.add_argument("--origins", nargs="+", default=list(CKPTS),
                    help="the per-origin collections FALSE claims are harvested from, deduped "
                         "across origins with the earliest winning. Default = the three RL "
                         "checkpoints; the SFT-holdout build passes independent base-policy "
                         "passes instead.")
    ap.add_argument("--audit-tmpl", default="audits/audit__{size}__{ck}__train.jsonl",
                    help="claim-level audit path relative to --report-dir, with {size}/{ck}")
    ap.add_argument("--results-json", default=None,
                    help="JSON map {'<size>_<origin>': <rollout path>} replacing the built-in "
                         "train-split map. Paths may be absolute or relative to --results-root.")
    ap.add_argument("--ids-key", default="rl_train_ids",
                    help="which id list in --split-file to restrict games to. rl_train_ids (921, "
                         "the GRPO split) by default; sft_train_ids (200) builds an awareness set "
                         "on the SAME games as the distillation corpus.")
    ap.add_argument("--per-game-cap", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0,
                    help="matches the probe builder's control_real per-game seeding")
    ap.add_argument("--max-prompt-tokens", type=int, default=8192)
    args = ap.parse_args()

    rd = Path(args.report_dir).resolve()
    from agents.agent_quality import AgentQuality
    from agents.model.model import ModelAPI
    from evaluation.source_of_fabrication import build_probe_items as bi
    from evaluation.source_of_fabrication.probe_prompts import compose
    from rl.sender_prompts import resolve_sender_prompt
    from transformers import AutoTokenizer
    # build_size uses bi._description (re-exported rl.monitors._description) so normalization
    # matches the item builder.
    assert "strategies" in args.sender_prompt, \
        f"the aux items are built under the strategies prompt, not {args.sender_prompt!r}"
    agent = AgentQuality(resolve_sender_prompt(bi.DOMAIN, args.sender_prompt), ModelAPI())

    CKPTS = tuple(args.origins)

    RB = Path(args.results_root)
    _default_results = {
        "4B_base": RB / "qwen3-4B-instruct-base/stubborn_oldbailey_rlrollout_train_sender-initial_base_oldbailey_strategies__recv_qwen3.5-35B.json",
        "8B_base": RB / "qwen3-8B-base/stubborn_oldbailey_rlrollout_train_sender-initial_base_oldbailey_strategies__recv_qwen3.5-35B.json",
        "4B_gs50": RB / "qwen3-4B-rl-stubborn-sftsplit-misrepv2-gs50/stubborn_oldbailey_rlrollout_train__recv_qwen3.5-35B.json",
        "8B_gs50": RB / "qwen3-8B-rl-stubborn-sftsplit-misrepv2-gs50/stubborn_oldbailey_rlrollout_train__recv_qwen3.5-35B.json",
        "4B_gs100": RB / "qwen3-4B-rl-stubborn-sftsplit-misrepv2/stubborn_oldbailey_rlrollout_train__recv_qwen3.5-35B.json",
        "8B_gs100": RB / "qwen3-8B-rl-stubborn-sftsplit-misrepv2/stubborn_oldbailey_rlrollout_train__recv_qwen3.5-35B.json",
    }
    if args.results_json:
        args.results = {k: (Path(v) if Path(v).is_absolute() else RB / v)
                        for k, v in json.load(open(args.results_json)).items()}
    else:
        args.results = _default_results
    _want = {f"{sz}_{ck}" for sz in args.sizes for ck in CKPTS}
    _miss = sorted(_want - set(args.results))
    if _miss:
        raise SystemExit(f"[sidecar] no rollout mapped for: {_miss} -- pass --results-json")

    split = json.load(open(args.split_file))
    if args.ids_key not in split:
        raise SystemExit(f"[sidecar] --split-file has no {args.ids_key!r} (has {sorted(split)})")
    train_ids = set(split[args.ids_key]) if isinstance(split[args.ids_key], list) else None
    if not train_ids:
        raise SystemExit(f"[sidecar] {args.ids_key!r} is empty")
    # 921 is an invariant of the GRPO split only.
    if args.ids_key == "rl_train_ids":
        assert len(train_ids) == 921, f"unexpected rl_train_ids: {len(train_ids)}"
    print(f"[sidecar] restricting to {len(train_ids)} games from {args.ids_key!r}")

    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    try:
        commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=str(_REPO), text=True,
                                capture_output=True, check=True).stdout.strip()
    except subprocess.CalledProcessError:
        commit = "unknown"
    meta = {"schema_version": SCHEMA_VERSION, "git_commit": commit, "arm": "in_pre_role",
            "sender_prompt": args.sender_prompt,
            "split_file": str(args.split_file), "ids_key": args.ids_key,
            "n_ids": len(train_ids), "origins": list(CKPTS),
            "audit_tmpl": args.audit_tmpl, "per_game_cap": args.per_game_cap,
            "seed": args.seed, "max_prompt_tokens_cap": args.max_prompt_tokens, "sizes": {}}

    for size in args.sizes:
        tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH[size])
        assert tok.chat_template, f"{TOKENIZER_PATH[size]} has no chat template"
        lines, m = build_size(size, args, rd, train_ids, compose, bi, agent, tok)
        m["tokenizer_path"] = TOKENIZER_PATH[size]
        m["chat_template_sha"] = hashlib.sha256(tok.chat_template.encode()).hexdigest()
        out_dir = out_root / size
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "sidecar.jsonl.gz"
        with gzip.GzipFile(out_path, "wb", mtime=0) as gz:
            for ln in lines:
                gz.write((json.dumps(ln, ensure_ascii=False) + "\n").encode())
        m["sidecar_sha256"] = _sha256(out_path)
        meta["sizes"][size] = m
        print(f"[write] {out_path}: {m['n_items']} items")

    (out_root / "_meta.json").write_text(json.dumps(meta, indent=1) + "\n")
    print(f"[done] meta -> {out_root / '_meta.json'}")


if __name__ == "__main__":
    main()
