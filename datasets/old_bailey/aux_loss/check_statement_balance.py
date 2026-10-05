#!/usr/bin/env python3
"""Gate an aux-CE probe sidecar on surface-form balance between its TRUE and FALSE classes.

In base_origin (the "fabrication awareness" step, used as an SFT stage and as AUX_CE=1) the classes
differ by form, not truth: FALSE items are judge-extracted claims (mean 60 chars), TRUE items are
verbatim record Description lines (mean 189 chars). A single length threshold classifies it at
95.0% (94.6% 4B / 100.0% 8B on the held-out probe split), an 8-feature surface classifier reaches
95.9% under game-grouped CV, and TRUE statements (almost never FALSE ones) are verbatim substrings
of the context. Probe accuracy on such a set says nothing about record-checking, so this script
fails on it.

  .venv/bin/python datasets/old_bailey/aux_loss/check_statement_balance.py --variant balanced
  # self-test: must trip on the known-confounded parent
  .venv/bin/python datasets/old_bailey/aux_loss/check_statement_balance.py --variant base_origin --expect-fail
  .venv/bin/python datasets/old_bailey/aux_loss/check_statement_balance.py \\
      --sidecar datasets/old_bailey/_generated/aux_ce_probe/variants/balanced/4B/sidecar.jsonl.gz \\
      --json-out /tmp/balance.json

--expect-fail inverts the exit code (0 only if some gate trips), so a broken checker cannot pass.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

_REPO = Path(__file__).resolve().parents[3]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from rl.fab_aux_common import (  # noqa: E402
    DEF_VERBATIM_RUN,
    DEFAULT_SRC,
    DEFAULT_TEMPLATE,
    SIZES,
    description,
    load_sidecar,
    longest_shared_run,
)
# A rewrite that flags its altered detail with caps or emphasis gives the probe the answer. The
# record has legitimate all-caps (NOT GUILTY, surnames, PC/AM/PM/DNA), so all-caps is gated on
# TRUE-vs-FALSE skew; emphasis markup is never legitimate.
_ALLCAPS = re.compile(r"\b[A-Z]{2,}\b")
_EMPHASIS = re.compile(r"\*\*|__|<b>|</b>|<em>|</em>|\[ALTERED\]|\[MODIFIED\]|<<|>>")

# Gate thresholds. The AUC band is two-sided: reliably shorter is as exploitable as longer.
DEF_AUC_LO, DEF_AUC_HI = 0.45, 0.55
DEF_MAX_LEN_ACC = 0.55          # best single-threshold accuracy from length alone
DEF_MAX_SURFACE_ACC = 0.55      # 8-feature surface-only logistic, game-grouped CV
DEF_MAX_PAIR_DWORDS = 3         # |dwords| between an item and its declared partner
DEF_MAX_VERBATIM_SKEW = 0.10    # |P(verbatim | TRUE) - P(verbatim | FALSE)|, run DEF_VERBATIM_RUN
DEF_MAX_ALLCAPS_SKEW = 0.05     # |P(ALLCAPS | TRUE) - P(ALLCAPS | FALSE)|
DEF_LABEL_TOL = 0.01            # |TRUE-FALSE|/n; matches the AUX_CE_SAMPLING=random submit gate


# ---------------------------------------------------------------- surface features


FEATNAMES = ("chars", "words", "n_commas", "n_periods", "ends_period", "starts_upper",
             "has_quote", "has_digit", "has_allcaps")


def features(s: str):
    # ends_period/starts_upper catch LLM-written TRUE sentences vs fabrication fragments cut from
    # arguments (29.8% end in a full stop, 69.2% open upper-case).
    return [len(s), len(s.split()), s.count(","), s.count("."),
            float(s.rstrip().endswith(".")), float(s[:1].isupper()),
            float("'" in s or '"' in s),
            float(any(c.isdigit() for c in s)),
            float(bool(_ALLCAPS.search(s)))]


def _fit_logreg(X, y, iters=4000, lr=0.5, l2=1e-3):
    """Plain-numpy logistic regression -- sklearn is not a dependency of this env."""
    X = np.hstack([np.ones((len(X), 1)), X])
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1.0 / (1.0 + np.exp(-X @ w))
        g = X.T @ (p - y) / len(y)
        g[1:] += l2 * w[1:]
        w -= lr * g
    return w


def _predict(w, X):
    X = np.hstack([np.ones((len(X), 1)), X])
    return (1.0 / (1.0 + np.exp(-X @ w))) >= 0.5


def grouped_cv_accuracy(X, y, groups, n_splits=5, seed=0):
    """Game-grouped k-fold accuracy (a game supplies several items, so rows are not independent)."""
    X = np.asarray(X, float)
    y = np.asarray(y, float)
    games = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(games)
    accs = []
    for fold in np.array_split(games, n_splits):
        te = np.isin(groups, fold)
        tr = ~te
        if tr.sum() == 0 or te.sum() == 0 or len(np.unique(y[tr])) < 2:
            continue
        # standardize with train-fold statistics only
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1.0
        Xtr, Xte = (X[tr] - mu) / sd, (X[te] - mu) / sd
        accs.append(float((_predict(_fit_logreg(Xtr, y[tr]), Xte) == y[te]).mean()))
    return float(np.mean(accs)) if accs else float("nan")


def _best_split_acc(y_sorted, boundary):
    """Best accuracy over all threshold positions, both polarities; y_sorted is label-by-value.

    Vectorized so each permutation is one cumsum (values do not move under a label shuffle).
    """
    n = len(y_sorted)
    cum_pos = np.concatenate([[0.0], np.cumsum(y_sorted)])   # cum_pos[i] = #TRUE in [0, i)
    idx = np.arange(n + 1, dtype=float)
    acc_ge = ((cum_pos[-1] - cum_pos) + (idx - cum_pos)) / n  # predict TRUE at positions >= i
    acc = np.maximum(acc_ge, 1.0 - acc_ge)                    # the other polarity is the mirror
    return float(acc[boundary].max())


def best_threshold_accuracy(pos, neg, n_perm=0, seed=0):
    """Best accuracy of one threshold on a scalar, both polarities.

    In-sample, so optimistic, and more so at small n (on 134 balanced items noise clears 56%). With
    n_perm > 0 it also returns a permutation p-value, which has no such bias.
    """
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    vals = np.concatenate([pos, neg])
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    order = np.argsort(vals, kind="mergesort")
    vs, ys = vals[order], y[order]
    # only cut between distinct values
    boundary = np.ones(len(vs) + 1, dtype=bool)
    boundary[1:-1] = vs[1:] != vs[:-1]
    best = _best_split_acc(ys, boundary)

    # Report the threshold itself for the human-readable line.
    thr, pol = None, None
    for t in np.unique(vs):
        for ge in (True, False):
            acc = float((((vs >= t) if ge else (vs < t)).astype(float) == ys).mean())
            if abs(acc - best) < 1e-12:
                thr, pol = float(t), (">=" if ge else "<")
                break
        if thr is not None:
            break

    p = None
    if n_perm:
        rng = np.random.default_rng(seed)
        ge = 0
        for _ in range(n_perm):
            if _best_split_acc(rng.permutation(ys), boundary) >= best - 1e-12:
                ge += 1
        p = (ge + 1) / (n_perm + 1)
    return best, thr, pol, p


def auc(pos, neg):
    """P(random TRUE > random FALSE), ties as half (Mann-Whitney U)."""
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    u = stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return float(u / (len(pos) * len(neg)))


def cohens_d(pos, neg):
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) < 2 or len(neg) < 2:
        return float("nan")
    sp = np.sqrt(((len(pos) - 1) * pos.var(ddof=1) + (len(neg) - 1) * neg.var(ddof=1))
                 / (len(pos) + len(neg) - 2))
    return float((pos.mean() - neg.mean()) / sp) if sp > 0 else 0.0


# ---------------------------------------------------------------- the gate


def record_descriptions(template: Path):
    """game_id -> [evidence Description lines]. Used for the verbatim-overlap check."""
    games = json.load(open(template))
    out = {}
    for g in games:
        info = (((g.get("params") or {}).get("private") or {}).get("information")) or []
        out[g["id"]] = [d for d in (description(e) for e in info) if d]
    return out


def audit(rows, records, args):
    """-> (report dict, list of failure strings)."""
    for r in rows:
        r["_s"] = r["statement"]
    pos = [r for r in rows if r["label"] == "true"]
    neg = [r for r in rows if r["label"] == "false"]
    rep = {
        "n_items": len(rows), "n_true": len(pos), "n_false": len(neg),
        "by_kind": dict(Counter(r.get("kind", "?") for r in rows)),
        "by_kind_label": {f"{k}/{v}": n for (k, v), n
                          in sorted(Counter((r.get("kind", "?"), r["label"]) for r in rows).items())},
        "n_games": len({r["game_id"] for r in rows}),
    }
    fails = []
    if not pos or not neg:
        return rep, ["one class is empty -- nothing to compare"]

    # --- 1. length, in characters and in words
    for metric, fn in (("chars", len), ("words", lambda s: len(s.split()))):
        p = [fn(r["_s"]) for r in pos]
        n = [fn(r["_s"]) for r in neg]
        a = auc(p, n)
        acc, thr, pol, perm_p = best_threshold_accuracy(p, n, n_perm=args.n_perm, seed=args.seed)
        mw = stats.mannwhitneyu(p, n, alternative="two-sided")
        rep[metric] = {
            "true_mean": float(np.mean(p)), "true_sd": float(np.std(p, ddof=1)),
            "false_mean": float(np.mean(n)), "false_sd": float(np.std(n, ddof=1)),
            "auc": a, "cohens_d": cohens_d(p, n), "mannwhitney_p": float(mw.pvalue),
            "best_threshold_acc": acc, "threshold": thr, "polarity": pol,
            "threshold_perm_p": perm_p,
        }
        # AUC's null is 0.5 at any n, so it gates alone.
        if not (args.auc_lo <= a <= args.auc_hi):
            fails.append(f"{metric}: AUC {a:.3f} outside [{args.auc_lo}, {args.auc_hi}] "
                         f"(TRUE {np.mean(p):.1f} vs FALSE {np.mean(n):.1f})")
        # The fitted threshold is biased up at small n, so it must also beat its permutation null.
        if acc > args.max_len_acc and (perm_p is None or perm_p < args.perm_alpha):
            fails.append(f"{metric}: a single threshold reaches {acc * 100:.1f}% "
                         f"(> {args.max_len_acc * 100:.0f}%, permutation p="
                         f"{'n/a' if perm_p is None else f'{perm_p:.3f}'}) -- length alone "
                         "classifies the set")

    # --- 2. every surface feature we can think of, jointly
    X = [features(r["_s"]) for r in rows]
    y = [r["label"] == "true" for r in rows]
    g = [r["game_id"] for r in rows]
    surf = grouped_cv_accuracy(X, y, np.asarray(g), seed=args.seed)
    rep["surface_cv_acc"] = surf
    rep["surface_feature_rates"] = {
        nm: {"true": float(np.mean([features(r["_s"])[i] for r in pos])),
             "false": float(np.mean([features(r["_s"])[i] for r in neg]))}
        for i, nm in enumerate(FEATNAMES)
    }
    if surf > args.max_surface_acc:
        fails.append(f"surface-only classifier reaches {surf * 100:.1f}% under game-grouped CV "
                     f"(> {args.max_surface_acc * 100:.0f}%) -- the classes differ by form, not truth")

    # --- 3. verbatim overlap with the case record (the record-matching shortcut)
    if records:
        def is_verbatim(r):
            lines = records.get(r["game_id"], [])
            return any(longest_shared_run(r["_s"], d, args.verbatim_run) for d in lines)
        vp = float(np.mean([is_verbatim(r) for r in pos]))
        vn = float(np.mean([is_verbatim(r) for r in neg]))
        rep["verbatim"] = {"true_rate": vp, "false_rate": vn, "skew": abs(vp - vn),
                           "run_words": args.verbatim_run}
        if abs(vp - vn) > args.max_verbatim_skew:
            fails.append(f"verbatim-overlap skew {abs(vp - vn) * 100:.1f}pp "
                         f"(TRUE {vp * 100:.1f}% vs FALSE {vn * 100:.1f}%) -- "
                         f"'appears in the record' predicts the label")

    # --- 3b. decoration: no emphasis markup, and no all-caps skew between the classes
    emph = [r["item_id"] for r in rows if _EMPHASIS.search(r["_s"])]
    cp = float(np.mean([bool(_ALLCAPS.search(r["_s"])) for r in pos]))
    cn = float(np.mean([bool(_ALLCAPS.search(r["_s"])) for r in neg]))
    rep["decoration"] = {"emphasis_markup": len(emph), "allcaps_true": cp, "allcaps_false": cn,
                         "allcaps_skew": abs(cp - cn)}
    if emph:
        fails.append(f"{len(emph)} statement(s) carry emphasis markup (e.g. {emph[:3]}) -- a "
                     "rewrite must never flag the detail it changed")
    if abs(cp - cn) > args.max_allcaps_skew:
        fails.append(f"all-caps skew {abs(cp - cn) * 100:.1f}pp (TRUE {cp * 100:.1f}% vs FALSE "
                     f"{cn * 100:.1f}%) -- capitalisation predicts the label")
    # Same-source twins only (paraphrase/altered rewrites of one record line), where differing
    # all-caps words mark the altered one. Block-A pairs are different texts, so not checked.
    twins = defaultdict(dict)
    for r in rows:
        if r.get("pair_id") and r.get("source_evidence"):
            twins[r["pair_id"]][r["label"]] = (r["_s"], r["source_evidence"])
    bad_twins = [k for k, d in twins.items() if len(d) == 2
                 and d["true"][1] == d["false"][1]
                 and set(_ALLCAPS.findall(d["true"][0])) != set(_ALLCAPS.findall(d["false"][0]))]
    rep["decoration"]["twin_allcaps_mismatch"] = len(bad_twins)
    if bad_twins:
        fails.append(f"{len(bad_twins)} same-source pair(s) whose two statements shout DIFFERENT "
                     f"words (e.g. {bad_twins[:3]}) -- that marks the altered one")

    # --- 4. declared pairs, when the builder recorded them
    dw = [abs(len(r["_s"].split()) - int(r["partner_words"]))
          for r in rows if r.get("partner_words") is not None]
    if dw:
        rep["pair_dwords"] = {"n": len(dw), "mean": float(np.mean(dw)),
                              "max": int(max(dw)), "over_tol": int(sum(d > args.max_pair_dwords
                                                                       for d in dw))}
        if max(dw) > args.max_pair_dwords:
            fails.append(f"{sum(d > args.max_pair_dwords for d in dw)} item(s) exceed the "
                         f"pair word tolerance (max |dwords| = {max(dw)} > {args.max_pair_dwords})")
    else:
        rep["pair_dwords"] = None

    # --- 5. label balance, and both classes drawn from the same games
    # 1% matches the AUX_CE_SAMPLING=random submit gate. A game-level split needs ~0.05, since
    # block A pairs some rows across games.
    skew = abs(len(pos) - len(neg)) / max(1, len(rows))
    rep["label_skew"] = skew
    if skew > args.label_tol:
        fails.append(f"label imbalance: {len(pos)} TRUE vs {len(neg)} FALSE "
                     f"({skew * 100:.1f}% > {args.label_tol * 100:.0f}%)")
    gt = {r["game_id"] for r in pos}
    gf = {r["game_id"] for r in neg}
    rep["games"] = {"true_only": len(gt - gf), "false_only": len(gf - gt), "both": len(gt & gf)}
    if gt ^ gf:
        fails.append(f"class/game confound: {len(gt - gf)} game(s) supply only TRUE items and "
                     f"{len(gf - gt)} only FALSE -- the case itself predicts the label")
    return rep, fails


def render(size, path, rep, fails):
    print(f"\n{'=' * 78}\n{size}  {path}\n{'=' * 78}")
    print(f"  items={rep['n_items']}  true={rep['n_true']}  false={rep['n_false']}  "
          f"games={rep['n_games']}")
    print(f"  kinds: {rep['by_kind_label']}")
    if "chars" in rep:
        for m in ("chars", "words"):
            d = rep[m]
            pp = d.get("threshold_perm_p")
            print(f"  {m:6s} TRUE {d['true_mean']:7.1f}+-{d['true_sd']:5.1f} | "
                  f"FALSE {d['false_mean']:7.1f}+-{d['false_sd']:5.1f} | "
                  f"AUC {d['auc']:.3f}  d {d['cohens_d']:+.2f}  "
                  f"1-threshold {d['best_threshold_acc'] * 100:.1f}%"
                  f"{'' if pp is None else f' (perm p={pp:.3f})'}")
    if rep.get("surface_cv_acc") == rep.get("surface_cv_acc"):   # not NaN
        print(f"  surface-only classifier (game-grouped CV): {rep['surface_cv_acc'] * 100:.1f}%")
    if rep.get("verbatim"):
        v = rep["verbatim"]
        print(f"  verbatim in record: TRUE {v['true_rate'] * 100:.1f}%  "
              f"FALSE {v['false_rate'] * 100:.1f}%  skew {v['skew'] * 100:.1f}pp")
    if rep.get("decoration"):
        d = rep["decoration"]
        print(f"  decoration: emphasis markup {d['emphasis_markup']}  |  ALLCAPS TRUE "
              f"{d['allcaps_true'] * 100:.2f}% vs FALSE {d['allcaps_false'] * 100:.2f}% "
              f"(skew {d['allcaps_skew'] * 100:.2f}pp)  |  same-source twin mismatches "
              f"{d['twin_allcaps_mismatch']}")
    if rep.get("pair_dwords"):
        p = rep["pair_dwords"]
        print(f"  pair |dwords|: mean {p['mean']:.2f}  max {p['max']}  over-tol {p['over_tol']}")
    print(f"  games: both-class {rep['games']['both']}  "
          f"true-only {rep['games']['true_only']}  false-only {rep['games']['false_only']}")
    if fails:
        print("  FAIL:")
        for f in fails:
            print(f"    - {f}")
    else:
        print("  PASS -- no surface shortcut found above threshold")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--variant", help="variant tag under <src>/variants/")
    src.add_argument("--sidecar", help="path to one sidecar.jsonl.gz")
    ap.add_argument("--src", default=str(DEFAULT_SRC), help="aux_ce_probe root")
    ap.add_argument("--sizes", nargs="+", default=list(SIZES))
    ap.add_argument("--template", default=str(DEFAULT_TEMPLATE),
                    help="game template JSON supplying the case record; '' to skip that check")
    ap.add_argument("--json-out", help="write the full report here")
    ap.add_argument("--expect-fail", action="store_true",
                    help="INVERT the exit code -- pass only if some gate trips (self-test on a "
                         "known-confounded variant)")
    ap.add_argument("--auc-lo", type=float, default=DEF_AUC_LO)
    ap.add_argument("--auc-hi", type=float, default=DEF_AUC_HI)
    ap.add_argument("--max-len-acc", type=float, default=DEF_MAX_LEN_ACC)
    ap.add_argument("--max-surface-acc", type=float, default=DEF_MAX_SURFACE_ACC)
    ap.add_argument("--max-pair-dwords", type=int, default=DEF_MAX_PAIR_DWORDS)
    ap.add_argument("--max-verbatim-skew", type=float, default=DEF_MAX_VERBATIM_SKEW)
    ap.add_argument("--max-allcaps-skew", type=float, default=DEF_MAX_ALLCAPS_SKEW)
    ap.add_argument("--label-tol", type=float, default=DEF_LABEL_TOL,
                    help="allowed |TRUE-FALSE|/n; raise to ~0.05 when auditing a game-level split")
    ap.add_argument("--verbatim-run", type=int, default=DEF_VERBATIM_RUN)
    ap.add_argument("--n-perm", type=int, default=200,
                    help="permutation draws calibrating the fitted-threshold accuracy (0 = off)")
    ap.add_argument("--perm-alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.sidecar:
        targets = [("-", Path(args.sidecar))]
    else:
        root = Path(args.src) / "variants" / args.variant
        targets = [(s, root / s / "sidecar.jsonl.gz") for s in args.sizes]
        # A missing requested size is an error, not a skip (use --sizes to gate a subset).
        missing = [str(p) for _, p in targets if not p.exists()]
        if missing:
            raise SystemExit(f"[balance] requested sizes {args.sizes} but these sidecars are "
                             f"missing: {missing}. Pass --sizes explicitly to gate a subset.")

    records = record_descriptions(Path(args.template)) if args.template else {}
    out, any_fail = {}, False
    for size, path in targets:
        rows = load_sidecar(path)
        rep, fails = audit(rows, records, args)
        rep["path"] = str(path)
        rep["failures"] = fails
        render(size, path, rep, fails)
        out[size] = rep
        any_fail |= bool(fails)

    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(out, indent=2))
        print(f"\n-> {args.json_out}")

    if args.expect_fail:
        if any_fail:
            print("\n[balance] --expect-fail: gates tripped as expected -- checker is live.")
            return 0
        print("\n[balance] --expect-fail: NOTHING tripped on a variant known to be confounded. "
              "The checker itself is broken.", file=sys.stderr)
        return 1
    if any_fail:
        print("\n[balance] FAILED -- this sidecar's classes are separable by surface form.",
              file=sys.stderr)
        return 1
    print("\n[balance] all gates passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
