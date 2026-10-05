#!/usr/bin/env python
"""Game-level correlation of technique presence with juror belief; the belief figure's only input.

Per game, joins the audit sidecar's 42-dim binary technique vector with the juror's final
P(guilty) (and the regex-parsed round-1 belief, for the shift outcome). Observational, so fixed
effects absorb two confounds: the cell (arm x sender x juror; single_strategy:X forces X and also
changes persuasion) and the case (game id / evidence strength; the same 100 held-out cases repeat
in every cell).

Estimators per technique x outcome (final P(guilty), or shift = final - round-1):
  r_naive   pooled point-biserial (confounded; reference only).
  r_within  Fisher-z average of per-cell point-biserials (controls cell, not case).
  r_adj     primary: Pearson r of two-way-FE residuals (cell + case, alternating demeaning);
            95% CI by delete-one-case jackknife (cluster-robust).
  beta_ols  joint OLS of the residualized outcome on all estimable residualized indicators: the
            partial Delta P(guilty) holding co-occurring techniques fixed (techniques travel in
            families, so univariate r smears credit); jackknife-by-case CI.

Sources: all_arms (base + guide + 42 single-technique arms) is primary, as only it makes every
technique estimable; base_guide (no steering) is the confound-free check on the ~half as many
techniques that occur unprompted. Estimable iff >= MIN_POS games observed present and >= MIN_POS
absent. Judge parse-fail (None) presence is cell-mean imputed (~0.1%, reported).

Final belief follows evaluation/old_bailey/evaluate_oldbailey.py (complete games, None receiver
rounds skipped, last remaining round via evaluation.belief.parse_belief). `--belief-source metrics`
reads it from the evaluator's `<result>.metrics.json` instead, for comparison.

  python evaluation/reward_landscape/analyze_belief_correlation.py [--results-root DIR]
      [--audit-dir DIR] [--manifest FILE] [--senders DIR ...] [--jurors PROFILE ...]
      [--belief-source rollout|metrics] [--out FILE]
"""
import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import scipy.sparse as sp

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from evaluation.belief import is_valid_belief, parse_belief  # noqa: E402
from evaluation.reward_landscape import audit_source  # noqa: E402
from rl.sender_prompts import all_specs, arm_key, parse_spec  # noqa: E402
from rl.strategy_audit.taxonomy import by_slug, slugs, slugs_by_legitimacy  # noqa: E402

# Display name (spells panel keys) -> results-directory slug (what --senders selects);
# juror key -> profile.
SENDERS = {"gpt-5.5": "gpt-5.5", "4B": "qwen3-4B-instruct-base",
           "8B": "qwen3-8B-base", "14B": "qwen3-14B-base"}
PROFILE = {"neut": "neutral", "stub": "stubborn"}
PAPER_SENDERS = ("qwen3-4B-instruct-base", "qwen3-8B-base")
PAPER_JURORS = ("stubborn",)

SLUGS = slugs()
N = len(SLUGS)
NAME = {s: by_slug()[s]["name"] for s in SLUGS}
LEGIT = slugs_by_legitimacy("legit")
CO = slugs_by_legitimacy("coercive")
DE = slugs_by_legitimacy("deceptive")

MIN_POS = 15            # technique estimable iff >= this many observed present AND absent games
MIN_CELL_GAMES = 30     # r_within: skip cells smaller than this
MIN_CELL_POS = 8        # r_within: within a cell the technique needs >= this present AND absent
PHI_CLIP = 0.999
DEMEAN_ITERS = 200
DEMEAN_TOL = 1e-10


# Inputs
def read_manifest(path):
    """{(sender dir, profile, arm): result path} from a build_manifest.py manifest."""
    lines = Path(path).read_text().splitlines()
    cols = lines[0].split("\t")
    out = {}
    for ln in lines[1:]:
        if not ln.strip():
            continue
        r = dict(zip(cols, ln.split("\t")))
        out[(r["sender"], r["profile"], r["arm"])] = r["result"]
    return out


def cell_stems(specs, senders, jurors, results_root, manifest):
    """(spec, display name, juror key, stem) per cell, in canonical order: arm, sender, juror.

    `senders` / `jurors` filter the tables above but never reorder them.
    """
    for spec in specs:
        for disp, sender_dir in SENDERS.items():
            if sender_dir not in senders:
                continue
            for rk, profile in PROFILE.items():
                if profile not in jurors:
                    continue
                if manifest is None:
                    res = audit_source.result_path(sender_dir, profile, spec, results_root)
                else:
                    res = manifest.get((sender_dir, profile, arm_key(spec)))
                    if res is None:
                        continue
                yield spec, disp, rk, res[:-5] if res.endswith(".json") else res


def per_game_belief_rollout(base_path):
    """{game id: final P(guilty)} straight from the rollout transcript (the evaluator's rule)."""
    p = base_path + ".json"
    if not os.path.exists(p):
        return None
    out = {}
    for g in json.load(open(p)):
        if not g.get("complete"):
            continue
        beliefs = [parse_belief(r.get("receiver")) for r in (g.get("responses") or [])
                   if isinstance(r, dict) and r.get("receiver") is not None]
        final = beliefs[-1] if beliefs else None
        if is_valid_belief(final):
            out[g["id"]] = float(final)
    return out


def per_game_belief_metrics(base_path):
    """The same number as written by the evaluator into `<result>.json.metrics.json`."""
    p = base_path + ".json.metrics.json"
    if not os.path.exists(p):
        return None
    m = json.load(open(p))
    return {r[0]: r[1] for r in m.get("rows", []) if isinstance(r[1], (int, float))}


def per_game_r1(base_path):
    """{game id: round-1 belief}, for the shift outcome."""
    p = base_path + ".json"
    if not os.path.exists(p):
        return {}
    out = {}
    for g in json.load(open(p)):
        resp = (g.get("responses") or [{}])[0].get("receiver") or ""
        b = parse_belief(resp)
        if isinstance(b, (int, float)):
            out[g["id"]] = float(b)
    return out


def load_long(specs, args, manifest, required=True):
    """One row per (game, cell): case, cell, sender, juror, final, shift, x[42] (nan = unjudged).

    Returns a dict of parallel arrays. With no cell found, exits if `required` (the primary source:
    a wrong --audit-dir / --results-root / --senders), else returns None (a secondary source may be
    missing from a partial grid).
    """
    belief_of = (per_game_belief_rollout if args.belief_source == "rollout"
                 else per_game_belief_metrics)
    case_ids, cell_ids, senders, jurors, finals, shifts = [], [], [], [], [], []
    X = []
    n_cells = 0
    for spec, disp, rk, spath in cell_stems(specs, args.senders, args.jurors,
                                            args.results_root, manifest):
        audit = audit_source.load_audit_for_stem(spath, args.audit_dir)
        if audit is None:
            continue
        fb = belief_of(spath)
        if not fb:
            continue
        r1 = per_game_r1(spath)
        n_cells += 1
        cell = f"{arm_key(spec)}/{disp}/{rk}"
        for g in audit["per_game"]:
            if not g.get("n_args"):
                continue
            gid = g["id"]
            if gid not in fb:
                continue
            v = g["vector"]
            case_ids.append(gid)
            cell_ids.append(cell)
            senders.append(disp)
            jurors.append(rk)
            finals.append(float(fb[gid]))
            shifts.append(float(fb[gid]) - r1[gid] if gid in r1 else np.nan)
            X.append([np.nan if v.get(s) is None else (1.0 if v[s] >= 0.5 else 0.0)
                      for s in SLUGS])
    if n_cells == 0:
        if not required:
            return None
        raise SystemExit(
            "[belief-corr] ERROR: no cell had both an audit artifact and a belief.\n"
            f"        results root {args.results_root}\n"
            f"        audit dir    {args.audit_dir}\n"
            f"        senders      {' '.join(args.senders)}\n"
            f"        jurors       {' '.join(args.jurors)}\n"
            "        Check that the audit was published and that --senders / --jurors name the "
            "grid that was played.")
    return {"case": np.array(case_ids), "cell": np.array(cell_ids),
            "sender": np.array(senders), "juror": np.array(jurors),
            "final": np.array(finals), "shift": np.array(shifts), "X": np.array(X),
            "n_cells_read": n_cells}


# FE machinery
def _codes(labels):
    uniq, idx = np.unique(labels, return_inverse=True)
    return idx, len(uniq)


def _indicator(idx, k):
    n = len(idx)
    return sp.csr_matrix((np.ones(n), (np.arange(n), idx)), shape=(n, k))


def two_way_residual(V, cell_idx, case_idx):
    """Residualize columns of V (n x k, no nan) on cell + case FE by alternating demeaning."""
    ci, nc = _codes(cell_idx)
    gi, ng = _codes(case_idx)
    C, G = _indicator(ci, nc), _indicator(gi, ng)
    cnt_c = np.asarray(C.sum(axis=0)).ravel()
    cnt_g = np.asarray(G.sum(axis=0)).ravel()
    R = V - V.mean(axis=0, keepdims=True)
    for _ in range(DEMEAN_ITERS):
        R1 = R - C @ ((C.T @ R) / cnt_c[:, None])
        R2 = R1 - G @ ((G.T @ R1) / cnt_g[:, None])
        if np.max(np.abs(R2 - R)) < DEMEAN_TOL:
            return R2
        R = R2
    return R


def _impute_by_cell(X, cell_idx):
    """Cell-mean-impute nan presence values (falls back to column mean if a cell is all-nan)."""
    Xi = X.copy()
    ci, nc = _codes(cell_idx)
    C = _indicator(ci, nc)
    M = np.nan_to_num(Xi)
    obs = (~np.isnan(Xi)).astype(float)
    sums = C.T @ M
    cnts = C.T @ obs
    with np.errstate(invalid="ignore", divide="ignore"):
        means = np.where(cnts > 0, sums / np.maximum(cnts, 1), np.nan)
    fill = np.asarray(means)[ci]
    col_mean = np.nanmean(Xi, axis=0)
    fill = np.where(np.isnan(fill), col_mean[None, :], fill)
    nanmask = np.isnan(Xi)
    Xi[nanmask] = fill[nanmask]
    return Xi, float(nanmask.mean())


def _pearson(a, b):
    if len(a) < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def _estimable(Xcol):
    obs = Xcol[~np.isnan(Xcol)]
    pos = obs.sum()
    return len(obs) >= 2 * MIN_POS and pos >= MIN_POS and (len(obs) - pos) >= MIN_POS


def analyze_source(D, outcome):
    """Full estimator suite for one (source rows, outcome name). Returns dict per slug + joint OLS."""
    y = D[outcome]
    keep = ~np.isnan(y)
    y, X = y[keep], D["X"][keep]
    cell, case = D["cell"][keep], D["case"][keep]
    n = len(y)
    if n == 0:
        return None
    est = [j for j in range(N) if _estimable(X[:, j])]
    n_pos = [int(np.nansum(X[:, j])) for j in range(N)]

    # naive pooled r (pairwise deletion)
    r_naive = [None] * N
    for j in range(N):
        m = ~np.isnan(X[:, j])
        if j in est:
            r_naive[j] = _pearson(X[m, j], y[m])

    # within-cell Fisher-z average
    r_within = [None] * N
    cells = np.unique(cell)
    for j in est:
        zs = []
        for c in cells:
            m = (cell == c) & ~np.isnan(X[:, j])
            if m.sum() < MIN_CELL_GAMES:
                continue
            xs = X[m, j]
            if xs.sum() < MIN_CELL_POS or (len(xs) - xs.sum()) < MIN_CELL_POS:
                continue
            r = _pearson(xs, y[m])
            if r is not None:
                zs.append(math.atanh(min(PHI_CLIP, max(-PHI_CLIP, r))))
        if zs:
            r_within[j] = math.tanh(sum(zs) / len(zs))

    # two-way FE residualization (cell + case) on cell-mean-imputed X
    Xi, imp_rate = _impute_by_cell(X, cell)
    V = np.column_stack([y, Xi])
    R = two_way_residual(V, cell, case)
    yr, Xr = R[:, 0], R[:, 1:]

    def _point_estimates(yr, Xr):
        r_adj = [None] * N
        for j in est:
            r_adj[j] = _pearson(Xr[:, j], yr)
        A = Xr[:, est]
        beta, *_ = np.linalg.lstsq(A, yr, rcond=None)
        b_ols = [None] * N
        for k, j in enumerate(est):
            b_ols[j] = float(beta[k])
        return r_adj, b_ols

    r_adj, b_ols = _point_estimates(yr, Xr)

    # delete-one-case jackknife for r_adj + beta_ols (cluster-robust over the ~100 cases)
    ucases = np.unique(case)
    G = len(ucases)
    jk_r = np.full((G, N), np.nan)
    jk_b = np.full((G, N), np.nan)
    for gi, cid in enumerate(ucases):
        m = case != cid
        Rg = two_way_residual(V[m], cell[m], case[m])
        rg, bg = _point_estimates(Rg[:, 0], Rg[:, 1:])
        jk_r[gi] = [np.nan if v is None else v for v in rg]
        jk_b[gi] = [np.nan if v is None else v for v in bg]

    def _jk_ci(theta, jk_col):
        v = jk_col[~np.isnan(jk_col)]
        if theta is None or len(v) < G // 2:
            return None, None, None
        se = math.sqrt((len(v) - 1) / len(v) * float(np.sum((v - v.mean()) ** 2)))
        return se, theta - 1.96 * se, theta + 1.96 * se

    out = {}
    for j, s in enumerate(SLUGS):
        se_r, lo_r, hi_r = _jk_ci(r_adj[j], jk_r[:, j])
        se_b, lo_b, hi_b = _jk_ci(b_ols[j], jk_b[:, j])
        out[s] = {
            "n_pos": n_pos[j], "estimable": j in est,
            "r_naive": _r4(r_naive[j]), "r_within": _r4(r_within[j]),
            "r_adj": _r4(r_adj[j]), "r_adj_se": _r4(se_r),
            "r_adj_ci": [_r4(lo_r), _r4(hi_r)],
            "beta_ols": _r4(b_ols[j]), "beta_ols_se": _r4(se_b),
            "beta_ols_ci": [_r4(lo_b), _r4(hi_b)],
        }
    return {"per_strategy": out, "n_games": n, "n_cells": int(len(cells)),
            "n_cases": G, "n_estimable": len(est), "impute_rate": round(imp_rate, 5)}


def subset_r_adj(D, outcome, mask):
    """r_adj only (no jackknife) on a row subset -- for the per-juror / per-sender splits."""
    y = D[outcome]
    keep = mask & ~np.isnan(y)
    y, X = y[keep], D["X"][keep]
    cell, case = D["cell"][keep], D["case"][keep]
    if len(y) == 0:
        return None
    est = [j for j in range(N) if _estimable(X[:, j])]
    Xi, _ = _impute_by_cell(X, cell)
    R = two_way_residual(np.column_stack([y, Xi]), cell, case)
    yr, Xr = R[:, 0], R[:, 1:]
    return {SLUGS[j]: _r4(_pearson(Xr[:, j], yr)) for j in est}


def _r4(v):
    return None if v is None else round(float(v), 4)


def _panel_rank(name):
    """Panel key order: the paper's panel, then per-sender panels, then family pools.

    A row subset selected by several keys is reported once, under the first key in this order
    (stable sort), so the default panel survives a grid that collapses onto it."""
    if name == "qwen4b8b_stub":
        return 0                  # 4B + 8B: the figure's default panel
    if name.startswith("qwen_"):
        return 2                  # 4B + 8B + 14B
    return 1                      # one sender


# The two sources
PRIMARY_SOURCE = "all_arms"


def sources():
    """{source name: the prompt arms it pools}. PRIMARY_SOURCE first."""
    every = all_specs(audit_source.DOMAIN)
    return {PRIMARY_SOURCE: every, "base_guide": [parse_spec("base"), parse_spec("strategies")]}


def build(args, manifest):
    out = {"meta": {
        "slugs": SLUGS, "names": NAME,
        "taxonomy": {"legit": LEGIT, "coercive": CO, "deceptive": DE},
        "outcomes": {"final": f"final P(guilty) ({args.belief_source} belief source)",
                     "shift": "final - round-1 belief (round-1 regex-parsed; rows without a "
                              "parseable round-1 belief dropped)"},
        "estimators": {
            "r_naive": "pooled point-biserial (confounded; reference)",
            "r_within": "Fisher-z average of per-cell point-biserial (cell-controlled)",
            "r_adj": "PRIMARY: Pearson r of two-way-FE residuals (cell + case demeaned); "
                     "95% CI = delete-one-case jackknife (cluster-robust)",
            "beta_ols": "joint OLS of residualized outcome on all estimable residualized "
                        "indicators: partial Delta P(guilty), jackknife CI",
        },
        "primary": "all_arms / final / r_adj and beta_ols",
        "panels": ("all_arms.panels.{qwen_neut,qwen_stub,gpt55_neut,gpt55_stub,4B_neut,4B_stub,"
                   "8B_neut,8B_stub,14B_neut,14B_stub,qwen4b8b_stub}: the full estimator suite "
                   "(jackknife CIs) on sender x juror subsets of the all_arms rows, for whichever "
                   "subsets the selected grid contains. qwen_* pool 4B+8B+14B; the per-size panels "
                   "are ~4400 rows each, so their CIs are wider than a pooled pair's; "
                   "qwen4b8b_stub pools the two SMALL Qwen senders against the stubborn juror "
                   "(~8800 rows) and is the paper's panel. A subset that more than one key "
                   "selects (with 4B+8B only, qwen_stub IS qwen4b8b_stub) is estimated and "
                   "reported once, under qwen4b8b_stub before a per-sender key before a family "
                   "pool. gpt-5.5 is per-model already and is "
                   "NOT duplicated under a gpt-5.5_* key"),
        "guards": {"min_pos": MIN_POS, "min_cell_games": MIN_CELL_GAMES,
                   "min_cell_pos": MIN_CELL_POS},
        "results_root": args.results_root,
        "audit_dir": args.audit_dir,
        "note": ("Observational: technique presence is chosen by the model within a cell, not "
                 "randomized. r_adj/beta_ols remove cell (prompt x sender x juror) and case "
                 "(evidence strength) confounds but not within-cell selection (e.g. the model "
                 "escalating on cases it is losing)."),
    }}

    for src, specs in sources().items():
        D = load_long(specs, args, manifest, required=(src == PRIMARY_SOURCE))
        if D is None or len(D["final"]) == 0:
            continue
        print(f"[belief-corr] {src}: {D['n_cells_read']} cells, {len(D['final'])} games, "
              f"{len(np.unique(D['case']))} cases")
        out[src] = {}
        for oc in ("final", "shift"):
            res = analyze_source(D, oc)
            if res:
                out[src][oc] = res
        if src == PRIMARY_SOURCE:                   # splits: same estimator, subset rows
            out[src]["splits"] = {"r_adj_final": {}}
            for rk in PROFILE:
                if PROFILE[rk] not in args.jurors:
                    continue
                out[src]["splits"]["r_adj_final"][f"juror_{rk}"] = \
                    subset_r_adj(D, "final", D["juror"] == rk)
            for disp, sender_dir in SENDERS.items():
                if sender_dir not in args.senders:
                    continue
                out[src]["splits"]["r_adj_final"][f"sender_{disp}"] = \
                    subset_r_adj(D, "final", D["sender"] == disp)
            # Figure panels: full estimator suite on sender x juror subsets. Each spans all 44
            # arms, so cell FE still absorb prompt steering; subsets not in the grid are not stored.
            # Per-size Qwen panels (~4400 rows / 44 cells / 100 cases, like gpt55_*) stop one size
            # hiding in the family average; their CIs are ~sqrt(3) wider than pooled Qwen.
            QWEN = ("4B", "8B", "14B")
            PER_MODEL = ("4B", "8B", "14B", "gpt-5.5")
            panel_masks = {
                "qwen_neut": np.isin(D["sender"], QWEN) & (D["juror"] == "neut"),
                "qwen_stub": np.isin(D["sender"], QWEN) & (D["juror"] == "stub"),
                "gpt55_neut": (D["sender"] == "gpt-5.5") & (D["juror"] == "neut"),
                "gpt55_stub": (D["sender"] == "gpt-5.5") & (D["juror"] == "stub"),
            }
            for _d in PER_MODEL:
                for _rk in PROFILE:
                    # already estimated under the gpt55_* keys
                    if _d == "gpt-5.5":
                        continue
                    panel_masks[f"{_d}_{_rk}"] = (D["sender"] == _d) & (D["juror"] == _rk)
            # Paper's panel: 4B + 8B vs the stubborn juror, pooled to double the thin per-size
            # panels without 14B, which differs on several slugs. Cell FE absorb sender identity.
            panel_masks["qwen4b8b_stub"] = (np.isin(D["sender"], ("4B", "8B"))
                                            & (D["juror"] == "stub"))
            # Keys can select identical rows (with 4B+8B only, qwen_stub == qwen4b8b_stub); each
            # row subset is estimated once and stored under its first key in _panel_rank order.
            out[src]["panels"] = {}
            by_subset = {}
            for pname in sorted(panel_masks, key=_panel_rank):
                pm = panel_masks[pname]
                sig = pm.tobytes()
                if sig in by_subset:
                    continue
                Dm = {k: (v[pm] if isinstance(v, np.ndarray) else v) for k, v in D.items()}
                by_subset[sig] = pres = analyze_source(Dm, "final")
                if pres:
                    out[src]["panels"][pname] = pres
    return out


def summarize(out):
    for src in ("all_arms", "base_guide"):
        if src not in out or "final" not in out[src]:
            continue
        res = out[src]["final"]
        print(f"\n=== {src} / final: n={res['n_games']} games, {res['n_cells']} cells, "
              f"{res['n_estimable']}/{N} estimable, impute_rate={res['impute_rate']}")
        per = res["per_strategy"]
        for key in ("r_adj", "beta_ols"):
            ranked = sorted(((per[s][key], s) for s in SLUGS if per[s][key] is not None),
                            reverse=True)
            print(f"  -- top {key} --")
            for v, s in ranked[:8]:
                ci = per[s][f"{key}_ci"]
                print(f"    {v:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]  {s}")
            print(f"  -- bottom {key} --")
            for v, s in ranked[-8:]:
                ci = per[s][f"{key}_ci"]
                print(f"    {v:+.3f} [{ci[0]:+.3f},{ci[1]:+.3f}]  {s}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--results-root", default=None,
                    help="root of every input and output (default: $RESULTS_ROOT)")
    ap.add_argument("--audit-dir", default=None,
                    help="the audit job's by_result/ directory (default: $AUDIT_DIR)")
    ap.add_argument("--manifest", default=os.getenv("MANIFEST"),
                    help="take the cells' result paths from this manifest instead of "
                         "constructing them (default: $MANIFEST)")
    ap.add_argument("--senders", nargs="+", default=list(PAPER_SENDERS),
                    help="sender results-directory slugs (default: the two paper senders)")
    ap.add_argument("--jurors", nargs="+", default=list(PAPER_JURORS),
                    help="juror profiles (default: stubborn)")
    ap.add_argument("--belief-source", choices=("rollout", "metrics"), default="rollout",
                    help="final belief from the rollout transcript (default) or from the "
                         "evaluator's metrics sidecar")
    ap.add_argument("--out", default=None,
                    help="stats json (default: "
                         "<results root>/reward_landscape/belief_correlation_stats.json)")
    args = ap.parse_args(argv)

    args.results_root = args.results_root or audit_source.default_results_root()
    args.audit_dir = args.audit_dir or audit_source.default_audit_dir(args.results_root)
    unknown = [s for s in args.senders if s not in SENDERS.values()]
    if unknown:
        ap.error(f"unknown sender directory/ies {unknown}; known: "
                 f"{', '.join(SENDERS.values())}")
    unknown = [j for j in args.jurors if j not in PROFILE.values()]
    if unknown:
        ap.error(f"unknown juror profile(s) {unknown}; known: {', '.join(PROFILE.values())}")
    manifest = read_manifest(args.manifest) if args.manifest else None
    out_path = Path(args.out or Path(audit_source.default_out_dir(args.results_root))
                    / "belief_correlation_stats.json")

    out = build(args, manifest)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    summarize(out)
    print(f"\nwrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
