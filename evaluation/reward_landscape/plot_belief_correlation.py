#!/usr/bin/env python3
r"""The paper figure: which persuasion techniques move the juror's belief, all 42 at once.

    x = technique (42 of the taxonomy, strongest effect first)
    y = adjusted point-biserial r between technique presence in a game and the juror's final
        P(guilty), with cell (prompt x sender x juror) and case fixed effects removed

Reads only the `--stats` json from analyze_belief_correlation.py
(`all_arms.panels.<panel>.per_strategy[*].{r_adj,r_adj_ci,n_pos}` plus `meta.{names,taxonomy,
estimators,outcomes,note}`). Default panel: Qwen3-4B + 8B vs the stubborn juror, where all 42
techniques are estimable.

Layout: a wide short strip for one column; strongest effect first; class shown as hue on marker
and label (green aligned / orange distortive / purple coercive); no title and ~0 outer margin (the
manuscript owns them); y axis cropped to the data (--pad-frac) to keep 42 categories printable.

Error bars are a delete-one-case jackknife 95% CI (100 Old Bailey cases), so a technique used on
easy cases does not get a narrow interval. Observational: within-cell selection survives both
fixed effects (e.g. escalating on losing games biases coercive slugs down), so read it as
co-occurrence with a convinced juror, not causation.

`evidence_based_persuasion` and `logical_appeal` appear in >85% of games, so their r rests on a few
absences. The figure does not mark them (the manuscript says so in prose); the sidecar json keeps a
`near_universal` flag on every point.

Usage
    python evaluation/reward_landscape/plot_belief_correlation.py --stats <stats json>
    python ... --stats <s> --panel 8B_stub      # any panel in the stats json
    python ... --stats <s> --labels slug        # raw slugs, not display names
    python ... --stats <s> --sort class         # group by class, r within class
"""
import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import MultipleLocator  # noqa: E402

# Purple coercive / orange distortive match the paper's other figures; green is unused there.
# "Distortive" is the paper's name for the taxonomy's `deceptive` key.
C = {"legit": "#00926b", "deceptive": "#d55e00", "coercive": "#9257cc"}
CLS_LBL = {"legit": "Aligned", "deceptive": "Distortive", "coercive": "Coercive"}
CLS_ORDER = ["legit", "deceptive", "coercive"]
NEAR_UNIVERSAL = 0.85       # r above this presence rate is driven by absences, not by contrast

# X-label overrides: the first three drop the gloss of `Name / gloss` names, whose rotated labels
# ran ~2x the plot height; `rumors` replaces "Rumor as Evidence", which reads as a mechanism.
# `--labels slug` bypasses these.
SHORT = {"authority_impersonation": "Authority Impersonation",
         "procedural_manipulation": "Procedural Manipulation",
         "belief_coercion": "Belief Coercion",
         "rumors": "Rumors"}

plt.rcParams.update({"font.size": 8, "axes.grid": True, "axes.axisbelow": True,
                     "grid.alpha": 0.28, "grid.linewidth": 0.6,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "sans-serif",
                     # Arial, then metric-compatible stand-ins for hosts without msttcorefonts.
                     "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"],
                     # keeps the y label's italic r in Arial instead of mathtext's DejaVu
                     "mathtext.fontset": "custom", "mathtext.rm": "Arial",
                     "mathtext.it": "Arial:italic", "mathtext.bf": "Arial:bold",
                     "pdf.fonttype": 42, "ps.fonttype": 42})   # embed TrueType, not Type-3


def load(path, panel):
    S = json.loads(Path(path).read_text())
    panels = S["all_arms"]["panels"]
    if panel not in panels:
        raise SystemExit(f"panel {panel!r} not in {path}\n  have: {', '.join(panels)}")
    cls_of = {s: c for c, slugs in S["meta"]["taxonomy"].items() for s in slugs}
    return S, panels[panel], cls_of


def rows(P, cls_of, names, args):
    """The plotted points, left -> right; non-estimable techniques are dropped, not left as gaps."""
    out, dropped = [], []
    for slug, v in P["per_strategy"].items():
        if v.get("r_adj") is None:
            dropped.append(slug)
            continue
        lo, hi = v["r_adj_ci"]
        out.append({"slug": slug, "label": names.get(slug) or slug.replace("_", " "),
                    "cls": cls_of[slug], "r_adj": v["r_adj"], "ci_lo": lo, "ci_hi": hi,
                    "n_pos": v["n_pos"],
                    "near_universal": v["n_pos"] >= NEAR_UNIVERSAL * P["n_games"]})
    # Descending: the paper's focus (misrepresentation, belief coercion, false information) is
    # the large positive end.
    if args.sort == "class":
        out.sort(key=lambda d: (CLS_ORDER.index(d["cls"]), -d["r_adj"]))
    else:
        out.sort(key=lambda d: -d["r_adj"])
    return out, dropped


def draw(R, P, args):
    """One wide strip. Axes are placed in absolute inches so --plot-w/--plot-h mean what they say."""
    # Figure box = axes + left gutter; x labels overhang it and bbox_inches="tight" pulls them into
    # the crop, so there is no hand-tuned bottom gutter.
    W, H = args.plot_w + args.left, args.plot_h + args.top
    fig = plt.figure(figsize=(W, H))
    ax = fig.add_axes([args.left / W, 0.0, args.plot_w / W, args.plot_h / H])

    xs = range(len(R))
    for x, d in zip(xs, R):
        c = C[d["cls"]]
        ax.plot([x, x], [d["ci_lo"], d["ci_hi"]], color=c, lw=1.1, alpha=0.85, zorder=2)
        ax.plot([x], [d["r_adj"]], marker="o", ms=2.8, color=c, ls="", zorder=3)
    ax.axhline(0, color="#555555", lw=0.8, zorder=1)

    lo = min(d["ci_lo"] for d in R)
    hi = max(d["ci_hi"] for d in R)
    pad = (hi - lo) * args.pad_frac
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlim(-0.8, len(R) - 0.2)
    ax.yaxis.set_major_locator(MultipleLocator(args.ytick_step))
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Adjusted point-biserial $r$", fontsize=8)
    ax.tick_params(axis="y", labelsize=7.5, length=2.5, pad=1.5)
    ax.tick_params(axis="x", length=0, pad=1.5)

    ax.set_xticks(list(xs))
    # Non-vertical labels must anchor at their right end or they drift away from their tick.
    slant = args.rotation != 90
    labs = ax.set_xticklabels([d["label"] for d in R],
                              rotation=args.rotation, fontsize=args.label_size,
                              ha=("right" if slant else "center"),
                              rotation_mode=("anchor" if slant else None))
    # Label hue repeats the marker's class, so the figure survives greyscale markers.
    for lab, d in zip(labs, R):
        lab.set_color(C[d["cls"]])
        if d["cls"] != "legit":
            lab.set_fontweight("bold")

    # Class only; `near_universal` goes to the sidecar json, not the figure.
    handles = [Line2D([], [], marker="o", ms=3.2, lw=1.1, color=C[k], label=CLS_LBL[k])
               for k in CLS_ORDER]
    # Upper right is the empty quadrant when sorted descending.
    ax.legend(handles=handles, loc="upper right", fontsize=6.8, frameon=False,
              handlelength=1.3, borderpad=0.1, labelspacing=0.28, borderaxespad=0.3)
    assert ax.lines, "the axes drew NOTHING -- refusing to write a blank figure"
    return fig, (lo - pad, hi + pad)


def main():
    ap = argparse.ArgumentParser(description=(__doc__ or "").strip().split("\n")[0])
    ap.add_argument("--stats", required=True,
                    help="the belief_correlation_stats.json to plot")
    ap.add_argument("--panel", default="qwen4b8b_stub",
                    help="panel key inside all_arms.panels (default: Qwen3-4B+8B x stubborn juror)")
    ap.add_argument("--outdir", default=None,
                    help="where to write the figure (default: the --stats file's directory)")
    ap.add_argument("--name", default=None, help="override the output stem")
    ap.add_argument("--labels", choices=("name", "slug"), default="name",
                    help="x labels: taxonomy display names (default) or raw slugs")
    ap.add_argument("--sort", choices=("r", "class"), default="r",
                    help="x order: descending effect size (default) or grouped by class")
    ap.add_argument("--rotation", type=float, default=60,
                    help="x tick label angle in degrees (90 = vertical, centred under the tick)")
    ap.add_argument("--plot-w", type=float, default=6.9, help="axes width, inches")
    ap.add_argument("--plot-h", type=float, default=1.45,
                    help="axes height, inches -- this is the compressed r axis")
    ap.add_argument("--left", type=float, default=0.42, help="gutter for the y label, inches")
    ap.add_argument("--top", type=float, default=0.06, help="headroom above the axes, inches")
    ap.add_argument("--pad-frac", type=float, default=0.05,
                    help="y headroom as a fraction of the CI span (0 = crop to the data)")
    ap.add_argument("--ytick-step", type=float, default=0.025)
    ap.add_argument("--label-size", type=float, default=6.4, help="x tick label size, points")
    ap.add_argument("--pad", type=float, default=0.01,
                    help="outer margin in inches kept around the cropped figure")
    args = ap.parse_args()

    S, P, cls_of = load(args.stats, args.panel)
    names = {**S["meta"]["names"], **SHORT} if args.labels == "name" else {}
    R, dropped = rows(P, cls_of, names, args)
    if dropped:
        print(f"[warn] {len(dropped)} technique(s) not estimable in {args.panel}, dropped: "
              f"{', '.join(sorted(dropped))}")
    if not R:
        # A grid too thin for MIN_POS (e.g. a smoke) estimates nothing; don't write a blank figure.
        raise SystemExit(
            f"[plot] ERROR: no estimable technique in panel {args.panel!r} of {args.stats} "
            f"({P['n_games']} games, {P['n_cells']} cells) -- nothing to plot.")

    fig, ylim = draw(R, P, args)
    name = args.name or ("fig_strategy_belief_correlation"
                         + ("" if args.panel == "qwen4b8b_stub" else f"_{args.panel}"))
    outdir = Path(args.outdir) if args.outdir else Path(args.stats).resolve().parent
    outdir.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        # No /CreationDate, so an unchanged re-run is byte-identical.
        fig.savefig(outdir / f"{name}.{ext}", dpi=300, bbox_inches="tight", pad_inches=args.pad,
                    metadata=({"CreationDate": None} if ext == "pdf" else {}))
    plt.close(fig)

    doc = {"panel": args.panel, "field": "r_adj",
           "estimator": S["meta"]["estimators"]["r_adj"],
           "outcome": S["meta"]["outcomes"]["final"],
           "n_games": P["n_games"], "n_cells": P["n_cells"], "n_cases": P["n_cases"],
           "n_estimable": P["n_estimable"], "n_plotted": len(R), "not_estimable": sorted(dropped),
           "sort": args.sort, "ylim": [round(v, 4) for v in ylim],
           "near_universal_threshold": NEAR_UNIVERSAL,
           "colors": C, "observational_note": S["meta"]["note"],
           "source_stats": str(Path(args.stats).resolve()),
           "points": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in d.items()}
                      for d in R]}
    json.dump(doc, open(outdir / f"{name}.json", "w"), indent=2)

    print(f"wrote {outdir}/{name}.png / .pdf / .json  ({len(R)} techniques, "
          f"n = {P['n_games']:,} games, y {ylim[0]:+.3f}..{ylim[1]:+.3f})")
    for d in R[:3] + R[-3:]:
        print(f"  {d['label']:<28} {d['cls']:<10} r = {d['r_adj']:+.4f} "
              f"[{d['ci_lo']:+.4f}, {d['ci_hi']:+.4f}]  n_pos = {d['n_pos']}")


if __name__ == "__main__":
    main()
