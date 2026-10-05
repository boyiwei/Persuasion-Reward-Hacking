#!/usr/bin/env python3
"""Paper figure: 1 x 4 row of square endorsement heatmaps.

    [ Qwen3-4B direct-ask ] [ Qwen3-4B in-role ] | [ Qwen3-8B direct-ask ] [ Qwen3-8B in-role ]

Each panel is 3x3: y = item origin (the checkpoint that fabricated the claim), x = prober (the
checkpoint answering); value = share of those fabrications the prober endorses as true. Down a
column (policy fixed): the detectability effect; across a row (evidence fixed): the behavioural.

Direct-ask is the out-of-context readout (fresh conversation, neutral analyst, no transcript or
role); in-role is the mid-game readout with the judge demanding an answer for the court record.
The gap between a model's two panels is assertion without belief.

Re-plots analyze.py's stats JSON (`cells[*].hallucination_rate`); nothing is recomputed. One
sequential single-hue ramp with shared vmin/vmax; every cell is labelled, so colour is never needed;
panels are placed in absolute inches so each is exactly square.

Usage
    python evaluation/source_of_fabrication/plot_heatmap.py --stats $SOF_ROOT/stats.json \
        --outdir $SOF_ROOT/figures                      # -> <name>.{png,pdf} + .json
    ... --show-n            # add n= under each percentage
    ... --tick-style ckpt   # base/gs50/gs100 instead of 0/50/100
    ... --diagonal          # outline the own-items diagonal
"""
import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["pdf.fonttype"] = 42  # embed TrueType so the PDF carries real Arial
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

CKPTS = ["base", "gs50", "gs100"]
STEP = {"base": "0", "gs50": "50", "gs100": "100"}

# (size, arm) in the order the panels appear, left to right.
PANELS = [("4B", "out"), ("4B", "in_pre_role"), ("8B", "out"), ("8B", "in_pre_role")]
ARM_TITLE = {"out": "Direct Ask", "in_pre_role": "In-Role"}

INK, MUTED = "#222222", "#666666"
CMAP = "Oranges"          # sequential, single hue, light -> dark (magnitude)


def cell(S, size, prober, origin, arm):
    for c in S["cells"]:
        if (c["size"] == size and c["prober_ckpt"] == prober
                and c["origin_ckpt"] == origin and c["arm"] == arm):
            return c
    raise KeyError((size, prober, origin, arm))


def draw_panel(ax, S, size, arm, vmax, show_n, tick_style, diagonal, yticks):
    # rows = origin (y), cols = prober (x)
    rate = [[cell(S, size, p, o, arm)["hallucination_rate"] for p in CKPTS] for o in CKPTS]
    nobs = [[cell(S, size, p, o, arm)["n_classified"] for p in CKPTS] for o in CKPTS]

    ax.imshow(rate, cmap=CMAP, vmin=0.0, vmax=vmax, aspect="equal",
              interpolation="nearest")

    for oi in range(3):
        for pi in range(3):
            v = rate[oi][pi]
            # white ink only on genuinely dark fill; dark ink reads better on mid-ramp orange
            col = "white" if v / vmax > 0.68 else INK
            if show_n:
                ax.text(pi, oi - 0.13, f"{100 * v:.0f}%", ha="center", va="center",
                        fontsize=9, color=col)
                ax.text(pi, oi + 0.2, f"n={nobs[oi][pi]}", ha="center", va="center",
                        fontsize=6.2, color=col, alpha=0.85)
            else:
                ax.text(pi, oi, f"{100 * v:.0f}%", ha="center", va="center",
                        fontsize=10, color=col)
            if diagonal and pi == oi:
                ax.add_patch(Rectangle((pi - .5, oi - .5), 1, 1, fill=False, lw=1.4,
                                       ec="#0072B2", zorder=5))

    labels = [STEP[c] for c in CKPTS] if tick_style == "step" else CKPTS
    ax.set_xticks(range(3))
    ax.set_xticklabels(labels, fontsize=12, color=MUTED)
    ax.set_yticks(range(3))
    ax.set_yticklabels(labels if yticks else [""] * 3, fontsize=12, color=MUTED)
    ax.tick_params(length=0, pad=2)
    for s in ax.spines.values():                       # recessive frame
        s.set_color("#DDDDDD")
        s.set_linewidth(0.6)
    return rate, nobs


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n", 1)[0])
    ap.add_argument("--stats", required=True, help="the stats JSON analyze.py wrote")
    ap.add_argument("--outdir", required=True, help="directory the figure files are written to")
    ap.add_argument("--name", default="fig_endorsement_grid_bal100",
                    help="output file stem under --outdir")
    ap.add_argument("--vmax", type=float, default=0.70,
                    help="top of the shared colour scale (fraction, not %%)")
    ap.add_argument("--show-n", action="store_true", help="print n= under each percentage")
    ap.add_argument("--tick-style", choices=("step", "ckpt"), default="step")
    ap.add_argument("--diagonal", action="store_true",
                    help="outline the prober==origin diagonal (the own-items configuration)")
    ap.add_argument("--panel-in", type=float, default=1.62, help="panel side length, inches")
    args = ap.parse_args()

    S = json.load(open(args.stats))
    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # Absolute-inch layout, so every panel is exactly square
    pw = ph = args.panel_in
    gap_in, gap_between = 0.08, 0.34          # within a model pair / between the two models
    left, right = 0.58, 0.72                  # y ticks + axis / colorbar + its label
    top, bottom = 0.42, 0.35                  # stacked titles / x ticks + axis
    fig_w = left + 4 * pw + 2 * gap_in + gap_between + right
    fig_h = bottom + ph + top

    fig = plt.figure(figsize=(fig_w, fig_h))
    xs, axes = [], []
    x = left
    for i, (size, arm) in enumerate(PANELS):
        if i > 0:
            x += gap_between if i == 2 else gap_in
        xs.append(x)
        axes.append(fig.add_axes([x / fig_w, bottom / fig_h, pw / fig_w, ph / fig_h]))
        x += pw

    # Record the source by name; an absolute path is machine-specific.
    stats_label = Path(args.stats).name
    out = {"source_stats": stats_label, "field": "hallucination_rate",
           "vmax": args.vmax, "panels": []}
    for ax, (size, arm) in zip(axes, PANELS):
        ytick = arm == "out"                   # left panel of each model block carries the y ticks
        rate, nobs = draw_panel(ax, S, size, arm, args.vmax, args.show_n,
                                args.tick_style, args.diagonal, ytick)
        ax.text(0.5, 1.01, ARM_TITLE[arm], transform=ax.transAxes, ha="center", va="bottom",
                fontsize=12, color=INK)
        out["panels"].append({
            "size": size, "arm": arm, "rows_origin": CKPTS, "cols_prober": CKPTS,
            "rate": rate, "n_classified": nobs,
            "row_means": [round(sum(r) / 3, 4) for r in rate],
            "col_means": [round(sum(rate[p][o] for p in range(3)) / 3, 4) for o in range(3)],
        })

    # model banners sit immediately above the arm titles
    for (i, j), size in (((0, 1), "4B"), ((2, 3), "8B")):
        cx = (xs[i] + xs[j] + pw) / 2 / fig_w
        fig.text(cx, (bottom + ph + 0.22) / fig_h, f"Qwen3-{size}", ha="center", va="bottom",
                 fontsize=12, color=INK)

    # shared axis labels: one per figure, not repeated under every panel
    fig.text((left + (4 * pw + 2 * gap_in + gap_between) / 2) / fig_w, 0.01 / fig_h,
             "Probe Step", ha="center", va="bottom",
             fontsize=12, color=INK)
    fig.text(0.2 / fig_w, (bottom + ph / 2) / fig_h,
             "Generation Step", rotation=90, ha="center", va="center",
             fontsize=12, color=INK)

    # colorbar (sequential ramp, shared by all four panels)
    cax = fig.add_axes([(left + 4 * pw + 2 * gap_in + gap_between + 0.16) / fig_w,
                        bottom / fig_h, 0.08 / fig_w, ph / fig_h])
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 100 * args.vmax), cmap=CMAP), cax=cax)
    cb.set_label("% judged as true evidence", fontsize=8, color=INK, labelpad=2)
    cb.ax.tick_params(labelsize=7, colors=MUTED, length=2)
    cb.outline.set_visible(False)

    for ext in ("png", "pdf"):
        # Suppress the PDF CreationDate so re-running an unchanged figure is byte-identical.
        meta = {"CreationDate": None} if ext == "pdf" else {}
        fig.savefig(Path(args.outdir) / f"{args.name}.{ext}", dpi=300, pad_inches=0,
                    metadata=meta)
    plt.close(fig)
    json.dump(out, open(Path(args.outdir) / f"{args.name}.json", "w"), indent=2)
    print(f"wrote {args.name}.png / .pdf / .json  ({fig_w:.2f} x {fig_h:.2f} in)")
    for p in out["panels"]:
        rm = " / ".join(f"{100 * v:.1f}%" for v in p["row_means"])
        n = p["n_classified"]
        print(f"  Qwen3-{p['size']:<3} {ARM_TITLE[p['arm']]:<11} row means (base/gs50/gs100): {rm}"
              f"   n per column: {n[0][0]}/{n[0][1]}/{n[0][2]}")


if __name__ == "__main__":
    main()
