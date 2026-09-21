"""End-to-end methodology flowchart (rubric B1): ingestion -> preprocessing -> model fitting -> inference -> output.

Decision nodes are diamonds; bold blue arrows show the path actually taken in this project, grey arrows the
alternative branches. Outputs: outputs/figures/flowchart.png and outputs/figures/flowchart.pdf
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "outputs" / "figures"
INK, MUTED, SURFACE = "#0b0b0b", "#898781", "#fcfcfb"
TAKEN, ALT = "#1c5cab", "#898781"
DEC_FILL, PROC_FILL, ALT_FILL = "#cde2fb", "#ffffff", "#f0efec"
CX, SIDE_L, SIDE_R = 10.0, 2.7, 17.3
Y = [26.2 - 2.2 * i - (1.2 if i >= 9 else 0) for i in range(12)]   # extra gap below the MVN decision
FS = 8.2


def box(ax, cx, cy, w, h, text, fill=PROC_FILL, edge=INK, bold_first=True):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.25",
                                fc=fill, ec=edge, lw=1.1, zorder=2))
    lines = text.split("\n")
    ax.text(cx, cy, "\n".join(lines), ha="center", va="center", fontsize=FS, color=INK, zorder=3,
            linespacing=1.35)


def diamond(ax, cx, cy, w, h, text):
    ax.add_patch(Polygon([(cx, cy + h / 2), (cx + w / 2, cy), (cx, cy - h / 2), (cx - w / 2, cy)],
                         closed=True, fc=DEC_FILL, ec="#256abf", lw=1.3, zorder=2))
    ax.text(cx, cy, text, ha="center", va="center", fontsize=FS, color=INK, zorder=3, linespacing=1.3,
            fontweight="bold")


def arrow(ax, pts, taken=True, label=None, label_at=None):
    """Polyline through pts, arrowhead on the last segment."""
    color, lw = (TAKEN, 2.0) if taken else (ALT, 1.1)
    for a, b in zip(pts[:-2], pts[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=lw, zorder=1, solid_capstyle="round")
    ax.annotate("", xy=pts[-1], xytext=pts[-2],
                arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=0, shrinkB=0), zorder=1)
    if label:
        x, y = label_at
        ax.text(x, y, label, fontsize=8, color=color, fontweight="bold", ha="center", va="bottom", zorder=4)


def main():
    fig, ax = plt.subplots(figsize=(8.5, 11.6), facecolor=SURFACE)
    ax.set(xlim=(-0.4, 20.4), ylim=(-1.3, 28.3))
    ax.axis("off")
    W, H, DW, DH, SW, SH = 9.8, 1.5, 6.6, 2.0, 5.8, 1.4

    # main column
    box(ax, CX, Y[0], W, H, "1. Data ingestion and event inference\nKaggle mm_master + mm_grenades demo logs\n"
                            "death = victim's cumulative damage first >= 100 HP")
    box(ax, CX, Y[1], W, H, "2. KPI matrix  $X\\in\\mathbb{R}^{n\\times p}$, p = 6\n"
                            "K/D, headshot share, ADR, opening-kill rate,\nutility/round, loadout value (rounds $\\geq$ 10)")
    diamond(ax, CX, Y[2], DW, DH, "Missing values\nin X?")
    box(ax, CX, Y[3], W, H, "3. Standardise  $Z=(X-\\bar{x})/s$\nEDA: mean vector $\\bar{x}$,\n"
                            "sample covariance S, correlation R")
    diamond(ax, CX, Y[4], DW, DH, "Is S positive definite?\n(Cholesky, min eig. > 0)")
    box(ax, CX, Y[5], W, H, "4. PCA on R, retain q components\nKaiser, parallel analysis, interpretability\n"
                            "q = 3 (75.4% of variance)")
    box(ax, CX, Y[6], W, H, "5. K-Means on PC scores, k = 2 ... 8\nsilhouette, Davies-Bouldin, bootstrap ARI\n"
                            "k = 4 archetypes")
    box(ax, CX, Y[7], W, H, "6. Team level (ties excluded)\nmean KPI vector per (match, team)\n"
                            "winner / loser by rounds won (1,165 pairs)")
    diamond(ax, CX, Y[8], DW, DH, "MVN (Mardia) and\nBox's M hold?")
    diamond(ax, CX, Y[9], DW, DH, "Wilks' $\\Lambda$ significant?\n(permutation p < 0.05)")
    box(ax, CX, Y[10], W, H, "7. Follow-up and win model\nper-KPI paired tests (Bonferroni)\n"
                             "logistic win probability, entropy balance score", )
    box(ax, CX, Y[11], W, H, "8. Interpretation and dashboard\narchetype labels, team balance score,\n"
                             "win probability (Streamlit app)")

    # side boxes
    box(ax, SIDE_R, Y[2], SW, SH, "Drop those rows\n(3 rows: no gun damage)", fill=ALT_FILL, edge=MUTED)
    box(ax, SIDE_R, Y[4], SW, SH, "Drop collinear KPI or\nuse shrinkage covariance", fill=ALT_FILL, edge=MUTED)
    box(ax, SIDE_L, Y[8], SW, 1.7, "Classical MANOVA\nindependent groups,\nWilks' $\\Lambda$ and F", fill=ALT_FILL, edge=MUTED)
    box(ax, SIDE_R, Y[8], SW, 1.7, "Paired within-match test:\nHotelling $T^2$ on winner-loser\ndifferences + permutation p",
        fill=PROC_FILL, edge=TAKEN)
    box(ax, SIDE_R, Y[9], SW, SH, "Report: no detectable\nKPI-vector difference", fill=ALT_FILL, edge=MUTED)

    top = lambda y, h: y + h / 2
    bot = lambda y, h: y - h / 2
    # main-column arrows (taken)
    arrow(ax, [(CX, bot(Y[0], H)), (CX, top(Y[1], H))])
    arrow(ax, [(CX, bot(Y[1], H)), (CX, top(Y[2], DH))])
    arrow(ax, [(CX, bot(Y[2], DH)), (CX, top(Y[3], H))], label="No", label_at=(CX + 0.5, bot(Y[2], DH) - 0.42))
    arrow(ax, [(CX, bot(Y[3], H)), (CX, top(Y[4], DH))])
    arrow(ax, [(CX, bot(Y[4], DH)), (CX, top(Y[5], H))], label="Yes", label_at=(CX + 0.6, bot(Y[4], DH) - 0.42))
    arrow(ax, [(CX, bot(Y[5], H)), (CX, top(Y[6], H))])
    arrow(ax, [(CX, bot(Y[6], H)), (CX, top(Y[7], H))])
    arrow(ax, [(CX, bot(Y[7], H)), (CX, top(Y[8], DH))])
    arrow(ax, [(CX, bot(Y[9], DH)), (CX, top(Y[10], H))], label="Yes", label_at=(CX + 0.6, bot(Y[9], DH) - 0.42))
    arrow(ax, [(CX, bot(Y[10], H)), (CX, top(Y[11], H))])

    # branch arrows
    xr = CX + DW / 2
    xl = CX - DW / 2
    arrow(ax, [(xr, Y[2]), (SIDE_R - SW / 2, Y[2])], taken=False, label="Yes", label_at=((xr + SIDE_R - SW / 2) / 2, Y[2] + 0.1))
    arrow(ax, [(SIDE_R, bot(Y[2], SH)), (SIDE_R, Y[3]), (CX + W / 2, Y[3])], taken=False)
    arrow(ax, [(xr, Y[4]), (SIDE_R - SW / 2, Y[4])], taken=False, label="No", label_at=((xr + SIDE_R - SW / 2) / 2, Y[4] + 0.1))
    arrow(ax, [(SIDE_R, bot(Y[4], SH)), (SIDE_R, Y[5]), (CX + W / 2, Y[5])], taken=False)
    # D4 branches
    arrow(ax, [(xl, Y[8]), (SIDE_L + SW / 2, Y[8])], taken=False, label="Yes", label_at=((xl + SIDE_L + SW / 2) / 2, Y[8] + 0.1))
    arrow(ax, [(xr, Y[8]), (SIDE_R - SW / 2, Y[8])], taken=True, label="No", label_at=((xr + SIDE_R - SW / 2) / 2, Y[8] + 0.1))
    merge_y = top(Y[9], DH) + 0.55
    arrow(ax, [(SIDE_L, bot(Y[8], 1.7)), (SIDE_L, merge_y), (CX - 0.0, merge_y)], taken=False)
    arrow(ax, [(SIDE_R, bot(Y[8], 1.7)), (SIDE_R, merge_y), (CX, merge_y), (CX, top(Y[9], DH))], taken=True)
    # D5 No branch
    arrow(ax, [(xr, Y[9]), (SIDE_R - SW / 2, Y[9])], taken=False, label="No", label_at=((xr + SIDE_R - SW / 2) / 2, Y[9] + 0.1))

    # legend
    ax.plot([0.6, 2.2], [-0.6, -0.6], color=TAKEN, lw=2.0)
    ax.text(2.5, -0.6, "path taken in this project", fontsize=8, va="center", color=INK)
    ax.plot([9.2, 10.8], [-0.6, -0.6], color=ALT, lw=1.1)
    ax.text(11.1, -0.6, "alternative branch (not needed)", fontsize=8, va="center", color=INK)
    ax.text(0.4, 28.1, "Methodology flowchart: CS:GO player profiling and win prediction", fontsize=10.5,
            fontweight="bold", va="top", color=INK)

    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "flowchart.png", dpi=200, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(FIG / "flowchart.pdf", facecolor=SURFACE, bbox_inches="tight")
    print("saved flowchart.png / flowchart.pdf")


if __name__ == "__main__":
    main()
