"""Phase 3d: bivariate normal density of each archetype over kill-rate (K/D) vs accuracy (headshot %).

Inputs : data/processed/player_match_kpis.csv, data/processed/cluster_labels_k4.csv
Outputs: outputs/figures/archetype_bivariate_density.png
         outputs/tables/archetype_bivariate_params.csv  (fitted mean vector, covariance, correlation)

Design: 2x2 small multiples (one panel per archetype) with all players as muted grey context. Four
categorical hues fail the all-pairs colour-separation floor (orange vs yellow), so identity is carried by
the panel + direct label rather than by colour alone. Contours are the fitted bivariate normal density at
1 and 2 Mahalanobis SDs (about 39% and 86% of mass if the archetype were truly bivariate normal).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG, TAB = ROOT / "outputs" / "figures", ROOT / "outputs" / "tables"
X, Y = "kd", "hs_pct"
X_LABEL, Y_LABEL = "Kill rate (K/D)", "Accuracy (headshot kill share)"
ORDER = ["Star Fragger", "Utility Support", "Heavy-Weapon Anchor", "Budget Headshotter"]
COLORS = dict(zip(ORDER, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]))  # categorical slots 1-4, fixed order
INK, MUTED, GRID, SURFACE, CONTEXT = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb", "#c3c2b7"
XLIM, YLIM = (0, 4), (0, 1)


def fit_gaussians(df, group="archetype"):
    """Bivariate normal (mean, covariance) of (X, Y) per archetype."""
    return {g: (d[[X, Y]].mean().to_numpy(), np.cov(d[[X, Y]].to_numpy(), rowvar=False))
            for g, d in df.groupby(group)}


def plot_density(df, gauss, path=None, team=None):
    """team (optional): DataFrame with columns kd, hs_pct, archetype, label; drawn as black markers."""
    plt.rcParams.update({"font.family": "sans-serif", "text.color": INK, "axes.labelcolor": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID})
    gx, gy = np.meshgrid(np.linspace(*XLIM, 300), np.linspace(*YLIM, 300))
    grid = np.dstack([gx, gy])
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True, facecolor=SURFACE)
    for ax, name in zip(axes.ravel(), ORDER):
        mu, cov = gauss[name]
        mvn = multivariate_normal(mu, cov)
        d = df[df.archetype == name]
        ax.set_facecolor(SURFACE)
        ax.scatter(df[X], df[Y], s=4, color=CONTEXT, alpha=0.25, linewidths=0, rasterized=True)
        ax.scatter(d[X], d[Y], s=6, color=COLORS[name], alpha=0.35, linewidths=0, rasterized=True)
        pdf = mvn.pdf(grid)
        lv = [mvn.pdf(mu) * np.exp(-0.5 * c ** 2) for c in (2, 1)]
        ax.contourf(gx, gy, pdf, levels=[lv[0], lv[1]], colors=[COLORS[name]], alpha=0.20)
        ax.contourf(gx, gy, pdf, levels=[lv[1], pdf.max() * 1.001], colors=[COLORS[name]], alpha=0.35)
        ax.contour(gx, gy, pdf, levels=lv, colors=[COLORS[name]], linewidths=1.5)
        ax.plot(*mu, marker="o", ms=8, mfc=COLORS[name], mec=SURFACE, mew=2, ls="none")
        if team is not None:
            tp = team[team.archetype == name]
            ax.scatter(tp[X], tp[Y], s=55, color=INK, edgecolors=SURFACE, linewidths=1.5, zorder=5)
            for j, (_, r) in enumerate(tp.iterrows()):
                ax.annotate(r["label"], (r[X], r[Y]), xytext=(6, 6 if j % 2 == 0 else -13),
                            textcoords="offset points", fontsize=8.5, color=INK, zorder=6)
        r = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        ax.set_title(f"{name}  (n={len(d):,}, {len(d) / len(df):.0%})", loc="left", fontsize=11,
                     fontweight="bold", color=INK, pad=17)
        ax.text(0, 1.012, f"mean K/D {mu[0]:.2f}, HS {mu[1]:.0%}, r = {r:.2f}", transform=ax.transAxes,
                ha="left", va="bottom", fontsize=8.5, color=MUTED)
        ax.set(xlim=XLIM, ylim=YLIM)
        ax.grid(color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    for ax in axes[1]:
        ax.set_xlabel(X_LABEL)
    for ax in axes[:, 0]:
        ax.set_ylabel(Y_LABEL)
    n_clip = int((df[X] > XLIM[1]).sum())
    fig.suptitle("Player archetypes: fitted bivariate normal density in kill-rate vs accuracy space",
                 x=0.06, ha="left", fontsize=13, fontweight="bold")
    note = (f"Contours: 1 and 2 SD of each archetype's fitted bivariate normal. Grey = all players. "
            f"{n_clip} players with K/D > {XLIM[1]} not shown.")
    if team is not None:
        note += "\nBlack markers = your team's players."
    fig.text(0.06, 0.005, note, fontsize=8.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.035 if team is not None else 0.02, 1, 0.96))
    if path:
        fig.savefig(path, dpi=150, facecolor=SURFACE)
    return fig


def main():
    kpi = pd.read_csv(PROC / "player_match_kpis.csv")
    lab = pd.read_csv(PROC / "cluster_labels_k4.csv")
    df = kpi.merge(lab[["file", "player_id", "archetype"]], on=["file", "player_id"], how="inner")
    assert len(df) == len(kpi)
    gauss = fit_gaussians(df)
    rows = [{"archetype": g, "mean_kd": m[0], "mean_hs": m[1], "var_kd": c[0, 0], "var_hs": c[1, 1],
             "cov": c[0, 1], "corr": c[0, 1] / np.sqrt(c[0, 0] * c[1, 1])} for g, (m, c) in gauss.items()]
    params = pd.DataFrame(rows).set_index("archetype").loc[ORDER]
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    params.to_csv(TAB / "archetype_bivariate_params.csv")
    print(params.round(4).to_string())
    plot_density(df, gauss, FIG / "archetype_bivariate_density.png")
    print("saved archetype_bivariate_density.png")


if __name__ == "__main__":
    main()
