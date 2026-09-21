"""Exploratory analysis of the KPI matrix X (n x p): mean vector, sample covariance, correlation, and
diagnostics that the multivariate methods depend on (is the sample covariance positive definite?).

Inputs : data/processed/player_match_kpis.csv
Outputs: outputs/tables/{mean_vector_summary,sample_covariance,correlation_matrix,sigma_diagnostics}.csv
         outputs/figures/correlation_heatmap.png, outputs/figures/pca_biplot.png
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG, TAB = ROOT / "outputs" / "figures", ROOT / "outputs" / "tables"
KPIS = ["kd", "hs_pct", "adr", "opening_rate", "utility_pr", "loadout_value"]
LABELS = {"kd": "K/D", "hs_pct": "Headshot share", "adr": "ADR", "opening_rate": "Opening-kill rate",
          "utility_pr": "Utility / round", "loadout_value": "Loadout value"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"
DIVERGING = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])   # blue - neutral - red
SEQUENTIAL = LinearSegmentedColormap.from_list("seq", ["#cde2fb", "#6da7ec", "#256abf", "#0d366b"])


def sigma_diagnostics(S, R):
    """Checks that matter for PCA / MANOVA: positive definiteness and conditioning."""
    try:
        np.linalg.cholesky(S)
        pd_ok = True
    except np.linalg.LinAlgError:
        pd_ok = False
    eig = np.linalg.eigvalsh(R)
    return {"positive_definite (Cholesky succeeds)": pd_ok, "det(S)": np.linalg.det(S), "det(R)": np.linalg.det(R),
            "min eigenvalue of R": eig.min(), "max eigenvalue of R": eig.max(),
            "condition number of R": eig.max() / eig.min(), "rank(S)": int(np.linalg.matrix_rank(S))}


def heatmap(R, path):
    fig, ax = plt.subplots(figsize=(6.4, 5.4), facecolor=SURFACE)
    im = ax.imshow(R.to_numpy(), cmap=DIVERGING, vmin=-1, vmax=1)
    names = [LABELS[k] for k in R.columns]
    ax.set_xticks(range(len(names)), names, rotation=35, ha="right")
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            v = R.iloc[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                    color="white" if abs(v) > 0.6 else INK)
    ax.set_title("Correlation matrix of the six KPIs (standardised covariance)", loc="left", fontsize=11,
                 fontweight="bold")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("Pearson correlation")
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")


def biplot(df, path):
    Z = StandardScaler().fit_transform(df[KPIS])
    pca = PCA().fit(Z)
    scores = pca.transform(Z)[:, :2]
    load = pca.components_[:2].T * np.sqrt(pca.explained_variance_[:2])       # KPI-PC correlations
    ev = pca.explained_variance_ratio_[:2]
    fig, ax = plt.subplots(figsize=(7.4, 6.2), facecolor=SURFACE)
    hb = ax.hexbin(scores[:, 0], scores[:, 1], gridsize=38, cmap=SEQUENTIAL, mincnt=1, bins="log", linewidths=0.2,
                   extent=(-5, 8, -5, 5))
    scale = 3.2
    for (x, y), k in zip(load, KPIS):
        ax.annotate("", xy=(x * scale, y * scale), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6, shrinkA=0, shrinkB=0))
        ax.text(x * scale * 1.09, y * scale * 1.09 + 0.05, LABELS[k], fontsize=9.5, color=INK,
                ha="left" if x >= 0 else "right", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc=SURFACE, ec="none", alpha=0.85))
    ax.axhline(0, color=GRID, lw=0.8, zorder=0)
    ax.axvline(0, color=GRID, lw=0.8, zorder=0)
    ax.set(xlim=(-4.6, 7.4), ylim=(-4.6, 4.6),
           xlabel=f"PC1: aggression / impact ({ev[0]:.1%} of variance)",
           ylabel=f"PC2: precision vs weapon tier ({ev[1]:.1%})")
    ax.set_title("PCA biplot: player-match rows and KPI loading directions", loc="left", fontsize=11,
                 fontweight="bold")
    cb = fig.colorbar(hb, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label("player-match rows per cell (log scale)")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")


def main():
    df = pd.read_csv(PROC / "player_match_kpis.csv")
    X = df[KPIS]
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    print(f"n = {len(X)} observations, p = {X.shape[1]} variables, missing values = {int(X.isna().sum().sum())}")

    summary = pd.DataFrame({"mean (x-bar)": X.mean(), "sd": X.std(), "min": X.min(), "median": X.median(),
                            "max": X.max(), "skewness": X.skew(), "excess kurtosis": X.kurt()})
    S, R = X.cov(), X.corr()
    diag = sigma_diagnostics(S.to_numpy(), R.to_numpy())
    summary.to_csv(TAB / "mean_vector_summary.csv")
    S.to_csv(TAB / "sample_covariance.csv")
    R.to_csv(TAB / "correlation_matrix.csv")
    pd.Series(diag).to_csv(TAB / "sigma_diagnostics.csv", header=["value"])
    print("\nMean vector and summary:\n", summary.round(3).to_string())
    print("\nSample covariance matrix S:\n", S.round(3).to_string())
    print("\nCorrelation matrix R:\n", R.round(2).to_string())
    print("\nSigma diagnostics:")
    for k, v in diag.items():
        print(f"  {k}: {v:.4g}" if isinstance(v, float) else f"  {k}: {v}")

    heatmap(R, FIG / "correlation_heatmap.png")
    biplot(df, FIG / "pca_biplot.png")
    print("\nsaved tables and figures to", ROOT / "outputs")


if __name__ == "__main__":
    main()
