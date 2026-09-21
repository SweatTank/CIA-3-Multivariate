"""Phase 3a: PCA on the six approved KPIs (z-scored, i.e. PCA of the correlation matrix).

Inputs : data/processed/player_match_kpis.csv (all rows, ties included)
Outputs: data/processed/pca_scores.csv    (all 6 PC scores per player-match row)
         outputs/tables/pca_loadings.csv  (loadings, eigenvalues, explained variance)
         outputs/figures/scree_plot.png
The number of components to retain is NOT decided here; this script reports the evidence for it.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG, TAB = ROOT / "outputs" / "figures", ROOT / "outputs" / "tables"
KPIS = ["kd", "hs_pct", "adr", "opening_rate", "utility_pr", "loadout_value"]


def bartlett_sphericity(X):
    n, p = X.shape
    R = np.corrcoef(X, rowvar=False)
    chi2 = -(n - 1 - (2 * p + 5) / 6) * np.log(np.linalg.det(R))
    df = p * (p - 1) / 2
    return chi2, df, stats.chi2.sf(chi2, df)


def kmo(X):
    """Kaiser-Meyer-Olkin measure of sampling adequacy (overall and per variable)."""
    R = np.corrcoef(X, rowvar=False)
    Rinv = np.linalg.inv(R)
    d = np.sqrt(np.outer(np.diag(Rinv), np.diag(Rinv)))
    partial = -Rinv / d
    np.fill_diagonal(R, 0)
    np.fill_diagonal(partial, 0)
    r2, p2 = (R ** 2), (partial ** 2)
    return r2.sum() / (r2.sum() + p2.sum()), r2.sum(0) / (r2.sum(0) + p2.sum(0))


def main():
    df = pd.read_csv(PROC / "player_match_kpis.csv")
    X = StandardScaler().fit_transform(df[KPIS])
    print(f"rows: {len(df)}   KPIs: {KPIS}")
    print("skewness:", df[KPIS].skew().round(2).to_dict())

    chi2, dof, p = bartlett_sphericity(X)
    k_all, k_var = kmo(X)
    print(f"\nBartlett sphericity: chi2={chi2:.1f}, df={dof:.0f}, p={p:.3g}")
    print(f"KMO overall = {k_all:.3f}; per variable: {dict(zip(KPIS, k_var.round(3)))}")

    pca = PCA().fit(X)
    eig = pca.explained_variance_
    evr = pca.explained_variance_ratio_
    names = [f"PC{i + 1}" for i in range(len(KPIS))]
    summary = pd.DataFrame({"eigenvalue": eig, "explained_var": evr, "cumulative": evr.cumsum()}, index=names)
    print("\n", summary.round(3))

    loadings = pd.DataFrame(pca.components_.T * np.sqrt(eig), index=KPIS, columns=names)
    print("\nLoadings (eigenvector * sqrt(eigenvalue) = correlation of KPI with PC):\n", loadings.round(2))

    # Parallel analysis: eigenvalues of the correlation matrix of random normal data of the same shape.
    rng = np.random.default_rng(42)
    sims = np.array([
        np.sort(np.linalg.eigvalsh(np.corrcoef(rng.standard_normal(X.shape), rowvar=False)))[::-1]
        for _ in range(100)
    ])
    pa95 = np.percentile(sims, 95, axis=0)
    print("\nRetention evidence:")
    print("  Kaiser (eigenvalue > 1):", int((eig > 1).sum()))
    print("  Parallel analysis (eig > 95th pct of random):", int((eig > pa95).sum()))
    print("  Components needed for 70% / 80% / 90% variance:",
          [int(np.searchsorted(summary.cumulative.values, t) + 1) for t in (0.7, 0.8, 0.9)])

    scores = pd.DataFrame(pca.transform(X), columns=names)
    out = pd.concat([df[["file", "player_id"]].reset_index(drop=True), scores], axis=1)
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    out.to_csv(PROC / "pca_scores.csv", index=False)
    pd.concat([summary.T, loadings]).to_csv(TAB / "pca_loadings.csv")

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(1, len(eig) + 1)
    ax.plot(x, eig, "o-", label="Observed eigenvalue")
    ax.plot(x, pa95, "s--", color="gray", label="Parallel analysis (95th pct)")
    ax.axhline(1, color="red", lw=0.8, ls=":", label="Kaiser (=1)")
    ax.set(xlabel="Principal component", ylabel="Eigenvalue", title="Scree plot", xticks=x)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "scree_plot.png", dpi=150)
    print(f"\nsaved scores to {PROC}; figure and table to {ROOT / 'outputs'}")


if __name__ == "__main__":
    main()
