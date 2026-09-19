"""Phase 3c: team-level MANOVA (Wilks' Lambda), winners vs losers on the mean KPI vector.

Unit: one row per (match, team) = mean of the six KPIs over that team's players. Ties are excluded
(approved decision: option A). Two modes:
  python manova.py check   -> assumption checks only (Mardia MVN, Box's M, outliers, match dependence)
  python manova.py run     -> approved plan: paired within-match Wilks' Lambda (primary, with sign-flip
                              permutation p-value) + spec-style two-group MANOVA (secondary, caveated)
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import PowerTransformer

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
KPIS = ["kd", "hs_pct", "adr", "opening_rate", "utility_pr", "loadout_value"]


def team_table():
    df = pd.read_csv(PROC / "player_match_kpis.csv")
    df = df[df.team_won.notna()]
    t = df.groupby(["file", "team"]).agg(**{k: (k, "mean") for k in KPIS},
                                         team_won=("team_won", "first"), n_players=("player_id", "nunique"))
    return t.reset_index()


def mardia(X):
    """Mardia's multivariate skewness and kurtosis tests. Returns (b1, skew_p, b2, kurt_z, kurt_p)."""
    n, p = X.shape
    Xc = X - X.mean(0)
    Sinv = np.linalg.inv(np.cov(Xc, rowvar=False, bias=True))
    D = Xc @ Sinv @ Xc.T
    b1 = (D ** 3).sum() / n ** 2
    skew_stat = n * b1 / 6
    skew_p = stats.chi2.sf(skew_stat, p * (p + 1) * (p + 2) / 6)
    b2 = (np.diag(D) ** 2).mean()
    z = (b2 - p * (p + 2)) / np.sqrt(8 * p * (p + 2) / n)
    return b1, skew_p, b2, z, 2 * stats.norm.sf(abs(z))


def box_m(groups):
    """Box's M test for equality of covariance matrices (chi-square approximation)."""
    g, p = len(groups), groups[0].shape[1]
    ns = np.array([len(x) for x in groups])
    covs = [np.cov(x, rowvar=False) for x in groups]
    N = ns.sum()
    Sp = sum((n - 1) * S for n, S in zip(ns, covs)) / (N - g)
    M = (N - g) * np.linalg.slogdet(Sp)[1] - sum((n - 1) * np.linalg.slogdet(S)[1] for n, S in zip(ns, covs))
    c = (np.sum(1 / (ns - 1)) - 1 / (N - g)) * (2 * p ** 2 + 3 * p - 1) / (6 * (p + 1) * (g - 1))
    chi2 = M * (1 - c)
    df = p * (p + 1) * (g - 1) / 2
    return M, chi2, df, stats.chi2.sf(chi2, df)


def report(label, X, y):
    print(f"\n===== {label} =====")
    p = X.shape[1]
    groups = [X[y == v] for v in (1.0, 0.0)]
    for name, G in zip(("winners", "losers"), groups):
        b1, sp, b2, z, kp = mardia(G)
        print(f"Mardia [{name}, n={len(G)}]: skew b1={b1:.3f} (p={sp:.3g}); "
              f"kurtosis b2={b2:.2f} vs expected {p * (p + 2)} (z={z:.2f}, p={kp:.3g})")
    M, chi2, df, pv = box_m(groups)
    print(f"Box's M = {M:.1f}, chi2 = {chi2:.1f}, df = {df:.0f}, p = {pv:.3g}")
    d2 = np.concatenate([((G - G.mean(0)) @ np.linalg.inv(np.cov(G, rowvar=False)) * (G - G.mean(0))).sum(1)
                         for G in groups])
    crit = stats.chi2.ppf(0.999, p)
    print(f"Mahalanobis outliers (> chi2 99.9% = {crit:.1f}): {(d2 > crit).sum()} of {len(d2)}")
    return groups


def qq_plot(X, y, path, title):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    for ax, v, name in zip(axes, (1.0, 0.0), ("Winners", "Losers")):
        G = X[y == v]
        d2 = np.sort(((G - G.mean(0)) @ np.linalg.inv(np.cov(G, rowvar=False)) * (G - G.mean(0))).sum(1))
        q = stats.chi2.ppf((np.arange(1, len(d2) + 1) - 0.5) / len(d2), G.shape[1])
        ax.scatter(q, d2, s=6)
        ax.plot([0, q.max()], [0, q.max()], "r--", lw=1)
        ax.set(title=f"{name}: Mahalanobis d² vs chi2({G.shape[1]})", xlabel="chi2 quantile", ylabel="d²")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)


def check():
    t = team_table()
    y = t.team_won.to_numpy()
    X = t[KPIS].to_numpy()
    print(f"team-match rows: {len(t)} (winners {int((y == 1).sum())}, losers {int((y == 0).sum())}); "
          f"team sizes: {t.n_players.value_counts().sort_index().to_dict()}")
    print("univariate skew:", t[KPIS].skew().round(2).to_dict())
    print("univariate excess kurtosis:", t[KPIS].kurt().round(2).to_dict())

    report("RAW team-mean KPIs", X, y)
    qq_plot(X, y, PROC / "mvn_qq_raw.png", "Raw team-mean KPIs")

    Xt = PowerTransformer(method="yeo-johnson").fit_transform(X)
    report("YEO-JOHNSON transformed team-mean KPIs (option, for comparison)", Xt, y)
    qq_plot(Xt, y, PROC / "mvn_qq_yeojohnson.png", "Yeo-Johnson transformed team-mean KPIs")

    # Independence: winner and loser of the same match are not independent observations.
    w = t[t.team_won == 1].set_index("file")[KPIS]
    l = t[t.team_won == 0].set_index("file")[KPIS]
    both = w.index.intersection(l.index)
    print(f"\nMatches with both a winner and loser row: {len(both)}")
    print("Winner-vs-loser within-match correlation per KPI:",
          {k: round(float(np.corrcoef(w.loc[both, k], l.loc[both, k])[0, 1]), 2) for k in KPIS})


def paired_manova(t, n_perm=20000, seed=42):
    """Primary test (approved): one-sample Hotelling T2 on within-match (winner - loser) KPI differences.
    Wilks' Lambda = 1 / (1 + T2/(n-1)); p-value from a sign-flip permutation test (distribution-free,
    valid because winner/loser labels are exchangeable within a match under H0)."""
    w = t[t.team_won == 1].set_index("file")[KPIS]
    l = t[t.team_won == 0].set_index("file")[KPIS]
    idx = w.index.intersection(l.index)
    D = (w.loc[idx] - l.loc[idx]).to_numpy()
    n, p = D.shape
    m, DtD = D.mean(0), D.T @ D
    T2 = n * m @ np.linalg.solve(np.cov(D, rowvar=False), m)
    lam = 1 / (1 + T2 / (n - 1))
    F = (n - p) / (p * (n - 1)) * T2
    p_asym = stats.f.sf(F, p, n - p)

    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, n))
    mb = signs @ D / n                                             # (B, p)
    Sb = (DtD - n * mb[:, :, None] * mb[:, None, :]) / (n - 1)     # sign flips leave D'D unchanged
    T2b = n * np.einsum("bi,bi->b", mb, np.linalg.solve(Sb, mb[:, :, None])[:, :, 0])
    p_perm = (1 + (T2b >= T2).sum()) / (1 + n_perm)

    print("\n=== PRIMARY: paired within-match MANOVA (Wilks' Lambda) ===")
    print(f"pairs n={n}, p={p}; Hotelling T2={T2:.1f}; Wilks' Lambda={lam:.4f}; F({p},{n - p})={F:.1f}; "
          f"asymptotic p={p_asym:.3g}")
    print(f"sign-flip permutation p (B={n_perm}) = {p_perm:.2g}  (smallest possible {1 / (1 + n_perm):.2g}); "
          f"max permuted T2 = {T2b.max():.1f}")
    print(f"effect size: partial eta^2 = 1 - Lambda = {1 - lam:.3f}")

    sd = D.std(0, ddof=1)
    t_stat = m / (sd / np.sqrt(n))
    p_uni = 2 * stats.t.sf(np.abs(t_stat), n - 1)
    uni = pd.DataFrame({"mean_diff": m, "sd_diff": sd, "cohens_dz": m / sd, "t": t_stat,
                        "p_bonf": np.minimum(1, p_uni * p)}, index=KPIS)
    # Standardised discriminant weights: which KPIs carry the winner-loser separation (jointly).
    disc = np.linalg.solve(np.cov(D, rowvar=False), m) * sd
    uni["std_discriminant_weight"] = disc / np.abs(disc).sum()
    print("\nPer-KPI follow-up (Bonferroni over 6 KPIs) and standardised discriminant weights:")
    print(uni.round(4).to_string())
    uni.to_csv(PROC / "manova_paired_followup.csv")
    return lam


def twogroup_manova(t):
    """Secondary (spec-style): independent two-group MANOVA. Violations documented in the README."""
    from statsmodels.multivariate.manova import MANOVA
    d = t.rename(columns={"team_won": "won"}).assign(won=lambda x: x.won.astype(int))
    res = MANOVA.from_formula(" + ".join(KPIS) + " ~ won", data=d).mv_test()
    stat = res.results["won"]["stat"]
    print("\n=== SECONDARY: spec-style two-group MANOVA (independent groups) ===")
    print(stat.round(6).to_string())
    lam = float(stat.loc["Wilks' lambda", "Value"])
    print(f"Wilks' Lambda = {lam:.4f}; partial eta^2 = 1 - Lambda = {1 - lam:.3f}")
    print("Caveat: winner/loser rows come from the same match (dependent), and MVN and Box's M fail; "
          "treat this p-value as descriptive only.")


def run():
    t = team_table()
    paired_manova(t)
    twogroup_manova(t)


if __name__ == "__main__":
    {"check": check, "run": run}[sys.argv[1] if len(sys.argv) > 1 else "check"]()
