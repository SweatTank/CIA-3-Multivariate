"""Phase 4: fitted scoring pipeline used by the dashboard.

  player KPIs --StandardScaler--> PCA (3 PCs kept) --K-Means (k=4)--> archetype per player
  team-mean KPIs --StandardScaler--> logistic regression --> win probability
  archetype counts --> team balance score (normalised Shannon entropy of the archetype mix)

Interpretation (approved framing): win probability is "how likely a team with this KPI profile is a winning
team", computed from the KPIs of the very match being scored; it is not a pre-match forecast.
Fitting mirrors pca.py / clustering.py exactly (same seed and settings) and is checked against the saved labels.

  python model.py   -> fit, validate, save data/processed/model_artifacts.joblib
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from clustering import N_PC, SEED, name_k4
from manova import team_table

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
ARTIFACTS = PROC / "model_artifacts.joblib"
KPIS = ["kd", "hs_pct", "adr", "opening_rate", "utility_pr", "loadout_value"]
K = 4


def fit():
    kpi = pd.read_csv(PROC / "player_match_kpis.csv")
    scaler = StandardScaler().fit(kpi[KPIS])
    pca = PCA().fit(scaler.transform(kpi[KPIS]))
    scores = pca.transform(scaler.transform(kpi[KPIS]))[:, :N_PC]
    km = KMeans(n_clusters=K, n_init=20, random_state=SEED).fit(scores)
    profile = kpi.assign(cluster=km.labels_).groupby("cluster")[KPIS].mean()
    names = name_k4(profile)

    # Check: identical to the labels saved by clustering.py
    saved = pd.read_csv(PROC / "cluster_labels_k4.csv")
    assert (saved.archetype.to_numpy() == pd.Series(km.labels_).map(names).to_numpy()).all(), \
        "refit archetypes differ from saved cluster_labels_k4.csv"

    # Win model on team-mean KPIs (ties excluded), validated with match-grouped CV
    t = team_table()
    y = t.team_won.astype(int).to_numpy()
    make = lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    p = cross_val_predict(make(), t[KPIS], y, groups=t.file, cv=GroupKFold(5), method="predict_proba")[:, 1]
    metrics = {"auc": roc_auc_score(y, p), "accuracy": accuracy_score(y, p > 0.5),
               "brier": brier_score_loss(y, p), "n_teams": len(t)}
    win_model = make().fit(t[KPIS], y)
    calib = pd.DataFrame({"p": p, "y": y}).assign(bin=lambda d: pd.cut(d.p, [0, .1, .3, .5, .7, .9, 1.0], include_lowest=True)) \
        .groupby("bin", observed=True).agg(n=("y", "size"), predicted=("p", "mean"), observed=("y", "mean"))
    return {"scaler": scaler, "pca": pca, "kmeans": km, "names": names, "win_model": win_model,
            "kpis": KPIS, "metrics": metrics}, calib, win_model


def load():
    return joblib.load(ARTIFACTS)


def balance_score(archetypes, all_names):
    """Normalised Shannon entropy of the archetype mix: 0 = every player the same archetype,
    1 = players spread evenly across all archetypes."""
    counts = pd.Series(archetypes).value_counts().reindex(all_names, fill_value=0).to_numpy(dtype=float)
    q = counts[counts > 0] / counts.sum()
    return float(-(q * np.log(q)).sum() / np.log(len(all_names)))


def score_team(art, players):
    """players: DataFrame with the six KPI columns, one row per player on ONE team.
    Returns per-player archetypes and PC scores, the team balance score, and the win probability."""
    X = players[art["kpis"]].astype(float)
    if X.isna().any().any():
        raise ValueError("KPI columns contain missing values")
    pcs = art["pca"].transform(art["scaler"].transform(X))[:, :N_PC]
    arche = pd.Series(art["kmeans"].predict(pcs)).map(art["names"]).to_numpy()
    team_mean = X.mean().to_frame().T
    prob = float(art["win_model"].predict_proba(team_mean)[0, 1])
    return {"archetypes": arche, "pc_scores": pcs, "team_mean_kpis": team_mean.iloc[0],
            "balance": balance_score(arche, sorted(art["names"].values())), "win_probability": prob}


def main():
    art, calib, win_model = fit()
    joblib.dump(art, ARTIFACTS)
    m = art["metrics"]
    print(f"refit reproduces saved archetypes; archetype names: {art['names']}")
    print(f"win model (match-grouped 5-fold CV, {m['n_teams']} team-matches): "
          f"AUC={m['auc']:.3f}, accuracy={m['accuracy']:.3f}, Brier={m['brier']:.3f}")
    print("\nCalibration (CV predictions):\n", calib.round(3).to_string())
    coefs = pd.Series(win_model[-1].coef_[0], index=KPIS)
    print("\nStandardised logistic coefficients:\n", coefs.round(2).to_string())

    # Smoke test on two real teams from the data (one winner, one loser)
    kpi = pd.read_csv(PROC / "player_match_kpis.csv")
    for won in (1.0, 0.0):
        row = kpi[(kpi.team_won == won) & (kpi.team_size == 5)]
        f, tm = row.iloc[0][["file", "team"]]
        team = kpi[(kpi.file == f) & (kpi.team == tm)]
        r = score_team(art, team)
        print(f"\nsmoke test, {'winning' if won else 'losing'} team: P(win)={r['win_probability']:.3f}, "
              f"balance={r['balance']:.2f}, archetypes={list(r['archetypes'])}")
    print(f"\nsaved {ARTIFACTS}")


if __name__ == "__main__":
    main()
