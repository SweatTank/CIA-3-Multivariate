"""Phase 3b: K-Means on the retained PC scores (PC1-PC3, approved), evaluated over a range of k.

Inputs : data/processed/pca_scores.csv, data/processed/player_match_kpis.csv
Outputs: outputs/figures/kmeans_k_selection.png
         data/processed/cluster_labels_k{K}.csv, only when run as `python clustering.py K` (final k).
Choosing k is a checkpoint decision, so with no argument this script only reports evidence.
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, calinski_harabasz_score,
                             davies_bouldin_score, silhouette_score)

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
FIG = ROOT / "outputs" / "figures"
N_PC = 3
KPIS = ["kd", "hs_pct", "adr", "opening_rate", "utility_pr", "loadout_value"]
K_RANGE = range(2, 9)
SEED = 42


def fit(Z, k, seed=SEED):
    return KMeans(n_clusters=k, n_init=20, random_state=seed).fit(Z)


def stability(Z, k, n_boot=20):
    """Mean adjusted Rand index between the full-data solution and solutions refit on bootstrap samples."""
    rng = np.random.default_rng(SEED)
    base = fit(Z, k)
    aris = []
    for b in range(n_boot):
        idx = rng.choice(len(Z), len(Z), replace=True)
        km = KMeans(n_clusters=k, n_init=5, random_state=b).fit(Z[idx])
        aris.append(adjusted_rand_score(base.labels_, km.predict(Z)))
    return float(np.mean(aris))


def profile(df, Z, k):
    km = fit(Z, k)
    d = df.copy()
    d["cluster"] = km.labels_
    cent = pd.DataFrame(km.cluster_centers_, columns=[f"PC{i + 1}" for i in range(N_PC)])
    prof = d.groupby("cluster")[KPIS].mean().join(cent)
    prof.insert(0, "n", d.cluster.value_counts().sort_index())
    prof.insert(1, "share", (prof.n / len(d)).round(3))
    return d, prof


def name_k4(prof):
    """Approved k=4 archetype names, assigned from cluster profiles (not from arbitrary cluster ids)."""
    names = {}
    names[prof.kd.idxmax()] = "Star Fragger"
    rest = prof.drop(index=list(names))
    names[rest.utility_pr.idxmax()] = "Utility Support"
    rest = prof.drop(index=list(names))
    names[rest.loadout_value.idxmin()] = "Budget Headshotter"
    names[prof.drop(index=list(names)).index[0]] = "Heavy-Weapon Anchor"
    return names


def main():
    scores = pd.read_csv(PROC / "pca_scores.csv")
    kpi = pd.read_csv(PROC / "player_match_kpis.csv")
    assert (scores.file == kpi.file).all() and (scores.player_id == kpi.player_id).all()
    Z = scores[[f"PC{i + 1}" for i in range(N_PC)]].to_numpy()

    if len(sys.argv) > 1:
        k = int(sys.argv[1])
        d, prof = profile(kpi, Z, k)
        out = scores[["file", "player_id"]].assign(cluster=d.cluster.to_numpy())
        if k == 4:
            names = name_k4(prof)
            out["archetype"] = out.cluster.map(names)
            print("archetype names:", names)
        out.to_csv(PROC / f"cluster_labels_k{k}.csv", index=False)
        print(prof.round(3).to_string())
        print(f"saved cluster_labels_k{k}.csv")
        return

    rows = []
    for k in K_RANGE:
        km = fit(Z, k)
        rows.append({
            "k": k, "inertia": km.inertia_,
            "silhouette": silhouette_score(Z, km.labels_),
            "calinski_harabasz": calinski_harabasz_score(Z, km.labels_),
            "davies_bouldin": davies_bouldin_score(Z, km.labels_),
            "min_cluster_share": np.bincount(km.labels_).min() / len(Z),
            "bootstrap_ARI": stability(Z, k),
        })
    res = pd.DataFrame(rows).set_index("k")
    print(res.round(3).to_string())

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, col, title in zip(axes, ["inertia", "silhouette", "bootstrap_ARI"],
                              ["Elbow (inertia)", "Silhouette (higher = better)", "Bootstrap stability (ARI)"]):
        ax.plot(res.index, res[col], "o-")
        ax.set(xlabel="k", title=title)
    fig.tight_layout()
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / "kmeans_k_selection.png", dpi=150)

    for k in (3, 4, 5):
        _, prof = profile(kpi, Z, k)
        print(f"\n--- k={k}: cluster sizes, mean KPIs, PC centroids ---")
        print(prof.round(2).to_string())


if __name__ == "__main__":
    main()
