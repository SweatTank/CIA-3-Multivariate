"""Streamlit dashboard: upload a team's player KPIs -> archetype labels, team balance score, win probability.

Run locally:  streamlit run app.py
Pure statistics (scaler + PCA + K-Means + logistic regression); no external services or APIs.
"""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from model import KPIS, load, score_team  # noqa: E402
from visualize import ORDER, fit_gaussians, plot_density  # noqa: E402

PROC = ROOT / "data" / "processed"
KPI_HELP = {
    "kd": "Kills / deaths (deaths floored at 1)",
    "hs_pct": "Share of kills that were headshots, as a fraction 0-1 (accuracy proxy)",
    "adr": "Average damage per round dealt to opponents",
    "opening_rate": "Share of rounds where the player got the round's first kill (reaction-time proxy)",
    "utility_pr": "Grenades thrown per round (map-control proxy)",
    "loadout_value": "Damage-weighted average weapon price in dollars (economy proxy)",
}

st.set_page_config(page_title="CS:GO Team Profiler", layout="wide")


@st.cache_resource
def get_artifacts():
    return load()


@st.cache_data
def get_background():
    kpi = pd.read_csv(PROC / "player_match_kpis.csv")
    lab = pd.read_csv(PROC / "cluster_labels_k4.csv")
    df = kpi.merge(lab[["file", "player_id", "archetype"]], on=["file", "player_id"])
    return kpi, df


@st.cache_data
def get_gauss(df):
    return fit_gaussians(df)


def examples(kpi):
    out = {}
    for won, label in ((1.0, "Example: a winning team"), (0.0, "Example: a losing team")):
        r = kpi[(kpi.team_won == won) & (kpi.team_size == 5)].iloc[5]
        t = kpi[(kpi.file == r.file) & (kpi.team == r.team)][KPIS].reset_index(drop=True)
        t.insert(0, "player", [f"Player {i + 1}" for i in range(len(t))])
        out[label] = t
    return out


def read_upload(f):
    df = pd.read_csv(f)
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [k for k in KPIS if k not in df.columns]
    if missing:
        raise ValueError(f"Missing KPI column(s): {', '.join(missing)}")
    if "player" not in df.columns:
        df.insert(0, "player", [f"Player {i + 1}" for i in range(len(df))])
    return df[["player", *KPIS]]


st.title("CS:GO team profiler")
st.caption("Multivariate player profiling (PCA + K-Means archetypes) and MANOVA-informed win probability. "
           "Trained on 12,767 player-match rows from 1,289 competitive matchmaking matches.")

kpi, bg = get_background()
art = get_artifacts()
gauss = get_gauss(bg)
ex = examples(kpi)

with st.sidebar:
    st.header("Team input")
    up = st.file_uploader("Upload a CSV with one row per player", type="csv")
    choice = st.selectbox("...or start from an example", list(ex))
    template = ex[choice].to_csv(index=False).encode()
    st.download_button("Download CSV template", template, "team_kpis_template.csv", "text/csv")
    with st.expander("Column definitions"):
        for k in KPIS:
            st.markdown(f"**{k}**: {KPI_HELP[k]}")
        st.caption("Optional `player` column for names. Extra columns are ignored.")

try:
    players = read_upload(up) if up is not None else ex[choice]
except Exception as e:  # unreadable or malformed upload
    st.error(f"Could not read the file: {e}")
    st.stop()

st.subheader("Team KPI data (editable)")
players = st.data_editor(players, num_rows="dynamic", width="stretch", hide_index=True)

if players[KPIS].isna().any().any() or len(players) == 0:
    st.warning("Fill in all six KPI values for every player to see results.")
    st.stop()
if (players.hs_pct > 1).any() or (players.hs_pct < 0).any():
    st.error("`hs_pct` must be a fraction between 0 and 1 (e.g. 0.35 for 35%).")
    st.stop()
if len(players) != 5:
    st.info(f"The model was trained mostly on 5-player teams; you entered {len(players)}.")

lo, hi = kpi[KPIS].quantile(0.01), kpi[KPIS].quantile(0.99)
outside = [k for k in KPIS if ((players[k] < lo[k]) | (players[k] > hi[k])).any()]
if outside:
    st.info("Some values fall outside the range of 98% of training players "
            f"({', '.join(outside)}); results there are extrapolation.")

res = score_team(art, players[KPIS])
c1, c2, c3 = st.columns(3)
pw = res["win_probability"]
c1.metric("Win probability", ">99%" if pw > 0.99 else "<1%" if pw < 0.01 else f"{pw:.0%}")
c2.metric("Team balance score", f"{res['balance']:.2f}")
mix = pd.Series(res["archetypes"]).value_counts()
c3.metric("Most common archetype", mix.index[0], f"{mix.iloc[0]} of {len(players)} players", delta_color="off")
st.caption("**Win probability** = how likely a team with this KPI profile is a winning team, from the KPIs of "
           "the match being scored (not a pre-match forecast). **Balance** = evenness of the archetype mix, "
           "0 (all players the same archetype) to 1 (spread evenly across all four).")

left, right = st.columns([1, 1])
with left:
    st.subheader("Archetype labels")
    out = players.copy()
    out.insert(1, "archetype", res["archetypes"])
    for i in range(res["pc_scores"].shape[1]):
        out[f"PC{i + 1}"] = res["pc_scores"][:, i].round(2)
    st.dataframe(out, width="stretch", hide_index=True)
    st.caption("PC1 ~ Aggression / Impact, PC2 ~ Precision vs weapon tier, PC3 ~ Utility / Support.")
with right:
    st.subheader("Archetype mix")
    st.bar_chart(mix.reindex(ORDER, fill_value=0), horizontal=True)

st.subheader("Where your players sit: kill rate vs accuracy")
team_pts = players.assign(archetype=res["archetypes"]).rename(columns={"player": "label"})
fig = plot_density(bg, gauss, team=team_pts)
st.pyplot(fig, width="stretch")

with st.expander("Method and limitations"):
    st.markdown(
        "- **Archetypes:** KPIs are z-scored, reduced to 3 principal components (75% of variance), then assigned "
        "to one of 4 K-Means clusters. Silhouette is only about 0.23, so archetypes are useful segments on a "
        "continuum, not sharply separated groups.\n"
        "- **Win test:** paired within-match MANOVA (Wilks' Lambda = 0.45, partial eta-squared = 0.55) plus a permutation test. "
        "Multivariate normality and Box's M assumptions failed, which is why the permutation test was used.\n"
        "- **Substitutions:** the data has no shots-fired or reaction-time telemetry. Accuracy is headshot "
        "share of kills and reaction time is opening-kill rate. Kills are inferred from cumulative damage "
        "reaching 100 HP.\n"
        "- **K/D dominates:** K/D alone predicts winning almost as well (AUC 0.977 vs 0.979), so the joint "
        "vector adds little for classification.\n"
        "- **Model quality:** match-grouped 5-fold CV, AUC "
        f"{art['metrics']['auc']:.3f}, accuracy {art['metrics']['accuracy']:.1%}.")

