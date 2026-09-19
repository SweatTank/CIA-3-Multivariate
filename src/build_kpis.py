"""Phase 2: aggregate CS:GO matchmaking event logs into one row per (match, player) with 6 KPIs
and a team win/loss label.

Inputs : data/raw/mm_master_demos.csv, data/raw/mm_grenades_demos.csv
Output : data/processed/player_match_kpis.csv (Steam IDs replaced by pseudonyms P00001..., see pseudonymise)

Definitions (approved):
  rounds_played = last round - first round + 1 in which the player appears (damage as attacker/victim,
                  or grenade thrower). Rows with rounds_played < MIN_ROUNDS are dropped.
  Kill (inferred) = the damage event on which a victim's running hp_dmg total in a round first reaches
                  100; the attacker on that row is credited. World (att_id 0), suicides and team kills
                  are deaths for the victim but not kills for anyone.
  kd            = kills / max(deaths, 1)
  hs_pct        = kills whose killing-blow hitbox is Head / kills (0 when kills = 0)
  adr           = hp_dmg dealt to opponents / rounds_played
  opening_rate  = rounds in which the player's kill was the round's first valid kill / rounds_played
  utility_pr    = unique grenade throws / rounds_played
  loadout_value = hp_dmg-weighted mean gun price over damage dealt to opponents
"""
from pathlib import Path

import pandas as pd

from weapon_prices import WEAPON_PRICES

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "player_match_kpis.csv"
MIN_ROUNDS = 10
KPIS = ["kd", "hs_pct", "adr", "opening_rate", "utility_pr", "loadout_value"]


def load():
    dmg = pd.read_csv(
        RAW / "mm_master_demos.csv",
        usecols=["file", "map", "date", "round", "tick", "att_team", "vic_team", "hp_dmg",
                 "hitbox", "wp", "att_id", "vic_id", "winner_team"],
    )
    nades = pd.read_csv(
        RAW / "mm_grenades_demos.csv",
        usecols=["file", "round", "seconds", "att_id", "nade", "nade_land_x", "nade_land_y"],
    )
    # A few matches exist only in the grenades file (no damage log, hence no team/winner): drop them.
    nades = nades[nades.file.isin(dmg.file.unique())]
    return dmg, nades


def infer_kills(dmg):
    """Return (deaths_df, valid_kills_df). Each victim-round yields at most one death."""
    dmg = dmg.sort_values(["file", "round", "tick"], kind="stable").copy()
    cum = dmg.groupby(["file", "round", "vic_id"]).hp_dmg.cumsum()
    deaths = dmg[(cum >= 100) & (cum - dmg.hp_dmg < 100)]
    valid = deaths[
        (deaths.att_id != 0) & (deaths.att_id != deaths.vic_id) & (deaths.att_team != deaths.vic_team)
    ]
    return deaths, valid


def player_presence(dmg, nades):
    """Rounds-played span per (file, player)."""
    a = dmg.loc[dmg.att_id != 0, ["file", "round", "att_id"]].rename(columns={"att_id": "pid"})
    v = dmg[["file", "round", "vic_id"]].rename(columns={"vic_id": "pid"})
    n = nades.loc[nades.att_id != 0, ["file", "round", "att_id"]].rename(columns={"att_id": "pid"})
    pr = pd.concat([a, v, n]).drop_duplicates()
    span = pr.groupby(["file", "pid"])["round"].agg(first_round="min", last_round="max")
    span["rounds_played"] = span.last_round - span.first_round + 1
    return span[["rounds_played"]]


def player_teams(dmg):
    """Team label per (file, player), from either attacker or victim rows (one label per match)."""
    a = dmg.loc[dmg.att_id != 0, ["file", "att_id", "att_team"]].set_axis(["file", "pid", "team"], axis=1)
    v = dmg[["file", "vic_id", "vic_team"]].set_axis(["file", "pid", "team"], axis=1)
    return pd.concat([a, v]).drop_duplicates().groupby(["file", "pid"]).team.first().to_frame()


def match_labels(dmg):
    """Per (file, team): rounds won, opponent rounds won, team win label (NaN for ties)."""
    rounds = dmg.drop_duplicates(["file", "round"])[["file", "round", "winner_team"]]
    won = rounds.groupby(["file", "winner_team"]).size().rename("team_rounds_won").reset_index()
    won = won.rename(columns={"winner_team": "team"})
    tot = rounds.groupby("file").size().rename("match_rounds")
    won = won.join(tot, on="file")
    # Teams that won zero rounds never appear above; add them from the team list.
    teams = pd.concat([dmg[["file", "att_team"]].set_axis(["file", "team"], axis=1),
                       dmg[["file", "vic_team"]].set_axis(["file", "team"], axis=1)]).drop_duplicates()
    teams = teams[teams.team != "World"]
    lab = teams.merge(won.drop(columns="match_rounds"), on=["file", "team"], how="left").join(tot, on="file")
    lab["team_rounds_won"] = lab.team_rounds_won.fillna(0).astype(int)
    lab["opp_rounds_won"] = lab.match_rounds - lab.team_rounds_won
    lab["is_tie"] = lab.team_rounds_won == lab.opp_rounds_won
    lab["team_won"] = (lab.team_rounds_won > lab.opp_rounds_won).astype(float)
    lab.loc[lab.is_tie, "team_won"] = float("nan")
    lab["match_complete"] = lab.groupby("file").team_rounds_won.transform("max") >= 16
    return lab.drop(columns="match_rounds")


def pseudonymise(ids, seed=0):
    """Replace Steam IDs with random pseudonyms (P00001, ...) so processed data can be published.
    A fixed-seed shuffle (not a hash: hashed Steam IDs can be brute-forced) makes the mapping
    deterministic given the raw data, but not recoverable from the output. IDs are only join keys."""
    uniq = pd.Series(ids.unique())
    order = uniq.sample(frac=1, random_state=seed).reset_index(drop=True)
    mapping = {pid: f"P{i + 1:05d}" for i, pid in enumerate(order)}
    return ids.map(mapping)


def main():
    dmg, nades = load()
    deaths, valid = infer_kills(dmg)

    df = player_presence(dmg, nades)
    df = df.join(player_teams(dmg))
    df = df[df.rounds_played >= MIN_ROUNDS]

    idx = ["file", "pid"]
    kills = valid.groupby(["file", "att_id"]).size().rename("kills").rename_axis(idx)
    hs = valid[valid.hitbox == "Head"].groupby(["file", "att_id"]).size().rename("hs_kills").rename_axis(idx)
    dths = deaths.groupby(["file", "vic_id"]).size().rename("deaths").rename_axis(idx)
    opening = valid.groupby(["file", "round"]).head(1).groupby(["file", "att_id"]).size() \
        .rename("opening_kills").rename_axis(idx)

    opp = dmg[(dmg.att_id != 0) & (dmg.att_id != dmg.vic_id) & (dmg.att_team != dmg.vic_team)]
    dmg_dealt = opp.groupby(["file", "att_id"]).hp_dmg.sum().rename("dmg_dealt").rename_axis(idx)
    gun = opp[opp.wp.isin(WEAPON_PRICES)]
    gun_dmg = gun.groupby(["file", "att_id"]).hp_dmg.sum()
    gun_val = (gun.wp.map(WEAPON_PRICES) * gun.hp_dmg).groupby([gun.file, gun.att_id]).sum()
    loadout = (gun_val / gun_dmg).rename("loadout_value").rename_axis(idx)

    throws = nades[nades.att_id != 0].drop_duplicates(
        ["file", "round", "att_id", "nade", "seconds", "nade_land_x", "nade_land_y"])
    n_throws = throws.groupby(["file", "att_id"]).size().rename("throws").rename_axis(idx)

    df = df.join([kills, hs, dths, opening, dmg_dealt, loadout, n_throws])
    df[["kills", "hs_kills", "deaths", "opening_kills", "dmg_dealt", "throws"]] = \
        df[["kills", "hs_kills", "deaths", "opening_kills", "dmg_dealt", "throws"]].fillna(0)

    df["kd"] = df.kills / df.deaths.clip(lower=1)
    df["hs_pct"] = (df.hs_kills / df.kills.where(df.kills > 0)).fillna(0.0)
    df["adr"] = df.dmg_dealt / df.rounds_played
    df["opening_rate"] = df.opening_kills / df.rounds_played
    df["utility_pr"] = df.throws / df.rounds_played
    # loadout_value is undefined for players who dealt no gun damage; approved decision: drop those rows.
    n_before = len(df)
    df = df[df.loadout_value.notna()]
    print(f"dropped {n_before - len(df)} rows with undefined loadout_value")

    df = df.reset_index().rename(columns={"pid": "player_id"})
    df["player_id"] = pseudonymise(df.player_id)
    meta = dmg.drop_duplicates("file")[["file", "map", "date"]]
    df = df.merge(meta, on="file", how="left").merge(match_labels(dmg), on=["file", "team"], how="left")
    df["team_size"] = df.groupby(["file", "team"]).player_id.transform("nunique")

    cols = ["file", "player_id", "team", "map", "date", "rounds_played", "team_size", *KPIS,
            "kills", "deaths", "team_rounds_won", "opp_rounds_won", "is_tie", "match_complete", "team_won"]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df[cols].to_csv(OUT, index=False)
    print(f"wrote {OUT} ({len(df)} rows)")


if __name__ == "__main__":
    main()
