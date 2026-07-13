"""
Feature engineering for match-outcome prediction -- LEAKAGE-SAFE.

Golden rule: every feature describing a match is computed using ONLY matches
that finished BEFORE it. We enforce this with chronological `shift(1)` before
any rolling/expanding window, so a match never "sees" its own result or any
future result.

Pipeline
--------
1. Explode each match into two team-perspective rows (home + away).
2. Sort chronologically per team and compute rolling / season-to-date form
   with shift(1).
3. Pivot the team features back to one row per match (home_* / away_* / *_diff).
4. Add head-to-head features (chronological loop, also leakage-safe).
5. Add market features from the odds (already pre-match, no leakage).
6. Add the targets: FTR (3-class) and a binary "home vs away" label.

Run:  python src/features.py        ->  data/processed/matches_features.csv
"""

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"

FORM_WINDOW = 5      # "recent form" = last 5 matches
H2H_WINDOW = 5       # head-to-head = last 5 meetings


# --- helpers -------------------------------------------------------------

def implied_probs(h, d, a):
    """De-vig 3-way decimal odds into probabilities that sum to 1."""
    rh, rd, ra = 1 / h, 1 / d, 1 / a
    s = rh + rd + ra
    return rh / s, rd / s, ra / s, s


def _points(gf, ga):
    return np.where(gf > ga, 3, np.where(gf == ga, 1, 0))


# --- 1. explode to team-perspective long format --------------------------

def to_long(df: pd.DataFrame) -> pd.DataFrame:
    """One row per team per match (home perspective + away perspective)."""
    base = ["match_id", "League", "Season", "Date"]

    home = df[base].copy()
    home["team"] = df["HomeTeam"]
    home["opp"] = df["AwayTeam"]
    home["venue"] = "H"
    home["gf"], home["ga"] = df["FTHG"], df["FTAG"]
    home["xg"], home["xga"] = df["home_xG"], df["away_xG"]
    home["sot_for"], home["sot_against"] = df["HST"], df["AST"]

    away = df[base].copy()
    away["team"] = df["AwayTeam"]
    away["opp"] = df["HomeTeam"]
    away["venue"] = "A"
    away["gf"], away["ga"] = df["FTAG"], df["FTHG"]
    away["xg"], away["xga"] = df["away_xG"], df["home_xG"]
    away["sot_for"], away["sot_against"] = df["AST"], df["HST"]

    long = pd.concat([home, away], ignore_index=True)
    long["points"] = _points(long["gf"], long["ga"])
    long["gd"] = long["gf"] - long["ga"]
    long["xgd"] = long["xg"] - long["xga"]
    long["win"] = (long["points"] == 3).astype(int)
    return long.sort_values(["League", "team", "Date"]).reset_index(drop=True)


# --- 2. rolling / season-to-date features (leakage-safe) -----------------

def add_team_features(long: pd.DataFrame) -> pd.DataFrame:
    season_keys = ["League", "Season", "team"]

    # rolling mean of the PRIOR FORM_WINDOW matches (shift(1) excludes current)
    roll_cols = ["points", "gf", "ga", "xg", "xga", "xgd",
                 "sot_for", "sot_against", "win"]

    def roll_mean(s):
        return s.shift(1).rolling(FORM_WINDOW, min_periods=1).mean()

    for c in roll_cols:
        long[f"form_{c}"] = long.groupby(season_keys)[c].transform(roll_mean)

    # season-to-date (strictly before this match)
    long["played_sofar"] = long.groupby(season_keys).cumcount()
    long["pts_sofar"] = long.groupby(season_keys)["points"].transform(
        lambda s: s.shift(1).cumsum())
    long["gd_sofar"] = long.groupby(season_keys)["gd"].transform(
        lambda s: s.shift(1).cumsum())
    long["ppg_sofar"] = long["pts_sofar"] / long["played_sofar"].replace(0, np.nan)

    # momentum: points in the last 3 matches
    long["mom_pts_last3"] = long.groupby(season_keys)["points"].transform(
        lambda s: s.shift(1).rolling(3, min_periods=1).sum())

    # venue-specific form (home team's home record / away team's away record)
    venue_keys = ["League", "Season", "team", "venue"]
    long["form_venue_ppg"] = long.groupby(venue_keys)["points"].transform(roll_mean)
    long["form_venue_xgd"] = long.groupby(venue_keys)["xgd"].transform(roll_mean)

    # rest days (crosses seasons within a league; season opener -> NaN)
    long["rest_days"] = (
        long.groupby(["League", "team"])["Date"].diff().dt.days
    )
    return long


# --- 3. pivot team features back to one row per match --------------------

def pivot_back(df: pd.DataFrame, long: pd.DataFrame) -> pd.DataFrame:
    feat_cols = [c for c in long.columns if c.startswith(
        ("form_", "pts_", "gd_", "ppg_", "mom_", "played_", "rest_"))]

    home = (long[long["venue"] == "H"]
            .set_index("match_id")[feat_cols].add_prefix("home_"))
    away = (long[long["venue"] == "A"]
            .set_index("match_id")[feat_cols].add_prefix("away_"))

    out = df.set_index("match_id").join(home).join(away)

    # difference features (home - away): usually the strongest signals
    for c in feat_cols:
        out[f"diff_{c}"] = out[f"home_{c}"] - out[f"away_{c}"]
    return out.reset_index()


# --- 4. head-to-head (chronological, leakage-safe) -----------------------

def add_h2h(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("Date").reset_index(drop=True)
    history = defaultdict(list)   # unordered pair -> [(winner_team|'D', total_goals)]
    hr, hg, hn = {}, {}, {}

    for row in df.itertuples():
        pair = tuple(sorted([row.HomeTeam, row.AwayTeam]))
        past = history[pair][-H2H_WINDOW:]
        if past:
            n = len(past)
            # win rate from the CURRENT home team's perspective
            hr[row.match_id] = sum(w == row.HomeTeam for w, _ in past) / n
            hg[row.match_id] = float(np.mean([g for _, g in past]))
            hn[row.match_id] = n
        else:
            hr[row.match_id] = np.nan
            hg[row.match_id] = np.nan
            hn[row.match_id] = 0

        # record this match AFTER reading (so it can't leak into itself)
        winner = row.HomeTeam if row.FTR == "H" else (
            row.AwayTeam if row.FTR == "A" else "D")
        history[pair].append((winner, row.FTHG + row.FTAG))

    df["h2h_home_winrate"] = df["match_id"].map(hr)
    df["h2h_avg_goals"] = df["match_id"].map(hg)
    df["h2h_n"] = df["match_id"].map(hn)
    return df


# --- 5. market features from odds ----------------------------------------

def add_market(df: pd.DataFrame) -> pd.DataFrame:
    ph, pd_, pa, over = implied_probs(df["B365H"], df["B365D"], df["B365A"])
    df["mkt_b365_p_home"], df["mkt_b365_p_draw"], df["mkt_b365_p_away"] = ph, pd_, pa
    df["mkt_b365_margin"] = over - 1

    psh, psd, psa, over_ps = implied_probs(df["PSH"], df["PSD"], df["PSA"])
    df["mkt_pin_p_home"], df["mkt_pin_p_draw"], df["mkt_pin_p_away"] = psh, psd, psa
    df["mkt_pin_margin"] = over_ps - 1
    return df


# --- 6. targets ----------------------------------------------------------

def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    df["target_3way"] = df["FTR"]                       # H / D / A
    # binary "Draw No Bet": home vs away; draws -> NA (excluded at modeling time)
    df["target_binary"] = df["FTR"].map({"H": "H", "A": "A", "D": pd.NA})
    return df


# --- orchestrator --------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    df["match_id"] = df.index

    long = to_long(df)
    long = add_team_features(long)
    out = pivot_back(df, long)
    out = add_h2h(out)
    out = add_market(out)
    out = add_targets(out)
    return out.sort_values("Date").reset_index(drop=True)


# --- model-safe feature selection ----------------------------------------

# Columns that describe the CURRENT match outcome -> using them would be
# leakage. They stay in the file (for reference / the betting sim) but must
# NEVER be fed to a model.
LEAKAGE_COLS = [
    "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR",
    "HS", "AS", "HST", "AST", "HC", "AC", "HF", "AF",
    "HY", "AY", "HR", "AR", "home_xG", "away_xG",
    "p_home", "p_draw", "p_away",                 # understat's own post-hoc forecast
    "B365H", "B365D", "B365A", "PSH", "PSD", "PSA",
    "PSCH", "PSCD", "PSCA", "WHH", "WHD", "WHA",
    "MaxH", "MaxD", "MaxA", "AvgH", "AvgD", "AvgA",
]

# Prefixes of engineered, leakage-safe model features.
SAFE_PREFIXES = ("home_", "away_", "diff_", "h2h_", "mkt_")


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the leakage-safe engineered feature columns present in df.

    Note the explicit LEAKAGE_COLS exclusion: raw current-match columns like
    `home_xG` / `away_xG` share the `home_`/`away_` prefix but are POST-match,
    so they must be filtered out here.
    """
    return [c for c in df.columns
            if c.startswith(SAFE_PREFIXES) and c not in LEAKAGE_COLS]


def main() -> None:
    df = pd.read_csv(PROC / "matches_all_leagues.csv", dtype={"Season": str})
    feats = build_features(df)

    out_path = PROC / "matches_features.csv"
    feats.to_csv(out_path, index=False)

    n_feat = len(feature_columns(feats))
    print("Feature engineering complete.")
    print(f"  rows           : {len(feats)}")
    print(f"  total columns  : {feats.shape[1]}")
    print(f"  model features : {n_feat}")
    print(f"  saved to       : {out_path}")


if __name__ == "__main__":
    main()
