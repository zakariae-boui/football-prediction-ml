"""
Merge football-data.co.uk (results + odds) with Understat (xG) for ONE league.

Merge key: (Season, HomeTeam, AwayTeam)

Team names differ between the two sources; we normalise understat names onto
the football-data convention using the per-league map in leagues.py, then
cross-check goals between the two sources to prove no fixture was mismatched.

Output: data/processed/matches_merged{suffix}.csv

Run:  python src/merge_data.py --league epl
      python src/merge_data.py --league laliga
"""

import argparse
from pathlib import Path

import pandas as pd

from leagues import get

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="epl", help="league key (epl, laliga)")
    args = ap.parse_args()
    cfg = get(args.league)
    sfx = cfg["suffix"]

    fd = pd.read_csv(PROC / f"footballdata_combined{sfx}.csv", dtype={"Season": str})
    us = pd.read_csv(PROC / f"understat{sfx}.csv", dtype={"Season": str})

    # --- normalise understat team names onto football-data convention -----
    us["HomeTeam"] = us["HomeTeam"].replace(cfg["name_map"])
    us["AwayTeam"] = us["AwayTeam"].replace(cfg["name_map"])

    us_slim = us[
        [
            "Season", "HomeTeam", "AwayTeam",
            "home_goals", "away_goals",        # kept ONLY for the cross-check
            "home_xG", "away_xG",
            "p_home", "p_draw", "p_away",
            "understat_id",
        ]
    ].copy()

    # --- merge ------------------------------------------------------------
    merged = fd.merge(
        us_slim,
        on=["Season", "HomeTeam", "AwayTeam"],
        how="left",
        validate="one_to_one",
    )

    n_total = len(merged)
    n_matched = int(merged["home_xG"].notna().sum())
    n_missing = n_total - n_matched

    print(f"Merge report -- {cfg['name']}")
    print(f"  merged rows : {n_total}")
    print(f"  with xG     : {n_matched} ({n_matched / n_total:.1%})")
    print(f"  missing xG  : {n_missing}")

    if n_missing:
        unmatched = merged[merged["home_xG"].isna()][
            ["Season", "Date", "HomeTeam", "AwayTeam"]
        ]
        print("\n  Unmatched rows (no xG found) -- check name_map:")
        print(unmatched.to_string(index=False))

    # --- integrity cross-check: goals must agree between both sources ------
    chk = merged.dropna(subset=["home_goals"]).copy()
    bad = chk[(chk["FTHG"] != chk["home_goals"]) | (chk["FTAG"] != chk["away_goals"])]
    print(f"\n  Goal cross-check: {len(chk) - len(bad)}/{len(chk)} agree, "
          f"{len(bad)} mismatches")
    if len(bad):
        print("  !! MISMATCHED FIXTURES (wrong name mapping?):")
        print(bad[["Season", "Date", "HomeTeam", "AwayTeam",
                   "FTHG", "FTAG", "home_goals", "away_goals"]].to_string(index=False))

    # drop the cross-check helper columns; goals come from football-data (FTHG/FTAG)
    merged = merged.drop(columns=["home_goals", "away_goals"])

    out_path = PROC / f"matches_merged{sfx}.csv"
    merged.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}  ({merged.shape[0]} rows x {merged.shape[1]} cols)")


if __name__ == "__main__":
    main()
