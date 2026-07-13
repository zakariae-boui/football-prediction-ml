"""
Combine the per-league merged datasets into ONE master file with a `League`
column. This is the file the notebook actually uses.

It stacks every matches_merged{suffix}.csv defined in leagues.py.

Output: data/processed/matches_all_leagues.csv

Run:  python src/combine_leagues.py
"""

from pathlib import Path

import pandas as pd

from leagues import LEAGUES

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"


def main() -> None:
    frames = []
    for key, cfg in LEAGUES.items():
        path = PROC / f"matches_merged{cfg['suffix']}.csv"
        if not path.exists():
            print(f"  [skip] {key}: {path.name} not found "
                  f"(run merge_data.py --league {key} first)")
            continue
        df = pd.read_csv(path, dtype={"Season": str})
        df.insert(0, "League", key)          # 'epl' / 'laliga'
        frames.append(df)
        print(f"  [ok]   {key:7s}: {len(df)} matches")

    if not frames:
        print("Nothing to combine.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"], errors="coerce")
    combined = combined.sort_values(["Date", "League"]).reset_index(drop=True)

    out_path = PROC / "matches_all_leagues.csv"
    combined.to_csv(out_path, index=False)

    print("\nDone.")
    print(f"  Total matches : {len(combined)}")
    print(f"  Leagues       : {combined['League'].value_counts().to_dict()}")
    print(f"  Seasons       : {sorted(combined['Season'].unique())}")
    print(f"  Saved to      : {out_path}")
    print(f"  shape         : {combined.shape[0]} rows x {combined.shape[1]} cols")


if __name__ == "__main__":
    main()
