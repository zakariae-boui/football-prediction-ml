"""
Download historical match results + bookmaker odds from football-data.co.uk

URL pattern: https://www.football-data.co.uk/mmz4281/{SEASON}/{LEAGUE}.csv
  SEASON  e.g. "2324"  -> season 2023/24  (last two digits of each year)
  LEAGUE  e.g. "E0"    -> Premier League, "SP1" -> La Liga

Each CSV already contains BOTH match stats (goals, shots, corners, cards)
AND bookmaker odds (Bet365, Pinnacle, William Hill, ...). No merge needed
here -- everything for one season/league is in a single file.

Run:  python src/download_footballdata.py --league epl
      python src/download_footballdata.py --league laliga
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import requests

from leagues import SEASONS_FD, get

BASE_URL = "https://www.football-data.co.uk/mmz4281"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "footballdata"

# Columns we care about (a superset; we keep whatever exists in each file).
KEEP_COLS = [
    # identity
    "Div", "Date", "Time", "HomeTeam", "AwayTeam",
    # full-time / half-time results
    "FTHG", "FTAG", "FTR", "HTHG", "HTAG", "HTR",
    # match stats
    "HS", "AS", "HST", "AST", "HC", "AC",
    "HF", "AF", "HY", "AY", "HR", "AR",
    # Bet365 odds (early/opening market)
    "B365H", "B365D", "B365A",
    # Pinnacle odds (sharp / closing benchmark)
    "PSH", "PSD", "PSA",
    # Pinnacle closing odds (when present in newer files)
    "PSCH", "PSCD", "PSCA",
    # William Hill (extra bookmaker for robustness)
    "WHH", "WHD", "WHA",
    # Market consensus (max / avg across bookmakers)
    "MaxH", "MaxD", "MaxA", "AvgH", "AvgD", "AvgA",
]


def download_one(season: str, fd_code: str) -> pd.DataFrame | None:
    url = f"{BASE_URL}/{season}/{fd_code}.csv"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  [skip] {fd_code} {season}: {exc}")
        return None

    raw_path = OUT_DIR / f"{fd_code}_{season}.csv"
    raw_path.write_bytes(resp.content)

    # football-data.co.uk files are latin-1 and sometimes have trailing junk cols
    df = pd.read_csv(raw_path, encoding="latin-1", on_bad_lines="skip")
    df = df.dropna(how="all", axis=1)                # drop empty columns
    df = df.dropna(subset=["HomeTeam", "AwayTeam"])   # drop empty rows

    cols = [c for c in KEEP_COLS if c in df.columns]
    df = df[cols].copy()
    df.insert(1, "Season", season)

    print(f"  [ok]   {fd_code} {season}: {len(df):4d} matches, {len(cols)} cols")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="epl", help="league key (epl, laliga)")
    args = ap.parse_args()
    cfg = get(args.league)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []

    print(f"{cfg['name']} ({cfg['fd_code']})")
    for season in SEASONS_FD:
        df = download_one(season, cfg["fd_code"])
        if df is not None and len(df):
            frames.append(df)
        time.sleep(0.5)  # be polite

    if not frames:
        print("No data downloaded.")
        return

    combined = pd.concat(frames, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"], dayfirst=True, errors="coerce")
    combined = combined.sort_values("Date").reset_index(drop=True)

    out_path = OUT_DIR.parent.parent / "processed" / f"footballdata_combined{cfg['suffix']}.csv"
    combined.to_csv(out_path, index=False)

    print("\nDone.")
    print(f"  Total matches : {len(combined)}")
    print(f"  Seasons       : {sorted(combined['Season'].unique())}")
    print(f"  Date range    : {combined['Date'].min().date()} -> {combined['Date'].max().date()}")
    print(f"  Saved to      : {out_path}")


if __name__ == "__main__":
    main()
