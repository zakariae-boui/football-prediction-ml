"""
Download match-level Expected Goals (xG) from understat.com

Understat has no official API, and (as of 2026) no longer embeds the data
inline in the page -- the league page is just a shell that loads its data
via AJAX. We hit that same internal JSON endpoint directly:

    GET https://understat.com/main/getLeagueData/{SLUG}/{YEAR}
        headers: X-Requested-With: XMLHttpRequest

It returns {"teams": ..., "players": ..., "dates": [...]}. The `dates` array
is every match of the season with home/away xG, goals and understat's own
pre-match forecast -- exactly what we need to merge onto football-data.co.uk.

Understat season years use the STARTING year:
    2023  ->  season 2023/24  (football-data code "2324")

Run:  python src/download_understat.py --league epl
      python src/download_understat.py --league laliga
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import requests

from leagues import YEARS_US, get

BASE_URL = "https://understat.com/main/getLeagueData"
OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "raw" / "understat"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "X-Requested-With": "XMLHttpRequest",
}


def year_to_season_code(year: int) -> str:
    """2023 -> '2324' (football-data.co.uk season code)."""
    return f"{year % 100:02d}{(year + 1) % 100:02d}"


def download_one(year: int, slug: str) -> pd.DataFrame | None:
    url = f"{BASE_URL}/{slug}/{year}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"  [skip] {slug} {year}: {exc}")
        return None

    payload = resp.json()
    (OUT_DIR / f"{slug}_{year}.json").write_text(json.dumps(payload), encoding="utf-8")

    rows = []
    for m in payload["dates"]:
        if not m.get("isResult"):       # skip not-yet-played fixtures
            continue
        rows.append(
            {
                "understat_id": m["id"],
                "datetime": m["datetime"],
                "HomeTeam": m["h"]["title"],
                "AwayTeam": m["a"]["title"],
                "home_goals": int(m["goals"]["h"]),
                "away_goals": int(m["goals"]["a"]),
                "home_xG": float(m["xG"]["h"]),
                "away_xG": float(m["xG"]["a"]),
                "p_home": float(m["forecast"]["w"]),
                "p_draw": float(m["forecast"]["d"]),
                "p_away": float(m["forecast"]["l"]),
            }
        )

    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["Date"] = df["datetime"].dt.date
    df.insert(0, "Season", year_to_season_code(year))

    print(f"  [ok]   {slug} {year}: {len(df):4d} matches with xG")
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default="epl", help="league key (epl, laliga)")
    args = ap.parse_args()
    cfg = get(args.league)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = []

    print(f"{cfg['name']} ({cfg['understat_slug']})")
    for year in YEARS_US:
        df = download_one(year, cfg["understat_slug"])
        if df is not None and len(df):
            frames.append(df)
        time.sleep(1.0)  # be polite

    if not frames:
        print("No data downloaded.")
        return

    combined = pd.concat(frames, ignore_index=True)
    out_path = OUT_DIR.parent.parent / "processed" / f"understat{cfg['suffix']}.csv"
    combined.to_csv(out_path, index=False)

    print("\nDone.")
    print(f"  Total matches : {len(combined)}")
    print(f"  Seasons       : {sorted(combined['Season'].unique())}")
    print(f"  Saved to      : {out_path}")


if __name__ == "__main__":
    main()
