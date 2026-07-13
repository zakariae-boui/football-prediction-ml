"""
Shared configuration for all data scripts.

One place that ties together, for each league:
  - the football-data.co.uk league code   (E0, SP1, ...)
  - the understat league slug              (EPL, La_liga, ...)
  - the file suffix used for outputs       (_epl, _laliga)
  - the team-name map (understat name -> football-data name)

The download/merge scripts take a --league argument (key of LEAGUES below).
"""

# football-data.co.uk season codes ("1819" = 2018/19) and the matching
# understat starting years (2018 = 2018/19). Kept aligned by index.
SEASONS_FD = ["1819", "1920", "2021", "2122", "2223", "2324", "2425", "2526"]
YEARS_US = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]

LEAGUES = {
    "epl": {
        "name": "English Premier League",
        "fd_code": "E0",
        "understat_slug": "EPL",
        "suffix": "_epl",
        # understat name -> football-data name (only the ones that differ)
        "name_map": {
            "Manchester City": "Man City",
            "Manchester United": "Man United",
            "Newcastle United": "Newcastle",
            "Nottingham Forest": "Nott'm Forest",
            "West Bromwich Albion": "West Brom",
            "Wolverhampton Wanderers": "Wolves",
        },
    },
    "laliga": {
        "name": "Spanish La Liga",
        "fd_code": "SP1",
        "understat_slug": "La_liga",
        "suffix": "_laliga",
        # understat name -> football-data name (only the ones that differ)
        "name_map": {
            "Athletic Club": "Ath Bilbao",
            "Atletico Madrid": "Ath Madrid",
            "Celta Vigo": "Celta",
            "Espanyol": "Espanol",
            "Rayo Vallecano": "Vallecano",
            "Real Betis": "Betis",
            "Real Sociedad": "Sociedad",
            "Real Valladolid": "Valladolid",
            "Real Oviedo": "Oviedo",
            "SD Huesca": "Huesca",
        },
    },
}


def get(league_key: str) -> dict:
    if league_key not in LEAGUES:
        raise SystemExit(
            f"Unknown league '{league_key}'. Choose from: {list(LEAGUES)}"
        )
    return LEAGUES[league_key]
