# Dataset Documentation

## Pipeline

```
                 per league (epl, laliga)
download_footballdata.py ─┐
                          ├─► merge_data.py ─► matches_merged_{league}.csv ─┐
download_understat.py    ─┘                                                 │
                                                                            ▼
                                              combine_leagues.py ─► matches_all_leagues.csv
```

Run order (all driven by `src/leagues.py` config):
```bash
# English Premier League
python src/download_footballdata.py --league epl
python src/download_understat.py    --league epl
python src/merge_data.py            --league epl

# Spanish La Liga
python src/download_footballdata.py --league laliga
python src/download_understat.py    --league laliga
python src/merge_data.py            --league laliga

# Stack both into the master file
python src/combine_leagues.py
```

## The 7 files

| File | What |
|---|---|
| `footballdata_combined_epl.csv` | PL results + odds |
| `understat_epl.csv` | PL xG |
| `matches_merged_epl.csv` | PL joined (3,040) |
| `footballdata_combined_laliga.csv` | La Liga results + odds |
| `understat_laliga.csv` | La Liga xG |
| `matches_merged_laliga.csv` | La Liga joined (3,040) |
| **`matches_all_leagues.csv`** | **← THE FILE THE NOTEBOOK USES (6,080)** |

## Master dataset: `data/processed/matches_all_leagues.csv`

- **Rows:** 6,080 matches (3,040 EPL + 3,040 La Liga)
- **Leagues:** English Premier League (E0) + Spanish La Liga (SP1)
- **Seasons:** 2018/19 → 2025/26 (8 seasons each)
- **Extra column:** `League` (`epl` / `laliga`) — enables per-league analysis
- **Merge integrity:** 100% joined in both leagues; goals cross-checked
  between sources → 3040/3040 agree in *each* league (no mis-mapped fixtures).

## Data dictionary

### Identity & result
| Column | Meaning |
|---|---|
| `League` | `epl` or `laliga` (master file only) |
| `Season` | Season code, e.g. `2324` = 2023/24 |
| `Date`, `Time` | Kickoff date / time |
| `HomeTeam`, `AwayTeam` | Clubs |
| `FTHG`, `FTAG` | Full-time goals home / away |
| **`FTR`** | **Full-time result: H / D / A — the TARGET** |
| `HTHG`, `HTAG`, `HTR` | Half-time goals / result |

### Match stats
| Column | Meaning |
|---|---|
| `HS`, `AS` | Shots home / away |
| `HST`, `AST` | Shots on target |
| `HC`, `AC` | Corners |
| `HF`, `AF` | Fouls |
| `HY`, `AY`, `HR`, `AR` | Yellow / Red cards |

### Expected goals (Understat)
| Column | Meaning |
|---|---|
| `home_xG`, `away_xG` | Expected goals per team for the match |
| `p_home`, `p_draw`, `p_away` | Understat's own pre-match forecast probabilities |
| `understat_id` | Understat match id (used for the goal cross-check) |

### Bookmaker odds
| Column | Bookmaker / type |
|---|---|
| `B365H/D/A` | Bet365 (pre-match) |
| `PSH/PSD/PSA` | Pinnacle (pre-match) |
| `PSCH/PSCD/PSCA` | **Pinnacle closing** (the sharp benchmark) |
| `WHH/WHD/WHA` | William Hill |
| `MaxH/D/A` | Best price across all bookmakers |
| `AvgH/D/A` | Average price across all bookmakers |

## Known missing values (not a problem)

| Columns | Missing | Reason |
|---|---|---|
| `Time`, `Max*`, `Avg*` | 380 (1 season) | Oldest season (2018/19) lacked these columns |
| `WHH/WHD/WHA` | 91 | William Hill absent for some fixtures |
| `Div` | 760 | League-code label, unused for modeling |

All **core** columns (result, goals, shots, xG, Bet365 & Pinnacle odds) are
100% complete.

## Extending to other leagues

Everything is driven by `src/leagues.py`. To add e.g. Serie A:
1. Add an entry to `LEAGUES` with its `fd_code` (`I1`), `understat_slug`
   (`Serie_A`), a `suffix` (`_seriea`), and an empty `name_map`.
2. Run the download + merge scripts with `--league seriea`.
3. The merge's goal cross-check will print any mismatched fixtures — fill the
   `name_map` for those teams and re-run.
4. `combine_leagues.py` automatically picks up every league in `LEAGUES`.
