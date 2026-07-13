# ⚽ Football Prediction with Machine Learning — Two Applications

> **Course:** Machine Learning
> **Type:** Classification + Model Comparison
> **Deliverable:** Jupyter Notebooks (+ a French presentation)
> **Team:** 2 members

---

## 0. The Big Picture — One Method, Two Applications

We build and compare **three models — Random Forest, XGBoost, SVM** — and apply the *same methodology* to **two real decisions**:

| | **Application A — League Betting** | **Application B — World Cup 2026** |
|---|---|---|
| **Question** | Can we predict league matches and **beat the bookmaker**? | Can we **forecast the 2026 World Cup** — who advances and who wins? |
| **Data** | EPL + La Liga, 6 080 matches, odds + xG | 49 493 international matches (1872→2026) |
| **Team strength** | club form, xG, market odds | a self-computed **Elo** rating |
| **Models** | RF / XGBoost / SVM, time-based split | the *same* 3 models, train / val / test |
| **From probabilities to a decision** | **value betting** + ROI + Kelly | **Monte Carlo** simulation of the bracket |
| **Notebook** | `01`–`04` | `05` |

> **One-sentence thesis:** *we built RF / XGBoost / SVM to predict football, then used the same methodology for two real decisions — trying to beat a betting market (A) and forecasting a live World Cup (B).*

---

## 1. Algorithms We Compare

As required by the course, we compare **three** algorithms in both applications:

| Algorithm | Why it's included | Observed behavior |
|---|---|---|
| **Random Forest** | Robust ensemble, handles non-linear features, little tuning | Strong, stable baseline |
| **XGBoost** | State-of-the-art for tabular data | Best probabilities for the betting layer |
| **SVM** | Classic margin-based classifier, needs scaling | Comparable accuracy, slowest to train |

We discuss *why* they behave differently (e.g. why boosting handles noisy tabular sports data well, why SVM scales poorly to tens of thousands of rows).

---

# APPLICATION A — League Betting

## 2. The Data (App A)

### 2.1 Primary source — `football-data.co.uk`
Free CSVs, results **and** bookmaker odds in one file.

| Column | Meaning |
|---|---|
| `Date`, `HomeTeam`, `AwayTeam` | Match identity |
| `FTHG` / `FTAG` | Full-time goals home / away |
| `FTR` | Full-time result (H/D/A) — **our target** |
| `HS/AS`, `HST/AST`, `HC/AC`, `HY/AY/HR/AR` | Shots, shots on target, corners, cards |
| `B365H/D/A` | Bet365 odds |
| `PSH/PSD/PSA`, `PSCH/PSCD/PSCA` | **Pinnacle** odds (open / closing) — the "sharp" benchmark |

### 2.2 Secondary source — `Understat` (Expected Goals)
| Feature | Meaning |
|---|---|
| `xG` | Expected goals — shot **quality**, not just quantity |
| `xGA` | Expected goals conceded |
| `xG difference` | created − conceded — one of the strongest single signals in football |

> **Why xG matters:** a team can win 3–0 on lucky long shots while losing the xG battle 0.4 vs 2.1. xG predicts *future* performance far better than past goals.

### 2.3 The merge
```
football-data.co.uk (results + odds)
   + merge on (Date, HomeTeam, AwayTeam)
Understat (xG per team per match)
   = one clean DataFrame → 114 columns, 62 used as model features
```

**Scope:** Premier League + La Liga, **8 seasons (2018/19 → 2025/26), 6 080 matches.**

---

## 3. Feature Engineering (App A)

We build **62 leakage-safe features**, in four families:

| Family | Count | Examples |
|---|---|---|
| **Recent form (last 5)** | 33 | points-per-game, goals for/against, shots on target, win rate, *venue-specific* form |
| **xG form (last 5)** | 14 | xG for/against, xG difference — underlying quality |
| **Market signal** | 23 | de-vigged implied P(H/D/A) from Bet365 / Pinnacle, bookmaker margin |
| **Context** | 6 | days rest, head-to-head record |
| **Difference features** | (within above) | home − away deltas — usually the strongest |

> ⚠️ **Leakage rule (validated in notebook `02`):** every feature uses **only matches played before the current one** (chronological `shift(1)`); rolling windows reset per season. We *banned* 41 post-match columns (score, shots, in-match xG…). Proof: the honest feature `diff_form_xgd` correlates **0.288** with the result, while the cheating in-match xG correlates **0.507** — "too strong" is the red flag for leakage.

### 3.1 Two classification framings
| # | Framing | Classes | Purpose |
|---|---|---|---|
| **1** | **3-class** (main) | Home / Draw / Away | The full, realistic problem (course requirement) |
| **2** | **Binary** ("Draw No Bet") | Home / Away | Isolates *how much* the draw hurts; maps to a real betting market |

> Binary accuracy (~73 %) is **not** comparable to 3-class accuracy (~54 %): it answers an easier question. We use it only to show that **the draw is the hard part**, and to stake the "Draw No Bet" market in the betting layer.

---

## 4. Methodology — Train / Validation / Test (App A)

Sports data is a time series, so a random split would leak the future.

```
WRONG ❌  random split            → model "sees the future" → fake accuracy
RIGHT ✅  TRAIN  2018/19–2021/22  (2 777 matches)  → models learn
          VAL    2022/23          (~920)            → tune hyper-parameters
          TEST   2023/24–2025/26  (2 220)           → final, untouched check
```

**Tuned hyper-parameters** (chosen on the validation set, saved to `data/processed/best_params.json`):

| Model | Best settings | Validation accuracy |
|---|---|---|
| Random Forest | `max_depth=6`, `min_samples_leaf=1` | 53.4 % |
| XGBoost | `max_depth=3`, `learning_rate=0.03` | 51.2 % |
| SVM | `C=0.5`, `gamma=scale` (RBF) | 54.3 % |

Settings are deliberately "soft" (shallow trees, slow learning): football is noisy, so simpler models avoid overfitting.

### 4.1 The betting-simulation logic
```
Bookmaker odds 2.50  → implied probability = 1 / 2.50 = 40 %
Our model says       → Home Win = 55 %
55 % > 40 %          → VALUE BET (market underpricing) → place a bet
```
We stake this across the whole test set, track the **bankroll over time**, and measure **Closing Line Value (CLV)** against Pinnacle.

---

## 5. Results (App A) — Honest Findings

| Model | 3-class accuracy | Draw recall | Best ROI (Draw-No-Bet) | Avg CLV |
|---|---|---|---|---|
| Random Forest | 53.9 % | 0.7 % | −7.0 % | −0.004 |
| XGBoost | 52.8 % | 6.3 % | **−2.9 %** | −0.001 |
| SVM | 54.2 % | 0.2 % | −8.4 % | −0.014 |
| **Bookmaker baseline** | **54.7 %** | — | — | — |

**What we learned:**
- **Accuracy ≈ the bookmaker (~54 %), never above it.** The closing odds already price all public information → there is no edge to exploit.
- **The draw is the enemy:** rare (~25 %) and dispersed, its recall collapses to ~0 %. We judge models by F1 / per-class recall, not accuracy alone.
- **ROI is negative** for every model (best ≈ −2.9 %): the bookmaker's margin wins, exactly as theory predicts.
- **Kelly staking:** with no real edge, every staking plan loses; **full Kelly busts the bankroll fastest**, quarter-Kelly survives longest. Kelly controls *how fast you lose*, it cannot create an edge.

> **App A verdict:** even with good data, our models only **reach** the market's accuracy ceiling and cannot beat it — a strong, honest result about market efficiency.

---

# APPLICATION B — World Cup 2026 Forecast

## 6. The Data & Team Strength (App B)

| Source | What | Use |
|---|---|---|
| `international_results.csv` | **49 493** international matches, 1872→2026 (incl. the live 2026 WC) | Elo + match results |
| FIFA ranking | stops in **2024** | **unused** → we compute our own Elo |

**Elo rating (computed by us):** we walk every international match in date order; each team's rating updates after every game (bigger wins and more important matches move it more; home advantage folded in). Elo needs the **full history** to be correct, so it uses **all 49 493 matches**. Sanity check: the top of the table is **Argentina** (World Cup + Copa holders) and **Spain** (Euro 2024) — with no manual tuning.

## 7. Features, Split & Validation (App B)

**Features:** Elo difference (strongest signal), each team's Elo level, recent form (last-5 points & goal difference), and neutral-venue flag.

**Important distinction:**
- **Elo** is built from **all 49 493** matches (full history).
- **The 3 models** train only on the **modern era (2002 onward)** — *all* international games (friendlies, qualifiers, Euro, Copa, Nations League, **and** the 2002–2022 World Cups), not just World Cup games.

```
TRAIN  2002 → 2021         (17 588 matches)
VAL    2021 → Jun 2026     ( 5 683)  ← includes the 2022 World Cup
TEST   2026 group stage    (    72)  ← already played → real-world check
```
(SVM is capped to the most recent ~8 000 matches for speed; RF/XGBoost use all.)

**Validation on the real 2026 group stage:**
- Match accuracy **≈ 60 %** (matches the Elo baseline).
- **25–26 of 32** advancing teams predicted correctly (2026 format: top-2 per group + 8 best third-placed). The misses are the *bubble* third-placed teams — the hardest calls by design.

## 8. Monte Carlo — Who Wins the Cup? (App B)

A group table can use averages, but a knockout needs an actual *winner* each match. So we **simulate**: for every tie, flip a weighted coin using the model's win probability (a draw is settled like a shootout, 50/50), advance the winner, and play it through to the final. **20 000 simulations.**

| Team | Reach Final | **Win Cup** |
|---|---|---|
| **Argentina** | 35.3 % | **22.3 %** |
| Spain | 25.9 % | 15.2 % |
| France | 24.5 % | 14.1 % |
| Colombia | 11.8 % | 5.5 % |
| Brazil | 10.2 % | 5.1 % |

> **App B verdict:** Argentina ≈ 22 %, then Spain and France — consistent with the real 2026 bookmaker market, and **robust across all three models** (RF, XGBoost, SVM agree on the podium). The Round-of-32 ties are already drawn, so we also report each one head-to-head.

---

## 9. Notebook Structure

```
01_eda.ipynb              EDA: outcome distribution, home advantage (+ COVID dip),
                          goals (≈ Poisson), xG, odds → implied probability
02_feature_engineering    Build 62 leakage-safe features; validate no leakage
03_modeling               Hyper-parameter tuning; train RF/XGBoost/SVM;
                          confusion matrices, F1/recall, calibration; 3-class vs binary
04_simulation             Value-betting simulation, ROI, CLV vs Pinnacle, Kelly staking
05_worldcup               App B: Elo, train/val/test, group test, Monte Carlo bracket
```

**Code lives in `src/`** (imported by the notebooks):
`features.py` (feature building), `modeling.py` (model defs + tuning), `simulation.py` (betting + Kelly), `worldcup.py` (Elo + Monte Carlo). The `download_*`, `merge_data`, `combine_leagues`, `leagues` scripts are the **data-prep pipeline** (already run — the processed CSVs are included).

---

## 10. Evaluation — Three Dimensions

| Dimension | Metrics | Question it answers |
|---|---|---|
| **ML quality** | Confusion matrix, Accuracy, Precision, Recall, F1 | Is the model correct? |
| **Probability quality** | Calibration curve | Are the probabilities trustworthy? |
| **Real-world value** | ROI, equity curve, CLV (App A); advancement accuracy, Monte Carlo (App B) | Is the model actually useful for a decision? |

---

## 11. Discussion — Talking Points

- **Accuracy ≠ profit.** A model can match the bookmaker on accuracy yet still lose money to the margin.
- **Markets are efficient.** Closing odds price the public information; reaching the ceiling without beating it is the expected, honest result.
- **The draw / class imbalance.** Draws are rare and the hardest class — accuracy hides this, so we report F1 and recall.
- **Why XGBoost gives the best bets** — better-calibrated probabilities on noisy tabular data.
- **Same method, two decisions.** The league pipeline transfers cleanly to a live tournament via Elo + Monte Carlo.

---

## 12. Tools & Libraries

```python
pandas, numpy          # data manipulation
matplotlib, seaborn    # visualization
scikit-learn           # RF, SVM, preprocessing, metrics, calibration, splits
xgboost                # XGBoost
requests               # data download scripts (src/, already run)
```
See `requirements.txt` to install everything.

---

## 13. Roadmap / Milestones

| Step | Task | Status |
|---|---|---|
| 1 | Download & merge data — 2 leagues, 8 seasons, 6 080 matches | ✅ |
| 2 | EDA & visualization | ✅ |
| 3 | Feature engineering — 62 leakage-safe features, validated | ✅ |
| 4 | Train/val/test split + hyper-parameter tuning | ✅ |
| 5 | Train RF / XGBoost / SVM — 3-class + binary | ✅ |
| 6 | Metrics, confusion matrices, calibration | ✅ |
| 7 | Betting simulation, ROI, CLV, Kelly | ✅ |
| 8 | **Application B — World Cup 2026** (Elo + Monte Carlo) | ✅ |
| 9 | Presentation (French) & write-up | ✅ |

---

## 14. Data Sources Reference

| Source | What | Link |
|---|---|---|
| football-data.co.uk | Results + odds (Bet365, Pinnacle, …) | https://www.football-data.co.uk/data.php |
| Understat | xG per team per match | https://understat.com |
| International results | National-team matches 1872→2026 | Kaggle (international football results) |

---

## 15. One-Sentence Summary

> *We build and compare Random Forest, XGBoost, and SVM, and apply the same methodology to two real decisions: in **Application A** we predict league matches and simulate value-betting against Pinnacle's closing odds (finding the market is efficient and unbeatable), and in **Application B** we forecast the live 2026 World Cup with a self-computed Elo and a Monte Carlo of the bracket (Argentina ≈ 22 % to win, robust across all three models) — answering not just "which model is most accurate?" but "what real decision can it actually support?"*
