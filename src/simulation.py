"""
src/simulation.py  —  value-betting simulation engine.

Core idea
---------
For every match in the test set:
  1. De-vig the bookmaker's 3-way odds → implied probabilities (sum to 1).
  2. Compare model's predicted probability for each outcome vs. implied prob.
  3. If  model_prob > implied_prob + min_edge  → VALUE BET → place flat stake.
  4. Settle: win (odds-1), loss (-1), or push/refund (0, DNB only).
  5. Track cumulative profit → equity curve.

Two modes
---------
  standard  : 3-class, bet on any outcome with value (H, D, or A)
  dnb       : Draw No Bet — only bet H or A; draw result → stake refunded

Benchmark
---------
  Closing Line Value (CLV): for each bet placed, compare the odds we used
  (Bet365) against Pinnacle's closing line.  CLV > 0 means we got better
  value than the sharpest market.  Consistently positive CLV = real edge.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

LEAGUE_LABEL = {"epl": "Premier League", "laliga": "La Liga"}
OUTCOME_COLS = {
    "H": ("B365H", "PSCH"),
    "D": ("B365D", "PSCD"),
    "A": ("B365A", "PSCA"),
}


# ── de-vig helper ─────────────────────────────────────────────────────────────

def devig(row, outcomes=("H", "D", "A")):
    """Return de-vigged implied probabilities from Bet365 odds."""
    raw = {}
    for o in outcomes:
        col = OUTCOME_COLS[o][0]
        v = row.get(col, float("nan"))
        if pd.notna(v) and v > 1:
            raw[o] = 1.0 / v
    if not raw:
        return {}
    total = sum(raw.values())
    return {o: v / total for o, v in raw.items()}


# ── core simulation ───────────────────────────────────────────────────────────

def run_simulation(
    test_df: pd.DataFrame,
    proba: np.ndarray,
    le,
    stake: float = 1.0,
    min_edge: float = 0.0,
    mode: str = "standard",   # "standard" or "dnb"
) -> pd.DataFrame:
    """
    Parameters
    ----------
    test_df   : test-set DataFrame (must contain B365H/D/A, PSCH/D/A, target_3way)
    proba     : model predict_proba output, shape (n, n_classes)
    le        : fitted LabelEncoder (defines column order of proba)
    stake     : flat stake per bet (default 1 unit)
    min_edge  : minimum edge over implied prob required to place bet (default 0)
    mode      : "standard" = 3-class; "dnb" = Draw No Bet (H/A only, draw → push)

    Returns
    -------
    DataFrame with one row per bet placed, plus cumulative profit column.
    Empty DataFrame if no bets were placed.
    """
    classes = list(le.classes_)
    records = []

    for i in range(len(test_df)):
        row = test_df.iloc[i]
        actual = str(row["target_3way"])

        implied = devig(row)
        if not implied:
            continue

        # model probabilities for each outcome
        model_probs = {}
        for o in implied:
            if o in classes:
                model_probs[o] = float(proba[i, classes.index(o)])

        # skip draw bets in DNB mode
        if mode == "dnb":
            model_probs.pop("D", None)
            implied.pop("D", None)
            # Re-normalise the bookmaker's implied probs over ONLY H and A so
            # they sum to 1 — the same "no draw" world the binary model lives in.
            # Without this we'd compare the model's P(H | no draw) against the
            # bookmaker's unconditional P(H) (3-way), manufacturing a fake edge.
            tot = sum(implied.values())
            if tot > 0:
                implied = {o: p / tot for o, p in implied.items()}

        if not model_probs:
            continue

        # find the outcome with the highest positive edge
        edges = {o: model_probs[o] - implied.get(o, 1.0) for o in model_probs}
        best_o = max(edges, key=edges.get)
        best_edge = edges[best_o]

        if best_edge <= min_edge:
            continue

        # raw 3-way odds for the chosen outcome (Bet365 + Pinnacle closing)
        b365_odds = row.get(OUTCOME_COLS[best_o][0], float("nan"))
        if pd.isna(b365_odds) or b365_odds <= 1:
            continue
        pin_odds = row.get(OUTCOME_COLS[best_o][1], float("nan"))

        if mode == "dnb":
            # Convert the raw 3-way price into a realistic Draw-No-Bet price.
            # The draw refund is NOT free: a real DNB market prices it in, so the
            # DNB odds are LOWER than the raw 3-way odds. Fair DNB odds for a side
            # = 1 + odds(side) / odds(other side). Settling at the full 3-way odds
            # while ALSO refunding draws would hand the bettor free value (this was
            # the second half of the fake-profit bug).
            other = "A" if best_o == "H" else "H"
            other_b365 = row.get(OUTCOME_COLS[other][0], float("nan"))
            if pd.isna(other_b365) or other_b365 <= 1:
                continue
            bet_odds = 1.0 + b365_odds / other_b365
            other_pin = row.get(OUTCOME_COLS[other][1], float("nan"))
            pin_bet_odds = (1.0 + pin_odds / other_pin
                            if pd.notna(pin_odds) and pd.notna(other_pin)
                            and pin_odds > 1 and other_pin > 1 else float("nan"))
        else:
            bet_odds = b365_odds
            pin_bet_odds = pin_odds

        # settle the bet
        if mode == "dnb" and actual == "D":
            profit, outcome = 0.0, "push"
        elif actual == best_o:
            profit, outcome = bet_odds * stake - stake, "win"
        else:
            profit, outcome = -stake, "loss"

        # closing line value vs Pinnacle (compared on the SAME market as the bet)
        clv = (bet_odds / pin_bet_odds - 1.0) if pd.notna(pin_bet_odds) and pin_bet_odds > 1 else float("nan")

        records.append({
            "Date":       pd.to_datetime(row["Date"]),
            "HomeTeam":   row["HomeTeam"],
            "AwayTeam":   row["AwayTeam"],
            "League":     row["League"],
            "bet_on":     best_o,
            "model_prob": model_probs[best_o],
            "implied":    implied.get(best_o, float("nan")),
            "edge":       best_edge,
            "b365_odds":  bet_odds,
            "actual":     actual,
            "outcome":    outcome,
            "profit":     profit,
            "clv":        clv,
        })

    if not records:
        return pd.DataFrame()

    bets = (pd.DataFrame(records)
            .sort_values("Date")
            .reset_index(drop=True))
    bets["cumprof"] = bets["profit"].cumsum()
    bets["roi"]     = bets["cumprof"] / (stake * (bets.index + 1))
    return bets


# ── summary stats ─────────────────────────────────────────────────────────────

def sim_stats(bets: pd.DataFrame, stake: float = 1.0) -> dict:
    if bets.empty:
        return {"bets": 0}
    wins   = (bets["outcome"] == "win").sum()
    losses = (bets["outcome"] == "loss").sum()
    pushes = (bets["outcome"] == "push").sum()
    n = len(bets)
    total_staked  = stake * n
    total_profit  = bets["profit"].sum()
    roi           = total_profit / total_staked
    avg_odds      = bets["b365_odds"].mean()
    avg_clv       = bets["clv"].mean()
    return {
        "Bets":          n,
        "Wins":          wins,
        "Losses":        losses,
        "Pushes":        pushes,
        "Win rate":      f"{wins/n:.1%}",
        "Total profit":  f"{total_profit:+.1f} units",
        "ROI":           f"{roi:+.2%}",
        "Avg odds":      f"{avg_odds:.2f}",
        "Avg CLV":       f"{avg_clv:+.3f}" if not np.isnan(avg_clv) else "n/a",
    }


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_equity_curves(sim_results: dict, title: str):
    """
    sim_results : {model_name: bets_df}
    """
    colors = {"Random Forest": "#1f77b4", "XGBoost": "#ff7f0e", "SVM": "#2ca02c"}
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)

    for name, bets in sim_results.items():
        if bets.empty:
            ax.plot([], [], label=f"{name} (no bets)")
            continue
        ax.plot(range(len(bets)), bets["cumprof"],
                label=f"{name}  ROI={bets['roi'].iloc[-1]:+.1%}",
                color=colors.get(name, "grey"), lw=1.8)

    ax.set_xlabel("bet number (chronological)")
    ax.set_ylabel("cumulative profit (units, flat stake=1)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_bets_per_outcome(sim_results: dict, title: str):
    rows = []
    for name, bets in sim_results.items():
        if bets.empty:
            continue
        for o, grp in bets.groupby("bet_on"):
            rows.append({"Model": name, "Bet on": o,
                         "Count": len(grp),
                         "ROI": grp["profit"].sum() / len(grp)})
    if not rows:
        return
    df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    df.pivot(index="Model", columns="Bet on", values="Count").plot(
        kind="bar", ax=axes[0], rot=0)
    axes[0].set_title(f"{title} — bets placed per outcome")
    axes[0].set_ylabel("number of bets")

    df.pivot(index="Model", columns="Bet on", values="ROI").plot(
        kind="bar", ax=axes[1], rot=0, color=["#2ca02c", "#7f7f7f", "#d62728"])
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_title(f"{title} — ROI per outcome type")
    axes[1].set_ylabel("ROI (unit profit per bet)")
    plt.tight_layout()
    plt.show()


def plot_per_league(sim_results: dict, title: str):
    rows = []
    for name, bets in sim_results.items():
        if bets.empty:
            continue
        for lg, grp in bets.groupby("League"):
            n = len(grp)
            roi = grp["profit"].sum() / n if n else float("nan")
            rows.append({"Model": name, "League": LEAGUE_LABEL.get(lg, lg), "ROI": roi})
    if not rows:
        return
    pvt = pd.DataFrame(rows).pivot(index="Model", columns="League", values="ROI")
    fig, ax = plt.subplots(figsize=(8, 4))
    pvt.plot(kind="bar", ax=ax, rot=0)
    ax.axhline(0, color="black", lw=0.8, ls="--")
    ax.set_title(f"{title} — ROI by league")
    ax.set_ylabel("ROI (profit per bet)")
    plt.tight_layout()
    plt.show()
    return pvt


# ── Kelly staking & bankroll management ──────────────────────────────────────
#
# The flat-stake sim above answers "is there an edge?".  Staking answers a
# second question: "given an edge, how much should you bet, and how rough is
# the ride?".  We size each bet as a fraction of the CURRENT bankroll (so every
# strategy compounds and is directly comparable on growth and drawdown).

def kelly_fraction(p: float, odds: float) -> float:
    """Optimal Kelly fraction of bankroll for a single bet.

    p    : model probability the bet wins, expressed in the bet's OWN market
           (3-way prob in standard mode, binary prob in DNB mode).
    odds : decimal odds actually taken for that bet (already DNB-adjusted in
           dnb mode, so `b = odds - 1` is the correct net return).

    f* = (b·p − q) / b = p − q/b   with q = 1 − p.

    Pushes (DNB draws) do NOT change f*: a refunded stake neither grows nor
    shrinks the bankroll, so the standard formula applied with the model's
    conditional (no-draw) win probability is exact. Returns 0 when the edge is
    non-positive — i.e. Kelly simply declines to bet.
    """
    b = odds - 1.0
    if b <= 0 or not np.isfinite(p):
        return 0.0
    f = p - (1.0 - p) / b
    return max(0.0, f)


def apply_staking(bets: pd.DataFrame, strategy: str = "flat",
                  bankroll0: float = 100.0, flat_frac: float = 0.02,
                  kelly_mult: float = 1.0, cap: float = 0.10) -> pd.DataFrame:
    """Replay already-placed bets under a bankroll-based staking rule.

    strategy : "flat"  → stake `flat_frac` of the current bankroll every bet
               "kelly" → stake `kelly_mult · f*` of the current bankroll
    kelly_mult : 1.0 = full Kelly, 0.5 = half-Kelly, 0.25 = quarter-Kelly
    cap        : hard ceiling on any single stake as a fraction of bankroll
                 (protects against over-betting when probabilities are off)

    Returns a copy of `bets` with `stake`, `bankroll` and `drawdown` columns.
    """
    out = bets.copy().reset_index(drop=True)
    if out.empty:
        return out

    bankroll = bankroll0
    peak = bankroll0
    stakes, banks, dds = [], [], []

    for r in out.itertuples():
        if strategy == "flat":
            frac = flat_frac
        else:  # kelly
            frac = kelly_mult * kelly_fraction(r.model_prob, r.b365_odds)
        frac = min(max(frac, 0.0), cap)
        stake = frac * bankroll

        # per-unit return of the bet: win → odds−1, loss → −1, push → 0
        if r.outcome == "win":
            unit_ret = r.b365_odds - 1.0
        elif r.outcome == "loss":
            unit_ret = -1.0
        else:  # push (DNB draw)
            unit_ret = 0.0

        bankroll += stake * unit_ret
        peak = max(peak, bankroll)
        stakes.append(stake)
        banks.append(bankroll)
        dds.append(bankroll / peak - 1.0)

    out["stake"] = stakes
    out["bankroll"] = banks
    out["drawdown"] = dds
    return out


def staking_metrics(staked: pd.DataFrame, bankroll0: float = 100.0) -> dict:
    """Risk/return summary for one staked equity curve."""
    if staked.empty or "bankroll" not in staked:
        return {}
    final = float(staked["bankroll"].iloc[-1])
    total_return = final / bankroll0 - 1.0
    max_dd = float(staked["drawdown"].min())          # most negative point

    bvals = np.concatenate([[bankroll0], staked["bankroll"].values])
    step_returns = np.diff(bvals) / bvals[:-1]
    vol = float(np.std(step_returns))
    calmar = (total_return / abs(max_dd)) if max_dd < 0 else float("nan")
    busted = bool((staked["bankroll"] <= bankroll0 * 0.05).any())

    return {
        "Final bankroll": round(final, 1),
        "Total return":   f"{total_return:+.1%}",
        "Max drawdown":   f"{max_dd:.1%}",
        "Calmar":         round(calmar, 2) if np.isfinite(calmar) else float("nan"),
        "Volatility":     round(vol, 4),
        "Busted (<5%)":   busted,
    }


# the staking ladder we compare in the notebook
STAKING_LADDER = {
    "Flat 2%":        dict(strategy="flat",  flat_frac=0.02),
    "Full Kelly":     dict(strategy="kelly", kelly_mult=1.0),
    "Half Kelly":     dict(strategy="kelly", kelly_mult=0.5),
    "Quarter Kelly":  dict(strategy="kelly", kelly_mult=0.25),
}


def compare_staking(bets: pd.DataFrame, bankroll0: float = 100.0,
                    cap: float = 0.10, ladder: dict = None) -> tuple:
    """Run every staking strategy on one model's bets.

    Returns (metrics_df, {strategy_name: staked_df}) so the caller can both
    tabulate the risk metrics and plot the bankroll curves.
    """
    ladder = ladder or STAKING_LADDER
    curves, rows = {}, []
    for name, kw in ladder.items():
        staked = apply_staking(bets, bankroll0=bankroll0, cap=cap, **kw)
        curves[name] = staked
        m = staking_metrics(staked, bankroll0)
        m["Strategy"] = name
        rows.append(m)
    metrics = pd.DataFrame(rows).set_index("Strategy")
    return metrics, curves


def plot_bankrolls(curves: dict, title: str, bankroll0: float = 100.0):
    """Plot bankroll-over-time for each staking strategy (one model)."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axhline(bankroll0, color="black", lw=0.8, ls="--", alpha=0.5,
               label=f"start ({bankroll0:.0f})")
    for name, staked in curves.items():
        if staked.empty:
            continue
        ax.plot(range(len(staked)), staked["bankroll"], lw=1.7, label=name)
    ax.set_xlabel("bet number (chronological)")
    ax.set_ylabel("bankroll (units)")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()
