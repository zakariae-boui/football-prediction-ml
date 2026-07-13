"""
src/worldcup.py  —  World Cup forecasting (Application B).

Step 1 (this file, for now): a leakage-safe **Elo rating engine** for national
teams, computed from the full international match history (1872 → 2026).

Why Elo and not the FIFA ranking?
---------------------------------
The FIFA ranking CSV stops in 2024, so it cannot rate teams for the 2026 World
Cup. Elo is self-contained: it updates after every match, so it stays current
right up to today's games. It is also the method behind eloratings.net and most
serious international-football models.

How Elo works (World Football Elo, the standard variant)
--------------------------------------------------------
Every team has a rating (new teams start at 1500). For each match:

    expected_home = 1 / (1 + 10 ** (-(elo_home - elo_away + ha) / 400))

where `ha` is a home-advantage bonus (0 at a neutral venue). After the match:

    elo_home += K * G * (result_home - expected_home)      # result: 1/0.5/0
    elo_away -= same amount                                # zero-sum

  * K  — match importance (World Cup final game > friendly).
  * G  — goal-difference multiplier (a 4-0 moves the rating more than a 1-0).

Leakage safety
--------------
For every match we store the ratings **as they were BEFORE kickoff**
(`home_elo_pre`, `away_elo_pre`) and only update *after* recording them, so a
match never sees its own result. These pre-match ratings are the model features.
"""

from pathlib import Path

import numpy as np
import pandas as pd

WC_DIR = Path(__file__).resolve().parent.parent / "data" / "worldcup"
RESULTS_CSV = WC_DIR / "international_results.csv"

START_ELO = 1500.0       # rating for a team's first-ever appearance
HOME_ADVANTAGE = 100.0   # Elo points added to the home side at a non-neutral venue


# ── match-importance weight (K) ──────────────────────────────────────────────

def match_weight(tournament: str) -> float:
    """K factor by competition importance (eloratings-style)."""
    t = str(tournament).lower()
    if "world cup" in t and "qualif" not in t:
        return 60.0                       # World Cup finals
    if any(k in t for k in ("euro", "copa am", "african cup", "asian cup",
                            "gold cup", "confederations")) and "qualif" not in t:
        return 50.0                       # continental finals
    if "nations league" in t:
        return 45.0
    if "qualif" in t:
        return 40.0                       # any qualifier
    if "friendly" in t:
        return 20.0
    return 30.0                           # other competitive matches


def _goal_multiplier(margin: int) -> float:
    """G: bigger wins move the rating more (capped, eloratings formula)."""
    if margin <= 1:
        return 1.0
    if margin == 2:
        return 1.5
    return (11.0 + margin) / 8.0


# ── core Elo computation ─────────────────────────────────────────────────────

def load_results(path: Path = RESULTS_CSV) -> pd.DataFrame:
    """Load and clean the international results file."""
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df["neutral"] = df["neutral"].astype(bool)
    for c in ("home_score", "away_score"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.sort_values("date").reset_index(drop=True)


def compute_elo(df: pd.DataFrame, start_elo: float = START_ELO,
                home_adv: float = HOME_ADVANTAGE):
    """Walk matches in date order, attaching pre-match Elo and updating after.

    Returns
    -------
    out     : df + columns home_elo_pre, away_elo_pre, elo_diff (home-away,
              home advantage folded in), plus exp_home (model-free Elo forecast).
    ratings : {team: final_elo}  — current strength of every team.
    """
    df = df.sort_values("date").reset_index(drop=True)
    ratings: dict[str, float] = {}

    home_pre = np.empty(len(df))
    away_pre = np.empty(len(df))
    exp_home = np.empty(len(df))

    for i, row in enumerate(df.itertuples()):
        h, a = row.home_team, row.away_team
        rh = ratings.get(h, start_elo)
        ra = ratings.get(a, start_elo)
        home_pre[i] = rh
        away_pre[i] = ra

        ha = 0.0 if row.neutral else home_adv
        we_home = 1.0 / (1.0 + 10 ** (-(rh - ra + ha) / 400.0))
        exp_home[i] = we_home

        # only PLAYED matches (valid scores) update the ratings
        if np.isnan(row.home_score) or np.isnan(row.away_score):
            continue

        hs, as_ = int(row.home_score), int(row.away_score)
        result = 1.0 if hs > as_ else (0.5 if hs == as_ else 0.0)
        k = match_weight(row.tournament)
        g = _goal_multiplier(abs(hs - as_))
        delta = k * g * (result - we_home)

        ratings[h] = rh + delta
        ratings[a] = ra - delta

    out = df.copy()
    out["home_elo_pre"] = home_pre
    out["away_elo_pre"] = away_pre
    out["elo_diff"] = (home_pre - away_pre) + np.where(out["neutral"], 0.0, home_adv)
    out["exp_home"] = exp_home      # Elo's own (model-free) home-win expectation
    return out, ratings


def current_ratings(path: Path = RESULTS_CSV) -> pd.Series:
    """Convenience: final Elo of every team, highest first."""
    _, ratings = compute_elo(load_results(path))
    return pd.Series(ratings, name="elo").sort_values(ascending=False)


# ── recent-form features (leakage-safe) ──────────────────────────────────────

FORM_WINDOW = 5


def add_form_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add each team's recent form (avg points & goal-diff over its last 5
    matches, computed with shift(1) so the current game is excluded).

    Built on a team-perspective long table, then folded back to one row per
    match as home_form_* / away_form_*.
    """
    df = df.sort_values("date").reset_index(drop=True)
    df["match_id"] = df.index

    played = df.dropna(subset=["home_score", "away_score"]).copy()

    # explode to two rows per match (home view + away view)
    h = played[["match_id", "date", "home_team", "home_score", "away_score"]].copy()
    h.columns = ["match_id", "date", "team", "gf", "ga"]
    a = played[["match_id", "date", "away_team", "away_score", "home_score"]].copy()
    a.columns = ["match_id", "date", "team", "gf", "ga"]
    long = pd.concat([h, a], ignore_index=True).sort_values(["team", "date"])

    long["pts"] = np.where(long.gf > long.ga, 3, np.where(long.gf == long.ga, 1, 0))
    long["gd"] = long.gf - long.ga

    def roll(s):
        return s.shift(1).rolling(FORM_WINDOW, min_periods=1).mean()

    long["form_pts"] = long.groupby("team")["pts"].transform(roll)
    long["form_gd"] = long.groupby("team")["gd"].transform(roll)

    # map back to home / away of each match
    fp = long.set_index(["match_id", "team"])[["form_pts", "form_gd"]]
    out = df.copy()
    hi = out.set_index(["match_id", "home_team"]).index
    ai = out.set_index(["match_id", "away_team"]).index
    out["home_form_pts"] = fp["form_pts"].reindex(hi).values
    out["home_form_gd"] = fp["form_gd"].reindex(hi).values
    out["away_form_pts"] = fp["form_pts"].reindex(ai).values
    out["away_form_gd"] = fp["form_gd"].reindex(ai).values
    return out


# ── modeling table ───────────────────────────────────────────────────────────

WC_FEATURES = [
    "elo_diff", "home_elo_pre", "away_elo_pre", "neutral_int",
    "home_form_pts", "away_form_pts", "home_form_gd", "away_form_gd",
    "form_pts_diff", "form_gd_diff",
]


def build_model_table(path: Path = RESULTS_CSV):
    """Full pipeline: load → Elo → form → features → 3-class target.

    Returns
    -------
    table   : one row per match with WC_FEATURES + `result` (H/D/A) target.
              Includes unplayed (NA) rows too — their features are valid even
              though `result` is NaN (these are the matches we will predict).
    ratings : {team: current Elo}
    """
    df = load_results(path)
    df, ratings = compute_elo(df)
    df = add_form_features(df)

    df["neutral_int"] = df["neutral"].astype(int)
    df["form_pts_diff"] = df["home_form_pts"] - df["away_form_pts"]
    df["form_gd_diff"] = df["home_form_gd"] - df["away_form_gd"]

    df["result"] = pd.NA
    played = df["home_score"].notna() & df["away_score"].notna()
    df.loc[played & (df.home_score > df.away_score), "result"] = "H"
    df.loc[played & (df.home_score == df.away_score), "result"] = "D"
    df.loc[played & (df.home_score < df.away_score), "result"] = "A"
    return df, ratings


# ── modeling: train / validation / test ──────────────────────────────────────

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import xgboost as xgb

MODEL_START = "2002-01-01"   # use the modern era for training
TRAIN_END   = "2021-01-01"   # TRAIN  : MODEL_START .. TRAIN_END
WC_START    = "2026-06-11"   # VAL    : TRAIN_END .. WC_START  (incl. 2022 WC, 2024 Euro/Copa)
KO_START    = "2026-06-28"   # TEST   : 2026 group stage only (first R32 match is Jun 28)
CLASSES     = np.array(["A", "D", "H"])   # fixed label order


def wc_split(table: pd.DataFrame):
    """Time-ordered split. Returns train, val, test(=2026 group stage), dev."""
    m = table[table["result"].notna() & (table["date"] >= MODEL_START)].copy()
    train = m[m["date"] < TRAIN_END]
    val   = m[(m["date"] >= TRAIN_END) & (m["date"] < WC_START)]
    test  = m[(m["date"] >= WC_START) & (m["date"] < KO_START)]   # group stage
    dev   = m[m["date"] < WC_START]                               # train + val
    return train, val, test, dev


def _model_defs(params: dict = None) -> dict:
    params = params or {}
    rf  = dict(n_estimators=300, max_depth=8, min_samples_leaf=3,
               random_state=42, n_jobs=-1)
    xgbp = dict(n_estimators=300, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0, eval_metric="mlogloss")
    svmp = dict(C=1.0, kernel="rbf", probability=True, random_state=42)
    rf.update(params.get("Random Forest", {}))
    xgbp.update(params.get("XGBoost", {}))
    svmp.update(params.get("SVM", {}))
    return {"Random Forest": RandomForestClassifier(**rf),
            "XGBoost": xgb.XGBClassifier(**xgbp),
            "SVM": SVC(**svmp)}


def train_wc_models(train_frame: pd.DataFrame, params: dict = None,
                    max_train: int = None):
    """Fit imputer+scaler on the train frame, then fit all three models.
    Returns (models, pipe, le).

    max_train : if set, train on only the most recent `max_train` matches.
    SVM with an RBF kernel scales poorly (~O(n²)); capping to the most recent
    ~8000 internationals keeps it fast and uses the most relevant games, while
    RF/XGBoost are happy either way."""
    tf = train_frame
    if max_train is not None and len(tf) > max_train:
        tf = tf.sort_values("date").iloc[-max_train:]

    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")),
                     ("scaler",  StandardScaler())])
    X = pipe.fit_transform(tf[WC_FEATURES].values)
    le = LabelEncoder().fit(CLASSES)
    y = le.transform(tf["result"].astype(str).values)
    models = {}
    for name, clf in _model_defs(params).items():
        clf.fit(X, y)
        models[name] = clf
    return models, pipe, le


def proba_for(models: dict, frame: pd.DataFrame, pipe, le) -> dict:
    """{model_name: DataFrame[A,D,H]} of predicted probabilities for `frame`."""
    X = pipe.transform(frame[WC_FEATURES].values)
    cols = list(le.classes_)
    return {name: pd.DataFrame(clf.predict_proba(X), columns=cols,
                               index=frame.index)
            for name, clf in models.items()}


# ── group-stage advancement check (your requested validation) ────────────────

def _reconstruct_groups(group_matches: pd.DataFrame) -> list:
    """Recover the 12 groups as connected components of the group-stage graph
    (teams that meet in the group stage share a group)."""
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root
    for r in group_matches.itertuples():
        parent[find(r.home_team)] = find(r.away_team)
    groups = {}
    teams = set(group_matches.home_team) | set(group_matches.away_team)
    for t in teams:
        groups.setdefault(find(t), []).append(t)
    return list(groups.values())


def advancement_accuracy(group_matches: pd.DataFrame, proba: pd.DataFrame,
                         ratings: dict, top_n: int = 2):
    """Compare the model's predicted top-`top_n` per group against reality.

    Predicted points use the model's expected points per match
    (3·P(win) + 1·P(draw)); actual points use the real results. Returns
    (n_correct, n_total, details_df)."""
    gm = group_matches.copy()
    # actual points / goal-diff
    act = {}
    for r in gm.itertuples():
        ph = 3 if r.home_score > r.away_score else (1 if r.home_score == r.away_score else 0)
        pa = 3 if r.away_score > r.home_score else (1 if r.home_score == r.away_score else 0)
        for team, pts, gd in [(r.home_team, ph, r.home_score - r.away_score),
                              (r.away_team, pa, r.away_score - r.home_score)]:
            d = act.setdefault(team, {"pts": 0, "gd": 0})
            d["pts"] += pts; d["gd"] += gd

    # predicted expected points
    pred = {}
    for idx, r in zip(gm.index, gm.itertuples()):
        pH, pD, pA = proba.loc[idx, "H"], proba.loc[idx, "D"], proba.loc[idx, "A"]
        for team, ep in [(r.home_team, 3 * pH + pD), (r.away_team, 3 * pA + pD)]:
            pred[team] = pred.get(team, 0.0) + ep

    rows, correct, total = [], 0, 0
    for grp in _reconstruct_groups(gm):
        actual_rank = sorted(grp, key=lambda t: (act[t]["pts"], act[t]["gd"]),
                             reverse=True)
        pred_rank = sorted(grp, key=lambda t: (pred[t], ratings.get(t, 1500)),
                           reverse=True)
        a_top, p_top = set(actual_rank[:top_n]), set(pred_rank[:top_n])
        hit = len(a_top & p_top)
        correct += hit; total += top_n
        rows.append({"group": "/".join(sorted(grp)),
                     "actual_top2": ", ".join(actual_rank[:top_n]),
                     "pred_top2": ", ".join(pred_rank[:top_n]),
                     "correct": hit})
    return correct, total, pd.DataFrame(rows)


def _expected_points(group_matches: pd.DataFrame, proba: pd.DataFrame) -> dict:
    """Sum of each team's expected group points (3·P(win)+1·P(draw))."""
    pts = {}
    for idx, r in zip(group_matches.index, group_matches.itertuples()):
        pH, pD, pA = proba.loc[idx, "H"], proba.loc[idx, "D"], proba.loc[idx, "A"]
        pts[r.home_team] = pts.get(r.home_team, 0.0) + 3 * pH + pD
        pts[r.away_team] = pts.get(r.away_team, 0.0) + 3 * pA + pD
    return pts


def predict_advancers_32(group_matches: pd.DataFrame, proba: pd.DataFrame,
                         ratings: dict) -> set:
    """The 32 teams the model thinks advance under the 2026 format:
    top 2 of every group (24) + the 8 best third-placed teams (8).
    Ranking uses expected points, ties broken by Elo."""
    pred = _expected_points(group_matches, proba)
    key = lambda t: (pred.get(t, 0.0), ratings.get(t, 1500))

    advancers, thirds = set(), []
    for grp in _reconstruct_groups(group_matches):
        rank = sorted(grp, key=key, reverse=True)
        advancers.update(rank[:2])      # top 2 qualify
        if len(rank) >= 3:
            thirds.append(rank[2])      # 3rd-place candidate
    best_thirds = sorted(thirds, key=key, reverse=True)[:8]
    advancers.update(best_thirds)
    return advancers


def actual_knockout_teams(table: pd.DataFrame) -> set:
    """Ground truth: the 32 teams that really reached the Round of 32 =
    every team appearing in the first knockout round in the data."""
    ko = table[(table["tournament"] == "FIFA World Cup") &
               (table["date"] >= KO_START)]
    return set(ko["home_team"]) | set(ko["away_team"])


# ── group-stage Monte Carlo (advancement probabilities) ──────────────────────

def simulate_groups(group_matches: pd.DataFrame, proba: pd.DataFrame,
                    ratings: dict, n_sims: int = 10000, seed: int = 42):
    """Play the group stage `n_sims` times using the model's match
    probabilities, applying the 2026 rule (top 2 of each group + 8 best
    third-placed teams advance). Tie-breaks use Elo (we model results, not
    goals, so we approximate goal-difference tie-breaks by team strength).

    Returns {team: probability of reaching the Round of 32}.
    """
    rng = np.random.default_rng(seed)
    groups = _reconstruct_groups(group_matches)
    teams = sorted(set(group_matches.home_team) | set(group_matches.away_team))
    elo = {t: ratings.get(t, 1500) for t in teams}

    # pre-sample every match's outcome for all simulations at once
    pairs, outcomes = [], []
    for idx, r in zip(group_matches.index, group_matches.itertuples()):
        p = proba.loc[idx, ["H", "D", "A"]].to_numpy(float)
        p = p / p.sum()
        pairs.append((r.home_team, r.away_team))
        outcomes.append(rng.choice(3, size=n_sims, p=p))   # 0=H,1=D,2=A
    outcomes = np.array(outcomes)

    adv = {t: 0 for t in teams}
    for s in range(n_sims):
        pts = {t: 0 for t in teams}
        col = outcomes[:, s]
        for (h, a), o in zip(pairs, col):
            if o == 0:
                pts[h] += 3
            elif o == 1:
                pts[h] += 1; pts[a] += 1
            else:
                pts[a] += 3
        thirds = []
        for grp in groups:
            rank = sorted(grp, key=lambda t: (pts[t], elo[t]), reverse=True)
            adv[rank[0]] += 1; adv[rank[1]] += 1
            thirds.append(rank[2])
        for t in sorted(thirds, key=lambda t: (pts[t], elo[t]), reverse=True)[:8]:
            adv[t] += 1

    return {t: adv[t] / n_sims for t in teams}


def group_tables(group_matches: pd.DataFrame, proba: pd.DataFrame,
                 ratings: dict, adv_prob: dict = None) -> list:
    """One tidy standings DataFrame per group, ranked by expected points
    (3·P(win)+1·P(draw)). Adds an advance_% column if `adv_prob` is given."""
    exp_pts = _expected_points(group_matches, proba)
    out = []
    for grp in _reconstruct_groups(group_matches):
        rows = []
        for t in grp:
            row = {"team": t, "xPoints": round(exp_pts.get(t, 0.0), 2)}
            if adv_prob is not None:
                row["advance_%"] = round(100 * adv_prob.get(t, 0.0), 1)
            rows.append(row)
        sort_col = "advance_%" if adv_prob is not None else "xPoints"
        out.append(pd.DataFrame(rows).sort_values(sort_col, ascending=False)
                   .reset_index(drop=True))
    return out


# ── knockout bracket (decoded from the official 2026 bracket) ─────────────────
import re

# Each match: (slotA, slotB). A slot is a team name (Round of 32) or "W<id>"
# meaning "winner of match M<id>".
BRACKET = {
    # Round of 32
    "M73": ("South Africa", "Canada"),            # already played -> Canada
    "M74": ("Germany", "Paraguay"),
    "M75": ("Netherlands", "Morocco"),
    "M76": ("Brazil", "Japan"),
    "M77": ("France", "Sweden"),
    "M78": ("Ivory Coast", "Norway"),
    "M79": ("Mexico", "Ecuador"),
    "M80": ("England", "DR Congo"),
    "M81": ("United States", "Bosnia and Herzegovina"),
    "M82": ("Belgium", "Senegal"),
    "M83": ("Portugal", "Croatia"),
    "M84": ("Spain", "Austria"),
    "M85": ("Switzerland", "Algeria"),
    "M86": ("Argentina", "Cape Verde"),
    "M87": ("Colombia", "Ghana"),
    "M88": ("Australia", "Egypt"),
    # Round of 16
    "M89": ("W74", "W77"), "M90": ("W73", "W75"),
    "M91": ("W76", "W78"), "M92": ("W79", "W80"),
    "M93": ("W83", "W84"), "M94": ("W81", "W82"),
    "M95": ("W86", "W88"), "M96": ("W85", "W87"),
    # Quarter-finals
    "M97": ("W89", "W90"), "M98": ("W93", "W94"),
    "M99": ("W91", "W92"), "M100": ("W95", "W96"),
    # Semi-finals
    "M101": ("W97", "W98"), "M102": ("W99", "W100"),
    # Final
    "M104": ("W101", "W102"),
}
PLAYED = {"M73": "Canada"}     # knockout results already known, locked in
ROUNDS = {
    "R32": ["M73", "M74", "M75", "M76", "M77", "M78", "M79", "M80",
            "M81", "M82", "M83", "M84", "M85", "M86", "M87", "M88"],
    "R16": ["M89", "M90", "M91", "M92", "M93", "M94", "M95", "M96"],
    "QF":  ["M97", "M98", "M99", "M100"],
    "SF":  ["M101", "M102"],
    "F":   ["M104"],
}
KO_TEAMS = sorted({t for m in ROUNDS["R32"] for t in BRACKET[m]})


# ── "given any two teams, who advances?" ─────────────────────────────────────

def current_form(table: pd.DataFrame, window: int = FORM_WINDOW) -> dict:
    """Each team's form (avg points, avg goal-diff) over its last `window`
    PLAYED matches — the state going INTO the knockout."""
    from collections import defaultdict
    played = table.dropna(subset=["home_score", "away_score"]).sort_values("date")
    hist = defaultdict(list)
    for r in played.itertuples():
        gd = r.home_score - r.away_score
        hist[r.home_team].append((3 if gd > 0 else (1 if gd == 0 else 0), gd))
        hist[r.away_team].append((3 if gd < 0 else (1 if gd == 0 else 0), -gd))
    return {t: (float(np.mean([p for p, _ in lst[-window:]])),
               float(np.mean([g for _, g in lst[-window:]])))
            for t, lst in hist.items()}


def _feat_row(A: str, B: str, ratings: dict, form: dict) -> dict:
    """Feature row for a NEUTRAL-venue match, team A as 'home' slot."""
    ea, eb = ratings.get(A, 1500), ratings.get(B, 1500)
    fa = form.get(A, (1.0, 0.0)); fb = form.get(B, (1.0, 0.0))
    return {"elo_diff": ea - eb, "home_elo_pre": ea, "away_elo_pre": eb,
            "neutral_int": 1,
            "home_form_pts": fa[0], "away_form_pts": fb[0],
            "home_form_gd": fa[1], "away_form_gd": fb[1],
            "form_pts_diff": fa[0] - fb[0], "form_gd_diff": fa[1] - fb[1]}


def advance_matrix(model, pipe, le, teams, ratings, form) -> dict:
    """P(A beats B) for every ordered pair, at a neutral venue. A draw is
    resolved like a penalty shootout (50/50). Home-slot bias is removed by
    averaging both orientations."""
    rows, pairs = [], []
    for A in teams:
        for B in teams:
            if A != B:
                rows.append(_feat_row(A, B, ratings, form)); pairs.append((A, B))
    X = pipe.transform(pd.DataFrame(rows)[WC_FEATURES].values)
    pr = model.predict_proba(X)
    iH, iD = list(le.classes_).index("H"), list(le.classes_).index("D")
    raw = {pair: pr[i][iH] + 0.5 * pr[i][iD] for i, pair in enumerate(pairs)}
    return {(A, B): 0.5 * (raw[(A, B)] + (1 - raw[(B, A)]))
            for A in teams for B in teams if A != B}


def _resolve(slot, winners):
    return winners["M" + slot[1:]] if re.fullmatch(r"W\d+", slot) else slot


def simulate_knockout(adv: dict, n_sims: int = 10000, seed: int = 42):
    """Play the bracket `n_sims` times. Returns a DataFrame: for each team,
    the probability of reaching each round and of winning the cup."""
    from collections import defaultdict
    rng = np.random.default_rng(seed)
    reach = defaultdict(lambda: defaultdict(int))   # team -> round -> count
    champ = defaultdict(int)
    order = ["R32", "R16", "QF", "SF", "F"]

    for _ in range(n_sims):
        W = {}
        for rnd in order:
            for mid in ROUNDS[rnd]:
                a = _resolve(BRACKET[mid][0], W)
                b = _resolve(BRACKET[mid][1], W)
                reach[a][rnd] += 1; reach[b][rnd] += 1
                if mid in PLAYED:
                    w = PLAYED[mid]
                else:
                    w = a if rng.random() < adv[(a, b)] else b
                W[mid] = w
                if rnd == "F":
                    champ[w] += 1

    rows = []
    for t in KO_TEAMS:
        rows.append({
            "team": t,
            "reach_R16": 100 * reach[t]["R16"] / n_sims,
            "reach_QF":  100 * reach[t]["QF"] / n_sims,
            "reach_SF":  100 * reach[t]["SF"] / n_sims,
            "reach_Final": 100 * reach[t]["F"] / n_sims,
            "win_cup":   100 * champ[t] / n_sims,
        })
    return (pd.DataFrame(rows).sort_values("win_cup", ascending=False)
            .reset_index(drop=True).round(1))


def round32_probabilities(adv: dict) -> pd.DataFrame:
    """Head-to-head advance probability for each of the 16 Round-of-32 ties
    (the only round where both teams are already known)."""
    rows = []
    for mid in ROUNDS["R32"]:
        a, b = BRACKET[mid]
        pa = round(float(100 * adv[(a, b)]), 1)   # float() avoids float32 display noise
        rows.append({"match": mid, "team_A": a, "win_A_%": pa,
                     "win_B_%": round(100 - pa, 1), "team_B": b,
                     "favorite": a if pa >= 50 else b,
                     "note": f"played: {PLAYED[mid]} won" if mid in PLAYED else ""})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    s = current_ratings()
    print("Top 20 national teams by Elo (as of latest match in the data):\n")
    print(s.head(20).round(0).to_string())
