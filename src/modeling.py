"""
src/modeling.py  —  reusable ML pipeline for Step 4.

Functions cover:
  preprocess()          load, drop NaN, impute, split, scale
  get_model_defs()      fresh RF / XGBoost / SVM instances
  train_all()           fit every model, return dict
  evaluate()            accuracy + full classification_report per model
  metrics_table()       DataFrame summary
  plot_cms()            3 confusion matrices side-by-side
  plot_per_league()     bar chart of accuracy by league
  plot_importances()    top-N features for RF and XGBoost
  plot_calibration()    calibration curves for P(home win)
"""

import sys
import json
import itertools
from copy import deepcopy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.calibration import CalibrationDisplay
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import xgboost as xgb

sys.path.append(str(Path(__file__).parent))
from features import feature_columns

PROC = Path(__file__).resolve().parent.parent / "data" / "processed"

# Time-ordered splits (never random — sports data is a time series).
#   TRAIN      : the model learns here.
#   VALIDATION : we pick the best hyperparameters here, WITHOUT touching test.
#   TEST       : the final score, untouched until the very end.
# The final models are refit on DEV (= train + validation) before the test
# evaluation, so no pre-test data is wasted.
TRAIN_SEASONS = ["1819", "1920", "2021", "2122"]   # 4 seasons — fit candidates
VAL_SEASONS   = ["2223"]                            # 1 season  — tune hyperparameters
TEST_SEASONS  = ["2324", "2425", "2526"]            # 3 seasons — report once
DEV_SEASONS   = TRAIN_SEASONS + VAL_SEASONS         # train+val — final fit
LEAGUE_LABEL  = {"epl": "Premier League", "laliga": "La Liga"}

# Tuned hyperparameters are persisted here so BOTH notebooks (modeling +
# simulation) reuse the exact same winners without re-tuning.
BEST_PARAMS_PATH = PROC / "best_params.json"


# ── model definitions ────────────────────────────────────────────────────────

def _load_best_params() -> dict:
    """Load tuned hyperparameters if tune_models() has been run, else {}."""
    if BEST_PARAMS_PATH.exists():
        with open(BEST_PARAMS_PATH) as f:
            return json.load(f)
    return {}


def get_model_defs(params: dict = None) -> dict:
    """Return a fresh (unfitted) copy of each model.

    Hyperparameters: start from sensible defaults, then overlay any tuned
    values. If `params` is None we auto-load best_params.json (written by
    tune_models), so once tuning has run every caller uses the tuned models.
    """
    params = params if params is not None else _load_best_params()

    rf  = dict(n_estimators=300, max_depth=10, min_samples_leaf=3,
               random_state=42, n_jobs=-1)
    xgbp = dict(n_estimators=300, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                random_state=42, verbosity=0, eval_metric="mlogloss")
    svmp = dict(C=1.0, kernel="rbf", probability=True, random_state=42)

    rf.update(params.get("Random Forest", {}))
    xgbp.update(params.get("XGBoost", {}))
    svmp.update(params.get("SVM", {}))

    return {
        "Random Forest": RandomForestClassifier(**rf),
        "XGBoost":       xgb.XGBClassifier(**xgbp),
        "SVM":           SVC(**svmp),
    }


# ── hyperparameter tuning on the validation set ──────────────────────────────

# Small, readable grids — enough to show real tuning without a long runtime.
PARAM_GRID = {
    "Random Forest": {"max_depth": [6, 10, 16], "min_samples_leaf": [1, 3]},
    "XGBoost":       {"max_depth": [3, 4, 6], "learning_rate": [0.03, 0.05, 0.1]},
    "SVM":           {"C": [0.5, 1.0, 5.0], "gamma": ["scale", 0.1]},
}


def _build_candidate(name: str, override: dict):
    """A model of the given family with base params + grid overrides.
    SVM uses probability=False here (faster) — we only need predictions to
    score validation accuracy; the final model re-enables probabilities."""
    if name == "Random Forest":
        p = dict(n_estimators=300, random_state=42, n_jobs=-1); p.update(override)
        return RandomForestClassifier(**p)
    if name == "XGBoost":
        p = dict(n_estimators=300, subsample=0.8, colsample_bytree=0.8,
                 random_state=42, verbosity=0, eval_metric="mlogloss"); p.update(override)
        return xgb.XGBClassifier(**p)
    if name == "SVM":
        p = dict(kernel="rbf", probability=False, random_state=42); p.update(override)
        return SVC(**p)
    raise ValueError(name)


def tune_models(df: pd.DataFrame, feat_cols: list,
                target_col: str = "target_3way", save: bool = True):
    """Grid-search each model on TRAIN, select the winner by VALIDATION accuracy.

    Tunes on the 3-class target by default and reuses those settings for the
    binary experiment (hyperparameters are robust across the two framings).
    Writes winners to best_params.json so the rest of the project picks them up.

    Returns
    -------
    best_params : {model_name: {param: value}}
    results     : tidy DataFrame of every candidate's validation accuracy
    """
    X_tr, X_val, y_tr, y_val, _, _, _, _ = preprocess(
        df, feat_cols, target_col, split="train_val")

    best_params, rows = {}, []
    for name, grid in PARAM_GRID.items():
        keys = list(grid)
        best = None
        for combo in itertools.product(*(grid[k] for k in keys)):
            override = dict(zip(keys, combo))
            clf = _build_candidate(name, override)
            clf.fit(X_tr, y_tr)
            acc = accuracy_score(y_val, clf.predict(X_val))
            rows.append({"Model": name, **override, "val_accuracy": round(acc, 4)})
            if best is None or acc > best[0]:
                best = (acc, override)
        best_params[name] = best[1]

    results = pd.DataFrame(rows)
    if save:
        with open(BEST_PARAMS_PATH, "w") as f:
            json.dump(best_params, f, indent=2)
    return best_params, results


# ── preprocessing ─────────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, feat_cols: list, target_col: str,
               split: str = "dev_test"):
    """
    1. Drop season-openers (form features NaN because no prior history).
    2. Time-based split by Season (never random).
    3. Impute remaining NaN with first-split median (via sklearn Pipeline).
    4. StandardScale (fit on first split only).

    split
    -----
    "dev_test"  (default) : first = DEV (train+val, 5 seasons), second = TEST.
                            Use for the FINAL models — same data as before, so
                            the simulation notebook is unaffected.
    "train_val"           : first = TRAIN (4 seasons), second = VALIDATION.
                            Use for hyperparameter tuning only.

    Returns (first/second are train/test or train/val depending on `split`)
    -------
    X_a, X_b : np.ndarray   (imputed + scaled)
    y_a, y_b : np.ndarray   (integer-encoded labels)
    a_df, b_df : DataFrame  (raw rows, for per-league eval)
    pipe : fitted sklearn Pipeline (imputer + scaler)
    le   : fitted LabelEncoder
    """
    if split == "dev_test":
        a_seasons, b_seasons = DEV_SEASONS, TEST_SEASONS
    elif split == "train_val":
        a_seasons, b_seasons = TRAIN_SEASONS, VAL_SEASONS
    else:
        raise ValueError(f"split must be 'dev_test' or 'train_val', got {split!r}")

    # drop season-openers
    df = df.dropna(subset=["home_form_points"]).copy()

    # for binary experiment: also drop draws (target is NaN for draws)
    df = df.dropna(subset=[target_col]).copy()

    train_df = df[df["Season"].isin(a_seasons)].reset_index(drop=True)
    test_df  = df[df["Season"].isin(b_seasons)].reset_index(drop=True)

    X_train_raw = train_df[feat_cols].values
    X_test_raw  = test_df[feat_cols].values

    # impute then scale (fit entirely on train)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])
    X_train = pipe.fit_transform(X_train_raw)
    X_test  = pipe.transform(X_test_raw)

    # encode labels
    le = LabelEncoder()
    y_train = le.fit_transform(train_df[target_col].astype(str))
    y_test  = le.transform(test_df[target_col].astype(str))

    return X_train, X_test, y_train, y_test, train_df, test_df, pipe, le


# ── training ──────────────────────────────────────────────────────────────────

def train_all(X_train: np.ndarray, y_train: np.ndarray,
              model_defs: dict = None) -> dict:
    """Fit a fresh copy of each model. Returns {name: fitted_model}.

    model_defs : optional {name: unfitted_model}. If None, get_model_defs()
    is used, which auto-loads tuned hyperparameters when available.
    """
    defs = model_defs if model_defs is not None else get_model_defs()
    fitted = {}
    for name, clf in defs.items():
        clf = deepcopy(clf)
        print(f"  training {name}...", end=" ", flush=True)
        clf.fit(X_train, y_train)
        print("done")
        fitted[name] = clf
    return fitted


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(fitted_models: dict, X_test, y_test, le) -> dict:
    """Return per-model dict with accuracy, report, cm, y_pred, y_proba."""
    results = {}
    for name, clf in fitted_models.items():
        y_pred  = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)
        results[name] = {
            "accuracy":    accuracy_score(y_test, y_pred),
            "report":      classification_report(
                               y_test, y_pred, target_names=le.classes_),
            "report_dict": classification_report(
                               y_test, y_pred, target_names=le.classes_,
                               output_dict=True),
            "cm":     confusion_matrix(y_test, y_pred),
            "y_pred": y_pred,
            "y_proba": y_proba,
        }
    return results


def metrics_table(results: dict, le) -> pd.DataFrame:
    rows = []
    for name, r in results.items():
        rd = r["report_dict"]
        row = {"Model": name, "Accuracy": r["accuracy"]}
        for cls in le.classes_:
            row[f"F1 ({cls})"] = rd.get(cls, {}).get("f1-score", float("nan"))
            row[f"Recall ({cls})"] = rd.get(cls, {}).get("recall", float("nan"))
        row["F1 weighted"] = rd["weighted avg"]["f1-score"]
        rows.append(row)
    return pd.DataFrame(rows).set_index("Model").round(3)


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_cms(results: dict, le, title: str = ""):
    names = list(results.keys())
    fig, axes = plt.subplots(1, 3, figsize=(17, 4))
    for ax, name in zip(axes, names):
        cm = results[name]["cm"]
        # row-normalised so small classes are still visible
        cm_pct = cm.astype(float) / cm.sum(axis=1, keepdims=True)
        sns.heatmap(cm_pct, annot=cm, fmt="d", cmap="Blues",
                    xticklabels=le.classes_, yticklabels=le.classes_,
                    vmin=0, vmax=1, ax=ax, cbar=False)
        ax.set_title(f"{name}\naccuracy = {results[name]['accuracy']:.1%}")
        ax.set_xlabel("predicted"); ax.set_ylabel("actual")
    fig.suptitle(title, fontsize=13, y=1.02)
    plt.tight_layout(); plt.show()


def plot_per_league(fitted_models: dict, test_df, feat_cols, pipe, le):
    rows = []
    for lg in ["epl", "laliga"]:
        sub = test_df[test_df["League"] == lg]
        if len(sub) == 0:
            continue
        X = pipe.transform(sub[feat_cols].fillna(0).values)
        target_col = "target_3way" if set(le.classes_) >= {"H", "D", "A"} else "target_binary"
        y = le.transform(sub[target_col].astype(str))
        for name, clf in fitted_models.items():
            acc = accuracy_score(y, clf.predict(X))
            rows.append({"League": LEAGUE_LABEL[lg], "Model": name, "Accuracy": acc})

    pvt = (pd.DataFrame(rows)
           .pivot(index="Model", columns="League", values="Accuracy"))

    fig, ax = plt.subplots(figsize=(8, 4))
    pvt.plot(kind="bar", ax=ax, rot=0,
             color=["#1f77b4", "#ff7f0e"])
    ax.set_ylim(0.4, 0.75); ax.set_ylabel("accuracy")
    ax.set_title("Accuracy by model and league (test set)")
    ax.axhline(0.54, ls="--", color="grey", alpha=.7, label="overall baseline (54%)")
    ax.legend(title="League")
    plt.tight_layout(); plt.show()
    return pvt


def plot_importances(fitted_models: dict, feat_cols: list, top_n: int = 15):
    fig, axes = plt.subplots(1, 2, figsize=(17, 6))
    for ax, name in zip(axes, ["Random Forest", "XGBoost"]):
        clf = fitted_models[name]
        imp = pd.Series(clf.feature_importances_, index=feat_cols)
        imp.nlargest(top_n).sort_values().plot(kind="barh", ax=ax, color="#1f77b4")
        ax.set_title(f"{name} — top {top_n} features by importance")
        ax.set_xlabel("importance")
    plt.tight_layout(); plt.show()


def plot_calibration(fitted_models: dict, X_test, y_test, le):
    if "H" not in le.classes_:
        print("Calibration only shown for 3-class (H class).")
        return
    class_idx = list(le.classes_).index("H")
    y_bin = (y_test == le.transform(["H"])[0]).astype(int)
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", alpha=.6, label="perfect calibration")
    colors = {"Random Forest": "#1f77b4", "XGBoost": "#ff7f0e", "SVM": "#2ca02c"}
    for name, clf in fitted_models.items():
        proba = clf.predict_proba(X_test)[:, class_idx]
        CalibrationDisplay.from_predictions(
            y_bin, proba, n_bins=10, ax=ax,
            name=name, color=colors.get(name, "grey"),
        )
    ax.set_title("Calibration curves — P(home win)\n(do predicted probabilities match actual rates?)")
    ax.legend(); plt.tight_layout(); plt.show()
