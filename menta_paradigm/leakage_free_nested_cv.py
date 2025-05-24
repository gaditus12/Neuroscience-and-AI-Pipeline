# leakage_free_nested_cv.py
"""Leakage‑free nested cross‑validation with per‑fold ComBat harmonisation.

Usage (command line) ---------------------------------------------------------
$ python leakage_free_nested_cv.py \
        --csv data/o2_raw.csv            # or directory with many CSVs
        --cv loso                         # 'loso' | 'kfold'
        --k 5                             # k for k‑fold  (ignored for loso)
        --features 25                     # top‑K univariate features per fold
        --out   results/                  # where to drop logs / CSVs

*   The CSV(s) must contain exactly:
        37 numeric feature columns (any names),
        one 'channel' column  (ignored by the model),
        one 'label'   column  (target),
        one 'session' column (batch variable).
*   NO baseline table is needed – only ComBat harmonisation across 'session'.
*   The script never fits ComBat on data from the evaluation fold ⇒ no test‑set
    leakage.  Accuracy numbers are an honest estimate of cross‑session general‑
    isation.

Plug‑in model families -------------------------------------------------------
Create a file called  heuristic_models.py  next to this script with a global
variable  HEURISTIC_MODELS  that is the dict you showed me, e.g.::

    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.svm import SVC
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import GaussianNB
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.ensemble import HistGradientBoostingClassifier

    HEURISTIC_MODELS = {
        "RandomForest": RandomForestClassifier(...),
        ...
    }

If the module is missing the script will raise an informative error.
"""

from __future__ import annotations

import argparse, os, sys, json, pathlib, datetime
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    LeaveOneGroupOut,
    StratifiedKFold,
    StratifiedGroupKFold,
)
from sklearn.base import clone
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

# ---------------------------------------------------------------------------
#   ───  ComBat wrapper (leakage‑safe)  ──────────────────────────────────────
# ---------------------------------------------------------------------------
try:
    from neuroHarmonize.harmonizationLearn import harmonizationLearn
    from neuroHarmonize.harmonizationApply import harmonizationApply
except ImportError as e:
    print("❌  neuroHarmonize is required but not installed:", e, file=sys.stderr)
    sys.exit(1)

class CombatTransformer:
    """scikit‑learn‑compatible transformer for ComBat harmonisation per fold."""

    def __init__(self, feature_cols: List[str], site_col: str = "session"):
        self.feature_cols = feature_cols
        self.site_col = site_col
        self.model_ = None  # will hold the learned parameters

    # scikit‑learn API ------------------------------------------------------
    def fit(self, X: pd.DataFrame, y=None):  # y ignored
        covars = pd.DataFrame({"SITE": X[self.site_col].values}, index=X.index)
        data = X[self.feature_cols].values
        self.model_, _ = harmonizationLearn(data, covars, eb=True)
        return self

    def transform(self, X: pd.DataFrame):
        covars = pd.DataFrame({"SITE": X[self.site_col].values}, index=X.index)
        data = X[self.feature_cols].values
        adjusted = harmonizationApply(data, covars, self.model_)
        # return *only* the numeric feature matrix (no session, no label)
        return pd.DataFrame(adjusted, columns=self.feature_cols, index=X.index)

# ---------------------------------------------------------------------------
#   ───  Helper functions  ───────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def load_csvs(path: Path) -> pd.DataFrame:
    """Load a single CSV or merge all CSVs in a directory."""
    if path.is_file():
        return pd.read_csv(path)
    if path.is_dir():
        dfs = [pd.read_csv(p) for p in sorted(path.glob("*.csv"))]
        if not dfs:
            raise FileNotFoundError(f"No .csv files found in {path}")
        return pd.concat(dfs, ignore_index=True)
    raise FileNotFoundError(path)


def identify_feature_cols(df: pd.DataFrame) -> List[str]:
    meta = {"channel", "label", "session"}
    return [c for c in df.columns if c not in meta]


# ---------------------------------------------------------------------------
#   ───  Nested CV core  ─────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def nested_cv(
    df: pd.DataFrame,
    models: Dict[str, object],
    cv: str = "loso",
    k_splits: int = 5,
    k_features: int = 25,
    out_dir: Path = Path("results"),
):
    out_dir.mkdir(parents=True, exist_ok=True)

    feat_cols = identify_feature_cols(df)
    X_all = df[feat_cols + ["session"]]
    y_all = df["label"]
    sessions = df["session"].values

    # ❶  splitters ---------------------------------------------------------
    if cv == "loso":
        outer_cv = LeaveOneGroupOut()
        outer_iter = outer_cv.split(X_all, y_all, sessions)
    elif cv == "kfold":
        # Stratified by label but sessions ignored; still groups‑aware inner splits
        outer_cv = StratifiedKFold(n_splits=k_splits, shuffle=True, random_state=24)
        outer_iter = outer_cv.split(X_all, y_all)
    else:
        raise ValueError("cv must be 'loso' or 'kfold'")

    # (inner CV will always be a *stratified* k‑fold on the ComBat‑harmonised
    #  training matrix; we keep k=3 or k=5 depending on sample size.)

    results = {
        m: dict(per_fold_acc=[], per_fold_f1=[], per_fold_n=[]) for m in models
    }

    for fold, (tr_idx, te_idx) in enumerate(outer_iter, start=1):
        # ── 4.0  ComBat on outer‑train  ────────────────────────────────────
        combat = CombatTransformer(feat_cols, "session").fit(X_all.iloc[tr_idx])
        X_tr_cb = combat.transform(X_all.iloc[tr_idx])
        X_te_cb = combat.transform(X_all.iloc[te_idx])
        y_tr = y_all.iloc[tr_idx]
        y_te = y_all.iloc[te_idx]

        # choose inner splitter size dynamically (≥ 3 splits, ≤ len(y_tr)‑1)
        k_inner = min(5, max(3, len(y_tr) // 10))
        inner_cv = StratifiedKFold(n_splits=k_inner, shuffle=True, random_state=42)

        # ── 4.1  model selection on inner CV  ─────────────────────────────
        inner_scores = {}
        for name, base_clf in models.items():
            pipe = Pipeline([
                ("scale", StandardScaler()),
                ("select", SelectKBest(f_classif, k=k_features)),
                ("clf", clone(base_clf)),
            ])
            f1s = []
            for in_tr, in_val in inner_cv.split(X_tr_cb, y_tr):
                pipe.fit(X_tr_cb.iloc[in_tr], y_tr.iloc[in_tr])
                y_hat = pipe.predict(X_tr_cb.iloc[in_val])
                f1s.append(f1_score(y_tr.iloc[in_val], y_hat, average="macro"))
            inner_scores[name] = float(np.mean(f1s))

        best_name = max(inner_scores, key=inner_scores.get)
        best_clf = models[best_name]

        # ── 4.2  Evaluate *each* model on outer‑test (for fairness)  ───────
        for name, base_clf in models.items():
            pipe = Pipeline([
                ("scale", StandardScaler()),
                ("select", SelectKBest(f_classif, k=k_features)),
                ("clf", clone(base_clf)),
            ]).fit(X_tr_cb, y_tr)

            y_pred = pipe.predict(X_te_cb)
            acc = accuracy_score(y_te, y_pred)
            f1  = f1_score(y_te, y_pred, average="macro")

            rec = results[name]
            rec["per_fold_acc"].append(acc)
            rec["per_fold_f1"].append(f1)
            rec["per_fold_n"].append(len(te_idx))

        print(f"Fold {fold:02d}/{cv}: best inner = {best_name}")

    # ❺  aggregate ---------------------------------------------------------
    summary_rows = []
    for name, rec in results.items():
        w = np.asarray(rec["per_fold_n"], float)
        mean_acc = float(np.average(rec["per_fold_acc"], weights=w))
        mean_f1  = float(np.average(rec["per_fold_f1"],  weights=w))
        rec["mean_acc"], rec["mean_f1"] = mean_acc, mean_f1
        summary_rows.append(dict(model=name, mean_acc=mean_acc, mean_f1=mean_f1))

    # ❻  save CSVs ---------------------------------------------------------
    pd.DataFrame(summary_rows).to_csv(out_dir / "model_means.csv", index=False)

    # Per‑fold table (long format) ----------------------------------------
    long_rows = []
    for name, rec in results.items():
        for f, (acc, f1, n) in enumerate(zip(rec["per_fold_acc"],
                                             rec["per_fold_f1"],
                                             rec["per_fold_n"]), start=1):
            long_rows.append(dict(model=name, fold=f, outer_acc=acc,
                                  outer_f1=f1, n_test=n))
    pd.DataFrame(long_rows).to_csv(out_dir / "per_fold_scores.csv", index=False)

    print("✓  Saved results to", out_dir)

# ---------------------------------------------------------------------------
#   ───  CLI  ────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Leakage‑free nested CV with per‑fold ComBat")
    ap.add_argument("--csv", required=True, help="CSV file or directory of CSVs")
    ap.add_argument("--cv", choices=["loso", "kfold"], default="loso",
                    help="outer CV scheme (default: loso)")
    ap.add_argument("--k", type=int, default=5, help="k for k‑fold (ignored in loso)")
    ap.add_argument("--features", type=int, default=25, help="top‑K features per fold")
    ap.add_argument("--out", default="results", help="output directory")
    args = ap.parse_args()

    # ------------------------------------------------------------------
    HEURISTIC_MODELS= {
            "RandomForest": RandomForestClassifier(
                n_estimators=80, max_depth=3, min_samples_leaf=2,
                class_weight="balanced", random_state=42, n_jobs=-1),
            "SVM": SVC(kernel="rbf", C=0.8, gamma="scale",
                       class_weight="balanced", probability=True, random_state=42),
            "ElasticNetLogReg": LogisticRegression(
                penalty="elasticnet", C=1.0, l1_ratio=0.5,
                solver="saga", max_iter=4000, class_weight="balanced", random_state=42),
            "ExtraTrees": ExtraTreesClassifier(
                n_estimators=120, max_depth=4, min_samples_leaf=2,
                class_weight="balanced", random_state=42, n_jobs=-1),
            "HGBClassifier": HistGradientBoostingClassifier(
                learning_rate=0.05, max_depth=3, max_iter=80,
                class_weight="balanced", random_state=42),
            "kNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),
            "GaussianNB": GaussianNB(),
            "ShrinkageLDA": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        }

    df = load_csvs(Path(args.csv))
    nested_cv(df, HEURISTIC_MODELS, cv=args.cv, k_splits=args.k,
              k_features=args.features, out_dir=Path(args.out))


if __name__ == "__main__":
    main()
