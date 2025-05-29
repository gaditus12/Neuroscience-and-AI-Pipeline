#!/usr/bin/env python
#  estimate_brown_null.py
#
#  Light-weight null-covariance experiment for Brown correction.
#  -------------------------------------------------------------
#  1.  Edit HEURISTIC_MODELS  and K_GRID below.
#  2.  Run with a CSV of features that contains at least:
#        • feature columns
#        • one label column      (default --label-col label)
#        • one group column      (needed only for LOSO)
#
#  Requires: scikit-learn ≥0.24, pandas, numpy, scipy.

import argparse, sys, pathlib, math
import time

import numpy as np
import pandas as pd
from itertools import combinations
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import (
    StratifiedKFold, GroupKFold, LeaveOneGroupOut
)
from scipy.stats import chi2

# --------------------------------------------------------------------------
#  🔧 1. PASTE YOUR MODEL FAMILY DEFINITIONS  HERE
# --------------------------------------------------------------------------
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier
HEURISTIC_MODELS = {
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

#  Number-of-features grid used inside each model’s inner CV
K_GRID = [2, 3, 5, 10]

# --------------------------------------------------------------------------
def make_outer_inner(cv_method, y, groups, random_state=24):
    """Return outer splitter and a prototype inner splitter."""
    if cv_method == "loso":
        outer = LeaveOneGroupOut()
        inner = StratifiedKFold(n_splits=3,
                                shuffle=True, random_state=random_state)
    elif cv_method == "kfold":
        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)
    else:
        raise ValueError("cv_method must be 'loso' or 'kfold'")
    return outer, inner

def estimate_null_cov(
        X: pd.DataFrame,
        y: np.ndarray,
        models: dict,
        k_grid,
        cv_method="loso",
        groups=None,
        n_perm=300,
        rng_seed=42,
        verbose=True):
    """
    Run n_perm label shuffles and record outer-mean F1 for every model.
    Returns R matrix, rhō, c, df_eff.
    """
    rng = np.random.default_rng(rng_seed)
    clf_names = list(models)
    k = len(clf_names)

    null_stats = {m: [] for m in clf_names}

    outer, inner_proto = make_outer_inner(cv_method, y, groups)
    # Helper to iterate outer splits once per permutation
    def outer_split():
        if cv_method == "loso":
            for train, test in outer.split(X, y, groups):
                yield train, test
        else:   # kfold
            for train, test in outer.split(X, y):
                yield train, test

    for p in range(n_perm):
        y_perm = rng.permutation(y)

        for tr_idx, te_idx in outer_split():
            X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
            y_tr, y_te = y_perm[tr_idx], y_perm[te_idx]

            for m_name, mdl in models.items():
                # ---------- inner loop to choose k ----------
                best_k, best_cv = None, -np.inf
                for k_feats in k_grid:
                    cv_scores = []
                    for in_tr, in_val in inner_proto.split(X_tr, y_tr):
                        scaler = StandardScaler().fit(X_tr.iloc[in_tr])
                        sel = SelectKBest(f_classif, k=k_feats).fit(
                            scaler.transform(X_tr.iloc[in_tr]),
                            y_tr[in_tr])
                        Xtr_sel = sel.transform(scaler.transform(X_tr.iloc[in_tr]))
                        Xval_sel = sel.transform(scaler.transform(X_tr.iloc[in_val]))
                        y_hat = clone(mdl).fit(Xtr_sel, y_tr[in_tr]).predict(Xval_sel)
                        cv_scores.append(
                            f1_score(y_tr[in_val], y_hat, average="macro"))
                    mean_cv = np.mean(cv_scores)
                    if mean_cv > best_cv:
                        best_cv, best_k = mean_cv, k_feats

                # ---------- train on full outer-train, test on outer-test ----------
                scaler = StandardScaler().fit(X_tr)
                sel = SelectKBest(f_classif, k=best_k).fit(
                    scaler.transform(X_tr), y_tr)
                Xtr_sel = sel.transform(scaler.transform(X_tr))
                Xte_sel = sel.transform(scaler.transform(X_te))
                y_hat = clone(mdl).fit(Xtr_sel, y_tr).predict(Xte_sel)
                outer_f1 = f1_score(y_te, y_hat, average="macro")
                null_stats[m_name].append(outer_f1)

        if verbose and (p+1) % 50 == 0:
            print_log(f"  finished {p+1}/{n_perm} permutations")

    # ---------- correlation of −log stats ----------
    S = pd.DataFrame(null_stats)            # n_perm × k
    W = -np.log(S)                          # transform
    R = W.corr().to_numpy()
    rho_bar = R[np.triu_indices(k, 1)].mean()
    c       = 1 + (k - 1)*rho_bar
    df_eff  = 2 * k / c
    return R, rho_bar, c, df_eff

def brown_p(real_pvals, rho_bar):
    real_pvals = np.asarray(real_pvals, float)
    k = len(real_pvals)
    X = -2 * np.sum(np.log(real_pvals))
    c = 1 + (k - 1)*rho_bar
    df_eff = 2*k / c
    return chi2.sf(X / c, df_eff)




# --------------------------------------------------------------------------
def main():

    print_log(f'-----STARTING------')
    print_log(f'\np values used:{p_values}')
    parser = argparse.ArgumentParser(
        description="Estimate Brown correction (ρ̄, df_eff) from shuffled labels.")
    parser.add_argument("--csv_path", default=f"brown_corrections/{file}.csv", help="CSV with features + labels (+ group column)")
    parser.add_argument("--label-col", default="label",
                        help="Column name for class labels [default: label]")
    parser.add_argument("--group-col", default="session",
                        help="Column with group IDs for LOSO [default: session]")
    parser.add_argument("--cv", choices=["loso", "kfold"], default=cv,
                        help="Outer-CV scheme [loso | kfold] (default: loso)")
    parser.add_argument("--n-perm", type=int, default=n_perm,
                        help="Number of shuffled-label permutations [default: 300]")
    parser.add_argument("--seed", type=int, default=42, help="RNG seed [42]")
    parser.add_argument("--real-pvals", default=p_values,
                        help="Comma-sep list of eight real-label p-values to get Brown-adjusted p.")
    parser.add_argument("--save-R", metavar="FILE",
                        help="Optional path to save the correlation matrix as CSV")
    args = parser.parse_args()

    # ---------- load data ----------
    df = pd.read_csv(args.csv_path)
    if args.label_col not in df.columns:
        sys.exit(f"Label column '{args.label_col}' not found in CSV.")
    if args.cv == "loso" and args.group_col not in df.columns:
        sys.exit(f"Group column '{args.group_col}' required for LOSO not found.")
    if 'channel' in df.columns:
        df=df.drop(columns=['channel'])
    y = df[args.label_col].values
    groups = df[args.group_col].values if args.cv == "loso" else None
    X = df.drop(columns=[args.label_col] + ([args.group_col])) # if args.cv=="loso" else []))
    print_log(f"Data shape: {X.shape},  classes: {np.unique(y)},  n_perm={args.n_perm}")

    # ---------- estimate null covariance ----------
    R, rho_bar, c, df_eff = estimate_null_cov(
        X, y, HEURISTIC_MODELS, K_GRID,
        cv_method=args.cv, groups=groups,
        n_perm=args.n_perm, rng_seed=args.seed)

    print_log("\n--- Results ---")
    print_log(("ρ̄  (mean off-diag corr) :", round(rho_bar, 4)))
    print_log(("c   (Brown scale factor) :", round(c, 4)))
    print_log(("df_eff                  :", round(df_eff, 3)))

    if args.save_R:
        pd.DataFrame(R, index=HEURISTIC_MODELS, columns=HEURISTIC_MODELS)\
            .to_csv(args.save_R)
        print_log(f"Correlation matrix saved → {args.save_R}")

    # ---------- Brown-adjusted p, if requested ----------
    if args.real_pvals:
        real_pvals = [float(x) for x in args.real_pvals.split(",")]
        if len(real_pvals) != len(HEURISTIC_MODELS):
            sys.exit("Need exactly %d p-values (got %d)" %
                     (len(HEURISTIC_MODELS), len(real_pvals)))
        p_brown = brown_p(real_pvals, rho_bar)
        print_log(("Brown-adjusted combined p =", "{:.3e}".format(p_brown)))

    print_log(f'-----ENDED------')
    timeTook=time.time()-t1
    print_log(f'\n\nTime took:{timeTook}s\n\tFor {n_perm} permutations using {cv}\n\nAverage time per permutation using {cv} is {timeTook/n_perm}')

def print_log(x):
    message = str(x)
    print(message)
    with open('brown_corrections' +  f"/[{cv}]_[{n_perm}_perm]_[{file}]_[{K_GRID}].txt", "a", encoding="utf-8") as f:
        f.write(message + "\n")


if __name__ == "__main__":
    t1 = time.time()
    cv = 'kfold'
    file="O2_SPI_norm-Z"
    p_values=("0.016,"
              "0.007,"
              "0.003,"
              "0.002,"
              "0.005,"
              "0.015,"
              "0.005," #rf
              "0.012")
    n_perm = 100
    main()
