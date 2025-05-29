#!/usr/bin/env python
"""
eff_df_4variants.py
-------------------
Estimate the effective degrees of freedom (k_eff) for the four
pipeline variants:

    • Z-score  + k-fold
    • Z-score  + LOSO
    • ComBat   + k-fold
    • ComBat   + LOSO

For each label permutation we run a full nested CV, keep the
outer-mean macro-F1 of the *inner-winner*, and then compute the
4 × 4 null-correlation matrix, ρ̄, and k_eff (Brown–Kost).

Inputs
------
--norm_csv   <file>   z-scored features
--combat_csv <file>   ComBat-harmonised features

Each CSV must contain:  feature columns,  ‘label’,  ‘session’.

Example
-------
python eff_df_4variants.py \
       --norm_csv   o2_norm-z.csv \
       --combat_csv o2_norm-combat.csv \
       --n_perm 300 \
       --real_pvals 0.002,0.016,0.015,0.002
"""

import argparse, time, sys, os
import numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, LeaveOneGroupOut
from scipy.stats import chi2

# ──────────────────────────────────────────────────────────────────────────
#  Model zoo  +  k-grid
# ──────────────────────────────────────────────────────────────────────────
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

HEURISTIC_MODELS = {
    "RandomForest": RandomForestClassifier(
        n_estimators=80, max_depth=3, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=1),          # n_jobs=1 avoids psutil bug
    "ExtraTrees": ExtraTreesClassifier(
        n_estimators=120, max_depth=4, min_samples_leaf=2,
        class_weight="balanced", random_state=42, n_jobs=1),
    "SVM": SVC(kernel="rbf", C=0.8, gamma="scale",
               class_weight="balanced", probability=False, random_state=42),
    "ElasticNetLogReg": LogisticRegression(
        penalty="elasticnet", C=1.0, l1_ratio=0.5,
        solver="saga", max_iter=4000, class_weight="balanced", random_state=42),
    "HGBClassifier": HistGradientBoostingClassifier(
        learning_rate=0.05, max_depth=3, max_iter=80,
        class_weight="balanced", random_state=42),
    "kNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),
    "GaussianNB": GaussianNB(),
    "ShrinkageLDA": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
}

K_GRID = [2, 3, 5, 10]

# ──────────────────────────────────────────────────────────────────────────
#  Helpers
# ──────────────────────────────────────────────────────────────────────────
def make_splitters(method):
    if method == "loso":
        outer = LeaveOneGroupOut()
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=24)
    elif method == "kfold":
        outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=24)
        inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=24)
    else:
        raise ValueError("cv method must be 'loso' or 'kfold'")
    return outer, inner

def needs_groups(splitter):
    """True if the splitter requires a groups array."""
    return isinstance(splitter, LeaveOneGroupOut)

# -------------------------------------------------------------------------
def best_model_outer_mean_f1(X, y, groups, outer_cv, inner_proto):
    """
    One full nested-CV run → mean outer-fold F1 of the inner-winner.
    """
    outer_scores = []

    # prepare outer split iterator with / without groups
    outer_iter = (outer_cv.split(X, y, groups) if needs_groups(outer_cv)
                  else outer_cv.split(X, y))

    for tr_idx, te_idx in outer_iter:
        X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]

        # ---------------- inner grid-search (model × k) -------------------
        best_name, best_k, best_inner = None, None, -np.inf
        for mdl_name, base_mdl in HEURISTIC_MODELS.items():
            for k in K_GRID:
                scores = []
                inner_iter = (inner_proto.split(X_tr, y_tr)  # Stratified K-fold – no groups
                              if not needs_groups(inner_proto) else
                              inner_proto.split(X_tr, y_tr, groups[tr_idx]))
                for in_tr, in_val in inner_iter:
                    scaler = StandardScaler().fit(X_tr.iloc[in_tr])
                    sel = SelectKBest(f_classif, k=k).fit(
                        scaler.transform(X_tr.iloc[in_tr]), y_tr[in_tr])
                    Xtr_sel = sel.transform(scaler.transform(X_tr.iloc[in_tr]))
                    Xval_sel = sel.transform(scaler.transform(X_tr.iloc[in_val]))
                    y_hat = clone(base_mdl).fit(Xtr_sel, y_tr[in_tr]).predict(Xval_sel)
                    scores.append(f1_score(y_tr[in_val], y_hat, average="macro"))
                if np.mean(scores) > best_inner:
                    best_inner = np.mean(scores)
                    best_name  = mdl_name
                    best_k     = k

        # ------------- retrain best on full outer train --------------------
        scaler = StandardScaler().fit(X_tr)
        sel    = SelectKBest(f_classif, k=best_k).fit(
                    scaler.transform(X_tr), y_tr)
        Xtr_sel = sel.transform(scaler.transform(X_tr))
        Xte_sel = sel.transform(scaler.transform(X_te))
        y_hat   = clone(HEURISTIC_MODELS[best_name]).fit(Xtr_sel, y_tr).predict(Xte_sel)
        outer_scores.append(f1_score(y_te, y_hat, average="macro"))

    return np.mean(outer_scores)

# -------------------------------------------------------------------------
def brown_stats(null_mat):
    W = -np.log(null_mat)
    R = np.corrcoef(W, rowvar=False)
    rho_bar = R[np.triu_indices(4, 1)].mean()
    c = 1 + (4 - 1) * rho_bar
    k_eff = 2 * 4 / c
    return rho_bar, c, k_eff, R



def print_log(x):
    message = str(x)
    print(message)
    with open('brown_corrections' +  f"/ensemble_corr_[{n_perm}_perm]_[{combat_csv}_{norm_csv}].txt", "a", encoding="utf-8") as f:
        f.write(message + "\n")


def brown_p(real_p, rho_bar):
    X = -2 * np.sum(np.log(real_p))
    c = 1 + (4 - 1) * rho_bar
    df_eff = 2 * 4 / c
    return chi2.sf(X / c, df_eff)


# -------------------------------------------------------------------------
def main():
    print_log('melo')
    # ---------- load data -------------------------------------------------
    def load_df(path):
        df = pd.read_csv(path)
        X = df.drop(columns=[args.label_col, args.group_col, 'channel'])
        y = df[args.label_col].values
        groups = df[args.group_col].values
        return X, y, groups

    X_norm,   y_norm,   g_norm   = load_df('final_final_set/'+args.norm_csv)
    X_comb,   y_comb,   g_comb   = load_df('final_final_set/'+args.combat_csv)

    variants = [
        ("Z-kfold",  X_norm,  y_norm,  g_norm,  "kfold"),
        ("Z-loso",   X_norm,  y_norm,  g_norm,  "loso"),
        ("CB-kfold", X_comb,  y_comb,  g_comb,  "kfold"),
        ("CB-loso",  X_comb,  y_comb,  g_comb,  "loso"),
    ]

    # ---------- permutation loop ------------------------------------------
    null_mat = np.zeros((args.n_perm, 4))
    for p in range(args.n_perm):
        for j, (name, X, y, groups, cvtype) in enumerate(variants):
            y_perm = rng.permutation(y)                 # shared RNG keeps dependence
            outer, inner = make_splitters(cvtype)
            null_mat[p, j] = best_model_outer_mean_f1(
                X, y_perm, groups, outer, inner)
        if (p + 1) % 50 == 0:
            print_log((f"  finished {p+1}/{args.n_perm} permutations"))

    rho_bar, c, k_eff, R = brown_stats(null_mat)
    print_log(("\n--- Brown/Kost statistics ---"))
    print_log((f"mean off-diag ρ   : {rho_bar:.4f}"))
    print_log((f"scale factor  c   : {c:.4f}"))
    print_log((f"effective df      : {k_eff:.2f}"))

    # ---------- optional Brown combined p --------------------------------
    if args.real_pvals:
        pvec = np.asarray([float(x) for x in args.real_pvals.split(",")])
        if len(pvec) != 4:
            sys.exit("Need exactly four p-values.")
        print_log((f"Brown-adjusted p  : {brown_p(pvec, rho_bar):.3e}"))

    pd.DataFrame(R, index=[v[0] for v in variants],
                    columns=[v[0] for v in variants]).to_csv("R_4variants.csv")
    print_log(("✓ saved correlation matrix → R_4variants.csv"))

# -------------------------------------------------------------------------
if __name__ == "__main__":
    global n_perm
    global combat_csv, norm_csv

    ap = argparse.ArgumentParser()
    ap.add_argument("--norm_csv", required=True)
    ap.add_argument("--combat_csv", required=True)
    ap.add_argument("--label_col", default="label")
    ap.add_argument("--group_col", default="session")
    ap.add_argument("--n_perm", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--real_pvals",
                    help="comma-separated list of the four real-data p-values")
    args = ap.parse_args()
    combat_csv= args.combat_csv
    norm_csv= args.norm_csv
    n_perm = args.n_perm

    rng = np.random.default_rng(args.seed)
    print_log(f'\nP values are given as follows: {args.real_pvals}\nWith input files: {combat_csv} and {norm_csv}\nFollowing {n_perm} permutations')

    t0 = time.time()
    main()
    print_log((f"\nTotal wall-time: {time.time() - t0:.1f} s"))


