import os
import time

from itertools import combinations
from tqdm import tqdm

from sklearn.decomposition import PCA
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import plotly.express as px
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from sklearn.model_selection import LeaveOneGroupOut, LeavePGroupsOut  # NEW
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Add these to your imports at the top of the file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Figure clean look
plt.style.use("ggplot")

# Optional: customize global defaults if you want more control
plt.rcParams.update({
    "figure.figsize": (10, 5),          # Default figure size
    "axes.edgecolor": "0.5",            # Axis color
    "axes.grid": True,                  # Show grid
    "grid.alpha": 0.4,                  # Grid transparency
    "grid.linestyle": "--",             # Dashed grid lines
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "patch.force_edgecolor": True,      # Outline histogram bars
    "patch.linewidth": 0.5,
})


import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import (
    LeaveOneOut,
    cross_val_score,
    StratifiedKFold,
    train_test_split,
)
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score,
)
from sklearn.pipeline import Pipeline

# scikit-optimize for Bayesian optimization
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical

    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print(
        "Warning: scikit-optimize not available. Install with 'pip install scikit-optimize' to use hyperparameter optimization."
    )

from sklearn.model_selection import (
    LeaveOneGroupOut,
    StratifiedGroupKFold,
    GroupShuffleSplit,
    StratifiedKFold,
)


class EEGAnalyzer:
    def __init__(
        self,
        features_file,
        top_n_labels=2,
        n_features_to_select=15,
        channel_approach="pooled",  # Options: "pooled", "separate", "features"
        cv_method="kfold",  # Options: "loo", "kfold", "holdout"
        cv_version="extended",  # extended or simple with a simple feature selection
        kfold_splits=5,
        test_size=0.2,
        lmoso_leftout: int = 2,
        permu_count=1_000,
        optimize=False,
        eval=False,
        frozen_features=None,
        run_directory=None
    ):  # For holdout validation

        # Configuration
        if frozen_features is None:
            frozen_features = [2,3,5,10,15]
        else:
            frozen_features = [frozen_features]
        self.features_file = (
            "data/final_sets/all_channels_binary/no_leak/final_final_set/" + features_file + ".csv"
        )
        self.top_n_labels = top_n_labels
        self.n_features_to_select = n_features_to_select
        self.channel_approach = channel_approach.lower()
        self.cv_method = cv_method.lower()
        self.cv_version = cv_version.lower()
        self.kfold_splits = kfold_splits
        self.test_size = test_size
        self.permu_count = permu_count
        self.lmoso_leftout = lmoso_leftout
        self.frozen_features = frozen_features

        # helper to render 1000→"1k"
        to_k = lambda n: (
            f"{n / 1000:.1f}k".rstrip("0").rstrip(".")
            if n >= 1000 else str(n)
        )
        permu_str = to_k(self.permu_count)

        ts = int(time.time())
        channel_norm = features_file.split("/")[-1].split(".")[0]
        ff=self.frozen_features[0]

        dirname = (
            f"[{channel_norm}]_"
            f"[{self.cv_method}]_[{permu_str}perm]_"
            f"[{ff}feat]_"
            f"run_{ts}"
        )
        # --- new run_directory logic ---
        if run_directory:
            # user specified hierarchy ⇒ trust it
            self.run_directory = os.path.join(run_directory, dirname)
        else:
            # fallback to your old timestamp‐based naming
            if eval:
                tag = "EVAL"
            elif optimize:
                tag = "optimize"
            else:
                tag = "standard"
            dirname = (
                f"{tag}_[{channel_norm}]_"
                f"[{self.cv_method}]_[{permu_str}perm]_"
                f"[{ff}feat]_"
                f"run_{ts}"
            )
            self.run_directory = os.path.join("ml_model_outputs", dirname)

        os.makedirs(self.run_directory, exist_ok=True)
        # dump the features file
        with open(os.path.join(self.run_directory, "features_file.txt"), "w") as f:
            f.write(self.features_file)

        self.heuristic_models=self._default_heuristic_models()
        # Initialize placeholders for later use
        self.df = None
        self.feature_columns = None
        self.unique_labels = None
        self.unique_channels = None
        self.X = None
        self.y = None
        self.scaler = StandardScaler()
        self.X_scaled = None
        self.X_scaled_df = None
        self.X_selected = None
        self.selected_features = None
        self.selector = None
        self.separability_scores = {}
        self.detailed_results = {}
        self.best_labels = None


    # [load_data, preprocess_data, feature_selection, and DummySelector remain unchanged]
    def load_data(self):
        print_log("---- LOADING DATA ----")
        self.df = pd.read_csv(self.features_file)

        # Ensure there is a 'channel' column (if missing, assume first column is channel)
        if "channel" not in self.df.columns:
            self.df["channel"] = self.df.iloc[:, 0]

        # Define feature columns (all columns except metadata)
        self.feature_columns = [
            col for col in self.df.columns if col not in ["label", "channel", "session"]
        ]

        # Get unique labels and channels
        self.unique_labels = self.df["label"].unique()
        self.unique_channels = (
            self.df["channel"].unique() if "channel" in self.df.columns else ["unknown"]
        )

        print_log(
            f"Dataset has {len(self.df)} rows with {len(self.feature_columns)} features"
        )
        print_log(
            f"Found {len(self.unique_labels)} unique labels: {self.unique_labels}"
        )
        print_log(
            f"Found {len(self.unique_channels)} unique channels: {self.unique_channels}\n"
        )

        # Print sample counts
        print_log("Sample counts per label:")
        print_log(self.df.groupby("label").size())
        if "channel" in self.df.columns:
            print_log("\nSample counts per label and channel:")
            print_log(self.df.groupby(["label", "channel"]).size())

        # Handle channels based on approach
        if self.channel_approach == "pooled":
            print_log(
                "\nUsing POOLED approach: Treating each channel reading as an independent sample"
            )
            self.X = self.df[self.feature_columns]
            self.y = self.df["label"]

        elif self.channel_approach == "separate":
            print_log("\nUsing SEPARATE approach: Analyzing each channel independently")
            # For now we use the same data – further processing could be added later.
            self.X = self.df[self.feature_columns]
            self.y = self.df["label"]

        else:  # "features" approach
            print_log(
                "\nUsing FEATURES approach: Combining channels as additional features"
            )
            if "session" not in self.df.columns:
                print_log(
                    "Error: This approach requires a 'session' column to identify unique recordings. Falling back to POOLED."
                )
                self.channel_approach = "pooled"
                self.X = self.df[self.feature_columns]
                self.y = self.df["label"]
            else:
                # Placeholder: For a proper implementation, you would combine rows per session.
                self.X = self.df[self.feature_columns]
                self.y = self.df["label"]

    def preprocess_data(self, debug_plots_only=True):
        print_log("\n---- PREPROCESSING DATA ----")
        # This method should only perform exploratory analysis or simple data cleaning.
        # The actual standardization will occur in cross-validation.

        # Basic data examination
        print_log(f"Feature value ranges (min, max):")
        for col in self.feature_columns[:5]:  # Print first few for brevity
            print_log(f"  {col}: ({self.X[col].min():.4f}, {self.X[col].max():.4f})")

        # Check for missing values
        missing_values = self.X.isnull().sum().sum()
        print_log(f"Total missing values: {missing_values}")

        if missing_values > 0:
            print_log(
                "WARNING: Dataset contains missing values which may affect analysis"
            )

        # Note about standardization
        print_log(
            "\nNOTE: Feature standardization will be performed within each cross-validation fold"
        )
        print_log("to prevent information leakage between training and test data.")

        # For backward compatibility with the rest of the code, we'll still fit a scaler
        # but with a clear warning that it should not be used for evaluation
        if debug_plots_only:
            self.X_scaled = self.scaler.fit_transform(self.X)
            self.X_scaled_df = pd.DataFrame(self.X_scaled, columns=self.feature_columns)
            print_log(
                "\nWARNING: Global standardization performed for visualization purposes only."
            )
            print_log("DO NOT use self.X_scaled for model evaluation/training.")

    def feature_selection(self):
        """
        Feature selection method - To be called before cross-validation
        NOTE: This method is for informational purposes only. The actual feature selection
        is performed within the cross-validation/holdout methods to avoid information leakage.
        """
        print_log("\n---- FEATURE SELECTION ANALYSIS (Informational Only) ----")
        print_log(
            "Note: The actual feature selection is performed within each cross-validation fold"
        )

        # This is only for initial visualization and exploration
        selector = SelectKBest(f_classif, k=self.n_features_to_select)
        selector.fit(self.X_scaled, self.y)
        selected_indices = selector.get_support(indices=True)

        # Store for visualization purposes only
        self.selected_features = [self.feature_columns[i] for i in selected_indices]

        print_log("Top features based on F-test (for informational purposes only):")
        f_scores = selector.scores_
        p_values = selector.pvalues_

        feature_scores = pd.DataFrame(
            {"feature": self.feature_columns, "f_score": f_scores, "p_value": p_values}
        ).sort_values("f_score", ascending=False)

        for i, (_, row) in enumerate(
            feature_scores.head(self.n_features_to_select).iterrows()
        ):
            print_log(
                f"  {i + 1}. {row['feature']} (F-score: {row['f_score']:.4f}, p-value: {row['p_value']:.4f})"
            )

        # Create a dummy selector for compatibility
        self.selector = self.DummySelector(self.feature_columns, self.selected_features)

        print_log(
            "\nWARNING: This is preliminary analysis only. Proper feature selection"
        )
        print_log("will be performed within each fold of cross-validation.")

    # ─── NEW util  ──────────────────────────────────────────────────────────
    def _get_nested_splitters(self, groups):
        """
        Decide which outer / inner CV objects to use, given current
        self.cv_method and the session-id array `groups`.
        Returns: (outer_cv, inner_cv)
        """
        if self.cv_method == "loso":
            outer = LeaveOneGroupOut()  # each outer fold = 1 session
            #   inner: random 20 % of the *remaining* sessions
            inner = GroupShuffleSplit(n_splits=3, test_size=0.20, random_state=42)
        elif self.cv_method == "kfold":
            outer = StratifiedKFold(
                n_splits=self.kfold_splits, shuffle=True, random_state=42
            )
            inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=24)
        elif self.cv_method == "loo":
            outer = LeaveOneGroupOut()
            inner = GroupShuffleSplit(n_splits=3, test_size=0.20, random_state=42)

        elif self.cv_method == "lmoso":
            outer = LeavePGroupsOut(self.lmoso_leftout)  # outer: leave-P-groups-out
            inner = GroupShuffleSplit(
                n_splits=3, test_size=0.20, random_state=42
            )  # inner: shuffled groups
        else:  # "holdout" → no outer CV, keep inner only
            outer = None
            inner = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

        return outer, inner

    class DummySelector:
        def __init__(self, feature_columns, selected_features):
            self.feature_columns = feature_columns
            self.selected_features = selected_features

        def transform(self, X):
            indices = [self.feature_columns.index(f) for f in self.selected_features]
            return X[:, indices]

        def get_support(self, indices=False):
            mask = [f in self.selected_features for f in self.feature_columns]
            if indices:
                return [i for i, m in enumerate(mask) if m]
            return mask

    # --------------------------------------------------------------
    #  NEW : evaluation helper for Leave‑One‑Session‑Out
    # --------------------------------------------------------------
    def _evaluate_with_cv_loso(self, X, y, models, cv, groups):
        """
        Exactly the same logic as _evaluate_with_cv, but
        the splitter requires `groups` as third argument.
        """
        results = {
            "model_metrics": {},
            "confusion_matrices": {},
            "misclassified_samples": {},
        }

        X_array = X.values if isinstance(X, pd.DataFrame) else X

        for name, model in models.items():
            y_true_all, y_pred_all, misclassified = [], [], []

            for train_idx, test_idx in cv.split(X_array, y, groups):
                X_tr, X_te = X_array[train_idx], X_array[test_idx]
                y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
                test_indices = y.iloc[test_idx].index.tolist()

                # scale + select on training fold only
                scaler = StandardScaler().fit(X_tr)
                X_tr_s = scaler.transform(X_tr)
                X_te_s = scaler.transform(X_te)

                selector = SelectKBest(f_classif, k=self.n_features_to_select).fit(
                    X_tr_s, y_tr
                )
                X_tr_sel = selector.transform(X_tr_s)
                X_te_sel = selector.transform(X_te_s)

                model.fit(X_tr_sel, y_tr)
                y_pred = model.predict(X_te_sel)

                y_true_all.extend(y_te.tolist())
                y_pred_all.extend(y_pred)

                for idx, t, p in zip(test_indices, y_te.tolist(), y_pred):
                    if t != p:
                        misclassified.append(
                            {
                                "index": idx,
                                "true_label": t,
                                "predicted_label": p,
                                "session_left_out": groups[test_idx[0]],
                            }
                        )

            results["model_metrics"][name] = self._calculate_metrics(
                y_true_all, y_pred_all
            )
            results["confusion_matrices"][name] = confusion_matrix(
                y_true_all, y_pred_all
            )
            results["misclassified_samples"][name] = misclassified

            print_log(f"\n{name} (LOSO) Classification Report:")
            print_log(classification_report(y_true_all, y_pred_all))

        return results

    def evaluate_label_combinations(self):
        print_log("\n---- EVALUATING LABEL COMBINATIONS ----")
        # Set up models
        models = {
            "RandomForest": RandomForestClassifier(
                n_estimators=50,
                max_depth=2,
                class_weight="balanced",
                random_state=48,
                min_samples_leaf=2,
            ),
            "SVM": SVC(
                kernel="rbf",
                C=0.1,
                gamma="scale",
                class_weight="balanced",
                random_state=42,
                probability=True,
            ),
            # new ✱
            "ElasticNetLogReg": LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                class_weight="balanced",
                max_iter=5000,
                random_state=42,
                l1_ratio=0.5,
            ),
            "ShrinkageLDA": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            "ExtraTrees": ExtraTreesClassifier(
                n_estimators=100,
                max_depth=None,
                class_weight="balanced",
                random_state=48,
            ),
            "HGBClassifier": HistGradientBoostingClassifier(
                learning_rate=0.1, max_depth=3, class_weight="balanced", random_state=48
            ),
            "GaussianNB": GaussianNB(),
            "kNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        }

        search_spaces = self._define_search_spaces()
        for key, val in search_spaces.items():
            if key not in models:
                models.pop(key, None)
        # Choose CV method based on parameter
        if self.cv_method == "loo":
            print_log("Using Leave-One-Out cross validation")
            cv = LeaveOneOut()
            if self.cv_version == "simple":
                cv_method = self._evaluate_with_cv
            elif self.cv_version == "extended":
                cv_method = self._evaluate_with_cv_extended
        elif self.cv_method == "kfold":
            print_log(
                f"Using Stratified K-Fold cross validation with {self.kfold_splits} splits"
            )
            cv = StratifiedKFold(n_splits=self.kfold_splits)
            if self.cv_version == "simple":
                cv_method = self._evaluate_with_cv
            elif self.cv_version == "extended":
                cv_method = self._evaluate_with_cv_extended
        elif self.cv_method == "holdout":
            print_log(f"Using holdout validation with test_size={self.test_size}")
            cv = None
            cv_method = self._evaluate_with_holdout
        elif self.cv_method == "loso":
            print_log("Using Leave‑One‑SESSION‑Out CV")
            cv = LeaveOneGroupOut()
            # always use the *simple* evaluator, but with groups
            cv_method = self._evaluate_with_cv_loso

        elif self.cv_method == "lmoso":
            print_log(f"Using Leave-{self.lmoso_leftout}-Sessions-Out CV")
            cv = LeavePGroupsOut(self.lmoso_leftout)
            cv_method = self._evaluate_with_cv_loso  # reuse the same evaluator

        else:
            raise ValueError(
                "Invalid cv_method provided. Choose 'loo', 'kfold', 'lmoso' or 'holdout'"
            )

        # Iterate over all possible label combinations (pairs or triplets)
        for label_combo in combinations(self.unique_labels, self.top_n_labels):
            df_subset = self.df[self.df["label"].isin(label_combo)]
            X_subset = df_subset[self.feature_columns]
            y_subset = df_subset["label"]

            # Print sample counts for this combination
            combo_sample_count = df_subset.groupby("label").size()
            print_log(f"\nLabel combination {label_combo}: {len(df_subset)} samples")
            print_log(f"  Per label: {combo_sample_count.to_dict()}")

            # FIX: Pass raw data to cv_method, which will handle scaling and feature selection properly
            # ---- call the correct evaluator ----
            if self.cv_method in ("loso", "lmoso"):
                groups = df_subset["session"].values  # 1‑D array of session IDs
                results = cv_method(X_subset, y_subset, models, cv, groups)
            else:
                results = cv_method(X_subset, y_subset, models, cv)

            # Store results
            best_model = max(
                results["model_metrics"],
                key=lambda m: results["model_metrics"][m]["f1_macro"],
            )
            best_score = results["model_metrics"][best_model]["accuracy"]

            self.separability_scores[label_combo] = best_score
            self.detailed_results[label_combo] = {
                "best_model": best_model,
                "metrics": results["model_metrics"],
                "confusion_matrices": results.get("confusion_matrices", {}),
                "misclassified_samples": results.get("misclassified_samples", {}),
                "n_samples": len(y_subset),
                "sample_counts": combo_sample_count.to_dict(),
            }

            # Print detailed results
            print_log(f"  Best model: {best_model}")
            for metric, value in results["model_metrics"][best_model].items():
                print_log(f"    {metric}: {value:.4f}")

        # Find best label combination based on F1 score
        self.best_labels = max(
            self.detailed_results,
            key=lambda c: self.detailed_results[c]["metrics"][
                self.detailed_results[c]["best_model"]
            ]["f1_macro"],
        )

        print_log(
            f"\nThe {self.top_n_labels} most diverse labels are: {self.best_labels}"
        )
        best_model = self.detailed_results[self.best_labels]["best_model"]
        print_log(f"Best model: {best_model}")

        metrics = self.detailed_results[self.best_labels]["metrics"][best_model]
        for metric, value in metrics.items():
            print_log(f"{metric}: {value:.4f}")

        print_log(
            f"Total samples: {self.detailed_results[self.best_labels]['n_samples']}"
        )

    def _evaluate_with_cv(self, X, y, models, cv):
        """Evaluate models with cross-validation - fixed to avoid information leaks"""
        results = {
            "model_metrics": {},
            "confusion_matrices": {},
            "misclassified_samples": {},
        }

        for name, model in models.items():
            y_true_all = []
            y_pred_all = []
            misclassified_samples = []

            # Convert X to numpy array if it's not already (for easier indexing)
            X_array = X if isinstance(X, np.ndarray) else X.values

            # Perform cross-validation
            for train_idx, test_idx in cv.split(X_array, y):
                # Get indices for this fold
                X_train, X_test = X_array[train_idx], X_array[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                test_indices = y.iloc[test_idx].index.tolist()

                # FIX 1: Fit the scaler only on training data for this fold
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # FIX 2: Perform feature selection only on training data for this fold
                selector = SelectKBest(f_classif, k=self.n_features_to_select)
                X_train_selected = selector.fit_transform(X_train_scaled, y_train)

                # Get the selected feature indices and transform test data
                selected_indices = selector.get_support(indices=True)
                X_test_selected = X_test_scaled[:, selected_indices]

                # Train model on this fold's training data
                model.fit(X_train_selected, y_train)
                y_pred = model.predict(X_test_selected)

                y_true_all.extend(y_test.tolist())
                y_pred_all.extend(y_pred)

                # Record misclassified samples
                for idx, true_val, pred_val in zip(
                    test_indices, y_test.tolist(), y_pred
                ):
                    if true_val != pred_val:
                        misclassified_samples.append(
                            {
                                "index": idx,
                                "true_label": true_val,
                                "predicted_label": pred_val,
                            }
                        )

            # Calculate metrics on all predictions
            results["model_metrics"][name] = self._calculate_metrics(
                y_true_all, y_pred_all
            )
            results["confusion_matrices"][name] = confusion_matrix(
                y_true_all, y_pred_all
            )
            results["misclassified_samples"][name] = misclassified_samples

            print_log(f"\n{name} Classification Report:")
            print_log(classification_report(y_true_all, y_pred_all))

        return results

    def _evaluate_with_cv_extended(self, X, y, models, cv):
        """
        Evaluate models with cross-validation using ensemble feature selection

        Args:
            X: Feature matrix
            y: Target labels
            models: Dictionary of models to evaluate
            cv: Cross-validation splitter

        Returns:
            dict: Results including metrics, confusion matrices, and misclassified samples
        """
        results = {
            "model_metrics": {},
            "confusion_matrices": {},
            "misclassified_samples": {},
            "feature_importance": {},
        }

        # Convert X to numpy array if it's not already (for easier indexing)
        X_array = X if isinstance(X, np.ndarray) else X.values
        X_df = X if isinstance(X, pd.DataFrame) else pd.DataFrame(X)

        # Track which features were selected in each fold for analysis
        feature_selection_frequency = {col: 0 for col in X_df.columns}

        for name, model in models.items():
            y_true_all = []
            y_pred_all = []
            misclassified_samples = []
            fold_importances = []

            # Perform cross-validation
            for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_array, y)):
                # Get indices for this fold
                X_train, X_test = X_array[train_idx], X_array[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                test_indices = y.iloc[test_idx].index.tolist()

                # Convert train data to DataFrame for feature selection
                X_train_df = pd.DataFrame(X_train, columns=X_df.columns)

                # Create temporary DataFrame with required structure for importance methods
                temp_df = X_train_df.copy()
                temp_df["label"] = y_train.values
                temp_df["channel"] = "combined"  # Placeholder
                temp_df["session"] = 0  # Placeholder

                # Calculate feature importance using multiple methods
                anova_importance = self.calculate_anova_importance(temp_df)
                mi_importance = self.calculate_mutual_info_importance(temp_df)
                rf_importance = self.calculate_random_forest_importance(temp_df)

                # Combine importance scores
                ensemble_importance = self.calculate_ensemble_importance(
                    anova_importance, mi_importance, rf_importance
                )

                # Store feature importance for this fold
                fold_importances.append(ensemble_importance)

                # Get the top N features
                top_features = (
                    ensemble_importance["feature"]
                    .head(self.n_features_to_select)
                    .tolist()
                )

                # Update selection frequency
                for feature in top_features:
                    feature_selection_frequency[feature] += 1

                # Scale the data
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Select the identified top features
                X_train_selected = X_train_scaled[
                    :, [list(X_df.columns).index(feat) for feat in top_features]
                ]
                X_test_selected = X_test_scaled[
                    :, [list(X_df.columns).index(feat) for feat in top_features]
                ]

                # Train model on this fold's training data
                model.fit(X_train_selected, y_train)
                y_pred = model.predict(X_test_selected)

                y_true_all.extend(y_test.tolist())
                y_pred_all.extend(y_pred)

                # Record misclassified samples
                for idx, true_val, pred_val in zip(
                    test_indices, y_test.tolist(), y_pred
                ):
                    if true_val != pred_val:
                        misclassified_samples.append(
                            {
                                "index": idx,
                                "true_label": true_val,
                                "predicted_label": pred_val,
                                "fold": fold_idx,
                            }
                        )

            # Aggregate feature importance across folds
            all_features = set()
            for fold_importance in fold_importances:
                all_features.update(fold_importance["feature"].tolist())

            # Create overall feature importance by averaging across folds
            overall_importance = []
            for feature in all_features:
                scores = [
                    fold_imp.loc[
                        fold_imp["feature"] == feature, "ensemble_score"
                    ].values[0]
                    for fold_imp in fold_importances
                    if feature in fold_imp["feature"].values
                ]

                overall_importance.append(
                    {
                        "feature": feature,
                        "avg_ensemble_score": sum(scores) / len(scores),
                        "selection_frequency": feature_selection_frequency[feature]
                        / len(fold_importances),
                    }
                )

            # Sort by average importance
            overall_importance_df = pd.DataFrame(overall_importance)
            overall_importance_df = overall_importance_df.sort_values(
                "avg_ensemble_score", ascending=False
            )

            # Calculate metrics on all predictions
            results["model_metrics"][name] = self._calculate_metrics(
                y_true_all, y_pred_all
            )
            results["confusion_matrices"][name] = confusion_matrix(
                y_true_all, y_pred_all
            )
            results["misclassified_samples"][name] = misclassified_samples
            results["feature_importance"][name] = overall_importance_df

            print_log(f"\n{name} Classification Report:")
            print_log(classification_report(y_true_all, y_pred_all))

            # Print top features by average importance
            print_log(f"\nTop {min(10, len(overall_importance_df))} Features:")
            for i, (_, row) in enumerate(overall_importance_df.head(10).iterrows()):
                print_log(
                    f"{i + 1}. {row['feature']} - Score: {row['avg_ensemble_score']:.4f}, "
                    + f"Selected in {int(row['selection_frequency'] * len(fold_importances))}/{len(fold_importances)} folds"
                )

        return results




    def calculate_anova_importance(self, df):
        """
        Calculate feature importance using ANOVA F-value (univariate feature selection).

        Args:
            df (pd.DataFrame): Input dataframe with features and labels

        Returns:
            pd.DataFrame: Dataframe with feature names and importance scores
        """
        # Prepare data
        X = df.drop(["label", "channel", "session"], axis=1)
        y = df["label"]

        # Calculate ANOVA F-values
        selector = SelectKBest(score_func=f_classif, k="all")
        selector.fit(X, y)

        # Create importance dataframe
        importance_df = pd.DataFrame(
            {
                "feature": X.columns,
                "f_value": selector.scores_,
                "p_value": selector.pvalues_,
            }
        )

        # Sort by importance (higher F-value = more important)
        importance_df = importance_df.sort_values(
            "f_value", ascending=False
        ).reset_index(drop=True)

        # Add normalized importance (0-100 scale)
        if importance_df["f_value"].max() > 0:
            importance_df["importance"] = (
                100.0 * importance_df["f_value"] / importance_df["f_value"].max()
            )
        else:
            importance_df["importance"] = 0

        return importance_df

    def calculate_mutual_info_importance(self, df):
        """
        Calculate feature importance using mutual information.

        Args:
            df (pd.DataFrame): Input dataframe with features and labels

        Returns:
            pd.DataFrame: Dataframe with feature names and importance scores
        """
        # Prepare data
        X = df.drop(["label", "channel", "session"], axis=1)
        y = df["label"]

        # Scale features (recommended for mutual information)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Calculate mutual information
        mi_scores = mutual_info_classif(X_scaled, y, random_state=42)

        # Create importance dataframe
        importance_df = pd.DataFrame({"feature": X.columns, "mutual_info": mi_scores})

        # Sort by importance (higher MI = more important)
        importance_df = importance_df.sort_values(
            "mutual_info", ascending=False
        ).reset_index(drop=True)

        # Add normalized importance (0-100 scale)
        if importance_df["mutual_info"].max() > 0:
            importance_df["importance"] = (
                100.0
                * importance_df["mutual_info"]
                / importance_df["mutual_info"].max()
            )
        else:
            importance_df["importance"] = 0

        return importance_df

    def calculate_random_forest_importance(self, df):
        """
        Calculate feature importance using Random Forest.

        Args:
            df (pd.DataFrame): Input dataframe with features and labels

        Returns:
            pd.DataFrame: Dataframe with feature names and importance scores
        """
        # Prepare data
        X = df.drop(["label", "channel", "session"], axis=1)
        y = df["label"]

        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)

        # Get feature importances
        importances = rf.feature_importances_

        # Create importance dataframe
        importance_df = pd.DataFrame(
            {"feature": X.columns, "rf_importance": importances}
        )

        # Sort by importance (higher = more important)
        importance_df = importance_df.sort_values(
            "rf_importance", ascending=False
        ).reset_index(drop=True)

        # Add normalized importance (0-100 scale)
        importance_df["importance"] = (
            100.0
            * importance_df["rf_importance"]
            / importance_df["rf_importance"].max()
        )

        return importance_df

    # ---------------------------------------------------------------------
    #  Helper: Ensemble importance is replaced with this class to be used in
    #  gold nested cv
    # ---------------------------------------------------------------------
    class EnsembleFilter:
        """
        A drop-in replacement for SelectKBest that ranks features with the
        calculate_ensemble_importance() routine you already have.
        """

        def __init__(self, k):
            self.k = k
            self.selected_idx_ = None  # indices in original feature order

        def fit(self, X, y):
            # X is numpy array here (after scaling); y is 1-d array / Series
            # Build a tiny dataframe to re-use your existing functions
            df_tmp = pd.DataFrame(X, columns=self.feature_names_)
            df_tmp["label"] = y
            df_tmp["channel"] = "dummy"
            df_tmp["session"] = 0

            anova = self.calculate_anova_importance(df_tmp)
            mi = self.calculate_mutual_info_importance(df_tmp)
            rf = self.calculate_random_forest_importance(df_tmp)

            ens = self.calculate_ensemble_importance(anova, mi, rf)

            top_feats = ens.head(self.k)["feature"].tolist()
            self.selected_idx_ = [self.feature_names_.index(f) for f in top_feats]
            return self

        def transform(self, X):
            if self.selected_idx_ is None:
                raise RuntimeError("Call fit() first")
            return X[:, self.selected_idx_]

        # scikit-learn compatibility helpers
        def get_support(self, indices=False):
            mask = np.zeros(len(self.feature_names_), dtype=bool)
            mask[self.selected_idx_] = True
            return self.selected_idx_ if indices else mask

    def calculate_ensemble_importance(self, anova_df, mi_df, rf_df):
        """
        Calculate ensemble importance by combining multiple methods.

        Args:
            anova_df (pd.DataFrame): ANOVA importance dataframe
            mi_df (pd.DataFrame): Mutual information importance dataframe
            rf_df (pd.DataFrame): Random forest importance dataframe

        Returns:
            pd.DataFrame: Dataframe with feature names and ensemble importance scores
        """
        # Normalize individual method scores to 0-1 scale
        anova_norm = anova_df.copy()
        anova_norm["norm_score"] = anova_norm["importance"] / 100.0

        mi_norm = mi_df.copy()
        mi_norm["norm_score"] = mi_norm["importance"] / 100.0

        rf_norm = rf_df.copy()
        rf_norm["norm_score"] = rf_norm["importance"] / 100.0

        # Create mappings of feature to normalized score
        anova_map = dict(zip(anova_norm["feature"], anova_norm["norm_score"]))
        mi_map = dict(zip(mi_norm["feature"], mi_norm["norm_score"]))
        rf_map = dict(zip(rf_norm["feature"], rf_norm["norm_score"]))

        # Get all features
        all_features = set(anova_map.keys()) | set(mi_map.keys()) | set(rf_map.keys())

        # Create ensemble scores
        ensemble_data = []
        for feature in all_features:
            # Get score from each method (default to 0 if not available)
            anova_score = anova_map.get(feature, 0.0)
            mi_score = mi_map.get(feature, 0.0)
            rf_score = rf_map.get(feature, 0.0)

            # Calculate ensemble score (weighted average)
            # Weights can be adjusted based on which method you trust more
            ensemble_score = 0.3 * anova_score + 0.3 * mi_score + 0.4 * rf_score

            ensemble_data.append(
                {
                    "feature": feature,
                    "anova_score": anova_score,
                    "mi_score": mi_score,
                    "rf_score": rf_score,
                    "ensemble_score": ensemble_score,
                }
            )

        # Create dataframe and sort by ensemble score
        ensemble_df = pd.DataFrame(ensemble_data)
        ensemble_df = ensemble_df.sort_values(
            "ensemble_score", ascending=False
        ).reset_index(drop=True)

        # Add normalized importance (0-100 scale)
        if ensemble_df["ensemble_score"].max() > 0:
            ensemble_df["importance"] = (
                100.0
                * ensemble_df["ensemble_score"]
                / ensemble_df["ensemble_score"].max()
            )
        else:
            ensemble_df["importance"] = 0

        return ensemble_df

    def _evaluate_with_holdout(self, X, y, models, _):
        """Evaluate models with holdout validation - fixed to avoid information leaks"""
        results = {
            "model_metrics": {},
            "confusion_matrices": {},
            "misclassified_samples": {},
        }

        # Split data once
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42, stratify=y
        )

        # Get the original indices for the test set
        test_indices = y_test.index.tolist()

        for name, model in models.items():
            # FIX 1: Fit the scaler only on training data
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # FIX 2: Perform feature selection only on training data
            selector = SelectKBest(f_classif, k=self.n_features_to_select)
            X_train_selected = selector.fit_transform(X_train_scaled, y_train)

            # Get the selected feature indices and transform test data
            selected_indices = selector.get_support(indices=True)
            X_test_selected = X_test_scaled[:, selected_indices]

            # Train the model and predict
            model.fit(X_train_selected, y_train)
            y_pred = model.predict(X_test_selected)

            results["model_metrics"][name] = self._calculate_metrics(
                y_test.tolist(), y_pred
            )
            results["confusion_matrices"][name] = confusion_matrix(
                y_test.tolist(), y_pred
            )

            misclassified_samples = []
            for idx, true_val, pred_val in zip(test_indices, y_test.tolist(), y_pred):
                if true_val != pred_val:
                    misclassified_samples.append(
                        {
                            "index": idx,
                            "true_label": true_val,
                            "predicted_label": pred_val,
                        }
                    )
            results["misclassified_samples"][name] = misclassified_samples

            print_log(f"\n{name} Classification Report:")
            print_log(classification_report(y_test.tolist(), y_pred))

        return results

    def export_misclassified_samples(self):
        """Export misclassified samples and calculate session-level accuracy."""
        print_log("\n---- EXPORTING MISCLASSIFIED SAMPLES & SESSION ACCURACY ----")

        for combo in self.detailed_results:
            combo_str = "_".join(combo)
            df_combo = self.df[
                self.df["label"].isin(combo)
            ]  # Get data for this label combo

            # Only proceed if sessions exist in the data
            if "session" not in df_combo.columns:
                print_log(
                    f"No session data for {combo_str}. Skipping session accuracy."
                )
                continue

            misclassified_dict = self.detailed_results[combo].get(
                "misclassified_samples", {}
            )

            for model, misclassified in misclassified_dict.items():
                if not misclassified:
                    continue

                # Export misclassified samples
                mis_df = pd.DataFrame(misclassified)
                merged = pd.merge(
                    mis_df,
                    self.df.reset_index(),
                    left_on="index",
                    right_on="index",
                    how="left",
                )
                filename = f"{self.run_directory}/misclassified_{combo_str}_{model}.csv"
                merged.to_csv(filename, index=False)

                # --- New: Calculate session accuracy ---
                session_accuracies = []
                for session in df_combo["session"].unique():
                    # Total samples in this session for current label combination
                    total_samples = df_combo[df_combo["session"] == session].shape[0]

                    # Misclassified samples in this session
                    mis_in_session = merged[merged["session"] == session].shape[0]

                    # Calculate accuracy
                    accuracy = (
                        (total_samples - mis_in_session) / total_samples
                        if total_samples > 0
                        else 0
                    )

                    session_accuracies.append(
                        {
                            "session": session,
                            "total_samples": total_samples,
                            "misclassified": mis_in_session,
                            "accuracy": accuracy,
                        }
                    )

                # Save session accuracy results
                session_acc_df = pd.DataFrame(session_accuracies)
                session_filename = (
                    f"{self.run_directory}/session_accuracy_{combo_str}_{model}.csv"
                )
                session_acc_df.to_csv(session_filename, index=False)

                print_log(
                    f"Exported session accuracy for {combo_str} ({model}) to '{session_filename}'"
                )

    def _calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive performance metrics"""
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro"
        )
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            y_true, y_pred, average="weighted"
        )

        # For binary classification, also calculate class-specific metrics
        if len(np.unique(y_true)) == 2:
            precision_class, recall_class, f1_class, _ = (
                precision_recall_fscore_support(y_true, y_pred, average=None)
            )
            class_metrics = {}
            for i, label in enumerate(np.unique(y_true)):
                class_metrics[f"precision_{label}"] = precision_class[i]
                class_metrics[f"recall_{label}"] = recall_class[i]
                class_metrics[f"f1_{label}"] = f1_class[i]
        else:
            class_metrics = {}

        # Calculate accuracy
        accuracy = np.mean(np.array(y_true) == np.array(y_pred))

        metrics = {
            "accuracy": accuracy,
            "precision_macro": precision,
            "recall_macro": recall,
            "f1_macro": f1,
            "precision_weighted": precision_w,
            "recall_weighted": recall_w,
            "f1_weighted": f1_w,
            **class_metrics,
        }

        return metrics

    # ── NEW helper ────────────────────────────────────────────────────────────────
    def _persist_nested_cv_log(self, model_name, outer_results):
        """
        outer_results: list[dict] with keys
            'fold', 'inner_best_f1', 'outer_f1', 'inner_best_params'
        """
        import pandas as pd, os, json, numpy as np

        df = pd.DataFrame(outer_results)

        # ----- weighted mean / std using n_samples -----
        w = df["n_samples"].to_numpy()
        mu_w = np.average(df["outer_f1"], weights=w)
        sigma_w = np.sqrt(np.average((df["outer_f1"] - mu_w) ** 2, weights=w))

        # add global summary (last row)
        df_summary = pd.DataFrame(
            [
                {
                    "fold": "MEAN±STD",
                    "inner_best_f1": f"{np.average(df.inner_best_f1, weights=w):.3f} ± "
                    f"{np.sqrt(np.average((df.inner_best_f1 -np.average(df.inner_best_f1, weights=w)) ** 2,weights=w)):.3f}",
                    "outer_f1": f"{mu_w:.3f} ± {sigma_w:.3f}",
                    "inner_best_params": "",
                }
            ]
        )
        out_df = pd.concat([df, df_summary], ignore_index=True)

        # pretty-print params as JSON
        out_df["inner_best_params"] = out_df["inner_best_params"].apply(
            lambda p: json.dumps(p, default=str) if isinstance(p, dict) else p
        )

        out_file = os.path.join(
            self.run_directory, f"nested_cv_log_{model_name.lower()}.csv"
        )
        out_df.to_csv(out_file, index=False)
        print_log(f"✓ Saved nested-CV log for {model_name} → {out_file}")

    def visualize_confusion_matrix(self):
        """Visualize confusion matrix for the best label combination"""
        print_log("\n---- VISUALIZING CONFUSION MATRIX ----")

        if "confusion_matrices" not in self.detailed_results[self.best_labels]:
            print_log("Confusion matrix not available. Skip visualization.")
            return

        best_model = self.detailed_results[self.best_labels]["best_model"]
        cm = self.detailed_results[self.best_labels]["confusion_matrices"][best_model]
        labels = list(self.best_labels)

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f'Confusion Matrix for {best_model} on {", ".join(labels)}')

        cm_file = f"{self.run_directory}/{self.channel_approach}_confusion_matrix_{self.cv_method}.png"
        plt.savefig(cm_file, dpi=300, bbox_inches="tight")
        print_log(f"Confusion matrix saved as '{cm_file}'")
        plt.close()

    # [confidence_ellipse, visualize_pca, visualize_feature_importance, analyze_channel_distribution methods remain unchanged]
    @staticmethod
    def confidence_ellipse(x, y, ax, n_std=2.0, facecolor="none", **kwargs):
        """
        Create a plot of the covariance confidence ellipse of x and y.
        """
        if x.size != y.size:
            raise ValueError("x and y must be the same size")
        cov = np.cov(x, y)
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = patches.Ellipse(
            (0, 0),
            width=ell_radius_x * 2,
            height=ell_radius_y * 2,
            facecolor=facecolor,
            **kwargs,
        )
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x)
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y)
        transf = (
            transforms.Affine2D()
            .rotate_deg(45)
            .scale(scale_x, scale_y)
            .translate(mean_x, mean_y)
        )
        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)

    def _get_best_model_selected_features(self):
        best_model = self.optimization_results["best_model"]
        feature_selector = best_model.named_steps["feature_selection"]
        selected_indices = feature_selector.get_support(indices=True)
        selected_features = [self.feature_columns[i] for i in selected_indices]

        df_best = self.df[self.df["label"].isin(self.best_labels)]
        X_best_full = df_best[self.feature_columns]
        y_best = df_best["label"]

        X_transformed = best_model.named_steps["scaler"].transform(X_best_full)
        X_selected = best_model.named_steps["feature_selection"].transform(
            X_transformed
        )

        return X_selected, y_best, df_best

    def visualize_pca(self):
        print_log(
            "\n---- VISUALIZING INTERACTIVE 2D PCA (both label-colored and distance-colored) ----"
        )

        if not hasattr(self, "optimization_results"):
            print_log("Run optimize_hyperparameters first.")
            return

        # 1) Prepare data
        X_selected, y_best, df_best = self._get_best_model_selected_features()

        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_selected)
        explained_variance = pca.explained_variance_ratio_ * 100

        df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        df_pca["label"] = y_best.values
        if "channel" in self.df.columns:
            df_pca["channel"] = df_best["channel"].values

        # ✨ Compute distance from center for dynamic color
        df_pca["distance_from_center"] = (
            df_pca["PC1"] ** 2 + df_pca["PC2"] ** 2
        ) ** 0.5

        # 2) Create first figure: color by label
        fig_label = px.scatter(
            df_pca,
            x="PC1",
            y="PC2",
            color="label",
            symbol="channel" if "channel" in df_pca.columns else None,
            hover_data=["label"] + (["channel"] if "channel" in df_pca.columns else []),
            title=f"PCA (2D) – Colored by Label\nPC1 {explained_variance[0]:.2f}%, PC2 {explained_variance[1]:.2f}%",
            labels={
                "PC1": f"PC1 ({explained_variance[0]:.2f}%)",
                "PC2": f"PC2 ({explained_variance[1]:.2f}%)",
            },
            width=1000,
            height=800,
            template="plotly_white",
        )
        fig_label.update_traces(
            marker=dict(size=8, line=dict(width=0.5, color="DarkSlateGrey"))
        )

        # Activate if you want circles around labels
        # label_groups = df_pca.groupby("label")
        # for label, group in label_groups:
        #     # Compute centroid
        #     center_x = group["PC1"].mean()
        #     center_y = group["PC2"].mean()
        #
        #     # Compute radius: std deviation or max distance from center
        #     radius = (
        #             ((group["PC1"] - center_x) ** 2 + (group["PC2"] - center_y) ** 2).mean() ** 0.5
        #     )
        #
        #     # Generate circle
        #     theta = np.linspace(0, 2 * np.pi, 100)
        #     circle_x = center_x + radius * np.cos(theta)
        #     circle_y = center_y + radius * np.sin(theta)
        #
        #     # Add to plot
        #     fig_label.add_scatter(
        #         x=circle_x,
        #         y=circle_y,
        #         mode="lines",
        #         line=dict(dash="dash", width=1),
        #         name=f"Circle: {label}",
        #         showlegend=True,
        #     )

        # Save
        output_file_label = f"{self.run_directory}/interactive_2d_pca_by_label_top{self.top_n_labels}_labels_top{self.n_features_to_select}_features_{self.cv_method}.html"
        fig_label.write_html(output_file_label)
        print_log(f"Saved label-colored 2D PCA → '{output_file_label}'")

        # 3) Create second figure: color by distance
        fig_dist = px.scatter(
            df_pca,
            x="PC1",
            y="PC2",
            color="distance_from_center",
            color_continuous_scale="Viridis",
            symbol="channel" if "channel" in df_pca.columns else None,
            hover_data=["label"] + (["channel"] if "channel" in df_pca.columns else []),
            title=f"PCA (2D) – Colored by Distance from Center\nPC1 {explained_variance[0]:.2f}%, PC2 {explained_variance[1]:.2f}%",
            labels={
                "PC1": f"PC1 ({explained_variance[0]:.2f}%)",
                "PC2": f"PC2 ({explained_variance[1]:.2f}%)",
                "distance_from_center": "Distance from Origin",
            },
            width=1000,
            height=800,
            template="plotly_white",
        )
        fig_dist.update_traces(
            marker=dict(size=8, line=dict(width=0.5, color="DarkSlateGrey"))
        )
        fig_dist.update_coloraxes(colorbar_title="Distance")

        # Save
        output_file_dist = f"{self.run_directory}/interactive_2d_pca_by_distance_top{self.top_n_labels}_labels_top{self.n_features_to_select}_features_{self.cv_method}.html"
        fig_dist.write_html(output_file_dist)
        print_log(f"Saved distance-colored 2D PCA → '{output_file_dist}'")

    def visualize_pca_3d(self):
        print_log(
            "\n---- VISUALIZING INTERACTIVE 3D PCA (both label-colored and distance-colored) ----"
        )

        if not hasattr(self, "optimization_results"):
            print_log("Run optimize_hyperparameters first.")
            return

        # 1) Prepare data
        X_selected, y_best, df_best = self._get_best_model_selected_features()

        pca = PCA(n_components=3)
        X_pca_3d = pca.fit_transform(X_selected)
        explained_var = pca.explained_variance_ratio_ * 100

        df_pca_3d = pd.DataFrame(X_pca_3d, columns=["PC1", "PC2", "PC3"])
        df_pca_3d["label"] = y_best.values
        if "channel" in self.df.columns:
            df_pca_3d["channel"] = df_best["channel"].values

        # ✨ Compute distance from center
        df_pca_3d["distance_from_center"] = (
            df_pca_3d["PC1"] ** 2 + df_pca_3d["PC2"] ** 2 + df_pca_3d["PC3"] ** 2
        ) ** 0.5

        # 2) Create first figure: color by label
        fig_label = px.scatter_3d(
            df_pca_3d,
            x="PC1",
            y="PC2",
            z="PC3",
            color="label",
            symbol="channel" if "channel" in df_pca_3d.columns else None,
            hover_data=["label"]
            + (["channel"] if "channel" in df_pca_3d.columns else []),
            title=f"PCA (3D) – Colored by Label\nPC1 {explained_var[0]:.2f}%, PC2 {explained_var[1]:.2f}%, PC3 {explained_var[2]:.2f}%",
            labels={
                "PC1": f"PC1 ({explained_var[0]:.2f}%)",
                "PC2": f"PC2 ({explained_var[1]:.2f}%)",
                "PC3": f"PC3 ({explained_var[2]:.2f}%)",
            },
            width=1000,
            height=800,
            template="plotly_white",
        )
        fig_label.update_traces(
            marker=dict(size=5, line=dict(width=0.5, color="DarkSlateGrey"))
        )

        # Activate if you want the spheres around labels
        # for label, group in df_pca_3d.groupby("label"):
        #     # Center
        #     cx = group["PC1"].mean()
        #     cy = group["PC2"].mean()
        #     cz = group["PC3"].mean()
        #
        #     # Radius (e.g. stddev-based)
        #     radius = (
        #             ((group["PC1"] - cx) ** 2 + (group["PC2"] - cy) ** 2 + (group["PC3"] - cz) ** 2).mean() ** 0.5
        #     )
        #
        #     # Sphere coordinates
        #     phi = np.linspace(0, np.pi, 15)
        #     theta = np.linspace(0, 2 * np.pi, 30)
        #     phi, theta = np.meshgrid(phi, theta)
        #     x_sphere = cx + radius * np.sin(phi) * np.cos(theta)
        #     y_sphere = cy + radius * np.sin(phi) * np.sin(theta)
        #     z_sphere = cz + radius * np.cos(phi)
        #     import plotly.graph_objects as go
        #     fig_label.add_trace(go.Surface(
        #         x=x_sphere,
        #         y=y_sphere,
        #         z=z_sphere,
        #         opacity=0.2,
        #         showscale=False,
        #         name=f"Sphere: {label}",
        #         hoverinfo='skip',
        #         colorscale=[[0, 'lightgray'], [1, 'lightblue']]
        #     ))

        # Save
        output_file_label = f"{self.run_directory}/interactive_3d_pca_by_label_top{self.top_n_labels}_labels_top{self.n_features_to_select}_features_{self.cv_method}.html"
        fig_label.write_html(output_file_label)
        print_log(f"Saved label-colored 3D PCA → '{output_file_label}'")

        # 3) Create second figure: color by distance
        fig_dist = px.scatter_3d(
            df_pca_3d,
            x="PC1",
            y="PC2",
            z="PC3",
            color="distance_from_center",
            color_continuous_scale="Viridis",
            symbol="channel" if "channel" in df_pca_3d.columns else None,
            hover_data=["label"]
            + (["channel"] if "channel" in df_pca_3d.columns else []),
            title=f"PCA (3D) – Colored by Distance from Center\nPC1 {explained_var[0]:.2f}%, PC2 {explained_var[1]:.2f}%, PC3 {explained_var[2]:.2f}%",
            labels={
                "PC1": f"PC1 ({explained_var[0]:.2f}%)",
                "PC2": f"PC2 ({explained_var[1]:.2f}%)",
                "PC3": f"PC3 ({explained_var[2]:.2f}%)",
                "distance_from_center": "Distance from Origin",
            },
            width=1000,
            height=800,
            template="plotly_white",
        )
        fig_dist.update_traces(
            marker=dict(size=5, line=dict(width=0.5, color="DarkSlateGrey"))
        )
        fig_dist.update_coloraxes(colorbar_title="Distance")

        # Save
        output_file_dist = f"{self.run_directory}/interactive_3d_pca_by_distance_top{self.top_n_labels}_labels_top{self.n_features_to_select}_features_{self.cv_method}.html"
        fig_dist.write_html(output_file_dist)
        print_log(f"Saved distance-colored 3D PCA → '{output_file_dist}'")

    def visualize_feature_importance(self):
        print_log("\n---- VISUALIZING FEATURE IMPORTANCE ----")
        # Focus on data corresponding to the best label combination
        df_best = self.df[self.df["label"].isin(self.best_labels)]
        X_best = df_best[self.feature_columns]
        y_best = df_best["label"]

        # Standardize and compute ANOVA F-values
        X_best_scaled = self.scaler.transform(X_best)
        f_values, p_values = f_classif(X_best_scaled, y_best)
        feature_scores = pd.DataFrame(
            {
                "Feature": self.feature_columns,
                "F_Score": f_values,
                "P_Value": p_values,
                "Log10_F": np.log10(f_values + 1),
            }
        )
        feature_scores = feature_scores.sort_values("F_Score", ascending=False)
        feature_scores["Significant"] = feature_scores["P_Value"] < 0.05
        feature_scores["Color"] = feature_scores["Significant"].map(
            {True: "darkblue", False: "lightblue"}
        )

        plt.figure(figsize=(14, 10))
        plt.barh(
            feature_scores["Feature"],
            feature_scores["Log10_F"],
            color=feature_scores["Color"],
        )
        plt.title(
            f"Feature Importance for Distinguishing Between {self.best_labels}",
            fontsize=16,
        )
        plt.xlabel(
            "Log10(F-Score+1) - Higher Values = More Discriminative", fontsize=12
        )
        plt.ylabel("EEG Feature", fontsize=12)
        plt.grid(axis="x", linestyle="--", alpha=0.7)

        # Draw significance threshold line if applicable
        sig_features = feature_scores[feature_scores["Significant"]]
        if not sig_features.empty:
            min_sig_log_f = np.log10(sig_features["F_Score"].min() + 1)
            plt.axvline(x=min_sig_log_f, color="red", linestyle="--", alpha=0.7)
            plt.text(
                min_sig_log_f + 0.1,
                1,
                "Significance Threshold (p<0.05)",
                rotation=90,
                color="red",
                verticalalignment="bottom",
            )

        # Annotate bar plot with F-scores and p-values
        for i, (_, row) in enumerate(feature_scores.iterrows()):
            plt.text(
                row["Log10_F"] + 0.1,
                i,
                f"F={row['F_Score']:.2f}, p={row['P_Value']:.4f}",
                va="center",
                fontsize=9,
            )

        plt.tight_layout()
        importance_file = f"{self.run_directory}/{self.channel_approach}_feature_importance_top{self.top_n_labels}_labels_top{self.n_features_to_select}_features_{self.cv_method}.png"
        plt.savefig(importance_file, dpi=300, bbox_inches="tight")
        print_log(f"Feature importance visualization saved as '{importance_file}'")
        plt.close()

    # Complete the analyze_channel_distribution method which was cut off
    def analyze_channel_distribution(self):
        print_log("\n---- ANALYZING CHANNEL DISTRIBUTION ----")
        if "channel" in self.df.columns and self.channel_approach == "pooled":
            label_channel_counts = pd.crosstab(self.df["label"], self.df["channel"])
            label_channel_pct = (
                label_channel_counts.div(label_channel_counts.sum(axis=1), axis=0) * 100
            )
            plt.figure(
                figsize=(len(self.unique_channels) * 1.5, len(self.unique_labels) * 1.2)
            )
            sns.heatmap(
                label_channel_pct,
                annot=label_channel_counts,
                fmt="d",
                cmap="YlGnBu",
                cbar_kws={"label": "Sample Percentage (%)"},
            )
            plt.title("Sample Distribution by Label and Channel", fontsize=16)
            plt.ylabel("Label", fontsize=12)
            plt.xlabel("Channel", fontsize=12)
            plt.tight_layout()

            # Save the channel distribution plot
            dist_file = f"{self.run_directory}/{self.channel_approach}_channel_distribution_{self.cv_method}.png"
            plt.savefig(dist_file, dpi=300, bbox_inches="tight")
            print_log(f"Channel distribution visualization saved as '{dist_file}'")
            plt.close()
        else:
            print_log(
                "Channel distribution analysis skipped (either no channel data or not using pooled approach)"
            )

    def visualize_metrics_comparison(self):
        """Visualize performance metrics comparison across label combinations"""
        print_log("\n---- VISUALIZING PERFORMANCE METRICS COMPARISON ----")

        # Prepare data for visualization
        metrics_data = []
        for combo in self.detailed_results:
            best_model = self.detailed_results[combo]["best_model"]
            metrics = self.detailed_results[combo]["metrics"][best_model]
            combo_str = ", ".join(combo)

            metrics_data.append(
                {
                    "Combination": combo_str,
                    "Accuracy": metrics["accuracy"],
                    "F1 Score": metrics["f1_macro"],
                    "Precision": metrics["precision_macro"],
                    "Recall": metrics["recall_macro"],
                }
            )

        # Convert to DataFrame and reshape for plotting
        metrics_df = pd.DataFrame(metrics_data)
        metrics_plot_df = pd.melt(
            metrics_df,
            id_vars=["Combination"],
            value_vars=["Accuracy", "F1 Score", "Precision", "Recall"],
            var_name="Metric",
            value_name="Score",
        )

        # Create the comparison plot
        plt.figure(figsize=(12, 8))
        g = sns.catplot(
            x="Combination",
            y="Score",
            hue="Metric",
            data=metrics_plot_df,
            kind="bar",
            height=6,
            aspect=1.5,
        )

        plt.title(
            "Performance Metrics Comparison Across Label Combinations", fontsize=16
        )
        plt.xlabel("Label Combination", fontsize=12)
        plt.ylabel("Score", fontsize=12)
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45, ha="right")
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()

        metrics_file = f"{self.run_directory}/{self.channel_approach}_metrics_comparison_{self.cv_method}.png"
        plt.savefig(metrics_file, dpi=300, bbox_inches="tight")
        print_log(f"Metrics comparison visualization saved as '{metrics_file}'")
        plt.close()

    def export_results(self):
        """Export detailed results to CSV files"""
        print_log("\n---- EXPORTING RESULTS ----")

        # Export metrics for all label combinations
        metrics_data = []
        for combo in self.detailed_results:
            best_model = self.detailed_results[combo]["best_model"]
            metrics = self.detailed_results[combo]["metrics"][best_model]
            combo_str = "_".join(combo)

            metrics_row = {
                "label_combination": combo_str,
                "best_model": best_model,
                "n_samples": self.detailed_results[combo]["n_samples"],
            }

            # Add sample counts per label
            for label, count in self.detailed_results[combo]["sample_counts"].items():
                metrics_row[f"samples_{label}"] = count

            # Add all metrics
            for metric, value in metrics.items():
                metrics_row[metric] = value

            metrics_data.append(metrics_row)

        # Save to CSV
        metrics_df = pd.DataFrame(metrics_data)
        metrics_file = f"{self.run_directory}/{self.channel_approach}_performance_metrics_{self.cv_method}.csv"
        metrics_df.to_csv(metrics_file, index=False)
        print_log(f"Performance metrics exported to '{metrics_file}'")

        # Export feature importance for best label combination
        df_best = self.df[self.df["label"].isin(self.best_labels)]
        X_best = df_best[self.feature_columns]
        y_best = df_best["label"]

        # Standardize and compute ANOVA F-values
        X_best_scaled = self.scaler.transform(X_best)
        f_values, p_values = f_classif(X_best_scaled, y_best)

        feature_scores = pd.DataFrame(
            {
                "feature": self.feature_columns,
                "f_score": f_values,
                "p_value": p_values,
                "selected": [f in self.selected_features for f in self.feature_columns],
            }
        )
        feature_scores = feature_scores.sort_values("f_score", ascending=False)

        features_file = f"{self.run_directory}/{self.channel_approach}_feature_importance_{self.cv_method}.csv"
        feature_scores.to_csv(features_file, index=False)
        print_log(f"Feature importance scores exported to '{features_file}'")
        self.export_misclassified_samples()

    # ---------------------------------------------------------------------
    #  Helper: export a nice CSV with weighted‑mean inner/outer scores
    # ---------------------------------------------------------------------
    # ---------------------------------------------------------------------
    #  Helper: export a nice CSV with weighted‑mean inner/outer scores
    # ---------------------------------------------------------------------
    def _export_cv_summary(self, model_logs: dict):
        """Save per‑model weighted mean/STD of inner & outer F1 to CSV."""
        import pandas as pd, numpy as np, os

        rows = []
        for model, log in model_logs.items():
            df = pd.DataFrame(log)
            w = df["n_samples"].to_numpy()
            # weighted means / stds
            mu_inner = np.average(df["inner_best_f1"], weights=w)
            mu_outer = np.average(df["outer_f1"], weights=w)
            sd_inner = np.sqrt(
                np.average((df["inner_best_f1"] - mu_inner) ** 2, weights=w)
            )
            sd_outer = np.sqrt(np.average((df["outer_f1"] - mu_outer) ** 2, weights=w))

            rows.append(
                {
                    "model": model,
                    "mean_inner_f1": mu_inner,
                    "std_inner_f1": sd_inner,
                    "mean_outer_f1": mu_outer,
                    "std_outer_f1": sd_outer,
                    "n_folds": len(df),
                    "n_total_samples": int(w.sum()),
                }
            )

        summary_df = pd.DataFrame(rows).sort_values("mean_outer_f1", ascending=False)
        out_csv = os.path.join(self.run_directory, "nested_cv_summary.csv")
        summary_df.to_csv(out_csv, index=False)
        print_log(f"✓ Saved nested‑CV summary → {out_csv}")
        return summary_df

    # ---------------------------------------------------------------------
    #  Helper: bar‑plot inner vs outer mean F1 for every model (annotated)
    # ---------------------------------------------------------------------
    def _plot_cv_summary(self, summary_df):
        import matplotlib.pyplot as plt, numpy as np, os

        idx = np.arange(len(summary_df))
        bar_w = 0.35
        plt.figure(figsize=(10, 6))
        ax = plt.gca()

        ax.bar(
            idx - bar_w / 2,
            summary_df["mean_inner_f1"],
            bar_w,
            label="Inner (best) F1",
            yerr=summary_df["std_inner_f1"],
            capsize=4,
        )
        ax.bar(
            idx + bar_w / 2,
            summary_df["mean_outer_f1"],
            bar_w,
            label="Outer (held‑out) F1",
            yerr=summary_df["std_outer_f1"],
            capsize=4,
        )

        # annotate bars with values
        for p in ax.patches:
            height = p.get_height()
            ax.text(
                p.get_x() + p.get_width() / 2,
                height + 0.01,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.7),
            )

        ax.set_xticks(idx)
        ax.set_xticklabels(summary_df["model"], rotation=45, ha="right")
        ax.set_ylabel("Macro‑F1")
        ax.set_ylim(0, 1)
        ax.set_title("Nested‑CV performance (inner vs outer)")
        ax.legend()
        plt.tight_layout()
        out_png = os.path.join(self.run_directory, "nested_cv_summary.png")
        plt.savefig(out_png, dpi=300)
        plt.close()
        print_log(f"✓ Saved nested‑CV summary plot → {out_png}")

    # ---------------------------------------------------------------------
    #  REPLACEMENT: optimise_hyperparameters now collects per‑model logs
    # ---------------------------------------------------------------------
    def optimize_hyperparameters(self, n_iter: int = 25):
        """Same logic as before, but now returns a nicer CSV/plot with inner vs outer F1."""
        from skopt import BayesSearchCV
        from skopt.space import Real, Integer, Categorical
        from tqdm import tqdm
        import numpy as np, joblib

        if not hasattr(self, "best_labels"):
            print_log("Run evaluate_label_combinations first to find the best labels")
            return

        df_best = self.df[self.df["label"].isin(self.best_labels)]

        X_best = df_best[self.feature_columns]
        y_best = df_best["label"]

        # --- define search spaces (identical to your previous dict) ---
        # [... keep the big search_spaces dict unchanged ...]
        search_spaces = (
            self._define_search_spaces()
        )

        # choose outer / inner splitters
        outer_cv, inner_cv_proto = self._get_nested_splitters(df_best["session"].values)

        best_results, all_results, model_logs = {}, {}, {}  # ← collect logs per model

        for model_name, space in tqdm(
            search_spaces.items(), desc="Optimising (nested CV)"
        ):
            print_log(f"\n⟹  Optimising {model_name}")
            outer_scores, fold_estimators, fold_sizes, outer_log, fold_acc = [], [], [], [],[]
            fold_true, fold_pred = [], []  # NEW

            # --- outer loop --------------------------------------------------
            outer_indices = outer_cv.split(
                X_best, y_best, groups=df_best["session"].values
            )
            for fold, (tr_idx, te_idx) in enumerate(outer_indices, start=1):
                X_tr, X_te = X_best.iloc[tr_idx], X_best.iloc[te_idx]
                y_tr, y_te = y_best.iloc[tr_idx], y_best.iloc[te_idx]

                inner_cv = list(
                    inner_cv_proto.split(
                        X_tr, y_tr, groups=df_best["session"].values[tr_idx]
                    )
                )

                pipe = Pipeline(
                    [
                        ("scaler", StandardScaler()),
                        ("feature_selection", SelectKBest(f_classif)),
                        ("model", RandomForestClassifier()),  # placeholder
                    ]
                )

                opt = BayesSearchCV(
                    pipe,
                    space,
                    n_iter=n_iter,
                    cv=inner_cv,
                    scoring="f1_macro",
                    n_jobs=-1,
                    random_state=42,
                    verbose=0,
                )
                opt.fit(X_tr, y_tr)

                y_pred = opt.predict(X_te)
                fold_true.append(y_te.to_numpy())  # NEW
                fold_pred.append(y_pred)  # NEW
                outer_f1 = f1_score(y_te, y_pred, average="macro")
                inner_best = opt.best_score_

                outer_scores.append(outer_f1)
                fold_sizes.append(len(te_idx))
                fold_estimators.append(opt.best_estimator_)

                outer_acc = accuracy_score(y_te, y_pred)
                fold_acc.append(outer_acc)  # inside the outer loop

                outer_log.append(
                    {
                        "fold": fold,
                        "n_samples": len(te_idx),
                        "inner_best_f1": inner_best,
                        "outer_f1": outer_f1,
                        "best_params": opt.best_params_,
                    }
                )
                print_log(
                    f"   fold {fold}: inner = {inner_best:.3f} | outer = {outer_f1:.3f}"
                )

            # ---------- weighted mean/std ----------
            mu = np.average(outer_scores, weights=fold_sizes)
            sig = np.sqrt(np.average((outer_scores - mu) ** 2, weights=fold_sizes))
            print_log(f"→ Nested‑CV F1 for {model_name}: {mu:.3f} ± {sig:.3f}")

            best_results[model_name] = {
                "outer_mean_f1": mu,
                "outer_std_f1": sig,
                "per_fold_f1": outer_scores,
                "per_fold_acc": fold_acc,  # ← store it here
                "per_fold_true": fold_true,  # NEW
                "per_fold_pred": fold_pred,  # NEW
                "best_fold_model": fold_estimators[int(np.argmax(outer_scores))],
            }
            all_results[model_name] = {
                "params": opt.cv_results_["params"],
                "scores": opt.cv_results_["mean_test_score"],
            }
            model_logs[model_name] = outer_log  # ← keep log

            # still save full per‑fold CSV for transparency
            self._persist_nested_cv_log(model_name, outer_log)
            joblib.dump(
                opt.best_estimator_,
                f"{self.run_directory}/best_{model_name.lower()}_model.pkl",
            )

        # -------- overall winner ----------
        best_model_name = max(
            best_results, key=lambda m: best_results[m]["outer_mean_f1"]
        )
        self.optimization_results = {
            "best_model_name": best_model_name,
            "best_model": best_results[best_model_name]["best_fold_model"],
            "best_score": best_results[best_model_name]["outer_mean_f1"],
            "best_results": best_results,
            "all_results": all_results,
        }

        # --- NEW: export nice CSV & figure ----------------------------------
        summary_df = self._export_cv_summary(model_logs)
        self._plot_cv_summary(summary_df)

        return best_results

    # ---------------------------------------------------------------------
    #  factor the huge search‑space dict into its own helper
    # ---------------------------------------------------------------------
    def _define_search_spaces(self):
        """Return the gigantic search_spaces dict (unchanged from your code)."""
        from skopt.space import Real, Integer, Categorical

        search_spaces = {
            # # ───────────────────────────────────────────────────────── Random Forest ──
            "RandomForest": {
                "model": Categorical(
                    [RandomForestClassifier(random_state=42, n_jobs=-1)]
                ),
                "model__n_estimators": Integer(50, 150),  # fewer trees → less variance
                "model__max_depth": Integer(2, 3),  # very shallow 2, 4
                "model__min_samples_split": Integer(2, 6), # 2, 6
                "model__min_samples_leaf": Integer(2, 8),  # prevents tiny leaves
                "model__class_weight": Categorical(["balanced"]),
                "feature_selection__k": Integer(3, 20), #3, 20
            },
            # # ──────────────────────────────────────────────────────────────   SVM  ──
            # "SVM": {
            #     "model": Categorical([SVC(random_state=42, probability=True)]),
            #     "model__kernel": Categorical(["linear", "rbf", "poly"]),  # poly removed
            #     "model__C": Real(1e-2, 5.0, prior="log-uniform"),
            #     "model__gamma": Real(1e-4, 5e-2, prior="log-uniform"),
            #     "model__class_weight": Categorical(["balanced", None]),
            #     "feature_selection__k": Integer(3, 20),
            # },
            # # # ─────────────────────────────────────────── Elastic‑Net Logistic Reg ──
            # "ElasticNetLogReg": {
            #     "model": Categorical(
            #         [
            #             LogisticRegression(
            #                 penalty="elasticnet",
            #                 solver="saga",
            #                 class_weight="balanced",
            #                 max_iter=4000,
            #                 random_state=42,
            #             )
            #         ]
            #     ),
            #     "model__C": Real(1e-2, 5.0, prior="log-uniform"),
            #     "model__l1_ratio": Real(0.1, 0.9),  # avoid extremes
            #     "feature_selection__k": Integer(3, 20),
            # },
            # # ───────────────────────────────────────────────────────── Extra Trees ──
            # "ExtraTrees": {
            #     "model": Categorical(
            #         [
            #             ExtraTreesClassifier(
            #                 class_weight="balanced", random_state=42, n_jobs=-1
            #             )
            #         ]
            #     ),
            #     "model__n_estimators": Integer(80, 200),
            #     "model__max_depth": Integer(2, 8),
            #     "model__min_samples_leaf": Integer(2, 6),
            #     "feature_selection__k": Integer(3, 20),
            # },
            # # # ────────────────────────────────────────── HistGradientBoosting ──
            # "HGBClassifier": {
            #     "model": Categorical(
            #         [
            #             HistGradientBoostingClassifier(
            #                 class_weight="balanced", random_state=42
            #             )
            #         ]
            #     ),
            #     "model__learning_rate": Real(0.02, 0.12, prior="log-uniform"),
            #     "model__max_depth": Integer(2, 4),
            #     "model__max_iter": Integer(60, 100),
            #     "feature_selection__k": Integer(3, 20),
            # },
            # # # ───────────────────────────────────────────────────── k‑Nearest Nbrs ──
            # "kNN": {
            #     "model": Categorical([KNeighborsClassifier()]),
            #     "model__n_neighbors": Integer(5, 15),  # ≥5 to limit variance
            #     "model__weights": Categorical(["uniform", "distance"]),
            #     "feature_selection__k": Integer(3, 20),
            # },
            # # # ────────────────────────────────────────────── Gaussian Naïve Bayes ──
            # "GaussianNB": {
            #     "model": Categorical([GaussianNB()])
            #     # no hyper‑params
            # },
            # # # ────────────────────────────────────────────── Shrinkage LDA ──
            # "ShrinkageLDA": {
            #     "model": Categorical([LinearDiscriminantAnalysis(solver="lsqr")]),
            #     "model__shrinkage": Categorical(["auto", 0.1, 0.3, None]),
            # },
        }
        return search_spaces

    # ── put this in EEGAnalyzer ──────────────────────────────────────────
    def _visualize_optimization_results(self, all_results):
        """
        Save optimisation-trajectory plots:
            • one pair of PNGs per model
            • one stacked overview (sorted-score & chronological)
        """
        import os, numpy as np, matplotlib.pyplot as plt, math

        # 1. create sub-folder
        prog_dir = os.path.join(self.run_directory, "optimization_progress")
        os.makedirs(prog_dir, exist_ok=True)

        # 2. helper that returns two matplotlib figures for a model
        def _make_figures(model, scores):
            scores = np.asarray(scores)  # list → ndarray
            sorted_idx = np.argsort(scores)
            sorted_scores = scores[sorted_idx]

            # ------- fig A : sorted by score -------
            fig_sorted, ax = plt.subplots(figsize=(8, 4))
            ax.plot(
                np.arange(1, len(sorted_scores) + 1),
                sorted_scores,
                "o-",
                label="inner-CV F1 (sorted)",
            )
            ax.axhline(
                sorted_scores.max(),
                color="r",
                ls="-",
                label=f"Best = {sorted_scores.max():.4f}",
            )
            ax.set(
                title=f"{model} – optimisation (best-first)",
                xlabel="Iteration (ranked)",
                ylabel="F1 score",
            )
            ax.legend()
            ax.grid(True)

            # ------- fig B : chronological -------
            fig_time, ax2 = plt.subplots(figsize=(8, 4))
            ax2.plot(
                np.arange(1, len(scores) + 1),
                scores,
                "o-",
                label="inner-CV F1 (chronological)",
            )
            ax2.axhline(
                scores.max(), color="r", ls="-", label=f"Best = {scores.max():.4f}"
            )
            ax2.set(
                title=f"{model} – optimisation (chronological)",
                xlabel="Iteration",
                ylabel="F1 score",
            )
            ax2.legend()
            ax2.grid(True)

            return fig_sorted, fig_time

        # 3. per-model PNGs
        for mdl, res in all_results.items():
            fig_s, fig_t = _make_figures(mdl, res["scores"])
            fig_s.savefig(os.path.join(prog_dir, f"{mdl}_progress_sorted.png"), dpi=300)
            fig_t.savefig(os.path.join(prog_dir, f"{mdl}_progress_time.png"), dpi=300)
            plt.close(fig_s)
            plt.close(fig_t)

        # 4. combined overview (same as original idea, but cleaner)
        n_models = len(all_results)
        fig1, axes1 = plt.subplots(
            n_models, 1, figsize=(12, 4 * n_models), sharex=False
        )
        fig2, axes2 = plt.subplots(
            n_models, 1, figsize=(12, 4 * n_models), sharex=False
        )

        if n_models == 1:  # keep iterable even for one model
            axes1, axes2 = [axes1], [axes2]

        for ax_s, ax_t, (mdl, res) in zip(axes1, axes2, all_results.items()):
            s = np.asarray(res["scores"])
            # --- sorted
            idx = np.argsort(s)
            ax_s.plot(np.arange(1, len(s) + 1), s[idx], "o-")
            ax_s.set(title=f"{mdl} (sorted)", ylabel="F1")
            ax_s.grid(True)
            # --- chronological
            ax_t.plot(np.arange(1, len(s) + 1), s, "o-")
            ax_t.set(title=f"{mdl} (chronological)", xlabel="Iteration", ylabel="F1")
            ax_t.grid(True)

        fig1.tight_layout()
        fig2.tight_layout()
        fig1.savefig(os.path.join(prog_dir, "overview_progress_sorted.png"), dpi=300)
        fig2.savefig(os.path.join(prog_dir, "overview_progress_time.png"), dpi=300)
        plt.close(fig1)
        plt.close(fig2)

        # 5. still call feature-importance visualisation
        self._visualize_best_model_feature_importance()

    def _visualize_best_model_feature_importance(self):
        """
        Visualize feature importance of the best optimized model.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd

        if not hasattr(self, "optimization_results"):
            print_log("Run optimize_hyperparameters first")
            return

        best_model = self.optimization_results["best_model"]
        best_model_name = self.optimization_results["best_model_name"]

        # Get the feature names that were selected (from the pipeline)
        feature_selector = best_model.named_steps["feature_selection"]
        selected_indices = feature_selector.get_support(indices=True)
        selected_features = [self.feature_columns[i] for i in selected_indices]

        # Get the actual model from the pipeline
        model = best_model.named_steps["model"]

        plt.figure(figsize=(12, 8))

        # Extract feature importance based on model type
        if best_model_name == "RandomForest":
            importances = model.feature_importances_
            std = np.std(
                [tree.feature_importances_ for tree in model.estimators_], axis=0
            )

            # Create DataFrame for plotting
            feature_importance = pd.DataFrame(
                {"feature": selected_features, "importance": importances, "std": std}
            ).sort_values("importance", ascending=False)

            plt.barh(
                feature_importance["feature"],
                feature_importance["importance"],
                xerr=feature_importance["std"],
            )
            plt.title(f"Feature Importance - Optimized {best_model_name}")

        elif best_model_name == "SVM" and model.kernel in ["linear"]:
            # For linear SVM, we can extract coefficients
            importances = np.abs(model.coef_[0])

            # Create DataFrame for plotting
            feature_importance = pd.DataFrame(
                {"feature": selected_features, "importance": importances}
            ).sort_values("importance", ascending=False)

            plt.barh(feature_importance["feature"], feature_importance["importance"])
            plt.title(
                f"Feature Importance (Coefficient Magnitude) - Optimized {best_model_name}"
            )

        else:
            # For other models, use permutation importance
            from sklearn.inspection import permutation_importance

            # Get data for the best label combination
            df_best = self.df[self.df["label"].isin(self.best_labels)]
            X_best = df_best[self.feature_columns]
            y_best = df_best["label"]

            # Transform data through the pipeline up to the model
            X_transformed = best_model.named_steps["scaler"].transform(X_best)
            X_selected = best_model.named_steps["feature_selection"].transform(
                X_transformed
            )

            # Calculate permutation importance
            result = permutation_importance(
                model, X_selected, y_best, n_repeats=10, random_state=42
            )

            # Create DataFrame for plotting
            feature_importance = pd.DataFrame(
                {
                    "feature": selected_features,
                    "importance": result.importances_mean,
                    "std": result.importances_std,
                }
            ).sort_values("importance", ascending=False)

            plt.barh(
                feature_importance["feature"],
                feature_importance["importance"],
                xerr=feature_importance["std"],
            )
            plt.title(f"Feature Importance (Permutation) - Optimized {best_model_name}")

        plt.xlabel("Importance")
        plt.ylabel("Features")
        plt.tight_layout()
        plt.savefig(f"{self.run_directory}/best_model_feature_importance.png", dpi=300)
        plt.close()

        # Save feature importance to CSV
        feature_importance.to_csv(
            f"{self.run_directory}/best_model_feature_importance.csv", index=False
        )

    # ------------------------------------------------------------
    #   ❶  Violin / strip plot of LOSO (or k-fold) outer scores
    # ------------------------------------------------------------
    def plot_outer_fold_distribution(self):
        """
        Violin/strip plot of outer‑fold macro‑F1 with bootstrap CI and
        (optionally) gold‑/OG‑null thresholds.

        Works for both:
            – self.optimization_results   (Bayes search route)
            – self.fixed_cv_results       (heuristic‑HP route)
        """
        import os, numpy as np, matplotlib.pyplot as plt, seaborn as sns

        # -----------------------------------------------------------------
        # 1) detect which result container is present
        # -----------------------------------------------------------------
        if hasattr(self, "optimization_results"):
            container = self.optimization_results
            best_name = container["best_model_name"]
            fold_scores = np.asarray(container["best_results"][best_name]["per_fold_f1"])
        elif hasattr(self, "fixed_cv_results"):
            best_name = self.fixed_best_model_name  # stored by run_nested_cv_fixed
            fold_scores = np.asarray(self.fixed_cv_results[best_name]["per_fold_f1"])
        else:
            print_log("No outer‑CV results found – run nested CV first.")
            return

        # -----------------------------------------------------------------
        # 2) basic statistics
        # -----------------------------------------------------------------
        mean_s = fold_scores.mean()
        sd_s = fold_scores.std(ddof=0)

        rng = np.random.default_rng(42)
        boot_means = rng.choice(fold_scores,
                                size=(5000, fold_scores.size),
                                replace=True).mean(axis=1)
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

        # -----------------------------------------------------------------
        # 3) plot
        # -----------------------------------------------------------------
        plt.figure(figsize=(5, 6))
        ax = plt.gca()
        sns.violinplot(data=fold_scores, inner=None, color="skyblue", ax=ax)
        sns.stripplot(data=fold_scores, color="black", size=6, jitter=False, ax=ax)

        ax.axhline(mean_s, ls="--", lw=2, color="red",
                   label=f"μ={mean_s:.2f}±{sd_s:.2f}")
        ax.axhspan(ci_low, ci_high, color="red", alpha=0.15,
                   label=f"95 % boot CI [{ci_low:.2f}, {ci_high:.2f}]")

        # optional null thresholds (OG or gold)
        for tag, style in [("null_95th_percentile.npy", (":", "α=0.05")),
                           ("gold_null95.npy", (":", "α=0.05 (gold)")),
                           ("null_50th_percentile.npy", ("--", "Null mean")),
                           ("gold_null50.npy", ("--", "Null mean (gold)"))]:
            f = os.path.join(self.run_directory, tag)
            if os.path.exists(f):
                val = float(np.load(f))
                ax.axhline(val, ls=style[0], lw=2,
                           color="black" if ':' in style[0] else "grey",
                           label=f"{style[1]} ({val:.2f})")

        ax.set_ylabel("Macro‑F1 (outer fold)")
        ax.set_title(f"{best_name} – distribution across {fold_scores.size} outer folds")
        ax.set_ylim(0, 1)
        ax.legend()
        out_png = f"{self.run_directory}/outer_fold_f1_distribution.png"
        plt.tight_layout();
        plt.savefig(out_png, dpi=300, bbox_inches="tight");
        plt.close()
        print_log(f"Saved violin plot → {out_png}")



    # NESTED CV without Bayes
    def run_nested_cv_fixed(self, models=None):
        """
        Nested CV **without** hyper‑parameter search.
        Uses exactly the same outer/inner splitters that
        `_get_nested_splitters` would give, but the inner loop is
        only there to keep the feature‑selection / scaling strictly
        train‑only – no optimisation is performed.

        After running, the per‑model outer‑fold predictions are stored in
        `self.fixed_cv_results` and the winning pipeline (re‑fitted on *all*
        data) in `self.fixed_best_model`.
        """
        from sklearn.base import clone
        from collections import defaultdict
        if self.best_labels is None:
            raise RuntimeError("call evaluate_label_combinations() first")

        # ❶  Data restricted to the best label combo
        df_best = self.df[self.df["label"].isin(self.best_labels)].reset_index(drop=True)
        X_all = df_best[self.feature_columns]
        y_all = df_best["label"]
        groups_all = df_best["session"].values

        # ❷  Default model dict (same heuristics you already use)
        if models is None:
            # ── Heuristic‑HP models (no tuning) ─────────────────────────────
            self.heuristic_models = {
                # 1) Random Forest – very small, shallow, balanced
                "RandomForest": RandomForestClassifier(
                    n_estimators=80,  # enough trees for stability, not overkill
                    max_depth=3,  # shallow → less over‑fitting on 118 samples
                    min_samples_split=2,
                    min_samples_leaf=2,  # avoid single‑sample leaves
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),

                # 2) SVM – RBF kernel with mild regularisation
                "SVM": SVC(
                    kernel="rbf",
                    C=0.8,  # soft margin
                    gamma="scale",  # 1 / (n_features × Var) – works fine here
                    probability=True,
                    class_weight="balanced",
                    random_state=42,
                ),

                # 3) Elastic‑Net Logistic Regression
                "ElasticNetLogReg": LogisticRegression(
                    penalty="elasticnet",
                    C=1.0,  # ≈ default strength
                    l1_ratio=0.5,  # 50‑50 L1/L2 mix
                    solver="saga",
                    max_iter=4000,
                    class_weight="balanced",
                    random_state=42,
                ),

                # 4) Extra‑Trees – slightly deeper than RF but still conservative
                "ExtraTrees": ExtraTreesClassifier(
                    n_estimators=120,
                    max_depth=4,
                    min_samples_leaf=2,
                    class_weight="balanced",
                    random_state=42,
                    n_jobs=-1,
                ),

                # 5) HistGradientBoosting – fast, regularised
                "HGBClassifier": HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_depth=3,
                    max_iter=80,
                    class_weight="balanced",
                    random_state=42,
                ),

                # 6) k‑Nearest Neighbours – distance weighting, odd k
                "kNN": KNeighborsClassifier(
                    n_neighbors=7,
                    weights="distance",
                ),

                # 7) Gaussian Naïve Bayes – no params to tweak
                "GaussianNB": GaussianNB(),

                # 8) Shrinkage LDA – automatic shrinkage
                "ShrinkageLDA": LinearDiscriminantAnalysis(
                    solver="lsqr",
                    shrinkage="auto",
                ),
            }

        outer_cv, _ = self._get_nested_splitters(groups_all)
        results = {}

        print_log("\n---- FIXED‑PARAM NESTED CV ----")
        for name, base_model in self.heuristic_models.items():
            fold_true, fold_pred, fold_f1, fold_acc = [],[], [], []
            print_log(f"\n{name}")
            for fold, (tr_idx, te_idx) in enumerate(
                    outer_cv.split(X_all, y_all, groups_all), start=1):
                X_tr, X_te = X_all.iloc[tr_idx], X_all.iloc[te_idx]
                y_tr, y_te = y_all.iloc[tr_idx], y_all.iloc[te_idx]

                # ── scale & select inside the outer fold ──────────────────
                scaler = StandardScaler().fit(X_tr)
                X_tr_s = scaler.transform(X_tr)
                X_te_s = scaler.transform(X_te)

                selector = SelectKBest(f_classif, k=self.n_features_to_select).fit(
                    X_tr_s, y_tr)
                X_tr_sel = selector.transform(X_tr_s)
                X_te_sel = selector.transform(X_te_s)

                clf = clone(base_model).fit(X_tr_sel, y_tr)
                y_hat = clf.predict(X_te_sel)

                acc_val = accuracy_score(y_te, y_hat)
                fold_acc.append(acc_val)  # store

                fold_true.append(y_te.to_numpy())
                fold_pred.append(y_hat)
                f1_val = f1_score(y_te, y_hat, average="macro")
                fold_f1.append(f1_val)
                print_log(f"  outer‑fold {fold}: F1={f1_val:.3f}")

            mu = np.mean(fold_f1)
            sig = np.std(fold_f1, ddof=0)
            print_log(f"→ mean outer F1 = {mu:.3f} ± {sig:.3f}")

            results[name] = dict(
                per_fold_true=fold_true,
                per_fold_pred=fold_pred,
                per_fold_f1=fold_f1,
                per_fold_acc=fold_acc,  # new
                mean_f1=mu,
                mean_acc=np.mean(fold_acc)
            )

        # store & pick the winner
        self.fixed_cv_results = results
        self.fixed_best_model_name = max(results, key=lambda k: results[k]["mean_f1"])

        # ❸  re‑fit winner on the FULL data (still scaling + selection)
        best_base = self.heuristic_models[self.fixed_best_model_name]
        scaler = StandardScaler().fit(X_all)
        X_all_s = scaler.transform(X_all)
        selector = SelectKBest(f_classif, k=self.n_features_to_select).fit(X_all_s, y_all)
        X_all_sel = selector.transform(X_all_s)

        self.fixed_best_model = clone(best_base).fit(X_all_sel, y_all)

        print_log(f"\nWinner (fixed‑param): {self.fixed_best_model_name} "
                  f"with mean F1 = {results[self.fixed_best_model_name]['mean_f1']:.3f}")



    # NESTED CV without Bayes
    def run_true_nested_cv(self):
        """
        Two‑level nested CV (no hyper‑parameter search).

        • Outer CV  = self.cv_method   (LOO, LOSO, LMOSO, k‑fold, …)
        • Inner CV  = chooses the best model family for *this* outer split,
                      but we also store outer‑fold predictions for every model
                      so downstream code (permutation tests, etc.) works unchanged.
        """
        from sklearn.base import clone
        import numpy as np
        import pandas as pd, os
        from sklearn.model_selection import (
            LeaveOneOut, StratifiedKFold,
            StratifiedGroupKFold, GroupShuffleSplit, LeavePGroupsOut,
        )

        # ─────────────────────────────────────────────────────────────────────
        # ❶  Sanity checks & data subsets
        # ─────────────────────────────────────────────────────────────────────
        if self.best_labels is None:
            print_log("sticking to the base labels m_et_s and n_3_s since best labels were not computed")
            self.best_labels=('m_et_s', 'n_3_s')
        df_best = self.df[self.df["label"].isin(self.best_labels)].reset_index(drop=True)
        X_all = df_best[self.feature_columns]
        y_all = df_best["label"]
        groups = df_best["session"].values

        # ─────────────────────────────────────────────────────────────────────
        # ❷  OUTER / INNER splitters
        # ─────────────────────────────────────────────────────────────────────
        if self.cv_method == "loo":
            outer_cv = LeaveOneOut()
            outer_iter = outer_cv.split(X_all, y_all)  # no groups
            inner_proto = StratifiedKFold(
                n_splits=min(3, len(X_all) - 1), shuffle=True, random_state=24)
        else:

            outer_cv, inner_proto = self._get_nested_splitters(groups)

            # does the OUTER splitter need a `groups` array?
            needs_groups_outer = isinstance(
                outer_cv,
                (StratifiedGroupKFold, GroupShuffleSplit, LeavePGroupsOut, LeaveOneGroupOut)
            )

            if needs_groups_outer:
                outer_iter = outer_cv.split(X_all, y_all, groups)
            else:
                outer_iter = outer_cv.split(X_all, y_all)

        # ─────────────────────────────────────────────────────────────────────
        # ❸  Prepare containers for *every* model family
        # ─────────────────────────────────────────────────────────────────────
        if not hasattr(self, "heuristic_models"):
            self.heuristic_models = self._default_heuristic_models()  # helper

        results = {
            m: dict(per_fold_true=[], per_fold_pred=[], per_fold_f1=[], per_fold_acc=[], per_fold_n=[], per_fold_inner_acc=[], per_fold_inner_f1=[])
            for m in self.heuristic_models
        }

        fold_logs = []  # keeps the “winning model per outer fold” story

        # -----------------------------------------------------------------
        k_grid = getattr(self, "k_grid", self.frozen_features)  # candidate #features
        from collections import defaultdict
        feature_score_accumulator, leaky_feature_score_accumulator = defaultdict(float),defaultdict(float)

        # -----------------------------------------------------------------
        # ─────────────────────────────────────────────────────────────────────
        # ❹  OUTER loop
        # ─────────────────────────────────────────────────────────────────────
        needs_groups_inner = isinstance(
            inner_proto, (StratifiedGroupKFold, GroupShuffleSplit, LeavePGroupsOut)
        )
        for f, (tr_idx, te_idx) in enumerate(outer_iter, start=1):

            X_tr, X_te = X_all.iloc[tr_idx], X_all.iloc[te_idx]
            y_tr, y_te = y_all.iloc[tr_idx], y_all.iloc[te_idx]

            # build inner splits once per outer fold
            inner_splits = (list(inner_proto.split(X_tr, y_tr, groups[tr_idx]))
                            if needs_groups_inner else
                            list(inner_proto.split(X_tr, y_tr)))

            # -----------------------------------------------------------------
            # 4.1  INNER loop – grid-search (model × k) on training only
            # -----------------------------------------------------------------
            inner_mean_f1, inner_mean_acc = {}, {}
            for mdl_name, base_mdl in self.heuristic_models.items():
                for k_feats in k_grid:
                    f1_inner, acc_inner = [], []

                    for in_tr, in_val in inner_splits:
                        scaler = StandardScaler().fit(X_tr.iloc[in_tr])

                        sel = SelectKBest(f_classif, k=k_feats).fit(
                            scaler.transform(X_tr.iloc[in_tr]),
                            y_tr.iloc[in_tr])

                        Xtr_sel = sel.transform(scaler.transform(X_tr.iloc[in_tr]))
                        Xval_sel = sel.transform(scaler.transform(X_tr.iloc[in_val]))

                        y_hat = clone(base_mdl).fit(Xtr_sel, y_tr.iloc[in_tr]).predict(Xval_sel)

                        f1_inner.append(f1_score(y_tr.iloc[in_val], y_hat, average="macro"))
                        acc_inner.append(accuracy_score(y_tr.iloc[in_val], y_hat))

                    key = (mdl_name, k_feats)  # ← unique combo
                    inner_mean_f1[key] = np.mean(f1_inner)
                    inner_mean_acc[key] = np.mean(acc_inner)

                    # ------------------------------------------------------------------
                    # NEW: keep a global log of inner-CV scores for each (model, k)
                    # ------------------------------------------------------------------
                    if not hasattr(self, "inner_grid_log"):
                        from collections import defaultdict
                        self.inner_grid_log = defaultdict(list)

                    self.inner_grid_log["model"].append(mdl_name)
                    self.inner_grid_log["k"].append(k_feats)
                    self.inner_grid_log["fold"].append(f)  # outer-fold index
                    self.inner_grid_log["inner_mean_f1"].append(inner_mean_f1[(mdl_name, k_feats)])
                    self.inner_grid_log["inner_mean_acc"].append(inner_mean_acc[(mdl_name, k_feats)])

            # choose the (model, k) that maximises inner-mean F1
            (best_name, best_k) = max(inner_mean_f1, key=inner_mean_f1.get)
            best_model = self.heuristic_models[best_name]

            # -----------------------------------------------------------------
            # 4.2  Re-train *each* model with its own best-k on outer-train,
            #      predict outer-test so downstream code (permutation tests)
            #      can still compare all models.
            # -----------------------------------------------------------------
            scaler_full = StandardScaler().fit(X_tr)

            for mdl_name, base_mdl in self.heuristic_models.items():
                # If this model wasn’t the winner we still need *some* k:
                k_star = max((k for (n, k) in inner_mean_f1 if n == mdl_name),
                             key=lambda k: inner_mean_f1[(mdl_name, k)])

                sel_full = SelectKBest(f_classif, k=k_star).fit(
                    scaler_full.transform(X_tr), y_tr)

                Xtr_sel = sel_full.transform(scaler_full.transform(X_tr))
                Xte_sel = sel_full.transform(scaler_full.transform(X_te))

                y_pred = clone(base_mdl).fit(Xtr_sel, y_tr).predict(Xte_sel)

                # bookkeeping (unchanged) …
                results[mdl_name]["per_fold_true"].append(y_te.to_numpy())
                results[mdl_name]["per_fold_pred"].append(y_pred)
                results[mdl_name]["per_fold_f1"].append(
                    f1_score(y_te, y_pred, average="macro"))
                results[mdl_name]["per_fold_acc"].append(
                    accuracy_score(y_te, y_pred))
                results[mdl_name]["per_fold_n"].append(len(te_idx))
                results[mdl_name]["per_fold_inner_acc"].append(
                    inner_mean_acc[(mdl_name, k_star)])
                results[mdl_name]["per_fold_inner_f1"].append(
                    inner_mean_f1[(mdl_name, k_star)])

            # --------------------------------------------------------------
            # ①  Feature-importance bookkeeping uses the *outer-winner*
            #    (best_name, best_k) determined above
            # --------------------------------------------------------------
            sel_full = SelectKBest(f_classif, k=best_k).fit(
                scaler_full.transform(X_tr), y_tr)
            X_tr_sel = sel_full.transform(scaler_full.transform(X_tr))
            selected_feats = [self.feature_columns[i]
                              for i in sel_full.get_support(indices=True)]
            # --------------------------------------------------------------
            #   ②  Extract importance scores for those features
            # --------------------------------------------------------------
            def _get_importances(model, X, y):
                """Return an array of importance scores *aligned with selected_feats*"""
                # tree-based models
                if hasattr(model, "feature_importances_"):
                    return model.feature_importances_
                # linear models with coef_
                if hasattr(model, "coef_"):
                    return np.abs(model.coef_).ravel()
                # fallback: permutation importance (quick, 5 repeats)
                from sklearn.inspection import permutation_importance
                pi = permutation_importance(model, X, y, n_repeats=5,
                                            random_state=42, n_jobs=-1)
                return pi.importances_mean

            X_tr_fold_sel = X_tr_sel  # already created above
            y_tr_fold = y_tr  # same alignment
            importances = _get_importances(
                clone(best_model).fit(X_tr_fold_sel, y_tr_fold),
                X_tr_fold_sel, y_tr_fold)

            # -----------------------------------------------------------------
            # 4.3  bookkeeping
            # -----------------------------------------------------------------
            outer_f1 = results[best_name]["per_fold_f1"][-1]
            print_log(f"Fold {f:02d}: best={best_name:<15}  "
          f"inner-F1={inner_mean_f1[(best_name, best_k)]:.3f}  "
                      f"outer‑F1={outer_f1:.3f}")
            print_log(f"Fold {f:02d} label counts: {dict(pd.Series(y_te).value_counts())}")
            print_log(
                f"k={best_k:<2} ")
            fold_logs.append(dict(
                fold=f,
                best_model=best_name,
                best_k=best_k,
                inner_f1=inner_mean_f1[(best_name, best_k)],
                outer_f1=outer_f1,
                n_test=len(te_idx),
            ))


            # --------------------------------------------------------------
            #   ③  Weight by this fold’s outer-F1 (or outer-ACC) -this, if used to select number of k's is a hyperparameter leakage
            # --------------------------------------------------------------
            # leaky_weighted_importances = importances * outer_f1  # or outer_acc
            # --------------------------------------------------------------
            #   ③  Weight by this fold’s iner-F1 (or outer-ACC) -insight leakage free
            # --------------------------------------------------------------
            inner_f1 = inner_mean_f1[(best_name, best_k)]
            weighted_importances = importances * inner_f1
            leaky_weighted_importances= importances * outer_f1
            # accumulate
            for feat, w_imp in zip(selected_feats, weighted_importances):
                feature_score_accumulator[feat] += w_imp
            # accumulate leaky importances
            for feat, leaky_w_imp in zip(selected_feats, leaky_weighted_importances):
                leaky_feature_score_accumulator[feat] += leaky_w_imp


            # keep a pretty DataFrame for optional inspection
            # TODO REVERT, not really necessary but breaks 3-class classification runs
            # per_fold_feature_tables.append(
            #     pd.DataFrame({"feature": selected_feats,
            #                   "importance": importances,
            #                   "weighted": weighted_importances,
            #                   "fold": f})
            # )


        # ─────────────────────────────────────────────────────────────────────
        # ❺  Aggregate metrics per model & choose global winner
        # ─────────────────────────────────────────────────────────────────────
        for mdl, rec in results.items():
            # below session weighted
            # rec["mean_f1"] = float(np.mean(rec["per_fold_f1"]))
            # rec["mean_acc"] = float(np.mean(rec["per_fold_acc"]))
            w = np.asarray(rec["per_fold_n"], dtype=float)
            rec["mean_f1"] = float(np.average(rec["per_fold_f1"], weights=w))
            rec["mean_acc"] = float(np.average(rec["per_fold_acc"], weights=w))
            rec["inner_mean_f1"] = float(np.average(rec["per_fold_inner_f1"], weights=w))
        # store & expose the familiar objects
        self.fixed_cv_results = results
        # this is a bad bad bad issue, that's why we no longer run the test
        # with multiple classifiers, this picks the best outer, which is a post-hoc
        # evaluation, this is INVALID
        # TODO we don't need to change this but we should not use anything from here to report
        # which means it is still fine to do it per model, but not aggregated
        # self.fixed_best_model_name = max(results, key=lambda m: results[m]["mean_f1"])
        # now fixed, we choose the inner f1 winner but it is not weighted correctly, so still do not use it for multiple classifiers
        self.fixed_best_model_name = max(results, key=lambda m: results[m]["inner_mean_f1"])



        # ---------------------------------------------------------------------------
        # Pretty plots of the CV results
        # ---------------------------------------------------------------------------
        import os, pandas as pd, matplotlib.pyplot as plt, seaborn as sns

        # ①  Tidy DataFrame with one row per outer fold
        rows = []
        for mdl, rec in results.items():
            for fold_i, (acc, f1) in enumerate(zip(rec["per_fold_acc"],
                                                   rec["per_fold_f1"]), start=1):
                rows.append(dict(model=mdl,
                                 fold=fold_i,
                                 outer_acc=acc,
                                 outer_f1=f1))
        df_folds = pd.DataFrame(rows)

        # ---------------------------------------------------------------------------
        # ②  Box / strip plot of outer-fold accuracies  – coloured by significance
        # ---------------------------------------------------------------------------
        if True:
            fig, ax = plt.subplots(figsize=(10, 6))

            # ------------------------------------------------------------------
            # A) decide the thresholds for chance and p=0.05
            # ------------------------------------------------------------------
            if self.top_n_labels == 2:  # binary, n = 118
                p10_thresh=0.568    # p10-p05 is treated as 'trend' though not considered 'significant'
                p05_thresh = 0.585
            elif self.top_n_labels == 3:  # three-class, n = 161
                p10_thresh=0.385
                p05_thresh = 0.404
            else:
                raise ValueError("top_n_labels must be 2 or 3")

            # ------------------------------------------------------------------
            # B) build a palette dict {model_name: colour}
            # ------------------------------------------------------------------
            palette = {}
            for m, rec in results.items():
                acc = rec["mean_acc"]
                if acc >= p05_thresh:
                    palette[m] = "#4daf4a"  # green  → significant
                elif p10_thresh <= acc < p05_thresh:
                    palette[m] = "#ffda66"  # yellow → “trend” p =[0.1,0.05]
                else:
                    palette[m] = "#bdbdbd"  # grey   → considered 'null'

            # ------------------------------------------------------------------
            # C) draw the box plot with that palette
            # ------------------------------------------------------------------
            sns.boxplot(data=df_folds, x="model", y="outer_acc",
                        palette=palette, ax=ax, showmeans=True,            meanprops=dict(marker="X",            # bigger, bold “X”
                           markeredgecolor="black",
                           markerfacecolor="white",
                           markersize=8),)
            sns.stripplot(data=df_folds, x="model", y="outer_acc",
                          color="black", alpha=0.6, jitter=0.25, size=4, ax=ax)

            ax.set_ylabel("Outer-fold accuracy")
            ax.set_xlabel("")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
            ax.set_title(f"Outer-fold accuracies (not sample weighted) per model with {self.cv_method} cross validation")
            ax.axhline(y=0.5, linestyle="--", color="black")
            import matplotlib.patches as mpatches
            green_patch = mpatches.Patch(color='green', label='Significant Accuracy (p<0.05)')
            yellow_patch = mpatches.Patch(color='yellow', label='Trend (0.05<p<0.1)')
            grey_patch = mpatches.Patch(color='grey', label='Not Significant or Trend (p>0.1)')

            # 2) Pass them into legend via the handles kwarg:
            ax.legend(handles=[green_patch, yellow_patch, grey_patch],
                      loc='lower right',
                      frameon=False)  # or any other style tweaks

            fig.tight_layout()

            out_png = os.path.join(self.run_directory, "outer_fold_accuracies.png")
            fig.savefig(out_png, dpi=300)
            plt.close()
            print_log(f"✓ Saved accuracy box plot → {out_png}")



        # ③  Scatter of mean accuracy vs mean F1
        mean_rows = [{"model": m,
                      "mean_acc": rec["mean_acc"],
                      "mean_f1": rec["mean_f1"]}
                     for m, rec in results.items()]
        df_mean = pd.DataFrame(mean_rows)

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.scatterplot(ax=ax, data=df_mean, x="mean_f1", y="mean_acc", s=80)
        for _, r in df_mean.iterrows():
            ax.text(r["mean_f1"] + 0.002, r["mean_acc"], r["model"],
                    fontsize=8, va="center")

        # horizontal reference lines (chance, p=0.05, p=0.005)
        self.add_accuracy_reference_lines(self.top_n_labels, ax=ax)

        plt.ylabel("Sample-weighted mean accuracy")
        plt.xlabel("Sample-weighted mean F1")
        plt.title(f"Model-level performance (outer CV, {self.cv_method})")
        plt.legend(loc='lower right')
        plt.tight_layout()
        scatter_png = os.path.join(self.run_directory, "model_mean_acc_vs_f1.png")
        plt.savefig(scatter_png, dpi=300)
        plt.close()
        print_log(f"✓ Saved accuracy-vs-F1 scatter → {scatter_png}")

        # ---------------------------------------------------------------------------
        # ④  Bar plot: inner-CV mean accuracy  vs  outer-fold mean accuracy
        # ---------------------------------------------------------------------------
        if True:
            # We need, for each model:
            #   • inner_mean_acc  – average of the accuracies the model got on its
            #                       inner-validation splits (across *all* outer folds)
            #   • outer_mean_acc  – already in results[model]["mean_acc"]
            #   • inner_sd, outer_sd  – standard deviations for error bars
            #

            inner_stats = pd.DataFrame({
                "inner_mean_acc": {m: np.mean(rec["per_fold_inner_acc"])
                                   for m, rec in results.items()},
                "inner_mean_f1": {m: np.mean(rec["per_fold_inner_f1"])
                                   for m, rec in results.items()},
                "inner_sd": {m: np.std(rec["per_fold_inner_acc"], ddof=1)
                             for m, rec in results.items()}
            })

            # ------------------------------------------------------------------
            # B) gather outer stats (already in results)
            # ------------------------------------------------------------------
            outer_stats = pd.DataFrame({
                "outer_mean_acc": {m: rec["mean_acc"] for m, rec in results.items()},
                "outer_sd": {m: np.std(rec["per_fold_acc"], ddof=1)
                             for m, rec in results.items()}
            })

            # C)  → sort models by *descending* outer mean accuracy  〈〈 NEW 〉〉
            ordered_models = (outer_stats["outer_mean_acc"]
                              .sort_values(ascending=False)
                              .index)

            # D) merge, then re-index to that order ----------------------------
            stats_df = (inner_stats.join(outer_stats, how="outer")
                        .loc[ordered_models]  # keep the order
                        .reset_index()
                        .rename(columns={"index": "model"}))

            # ------------------------------------------------------------------
            # E) make the bar plot
            # ------------------------------------------------------------------
            fig, ax = plt.subplots(figsize=(10, 6))

            x = np.arange(len(stats_df))  # model positions
            width = 0.35

            ax.bar(x - width / 2, stats_df["inner_mean_acc"],
                   width, yerr=stats_df["inner_sd"],
                   label="Inner (mean CV acc)", color="#3182bd", alpha=0.9,
                   capsize=4)
            ax.bar(x + width / 2, stats_df["outer_mean_acc"],
                   width, yerr=stats_df["outer_sd"],
                   label="Outer (held-out acc)", color="#fdb863", alpha=0.9,
                   capsize=4)

            # Add the text value on top of each bar
            for i, row in stats_df.iterrows():
                ax.text(i - width / 2, row["inner_mean_acc"] + 0.01,
                        f"{row['inner_mean_acc']:.3f}", ha="center", va="bottom", fontsize=8)
                ax.text(i + width / 2, row["outer_mean_acc"] + 0.01,
                        f"{row['outer_mean_acc']:.3f}", ha="center", va="bottom", fontsize=8)

            ax.set_xticks(x)
            ax.set_xticklabels(stats_df["model"], rotation=30, ha="right")
            ax.set_ylabel("Accuracy")
            ax.set_ylim(0, 1)
            ax.set_title("Nested-CV mean accuracy: inner selection vs outer test")
            ax.legend(frameon=False)
            fig.tight_layout()

            inner_outer_png = os.path.join(self.run_directory,
                                           "inner_vs_outer_mean_accuracy.png")
            fig.savefig(inner_outer_png, dpi=300)
            plt.close()
            print_log(f"✓ Saved inner-vs-outer accuracy plot → {inner_outer_png}")

        # ─────────────────────────────────────────────────────────────────────
        # ❻  Re‑fit the winner on ALL data and save in self.fixed_best_model
        # ─────────────────────────────────────────────────────────────────────
        best_base = self.heuristic_models[self.fixed_best_model_name]
        scaler = StandardScaler().fit(X_all)
        sel = SelectKBest(f_classif, k=self.n_features_to_select).fit(
            scaler.transform(X_all), y_all)
        X_all_sel = sel.transform(scaler.transform(X_all))
        self.fixed_best_model = clone(best_base).fit(X_all_sel, y_all)

        # --------------------------------------------------------------
        # ❺  Aggregate weighted scores across folds weighted by inner fold accuracies (no heuristic leakage)
        # --------------------------------------------------------------
        agg_df = (
            pd.DataFrame.from_dict(feature_score_accumulator, orient="index",
                                   columns=["cum_weighted_score"])
            .sort_values("cum_weighted_score", ascending=False)
            .reset_index()
            .rename(columns={"index": "feature"})
        )

        out_csv = os.path.join(self.run_directory, "crossval_feature_scores.csv")
        agg_df.to_csv(out_csv, index=False)
        print_log(f"✓ Saved cross-validated feature scores → {out_csv}")

        # optional bar-plot of top-N
        TOP_N = 25
        plt.figure(figsize=(8, 0.4 * TOP_N + 1))
        sns.barplot(data=agg_df.head(TOP_N),
                    y="feature", x="cum_weighted_score", orient="h")
        plt.xlabel("∑ (importance × inner-fold F1)")
        plt.title(f"Top {TOP_N} features across all outer folds weighted with INNER fold accuracy (no heuristic leakage)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_directory,
                                 f"feature_scores_top{TOP_N}.png"), dpi=300)
        plt.close()

        # --------------------------------------------------------------
        # ❺  Aggregate weighted scores across folds weighted by OUTER fold accuracies (heuristic leakage)
        # --------------------------------------------------------------
        agg_df = (
            pd.DataFrame.from_dict(leaky_feature_score_accumulator, orient="index",
                                   columns=["cum_weighted_score"])
            .sort_values("cum_weighted_score", ascending=False)
            .reset_index()
            .rename(columns={"index": "feature"})
        )

        out_csv = os.path.join(self.run_directory, "leaky_crossval_feature_scores.csv")
        agg_df.to_csv(out_csv, index=False)
        print_log(f"✓ Saved leaky cross-validated feature scores (only for post-hoc insights) → {out_csv}")

        # optional bar-plot of top-N
        TOP_N = 25
        plt.figure(figsize=(8, 0.4 * TOP_N + 1))
        sns.barplot(data=agg_df.head(TOP_N),
                    y="feature", x="cum_weighted_score", orient="h")
        plt.xlabel("∑ (importance × OUTER-fold F1)")
        plt.title(f"Top {TOP_N} features across all outer folds weighted with OUTER fold accuracy (post-hoc only)")
        plt.tight_layout()
        plt.savefig(os.path.join(self.run_directory,
                                 f"leaky_feature_scores_top{TOP_N}.png"), dpi=300)
        plt.close()








        print_log(f"\nWinner (fixed‑param): {self.fixed_best_model_name} "
                  f"with mean F1 = {results[self.fixed_best_model_name]['mean_f1']:.3f}")

        # ------------------------------------------------------------------
        # Save the per-(model, k, fold) grid search results
        # ------------------------------------------------------------------
        if hasattr(self, "inner_grid_log"):
            import pandas as pd, os
            grid_df = pd.DataFrame(self.inner_grid_log)
            grid_csv = os.path.join(self.run_directory, "inner_cv_gridsearch_log.csv")
            grid_df.to_csv(grid_csv, index=False)
            print_log(f"✓ Saved per-model-k inner-CV scores → {grid_csv}")

        # ─────────────────────────────────────────────────────────────────────
        # ❼  Save a log CSV (outer‑fold winners) – optional, nice to have
        # ─────────────────────────────────────────────────────────────────────

        log_df = pd.DataFrame(fold_logs)
        log_csv = os.path.join(self.run_directory, "nested_full_log.csv")
        log_df.to_csv(log_csv, index=False)
        with open (os.path.join(self.run_directory,"results.txt"), 'w') as f:
            f.write(str(results))
        print_log(f"✓ Saved nested‑CV log → {log_csv}")

    # ------------------------------------------------------------
    #  Permutation test – label‑shuffle on fixed CV predictions
    # ------------------------------------------------------------
    def permutation_test_best_model(self,
                                    n_perm: int = 1_000,
                                    random_state: int = 42):
        """
        Ojala‑Garriga label‑shuffle test
        (predictions are fixed; only labels are permuted).

        Saves the 95‑th and 50‑th percentiles for violin‑plot overlays.
        """
        n_perm = self.permu_count  # honour CLI value
        if not hasattr(self, "optimization_results"):
            print_log("Run optimise_hyperparameters first.")
            return

        import numpy as np
        from sklearn.metrics import f1_score, accuracy_score
        rng = np.random.default_rng(random_state)

        best_name = self.optimization_results["best_model_name"]
        fold_f1 = np.asarray(self.optimization_results["best_results"]
                             [best_name]["per_fold_f1"])
        fold_acc = np.asarray(self.optimization_results["best_results"]
                              [best_name]["per_fold_acc"])
        per_fold_true = self.optimization_results["best_results"] \
            [best_name]["per_fold_true"]
        per_fold_pred = self.optimization_results["best_results"] \
            [best_name]["per_fold_pred"]

        # ---------- observed statistic (mean of per‑fold scores) ----------
        obs_mean_f1 = float(fold_f1.mean())
        obs_mean_acc = float(fold_acc.mean())
        print_log(f"Observed μ F1 = {obs_mean_f1:.3f} | μ ACC = {obs_mean_acc:.3f}")

        # ---------- build null distribution ------------------------------
        # concat once for easier shuffling
        y_true_all = np.concatenate(per_fold_true)
        y_pred_all = np.concatenate(per_fold_pred)
        fold_sizes = [len(t) for t in per_fold_true]

        null_mean_f1 = np.empty(n_perm)
        null_mean_acc = np.empty(n_perm)

        for i in tqdm( range(n_perm), desc='Running Permutations'):
            y_perm = rng.permutation(y_true_all)  # shuffle labels
            start = 0
            f1_list, acc_list = [], []
            for n in fold_sizes:  # slice back into folds
                ys = y_perm[start:start + n]
                ps = y_pred_all[start:start + n]
                f1_list.append(f1_score(ys, ps, average="macro"))
                acc_list.append(accuracy_score(ys, ps))
                start += n
            null_mean_f1[i] = np.mean(f1_list)
            null_mean_acc[i] = np.mean(acc_list)

        # ---------- p‑values & critical values ----------------------------
        p_f1 = (np.sum(null_mean_f1 >= obs_mean_f1) + 1) / (n_perm + 1)
        p_acc = (np.sum(null_mean_acc >= obs_mean_acc) + 1) / (n_perm + 1)

        crit95_f1, crit50_f1 = np.percentile(null_mean_f1, [95, 50]).astype(float)
        crit95_acc, crit50_acc = np.percentile(null_mean_acc, [95, 50]).astype(float)

        print_log(f"Permutation p‑values →  F1: {p_f1:.4f} | ACC: {p_acc:.4f}")

        # ---------- save thresholds for violin plot -----------------------
        np.save(os.path.join(self.run_directory, "null_95th_percentile.npy"),
                crit95_f1)
        np.save(os.path.join(self.run_directory, "null_50th_percentile.npy"),
                crit50_f1)

        # ---------- histogram figure --------------------------------------
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

        ax[0].hist(null_mean_f1, 30, alpha=.7)
        ax[0].axvline(obs_mean_f1, color="red", lw=2, label=f"Obs {obs_mean_f1:.3f}")
        ax[0].axvline(crit95_f1, color="black", ls=":", lw=2, label="95 % null")
        ax[0].axvline(crit50_f1, color="grey", ls=":", lw=2, label="Null mean")
        ax[0].set(title=f"Macro‑F1 null distribution\np={p_f1:.4f}",
                  xlabel="Mean F1", ylabel="Count")
        ax[0].legend()

        ax[1].hist(null_mean_acc, 30, alpha=.7, color="mediumaquamarine")
        ax[1].axvline(obs_mean_acc, color="red", lw=2, label=f"Obs {obs_mean_acc:.3f}")
        ax[1].axvline(crit95_acc, color="black", ls=":", lw=2, label="95 % null")
        ax[1].axvline(crit50_acc, color="grey", ls=":", lw=2, label="Null mean")
        ax[1].set(title=f"Accuracy null distribution\np={p_acc:.4f}",
                  xlabel="Mean Accuracy")
        ax[1].legend()

        fig.suptitle(f"Permutation test – {best_name} ({n_perm} permutations)")
        out_png = f"{self.run_directory}/permutation_histogram_{best_name}.png"
        fig.tight_layout()
        fig.savefig(out_png, dpi=300)
        plt.close()
        print_log(f"Saved permutation histogram → {out_png}")

        self.permutation_p_value = {"f1": p_f1, "accuracy": p_acc}
        return self.permutation_p_value, crit95_f1, crit50_f1


    # ──────────────────────────────────────────────────────────────────────────────
    #  GOLD‑STANDARD PERMUTATION TEST  (full re‑training for every shuffle)
    # ──────────────────────────────────────────────────────────────────────────────
    def permutation_test_gold(self,
                              n_perm: int = 1000,
                              random_state: int = 42):
        """
        Fully re‑trains the **winning fixed‑hyper‑param pipeline**
        on *every* label shuffle → unbiased null distribution.

        • Requires that `run_nested_cv_fixed()` has been executed first
          (it decides the “winning” pipeline + stores heuristic models).

        Returns
        -------
        dict  with p‑values and percentile thresholds.
        """
        n_perm=self.permu_count
        if not hasattr(self, "fixed_cv_results"):
            raise RuntimeError("Call run_nested_cv_fixed() before the gold permutation test")

        import numpy as np, matplotlib.pyplot as plt, os
        from tqdm import trange
        from sklearn.base import clone
        from sklearn.metrics import f1_score, accuracy_score

        rng = np.random.default_rng(random_state)

        # ----- data restricted to best‑label pair/triplet --------------------------
        df_best = self.df[self.df["label"].isin(self.best_labels)].reset_index(drop=True)
        X_all = df_best[self.feature_columns]
        y_orig = df_best["label"].to_numpy()
        groups = df_best["session"].values

        outer_cv, _ = self._get_nested_splitters(groups)

        # ----- use the SAME base model that won in run_nested_cv_fixed -------------
        base_model_name = self.fixed_best_model_name
        base_model_dict = {base_model_name:
                               self.fixed_cv_results[base_model_name]
                           }  # only need the name for printing
        # you still have the original model object:
        heuristic_models = self.heuristic_models
        base_model = heuristic_models[base_model_name]

        # ----- observed statistic (already computed) ------------------------------
        obs_f1 = self.fixed_cv_results[base_model_name]["mean_f1"]

        # containers
        null_mean_f1 = np.empty(n_perm)
        null_mean_acc = np.empty(n_perm)

        print_log(f"\n---- GOLD permutation test ({n_perm} shuffles) "
                  f"– evaluating {base_model_name} ----")

        for p in trange(n_perm, desc="Permutations"):
            y_perm = rng.permutation(y_orig)  # shuffle labels **before** CV
            f1_per_fold, acc_per_fold = [], []

            for tr_idx, te_idx in outer_cv.split(X_all, y_perm, groups):
                # split the *permuted* labels exactly like the original CV
                X_tr, X_te = X_all.iloc[tr_idx], X_all.iloc[te_idx]
                y_tr, y_te = y_perm[tr_idx], y_perm[te_idx]

                # ----- identical preprocessing as in run_nested_cv_fixed ----------
                scaler = StandardScaler().fit(X_tr)
                X_tr_s = scaler.transform(X_tr)
                X_te_s = scaler.transform(X_te)

                selector = SelectKBest(f_classif, k=self.n_features_to_select).fit(
                    X_tr_s, y_tr)
                X_tr_sel = selector.transform(X_tr_s)
                X_te_sel = selector.transform(X_te_s)

                mdl = clone(base_model).fit(X_tr_sel, y_tr)
                y_hat = mdl.predict(X_te_sel)

                f1_per_fold.append(f1_score(y_te, y_hat, average="macro"))
                acc_per_fold.append(accuracy_score(y_te, y_hat))

            null_mean_f1[p] = np.mean(f1_per_fold)
            null_mean_acc[p] = np.mean(acc_per_fold)

        # ----- p‑values -----------------------------------------------------------
        obs_acc = self.fixed_cv_results[base_model_name]["mean_acc"]
        p_f1 = (np.sum(null_mean_f1 >= obs_f1) + 1) / (n_perm + 1)
        p_acc = (np.sum(null_mean_acc >= obs_acc) + 1) / (n_perm + 1)

        crit95_f1 = float(np.percentile(null_mean_f1, 95))
        crit50_f1 = float(np.percentile(null_mean_f1, 50))

        # save thresholds for overlay plots if you like
        np.save(os.path.join(self.run_directory, "gold_null95.npy"), crit95_f1)
        np.save(os.path.join(self.run_directory, "gold_null50.npy"), crit50_f1)

        # ----- histogram figure ---------------------------------------------------
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(null_mean_f1, 30, alpha=.7, label="null μ‑F1")
        ax.axvline(obs_f1, color="red", lw=2, label=f"observed {obs_f1:.3f}")
        ax.axvline(crit95_f1, color="black", ls=":", lw=2, label="95 % null")
        ax.set(title=f"Gold permutation – {base_model_name}\n"
                     f"p={p_f1:.4f}", xlabel="mean outer‑fold F1", ylabel="count")
        ax.legend()
        out_png = f"{self.run_directory}/gold_permutation_hist_{base_model_name}.png"
        fig.tight_layout()
        fig.savefig(out_png, dpi=300)
        plt.close()
        print_log(f"Saved gold permutation histogram → {out_png}")

        return dict(p_f1=p_f1, p_acc=p_acc, crit95_f1=crit95_f1)



    # ──────────────────────────────────────────────────────────────────────────────
    #  GOLD‑STANDARD PERMUTATION TEST  (full re‑training for every shuffle)
    # ──────────────────────────────────────────────────────────────────────────────
    def permutation_test_true_cv_gold(self,
                              n_perm: int = 1_000,
                              random_state: int = 42):
        """
        Label‑shuffle permutation test that *re‑trains the winning pipeline*
        inside the **same outer‑CV scheme** used in run_nested_cv_fixed().

        Returns
        -------
        dict with p‑values and critical values for overlay plots.
        """
        # honour CLI / ctor argument
        n_perm = self.permu_count if hasattr(self, "permu_count") else n_perm

        if not hasattr(self, "fixed_cv_results"):
            raise RuntimeError("Run run_nested_cv_fixed() first")

        import numpy as np, matplotlib.pyplot as plt, os
        from tqdm import trange
        from sklearn.base import clone
        from sklearn.metrics import f1_score, accuracy_score
        from sklearn.model_selection import LeaveOneOut

        rng = np.random.default_rng(random_state)

        # ─────────────────────────────────────────────────────────────────
        # 1) Data restricted to winning label pair / triplet
        # ─────────────────────────────────────────────────────────────────
        df_best = self.df[self.df["label"].isin(self.best_labels)].reset_index(drop=True)
        X_all = df_best[self.feature_columns]
        y_orig = df_best["label"].to_numpy()
        groups = df_best["session"].values

        # ─────────────────────────────────────────────────────────────────
        # 2) Outer splitter identical to nested‑CV
        # ─────────────────────────────────────────────────────────────────
        if self.cv_method == "loo":
            outer_cv = LeaveOneOut()
            outer_iter = outer_cv.split(X_all, y_orig)  # no groups
        else:
            outer_cv, _ = self._get_nested_splitters(groups)
            outer_iter = outer_cv.split(X_all, y_orig, groups)

        # ─────────────────────────────────────────────────────────────────
        # 3) Winning base model from nested‑CV
        # ─────────────────────────────────────────────────────────────────
        base_name = self.fixed_best_model_name
        base_model = self.heuristic_models[base_name]
        obs_f1 = self.fixed_cv_results[base_name]["mean_f1"]
        obs_acc = self.fixed_cv_results[base_name]["mean_acc"]

        null_mean_f1 = np.empty(n_perm, dtype=float)
        null_mean_acc = np.empty(n_perm, dtype=float)

        print_log(f"\n---- GOLD permutation ({n_perm} shuffles) "
                  f"– evaluating {base_name} ----")

        # ─────────────────────────────────────────────────────────────────
        # 4) Permutation loop
        # ─────────────────────────────────────────────────────────────────
        for p in trange(n_perm, desc="Permutations"):
            y_perm = rng.permutation(y_orig)  # shuffle labels BEFORE outer CV
            f1_per_fold, acc_per_fold, fold_sizes = [], [], []

            # iterate outer folds with the *same* splitter
            if self.cv_method == "loo":
                split_iter = outer_cv.split(X_all, y_perm)
            else:
                split_iter = outer_cv.split(X_all, y_perm, groups)

            for tr_idx, te_idx in split_iter:
                X_tr, X_te = X_all.iloc[tr_idx], X_all.iloc[te_idx]
                y_tr, y_te = y_perm[tr_idx], y_perm[te_idx]

                # identical preprocessing as in run_nested_cv_fixed
                scaler = StandardScaler().fit(X_tr)
                sel = SelectKBest(f_classif,
                                  k=self.n_features_to_select).fit(
                    scaler.transform(X_tr), y_tr)

                X_tr_sel = sel.transform(scaler.transform(X_tr))
                X_te_sel = sel.transform(scaler.transform(X_te))

                y_hat = clone(base_model).fit(X_tr_sel, y_tr).predict(X_te_sel)
                f1_per_fold.append(f1_score(y_te, y_hat, average="macro"))
                acc_per_fold.append(accuracy_score(y_te, y_hat))
                fold_sizes.append(len(te_idx))  # ← NEW

            # these were session-meaned, we need sample meaned
            # null_mean_f1[p] = float(np.mean(f1_per_fold))
            # null_mean_acc[p] = float(np.mean(acc_per_fold))
            w = np.asarray(fold_sizes, dtype=float)
            null_mean_f1[p] = float(np.average(f1_per_fold, weights=w))
            null_mean_acc[p] = float(np.average(acc_per_fold, weights=w))

        # ─────────────────────────────────────────────────────────────────
        # 5) p‑values & critical values
        # ─────────────────────────────────────────────────────────────────
        p_f1 = (np.sum(null_mean_f1 >= obs_f1) + 1) / (n_perm + 1)
        p_acc = (np.sum(null_mean_acc >= obs_acc) + 1) / (n_perm + 1)

        crit95_f1, crit50_f1 = np.percentile(null_mean_f1, [95, 50]).astype(float)

        # save thresholds for overlay plots
        np.save(os.path.join(self.run_directory, "gold_null95.npy"), crit95_f1)
        np.save(os.path.join(self.run_directory, "gold_null50.npy"), crit50_f1)

        # ─────────────────────────────────────────────────────────────────
        # 6) Quick histogram figure (optional)
        # ─────────────────────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist(null_mean_f1, 30, alpha=.7, label="null μ‑F1")
        ax.axvline(obs_f1, color="red", lw=2, label=f"observed {obs_f1:.3f}")
        ax.axvline(crit50_f1, color="brown", ls="-", lw=2,
                   label=f"50 % null ({crit50_f1:.3f})")
        ax.axvline(crit95_f1, color="black", ls=":", lw=2,
                   label=f"95 % null ({crit95_f1:.3f})")
        ax.set(title=f"Gold permutation – {base_name}\n"
                     f"p={p_f1:.4f}",
               xlabel="mean outer‑fold F1", ylabel="count")
        ax.legend()
        out_png = os.path.join(self.run_directory,
                               f"gold_permutation_hist_{base_name}.png")
        fig.tight_layout()
        fig.savefig(out_png, dpi=300)
        plt.close()
        print_log(f"Saved gold permutation histogram → {out_png}")

        return dict(p_f1=p_f1, p_acc=p_acc,
                    crit95_f1=crit95_f1, crit50_f1=crit50_f1)

    def permutation_test_final_true_cv_gold(self,
                                        n_perm: int = 1_000,
                                        random_state: int = 42,
                                        alpha: float = 0.05,
                                        n_jobs: int = -1,
                                        batch: int = 50):
        """
        Full nested-CV permutation test (multicore + progress bar) with separate histograms
        for F1 and accuracy, and theoretical binomial overlay on accuracy plot.

        Parameters
        ----------
        n_perm       Number of permutations
        random_state RNG seed
        alpha        Kept for API compat (not used here)
        n_jobs       CPU cores for joblib (-1 = all)
        batch        How many permutations each Parallel call runs before tqdm updates

        Returns
        -------
        dict with p-values and critical values for overlay plots
        """
        # honour CLI / ctor argument
        n_perm = self.permu_count if hasattr(self, "permu_count") else n_perm
        if not hasattr(self, "fixed_cv_results"):
            raise RuntimeError("Run run_nested_cv_fixed() first")

        import os, numpy as np
        from joblib import Parallel, delayed
        from tqdm import tqdm
        from sklearn.base import clone
        from sklearn.metrics import f1_score, accuracy_score
        from sklearn.model_selection import (LeaveOneOut, StratifiedKFold,
                                             StratifiedGroupKFold, GroupShuffleSplit,
                                             LeavePGroupsOut)
        from sklearn.preprocessing import StandardScaler
        from sklearn.feature_selection import SelectKBest, f_classif
        # ensure scipy available for binomial
        try:
            from scipy.stats import binom
        except ImportError:
            binom = None

        rng_master = np.random.default_rng(random_state)

        # Data restricted to winning label set
        # labels are predefined so this is not relevant, these did not 'win' anyting as there is only 2 labels passed to binary classifiers
        df_best = self.df[self.df["label"].isin(self.best_labels)].reset_index(drop=True)
        X_all = df_best[self.feature_columns]
        y_orig = df_best["label"].to_numpy()
        groups = df_best["session"].values

        # Outer splitter identical to nested-CV
        if self.cv_method == "loo":
            outer_cv = LeaveOneOut()
            outer_split = lambda X, y: outer_cv.split(X, y)
            inner_proto_template = None
        else:
            outer_cv, inner_proto_template = self._get_nested_splitters(groups)
            outer_split = lambda X, y: outer_cv.split(X, y, groups)

        # observed metrics
        best_name = self.fixed_best_model_name
        obs_f1 = self.fixed_cv_results[best_name]["mean_f1"]
        obs_acc = self.fixed_cv_results[best_name]["mean_acc"]

        print_log(f"\n---- GOLD permutation ({n_perm} shuffles, batch={batch}, n_jobs={n_jobs}) ----")

        # ------------------------------------------------------------------
        k_grid = getattr(self, "k_grid", self.frozen_features)  # candidate #features

        # ------------------------------------------------------------------

        def _one_permutation(seed: int) -> tuple[float, float]:
            rng = np.random.default_rng(seed)
            y_perm = rng.permutation(y_orig)

            f1_per_fold, acc_per_fold, sizes = [], [], []

            for tr_idx, te_idx in outer_split(X_all, y_perm):
                X_tr, X_te = X_all.iloc[tr_idx], X_all.iloc[te_idx]
                y_tr, y_te = y_perm[tr_idx], y_perm[te_idx]
                g_tr = groups[tr_idx]

                # ---------- build inner splitter *once* for this outer fold ----------
                if self.cv_method == "loo":
                    inner_cv = StratifiedKFold(n_splits=min(3, len(X_tr) - 1),
                                               shuffle=True, random_state=24)
                    inner_split = lambda X, y: inner_cv.split(X, y)
                else:
                    proto = inner_proto_template
                    needs_g = isinstance(proto, (StratifiedGroupKFold,
                                                 GroupShuffleSplit,
                                                 LeavePGroupsOut))
                    inner_split = (lambda X, y: proto.split(X, y, g_tr)
                    if needs_g else lambda X, y: proto.split(X, y))

                # ---------- inner loop: score every (model, k) combination ----------
                inner_mean_f1 = {}
                for mdl_name, base_mdl in self.heuristic_models.items():
                    for k_feats in k_grid:
                        scores = []
                        for in_tr, in_val in inner_split(X_tr, y_tr):
                            scaler = StandardScaler().fit(X_tr.iloc[in_tr])

                            sel = SelectKBest(f_classif, k=k_feats).fit(
                                scaler.transform(X_tr.iloc[in_tr]),
                                y_tr[in_tr])

                            Xtr_sel = sel.transform(scaler.transform(X_tr.iloc[in_tr]))
                            Xval_sel = sel.transform(scaler.transform(X_tr.iloc[in_val]))

                            y_hat = clone(base_mdl).fit(Xtr_sel, y_tr[in_tr]).predict(Xval_sel)
                            scores.append(f1_score(y_tr[in_val], y_hat, average="macro"))

                        inner_mean_f1[(mdl_name, k_feats)] = float(np.mean(scores))

                # winner = model–k pair with best inner F1
                (winner_name, winner_k) = max(inner_mean_f1, key=inner_mean_f1.get)
                win_mdl = self.heuristic_models[winner_name]

                # ---------- retrain winner on full outer-train, test on outer-test ----------
                scaler_full = StandardScaler().fit(X_tr)
                sel_full = SelectKBest(f_classif, k=winner_k).fit(
                    scaler_full.transform(X_tr), y_tr)

                Xtr_sel = sel_full.transform(scaler_full.transform(X_tr))
                Xte_sel = sel_full.transform(scaler_full.transform(X_te))

                y_pred = clone(win_mdl).fit(Xtr_sel, y_tr).predict(Xte_sel)

                f1_per_fold.append(f1_score(y_te, y_pred, average="macro"))
                acc_per_fold.append(accuracy_score(y_te, y_pred))
                sizes.append(len(te_idx))

            w = np.asarray(sizes, float)
            return (float(np.average(f1_per_fold, weights=w)),
                    float(np.average(acc_per_fold, weights=w)))
        # Parallel shuffle
        seeds = rng_master.integers(0, 2**32-1, size=n_perm)
        null_f1, null_acc = [], []
        for start in tqdm(range(0, n_perm, batch), desc="Permutations", unit="batch"):
            batch_seeds = seeds[start:start+batch]
            batch_res = Parallel(n_jobs=n_jobs, backend="loky")(delayed(_one_permutation)(s)
                                                                for s in batch_seeds)
            bf1, bacc = zip(*batch_res)
            null_f1.extend(bf1)
            null_acc.extend(bacc)

        null_mean_f1 = np.asarray(null_f1)
        null_mean_acc = np.asarray(null_acc)

        # p-values & thresholds
        p_f1 = (np.sum(null_mean_f1 >= obs_f1) + 1) / (n_perm + 1)
        p_acc = (np.sum(null_mean_acc >= obs_acc) + 1) / (n_perm + 1)
        crit99_f1, crit95_f1, crit50_f1 = np.percentile(null_mean_f1, [99, 95, 50]).astype(float)
        crit99_acc, crit95_acc, crit50_acc = np.percentile(null_mean_acc, [99, 95, 50]).astype(float)

        # Save thresholds
        np.save(os.path.join(self.run_directory, "gold_null99_f1.npy"), crit99_f1)
        np.save(os.path.join(self.run_directory, "gold_null95_f1.npy"), crit95_f1)
        np.save(os.path.join(self.run_directory, "gold_null50_f1.npy"), crit50_f1)
        np.save(os.path.join(self.run_directory, "gold_null99_acc.npy"), crit99_acc)
        np.save(os.path.join(self.run_directory, "gold_null95_acc.npy"), crit95_acc)
        np.save(os.path.join(self.run_directory, "gold_null50_acc.npy"), crit50_acc)

        # Plot F1 histogram
        import matplotlib.pyplot as plt
        fig_f1, ax_f1 = plt.subplots(figsize=(6,4))
        ax_f1.hist(null_mean_f1, bins=75, density=True, alpha=0.7, label="null μ-F1")
        ax_f1.axvline(obs_f1, color="red", lw=2, label=f"observed {obs_f1:.3f}")
        ax_f1.axvline(crit50_f1, color="brown", lw=2, label=f"50% null ({crit50_f1:.3f})")
        ax_f1.axvline(crit95_f1, color="black", ls=":", lw=2, label=f"95% null ({crit95_f1:.3f})")
        ax_f1.set(title=f"Null F1 distribution (p={p_f1:.4f})",
                  xlabel="mean outer-fold F1", ylabel="density")
        ax_f1.legend(frameon=False)
        fig_f1.tight_layout()
        path_f1 = os.path.join(self.run_directory, "gold_perm_hist_f1.png")
        fig_f1.savefig(path_f1, dpi=300)
        plt.close(fig_f1)
        print_log(f"✓ Saved F1 permutation histogram → {path_f1}")

        # -------------------------------------------------------------------------
        # Plot accuracy histogram *as density* with Binomial–PMF overlay
        # -------------------------------------------------------------------------
        fig_acc, ax_acc = plt.subplots(figsize=(6, 4))

        # 1) permutation null as a *density* histogram
        ax_acc.hist(
            null_mean_acc,
            bins=50,
            density=True,  # ← makes area = 1
            alpha=0.70,
            label="null μ-accuracy (density)",
            color="#e6550d"
        )
        from scipy.stats import gaussian_kde
        # KDE curve
        kde = gaussian_kde(null_mean_acc)
        xs = np.linspace(null_mean_acc.min(), null_mean_acc.max(), 400)
        ax_acc.plot(xs, kde(xs), lw=1, color="black",
                    label="null KDE")
        ax_acc.fill_between(xs, kde(xs), alpha=0.20, color="black")

        # 2) theoretical Binomial PMF, also a density (area = 1)
        if binom is not None:
            p0 = 1.0 / self.top_n_labels  # chance level
            N_total = int(sum(self.fixed_cv_results[best_name]["per_fold_n"]))
            k = np.arange(0, N_total + 1)
            pmf = binom.pmf(k, N_total, p0)  # already a density
            delta = 1.0 / N_total  # step between adjacent k/N
            pmf_dens = pmf / delta  # ← divide by Δx
            x = k / N_total  # accuracy axis

            # plot & optionally fill under the curve
            ax_acc.plot(x, pmf_dens, '-', lw=1,
                        label=f"Binomial PMF (p={p0:.2f})", color="#3182bd")
            ax_acc.fill_between(x, pmf_dens, alpha=0.20, color="#9ecae1")

        # 3) observed & null-percentile lines
        ax_acc.axvline(obs_acc, color="red", lw=1,
                       label=f"observed {obs_acc:.3f}")
        ax_acc.axvline(crit50_acc, color="grey", lw=1,
                       label=f"50 % null ({crit50_acc:.3f})")
        ax_acc.axvline(crit95_acc, color="green", lw=1,
                       label=f"95 % null ({crit95_acc:.3f})")
        ax_acc.axvline(crit99_acc, color="purple", lw=1,
                       label=f"99 % null ({crit99_acc:.3f})")

        # 4) axes, legend, save
        ax_acc.set(
            title=f"Null accuracy distribution vs. Binomial; p={p_acc:.4f}",
            xlabel="mean outer-fold accuracy",
            ylabel="density"  # ← density now
        )
        ax_acc.set_xlim(0.3, 0.7)
        ax_acc.legend(frameon=False, loc="upper left")

        fig_acc.tight_layout()
        path_acc = os.path.join(self.run_directory, "gold_perm_hist_accuracy.png")
        fig_acc.savefig(path_acc, dpi=300)
        plt.close(fig_acc)
        print_log(f"✓ Saved accuracy permutation histogram → {path_acc}")

        return dict(
            p_f1=p_f1, p_acc=p_acc,
            crit95_f1=crit95_f1, crit50_f1=crit50_f1,
            crit95_acc=crit95_acc, crit50_acc=crit50_acc
        )

    def add_accuracy_reference_lines(self, n_classes: int,
                                     ax: plt.Axes | None = None,
                                     colors: dict | None = None,
                                     loc: str = "lower right"):
        """
        Draw chance + two significance thresholds on an existing plot.

        Parameters
        ----------
        n_classes : int           2 or 3
        ax        : matplotlib Axes (default: current axes)
        colors    : dict with keys 'chance', 'p05', 'p005'
        loc       : legend location
        """
        if ax is None:
            ax = plt.gca()

        # -------------------- thresholds for your sample sizes -------------------
        thresh = {
            2: dict(chance=0.50, p05=0.585, p01=0.617),  # n = 120
            3: dict(chance=1 / 3, p05=0.39345, p01=0.42077),  # n = 161 # to be fixed, trio count is probably different now
        }
        if n_classes not in thresh:
            raise ValueError("n_classes must be 2 or 3")

        col = colors or dict(chance="grey", p05="green", p01="purple")



        ax.axhline(thresh[n_classes]["p01"],
                   ls="--", lw=1.5, color=col["p01"],
                   label=f"p = 0.01 ({thresh[n_classes]['p01']:.3f})")
        ax.axhline(thresh[n_classes]["p05"],
                   ls="--", lw=1.5, color=col["p05"],
                   label=f"p = 0.05 ({thresh[n_classes]['p05']:.3f})")
        ax.axhline(thresh[n_classes]["chance"],
                   ls="--", lw=1.5, color=col["chance"],
                   label=f"Chance ({thresh[n_classes]['chance']:.3f})")


        ax.legend(loc='lower right', frameon=False, fontsize=9)

    def evaluate_best_model(self):
        """
        Evaluate the best model from hyperparameter optimization using the same
        cross-validation approach as the original models.
        """
        print_log("\n---- EVALUATING BEST OPTIMIZED MODEL ----")

        if not hasattr(self, "optimization_results"):
            print_log("Run optimize_hyperparameters first")
            return

        best_model = self.optimization_results["best_model"]
        best_model_name = self.optimization_results["best_model_name"]

        # Focus on data for the best label combination
        df_best = self.df[self.df["label"].isin(self.best_labels)]
        X_best = df_best[self.feature_columns]
        y_best = df_best["label"]

        # Set up cross-validation
        if self.cv_method == "loo":
            from sklearn.model_selection import LeaveOneOut

            cv = LeaveOneOut()
            groups_eval = df_best["session"].values
        elif self.cv_method == "kfold":
            from sklearn.model_selection import StratifiedKFold

            cv = StratifiedKFold(n_splits=self.kfold_splits)
            groups_eval = df_best["session"].values
        elif self.cv_method == "loso":
            cv = LeaveOneGroupOut()
            groups_eval = df_best["session"].values
        elif self.cv_method == "lmoso":
            cv = LeavePGroupsOut(self.lmoso_leftout)
            groups_eval = df_best["session"].values

        else:  # TODO Holdout is not working properly, fix it
            from sklearn.model_selection import train_test_split

            X_train, X_test, y_train, y_test = train_test_split(
                X_best,
                y_best,
                test_size=self.test_size,
                random_state=42,
                stratify=y_best,
            )
            groups_eval = df_best["session"].values

        y_true_all = []
        y_pred_all = []
        misclassified_samples = []

        from sklearn.base import clone
        if self.cv_method in ["loo", "kfold", "loso", "lmoso"]:
            # Cross-validation approach (LOO or k-fold)

            for fold_idx, (train_idx, test_idx) in enumerate(
                cv.split(X_best, y_best, groups=groups_eval)
                if groups_eval is not None
                else cv.split(X_best, y_best)
            ):
                X_train, X_test = X_best.iloc[train_idx], X_best.iloc[test_idx]
                y_train, y_test = y_best.iloc[train_idx], y_best.iloc[test_idx]
                test_indices = y_test.index.tolist()

                # Clone the model for each fold to ensure independence

                model_clone = clone(best_model)

                # Fit and predict
                model_clone.fit(X_train, y_train)
                y_pred = model_clone.predict(X_test)

                y_true_all.extend(y_test.tolist())
                y_pred_all.extend(y_pred)

                # Record misclassified samples
                for idx, true_val, pred_val in zip(test_indices, y_test, y_pred):
                    if true_val != pred_val:
                        misclassified_samples.append(
                            {
                                "index": idx,
                                "true_label": true_val,
                                "predicted_label": pred_val,
                                "fold": fold_idx,
                            }
                        )
        else:
            # Holdout approach
            best_model.fit(X_train, y_train)
            y_pred = best_model.predict(X_test)
            y_true_all = y_test.tolist()
            y_pred_all = y_pred.tolist()
            test_indices = y_test.index.tolist()

            # Record misclassified samples
            for idx, true_val, pred_val in zip(test_indices, y_test, y_pred):
                if true_val != pred_val:
                    misclassified_samples.append(
                        {
                            "index": idx,
                            "true_label": true_val,
                            "predicted_label": pred_val,
                        }
                    )

        # Calculate and display metrics
        metrics = self._calculate_metrics(y_true_all, y_pred_all)
        print_log(f"Optimized {best_model_name} Performance:")
        for metric, value in metrics.items():
            print_log(f"  {metric}: {value:.4f}")

        # Generate confusion matrix
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(y_true_all, y_pred_all)

        # Visualize confusion matrix
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=self.best_labels,
            yticklabels=self.best_labels,
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix for Optimized {best_model_name}")
        plt.tight_layout()
        plt.savefig(
            f"{self.run_directory}/optimized_model_confusion_matrix.png", dpi=300
        )
        plt.close()

        # Update detailed_results with optimized model's performance
        self.detailed_results[self.best_labels]["misclassified_samples"][
            best_model_name
        ] = misclassified_samples
        self.detailed_results[self.best_labels]["metrics"][best_model_name] = metrics
        self.detailed_results[self.best_labels]["confusion_matrices"][
            best_model_name
        ] = cm

        # Check if optimized model is better than previous best
        current_best_model = self.detailed_results[self.best_labels]["best_model"]
        current_f1 = self.detailed_results[self.best_labels]["metrics"][
            current_best_model
        ]["f1_macro"]
        optimized_f1 = metrics["f1_macro"]

        if optimized_f1 > current_f1:
            print_log(
                f"Optimized model {best_model_name} outperforms previous best {current_best_model}. Updating best model."
            )
            self.detailed_results[self.best_labels]["best_model"] = best_model_name

    def save_best_models(self):
        """Save both SVM and Random Forest models for the best label combination"""
        if not hasattr(self, "best_labels"):
            print_log(
                "No best labels identified. Run evaluate_label_combinations first."
            )
            return

        print_log("\n---- SAVING BASE MODELS ----")
        df_best = self.df[self.df["label"].isin(self.best_labels)]
        X_best = df_best[self.feature_columns]
        y_best = df_best["label"]

        # Define base models with original parameters
        models = {
            "RandomForest": RandomForestClassifier(
                n_estimators=200, max_depth=5, class_weight="balanced", random_state=48
            ),
            "SVM": SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                class_weight="balanced",
                random_state=42,
                probability=True,
            ),
        }

        # Create and save pipelines
        for name, model in models.items():
            # Create full preprocessing + classification pipeline
            pipeline = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("selector", SelectKBest(f_classif, k=self.n_features_to_select)),
                    ("classifier", model),
                ]
            )

            # Fit on entire best-label dataset
            pipeline.fit(X_best, y_best)

            # Save model
            filename = f"{self.run_directory}/base_{name.lower()}_model.pkl"
            joblib.dump(pipeline, filename)
            print_log(f"Saved {name} model to {filename}")

    # STEP 2: Modify the run_complete_analysis method to include hyperparameter optimization
    def run_complete_analysis(self, optimize_hyperparams=False, n_iter=25):
        """Run the complete analysis pipeline with fixed information leakage issues"""
        self.load_data()

        # Modified calls to prevent info leaks
        # self.preprocess_data()  # Now just for exploration
        # self.feature_selection()  # Now just for informational purposes
        # self.evaluate_label_combinations()  # This method now handles proper CV and feature selection

        # Main hyperparameter optimization (we simply always want to run with this option)
        if optimize_hyperparams:
            self.optimize_hyperparameters(n_iter=n_iter)
            self.evaluate_best_model() #already saves conf. matrx
            self.visualize_pca()
            self.visualize_pca_3d()
            self.permutation_test_best_model(n_perm=1_000)
            self.plot_outer_fold_distribution()
            self._visualize_optimization_results( # feature importance saved with progress
                self.optimization_results["all_results"]
            )

            # TODO change these so they report the correct model
            # self.visualize_pca()
            # self.visualize_pca_3d()

        # main pipeline we have right now
        else: # run nested-cv without bayesian with gold standard permuation
            self.run_true_nested_cv()  # chooses the heuristic winner
            if self.permu_count!=0:
                self.permutation_test_final_true_cv_gold()
                self.plot_outer_fold_distribution()
                self.visualize_confusion_matrix()
            # these are not relevant for this section
            # self.visualize_pca()
            # self.visualize_pca_3d()
            # self.analyze_channel_distribution()
            # self.visualize_feature_importance()
            # self.visualize_metrics_comparison()
            # self.export_results()



        print_log("\n---- ANALYSIS COMPLETE ----")
        print_log(f"Most separable label combination: {self.best_labels}")
        best_model = self.detailed_results[self.best_labels]["best_model"]
        metrics = self.detailed_results[self.best_labels]["metrics"][best_model]

        print_log(f"Best model: {best_model}")
        print_log(f"Performance metrics:")
        for metric, value in metrics.items():
            print_log(f"  {metric}: {value:.4f}")

        print_log(f"\nTop {self.n_features_to_select} informational features:")
        for i, feature in enumerate(self.selected_features):
            print_log(f"  {i + 1}. {feature}")

        print_log("\nNOTE: The actual features used in model evaluation were selected")
        print_log(
            "independently within each cross-validation fold to prevent information leakage."
        )
        self.save_best_models()  # <-- New addition

        # Add hyperparameter optimization results if applicable
        if optimize_hyperparams and hasattr(self, "optimization_results"):
            print_log("\n---- HYPERPARAMETER OPTIMIZATION RESULTS ----")
            best_model_name = self.optimization_results["best_model_name"]
            best_score = self.optimization_results["best_score"]
            print_log(f"Best optimized model: {best_model_name}")
            print_log(f"Best optimized score (F1): {best_score:.4f}")
            print_log(
                f"Optimized model saved to: {self.run_directory}/best_{best_model_name.lower()}_model.pkl"
            )

    def _default_heuristic_models(self):
        """Return the fixed‑parameter models you used before."""
        return {
            # "RandomForest": RandomForestClassifier(
            #     n_estimators=80, max_depth=3, min_samples_leaf=2,
            #     class_weight="balanced", random_state=42, n_jobs=-1),
            # "SVM": SVC(kernel="rbf", C=0.8, gamma="scale",
            #            class_weight="balanced", probability=True, random_state=42),
            # "ElasticNetLogReg": LogisticRegression(
            #     penalty="elasticnet", C=1.0, l1_ratio=0.5,
            #     solver="saga", max_iter=4000, class_weight="balanced", random_state=42),
            "ExtraTrees": ExtraTreesClassifier(
                n_estimators=120, max_depth=4, min_samples_leaf=2,
                class_weight="balanced", random_state=42, n_jobs=-1),
            # "HGBClassifier": HistGradientBoostingClassifier(
            #     learning_rate=0.05, max_depth=3, max_iter=80,
            #     class_weight="balanced", random_state=42),
            "kNN": KNeighborsClassifier(n_neighbors=7, weights="distance"),
            "GaussianNB": GaussianNB(),
            "ShrinkageLDA": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
        }


def print_log(x):
    message = str(x)
    print(message)
    with open(analyzer.run_directory + "/log.txt", "a", encoding="utf-8") as f:
        f.write(message + "\n")


# STEP 3: Update the main script to accept a hyperparameter optimization flag

if __name__ == "__main__":
    import argparse
    import time

    # parser.add_argument('--features_file', type=str, default="data/normalized_merges/14_all_channels/oz_norm-ComBat.csv",
    # parser.add_argument('--features_file', type=str, default="data/merged_features/14_sessions_merge_1743884222/59_balanced_o1.csv",

    start_time = time.time()
    parser = argparse.ArgumentParser(description="EEG Data Analysis Tool")
    parser.add_argument(
        "--features_file",
        type=str,
        default="o2_comBat",
        help="Path to the CSV file containing EEG features",
    )
    parser.add_argument(
        "--run_directory",
        type=str,
        default=None,
        help="Path to save"
    )
    parser.add_argument(
        "--top_n_labels",
        type=int,
        default=2,
        help="Number of labels to analyze (default: 2)",
    )
    parser.add_argument(
        "--n_features",
        type=int,
        default=20,
        help="Number of top features to select (default: 8)",
    )
    parser.add_argument(
        "--channel_approach",
        type=str,
        default="pooled",
        choices=["pooled", "separate", "features"],
        help="How to handle channel data (default: pooled)",
    )
    parser.add_argument(
        "--cv_method",
        type=str,
        default="loso",
        choices=["loo", "kfold", "holdout", "loso", "lmoso"],
        help="Cross-validation method (default: loso)",
    )

    parser.add_argument(
        "--lmoso_leftout",
        type=int,
        default=2,
        help="How many sessions to leave out per fold when cv_method='lmoso'",
    )

    parser.add_argument(
        "--cv_version",
        type=str,
        default="simple",
        choices=["extended", "simple"],
        help="Cross-validation method (default: extended)",
    )
    parser.add_argument(
        "--kfold_splits",
        type=int,
        default=5,
        help="Number of splits for k-fold cross-validation (default: 5)",
    )
    parser.add_argument(
        "--test_size",
        type=float,
        default=0.25,
        help="Test set size for holdout validation (default: 0.2)",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help='Run an evaluation task, saves them under eval_preceding'
    )
    parser.add_argument(
        "--optimize", action="store_true", help="Perform hyperparameter optimization"
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=25,
        help="Number of iterations for hyperparameter optimization (default: 25)",
    ),
    parser.add_argument(
        "--permu_count",
        type=int,
        default=0,
        help="Number of permutations to run optimization (default: 1k)",
    )
    parser.add_argument(
        "--frozen_features",
        type=int,
        default=None,
        help="Pass it to freeze features to a number of features (default: unfrozen, k=[2,3,5,10,15]"
    )

    args = parser.parse_args()

    analyzer = EEGAnalyzer(
        features_file=args.features_file,
        top_n_labels=args.top_n_labels,
        n_features_to_select=args.n_features,
        channel_approach=args.channel_approach,
        cv_method=args.cv_method,
        cv_version=args.cv_version,
        kfold_splits=args.kfold_splits,
        test_size=args.test_size,
        permu_count=args.permu_count,
        lmoso_leftout=args.lmoso_leftout,
        optimize=args.optimize,
        eval=args.eval,
        frozen_features=args.frozen_features,
        run_directory=args.run_directory,
    )

    analyzer.run_complete_analysis(
        optimize_hyperparams=args.optimize, n_iter=args.n_iter
    )
    open(f"{analyzer.run_directory}/Completed.bat", "w")
    print_log(f"Analysis took  {(time.time() - start_time):.2f} seconds")
