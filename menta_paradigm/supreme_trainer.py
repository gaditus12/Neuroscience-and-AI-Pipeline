import os
import time
import argparse
import numpy as np
import pandas as pd
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.calibration import CalibratedClassifierCV
from tqdm import tqdm
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import (
    LeaveOneGroupOut,
    LeavePGroupsOut,
    cross_val_predict,
    StratifiedKFold,
    StratifiedGroupKFold,
    GroupShuffleSplit,
)
from sklearn.metrics import (
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
    f1_score,
    accuracy_score,
)
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    HistGradientBoostingClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.base import clone

# Optional: Import Bayesian optimization if available
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical

    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print(
        "Warning: scikit-optimize not available. Install with 'pip install scikit-optimize' to use hyperparameter optimization."
    )


class SupremeTrainer:
    def __init__(
        self,
        channels_models,
        top_n_labels=2,
        n_features_to_select=15,
        cv_method="loso",
        kfold_splits=5,
        lmoso_leftout=2,
        permu_count=1000,
        data_path="data/final_sets/all_channels_binary",
        random_state=42,
    ):
        """
        Initialize the SupremeTrainer with multiple channel-model pairs and configuration.

        Args:
            channels_models (dict): Dictionary mapping channel names to model configs.
                Format: {'o1': {'model': 'rf', 'accuracy': 0.65}, ...}
            top_n_labels (int): Number of labels to use (typically 2 for binary)
            n_features_to_select (int): Number of features to select per channel-model
            cv_method (str): Cross-validation method ("loso", "lmoso", "kfold")
            kfold_splits (int): Number of splits for k-fold CV
            lmoso_leftout (int): Number of sessions to leave out for LMOSO CV
            permu_count (int): Number of permutations for significance testing
            data_path (str): Path to directory containing channel-specific data files
            random_state (int): Random seed for reproducibility
        """
        # Store configuration parameters
        self.channels_models = channels_models
        self.top_n_labels = top_n_labels
        self.n_features_to_select = n_features_to_select
        self.cv_method = cv_method.lower()
        self.kfold_splits = kfold_splits
        self.lmoso_leftout = lmoso_leftout
        self.permu_count = permu_count
        self.data_path = data_path
        self.random_state = random_state

        # -----------------  NEW CONFIG PARAMS for educated confidence  -----------------
        self.alpha_conf = 0.7  # 0 = ignore confidence, 1 = ignore global accuracy
        self.conf_scale = "softmax"  # or 'minmax'  (how to normalise confidences)
        # -------------------------------------------------------

        # Create model name formatter
        to_k = lambda n: (
            f"{n / 1000:.1f}k".rstrip("0").rstrip(".") if n >= 1000 else str(n)
        )
        permu_count_str = to_k(permu_count)

        # Create output directory
        timestamp = int(time.time())
        chan_models_str = "_".join(
            [f"{ch}_{cfg['model']}" for ch, cfg in channels_models.items()]
        )
        self.run_directory = os.path.join(
            "supreme_model_outputs",
            f"{chan_models_str}_{cv_method}_{permu_count_str}_run_{timestamp}",
        )
        os.makedirs(self.run_directory, exist_ok=True)

        # Initialize data structures
        self.channel_data = {}  # Will hold DataFrames for each channel
        self.feature_columns = {}  # Will store feature columns for each channel
        self.best_labels = None  # Will store the best label combination
        self.unique_labels = None  # Will store all unique labels from the data
        self.template_pipelines = {}  # Will store pipeline templates

        # Create subdirectory for fold predictions
        self.fold_predictions_dir = os.path.join(self.run_directory, "fold_predictions")
        os.makedirs(self.fold_predictions_dir, exist_ok=True)

        # Log configuration
        with open(f"{self.run_directory}/config.txt", "w") as f:
            f.write(f"Channels and models: {channels_models}\n")
            f.write(f"CV method: {cv_method}\n")
            f.write(f"Top N labels: {top_n_labels}\n")
            f.write(f"Features to select: {n_features_to_select}\n")
            f.write(f"Permutation count: {permu_count}\n")
            f.write(f"Random state: {random_state}\n")
            f.write(f"Alpha value {self.alpha_conf}\n")
            f.write(f"Confidence scale: {self.conf_scale}\n")

        # Log environment info
        self._log_environment_info()

        # Print welcome message
        print_log(self, "---- SUPREME TRAINER INITIALIZED ----")
        print_log(self, f"Output directory: {self.run_directory}")
        print_log(self, f"Channels and models: {channels_models}")
        print_log(self, f"CV method: {cv_method}")

    def _log_environment_info(self):
        """Log environment information for reproducibility."""
        import sys
        import pkg_resources
        import platform

        info = {
            "python_version": sys.version,
            "platform": platform.platform(),
            "packages": {pkg.key: pkg.version for pkg in pkg_resources.working_set},
        }

        # Save to file
        with open(f"{self.run_directory}/environment_info.txt", "w") as f:
            f.write("Python version: " + info["python_version"] + "\n")
            f.write("Platform: " + info["platform"] + "\n")
            f.write("\nPackage versions:\n")
            for pkg, version in info["packages"].items():
                f.write(f"{pkg}: {version}\n")

        return info

    def load_data(self):
        """
        Load data for each channel specified in channels_models.
        """
        print_log(self, "---- LOADING DATA ----")

        for channel, config in self.channels_models.items():
            # Construct file path
            file_path = os.path.join(self.data_path, f"{channel}.csv")

            # Check if file exists
            if not os.path.isfile(file_path):
                print_log(
                    self, f"Error: File not found for channel {channel}: {file_path}"
                )
                continue

            # Load data
            print_log(self, f"Loading data for channel {channel} from {file_path}")
            df = pd.read_csv(file_path)

            # Ensure 'session' column exists (required for LOSO/LMOSO CV)
            if "session" not in df.columns:
                print_log(self, f"Error: 'session' column missing in {channel} data")
                continue

            # Identify feature columns (all except metadata)
            feature_cols = [
                col for col in df.columns if col not in ["label", "channel", "session"]
            ]
            self.feature_columns[channel] = feature_cols

            # Store data
            self.channel_data[channel] = df

            # Log info
            print_log(
                self,
                f"Channel {channel}: {len(df)} samples, {len(feature_cols)} features",
            )
            print_log(self, f"Labels in {channel}: {df['label'].unique()}")
            print_log(self, f"Sessions in {channel}: {df['session'].unique()}")

        # Check if any data was loaded
        if not self.channel_data:
            raise ValueError(
                "No data was loaded for any channel. Please check file paths."
            )

        # Get unique labels across all channels (should be the same for all)
        all_labels = set()
        for channel, df in self.channel_data.items():
            all_labels.update(df["label"].unique())

        self.unique_labels = sorted(list(all_labels))
        print_log(
            self,
            f"Found {len(self.unique_labels)} unique labels across all channels: {self.unique_labels}",
        )

        # Restrict to top_n_labels if specified
        if self.top_n_labels < len(self.unique_labels):
            # For now, just take the first top_n_labels
            self.best_labels = self.unique_labels[: self.top_n_labels]
            print_log(self, f"Using {self.top_n_labels} labels: {self.best_labels}")
        else:
            self.best_labels = self.unique_labels

        # Ensure all channels have the same label distribution
        label_counts = {}
        for channel, df in self.channel_data.items():
            label_counts[channel] = df["label"].value_counts().to_dict()

        print_log(self, "Label distribution per channel:")
        for channel, counts in label_counts.items():
            print_log(self, f"  {channel}: {counts}")

        # Create template pipelines for each model type
        self._create_template_pipelines()

        return True

    def _create_template_pipelines(self):
        """Create template pipelines for each model type."""
        models = {
            "rf": RandomForestClassifier(
                n_estimators=100,
                max_depth=4,
                random_state=48,
                class_weight="balanced",
            ),
            "svm": SVC(
                kernel="rbf",
                C=1.0,
                gamma="scale",
                probability=True,
                random_state=42,
                class_weight="balanced",
            ),
            "logreg": LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=0.5,
                max_iter=5000,
                random_state=42,
                class_weight="balanced",
            ),
            "lda": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            "et": ExtraTreesClassifier(
                n_estimators=100,
                max_depth=4,
                random_state=48,
                class_weight="balanced",
            ),
            "hgb": HistGradientBoostingClassifier(
                max_depth=3, random_state=48, class_weight="balanced"
            ),
            "gnb": GaussianNB(),
            "knn": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        }

        for model_code, model in models.items():
            self.template_pipelines[model_code] = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "feature_selection",
                        SelectKBest(f_classif, k=self.n_features_to_select),
                    ),
                    ("model", model),
                ]
            )

    def _define_search_spaces(self):
        """
        Define hyperparameter search spaces for each model type.

        Returns:
            dict: Model name to search space mapping
        """
        search_spaces = {
            # Random Forest
            "rf": {
                "feature_selection__k": Integer(3, 20),
                "model__n_estimators": Integer(50, 150),
                "model__max_depth": Integer(2, 6),
                "model__min_samples_split": Integer(2, 6),
                "model__min_samples_leaf": Integer(1, 4),
            },
            # SVM
            "svm": {
                "feature_selection__k": Integer(3, 20),
                "model__C": Real(1e-2, 10.0, prior="log-uniform"),
                "model__gamma": Real(1e-4, 1.0, prior="log-uniform"),
                "model__kernel": Categorical(["linear", "rbf"]),
            },
            # Elastic Net Logistic Regression
            "logreg": {
                "feature_selection__k": Integer(3, 20),
                "model__C": Real(1e-2, 10.0, prior="log-uniform"),
                "model__l1_ratio": Real(0.1, 0.9),
            },
            # Shrinkage LDA
            "lda": {
                "feature_selection__k": Integer(3, 20),
                "model__shrinkage": Categorical(["auto", 0.1, 0.3, 0.5, 0.7, 0.9]),
            },
            # Extra Trees
            "et": {
                "feature_selection__k": Integer(3, 20),
                "model__n_estimators": Integer(50, 200),
                "model__max_depth": Integer(2, 8),
                "model__min_samples_leaf": Integer(1, 6),
            },
            # Histogram Gradient Boosting Classifier
            "hgb": {
                "feature_selection__k": Integer(3, 20),
                "model__learning_rate": Real(0.01, 0.2, prior="log-uniform"),
                "model__max_depth": Integer(2, 6),
                "model__max_iter": Integer(50, 150),
            },
            # k-Nearest Neighbors
            "knn": {
                "feature_selection__k": Integer(3, 20),
                "model__n_neighbors": Integer(3, 15),
                "model__weights": Categorical(["uniform", "distance"]),
            },
            # Gaussian Naive Bayes
            "gnb": {
                "feature_selection__k": Integer(3, 20)
                # No hyper-parameters for GNB
            },
        }
        return search_spaces

    def _get_model_name(self, model_code):
        """
        Convert short model code to full name.

        Args:
            model_code (str): Short code for model (rf, svm, etc.)

        Returns:
            str: Full model name
        """
        model_map = {
            "rf": "RandomForest",
            "svm": "SVM",
            "logreg": "ElasticNetLogReg",
            "lda": "ShrinkageLDA",
            "et": "ExtraTrees",
            "hgb": "HGBClassifier",
            "knn": "kNN",
            "gnb": "GaussianNB",
        }
        return model_map.get(model_code, model_code)

    def _get_cv_splitter(self):
        """
        Create appropriate CV splitters based on CV method.

        Returns:
            object: CV splitter object for the current CV method
        """
        if self.cv_method == "loso":
            return LeaveOneGroupOut()
        elif self.cv_method == "lmoso":
            return LeavePGroupsOut(self.lmoso_leftout)
        elif self.cv_method == "kfold":
            return StratifiedGroupKFold(
                n_splits=self.kfold_splits, shuffle=True, random_state=self.random_state
            )
        else:
            raise ValueError(f"Unsupported CV method: {self.cv_method}")

    def _tune_and_fit_pipeline(
        self, channel, model_code, X_train, y_train, groups_train=None, n_iter=25
    ):
        """
        Tune hyper‑parameters for <channel,model> and return a *calibrated*
        probability‑producing pipeline.

        Args:
            channel (str): Channel name
            model_code (str): Model code (rf, svm, etc.)
            X_train (DataFrame): Training features
            y_train (Series): Training labels
            groups_train (array, optional): Group labels for inner CV
            n_iter (int): Number of iterations for Bayesian optimization

        Returns:
            Pipeline: Tuned and fitted pipeline
        """

        # ---------- 1. Get a working pipeline (with or without Bayes‑opt) ----------
        if not SKOPT_AVAILABLE or model_code not in self._define_search_spaces():
            # fallback: use template as‑is
            best_pipe = clone(self.template_pipelines[model_code]).fit(X_train, y_train)
        else:
            # -------- inner CV splitter  --------
            if groups_train is not None:
                # choose a group‑aware inner splitter
                inner_cv = (
                    GroupShuffleSplit(
                        n_splits=3, test_size=0.2, random_state=self.random_state
                    )
                    if self.cv_method in ("loso", "lmoso")
                    else StratifiedGroupKFold(
                        n_splits=3, shuffle=True, random_state=self.random_state
                    )
                )
            else:
                inner_cv = StratifiedKFold(
                    n_splits=3, shuffle=True, random_state=self.random_state
                )

            # -------- BayesSearch --------
            base_pipe = clone(self.template_pipelines[model_code])
            opt = BayesSearchCV(
                base_pipe,
                self._define_search_spaces()[model_code],
                n_iter=n_iter,
                cv=inner_cv,
                scoring="f1_macro",
                n_jobs=min(4, os.cpu_count()),
                random_state=self.random_state,
                verbose=0,
            )
            opt.fit(X_train, y_train, groups=groups_train)
            best_pipe = opt.best_estimator_
            best_params = opt.best_params_
            best_score = opt.best_score_
            print_log(
                self,
                f"Best params {channel}({model_code}): {best_params} | "
                f"inner‑CV F1={best_score:.3f}",
            )

            # persist parameters
            with open(
                f"{self.run_directory}/{channel}_{model_code}_best_params.json",
                "w",
                encoding="utf-8",
            ) as f:
                json.dump(
                    {
                        k: (str(v) if not isinstance(v, (int, float, str)) else v)
                        for k, v in best_params.items()
                    },
                    f,
                    indent=2,
                )

        # ---------- 2.  CALIBRATE probabilities  ----------
        # many scikit models are over‑confident; use 3‑fold Platt scaling
        calibrate_ok = isinstance(
            best_pipe.named_steps["model"],
            (
                RandomForestClassifier,
                ExtraTreesClassifier,
                HistGradientBoostingClassifier,
                KNeighborsClassifier,
                LinearDiscriminantAnalysis,
                GaussianNB,
                LogisticRegression,
                SVC,  # when probability=True
            ),
        )

        if calibrate_ok:
            best_pipe = CalibratedClassifierCV(
                best_pipe, method="sigmoid", cv=3  # Platt scaling
            )
            best_pipe.fit(X_train, y_train)  # *** fit calibrator ***

        # ---------- 3.  return calibrated pipeline ----------
        return best_pipe

    def run_loso_evaluation(self, n_iter=25, online_fusion=True):
        """
        Run LOSO evaluation, training models for each fold and evaluating the supreme model.

        Args:
            n_iter (int): Number of iterations for hyperparameter optimization
            online_fusion (bool): Whether to compute ensemble prediction during training

        Returns:
            dict: Complete evaluation results
        """
        print_log(self, "---- RUNNING LOSO EVALUATION ----")

        ref_channel = next(iter(self.channel_data.keys()))
        all_sessions = np.unique(self.channel_data[ref_channel]["session"])
        print_log(self, f"Found {len(all_sessions)} unique sessions: {all_sessions}")

        for channel, df in self.channel_data.items():
            channel_sessions = np.unique(df["session"])
            if not np.array_equal(np.sort(all_sessions), np.sort(channel_sessions)):
                print_log(
                    self,
                    f"Warning: Channel {channel} has different sessions: {channel_sessions}",
                )
                print_log(self, "Continuing with sessions common to all channels...")
                all_sessions = np.intersect1d(all_sessions, channel_sessions)

        fold_results = []
        channel_metrics = {
            channel: {"accuracy": [], "f1": []} for channel in self.channels_models
        }
        supreme_metrics = {"accuracy": [], "f1": []}

        # --- NEW: for sample-weighted metrics ---
        all_supreme_true = []
        all_supreme_pred = []
        all_channel_true = {ch: [] for ch in self.channels_models}
        all_channel_pred = {ch: [] for ch in self.channels_models}

        cv_splitter = self._get_cv_splitter()

        for session_idx, test_session in enumerate(all_sessions):
            print_log(
                self,
                f"\n---- Fold {session_idx + 1}/{len(all_sessions)} (Test Session: {test_session}) ----",
            )

            fold_preds = {}
            fold_truths = None

            for channel, config in self.channels_models.items():
                model_code = config["model"]
                model_name = self._get_model_name(model_code)

                df = self.channel_data[channel]
                df_subset = df[df["label"].isin(self.best_labels)]

                train_mask = df_subset["session"] != test_session
                test_mask = df_subset["session"] == test_session

                X_train = df_subset.loc[train_mask, self.feature_columns[channel]]
                y_train = df_subset.loc[train_mask, "label"]
                groups_train = df_subset.loc[train_mask, "session"].values

                X_test = df_subset.loc[test_mask, self.feature_columns[channel]]
                y_test = df_subset.loc[test_mask, "label"]

                if fold_truths is None:
                    fold_truths = y_test.values

                best_pipe = self._tune_and_fit_pipeline(
                    channel, model_code, X_train, y_train, groups_train, n_iter
                )

                y_pred = best_pipe.predict(X_test)
                y_proba = best_pipe.predict_proba(X_test)

                acc = accuracy_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred, average="macro")

                channel_metrics[channel]["accuracy"].append(acc)
                channel_metrics[channel]["f1"].append(f1)

                print_log(self, f"{channel} ({model_name}): Acc={acc:.4f}, F1={f1:.4f}")

                fold_preds[channel] = {
                    "proba": y_proba,
                    "classes": best_pipe.classes_,
                    "pred": y_pred,
                }

                pred_data = {
                    "y_true": y_test.values.tolist(),
                    "y_pred": y_pred.tolist(),
                    "y_proba": y_proba.tolist(),
                    "classes": best_pipe.classes_.tolist(),
                }
                with open(
                    f"{self.fold_predictions_dir}/{channel}_{test_session}_preds.json",
                    "w",
                ) as f:
                    json.dump(pred_data, f)

                # --- NEW: accumulate per-sample predictions ---
                all_channel_true[channel].extend(y_test.values)
                all_channel_pred[channel].extend(y_pred)

            if online_fusion:
                if session_idx > 0:
                    weights = self._compute_channel_weights(
                        channel_metrics, session_idx
                    )
                else:
                    weights = {}
                    for channel, config in self.channels_models.items():
                        acc = np.clip(config["accuracy"], 0.51, 0.99)
                        weights[channel] = np.log(acc / (1 - acc))
                    total = sum(abs(w) for w in weights.values())
                    weights = {ch: w / total for ch, w in weights.items()}

                supreme_pred = self._weighted_voting(
                    fold_preds, weights, len(fold_truths)
                )

                supreme_acc = accuracy_score(fold_truths, supreme_pred)
                supreme_f1 = f1_score(fold_truths, supreme_pred, average="macro")

                supreme_metrics["accuracy"].append(supreme_acc)
                supreme_metrics["f1"].append(supreme_f1)

                # --- NEW: accumulate sample-level truth/pred ---
                all_supreme_true.extend(fold_truths)
                all_supreme_pred.extend(supreme_pred)

                print_log(
                    self, f"Supreme model: Acc={supreme_acc:.4f}, F1={supreme_f1:.4f}"
                )

                fold_results.append(
                    {
                        "fold": session_idx + 1,
                        "test_session": test_session,
                        "channel_metrics": {
                            ch: {
                                "accuracy": channel_metrics[ch]["accuracy"][
                                    session_idx
                                ],
                                "f1": channel_metrics[ch]["f1"][session_idx],
                            }
                            for ch in self.channels_models
                        },
                        "supreme_accuracy": supreme_acc,
                        "supreme_f1": supreme_f1,
                        "weights": weights,
                    }
                )

        overall_results = {
            "channel_metrics": {
                channel: {
                    "mean_accuracy": np.mean(metrics["accuracy"]),
                    "std_accuracy": np.std(metrics["accuracy"]),
                    "mean_f1": np.mean(metrics["f1"]),
                    "std_f1": np.std(metrics["f1"]),
                    "per_fold_accuracy": metrics["accuracy"],
                    "per_fold_f1": metrics["f1"],
                }
                for channel, metrics in channel_metrics.items()
            },
            "supreme_metrics": {
                "mean_accuracy": np.mean(supreme_metrics["accuracy"]),
                "std_accuracy": np.std(supreme_metrics["accuracy"]),
                "mean_f1": np.mean(supreme_metrics["f1"]),
                "std_f1": np.std(supreme_metrics["f1"]),
                "per_fold_accuracy": supreme_metrics["accuracy"],
                "per_fold_f1": supreme_metrics["f1"],
            },
            "fold_results": fold_results,
        }

        # --- NEW: Add sample-weighted metrics ---
        sample_weighted = {
            "supreme_accuracy": accuracy_score(all_supreme_true, all_supreme_pred),
            "supreme_f1": f1_score(all_supreme_true, all_supreme_pred, average="macro"),
            "channels": {},
        }
        for ch in self.channels_models:
            acc = accuracy_score(all_channel_true[ch], all_channel_pred[ch])
            f1 = f1_score(all_channel_true[ch], all_channel_pred[ch], average="macro")
            sample_weighted["channels"][ch] = {"accuracy": acc, "f1": f1}
        overall_results["sample_weighted"] = sample_weighted

        # --- Logging ---
        print_log(self, "\n---- OVERALL RESULTS ----")
        print_log(self, "Session-averaged channel performance:")
        for channel, metrics in overall_results["channel_metrics"].items():
            model_name = self._get_model_name(self.channels_models[channel]["model"])
            print_log(
                self,
                f"  {channel} ({model_name}): Acc={metrics['mean_accuracy']:.4f}±{metrics['std_accuracy']:.4f}, "
                f"F1={metrics['mean_f1']:.4f}±{metrics['std_f1']:.4f}",
            )
        sup = overall_results["supreme_metrics"]
        print_log(
            self,
            f"Supreme (session avg): Acc={sup['mean_accuracy']:.4f}±{sup['std_accuracy']:.4f}, "
            f"F1={sup['mean_f1']:.4f}±{sup['std_f1']:.4f}",
        )

        # --- Sample-weighted metrics logging ---
        print_log(self, "\n---- SAMPLE-WEIGHTED RESULTS ----")
        print_log(
            self,
            f"Supreme (SW): Acc={sample_weighted['supreme_accuracy']:.4f}, F1={sample_weighted['supreme_f1']:.4f}",
        )
        for ch, m in sample_weighted["channels"].items():
            print_log(self, f"{ch} (SW): Acc={m['accuracy']:.4f}, F1={m['f1']:.4f}")

        # Save results to disk
        with open(f"{self.run_directory}/evaluation_results.json", "w") as f:

            def ensure_list(v):
                return list(v) if isinstance(v, (np.ndarray, pd.Series)) else v

            results_json = overall_results.copy()
            for channel in results_json["channel_metrics"]:
                results_json["channel_metrics"][channel]["per_fold_accuracy"] = (
                    ensure_list(
                        results_json["channel_metrics"][channel]["per_fold_accuracy"]
                    )
                )
                results_json["channel_metrics"][channel]["per_fold_f1"] = ensure_list(
                    results_json["channel_metrics"][channel]["per_fold_f1"]
                )
            results_json["supreme_metrics"]["per_fold_accuracy"] = ensure_list(
                results_json["supreme_metrics"]["per_fold_accuracy"]
            )
            results_json["supreme_metrics"]["per_fold_f1"] = ensure_list(
                results_json["supreme_metrics"]["per_fold_f1"]
            )
            json.dump(results_json, f, indent=2)

        fold_df = pd.DataFrame(fold_results)
        fold_df.to_csv(f"{self.run_directory}/fold_results.csv", index=False)

        self.evaluation_results = overall_results
        return overall_results

    def _compute_channel_weights(self, channel_metrics, current_fold_idx):
        """
        Compute log-odds-based channel weights using performance in previous folds.

        Args:
            channel_metrics (dict): Dictionary of channel metrics
            current_fold_idx (int): Current fold index

        Returns:
            dict: Channel weights normalized to sum of absolute weights
        """
        weights = {}
        for channel, metrics in channel_metrics.items():
            if current_fold_idx > 0:
                acc = np.mean(metrics["accuracy"][:current_fold_idx])
            else:
                acc = self.channels_models[channel]["accuracy"]

            # Clip accuracy to avoid log(0) or log(inf)
            acc = np.clip(acc, 0.51, 0.99)
            weights[channel] = np.log(acc / (1 - acc))  # log-odds

        # Normalize weights to sum of absolute value 1
        total = sum(abs(w) for w in weights.values())
        if total == 0:
            # Fallback: equal weights if all weights are 0 (shouldn't happen)
            weights = {ch: 1.0 / len(weights) for ch in weights}
        else:
            weights = {ch: w / total for ch, w in weights.items()}

        return weights

    def _weighted_voting(self, fold_preds: dict, global_w: dict, n_samples: int):
        """Confidence‑aware dynamic fusion."""
        sup_pred = []

        # pre‑compute softmax normaliser if requested
        def _norm_conf(conf_dict):
            if self.conf_scale == "softmax":
                z = np.exp(list(conf_dict.values()))
                return {k: v / np.sum(z) for k, v in zip(conf_dict, z)}
            else:  # min‑max
                lo, hi = min(conf_dict.values()), max(conf_dict.values())
                rng = hi - lo or 1.0
                return {k: (v - lo) / rng for k, v in conf_dict.items()}

        for i in range(n_samples):

            # 1) raw confidence from each channel = max‑probability for that sample
            raw_conf = {
                ch: float(pred_pack["proba"][i].max())
                for ch, pred_pack in fold_preds.items()
            }
            loc_conf = _norm_conf(raw_conf)

            # 2) final weight = α·local + (1‑α)·global
            weights_i = {
                ch: self.alpha_conf * loc_conf + (1 - self.alpha_conf) * global_w[ch]
                for ch, loc_conf in loc_conf.items()
            }

            # 3) aggregate probabilities
            agg_probs = {lbl: 0.0 for lbl in self.best_labels}
            for ch, pred_pack in fold_preds.items():
                for cls_idx, cls_val in enumerate(pred_pack["classes"]):
                    agg_probs[cls_val] += weights_i[ch] * pred_pack["proba"][i, cls_idx]

            sup_pred.append(max(agg_probs, key=agg_probs.get))

        return sup_pred

    def permutation_test(self, n_perm=None):
        if n_perm is None:
            n_perm = self.permu_count
        if not hasattr(self, "evaluation_results"):
            print_log(self, "Run run_loso_evaluation first.")
            return {}

        print_log(self, f"---- PERMUTATION TEST (n={n_perm}) ----")

        # observed metrics
        obs_acc = self.evaluation_results["supreme_metrics"]["mean_accuracy"]
        obs_f1 = self.evaluation_results["supreme_metrics"]["mean_f1"]

        ref_channel = next(iter(self.channel_data.keys()))
        all_sessions = np.unique(self.channel_data[ref_channel]["session"])
        rng = np.random.RandomState(self.random_state)

        null_acc = np.zeros(n_perm)
        null_f1 = np.zeros(n_perm)

        for p in tqdm(range(n_perm), desc="Permutation"):
            perm_fold_acc, perm_fold_f1 = [], []

            # ---- outer LOSO loop (exactly like evaluation) ----
            for test_session in all_sessions:
                fold_preds = {}
                perm_true = None  # permuted ground‑truth for this fold

                for channel, cfg in self.channels_models.items():
                    model_code = cfg["model"]
                    df = self.channel_data[channel][
                        self.channel_data[channel]["label"].isin(self.best_labels)
                    ]

                    train_mask = df["session"] != test_session
                    test_mask = df["session"] == test_session

                    X_train = df.loc[train_mask, self.feature_columns[channel]]
                    y_train = df.loc[train_mask, "label"].values.copy()
                    X_test = df.loc[test_mask, self.feature_columns[channel]]
                    y_test = df.loc[test_mask, "label"].values

                    # ----- PERMUTE training labels only -----
                    rng.shuffle(y_train)

                    # fit model on permuted training set
                    pipe = self._tune_and_fit_pipeline(
                        channel,
                        model_code,
                        X_train,
                        y_train,
                        groups_train=df.loc[train_mask, "session"].values,
                        n_iter=5,  # keep search very small – speed!
                    )

                    y_proba = pipe.predict_proba(X_test)
                    y_pred = pipe.predict(X_test)

                    if perm_true is None:
                        perm_true = y_test  # unchanged!  Only predictions are random.

                    fold_preds[channel] = {
                        "proba": y_proba,
                        "classes": pipe.classes_,
                    }

                # equal weights removes any training‑set bias
                w = {ch: 1.0 / len(self.channels_models) for ch in self.channels_models}
                perm_sup = self._weighted_voting(fold_preds, w, len(perm_true))

                perm_fold_acc.append(accuracy_score(perm_true, perm_sup))
                perm_fold_f1.append(f1_score(perm_true, perm_sup, average="macro"))

            null_acc[p] = np.mean(perm_fold_acc)
            null_f1[p] = np.mean(perm_fold_f1)

        # ----- p‑values & summary -----
        p_acc = (np.sum(null_acc >= obs_acc) + 1) / (n_perm + 1)
        p_f1 = (np.sum(null_f1 >= obs_f1) + 1) / (n_perm + 1)

        crit95_acc = np.percentile(null_acc, 95)
        crit95_f1 = np.percentile(null_f1, 95)

        print_log(self, f"Permutation p‑values  acc={p_acc:.4f}  f1={p_f1:.4f}")

        return {
            "observed_acc": obs_acc,
            "observed_f1": obs_f1,
            "p_value_acc": p_acc,
            "p_value_f1": p_f1,
            "critical_acc_95": crit95_acc,
            "critical_f1_95": crit95_f1,
        }

    def bootstrap_confidence_intervals(self, n_boot=5000):
        """
        Calculate bootstrap confidence intervals for model performance.

        Args:
            n_boot (int): Number of bootstrap samples

        Returns:
            dict: Bootstrap results
        """
        if not hasattr(self, "evaluation_results"):
            print_log(
                self, "No evaluation results available. Run run_loso_evaluation first."
            )
            return {}

        print_log(self, f"---- BOOTSTRAP CONFIDENCE INTERVALS (n={n_boot}) ----")

        # Get actual performance metrics for all channels and supreme model
        channel_acc = {
            channel: metrics["per_fold_accuracy"]
            for channel, metrics in self.evaluation_results["channel_metrics"].items()
        }
        channel_f1 = {
            channel: metrics["per_fold_f1"]
            for channel, metrics in self.evaluation_results["channel_metrics"].items()
        }

        supreme_acc = self.evaluation_results["supreme_metrics"]["per_fold_accuracy"]
        supreme_f1 = self.evaluation_results["supreme_metrics"]["per_fold_f1"]

        # Set up random state
        rng = np.random.RandomState(self.random_state)

        # Number of folds
        n_folds = len(supreme_acc)

        # Prepare bootstrap results
        bootstrap_results = {
            "supreme": {"acc": [], "f1": []},
            "channels": {
                channel: {"acc": [], "f1": []} for channel in self.channels_models
            },
        }

        # Perform bootstrap resampling
        for _ in tqdm(range(n_boot), desc="Bootstrap"):
            # Sample with replacement
            boot_indices = rng.choice(n_folds, size=n_folds, replace=True)

            # Supreme model bootstrap
            bootstrap_results["supreme"]["acc"].append(
                np.mean(np.array(supreme_acc)[boot_indices])
            )
            bootstrap_results["supreme"]["f1"].append(
                np.mean(np.array(supreme_f1)[boot_indices])
            )

            # Channel models bootstrap
            for channel in self.channels_models:
                bootstrap_results["channels"][channel]["acc"].append(
                    np.mean(np.array(channel_acc[channel])[boot_indices])
                )
                bootstrap_results["channels"][channel]["f1"].append(
                    np.mean(np.array(channel_f1[channel])[boot_indices])
                )

        # Calculate confidence intervals
        ci_results = {
            "supreme": {
                "acc_mean": np.mean(bootstrap_results["supreme"]["acc"]),
                "acc_ci_low": np.percentile(bootstrap_results["supreme"]["acc"], 2.5),
                "acc_ci_high": np.percentile(bootstrap_results["supreme"]["acc"], 97.5),
                "f1_mean": np.mean(bootstrap_results["supreme"]["f1"]),
                "f1_ci_low": np.percentile(bootstrap_results["supreme"]["f1"], 2.5),
                "f1_ci_high": np.percentile(bootstrap_results["supreme"]["f1"], 97.5),
            },
            "channels": {},
        }

        for channel in self.channels_models:
            ci_results["channels"][channel] = {
                "acc_mean": np.mean(bootstrap_results["channels"][channel]["acc"]),
                "acc_ci_low": np.percentile(
                    bootstrap_results["channels"][channel]["acc"], 2.5
                ),
                "acc_ci_high": np.percentile(
                    bootstrap_results["channels"][channel]["acc"], 97.5
                ),
                "f1_mean": np.mean(bootstrap_results["channels"][channel]["f1"]),
                "f1_ci_low": np.percentile(
                    bootstrap_results["channels"][channel]["f1"], 2.5
                ),
                "f1_ci_high": np.percentile(
                    bootstrap_results["channels"][channel]["f1"], 97.5
                ),
            }

        # Print results
        print_log(self, "Bootstrap 95% confidence intervals:")
        print_log(
            self,
            f"Supreme model - Accuracy: {ci_results['supreme']['acc_mean']:.4f} [{ci_results['supreme']['acc_ci_low']:.4f}, {ci_results['supreme']['acc_ci_high']:.4f}]",
        )
        print_log(
            self,
            f"Supreme model - F1 Score: {ci_results['supreme']['f1_mean']:.4f} [{ci_results['supreme']['f1_ci_low']:.4f}, {ci_results['supreme']['f1_ci_high']:.4f}]",
        )

        for channel in self.channels_models:
            model_name = self._get_model_name(self.channels_models[channel]["model"])
            ch_ci = ci_results["channels"][channel]
            print_log(
                self,
                f"{channel} ({model_name}) - Accuracy: {ch_ci['acc_mean']:.4f} [{ch_ci['acc_ci_low']:.4f}, {ch_ci['acc_ci_high']:.4f}]",
            )
            print_log(
                self,
                f"{channel} ({model_name}) - F1 Score: {ch_ci['f1_mean']:.4f} [{ch_ci['f1_ci_low']:.4f}, {ch_ci['f1_ci_high']:.4f}]",
            )

        # Create visualization
        self._plot_bootstrap_results(bootstrap_results, ci_results)

        return ci_results

    def _plot_bootstrap_results(self, bootstrap_results, ci_results):
        """
        Plot bootstrap distributions and confidence intervals.

        Args:
            bootstrap_results (dict): Raw bootstrap results
            ci_results (dict): Confidence interval results
        """
        # Plot supreme model results
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(
            bootstrap_results["supreme"]["acc"], bins=30, alpha=0.7, color="skyblue"
        )
        plt.axvline(
            ci_results["supreme"]["acc_mean"],
            color="red",
            lw=2,
            label=f'Mean: {ci_results["supreme"]["acc_mean"]:.4f}',
        )
        plt.axvline(ci_results["supreme"]["acc_ci_low"], color="red", ls=":", lw=2)
        plt.axvline(
            ci_results["supreme"]["acc_ci_high"],
            color="red",
            ls=":",
            lw=2,
            label=f'95% CI: [{ci_results["supreme"]["acc_ci_low"]:.4f}, {ci_results["supreme"]["acc_ci_high"]:.4f}]',
        )
        plt.title("Supreme Model - Accuracy Bootstrap")
        plt.xlabel("Accuracy")
        plt.ylabel("Frequency")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.hist(bootstrap_results["supreme"]["f1"], bins=30, alpha=0.7, color="salmon")
        plt.axvline(
            ci_results["supreme"]["f1_mean"],
            color="red",
            lw=2,
            label=f'Mean: {ci_results["supreme"]["f1_mean"]:.4f}',
        )
        plt.axvline(ci_results["supreme"]["f1_ci_low"], color="red", ls=":", lw=2)
        plt.axvline(
            ci_results["supreme"]["f1_ci_high"],
            color="red",
            ls=":",
            lw=2,
            label=f'95% CI: [{ci_results["supreme"]["f1_ci_low"]:.4f}, {ci_results["supreme"]["f1_ci_high"]:.4f}]',
        )
        plt.title("Supreme Model - F1 Score Bootstrap")
        plt.xlabel("F1 Score")
        plt.ylabel("Frequency")
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"{self.run_directory}/bootstrap_supreme.png", dpi=300)
        plt.close()

        # Plot comparative confidence intervals
        n_channels = len(self.channels_models)
        channel_names = list(self.channels_models.keys())

        # Accuracy comparison
        plt.figure(figsize=(12, 6))

        acc_means = [ci_results["channels"][ch]["acc_mean"] for ch in channel_names]
        acc_ci_low = [ci_results["channels"][ch]["acc_ci_low"] for ch in channel_names]
        acc_ci_high = [
            ci_results["channels"][ch]["acc_ci_high"] for ch in channel_names
        ]
        acc_errors = np.array(
            [
                np.array(acc_means) - np.array(acc_ci_low),
                np.array(acc_ci_high) - np.array(acc_means),
            ]
        )

        # Add supreme model
        channel_names.append("Supreme")
        acc_means.append(ci_results["supreme"]["acc_mean"])
        acc_errors = np.hstack(
            [
                acc_errors,
                np.array(
                    [
                        [
                            ci_results["supreme"]["acc_mean"]
                            - ci_results["supreme"]["acc_ci_low"]
                        ],
                        [
                            ci_results["supreme"]["acc_ci_high"]
                            - ci_results["supreme"]["acc_mean"]
                        ],
                    ]
                ),
            ]
        )

        # Plot with error bars
        plt.figure(figsize=(12, 6))
        x_pos = np.arange(len(channel_names))
        plt.bar(
            x_pos,
            acc_means,
            yerr=acc_errors,
            align="center",
            alpha=0.8,
            color=["skyblue"] * n_channels + ["green"],
        )
        plt.xticks(x_pos, channel_names)
        plt.ylabel("Accuracy")
        plt.title("Model Accuracy with 95% Confidence Intervals")
        plt.grid(axis="y", linestyle="--", alpha=0.3)

        # Annotate with values
        for i, v in enumerate(acc_means):
            plt.text(i, v + 0.02, f"{v:.3f}", ha="center")

        plt.tight_layout()
        plt.savefig(f"{self.run_directory}/model_accuracy_comparison.png", dpi=300)
        plt.close()

        # F1 Score comparison (similar to above)
        plt.figure(figsize=(12, 6))

        f1_means = [ci_results["channels"][ch]["f1_mean"] for ch in channel_names[:-1]]
        f1_ci_low = [
            ci_results["channels"][ch]["f1_ci_low"] for ch in channel_names[:-1]
        ]
        f1_ci_high = [
            ci_results["channels"][ch]["f1_ci_high"] for ch in channel_names[:-1]
        ]
        f1_errors = np.array(
            [
                np.array(f1_means) - np.array(f1_ci_low),
                np.array(f1_ci_high) - np.array(f1_means),
            ]
        )

        # Add supreme model
        f1_means.append(ci_results["supreme"]["f1_mean"])
        f1_errors = np.hstack(
            [
                f1_errors,
                np.array(
                    [
                        [
                            ci_results["supreme"]["f1_mean"]
                            - ci_results["supreme"]["f1_ci_low"]
                        ],
                        [
                            ci_results["supreme"]["f1_ci_high"]
                            - ci_results["supreme"]["f1_mean"]
                        ],
                    ]
                ),
            ]
        )

        # Plot with error bars
        plt.figure(figsize=(12, 6))
        x_pos = np.arange(len(channel_names))
        plt.bar(
            x_pos,
            f1_means,
            yerr=f1_errors,
            align="center",
            alpha=0.8,
            color=["salmon"] * n_channels + ["green"],
        )
        plt.xticks(x_pos, channel_names)
        plt.ylabel("F1 Score")
        plt.title("Model F1 Score with 95% Confidence Intervals")
        plt.grid(axis="y", linestyle="--", alpha=0.3)

        # Annotate with values
        for i, v in enumerate(f1_means):
            plt.text(i, v + 0.02, f"{v:.3f}", ha="center")

        plt.tight_layout()
        plt.savefig(f"{self.run_directory}/model_f1_comparison.png", dpi=300)
        plt.close()

    def plot_confusion_matrices(self):
        """
        Plot confusion matrices for each channel and the supreme model.
        """
        if not hasattr(self, "evaluation_results"):
            print_log(
                self, "No evaluation results available. Run run_loso_evaluation first."
            )
            return

        print_log(self, "---- PLOTTING CONFUSION MATRICES ----")

        # Load fold predictions for each channel and the supreme model
        # For each session (fold)
        ref_channel = next(iter(self.channel_data.keys()))
        all_sessions = np.unique(self.channel_data[ref_channel]["session"])

        # Initialize confusion matrices
        cm_channels = {
            channel: np.zeros((len(self.best_labels), len(self.best_labels)))
            for channel in self.channels_models
        }
        cm_supreme = np.zeros((len(self.best_labels), len(self.best_labels)))

        # Collect predictions across all folds
        all_true = []
        all_pred_supreme = []
        all_pred_channels = {channel: [] for channel in self.channels_models}

        # For each session (fold)
        for test_session in all_sessions:
            # Load predictions for each channel
            for channel in self.channels_models:
                pred_file = (
                    f"{self.fold_predictions_dir}/{channel}_{test_session}_preds.json"
                )

                if not os.path.exists(pred_file):
                    print_log(
                        self,
                        f"Warning: Prediction file {pred_file} not found. Skipping.",
                    )
                    continue

                with open(pred_file, "r") as f:
                    pred_data = json.load(f)

                # Get true and predicted labels
                y_true = np.array(pred_data["y_true"])
                y_pred = np.array(pred_data["y_pred"])

                # Accumulate for channel confusion matrix
                all_true.extend(y_true)
                all_pred_channels[channel].extend(y_pred)

                # Update channel confusion matrix
                cm_channels[channel] += confusion_matrix(
                    y_true, y_pred, labels=self.best_labels
                )

        # Compute supreme confusion matrix
        # Note: This requires either recomputing supreme predictions or loading them
        # For simplicity, we'll use the fold results and assume supreme predictions are saved
        # In practice, you should save supreme predictions during run_loso_evaluation

        # Plot channel confusion matrices
        n_channels = len(self.channels_models)
        n_cols = min(3, n_channels)
        n_rows = (n_channels + n_cols - 1) // n_cols

        fig, ax_arr = plt.subplots(n_rows, n_cols, figsize=(n_cols * 5, n_rows * 4))
        ax_list: list[plt.Axes] = (
            [ax_arr] if not isinstance(ax_arr, np.ndarray) else ax_arr.ravel().tolist()
        )

        for i, (channel, cm) in enumerate(cm_channels.items()):
            ax = ax_list[i]
            sns.heatmap(
                cm,
                annot=True,
                fmt=".3f",
                cmap="Blues",
                xticklabels=self.best_labels,
                yticklabels=self.best_labels,
                ax=ax,
            )
            ax.set_xlabel("Predicted")
            ax.set_ylabel("True")
            model_name = self._get_model_name(self.channels_models[channel]["model"])
            ax.set_title(f"{channel} ({model_name}) Confusion Matrix")

        # hide unused axes
        for j in range(len(cm_channels), len(ax_list)):
            fig.delaxes(ax_list[j])

        plt.tight_layout()
        plt.savefig(f"{self.run_directory}/channel_confusion_matrices.png", dpi=300)
        plt.close()

    def plot_per_sample_agreement(self):
        """
        Plot per-sample agreement across channels.
        Shows which samples all channels agree on and which have disagreement.
        """
        if not hasattr(self, "evaluation_results"):
            print_log(
                self, "No evaluation results available. Run run_loso_evaluation first."
            )
            return

        print_log(self, "---- PLOTTING PER-SAMPLE AGREEMENT ----")

        # Get all sessions
        ref_channel = next(iter(self.channel_data.keys()))
        all_sessions = np.unique(self.channel_data[ref_channel]["session"])

        # Prepare data for heatmap
        sample_data = []

        # For each session (fold)
        for test_session in all_sessions:
            # Load predictions for each channel
            fold_preds = {}
            fold_true = None

            for channel in self.channels_models:
                pred_file = (
                    f"{self.fold_predictions_dir}/{channel}_{test_session}_preds.json"
                )

                if not os.path.exists(pred_file):
                    print_log(
                        self,
                        f"Warning: Prediction file {pred_file} not found. Skipping.",
                    )
                    continue

                with open(pred_file, "r") as f:
                    pred_data = json.load(f)

                # Get true and predicted labels
                if fold_true is None:
                    fold_true = np.array(pred_data["y_true"])

                fold_preds[channel] = np.array(pred_data["y_pred"])

            # For each sample in this fold
            for i in range(len(fold_true)):
                sample_row = {
                    "session": test_session,
                    "true_label": fold_true[i],
                    "sample_idx": i,
                }

                # Get predictions from each channel
                for channel in fold_preds:
                    sample_row[f"{channel}_pred"] = fold_preds[channel][i]

                # Count correct and agreement
                correct_count = sum(
                    1
                    for channel in fold_preds
                    if fold_preds[channel][i] == fold_true[i]
                )

                # Check if all channels predict the same (agreement)
                unique_preds = set(fold_preds[channel][i] for channel in fold_preds)
                agreement = len(unique_preds) == 1

                sample_row["correct_count"] = correct_count
                sample_row["total_channels"] = len(fold_preds)
                sample_row["agreement"] = agreement
                sample_row["agreement_label"] = (
                    next(iter(unique_preds)) if agreement else "mixed"
                )

                sample_data.append(sample_row)

        # Convert to DataFrame
        samples_df = pd.DataFrame(sample_data)

        # Save to CSV for detailed analysis
        samples_df.to_csv(f"{self.run_directory}/per_sample_agreement.csv", index=False)

        # Create heatmap for visualization
        # Reshape data for heatmap: samples vs. channels with correct/incorrect predictions
        channels = list(self.channels_models.keys())

        agreement_counts = samples_df["agreement"].value_counts()
        agreement_percent = (agreement_counts.get(True, 0) / len(samples_df)) * 100

        correct_by_channel = {}
        for channel in channels:
            mask = samples_df[f"{channel}_pred"] == samples_df["true_label"]
            correct_by_channel[channel] = mask.mean() * 100

        # Plot agreement statistics
        plt.figure(figsize=(12, 6))

        plt.subplot(1, 2, 1)
        plt.bar(
            ["Agreement", "Disagreement"],
            [agreement_counts.get(True, 0), agreement_counts.get(False, 0)],
            color=["green", "red"],
        )
        plt.title(f"Channel Agreement: {agreement_percent:.1f}% of samples")
        plt.ylabel("Number of Samples")

        plt.subplot(1, 2, 2)
        plt.bar(channels, [correct_by_channel[ch] for ch in channels], color="skyblue")
        plt.title("Accuracy by Channel")
        plt.ylabel("Accuracy (%)")
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.savefig(f"{self.run_directory}/channel_agreement_stats.png", dpi=300)
        plt.close()

        # For samples with disagreement, visualize which channels got it right
        disagreement_samples = samples_df[~samples_df["agreement"]].copy()

        if len(disagreement_samples) > 0:
            # Limit to 100 samples for visualization clarity
            if len(disagreement_samples) > 100:
                disagreement_samples = disagreement_samples.sample(
                    100, random_state=self.random_state
                )

            # Create heatmap data
            heatmap_data = np.zeros((len(disagreement_samples), len(channels)))

            for i, (_, row) in enumerate(disagreement_samples.iterrows()):
                for j, channel in enumerate(channels):
                    # 1 if correct, 0 if incorrect
                    heatmap_data[i, j] = int(
                        row[f"{channel}_pred"] == row["true_label"]
                    )

            plt.figure(figsize=(10, max(8, len(disagreement_samples) * 0.2)))
            sns.heatmap(
                heatmap_data,
                cmap=["red", "green"],
                cbar=False,
                xticklabels=channels,
                yticklabels=False,
            )
            plt.title("Channels Correctness on Disagreement Samples")
            plt.xlabel("Channel")
            plt.ylabel("Sample")
            plt.tight_layout()
            plt.savefig(
                f"{self.run_directory}/disagreement_samples_heatmap.png", dpi=300
            )
            plt.close()
        self.agg_perc = agreement_percent

    def run_complete_analysis(self, n_iter=25):
        """
        Run the complete supreme model analysis pipeline.

        Args:
            n_iter (int): Number of iterations for hyperparameter optimization

        Returns:
            dict: Complete analysis results
        """
        print_log(self, "---- STARTING COMPLETE ANALYSIS ----")

        # Step 1: Load data
        self.load_data()

        # Step 2: Run LOSO evaluation
        evaluation_results = self.run_loso_evaluation(n_iter=n_iter)

        # Step 3: Statistical evaluation
        permutation_results = self.permutation_test()
        bootstrap_results = self.bootstrap_confidence_intervals()

        # Step 4: Visualization
        self.plot_confusion_matrices()
        self.plot_per_sample_agreement()

        # Step 5: Generate final report
        self._generate_final_report(
            evaluation_results, permutation_results, bootstrap_results
        )

        print_log(self, "---- ANALYSIS COMPLETE ----")
        print_log(self, f"Results saved to {self.run_directory}")

        return {
            "evaluation_results": evaluation_results,
            "permutation_results": permutation_results,
            "bootstrap_results": bootstrap_results,
        }

    def _generate_final_report(
        self, evaluation_results, permutation_results, bootstrap_results
    ):
        """
        Generate a comprehensive final report of the analysis.

        Args:
            evaluation_results (dict): Results from LOSO evaluation
            permutation_results (dict): Results from permutation testing
            bootstrap_results (dict): Results from bootstrap confidence intervals
        """
        # Create a markdown report
        report = f"""# Supreme Model Analysis Report

        ## Overview
        - **Analysis Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}
        - **Channels & Models:** {', '.join([f"{ch} ({self._get_model_name(cfg['model'])})" for ch, cfg in self.channels_models.items()])}
        - **Cross-Validation Method:** {self.cv_method.upper()}
        - **Labels:** {', '.join(self.best_labels)}
        - **Features Selected per Model:** {self.n_features_to_select}
        - **Random Seed:** {self.random_state}
        
        ## Supreme Model Performance
        - **Accuracy:** {evaluation_results['supreme_metrics']['mean_accuracy']:.4f} ± {evaluation_results['supreme_metrics']['std_accuracy']:.4f}
        - **F1 Score:** {evaluation_results['supreme_metrics']['mean_f1']:.4f} ± {evaluation_results['supreme_metrics']['std_f1']:.4f}
        - **Bootstrap 95% CI (Accuracy):** [{bootstrap_results['supreme']['acc_ci_low']:.4f}, {bootstrap_results['supreme']['acc_ci_high']:.4f}]
        - **Bootstrap 95% CI (F1):** [{bootstrap_results['supreme']['f1_ci_low']:.4f}, {bootstrap_results['supreme']['f1_ci_high']:.4f}]
        - **Permutation Test p-value (Accuracy):** {permutation_results['p_value_acc']:.4f}
        - **Permutation Test p-value (F1):** {permutation_results['p_value_f1']:.4f}
        
        ## Individual Channel Performance
        
        | Channel | Model | Accuracy | F1 Score | Accuracy 95% CI | F1 Score 95% CI |
        |---------|-------|----------|----------|----------------|-----------------|
        """

        # Add channel performance rows
        for channel in self.channels_models:
            model_name = self._get_model_name(self.channels_models[channel]["model"])
            metrics = evaluation_results["channel_metrics"][channel]
            ch_ci = bootstrap_results["channels"][channel]

            report += f"| {channel} | {model_name} | {metrics['mean_accuracy']:.4f} ± {metrics['std_accuracy']:.4f} | {metrics['mean_f1']:.4f} ± {metrics['std_f1']:.4f} | [{ch_ci['acc_ci_low']:.4f}, {ch_ci['acc_ci_high']:.4f}] | [{ch_ci['f1_ci_low']:.4f}, {ch_ci['f1_ci_high']:.4f}] |\n"

        # Add fold-by-fold performance
        report += f"""
## Fold-by-Fold Performance

| Fold | Test Session | Supreme Accuracy | Supreme F1 |
|------|--------------|-----------------|------------|
"""

        for fold in evaluation_results["fold_results"]:
            report += f"| {fold['fold']} | {fold['test_session']} | {fold['supreme_accuracy']:.4f} | {fold['supreme_f1']:.4f} |\n"

        # Statistical significance section
        report += f"""
## Statistical Significance

The permutation test was performed by shuffling labels randomly and repeating the analysis {self.permu_count} times.

- **Observed Accuracy:** {permutation_results['observed_acc']:.4f}
- **Null Distribution 95th Percentile (Accuracy):** {permutation_results['critical_acc_95']:.4f}
- **p-value (Accuracy):** {permutation_results['p_value_acc']:.4f}

- **Observed F1 Score:** {permutation_results['observed_f1']:.4f}
- **Null Distribution 95th Percentile (F1):** {permutation_results['critical_f1_95']:.4f}
- **p-value (F1):** {permutation_results['p_value_f1']:.4f}

The supreme model's performance is{'' if permutation_results['p_value_acc'] < 0.05 else ' not'} statistically significant at α=0.05 for accuracy and{'' if permutation_results['p_value_f1'] < 0.05 else ' not'} for F1 score.

## Interpretation and Conclusion

This analysis demonstrates that combining predictions from multiple channel-specific models using a weighted voting approach{"" if evaluation_results['supreme_metrics']['mean_accuracy'] > max([metrics['mean_accuracy'] for metrics in evaluation_results['channel_metrics'].values()]) else " does not"} improve overall classification performance compared to the best single-channel model.

The supreme model achieved an accuracy of {evaluation_results['supreme_metrics']['mean_accuracy']:.4f}, which is {evaluation_results['supreme_metrics']['mean_accuracy'] - max([metrics['mean_accuracy'] for metrics in evaluation_results['channel_metrics'].values()]):.4f} points {"higher" if evaluation_results['supreme_metrics']['mean_accuracy'] > max([metrics['mean_accuracy'] for metrics in evaluation_results['channel_metrics'].values()]) else "lower"} than the best individual channel model ({max(evaluation_results['channel_metrics'], key=lambda ch: evaluation_results['channel_metrics'][ch]['mean_accuracy'])}).

Key findings:
1. Channel model agreement: On {self.agg_perc:.1f}% of samples, all channels produced the same prediction.
2. The supreme model leverages information from multiple channels to {"improve" if evaluation_results['supreme_metrics']['mean_accuracy'] > max([metrics['mean_accuracy'] for metrics in evaluation_results['channel_metrics'].values()]) else "maintain"} classification performance.
3. Statistical significance testing confirms that the observed performance {"is" if permutation_results['p_value_acc'] < 0.05 else "is not"} significantly better than chance.

## Next Steps

- Fine-tune channel weighting strategies to further optimize supreme model performance
- Investigate per-sample agreement patterns to identify challenging classification cases
- Consider additional channel sources for potential performance improvement
- Explore alternative ensemble methods beyond weighted voting

"""

        # Save report to markdown file
        with open(f"{self.run_directory}/final_report.md", "w", encoding="utf-8") as f:
            f.write(report)

        print_log(self, f"Final report saved to {self.run_directory}/final_report.md")


def print_log(trainer, message):
    """
    Print log message and save to log file.

    Args:
        trainer (SupremeTrainer): The trainer instance
        message (str): Message to log
    """
    print(message)
    with open(f"{trainer.run_directory}/log.txt", "a", encoding="utf-8") as f:
        f.write(message + "\n")


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Supreme EEG Model Trainer")

    parser.add_argument(
        "--channels_config",
        type=str,
        help="JSON file with channel:model:accuracy mapping",
        default=None,
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
        default=15,
        help="Number of top features to select (default: 15)",
    )
    parser.add_argument(
        "--cv_method",
        type=str,
        default="loso",
        choices=["loso", "lmoso", "kfold"],
        help="Cross-validation method (default: loso)",
    )
    parser.add_argument(
        "--kfold_splits",
        type=int,
        default=5,
        help="Number of splits for k-fold cross-validation (default: 5)",
    )
    parser.add_argument(
        "--lmoso_leftout",
        type=int,
        default=2,
        help="Number of sessions to leave out for LMOSO (default: 2)",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=25,
        help="Number of iterations for hyperparameter optimization (default: 25)",
    )
    parser.add_argument(
        "--permu_count",
        type=int,
        default=1000,
        help="Number of permutations for significance testing (default: 1000)",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/final_sets/all_channels_binary",
        help="Path to directory containing channel data files",
    )
    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Either load channels configuration from JSON file or use a default config
    if args.channels_config:
        with open(args.channels_config, "r") as f:
            channels_models = json.load(f)
    else:
        # TODO see if you can make use of model's 'strong guesses', if the models are strongly
        #  sure (for example for RF, it can say that the information gain is very high),
        #  then we give them additional bonus points on their decision, if they are simply not
        #  sure of their decision maybe we just make use of democracy then, or just trust the highest performing model,
        #  it might also be that the models are all strong in similar manners, but its just their data at hand is
        #  more or less difficult to deal with.
        # Example default configuration
        channels_models = {
            "o2_comBat": {"model": "rf", "accuracy": 0.718},
            # "po4_comBat": {"model": "knn", "accuracy": 0.613},
            "tp8_comBat": {"model": "rf", "accuracy": 0.682},
            "oz_comBat": {"model": "knn", "accuracy": 0.621},
        }

    # Create and run the supreme trainer
    trainer = SupremeTrainer(
        channels_models=channels_models,
        top_n_labels=args.top_n_labels,
        n_features_to_select=args.n_features,
        cv_method=args.cv_method,
        kfold_splits=args.kfold_splits,
        lmoso_leftout=args.lmoso_leftout,
        permu_count=args.permu_count,
        data_path=args.data_path,
        random_state=args.random_state,
    )

    # Run the complete analysis
    trainer.run_complete_analysis(n_iter=args.n_iter)
