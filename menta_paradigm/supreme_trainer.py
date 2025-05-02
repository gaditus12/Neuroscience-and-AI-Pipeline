import json
import os
import time
import argparse
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import (
    LeaveOneGroupOut,
    LeavePGroupsOut,
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
        cv_method="loso",  # Options: "loso", "lmoso", "kfold"
        kfold_splits=5,
        lmoso_leftout=2,
        permu_count=1000,
        data_path="data/final_sets/all_channels_binary",
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

        # Initialize data structures for storing results
        self.channel_data = {}  # Will hold DataFrames for each channel
        self.channel_models = {}  # Will hold trained models for each channel
        self.best_labels = None  # Will store the best label combination
        self.unique_labels = None  # Will store all unique labels from the data
        self.feature_columns = {}  # Will store feature columns for each channel
        self.fold_predictions = {}  # Will store predictions for each fold
        self.supreme_results = {}  # Will store combined results

        # Log initialization
        with open(f"{self.run_directory}/config.txt", "w") as f:
            f.write(f"Channels and models: {channels_models}\n")
            f.write(f"CV method: {cv_method}\n")
            f.write(f"Top N labels: {top_n_labels}\n")
            f.write(f"Features to select: {n_features_to_select}\n")
            f.write(f"Permutation count: {permu_count}\n")

        # Print welcome message
        print_log(self, "---- SUPREME TRAINER INITIALIZED ----")
        print_log(self, f"Output directory: {self.run_directory}")
        print_log(self, f"Channels and models: {channels_models}")
        print_log(self, f"CV method: {cv_method}")

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

        # Ensure all channels have the same label distribution
        label_counts = {}
        for channel, df in self.channel_data.items():
            label_counts[channel] = df["label"].value_counts().to_dict()

        print_log(self, "Label distribution per channel:")
        for channel, counts in label_counts.items():
            print_log(self, f"  {channel}: {counts}")

        return True

    def _define_search_spaces(self):
        """
        Define hyperparameter search spaces for each model type.

        Returns:
            dict: Model name to search space mapping
        """
        search_spaces = {
            # Random Forest
            "rf": {
                "model": Categorical(
                    [RandomForestClassifier(random_state=42, n_jobs=-1)]
                ),
                "model__n_estimators": Integer(50, 150),
                "model__max_depth": Categorical([4]),
                "model__min_samples_split": Integer(2, 6),
                "model__min_samples_leaf": Integer(2, 8),
                "model__class_weight": Categorical(["balanced"]),
                "feature_selection__k": Integer(3, 20),
            },
            # SVM
            "svm": {
                "model": Categorical([SVC(random_state=42, probability=True)]),
                "model__kernel": Categorical(["linear", "rbf", "poly"]),
                "model__C": Real(1e-2, 5.0, prior="log-uniform"),
                "model__gamma": Real(1e-4, 5e-2, prior="log-uniform"),
                "model__class_weight": Categorical(["balanced", None]),
                "feature_selection__k": Integer(3, 20),
            },
            # Elastic Net Logistic Regression
            "logreg": {
                "model": Categorical(
                    [
                        LogisticRegression(
                            penalty="elasticnet",
                            solver="saga",
                            class_weight="balanced",
                            max_iter=4000,
                            random_state=42,
                        )
                    ]
                ),
                "model__C": Real(1e-2, 5.0, prior="log-uniform"),
                "model__l1_ratio": Real(0.1, 0.9),
                "feature_selection__k": Integer(3, 20),
            },
            # Shrinkage LDA
            "lda": {
                "model": Categorical([LinearDiscriminantAnalysis(solver="lsqr")]),
                "model__shrinkage": Categorical(["auto", 0.1, 0.3, None]),
                "feature_selection__k": Integer(3, 20),
            },
            # Extra Trees
            "et": {
                "model": Categorical(
                    [
                        ExtraTreesClassifier(
                            class_weight="balanced", random_state=42, n_jobs=-1
                        )
                    ]
                ),
                "model__n_estimators": Integer(80, 200),
                "model__max_depth": Integer(2, 8),
                "model__min_samples_leaf": Integer(2, 6),
                "feature_selection__k": Integer(3, 20),
            },
            # Histogram Gradient Boosting Classifier
            "hgb": {
                "model": Categorical(
                    [
                        HistGradientBoostingClassifier(
                            class_weight="balanced", random_state=42
                        )
                    ]
                ),
                "model__learning_rate": Real(0.02, 0.12, prior="log-uniform"),
                "model__max_depth": Integer(2, 4),
                "model__max_iter": Integer(60, 100),
                "feature_selection__k": Integer(3, 20),
            },
            # k-Nearest Neighbors
            "knn": {
                "model": Categorical([KNeighborsClassifier()]),
                "model__n_neighbors": Integer(5, 15),
                "model__weights": Categorical(["uniform", "distance"]),
                "feature_selection__k": Integer(3, 20),
            },
            # Gaussian Naive Bayes
            "gnb": {
                "model": Categorical([GaussianNB()]),
                "feature_selection__k": Integer(3, 20),
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

    def _get_nested_splitters(self, groups):
        """
        Create appropriate CV splitters based on CV method.

        Args:
            groups (array-like): Array of session IDs

        Returns:
            tuple: (outer_cv, inner_cv) splitter objects
        """
        if self.cv_method == "loso":
            outer = LeaveOneGroupOut()
            inner = GroupShuffleSplit(n_splits=3, test_size=0.20, random_state=42)
        elif self.cv_method == "lmoso":
            outer = LeavePGroupsOut(self.lmoso_leftout)
            inner = GroupShuffleSplit(n_splits=3, test_size=0.20, random_state=42)
        elif self.cv_method == "kfold":
            outer = StratifiedGroupKFold(
                n_splits=self.kfold_splits, shuffle=True, random_state=42
            )
            inner = StratifiedGroupKFold(n_splits=3, shuffle=True, random_state=24)
        else:
            raise ValueError(f"Unsupported CV method: {self.cv_method}")

        return outer, inner

    def optimize_channel_models(self, n_iter=25):
        """
        Optimize hyperparameters for each channel-model pair using nested CV.
        Stores best parameters rather than trained models.
        """
        if not SKOPT_AVAILABLE:
            print_log(
                self,
                "Warning: scikit-optimize not available. Using default parameters.",
            )
            return {}

        print_log(self, "---- OPTIMIZING CHANNEL HYPERPARAMETERS ----")
        search_spaces = self._define_search_spaces()
        optimization_results = {}
        self.channel_hyperparams = {}  # Store best parameters instead of models

        # Rest of the function stays similar but instead of storing models:
        for channel, config in self.channels_models.items():
            # Optimization code...

            # Store best parameters instead of models
            self.channel_hyperparams[channel] = opt.best_params_

        return optimization_results

    def evaluate_supreme_model(self):
        """
        Evaluate the combined "supreme" model using weighted voting from individual channel models.

        Returns:
            dict: Supreme model evaluation results
        """
        print_log(self, "---- EVALUATING SUPREME MODEL ----")

        if not self.channel_models:
            print_log(
                self, "No channel models available. Run optimize_channel_models first."
            )
            return {}

        # Get selected labels
        selected_labels = self.best_labels

        # Prepare data for all channels
        channel_data = {}
        for channel in self.channels_models.keys():
            df = self.channel_data[channel]
            df_subset = df[df["label"].isin(selected_labels)]
            channel_data[channel] = {
                "X": df_subset[self.feature_columns[channel]],
                "y": df_subset["label"],
                "groups": df_subset["session"].values,
                "indices": df_subset.index,
            }

        # Ensure all channels have the same samples (by session)
        # This is important to have aligned predictions later
        common_sessions = set.intersection(
            *[set(data["groups"]) for data in channel_data.values()]
        )
        for channel in channel_data:
            mask = np.isin(channel_data[channel]["groups"], list(common_sessions))
            channel_data[channel]["X"] = channel_data[channel]["X"][mask]
            channel_data[channel]["y"] = channel_data[channel]["y"][mask]
            channel_data[channel]["groups"] = channel_data[channel]["groups"][mask]
            channel_data[channel]["indices"] = channel_data[channel]["indices"][mask]

        # Choose a reference channel for CV splitting and true labels
        ref_channel = list(channel_data.keys())[0]
        ref_X = channel_data[ref_channel]["X"]
        ref_y = channel_data[ref_channel]["y"]
        ref_groups = channel_data[ref_channel]["groups"]

        # Get outer CV splitter
        outer_cv, _ = self._get_nested_splitters(ref_groups)

        # Initialize results storage
        all_true = []
        all_pred_supreme = []
        all_pred_channels = {channel: [] for channel in channel_data}
        fold_results = []

        # Run CV evaluation with PROPER RETRAINING
        for fold, (train_idx, test_idx) in enumerate(
            outer_cv.split(ref_X, ref_y, ref_groups), start=1
        ):
            fold_true = ref_y.iloc[test_idx].values
            fold_size = len(test_idx)
            fold_session = ref_groups[test_idx[0]]

            # For each channel, TRAIN AND PREDICT for this fold (NO REUSE)
            fold_preds = {}
            for channel, config in self.channels_models.items():
                model_code = config["model"]

                # Get training data for this fold
                X_train = channel_data[channel]["X"].iloc[train_idx]
                y_train = channel_data[channel]["y"].iloc[train_idx]

                # Get test data for this fold
                X_test = channel_data[channel]["X"].iloc[test_idx]

                # Create and train a NEW pipeline for this fold
                pipe = self._create_and_fit_pipeline(model_code, X_train, y_train)

                # Predict with the freshly trained pipeline
                y_proba = pipe.predict_proba(X_test)
                y_pred = pipe.predict(X_test)

                fold_preds[channel] = {
                    "proba": y_proba,
                    "pred": y_pred,
                    "classes": pipe.classes_,  # Add this line to store the classes
                }

                # Store predictions for this channel
                all_pred_channels[channel].extend(y_pred)

            # Combine predictions using weighted voting
            supreme_pred = self._weighted_voting(fold_preds, fold_true.shape[0])

            # Store results
            all_true.extend(fold_true)
            all_pred_supreme.extend(supreme_pred)

            # Calculate fold metrics
            fold_acc = accuracy_score(fold_true, supreme_pred)
            fold_f1 = f1_score(fold_true, supreme_pred, average="macro")

            fold_results.append(
                {
                    "fold": fold,
                    "session": fold_session,
                    "n_samples": fold_size,
                    "accuracy": fold_acc,
                    "f1_macro": fold_f1,
                }
            )

            print_log(
                self,
                f"Fold {fold} (Session {fold_session}): Accuracy = {fold_acc:.3f}, F1 = {fold_f1:.3f}",
            )

        # Calculate overall metrics
        supreme_acc = accuracy_score(all_true, all_pred_supreme)
        supreme_f1 = f1_score(all_true, all_pred_supreme, average="macro")
        supreme_cm = confusion_matrix(all_true, all_pred_supreme)

        # Calculate individual channel metrics
        channel_metrics = {}
        for channel in all_pred_channels:
            channel_acc = accuracy_score(all_true, all_pred_channels[channel])
            channel_f1 = f1_score(all_true, all_pred_channels[channel], average="macro")
            channel_cm = confusion_matrix(all_true, all_pred_channels[channel])

            channel_metrics[channel] = {
                "accuracy": channel_acc,
                "f1_macro": channel_f1,
                "confusion_matrix": channel_cm,
            }

            print_log(
                self,
                f"{channel} metrics: Accuracy = {channel_acc:.3f}, F1 = {channel_f1:.3f}",
            )

        # Store results
        supreme_results = {
            "accuracy": supreme_acc,
            "f1_macro": supreme_f1,
            "confusion_matrix": supreme_cm,
            "fold_results": fold_results,
            "channel_metrics": channel_metrics,
        }

        print_log(
            self,
            f"Supreme model overall: Accuracy = {supreme_acc:.3f}, F1 = {supreme_f1:.3f}",
        )

        # Save results
        fold_df = pd.DataFrame(fold_results)
        fold_df.to_csv(f"{self.run_directory}/supreme_fold_results.csv", index=False)

        # Save channel metrics
        channel_metrics_rows = []
        for channel, metrics in channel_metrics.items():
            row = {
                "channel": channel,
                "model": self.channels_models[channel]["model"],
                "weight": self.channels_models[channel]["accuracy"],
                "accuracy": metrics["accuracy"],
                "f1_macro": metrics["f1_macro"],
            }
            channel_metrics_rows.append(row)

        channel_df = pd.DataFrame(channel_metrics_rows)
        channel_df.to_csv(f"{self.run_directory}/channel_metrics.csv", index=False)

        self.supreme_results = supreme_results

        return supreme_results

    def _create_and_fit_pipeline(self, model_code, X_train, y_train):
        """
        Create and fit a pipeline for the specified model type.
        This ensures a fresh model for each fold with no data leakage.
        """
        # Define model based on code
        models = {
            "rf": RandomForestClassifier(
                n_estimators=100, max_depth=4, random_state=42, class_weight="balanced"
            ),
            "svm": SVC(
                kernel="rbf",
                C=1.0,
                probability=True,
                random_state=42,
                class_weight="balanced",
            ),
            "hgb": HistGradientBoostingClassifier(
                max_depth=3, random_state=42, class_weight="balanced"
            ),
            "et": ExtraTreesClassifier(
                n_estimators=100, max_depth=4, random_state=42, class_weight="balanced"
            ),
            "logreg": LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=0.5,
                random_state=42,
                class_weight="balanced",
            ),
            "lda": LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            "gnb": GaussianNB(),
            "knn": KNeighborsClassifier(n_neighbors=7, weights="distance"),
        }

        # Create pipeline
        pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "feature_selection",
                    SelectKBest(f_classif, k=self.n_features_to_select),
                ),
                ("model", models[model_code]),
            ]
        )

        # Fit pipeline
        pipe.fit(X_train, y_train)

        return pipe

    def _weighted_voting(self, fold_preds, n_samples):
        """
        Combine predictions from multiple channel models using weighted voting.
        """
        # Initialize storage for supreme predictions
        supreme_pred = []

        # Normalize weights
        weights = np.array(
            [config["accuracy"] for _, config in self.channels_models.items()]
        )
        channel_names = list(self.channels_models.keys())
        normalized_weights = weights / weights.sum()

        # Ensure consistent class ordering using best_labels
        class_to_idx = {c: i for i, c in enumerate(self.best_labels)}

        # Get class probabilities for each sample
        for i in range(n_samples):
            weighted_probs = {
                label: 0.0 for label in self.best_labels
            }  # Initialize with all possible labels

            # Accumulate weighted probabilities for each class
            for idx, channel in enumerate(channel_names):
                weight = normalized_weights[idx]
                proba = fold_preds[channel]["proba"][i]
                classes = fold_preds[channel]["classes"]  # Get stored classes

                # Add weighted probabilities for each class
                for cls_idx, cls_val in enumerate(classes):
                    weighted_probs[cls_val] += proba[cls_idx] * weight

            # Choose class with highest weighted probability
            best_class = max(weighted_probs, key=weighted_probs.get)
            supreme_pred.append(best_class)

        return supreme_pred

    def permutation_test_supreme(self, n_perm=None):
        """
        Perform permutation testing to assess statistical significance.

        Args:
            n_perm (int, optional): Number of permutations. If None, uses self.permu_count.

        Returns:
            dict: Permutation test results
        """
        if n_perm is None:
            n_perm = self.permu_count

        print_log(self, f"---- PERMUTATION TEST (n={n_perm}) ----")

        if not hasattr(self, "supreme_results") or not self.supreme_results:
            print_log(
                self, "No supreme results available. Run evaluate_supreme_model first."
            )
            return {}

        # Get selected labels
        selected_labels = self.best_labels

        # Prepare data for all channels
        channel_data = {}
        for channel in self.channels_models.keys():
            df = self.channel_data[channel]
            df_subset = df[df["label"].isin(selected_labels)]
            channel_data[channel] = {
                "X": df_subset[self.feature_columns[channel]],
                "y": df_subset["label"],
                "groups": df_subset["session"].values,
                "indices": df_subset.index,
            }

        # Ensure all channels have the same samples (by session)
        common_sessions = set.intersection(
            *[set(data["groups"]) for data in channel_data.values()]
        )
        for channel in channel_data:
            mask = np.isin(channel_data[channel]["groups"], list(common_sessions))
            channel_data[channel]["X"] = channel_data[channel]["X"][mask]
            channel_data[channel]["y"] = channel_data[channel]["y"][mask]
            channel_data[channel]["groups"] = channel_data[channel]["groups"][mask]
            channel_data[channel]["indices"] = channel_data[channel]["indices"][mask]

        # Choose a reference channel for CV splitting
        ref_channel = list(channel_data.keys())[0]
        ref_X = channel_data[ref_channel]["X"]
        ref_y = channel_data[ref_channel]["y"]
        ref_groups = channel_data[ref_channel]["groups"]

        # Get outer CV splitter
        outer_cv, _ = self._get_nested_splitters(ref_groups)

        # Calculate observed scores
        # These should match what we calculated in evaluate_supreme_model
        obs_fold_f1 = [
            fold["f1_macro"] for fold in self.supreme_results["fold_results"]
        ]
        obs_fold_acc = [
            fold["accuracy"] for fold in self.supreme_results["fold_results"]
        ]
        obs_mean_f1 = np.mean(obs_fold_f1)
        obs_mean_acc = np.mean(obs_fold_acc)

        print_log(
            self, f"Observed mean F1: {obs_mean_f1:.3f}, Accuracy: {obs_mean_acc:.3f}"
        )

        # Set up arrays to store null distribution
        null_mean_f1 = np.empty(n_perm)
        null_mean_acc = np.empty(n_perm)

        # Perform permutations
        # Perform permutations
        rng = np.random.RandomState(42)
        for i in tqdm(range(n_perm), desc="Permutation test"):
            # Permute labels
            y_perm = rng.permutation(ref_y.values)

            # Store permutation fold scores
            perm_fold_f1 = []
            perm_fold_acc = []

            # Run CV with permuted labels - IMPORTANT: Retrain for each fold
            for fold_idx, (train_idx, test_idx) in enumerate(
                outer_cv.split(ref_X, ref_y, ref_groups)
            ):
                fold_true_perm = y_perm[test_idx]

                # For each channel, get predictions for this fold WITH NEW TRAINING
                fold_preds = {}
                for channel, config in self.channels_models.items():
                    model_code = config["model"]

                    # Get training data with permuted labels
                    X_train = channel_data[channel]["X"].iloc[train_idx]
                    y_train_perm = y_perm[train_idx]  # Use permuted labels for training

                    # Get test data
                    X_test = channel_data[channel]["X"].iloc[test_idx]

                    # Create and train new pipeline with permuted labels
                    pipe = self._create_and_fit_pipeline(
                        model_code, X_train, y_train_perm
                    )

                    # Predict
                    y_proba = pipe.predict_proba(X_test)
                    y_pred = pipe.predict(X_test)

                    fold_preds[channel] = {
                        "proba": y_proba,
                        "pred": y_pred,
                    }

                # Combine predictions using weighted voting
                supreme_pred = self._weighted_voting(
                    fold_preds, fold_true_perm.shape[0]
                )

                # Calculate metrics
                fold_acc = accuracy_score(fold_true_perm, supreme_pred)
                fold_f1 = f1_score(fold_true_perm, supreme_pred, average="macro")

                perm_fold_f1.append(fold_f1)
                perm_fold_acc.append(fold_acc)

            # Calculate mean scores for this permutation
            null_mean_f1[i] = np.mean(perm_fold_f1)
            null_mean_acc[i] = np.mean(perm_fold_acc)

            # Calculate p-values (proportion of permutation statistics >= observed statistic)
        p_f1 = (np.sum(null_mean_f1 >= obs_mean_f1) + 1) / (n_perm + 1)
        p_acc = (np.sum(null_mean_acc >= obs_mean_acc) + 1) / (n_perm + 1)

        # Calculate critical values (95th percentile of null distribution)
        crit95_f1 = float(np.percentile(null_mean_f1, 95))
        crit95_acc = float(np.percentile(null_mean_acc, 95))

        # Calculate median of null distribution
        crit50_f1 = float(np.percentile(null_mean_f1, 50))
        crit50_acc = float(np.percentile(null_mean_acc, 50))

        print_log(
            self, f"Permutation p-values → F1: {p_f1:.4f} | Accuracy: {p_acc:.4f}"
        )
        print_log(
            self,
            f"Critical values (α=0.05) → F1: {crit95_f1:.3f} | Accuracy: {crit95_acc:.3f}",
        )

        # Save null distribution thresholds for plotting
        np.save(
            os.path.join(self.run_directory, "null_95th_percentile_f1.npy"), crit95_f1
        )
        np.save(
            os.path.join(self.run_directory, "null_50th_percentile_f1.npy"), crit50_f1
        )
        np.save(
            os.path.join(self.run_directory, "null_95th_percentile_acc.npy"), crit95_acc
        )
        np.save(
            os.path.join(self.run_directory, "null_50th_percentile_acc.npy"), crit50_acc
        )

        # Create histogram visualization
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.hist(null_mean_f1, bins=30, alpha=0.7)
        plt.axvline(obs_mean_f1, color="red", lw=2, label=f"Observed {obs_mean_f1:.3f}")
        plt.axvline(crit95_f1, color="black", ls=":", lw=2, label=f"95% null")
        plt.axvline(crit50_f1, color="grey", ls=":", lw=2, label="Median null")
        plt.title(f"Macro-F1 null distribution\np={p_f1:.4f}")
        plt.xlabel("Mean F1")
        plt.ylabel("Count")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.hist(null_mean_acc, bins=30, alpha=0.7, color="mediumaquamarine")
        plt.axvline(
            obs_mean_acc, color="red", lw=2, label=f"Observed {obs_mean_acc:.3f}"
        )
        plt.axvline(crit95_acc, color="black", ls=":", lw=2, label="95% null")
        plt.axvline(crit50_acc, color="grey", ls=":", lw=2, label="Median null")
        plt.title(f"Accuracy null distribution\np={p_acc:.4f}")
        plt.xlabel("Mean Accuracy")
        plt.legend()

        plt.tight_layout()
        plt.savefig(
            f"{self.run_directory}/permutation_histogram.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Return results
        permutation_results = {
            "observed_f1": obs_mean_f1,
            "observed_acc": obs_mean_acc,
            "p_value_f1": p_f1,
            "p_value_acc": p_acc,
            "critical_f1_95": crit95_f1,
            "critical_acc_95": crit95_acc,
            "null_median_f1": crit50_f1,
            "null_median_acc": crit50_acc,
        }

        return permutation_results

    def plot_weighted_feature_importance(self):
        """
        Plot feature importance across all channels, weighted by channel accuracy.
        """
        print_log(self, "---- PLOTTING WEIGHTED FEATURE IMPORTANCE ----")

        if not self.channel_models:
            print_log(
                self, "No channel models available. Run optimize_channel_models first."
            )
            return

        # Collect feature importance from each channel
        all_features = set()
        feature_importance = {}

        for channel, model in self.channel_models.items():
            # Get feature selector
            feature_selector = model.named_steps["feature_selection"]
            selected_indices = feature_selector.get_support(indices=True)
            selected_features = [
                self.feature_columns[channel][i] for i in selected_indices
            ]

            # Get importance scores
            if hasattr(feature_selector, "scores_"):
                f_scores = feature_selector.scores_

                # Create importance map for selected features
                for idx, feature in enumerate(selected_features):
                    score = f_scores[selected_indices[idx]]

                    if feature not in feature_importance:
                        feature_importance[feature] = []

                    # Store (channel, score)
                    feature_importance[feature].append((channel, score))
                    all_features.add(feature)

        # Calculate weighted importance
        weighted_importance = {}
        for feature in all_features:
            weighted_sum = 0
            weight_sum = 0

            # Calculate weighted score
            for channel, score in feature_importance.get(feature, []):
                weight = self.channels_models[channel]["accuracy"]
                weighted_sum += score * weight
                weight_sum += weight

            if weight_sum > 0:
                weighted_importance[feature] = weighted_sum / weight_sum

        # Convert to sorted DataFrame
        importance_df = pd.DataFrame(
            {
                "Feature": list(weighted_importance.keys()),
                "Importance": list(weighted_importance.values()),
            }
        ).sort_values("Importance", ascending=False)

        # Save importance to CSV
        importance_df.to_csv(
            f"{self.run_directory}/weighted_feature_importance.csv", index=False
        )

        # Create visualization
        plt.figure(figsize=(14, 10))
        plt.barh(importance_df["Feature"], importance_df["Importance"], color="skyblue")
        plt.title("Weighted Feature Importance Across Channels", fontsize=16)
        plt.xlabel("Weighted Importance Score", fontsize=12)
        plt.ylabel("Feature", fontsize=12)
        plt.grid(axis="x", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.savefig(
            f"{self.run_directory}/weighted_feature_importance.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        print_log(self, f"Top 10 important features:\n{importance_df.head(10)}")

        return importance_df

    def visualize_confusion_matrices(self):
        """
        Visualize confusion matrices for each channel and the supreme model.
        """
        print_log(self, "---- VISUALIZING CONFUSION MATRICES ----")

        if not hasattr(self, "supreme_results") or not self.supreme_results:
            print_log(
                self, "No supreme results available. Run evaluate_supreme_model first."
            )
            return

        # Create a grid of confusion matrices
        n_channels = len(self.channels_models)
        n_cols = 3
        n_rows = (n_channels + 1) // n_cols + 1  # +1 for supreme model

        plt.figure(figsize=(15, 5 * n_rows))

        # Plot supreme model confusion matrix
        plt.subplot(n_rows, n_cols, 1)
        cm = self.supreme_results["confusion_matrix"]
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
        plt.title("Supreme Model Confusion Matrix")

        # Plot channel confusion matrices
        for i, (channel, metrics) in enumerate(
            self.supreme_results["channel_metrics"].items(), start=2
        ):
            plt.subplot(n_rows, n_cols, i)
            cm = metrics["confusion_matrix"]
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
            plt.title(
                f'{channel} ({self._get_model_name(self.channels_models[channel]["model"])}) Confusion Matrix'
            )

        plt.tight_layout()
        plt.savefig(
            f"{self.run_directory}/confusion_matrices.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def plot_channel_vs_supreme_performance(self):
        """
        Create a bar chart comparing channel performances with supreme model.
        """
        print_log(self, "---- PLOTTING CHANNEL VS SUPREME PERFORMANCE ----")

        if not hasattr(self, "supreme_results") or not self.supreme_results:
            print_log(
                self, "No supreme results available. Run evaluate_supreme_model first."
            )
            return

        # Prepare data for plotting
        channels = list(self.supreme_results["channel_metrics"].keys())
        channel_acc = [
            self.supreme_results["channel_metrics"][ch]["accuracy"] for ch in channels
        ]
        channel_f1 = [
            self.supreme_results["channel_metrics"][ch]["f1_macro"] for ch in channels
        ]
        supreme_acc = self.supreme_results["accuracy"]
        supreme_f1 = self.supreme_results["f1_macro"]

        # Create a DataFrame for easier plotting
        plot_data = pd.DataFrame(
            {
                "Channel": channels + ["Supreme"],
                "Accuracy": channel_acc + [supreme_acc],
                "F1 Score": channel_f1 + [supreme_f1],
                "Type": ["Channel"] * len(channels) + ["Supreme"],
            }
        )

        # Create visualization
        plt.figure(figsize=(12, 6))

        # Set up bar positions
        x = np.arange(len(channels) + 1)
        width = 0.35

        # Plot bars
        plt.bar(
            x - width / 2,
            plot_data["Accuracy"],
            width,
            label="Accuracy",
            color="skyblue",
        )
        plt.bar(
            x + width / 2,
            plot_data["F1 Score"],
            width,
            label="F1 Score",
            color="salmon",
        )

        # Add labels and legend
        plt.xlabel("Channel / Model")
        plt.ylabel("Score")
        plt.title("Channel vs Supreme Model Performance")
        plt.xticks(x, plot_data["Channel"])
        plt.ylim(0, 1.0)
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        plt.legend()

        # Highlight supreme model
        plt.axvspan(len(channels) - 0.5, len(channels) + 0.5, alpha=0.1, color="green")

        # Add value annotations
        for i, value in enumerate(plot_data["Accuracy"]):
            plt.text(
                i - width / 2, value + 0.02, f"{value:.3f}", ha="center", va="bottom"
            )

        for i, value in enumerate(plot_data["F1 Score"]):
            plt.text(
                i + width / 2, value + 0.02, f"{value:.3f}", ha="center", va="bottom"
            )

        plt.tight_layout()
        plt.savefig(
            f"{self.run_directory}/channel_vs_supreme_performance.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def plot_fold_performance(self):
        """
        Plot performance across folds/sessions.
        """
        print_log(self, "---- PLOTTING FOLD PERFORMANCE ----")

        if not hasattr(self, "supreme_results") or not self.supreme_results:
            print_log(
                self, "No supreme results available. Run evaluate_supreme_model first."
            )
            return

        # Get fold results
        fold_df = pd.DataFrame(self.supreme_results["fold_results"])

        # Create visualization
        plt.figure(figsize=(12, 6))

        # Set up bar positions
        x = np.arange(len(fold_df))
        width = 0.35

        # Plot bars
        plt.bar(
            x - width / 2, fold_df["accuracy"], width, label="Accuracy", color="skyblue"
        )
        plt.bar(
            x + width / 2, fold_df["f1_macro"], width, label="F1 Score", color="salmon"
        )

        # Add mean lines
        plt.axhline(
            fold_df["accuracy"].mean(),
            color="skyblue",
            ls="--",
            label=f'Mean Acc: {fold_df["accuracy"].mean():.3f}',
        )
        plt.axhline(
            fold_df["f1_macro"].mean(),
            color="salmon",
            ls="--",
            label=f'Mean F1: {fold_df["f1_macro"].mean():.3f}',
        )

        # Add labels and legend
        plt.xlabel("Fold / Session")
        plt.ylabel("Score")
        plt.title("Supreme Model Performance by Fold/Session")
        plt.xticks(
            x,
            [
                f"{fold} (S{session})"
                for fold, session in zip(fold_df["fold"], fold_df["session"])
            ],
        )
        plt.ylim(0, 1.0)
        plt.grid(axis="y", linestyle="--", alpha=0.3)
        plt.legend()

        # Add value annotations
        for i, value in enumerate(fold_df["accuracy"]):
            plt.text(
                i - width / 2, value + 0.02, f"{value:.3f}", ha="center", va="bottom"
            )

        for i, value in enumerate(fold_df["f1_macro"]):
            plt.text(
                i + width / 2, value + 0.02, f"{value:.3f}", ha="center", va="bottom"
            )

        plt.tight_layout()
        plt.savefig(
            f"{self.run_directory}/fold_performance.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def bootstrap_supreme_confidence(self, n_boot=5000):
        """
        Perform bootstrap resampling to estimate confidence intervals for supreme model.

        Args:
            n_boot (int): Number of bootstrap samples

        Returns:
            dict: Bootstrap results including confidence intervals
        """
        print_log(self, f"---- BOOTSTRAP CONFIDENCE INTERVALS (n={n_boot}) ----")

        if not hasattr(self, "supreme_results") or not self.supreme_results:
            print_log(
                self, "No supreme results available. Run evaluate_supreme_model first."
            )
            return {}

        # Get fold results
        fold_df = pd.DataFrame(self.supreme_results["fold_results"])
        fold_acc = fold_df["accuracy"].values
        fold_f1 = fold_df["f1_macro"].values
        fold_weights = fold_df["n_samples"].values

        # Calculate observed weighted statistics
        obs_acc = np.average(fold_acc, weights=fold_weights)
        obs_f1 = np.average(fold_f1, weights=fold_weights)

        # Perform weighted bootstrap resampling
        rng = np.random.RandomState(42)
        boot_acc = np.zeros(n_boot)
        boot_f1 = np.zeros(n_boot)

        for i in range(n_boot):
            # Sample with replacement
            boot_idx = rng.choice(len(fold_acc), size=len(fold_acc), replace=True)
            boot_acc[i] = np.average(fold_acc[boot_idx], weights=fold_weights[boot_idx])
            boot_f1[i] = np.average(fold_f1[boot_idx], weights=fold_weights[boot_idx])

        # Calculate 95% confidence intervals
        ci_low_acc, ci_high_acc = np.percentile(boot_acc, [2.5, 97.5])
        ci_low_f1, ci_high_f1 = np.percentile(boot_f1, [2.5, 97.5])

        print_log(
            self,
            f"Accuracy: {obs_acc:.3f} [95% CI: {ci_low_acc:.3f}-{ci_high_acc:.3f}]",
        )
        print_log(
            self, f"F1 Score: {obs_f1:.3f} [95% CI: {ci_low_f1:.3f}-{ci_high_f1:.3f}]"
        )

        # Create visualization
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.hist(boot_acc, bins=30, alpha=0.7, color="skyblue")
        plt.axvline(
            obs_acc, color="red", ls="-", lw=2, label=f"Observed: {obs_acc:.3f}"
        )
        plt.axvline(
            ci_low_acc,
            color="red",
            ls=":",
            lw=2,
            label=f"95% CI: [{ci_low_acc:.3f}, {ci_high_acc:.3f}]",
        )
        plt.axvline(ci_high_acc, color="red", ls=":", lw=2)
        plt.title("Bootstrap Distribution - Accuracy")
        plt.xlabel("Accuracy")
        plt.ylabel("Frequency")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.hist(boot_f1, bins=30, alpha=0.7, color="salmon")
        plt.axvline(obs_f1, color="red", ls="-", lw=2, label=f"Observed: {obs_f1:.3f}")
        plt.axvline(
            ci_low_f1,
            color="red",
            ls=":",
            lw=2,
            label=f"95% CI: [{ci_low_f1:.3f}, {ci_high_f1:.3f}]",
        )
        plt.axvline(ci_high_f1, color="red", ls=":", lw=2)
        plt.title("Bootstrap Distribution - F1 Score")
        plt.xlabel("F1 Score")
        plt.ylabel("Frequency")
        plt.legend()

        plt.tight_layout()
        plt.savefig(
            f"{self.run_directory}/bootstrap_confidence_intervals.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # Save bootstrap results
        bootstrap_results = {
            "observed_acc": obs_acc,
            "observed_f1": obs_f1,
            "ci_low_acc": ci_low_acc,
            "ci_high_acc": ci_high_acc,
            "ci_low_f1": ci_low_f1,
            "ci_high_f1": ci_high_f1,
        }

        return bootstrap_results

    def run_complete_analysis(self, n_iter=25):
        """
        Run the complete analysis pipeline.

        Args:
            n_iter (int): Number of iterations for hyperparameter optimization

        Returns:
            dict: Complete analysis results
        """
        print_log(self, "---- STARTING COMPLETE ANALYSIS ----")

        # Step 1: Load data
        self.load_data()

        # Step 2: Optimize each channel model
        optimization_results = self.optimize_channel_models(n_iter=n_iter)

        # Step 3: Evaluate supreme model
        supreme_results = self.evaluate_supreme_model()

        # Step 4: Statistical evaluation and visualization
        permutation_results = self.permutation_test_supreme()
        bootstrap_results = self.bootstrap_supreme_confidence()
        self.plot_weighted_feature_importance()
        self.visualize_confusion_matrices()
        self.plot_channel_vs_supreme_performance()
        self.plot_fold_performance()

        # Step 5: Final report
        print_log(self, "\n---- ANALYSIS COMPLETE ----")
        print_log(self, f"Supreme model accuracy: {supreme_results['accuracy']:.4f}")
        print_log(self, f"Supreme model F1 score: {supreme_results['f1_macro']:.4f}")
        print_log(
            self,
            f"Permutation test p-value (F1): {permutation_results['p_value_f1']:.4f}",
        )
        print_log(
            self,
            f"Bootstrap 95% CI for F1: [{bootstrap_results['ci_low_f1']:.4f}, {bootstrap_results['ci_high_f1']:.4f}]",
        )
        print_log(self, "Individual channel performances:")

        for channel, metrics in supreme_results["channel_metrics"].items():
            model_name = self._get_model_name(self.channels_models[channel]["model"])
            print_log(
                self,
                f"  {channel} ({model_name}): Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}",
            )

        # Save final report
        with open(f"{self.run_directory}/final_report.txt", "w") as f:
            f.write("SUPREME MODEL ANALYSIS REPORT\n")
            f.write("============================\n\n")
            f.write(f"Supreme model accuracy: {supreme_results['accuracy']:.4f}\n")
            f.write(f"Supreme model F1 score: {supreme_results['f1_macro']:.4f}\n")
            f.write(
                f"Permutation test p-value (F1): {permutation_results['p_value_f1']:.4f}\n"
            )
            f.write(
                f"Bootstrap 95% CI for F1: [{bootstrap_results['ci_low_f1']:.4f}, {bootstrap_results['ci_high_f1']:.4f}]\n\n"
            )
            f.write("Individual channel performances:\n")

            for channel, metrics in supreme_results["channel_metrics"].items():
                model_name = self._get_model_name(
                    self.channels_models[channel]["model"]
                )
                f.write(
                    f"  {channel} ({model_name}): Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}\n"
                )

            f.write("\nAnalysis complete.\n")

        # Save complete results
        complete_results = {
            "optimization_results": optimization_results,
            "supreme_results": supreme_results,
            "permutation_results": permutation_results,
            "bootstrap_results": bootstrap_results,
        }

        joblib.dump(complete_results, f"{self.run_directory}/complete_results.pkl")

        print_log(self, f"Results saved to {self.run_directory}")

        return complete_results


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
        required=True,
        help="JSON file with channel:model:accuracy mapping",
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

    args = parser.parse_args()

    # Load channels configuration from JSON file
    with open(args.channels_config, "r") as f:
        channels_models = json.load(f)

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
    )

    # Run the complete analysis
    trainer.run_complete_analysis(n_iter=args.n_iter)
