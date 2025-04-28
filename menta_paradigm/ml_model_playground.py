import os
import time

from itertools import combinations
from tqdm import tqdm

from sklearn.decomposition import PCA
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import matplotlib.patches as patches
import matplotlib.transforms as transforms
from sklearn.model_selection import LeaveOneGroupOut   # NEW
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Add these to your imports at the top of the file
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, cross_val_score, StratifiedKFold, train_test_split
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report, f1_score, \
    accuracy_score
from sklearn.pipeline import Pipeline
# scikit-optimize for Bayesian optimization
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
    SKOPT_AVAILABLE = True
except ImportError:
    SKOPT_AVAILABLE = False
    print("Warning: scikit-optimize not available. Install with 'pip install scikit-optimize' to use hyperparameter optimization.")

from sklearn.model_selection import (
        LeaveOneGroupOut, StratifiedGroupKFold, GroupShuffleSplit,
        StratifiedKFold)
class EEGAnalyzer:
    def __init__(self,
                 features_file,
                 top_n_labels=2,
                 n_features_to_select=15,
                 channel_approach="pooled",  # Options: "pooled", "separate", "features"
                 cv_method="kfold",  # Options: "loo", "kfold", "holdout"
                 cv_version='extended', #extended or simple with a simple feature selection
                 kfold_splits=5,
                 test_size=0.2):  # For holdout validation
        # Configuration
        self.features_file = features_file
        self.top_n_labels = top_n_labels
        self.n_features_to_select = n_features_to_select
        self.channel_approach = channel_approach.lower()
        self.cv_method = cv_method.lower()
        self.cv_version = cv_version.lower()
        self.kfold_splits = kfold_splits
        self.test_size = test_size

        # Create a unique directory for this run
        timestamp = int(time.time())
        self.run_directory = os.path.join("ml_model_outputs", f"run_{timestamp}")
        os.makedirs(self.run_directory, exist_ok=True)

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
        print("---- LOADING DATA ----")
        self.df = pd.read_csv(self.features_file)

        # Ensure there is a 'channel' column (if missing, assume first column is channel)
        if 'channel' not in self.df.columns:
            self.df['channel'] = self.df.iloc[:, 0]

        # Define feature columns (all columns except metadata)
        self.feature_columns = [col for col in self.df.columns if col not in ["label", "channel", "session"]]

        # Get unique labels and channels
        self.unique_labels = self.df["label"].unique()
        self.unique_channels = self.df['channel'].unique() if 'channel' in self.df.columns else ["unknown"]

        print(f"Dataset has {len(self.df)} rows with {len(self.feature_columns)} features")
        print(f"Found {len(self.unique_labels)} unique labels: {self.unique_labels}")
        print(f"Found {len(self.unique_channels)} unique channels: {self.unique_channels}\n")

        # Print sample counts
        print("Sample counts per label:")
        print(self.df.groupby("label").size())
        if 'channel' in self.df.columns:
            print("\nSample counts per label and channel:")
            print(self.df.groupby(["label", "channel"]).size())

        # Handle channels based on approach
        if self.channel_approach == "pooled":
            print("\nUsing POOLED approach: Treating each channel reading as an independent sample")
            self.X = self.df[self.feature_columns]
            self.y = self.df["label"]

        elif self.channel_approach == "separate":
            print("\nUsing SEPARATE approach: Analyzing each channel independently")
            # For now we use the same data – further processing could be added later.
            self.X = self.df[self.feature_columns]
            self.y = self.df["label"]

        else:  # "features" approach
            print("\nUsing FEATURES approach: Combining channels as additional features")
            if 'session' not in self.df.columns:
                print(
                    "Error: This approach requires a 'session' column to identify unique recordings. Falling back to POOLED.")
                self.channel_approach = "pooled"
                self.X = self.df[self.feature_columns]
                self.y = self.df["label"]
            else:
                # Placeholder: For a proper implementation, you would combine rows per session.
                self.X = self.df[self.feature_columns]
                self.y = self.df["label"]

    def preprocess_data(self, debug_plots_only=True):
        print("\n---- PREPROCESSING DATA ----")
        # This method should only perform exploratory analysis or simple data cleaning.
        # The actual standardization will occur in cross-validation.

        # Basic data examination
        print(f"Feature value ranges (min, max):")
        for col in self.feature_columns[:5]:  # Print first few for brevity
            print(f"  {col}: ({self.X[col].min():.4f}, {self.X[col].max():.4f})")

        # Check for missing values
        missing_values = self.X.isnull().sum().sum()
        print(f"Total missing values: {missing_values}")

        if missing_values > 0:
            print("WARNING: Dataset contains missing values which may affect analysis")

        # Note about standardization
        print("\nNOTE: Feature standardization will be performed within each cross-validation fold")
        print("to prevent information leakage between training and test data.")

        # For backward compatibility with the rest of the code, we'll still fit a scaler
        # but with a clear warning that it should not be used for evaluation
        if debug_plots_only:
            self.X_scaled = self.scaler.fit_transform(self.X)
            self.X_scaled_df = pd.DataFrame(self.X_scaled, columns=self.feature_columns)
            print("\nWARNING: Global standardization performed for visualization purposes only.")
            print("DO NOT use self.X_scaled for model evaluation/training.")
    def feature_selection(self):
        """
        Feature selection method - To be called before cross-validation
        NOTE: This method is for informational purposes only. The actual feature selection
        is performed within the cross-validation/holdout methods to avoid information leakage.
        """
        print("\n---- FEATURE SELECTION ANALYSIS (Informational Only) ----")
        print("Note: The actual feature selection is performed within each cross-validation fold")

        # This is only for initial visualization and exploration
        selector = SelectKBest(f_classif, k=self.n_features_to_select)
        selector.fit(self.X_scaled, self.y)
        selected_indices = selector.get_support(indices=True)

        # Store for visualization purposes only
        self.selected_features = [self.feature_columns[i] for i in selected_indices]

        print("Top features based on F-test (for informational purposes only):")
        f_scores = selector.scores_
        p_values = selector.pvalues_

        feature_scores = pd.DataFrame({
            'feature': self.feature_columns,
            'f_score': f_scores,
            'p_value': p_values
        }).sort_values('f_score', ascending=False)

        for i, (_, row) in enumerate(feature_scores.head(self.n_features_to_select).iterrows()):
            print(f"  {i + 1}. {row['feature']} (F-score: {row['f_score']:.4f}, p-value: {row['p_value']:.4f})")

        # Create a dummy selector for compatibility
        self.selector = self.DummySelector(self.feature_columns, self.selected_features)

        print("\nWARNING: This is preliminary analysis only. Proper feature selection")
        print("will be performed within each fold of cross-validation.")

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
            inner = GroupShuffleSplit(n_splits=3,
                                      test_size=0.20,
                                      random_state=42)
        elif self.cv_method == "kfold":
            outer = StratifiedGroupKFold(
                n_splits=self.kfold_splits, shuffle=True, random_state=42)
            inner = StratifiedGroupKFold(
                n_splits=3, shuffle=True, random_state=24)
        elif self.cv_method == "loo":
            outer = LeaveOneGroupOut()
            inner = GroupShuffleSplit(n_splits=3,
                                      test_size=0.20,
                                      random_state=42)
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
        results = {'model_metrics': {}, 'confusion_matrices': {}, 'misclassified_samples': {}}

        X_array = X.values if isinstance(X, pd.DataFrame) else X

        for name, model in models.items():
            y_true_all, y_pred_all, misclassified = [], [], []

            for train_idx, test_idx in cv.split(X_array, y, groups):
                X_tr, X_te = X_array[train_idx], X_array[test_idx]
                y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
                test_indices = y.iloc[test_idx].index.tolist()

                # scale + select on training fold only
                scaler   = StandardScaler().fit(X_tr)
                X_tr_s   = scaler.transform(X_tr)
                X_te_s   = scaler.transform(X_te)

                selector = SelectKBest(f_classif, k=self.n_features_to_select).fit(X_tr_s, y_tr)
                X_tr_sel = selector.transform(X_tr_s)
                X_te_sel = selector.transform(X_te_s)

                model.fit(X_tr_sel, y_tr)
                y_pred = model.predict(X_te_sel)

                y_true_all.extend(y_te.tolist())
                y_pred_all.extend(y_pred)

                for idx, t, p in zip(test_indices, y_te.tolist(), y_pred):
                    if t != p:
                        misclassified.append({'index': idx,
                                              'true_label': t,
                                              'predicted_label': p,
                                              'session_left_out': groups[test_idx[0]]})

            results['model_metrics'][name]        = self._calculate_metrics(y_true_all, y_pred_all)
            results['confusion_matrices'][name]   = confusion_matrix(y_true_all, y_pred_all)
            results['misclassified_samples'][name] = misclassified

            print(f"\n{name} (LOSO) Classification Report:")
            print(classification_report(y_true_all, y_pred_all))

        return results

    def evaluate_label_combinations(self):
        print("\n---- EVALUATING LABEL COMBINATIONS ----")
        # Set up models
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=50, max_depth=2,
                                                   class_weight='balanced', random_state=48, min_samples_leaf=2),
            'SVM': SVC(kernel='rbf', C=0.1, gamma='scale',
                       class_weight='balanced', random_state=42, probability=True),
            # new ✱
            'ElasticNetLogReg': LogisticRegression(
                penalty='elasticnet', solver='saga',
                class_weight='balanced', max_iter=5000, random_state=42, l1_ratio=0.5),

            'ShrinkageLDA': LinearDiscriminantAnalysis(
                solver='lsqr', shrinkage='auto'),

            'ExtraTrees': ExtraTreesClassifier(
                n_estimators=100, max_depth=None, class_weight='balanced',
                random_state=48),

            'HGBClassifier': HistGradientBoostingClassifier(
                learning_rate=0.1, max_depth=3,
                class_weight='balanced', random_state=48),

            'GaussianNB': GaussianNB(),

            'kNN': KNeighborsClassifier(
                n_neighbors=7, weights='distance')
        }

        # Choose CV method based on parameter
        if self.cv_method == "loo":
            print("Using Leave-One-Out cross validation")
            cv = LeaveOneOut()
            if self.cv_version=='simple':
                cv_method=self._evaluate_with_cv
            elif self.cv_version=='extended':
                cv_method=self._evaluate_with_cv_extended
        elif self.cv_method == "kfold":
            print(f"Using Stratified K-Fold cross validation with {self.kfold_splits} splits")
            cv = StratifiedKFold(n_splits=self.kfold_splits)
            if self.cv_version=='simple':
                cv_method=self._evaluate_with_cv
            elif self.cv_version=='extended':
                cv_method=self._evaluate_with_cv_extended
        elif self.cv_method == "holdout":
            print(f"Using holdout validation with test_size={self.test_size}")
            cv = None
            cv_method = self._evaluate_with_holdout
        elif self.cv_method == "loso":
            print("Using Leave‑One‑SESSION‑Out CV")
            cv         = LeaveOneGroupOut()
            # always use the *simple* evaluator, but with groups
            cv_method  = self._evaluate_with_cv_loso


        else:
            raise ValueError("Invalid cv_method provided. Choose 'loo', 'kfold', or 'holdout'.")

        # Iterate over all possible label combinations (pairs or triplets)
        for label_combo in combinations(self.unique_labels, self.top_n_labels):
            df_subset = self.df[self.df["label"].isin(label_combo)]
            X_subset = df_subset[self.feature_columns]
            y_subset = df_subset["label"]

            # Print sample counts for this combination
            combo_sample_count = df_subset.groupby("label").size()
            print(f"\nLabel combination {label_combo}: {len(df_subset)} samples")
            print(f"  Per label: {combo_sample_count.to_dict()}")

            # FIX: Pass raw data to cv_method, which will handle scaling and feature selection properly
            # ---- call the correct evaluator ----
            if self.cv_method == "loso":
                groups = df_subset["session"].values  # 1‑D array of session IDs
                results = cv_method(X_subset, y_subset, models, cv, groups)
            else:
                results = cv_method(X_subset, y_subset, models, cv)

            # Store results
            best_model = max(results['model_metrics'], key=lambda m: results['model_metrics'][m]['f1_macro'])
            best_score = results['model_metrics'][best_model]['accuracy']

            self.separability_scores[label_combo] = best_score
            self.detailed_results[label_combo] = {
                'best_model': best_model,
                'metrics': results['model_metrics'],
                'confusion_matrices': results.get('confusion_matrices', {}),
                'misclassified_samples': results.get('misclassified_samples', {}),
                'n_samples': len(y_subset),
                'sample_counts': combo_sample_count.to_dict()
            }

            # Print detailed results
            print(f"  Best model: {best_model}")
            for metric, value in results['model_metrics'][best_model].items():
                print(f"    {metric}: {value:.4f}")

        # Find best label combination based on F1 score
        self.best_labels = max(self.detailed_results,
                               key=lambda c:
                               self.detailed_results[c]['metrics'][self.detailed_results[c]['best_model']]['f1_macro'])

        print(f"\nThe {self.top_n_labels} most diverse labels are: {self.best_labels}")
        best_model = self.detailed_results[self.best_labels]['best_model']
        print(f"Best model: {best_model}")

        metrics = self.detailed_results[self.best_labels]['metrics'][best_model]
        for metric, value in metrics.items():
            print(f"{metric}: {value:.4f}")

        print(f"Total samples: {self.detailed_results[self.best_labels]['n_samples']}")
    def _evaluate_with_cv(self, X, y, models, cv):
        """Evaluate models with cross-validation - fixed to avoid information leaks"""
        results = {'model_metrics': {}, 'confusion_matrices': {}, 'misclassified_samples': {}}

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
                for idx, true_val, pred_val in zip(test_indices, y_test.tolist(), y_pred):
                    if true_val != pred_val:
                        misclassified_samples.append({
                            'index': idx,
                            'true_label': true_val,
                            'predicted_label': pred_val
                        })

            # Calculate metrics on all predictions
            results['model_metrics'][name] = self._calculate_metrics(y_true_all, y_pred_all)
            results['confusion_matrices'][name] = confusion_matrix(y_true_all, y_pred_all)
            results['misclassified_samples'][name] = misclassified_samples

            print(f"\n{name} Classification Report:")
            print(classification_report(y_true_all, y_pred_all))

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
        results = {'model_metrics': {}, 'confusion_matrices': {}, 'misclassified_samples': {}, 'feature_importance': {}}

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
                temp_df['label'] = y_train.values
                temp_df['channel'] = 'combined'  # Placeholder
                temp_df['session'] = 0  # Placeholder

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
                top_features = ensemble_importance['feature'].head(self.n_features_to_select).tolist()

                # Update selection frequency
                for feature in top_features:
                    feature_selection_frequency[feature] += 1

                # Scale the data
                scaler = StandardScaler()
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)

                # Select the identified top features
                X_train_selected = X_train_scaled[:, [list(X_df.columns).index(feat) for feat in top_features]]
                X_test_selected = X_test_scaled[:, [list(X_df.columns).index(feat) for feat in top_features]]

                # Train model on this fold's training data
                model.fit(X_train_selected, y_train)
                y_pred = model.predict(X_test_selected)

                y_true_all.extend(y_test.tolist())
                y_pred_all.extend(y_pred)

                # Record misclassified samples
                for idx, true_val, pred_val in zip(test_indices, y_test.tolist(), y_pred):
                    if true_val != pred_val:
                        misclassified_samples.append({
                            'index': idx,
                            'true_label': true_val,
                            'predicted_label': pred_val,
                            'fold': fold_idx
                        })

            # Aggregate feature importance across folds
            all_features = set()
            for fold_importance in fold_importances:
                all_features.update(fold_importance['feature'].tolist())

            # Create overall feature importance by averaging across folds
            overall_importance = []
            for feature in all_features:
                scores = [fold_imp.loc[fold_imp['feature'] == feature, 'ensemble_score'].values[0]
                          for fold_imp in fold_importances
                          if feature in fold_imp['feature'].values]

                overall_importance.append({
                    'feature': feature,
                    'avg_ensemble_score': sum(scores) / len(scores),
                    'selection_frequency': feature_selection_frequency[feature] / len(fold_importances)
                })

            # Sort by average importance
            overall_importance_df = pd.DataFrame(overall_importance)
            overall_importance_df = overall_importance_df.sort_values('avg_ensemble_score', ascending=False)

            # Calculate metrics on all predictions
            results['model_metrics'][name] = self._calculate_metrics(y_true_all, y_pred_all)
            results['confusion_matrices'][name] = confusion_matrix(y_true_all, y_pred_all)
            results['misclassified_samples'][name] = misclassified_samples
            results['feature_importance'][name] = overall_importance_df

            print(f"\n{name} Classification Report:")
            print(classification_report(y_true_all, y_pred_all))

            # Print top features by average importance
            print(f"\nTop {min(10, len(overall_importance_df))} Features:")
            for i, (_, row) in enumerate(overall_importance_df.head(10).iterrows()):
                print(f"{i + 1}. {row['feature']} - Score: {row['avg_ensemble_score']:.4f}, " +
                      f"Selected in {int(row['selection_frequency'] * len(fold_importances))}/{len(fold_importances)} folds")

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
        X = df.drop(['label', 'channel', 'session'], axis=1)
        y = df['label']

        # Calculate ANOVA F-values
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X, y)

        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'f_value': selector.scores_,
            'p_value': selector.pvalues_
        })

        # Sort by importance (higher F-value = more important)
        importance_df = importance_df.sort_values('f_value', ascending=False).reset_index(drop=True)

        # Add normalized importance (0-100 scale)
        if importance_df['f_value'].max() > 0:
            importance_df['importance'] = 100.0 * importance_df['f_value'] / importance_df['f_value'].max()
        else:
            importance_df['importance'] = 0

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
        X = df.drop(['label', 'channel', 'session'], axis=1)
        y = df['label']

        # Scale features (recommended for mutual information)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Calculate mutual information
        mi_scores = mutual_info_classif(X_scaled, y, random_state=42)

        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'mutual_info': mi_scores
        })

        # Sort by importance (higher MI = more important)
        importance_df = importance_df.sort_values('mutual_info', ascending=False).reset_index(drop=True)

        # Add normalized importance (0-100 scale)
        if importance_df['mutual_info'].max() > 0:
            importance_df['importance'] = 100.0 * importance_df['mutual_info'] / importance_df['mutual_info'].max()
        else:
            importance_df['importance'] = 0

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
        X = df.drop(['label', 'channel', 'session'], axis=1)
        y = df['label']

        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X, y)

        # Get feature importances
        importances = rf.feature_importances_

        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'rf_importance': importances
        })

        # Sort by importance (higher = more important)
        importance_df = importance_df.sort_values('rf_importance', ascending=False).reset_index(drop=True)

        # Add normalized importance (0-100 scale)
        importance_df['importance'] = 100.0 * importance_df['rf_importance'] / importance_df['rf_importance'].max()

        return importance_df

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
        anova_norm['norm_score'] = anova_norm['importance'] / 100.0

        mi_norm = mi_df.copy()
        mi_norm['norm_score'] = mi_norm['importance'] / 100.0

        rf_norm = rf_df.copy()
        rf_norm['norm_score'] = rf_norm['importance'] / 100.0

        # Create mappings of feature to normalized score
        anova_map = dict(zip(anova_norm['feature'], anova_norm['norm_score']))
        mi_map = dict(zip(mi_norm['feature'], mi_norm['norm_score']))
        rf_map = dict(zip(rf_norm['feature'], rf_norm['norm_score']))

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
            ensemble_score = (0.3 * anova_score + 0.3 * mi_score + 0.4 * rf_score)

            ensemble_data.append({
                'feature': feature,
                'anova_score': anova_score,
                'mi_score': mi_score,
                'rf_score': rf_score,
                'ensemble_score': ensemble_score
            })

        # Create dataframe and sort by ensemble score
        ensemble_df = pd.DataFrame(ensemble_data)
        ensemble_df = ensemble_df.sort_values('ensemble_score', ascending=False).reset_index(drop=True)

        # Add normalized importance (0-100 scale)
        if ensemble_df['ensemble_score'].max() > 0:
            ensemble_df['importance'] = 100.0 * ensemble_df['ensemble_score'] / ensemble_df['ensemble_score'].max()
        else:
            ensemble_df['importance'] = 0

        return ensemble_df

    def _evaluate_with_holdout(self, X, y, models, _):
        """Evaluate models with holdout validation - fixed to avoid information leaks"""
        results = {'model_metrics': {}, 'confusion_matrices': {}, 'misclassified_samples': {}}

        # Split data once
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42, stratify=y)

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

            results['model_metrics'][name] = self._calculate_metrics(y_test.tolist(), y_pred)
            results['confusion_matrices'][name] = confusion_matrix(y_test.tolist(), y_pred)

            misclassified_samples = []
            for idx, true_val, pred_val in zip(test_indices, y_test.tolist(), y_pred):
                if true_val != pred_val:
                    misclassified_samples.append({
                        'index': idx,
                        'true_label': true_val,
                        'predicted_label': pred_val
                    })
            results['misclassified_samples'][name] = misclassified_samples

            print(f"\n{name} Classification Report:")
            print(classification_report(y_test.tolist(), y_pred))

        return results

    def export_misclassified_samples(self):
        """Export misclassified samples and calculate session-level accuracy."""
        print("\n---- EXPORTING MISCLASSIFIED SAMPLES & SESSION ACCURACY ----")

        for combo in self.detailed_results:
            combo_str = '_'.join(combo)
            df_combo = self.df[self.df["label"].isin(combo)]  # Get data for this label combo

            # Only proceed if sessions exist in the data
            if 'session' not in df_combo.columns:
                print(f"No session data for {combo_str}. Skipping session accuracy.")
                continue

            misclassified_dict = self.detailed_results[combo].get('misclassified_samples', {})

            for model, misclassified in misclassified_dict.items():
                if not misclassified:
                    continue

                # Export misclassified samples
                mis_df = pd.DataFrame(misclassified)
                merged = pd.merge(mis_df, self.df.reset_index(),
                                  left_on='index', right_on='index', how='left')
                filename = f"{self.run_directory}/misclassified_{combo_str}_{model}.csv"
                merged.to_csv(filename, index=False)

                # --- New: Calculate session accuracy ---
                session_accuracies = []
                for session in df_combo['session'].unique():
                    # Total samples in this session for current label combination
                    total_samples = df_combo[df_combo['session'] == session].shape[0]

                    # Misclassified samples in this session
                    mis_in_session = merged[merged['session'] == session].shape[0]

                    # Calculate accuracy
                    accuracy = (total_samples - mis_in_session) / total_samples if total_samples > 0 else 0

                    session_accuracies.append({
                        'session': session,
                        'total_samples': total_samples,
                        'misclassified': mis_in_session,
                        'accuracy': accuracy
                    })

                # Save session accuracy results
                session_acc_df = pd.DataFrame(session_accuracies)
                session_filename = f"{self.run_directory}/session_accuracy_{combo_str}_{model}.csv"
                session_acc_df.to_csv(session_filename, index=False)

                print(f"Exported session accuracy for {combo_str} ({model}) to '{session_filename}'")
    def _calculate_metrics(self, y_true, y_pred):
        """Calculate comprehensive performance metrics"""
        precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='macro')
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')

        # For binary classification, also calculate class-specific metrics
        if len(np.unique(y_true)) == 2:
            precision_class, recall_class, f1_class, _ = precision_recall_fscore_support(y_true, y_pred, average=None)
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
            'accuracy': accuracy,
            'precision_macro': precision,
            'recall_macro': recall,
            'f1_macro': f1,
            'precision_weighted': precision_w,
            'recall_weighted': recall_w,
            'f1_weighted': f1_w,
            **class_metrics
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
        w = df['n_samples'].to_numpy()
        mu_w = np.average(df['outer_f1'], weights=w)
        sigma_w = np.sqrt(np.average((df['outer_f1'] - mu_w) ** 2, weights=w))

        # add global summary (last row)
        df_summary = pd.DataFrame([{
            'fold': 'MEAN±STD',
            'inner_best_f1': f"{np.average(df.inner_best_f1, weights=w):.3f} ± "
                             f"{np.sqrt(np.average((df.inner_best_f1 -np.average(df.inner_best_f1, weights=w)) ** 2,weights=w)):.3f}",
            'outer_f1': f"{mu_w:.3f} ± {sigma_w:.3f}",
            'inner_best_params': ''
        }])
        out_df = pd.concat([df, df_summary], ignore_index=True)

        # pretty-print params as JSON
        out_df['inner_best_params'] = out_df['inner_best_params'].apply(
            lambda p: json.dumps(p, default=str) if isinstance(p, dict) else p)

        out_file = os.path.join(
            self.run_directory,
            f"nested_cv_log_{model_name.lower()}.csv"
        )
        out_df.to_csv(out_file, index=False)
        print(f"✓ Saved nested-CV log for {model_name} → {out_file}")

    def visualize_confusion_matrix(self):
        """Visualize confusion matrix for the best label combination"""
        print("\n---- VISUALIZING CONFUSION MATRIX ----")

        if 'confusion_matrices' not in self.detailed_results[self.best_labels]:
            print("Confusion matrix not available. Skip visualization.")
            return

        best_model = self.detailed_results[self.best_labels]['best_model']
        cm = self.detailed_results[self.best_labels]['confusion_matrices'][best_model]
        labels = list(self.best_labels)

        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix for {best_model} on {", ".join(labels)}')

        cm_file = f"{self.run_directory}/{self.channel_approach}_confusion_matrix_{self.cv_method}.png"
        plt.savefig(cm_file, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved as '{cm_file}'")
        plt.close()

    # [confidence_ellipse, visualize_pca, visualize_feature_importance, analyze_channel_distribution methods remain unchanged]
    @staticmethod
    def confidence_ellipse(x, y, ax, n_std=2.0, facecolor='none', **kwargs):
        """
        Create a plot of the covariance confidence ellipse of x and y.
        """
        if x.size != y.size:
            raise ValueError("x and y must be the same size")
        cov = np.cov(x, y)
        pearson = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
        ell_radius_x = np.sqrt(1 + pearson)
        ell_radius_y = np.sqrt(1 - pearson)
        ellipse = patches.Ellipse((0, 0),
                                  width=ell_radius_x * 2,
                                  height=ell_radius_y * 2,
                                  facecolor=facecolor,
                                  **kwargs)
        scale_x = np.sqrt(cov[0, 0]) * n_std
        mean_x = np.mean(x)
        scale_y = np.sqrt(cov[1, 1]) * n_std
        mean_y = np.mean(y)
        transf = transforms.Affine2D().rotate_deg(45).scale(scale_x, scale_y).translate(mean_x, mean_y)
        ellipse.set_transform(transf + ax.transData)
        return ax.add_patch(ellipse)

    def visualize_pca(self):
        print("\n---- VISUALIZING PCA ----")
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(self.X_scaled)
        df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
        df_pca["label"] = self.y
        if 'channel' in self.df.columns:
            df_pca["channel"] = self.df["channel"]

        explained_variance = pca.explained_variance_ratio_ * 100
        plt.figure(figsize=(12, 10))

        # Set up color palette and markers
        palette = sns.color_palette("colorblind", n_colors=len(self.unique_labels))
        color_dict = {label: palette[i] for i, label in enumerate(self.unique_labels)}
        markers = ['o', 's', 'd', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'D', 'P', 'X']
        marker_dict = {channel: markers[i % len(markers)] for i, channel in enumerate(self.unique_channels)}

        # Plot each label (and channel if available)
        for label in self.unique_labels:
            label_data = df_pca[df_pca['label'] == label]
            if 'channel' in self.df.columns:
                for channel in self.unique_channels:
                    channel_data = label_data[label_data['channel'] == channel]
                    if not channel_data.empty:
                        plt.scatter(channel_data["PC1"],
                                    channel_data["PC2"],
                                    s=100,
                                    c=[color_dict[label]],
                                    marker=marker_dict[channel],
                                    alpha=0.8,
                                    edgecolor='w',
                                    linewidth=0.5,
                                    label=f"{label} ({channel})")
            else:
                plt.scatter(label_data["PC1"],
                            label_data["PC2"],
                            s=100,
                            c=[color_dict[label]],
                            alpha=0.8,
                            edgecolor='w',
                            linewidth=0.5,
                            label=label)

        ax = plt.gca()
        # Draw confidence ellipses for each label (if enough points exist)
        for label in self.unique_labels:
            label_data = df_pca[df_pca['label'] == label]
            if len(label_data) >= 3:
                self.confidence_ellipse(label_data["PC1"], label_data["PC2"],
                                        ax, n_std=2.0,
                                        edgecolor=color_dict[label],
                                        linewidth=2,
                                        alpha=0.5)

        # Add centroids and annotations for each label
        for label in self.unique_labels:
            label_data = df_pca[df_pca['label'] == label]
            centroid_x = label_data['PC1'].mean()
            centroid_y = label_data['PC2'].mean()
            plt.scatter(centroid_x, centroid_y,
                        s=200,
                        c=[color_dict[label]],
                        marker='X',
                        edgecolor='black',
                        linewidth=1.5,
                        alpha=1.0)
            plt.annotate(f"{label}",
                         (centroid_x, centroid_y),
                         fontsize=12,
                         fontweight='bold',
                         ha='center',
                         va='bottom',
                         xytext=(0, 10),
                         textcoords='offset points',
                         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.8))

        plt.title("PCA Visualization of EEG Features\nChannel markers show distribution of readings", fontsize=16,
                  pad=20)
        plt.xlabel(f"PC1 ({explained_variance[0]:.2f}% Variance)", fontsize=12)
        plt.ylabel(f"PC2 ({explained_variance[1]:.2f}% Variance)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Avoid duplicate legend entries
        handles, labels_leg = plt.gca().get_legend_handles_labels()
        by_label = {}
        for h, l in zip(handles, labels_leg):
            label_part = l.split(' ')[0]
            if label_part not in by_label:
                by_label[label_part] = h
        plt.legend(by_label.values(), by_label.keys(),
                   title="Labels", title_fontsize=12, fontsize=10,
                   loc='best', frameon=True, framealpha=0.95)

        # Add a second legend for channel markers if available
        if 'channel' in self.df.columns:
            marker_handles = [plt.Line2D([0], [0], marker=marker_dict[ch], color='gray',
                                         linestyle='None', markersize=10)
                              for ch in self.unique_channels]
            marker_labels = [f"Channel: {ch}" for ch in self.unique_channels]
            plt.figlegend(marker_handles, marker_labels,
                          loc='lower center', ncol=len(self.unique_channels),
                          bbox_to_anchor=(0.5, 0), fontsize=10, frameon=True)
            plt.subplots_adjust(bottom=0.15)

        # Get metrics for the best model
        best_model = self.detailed_results[self.best_labels]['best_model']
        metrics = self.detailed_results[self.best_labels]['metrics'][best_model]

        # Add text about the best label combination with multiple metrics
        plt.figtext(0.5, 0.01,
                    f"Most Separable Labels: {', '.join(self.best_labels)}\n"
                    f"Model: {best_model} | Accuracy: {metrics['accuracy']:.4f} | F1: {metrics['f1_macro']:.4f} | Precision: {metrics['precision_macro']:.4f} | Recall: {metrics['recall_macro']:.4f}",
                    ha="center", fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.8))
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        output_file = f"{self.run_directory}/{self.channel_approach}_eeg_pooled_visualization_top{self.top_n_labels}_labels_top{self.n_features_to_select}_features_{self.cv_method}.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"PCA visualization saved as '{output_file}'")
        plt.close()

    def visualize_pca_3d(self):
        """
        Similar to visualize_pca, but uses 3 principal components
        and plots them in a 3D scatter plot.
        """
        print("\n---- VISUALIZING 3D PCA ----")

        # 1) Fit PCA with 3 components on your globally-scaled data (self.X_scaled)
        pca = PCA(n_components=3)
        X_pca_3d = pca.fit_transform(self.X_scaled)
        df_pca_3d = pd.DataFrame(X_pca_3d, columns=["PC1", "PC2", "PC3"])
        df_pca_3d["label"] = self.y
        if 'channel' in self.df.columns:
            df_pca_3d["channel"] = self.df["channel"]

        # 2) Prepare color & marker mappings
        palette = sns.color_palette("colorblind", n_colors=len(self.unique_labels))
        color_dict = {label: palette[i] for i, label in enumerate(self.unique_labels)}
        markers = ['o', 's', 'd', '^', 'v', '<', '>', 'p', '*', 'h', 'H', 'D', 'P', 'X']
        marker_dict = {ch: markers[i % len(markers)] for i, ch in enumerate(self.unique_channels)}

        # 3) Create 3D figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 4) Scatter-plot each label/channel
        for label in self.unique_labels:
            subset_label = df_pca_3d[df_pca_3d['label'] == label]
            if 'channel' in self.df.columns:
                for channel in self.unique_channels:
                    subset_chan = subset_label[subset_label['channel'] == channel]
                    if not subset_chan.empty:
                        ax.scatter(
                            subset_chan["PC1"], subset_chan["PC2"], subset_chan["PC3"],
                            c=[color_dict[label]],
                            marker=marker_dict[channel],
                            alpha=0.8,
                            edgecolor='w',
                            linewidth=0.5,
                            label=f"{label} ({channel})"
                        )
            else:
                ax.scatter(
                    subset_label["PC1"], subset_label["PC2"], subset_label["PC3"],
                    c=[color_dict[label]],
                    alpha=0.8,
                    edgecolor='w',
                    linewidth=0.5,
                    label=label
                )

        # 5) Label axes with explained variance
        explained_var = pca.explained_variance_ratio_ * 100
        ax.set_xlabel(f"PC1 ({explained_var[0]:.2f}% var)")
        ax.set_ylabel(f"PC2 ({explained_var[1]:.2f}% var)")
        ax.set_zlabel(f"PC3 ({explained_var[2]:.2f}% var)")
        ax.set_title("3D PCA Visualization of EEG Features", pad=20)

        # Optional: remove duplicate legend entries
        handles, labels = ax.get_legend_handles_labels()
        unique_legend = {}
        for h, lbl in zip(handles, labels):
            if lbl not in unique_legend:
                unique_legend[lbl] = h
        ax.legend(unique_legend.values(), unique_legend.keys(), loc='best', frameon=True)

        # 6) Save figure (static PNG, not interactive)
        output_file = f"{self.run_directory}/{self.channel_approach}_3d_pca_visualization.png"
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"3D PCA plot saved as '{output_file}'")
        plt.close(fig)

    def visualize_feature_importance(self):
        print("\n---- VISUALIZING FEATURE IMPORTANCE ----")
        # Focus on data corresponding to the best label combination
        df_best = self.df[self.df["label"].isin(self.best_labels)]
        X_best = df_best[self.feature_columns]
        y_best = df_best["label"]

        # Standardize and compute ANOVA F-values
        X_best_scaled = self.scaler.transform(X_best)
        f_values, p_values = f_classif(X_best_scaled, y_best)
        feature_scores = pd.DataFrame({
            'Feature': self.feature_columns,
            'F_Score': f_values,
            'P_Value': p_values,
            'Log10_F': np.log10(f_values + 1)
        })
        feature_scores = feature_scores.sort_values('F_Score', ascending=False)
        feature_scores['Significant'] = feature_scores['P_Value'] < 0.05
        feature_scores['Color'] = feature_scores['Significant'].map({True: 'darkblue', False: 'lightblue'})

        plt.figure(figsize=(14, 10))
        plt.barh(feature_scores['Feature'], feature_scores['Log10_F'], color=feature_scores['Color'])
        plt.title(f'Feature Importance for Distinguishing Between {self.best_labels}', fontsize=16)
        plt.xlabel('Log10(F-Score+1) - Higher Values = More Discriminative', fontsize=12)
        plt.ylabel('EEG Feature', fontsize=12)
        plt.grid(axis='x', linestyle='--', alpha=0.7)

        # Draw significance threshold line if applicable
        sig_features = feature_scores[feature_scores['Significant']]
        if not sig_features.empty:
            min_sig_log_f = np.log10(sig_features['F_Score'].min() + 1)
            plt.axvline(x=min_sig_log_f, color='red', linestyle='--', alpha=0.7)
            plt.text(min_sig_log_f + 0.1, 1, 'Significance Threshold (p<0.05)',
                     rotation=90, color='red', verticalalignment='bottom')

        # Annotate bar plot with F-scores and p-values
        for i, (_, row) in enumerate(feature_scores.iterrows()):
            plt.text(row['Log10_F'] + 0.1, i,
                     f"F={row['F_Score']:.2f}, p={row['P_Value']:.4f}",
                     va='center', fontsize=9)

        plt.tight_layout()
        importance_file = f"{self.run_directory}/{self.channel_approach}_feature_importance_top{self.top_n_labels}_labels_top{self.n_features_to_select}_features_{self.cv_method}.png"
        plt.savefig(importance_file, dpi=300, bbox_inches='tight')
        print(f"Feature importance visualization saved as '{importance_file}'")
        plt.close()

    # Complete the analyze_channel_distribution method which was cut off
    def analyze_channel_distribution(self):
        print("\n---- ANALYZING CHANNEL DISTRIBUTION ----")
        if 'channel' in self.df.columns and self.channel_approach == "pooled":
            label_channel_counts = pd.crosstab(self.df['label'], self.df['channel'])
            label_channel_pct = label_channel_counts.div(label_channel_counts.sum(axis=1), axis=0) * 100
            plt.figure(figsize=(len(self.unique_channels) * 1.5, len(self.unique_labels) * 1.2))
            sns.heatmap(label_channel_pct, annot=label_channel_counts, fmt="d", cmap="YlGnBu",
                        cbar_kws={'label': 'Sample Percentage (%)'})
            plt.title('Sample Distribution by Label and Channel', fontsize=16)
            plt.ylabel('Label', fontsize=12)
            plt.xlabel('Channel', fontsize=12)
            plt.tight_layout()

            # Save the channel distribution plot
            dist_file = f"{self.run_directory}/{self.channel_approach}_channel_distribution_{self.cv_method}.png"
            plt.savefig(dist_file, dpi=300, bbox_inches='tight')
            print(f"Channel distribution visualization saved as '{dist_file}'")
            plt.close()
        else:
            print("Channel distribution analysis skipped (either no channel data or not using pooled approach)")

    def visualize_metrics_comparison(self):
        """Visualize performance metrics comparison across label combinations"""
        print("\n---- VISUALIZING PERFORMANCE METRICS COMPARISON ----")

        # Prepare data for visualization
        metrics_data = []
        for combo in self.detailed_results:
            best_model = self.detailed_results[combo]['best_model']
            metrics = self.detailed_results[combo]['metrics'][best_model]
            combo_str = ', '.join(combo)

            metrics_data.append({
                'Combination': combo_str,
                'Accuracy': metrics['accuracy'],
                'F1 Score': metrics['f1_macro'],
                'Precision': metrics['precision_macro'],
                'Recall': metrics['recall_macro']
            })

        # Convert to DataFrame and reshape for plotting
        metrics_df = pd.DataFrame(metrics_data)
        metrics_plot_df = pd.melt(metrics_df,
                                  id_vars=['Combination'],
                                  value_vars=['Accuracy', 'F1 Score', 'Precision', 'Recall'],
                                  var_name='Metric', value_name='Score')

        # Create the comparison plot
        plt.figure(figsize=(12, 8))
        g = sns.catplot(x='Combination', y='Score', hue='Metric',
                        data=metrics_plot_df, kind='bar', height=6, aspect=1.5)

        plt.title('Performance Metrics Comparison Across Label Combinations', fontsize=16)
        plt.xlabel('Label Combination', fontsize=12)
        plt.ylabel('Score', fontsize=12)
        plt.ylim(0, 1.0)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()

        metrics_file = f"{self.run_directory}/{self.channel_approach}_metrics_comparison_{self.cv_method}.png"
        plt.savefig(metrics_file, dpi=300, bbox_inches='tight')
        print(f"Metrics comparison visualization saved as '{metrics_file}'")
        plt.close()

    def export_results(self):
        """Export detailed results to CSV files"""
        print("\n---- EXPORTING RESULTS ----")

        # Export metrics for all label combinations
        metrics_data = []
        for combo in self.detailed_results:
            best_model = self.detailed_results[combo]['best_model']
            metrics = self.detailed_results[combo]['metrics'][best_model]
            combo_str = '_'.join(combo)

            metrics_row = {
                'label_combination': combo_str,
                'best_model': best_model,
                'n_samples': self.detailed_results[combo]['n_samples']
            }

            # Add sample counts per label
            for label, count in self.detailed_results[combo]['sample_counts'].items():
                metrics_row[f'samples_{label}'] = count

            # Add all metrics
            for metric, value in metrics.items():
                metrics_row[metric] = value

            metrics_data.append(metrics_row)

        # Save to CSV
        metrics_df = pd.DataFrame(metrics_data)
        metrics_file = f"{self.run_directory}/{self.channel_approach}_performance_metrics_{self.cv_method}.csv"
        metrics_df.to_csv(metrics_file, index=False)
        print(f"Performance metrics exported to '{metrics_file}'")

        # Export feature importance for best label combination
        df_best = self.df[self.df["label"].isin(self.best_labels)]
        X_best = df_best[self.feature_columns]
        y_best = df_best["label"]

        # Standardize and compute ANOVA F-values
        X_best_scaled = self.scaler.transform(X_best)
        f_values, p_values = f_classif(X_best_scaled, y_best)

        feature_scores = pd.DataFrame({
            'feature': self.feature_columns,
            'f_score': f_values,
            'p_value': p_values,
            'selected': [f in self.selected_features for f in self.feature_columns]
        })
        feature_scores = feature_scores.sort_values('f_score', ascending=False)

        features_file = f"{self.run_directory}/{self.channel_approach}_feature_importance_{self.cv_method}.csv"
        feature_scores.to_csv(features_file, index=False)
        print(f"Feature importance scores exported to '{features_file}'")
        self.export_misclassified_samples()

    def optimize_hyperparameters(self, n_iter=25):
        """
        Perform Bayesian hyperparameter optimization using scikit-optimize.

        Parameters:
            n_iter (int): Number of iterations for the optimization process.

        Returns:
            dict: Best hyperparameters found.
        """
        print("\n---- HYPERPARAMETER OPTIMIZATION ----")
        from skopt import BayesSearchCV
        from skopt.space import Real, Integer, Categorical
        from tqdm import tqdm

        # Focus on data corresponding to the best label combination
        if not hasattr(self, 'best_labels'):
            print("Run evaluate_label_combinations first to find the best labels")
            return

        df_best = self.df[self.df["label"].isin(self.best_labels)]
        X_best = df_best[self.feature_columns]
        y_best = df_best["label"]

        # Define the CV strategy based on existing settings
        if self.cv_method == "loo":
            from sklearn.model_selection import LeaveOneOut
            cv = LeaveOneOut()
        elif self.cv_method == "kfold":
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(n_splits=self.kfold_splits)
        else:  # holdout - we'll use k-fold for optimization
            from sklearn.model_selection import StratifiedKFold
            cv = StratifiedKFold(n_splits=5)

        # Create pipeline with preprocessing steps to avoid information leakage
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.feature_selection import SelectKBest, f_classif

        # Define search spaces for different models
        search_spaces = {
            # ───────────────────────────────────────────────────────── Random Forest ──
            'RandomForest': {
                'model': Categorical([RandomForestClassifier(
                    random_state=42, n_jobs=-1)]),
                'model__n_estimators': Categorical([50]),  # fewer trees → less variance
                'model__max_depth': Categorical([4]),  # very shallow
                'model__min_samples_split': Integer(2, 6),
                'model__min_samples_leaf': Integer(2, 8),  # prevents tiny leaves
                'model__class_weight': Categorical(['balanced']),
                'feature_selection__k': Categorical([20]) #TODO change back if you want search on that
            },

            # ──────────────────────────────────────────────────────────────   SVM  ──
            'SVM': {
                'model': Categorical([SVC(
                    random_state=42, probability=True)]),
                'model__kernel': Categorical(['linear', 'rbf']),  # poly removed
                'model__C': Real(1e-2, 5.0, prior = 'log-uniform'),
        'model__gamma': Real(1e-4, 5e-2, prior = 'log-uniform'),
        'model__class_weight': Categorical(['balanced', None]),
        'feature_selection__k': Integer(4, 15)
        },

        # # ─────────────────────────────────────────── Elastic‑Net Logistic Reg ──
        'ElasticNetLogReg': {
            'model': Categorical([LogisticRegression(
                penalty='elasticnet', solver='saga',
                class_weight='balanced',
                max_iter=4000, random_state=42)]),
            'model__C': Real(1e-2, 5.0, prior = 'log-uniform'),
        'model__l1_ratio': Real(0.1, 0.9),  # avoid extremes
        'feature_selection__k': Integer(5, 20)
        },

        # ───────────────────────────────────────────────────────── Extra Trees ──
        'ExtraTrees': {
            'model': Categorical([ExtraTreesClassifier(
                class_weight='balanced',
                random_state=42, n_jobs=-1)]),
            'model__n_estimators': Integer(80, 200),
            'model__max_depth': Integer(2, 8),
            'model__min_samples_leaf': Integer(2, 6),
            'feature_selection__k': Integer(5, 20)
        },

        # ────────────────────────────────────────── HistGradientBoosting ──
        'HGBClassifier': {
            'model': Categorical([HistGradientBoostingClassifier(
                class_weight='balanced', random_state=42)]),
            'model__learning_rate': Real(0.02, 0.12, prior='log-uniform'),
            'model__max_depth': Integer(2, 4),
            'model__max_iter': Integer(60, 100),
            'feature_selection__k': Integer(5, 20)
        },

        # ───────────────────────────────────────────────────── k‑Nearest Nbrs ──
        'kNN': {
            'model': Categorical([KNeighborsClassifier()]),
            'model__n_neighbors': Integer(5, 15),  # ≥5 to limit variance
            'model__weights': Categorical(['uniform', 'distance']),
            'feature_selection__k': Integer(5, 20)
        },

        # ────────────────────────────────────────────── Gaussian Naïve Bayes ──
        'GaussianNB': {
            'model': Categorical([GaussianNB()])
            # no hyper‑params
        },

        # ────────────────────────────────────────────── Shrinkage LDA ──
        'ShrinkageLDA': {
            'model': Categorical([LinearDiscriminantAnalysis(
                solver='lsqr')]),
            'model__shrinkage': Categorical(['auto', 0.1, 0.3, None])
        }
        }

        outer_cv, inner_cv_proto = self._get_nested_splitters(
            df_best["session"].values)  # session IDs array

        best_results, all_results = {}, {}

        for model_name, space in tqdm(search_spaces.items(),
                                      desc="Optimising (nested CV)"):
            print(f"\n⟹  Optimising {model_name}")

            outer_scores, outer_std, fold_estimators, fold_sizes= [], [], [], []
            outer_log = []  # ← new: per-model log

            # ───── outer loop (may be None for 'holdout') ───────────────────────
            if outer_cv is None:  # holdout   – single dev/test split
                outer_indices = [(np.arange(len(X_best)), np.arange(len(X_best)))]
            else:
                outer_indices = outer_cv.split(X_best, y_best,
                                               groups=df_best["session"].values)

            for fold, (tr_idx, te_idx) in enumerate(outer_indices, start=1):
                X_tr, X_te = X_best.iloc[tr_idx], X_best.iloc[te_idx]
                y_tr, y_te = y_best.iloc[tr_idx], y_best.iloc[te_idx]

                # make an *inner* CV object limited to the training sessions
                if outer_cv is None:
                    inner_cv = inner_cv_proto  # plain StratKFold
                else:
                    inner_cv = list(
                        inner_cv_proto.split(
                            X_tr, y_tr,
                            groups=df_best["session"].values[tr_idx])
                    )

                # ----------------  inner Bayesian search  ----------------------
                pipe = Pipeline([
                    ('scaler', StandardScaler()),
                    ('feature_selection', SelectKBest(f_classif)),
                    ('model', RandomForestClassifier())  # placeholder, overwritten by BayesSearch
                ])

                opt = BayesSearchCV(
                    pipe,
                    space,
                    n_iter=n_iter,
                    cv=inner_cv,
                    scoring='f1_macro',
                    n_jobs=-1,
                    random_state=42,
                    verbose=0
                )
                opt.fit(X_tr, y_tr)

                # unbiased score on the *outer-test* portion
                y_pred = opt.predict(X_te)
                outer_acc = accuracy_score(y_te, y_pred)
                prec, rec, f1c, _ = precision_recall_fscore_support(
                    y_te, y_pred, average='macro')

                outer_f1 = f1_score(y_te, y_pred, average='macro')
                inner_best_f = opt.best_score_

                outer_scores.append(outer_f1)
                fold_sizes.append(len(te_idx))  # <-- add this

                fold_estimators.append(opt.best_estimator_)

                outer_log.append({
                    'fold': fold,
                    'n_samples': len(te_idx),  # ← new column
                    'inner_best_f1': round(inner_best_f, 3),
                    'outer_f1': round(outer_f1, 3),
                    'outer_acc': round(outer_acc, 3),
                    'outer_prec': round(prec, 3),
                    'outer_recall': round(rec, 3),
                    'inner_best_params': opt.best_params_,

                })
                labels = y_te.unique()  # ['m_et_s', 'n_3_s']
                prec_c, rec_c, f1_c, _ = precision_recall_fscore_support(
                    y_te, y_pred, average=None, labels=labels)

                for lbl, p, r, f in zip(labels, prec_c, rec_c, f1_c):
                    outer_log[-1][f'f1_{lbl}'] = round(f, 3)
                    outer_log[-1][f'prec_{lbl}'] = round(p, 3)
                    outer_log[-1][f'recall_{lbl}'] = round(r, 3)

                print(f"   fold {fold}: best inner F-score = {opt.best_score_:.3f} | "
                      f"outer F-score = {outer_scores[-1]:.3f}")

            # summary across outer folds
            mu = np.average(outer_scores, weights=fold_sizes)
            sigma = np.sqrt(np.average((outer_scores - mu) ** 2,
                                       weights=fold_sizes))
            print(f"→ Nested-CV F1 for {model_name}: {mu:.3f} ± {sigma:.3f}")

            best_results[model_name] = {
                'outer_mean_f1': mu,
                'outer_std_f1': sigma,
                'per_fold_f1': outer_scores,
                'best_fold_model': fold_estimators[int(np.argmax(outer_scores))]
            }

            # keep raw BayesSearch history if you wish
            all_results[model_name] = {
                'params': opt.cv_results_['params'],
                'scores': opt.cv_results_['mean_test_score']
            }
            # NEW – save per-fold inner / outer log
            self._persist_nested_cv_log(model_name, outer_log)
            # Save the best model to a file
            import joblib
            joblib.dump(opt.best_estimator_, f"{self.run_directory}/best_{model_name.lower()}_model.pkl")

        # pick overall winner
        best_model_name = max(best_results, key=lambda m: best_results[m]['outer_mean_f1'])
        print(f"\n★★  Best (nested-CV) model = {best_model_name}  "
              f"{best_results[best_model_name]['outer_mean_f1']:.3f} ± "
              f"{best_results[best_model_name]['outer_std_f1']:.3f}")


        # Determine the best model overall
        # (store the real estimator + use the mean outer F1)
        best_model = best_results[best_model_name]['best_fold_model']  # ← a Pipeline
        best_score = best_results[best_model_name]['outer_mean_f1']

        print(f"\nBest overall model: {best_model_name}")
        print(f"Best overall score (F1): {best_score:.4f}")

        # Save the optimization results
        self.optimization_results = {
            'best_model_name': best_model_name,
            'best_model': best_model,
            'best_score': best_score,
            'best_results': best_results,
            'all_results': all_results
        }

        # Visualize the optimization results
        self._visualize_optimization_results(all_results)

        return best_results

    def _visualize_optimization_results(self, all_results):
        """
        Visualize the optimization results.

        Parameters:
        all_results (dict): Results from optimization for all models.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        plt.figure(figsize=(15, 10))

        for i, (model_name, results) in enumerate(all_results.items()):
            # Sort by score to see the improvement
            indices = np.argsort(results['scores'])
            sorted_scores = np.array(results['scores'])[indices]

            # Plot the optimization progress
            plt.subplot(len(all_results), 1, i + 1)
            plt.plot(range(1, len(sorted_scores) + 1), sorted_scores, 'o-', label=f'{model_name} optimization')
            plt.axhline(y=max(results['scores']), color='r', linestyle='-',
                        label=f'Best score: {max(results["scores"]):.4f}')
            plt.title(f'{model_name} Optimization Progress')
            plt.xlabel('Iteration (sorted by score)')
            plt.ylabel('F1 Score')
            plt.grid(True)
            plt.legend()

        plt.tight_layout()
        plt.savefig(f"{self.run_directory}/optimization_progress.png", dpi=300)
        plt.close()

        # Create feature importance for best model
        self._visualize_best_model_feature_importance()

    def _visualize_best_model_feature_importance(self):
        """
        Visualize feature importance of the best optimized model.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import pandas as pd

        if not hasattr(self, 'optimization_results'):
            print("Run optimize_hyperparameters first")
            return

        best_model = self.optimization_results['best_model']
        best_model_name = self.optimization_results['best_model_name']

        # Get the feature names that were selected (from the pipeline)
        feature_selector = best_model.named_steps['feature_selection']
        selected_indices = feature_selector.get_support(indices=True)
        selected_features = [self.feature_columns[i] for i in selected_indices]

        # Get the actual model from the pipeline
        model = best_model.named_steps['model']

        plt.figure(figsize=(12, 8))

        # Extract feature importance based on model type
        if best_model_name == 'RandomForest':
            importances = model.feature_importances_
            std = np.std([tree.feature_importances_ for tree in model.estimators_], axis=0)

            # Create DataFrame for plotting
            feature_importance = pd.DataFrame({
                'feature': selected_features,
                'importance': importances,
                'std': std
            }).sort_values('importance', ascending=False)

            plt.barh(feature_importance['feature'], feature_importance['importance'], xerr=feature_importance['std'])
            plt.title(f'Feature Importance - Optimized {best_model_name}')

        elif best_model_name == 'SVM' and model.kernel in ['linear']:
            # For linear SVM, we can extract coefficients
            importances = np.abs(model.coef_[0])

            # Create DataFrame for plotting
            feature_importance = pd.DataFrame({
                'feature': selected_features,
                'importance': importances
            }).sort_values('importance', ascending=False)

            plt.barh(feature_importance['feature'], feature_importance['importance'])
            plt.title(f'Feature Importance (Coefficient Magnitude) - Optimized {best_model_name}')

        else:
            # For other models, use permutation importance
            from sklearn.inspection import permutation_importance

            # Get data for the best label combination
            df_best = self.df[self.df["label"].isin(self.best_labels)]
            X_best = df_best[self.feature_columns]
            y_best = df_best["label"]

            # Transform data through the pipeline up to the model
            X_transformed = best_model.named_steps['scaler'].transform(X_best)
            X_selected = best_model.named_steps['feature_selection'].transform(X_transformed)

            # Calculate permutation importance
            result = permutation_importance(model, X_selected, y_best, n_repeats=10, random_state=42)

            # Create DataFrame for plotting
            feature_importance = pd.DataFrame({
                'feature': selected_features,
                'importance': result.importances_mean,
                'std': result.importances_std
            }).sort_values('importance', ascending=False)

            plt.barh(feature_importance['feature'], feature_importance['importance'], xerr=feature_importance['std'])
            plt.title(f'Feature Importance (Permutation) - Optimized {best_model_name}')

        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig(f"{self.run_directory}/best_model_feature_importance.png", dpi=300)
        plt.close()

        # Save feature importance to CSV
        feature_importance.to_csv(f"{self.run_directory}/best_model_feature_importance.csv", index=False)

    # ------------------------------------------------------------
    #   ❶  Violin / strip plot of LOSO (or k-fold) outer scores
    # ------------------------------------------------------------
    def plot_outer_fold_distribution(self):
        """
        Draw a violin/strip plot of the outer-fold F1 scores for the
        best model picked in optimise_hyperparameters().
        Works for LOSO or k-fold.
        """
        if not hasattr(self, 'optimization_results'):
            print("Run optimise_hyperparameters first.")
            return

        best_name = self.optimization_results['best_model_name']
        fold_scores = self.optimization_results['best_results'  # dict from optimise_hyperparameters
        ][best_name]['per_fold_f1']

        import matplotlib.pyplot as plt, seaborn as sns, numpy as np
        plt.figure(figsize=(5, 6))
        sns.violinplot(data=fold_scores, inner=None, color='skyblue')
        sns.stripplot(data=fold_scores, color='black', size=6, jitter=False)

        plt.ylabel('Macro-F1 (outer fold)')
        plt.title(f'{best_name} – distribution across {len(fold_scores)} outer folds')
        plt.ylim(0, 1)
        out_file = f"{self.run_directory}/outer_fold_f1_distribution.png"
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved violin plot →  {out_file}")

    # ------------------------------------------------------------
    #   ❷  Permutation test (label shuffle)
    # ------------------------------------------------------------
    def permutation_test_best_model(self, n_perm=100, random_state=42):
        """
        Runs a label-shuffle test on the *best optimised pipeline*
        using the same cross-validation splitter (LOSO or k-fold).

        Returns
        -------
        p_value   : fraction of permutations whose mean F1 ≥ observed
        """
        if not hasattr(self, 'optimization_results'):
            print("Run optimise_hyperparameters first.")
            return

        rng = np.random.default_rng(random_state)
        best_pipe = self.optimization_results['best_model']
        best_name = self.optimization_results['best_model_name']

        # data for the winning label combination
        df_best = self.df[self.df["label"].isin(self.best_labels)]
        X = df_best[self.feature_columns].values
        y = df_best["label"].values
        groups = df_best["session"].values if self.cv_method == 'loso' else None

        # choose the same outer splitter
        if self.cv_method == 'loso':
            outer_splitter = LeaveOneGroupOut()
            split_iter = outer_splitter.split(X, y, groups)
        else:
            outer_splitter = StratifiedKFold(n_splits=self.kfold_splits, shuffle=True,
                                             random_state=123)
            split_iter = outer_splitter.split(X, y)

        # --- observed score -------------------------------------------------
        from sklearn.metrics import f1_score
        obs_scores = []
        for tr, te in split_iter:
            best_pipe.fit(X[tr], y[tr])
            obs_scores.append(f1_score(y[te], best_pipe.predict(X[te]),
                                       average='macro'))
        obs_mean = np.mean(obs_scores)

        print(f"Observed mean outer-fold F1 for {best_name}: {obs_mean:.3f}")

        # --- permutations ---------------------------------------------------
        null_means = []
        for p in tqdm(range(n_perm), desc="Permutations"):
            y_perm = rng.permutation(y)  # shuffle labels globally
            split_iter = (outer_splitter.split(X, y_perm, groups)
                          if groups is not None else
                          outer_splitter.split(X, y_perm))

            fold_f1 = []
            for tr, te in split_iter:
                best_pipe.fit(X[tr], y_perm[tr])
                fold_f1.append(
                    f1_score(y_perm[te], best_pipe.predict(X[te]),
                             average='macro')
                )
            null_means.append(np.mean(fold_f1))

        null_means = np.array(null_means)
        p_val = (np.sum(null_means >= obs_mean) + 1) / (n_perm + 1)
        print(f"Permutation-test p-value: {p_val:.4f}")

        # -------- histogram ------------
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.hist(null_means, bins=30, alpha=0.7)
        plt.axvline(obs_mean, color='red', lw=2,
                    label=f'Observed ({obs_mean:.3f})')
        plt.xlabel('Mean macro-F1 (null)')
        plt.ylabel('Count')
        plt.title(f'Permutation test – {best_name}\n p = {p_val:.4f}')
        plt.legend()
        out_file = f"{self.run_directory}/permutation_histogram_{best_name}.png"
        plt.savefig(out_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved permutation histogram →  {out_file}")

        # store for later if you like
        self.permutation_p_value = p_val
        return p_val

    def evaluate_best_model(self):
        """
        Evaluate the best model from hyperparameter optimization using the same
        cross-validation approach as the original models.
        """
        print("\n---- EVALUATING BEST OPTIMIZED MODEL ----")

        if not hasattr(self, 'optimization_results'):
            print("Run optimize_hyperparameters first")
            return

        best_model = self.optimization_results['best_model']
        best_model_name = self.optimization_results['best_model_name']

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
        else:  # TODO Holdout is not working properly, fix it
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X_best, y_best, test_size=self.test_size, random_state=42, stratify=y_best)
            groups_eval = df_best["session"].values


        y_true_all = []
        y_pred_all = []
        misclassified_samples = []

        if self.cv_method in ["loo", "kfold", "loso"]:
            # Cross-validation approach (LOO or k-fold)
            for fold_idx, (train_idx, test_idx) in enumerate(
                    cv.split(X_best, y_best, groups=groups_eval) if groups_eval is not None
                    else cv.split(X_best, y_best)):
                X_train, X_test = X_best.iloc[train_idx], X_best.iloc[test_idx]
                y_train, y_test = y_best.iloc[train_idx], y_best.iloc[test_idx]
                test_indices = y_test.index.tolist()

                # Clone the model for each fold to ensure independence
                from sklearn.base import clone
                model_clone = clone(best_model)

                # Fit and predict
                model_clone.fit(X_train, y_train)
                y_pred = model_clone.predict(X_test)

                y_true_all.extend(y_test.tolist())
                y_pred_all.extend(y_pred)

                # Record misclassified samples
                for idx, true_val, pred_val in zip(test_indices, y_test, y_pred):
                    if true_val != pred_val:
                        misclassified_samples.append({
                            'index': idx,
                            'true_label': true_val,
                            'predicted_label': pred_val,
                            'fold': fold_idx
                        })
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
                    misclassified_samples.append({
                        'index': idx,
                        'true_label': true_val,
                        'predicted_label': pred_val
                    })

        # Calculate and display metrics
        metrics = self._calculate_metrics(y_true_all, y_pred_all)
        print(f"Optimized {best_model_name} Performance:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        # Generate confusion matrix
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_true_all, y_pred_all)

        # Visualize confusion matrix
        import matplotlib.pyplot as plt
        import seaborn as sns
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=self.best_labels, yticklabels=self.best_labels)
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title(f'Confusion Matrix for Optimized {best_model_name}')
        plt.tight_layout()
        plt.savefig(f"{self.run_directory}/optimized_model_confusion_matrix.png", dpi=300)
        plt.close()

        # Update detailed_results with optimized model's performance
        self.detailed_results[self.best_labels]['misclassified_samples'][best_model_name] = misclassified_samples
        self.detailed_results[self.best_labels]['metrics'][best_model_name] = metrics
        self.detailed_results[self.best_labels]['confusion_matrices'][best_model_name] = cm

        # Check if optimized model is better than previous best
        current_best_model = self.detailed_results[self.best_labels]['best_model']
        current_f1 = self.detailed_results[self.best_labels]['metrics'][current_best_model]['f1_macro']
        optimized_f1 = metrics['f1_macro']

        if optimized_f1 > current_f1:
            print(
                f"Optimized model {best_model_name} outperforms previous best {current_best_model}. Updating best model.")
            self.detailed_results[self.best_labels]['best_model'] = best_model_name
    def save_best_models(self):
        """Save both SVM and Random Forest models for the best label combination"""
        if not hasattr(self, 'best_labels'):
            print("No best labels identified. Run evaluate_label_combinations first.")
            return

        print("\n---- SAVING BASE MODELS ----")
        df_best = self.df[self.df["label"].isin(self.best_labels)]
        X_best = df_best[self.feature_columns]
        y_best = df_best["label"]

        # Define base models with original parameters
        models = {
            'RandomForest': RandomForestClassifier(
                n_estimators=200,
                max_depth=5,
                class_weight='balanced',
                random_state=48
            ),
            'SVM': SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced',
                random_state=42,
                probability=True
            )
        }

        # Create and save pipelines
        for name, model in models.items():
            # Create full preprocessing + classification pipeline
            pipeline = Pipeline([
                ('scaler', StandardScaler()),
                ('selector', SelectKBest(f_classif, k=self.n_features_to_select)),
                ('classifier', model)
            ])

            # Fit on entire best-label dataset
            pipeline.fit(X_best, y_best)

            # Save model
            filename = f"{self.run_directory}/base_{name.lower()}_model.pkl"
            joblib.dump(pipeline, filename)
            print(f"Saved {name} model to {filename}")

    # STEP 2: Modify the run_complete_analysis method to include hyperparameter optimization
    def run_complete_analysis(self, optimize_hyperparams=False, n_iter=25):
        """Run the complete analysis pipeline with fixed information leakage issues"""
        self.load_data()

        # Modified calls to prevent info leaks
        self.preprocess_data()  # Now just for exploration
        self.feature_selection()  # Now just for informational purposes
        self.evaluate_label_combinations()  # This method now handles proper CV and feature selection

        # Optional hyperparameter optimization
        if optimize_hyperparams:
            self.optimize_hyperparameters(n_iter=n_iter)
            self.evaluate_best_model()

        # Visualization and reporting can stay the same as they're for interpretation
        self.visualize_confusion_matrix()
        self.visualize_pca()
        self.visualize_pca_3d()
        self.visualize_feature_importance()
        self.analyze_channel_distribution()
        self.visualize_metrics_comparison()
        self.export_results()

        # ---------------------------------------------
        #   extra sanity-checks / visual diagnostics
        # ---------------------------------------------
        if optimize_hyperparams:
            self.plot_outer_fold_distribution()
            self.permutation_test_best_model(n_perm=1000)

        print("\n---- ANALYSIS COMPLETE ----")
        print(f"Most separable label combination: {self.best_labels}")
        best_model = self.detailed_results[self.best_labels]['best_model']
        metrics = self.detailed_results[self.best_labels]['metrics'][best_model]

        print(f"Best model: {best_model}")
        print(f"Performance metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        print(f"\nTop {self.n_features_to_select} informational features:")
        for i, feature in enumerate(self.selected_features):
            print(f"  {i + 1}. {feature}")

        print("\nNOTE: The actual features used in model evaluation were selected")
        print("independently within each cross-validation fold to prevent information leakage.")
        self.save_best_models()  # <-- New addition


        # Add hyperparameter optimization results if applicable
        if optimize_hyperparams and hasattr(self, 'optimization_results'):
            print("\n---- HYPERPARAMETER OPTIMIZATION RESULTS ----")
            best_model_name = self.optimization_results['best_model_name']
            best_score = self.optimization_results['best_score']
            print(f"Best optimized model: {best_model_name}")
            print(f"Best optimized score (F1): {best_score:.4f}")
            print(f"Optimized model saved to: {self.run_directory}/best_{best_model_name.lower()}_model.pkl")

# STEP 3: Update the main script to accept a hyperparameter optimization flag
if __name__ == "__main__":
    import argparse
    import time

    # parser.add_argument('--features_file', type=str, default="data/normalized_merges/14_full_o1_comBat/normalized_imagery.csv",
    # parser.add_argument('--features_file', type=str, default="data/merged_features/14_sessions_merge_1743884222/59_balanced_o1.csv",


    start_time=time.time()
    parser = argparse.ArgumentParser(description='EEG Data Analysis Tool')
    parser.add_argument('--features_file', type=str, default="data/normalized_merges/14_all_channels/tp7_norm-ComBat.csv",
                        help='Path to the CSV file containing EEG features')
    parser.add_argument('--top_n_labels', type=int, default=2,
                        help='Number of labels to analyze (default: 2)')
    parser.add_argument('--n_features', type=int, default=10,
                        help='Number of top features to select (default: 8)')
    parser.add_argument('--channel_approach', type=str, default="pooled",
                        choices=["pooled", "separate", "features"],
                        help='How to handle channel data (default: pooled)')
    parser.add_argument('--cv_method', type=str, default="loso",
                        choices=["loo", "kfold", "holdout", 'loso'],
                        help='Cross-validation method (default: loo)')
    parser.add_argument('--cv_version', type=str, default='simple', choices=['extended', 'simple'],help='Cross-validation method (default: extended)')
    parser.add_argument('--kfold_splits', type=int, default=7,
                        help='Number of splits for k-fold cross-validation (default: 5)')
    parser.add_argument('--test_size', type=float, default=0.25,
                        help='Test set size for holdout validation (default: 0.2)')
    parser.add_argument('--optimize', action='store_true',
                        help='Perform hyperparameter optimization')
    parser.add_argument('--n_iter', type=int, default=40,
                        help='Number of iterations for hyperparameter optimization (default: 25)')

    args = parser.parse_args()

    analyzer = EEGAnalyzer(
        features_file=args.features_file,
        top_n_labels=args.top_n_labels,
        n_features_to_select=args.n_features,
        channel_approach=args.channel_approach,
        cv_method=args.cv_method,
        cv_version=args.cv_version,
        kfold_splits=args.kfold_splits,
        test_size=args.test_size
    )

    analyzer.run_complete_analysis(optimize_hyperparams=args.optimize, n_iter=args.n_iter)
    print(f'Analysis took  {(time.time() - start_time):.2f} seconds')