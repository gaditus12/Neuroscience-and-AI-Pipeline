import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score
from sklearn.decomposition import PCA
import warnings
import os

warnings.filterwarnings('ignore')


class EEGFeatureSelector:
    """
    A class for selecting the most informative features from EEG data using multiple methods
    and evaluating their discriminative power.
    """

    def __init__(self, data_path, n_features=30, cv_folds=5, random_state=42, output_file='top_eeg_features.csv'):
        """
        Initialize the EEG feature selector.

        Args:
            data_path (str): Path to the CSV file containing the EEG data
            n_features (int): Number of top features to select
            cv_folds (int): Number of cross-validation folds for stability analysis
            random_state (int): Random seed for reproducibility
            output_file (str): Path to save the output CSV file with top features
        """
        self.data_path = data_path
        self.n_features = n_features
        self.cv_folds = cv_folds
        self.random_state = random_state
        self.output_file = output_file
        self.df = None
        self.X = None
        self.y = None
        self.feature_names = None
        self.non_feature_columns = ['label', 'channel', 'session']

    def load_data(self):
        """Load and prepare the EEG data"""
        print(f"Loading data from {self.data_path}...")
        self.df = pd.read_csv(self.data_path)

        # Identify non-feature columns that exist in the dataset
        self.existing_non_feature_columns = [col for col in self.non_feature_columns if col in self.df.columns]

        # Exclude non-feature columns
        self.X = self.df.drop(self.existing_non_feature_columns, axis=1, errors='ignore')
        self.feature_names = self.X.columns

        # Get the label column (assuming it always exists)
        if 'label' in self.df.columns:
            self.y = self.df['label']
            print(f"Class distribution: {self.y.value_counts().to_dict()}")
        else:
            raise ValueError("The 'label' column is required but not found in the dataset")

        print(f"Data loaded: {self.X.shape[0]} samples with {self.X.shape[1]} features")

        return self

    def calculate_anova_importance(self):
        """Calculate feature importance using ANOVA F-value"""
        print("Calculating ANOVA F-test importance...")

        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)

        # Calculate ANOVA F-values
        selector = SelectKBest(score_func=f_classif, k='all')
        selector.fit(X_scaled, self.y)

        # Create importance dataframe
        anova_df = pd.DataFrame({
            'feature': self.feature_names,
            'f_value': selector.scores_,
            'p_value': selector.pvalues_
        })

        # Sort by importance (higher F-value = more important)
        anova_df = anova_df.sort_values('f_value', ascending=False).reset_index(drop=True)

        # Add normalized importance (0-100 scale)
        if anova_df['f_value'].max() > 0:
            anova_df['importance'] = 100.0 * anova_df['f_value'] / anova_df['f_value'].max()
        else:
            anova_df['importance'] = 0

        return anova_df

    def calculate_mutual_info_importance(self):
        """Calculate feature importance using mutual information"""
        print("Calculating mutual information importance...")

        # Scale features (recommended for mutual information)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)

        # Calculate mutual information
        mi_scores = mutual_info_classif(X_scaled, self.y, random_state=self.random_state)

        # Create importance dataframe
        mi_df = pd.DataFrame({
            'feature': self.feature_names,
            'mutual_info': mi_scores
        })

        # Sort by importance (higher MI = more important)
        mi_df = mi_df.sort_values('mutual_info', ascending=False).reset_index(drop=True)

        # Add normalized importance (0-100 scale)
        if mi_df['mutual_info'].max() > 0:
            mi_df['importance'] = 100.0 * mi_df['mutual_info'] / mi_df['mutual_info'].max()
        else:
            mi_df['importance'] = 0

        return mi_df

    def calculate_random_forest_importance(self):
        """Calculate feature importance using Random Forest"""
        print("Calculating Random Forest importance...")

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)

        # Train Random Forest
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=3,
            class_weight='balanced',
            random_state=self.random_state
        )
        rf.fit(X_scaled, self.y)

        # Get feature importances
        importances = rf.feature_importances_

        # Create importance dataframe
        rf_df = pd.DataFrame({
            'feature': self.feature_names,
            'rf_importance': importances
        })

        # Sort by importance (higher = more important)
        rf_df = rf_df.sort_values('rf_importance', ascending=False).reset_index(drop=True)

        # Add normalized importance (0-100 scale)
        if rf_df['rf_importance'].max() > 0:
            rf_df['importance'] = 100.0 * rf_df['rf_importance'] / rf_df['rf_importance'].max()
        else:
            rf_df['importance'] = 0

        return rf_df

    def calculate_svm_importance(self):
        """Calculate feature importance using SVM coefficients (for linear kernel)"""
        print("Calculating SVM-based importance...")

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)

        # Create and train a linear SVM
        svm = SVC(kernel='linear', class_weight='balanced', random_state=self.random_state)
        svm.fit(X_scaled, self.y)

        # Get feature importances (absolute coefficient values)
        importances = np.abs(svm.coef_[0])

        # Create importance dataframe
        svm_df = pd.DataFrame({
            'feature': self.feature_names,
            'svm_importance': importances
        })

        # Sort by importance (higher = more important)
        svm_df = svm_df.sort_values('svm_importance', ascending=False).reset_index(drop=True)

        # Add normalized importance (0-100 scale)
        if svm_df['svm_importance'].max() > 0:
            svm_df['importance'] = 100.0 * svm_df['svm_importance'] / svm_df['svm_importance'].max()
        else:
            svm_df['importance'] = 0

        return svm_df

    def calculate_stability_importance(self):
        """
        Calculate feature stability across multiple folds of the data
        using ANOVA F-test to measure how consistently features are selected
        """
        print("Calculating feature stability across folds...")

        # Scale features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(self.X)

        # Create StratifiedKFold object
        skf = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)

        # Initialize dictionary to store feature ranks in each fold
        feature_ranks = {feature: [] for feature in self.feature_names}

        # Perform cross-validation
        for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X_scaled, self.y)):
            X_train = X_scaled[train_idx]
            y_train = self.y.iloc[train_idx]

            # Apply ANOVA F-test
            selector = SelectKBest(score_func=f_classif, k='all')
            selector.fit(X_train, y_train)

            # Get feature scores and their ranks
            scores = selector.scores_
            ranks = np.argsort(-scores)  # Sort indices by descending order of scores

            # Store the rank of each feature in this fold
            for rank, feature_idx in enumerate(ranks):
                feature_name = self.feature_names[feature_idx]
                feature_ranks[feature_name].append(rank)

        # Calculate mean rank and rank stability for each feature
        stability_data = []
        for feature, ranks in feature_ranks.items():
            mean_rank = np.mean(ranks)
            rank_std = np.std(ranks)
            stability_score = 1.0 / (1.0 + rank_std)  # Higher stability = lower std deviation

            stability_data.append({
                'feature': feature,
                'mean_rank': mean_rank,
                'rank_std': rank_std,
                'stability_score': stability_score
            })

        # Create and sort dataframe
        stability_df = pd.DataFrame(stability_data)
        stability_df = stability_df.sort_values(['mean_rank', 'rank_std'],
                                                ascending=[True, True]).reset_index(drop=True)

        # Add normalized importance (0-100 scale)
        max_stability = stability_df['stability_score'].max()
        if max_stability > 0:
            stability_df['importance'] = 100.0 * stability_df['stability_score'] / max_stability
        else:
            stability_df['importance'] = 0

        return stability_df

    def calculate_ensemble_importance(self, method_dfs, weights=None):
        """
        Calculate ensemble importance by combining multiple methods

        Args:
            method_dfs (dict): Dictionary of dataframes with feature importance from different methods
            weights (dict, optional): Dictionary with method weights. Defaults to equal weights.
        """
        print("Calculating ensemble feature importance...")

        # Default to equal weights if not provided
        if weights is None:
            weights = {method: 1.0 for method in method_dfs.keys()}

        # Normalize weights to sum to 1.0
        total_weight = sum(weights.values())
        norm_weights = {k: v / total_weight for k, v in weights.items()}

        # Create mappings of feature to normalized score for each method
        feature_scores = {}

        for method, df in method_dfs.items():
            score_map = dict(zip(df['feature'], df['importance'] / 100.0))  # Convert to 0-1 scale

            for feature, score in score_map.items():
                if feature not in feature_scores:
                    feature_scores[feature] = {}
                feature_scores[feature][method] = score

        # Create ensemble scores
        ensemble_data = []

        for feature, method_scores in feature_scores.items():
            # Get weighted average score
            ensemble_score = 0.0
            for method, score in method_scores.items():
                method_weight = norm_weights.get(method, 0.0)
                ensemble_score += score * method_weight

            # Add method-specific scores
            feature_data = {'feature': feature, 'ensemble_score': ensemble_score}
            for method in method_dfs.keys():
                feature_data[f"{method}_score"] = method_scores.get(method, 0.0)

            ensemble_data.append(feature_data)

        # Create dataframe and sort by ensemble score
        ensemble_df = pd.DataFrame(ensemble_data)
        ensemble_df = ensemble_df.sort_values('ensemble_score', ascending=False).reset_index(drop=True)

        # Add normalized importance (0-100 scale)
        ensemble_df['importance'] = 100.0 * ensemble_df['ensemble_score']

        return ensemble_df

    def evaluate_feature_subset(self, features, n_cv=5):
        """
        Evaluate a subset of features using cross-validation with SVM and Random Forest

        Args:
            features (list): List of feature names to evaluate
            n_cv (int): Number of cross-validation folds

        Returns:
            dict: Dictionary with evaluation metrics
        """
        print(f"Evaluating performance with {len(features)} features...")

        # Scale features
        scaler = StandardScaler()
        X_subset = self.X[features]
        X_scaled = scaler.fit_transform(X_subset)

        # Create models
        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=3,
            class_weight='balanced',
            random_state=self.random_state
        )

        svm = SVC(
            kernel='rbf',
            C=1.0,
            gamma='scale',
            class_weight='balanced',
            probability=True,
            random_state=self.random_state
        )

        # Evaluate with cross-validation
        cv = StratifiedKFold(n_splits=n_cv, shuffle=True, random_state=self.random_state)

        rf_scores = cross_val_score(rf, X_scaled, self.y, cv=cv, scoring='accuracy')
        svm_scores = cross_val_score(svm, X_scaled, self.y, cv=cv, scoring='accuracy')

        # Calculate additional metrics for SVM
        svm_auc_scores = []

        for train_idx, test_idx in cv.split(X_scaled, self.y):
            X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
            y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]

            svm.fit(X_train, y_train)
            svm_probs = svm.predict_proba(X_test)[:, 1]

            try:
                auc = roc_auc_score(y_test, svm_probs)
                svm_auc_scores.append(auc)
            except Exception:
                # Handle case where a fold might only have one class
                svm_auc_scores.append(0.5)

        # Return evaluation results
        return {
            'rf_accuracy_mean': rf_scores.mean(),
            'rf_accuracy_std': rf_scores.std(),
            'svm_accuracy_mean': svm_scores.mean(),
            'svm_accuracy_std': svm_scores.std(),
            'svm_auc_mean': np.mean(svm_auc_scores),
            'svm_auc_std': np.std(svm_auc_scores),
            'n_features': len(features)
        }

    def analyze_channel_distributions(self, top_features, n_top=30):
        """
        Analyze the distribution of channels in the top features

        Args:
            top_features (list): List of feature names
            n_top (int): Number of top features to analyze

        Returns:
            dict: Dictionary with channel distribution statistics
        """
        print("Analyzing channel distribution in top features...")

        # Limit to the top N features
        features = top_features[:n_top]

        # Extract channel from feature name (assumes format like "feature_channel")
        channel_counts = {}
        feature_types = {}

        for feature in features:
            parts = feature.split('_')
            if len(parts) > 1:
                # The last part should be the channel
                channel = parts[-1]

                # The rest is the feature type
                feature_type = '_'.join(parts[:-1])

                # Update counts
                channel_counts[channel] = channel_counts.get(channel, 0) + 1

                if feature_type not in feature_types:
                    feature_types[feature_type] = []
                feature_types[feature_type].append(channel)

        # Sort by count
        sorted_channels = sorted(channel_counts.items(), key=lambda x: x[1], reverse=True)
        sorted_features = sorted(feature_types.items(), key=lambda x: len(x[1]), reverse=True)

        return {
            'channel_counts': dict(sorted_channels),
            'feature_types': {k: sorted(v) for k, v in sorted_features}
        }

    def plot_feature_importance(self, df, title, n_features=30):
        """
        Plot feature importance for the top features

        Args:
            df (pd.DataFrame): DataFrame with feature importance
            title (str): Plot title
            n_features (int): Number of top features to plot
        """
        # Get top N features
        plot_df = df.head(n_features).copy()

        # Create plot
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=plot_df)
        plt.title(f"{title} (Top {n_features} Features)")
        plt.xlabel('Importance Score')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()

    def save_reduced_dataset(self, top_features, output_file=None):
        """
        Save a reduced dataset with only the top features plus non-feature columns

        Args:
            top_features (list): List of feature names to include
            output_file (str, optional): Path to save the file. If None, uses self.output_file

        Returns:
            str: Path to the saved file
        """
        if output_file is None:
            output_file = self.output_file

        # Get directory path and create if it doesn't exist
        output_dir = os.path.dirname(output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # Create file name with _reduced suffix if not already specified
        base, ext = os.path.splitext(output_file)
        if '_reduced' not in base:
            output_file = f"{base}_reduced{ext}"

        # Create a reduced DataFrame with non-feature columns and top features
        columns_to_keep = self.existing_non_feature_columns + top_features
        reduced_df = self.df[columns_to_keep]

        # Save to CSV
        reduced_df.to_csv(output_file, index=False)
        print(f"Reduced dataset with {len(top_features)} features saved to '{output_file}'")

        return output_file

    def run_complete_analysis(self):
        """Run a complete feature selection analysis and return the top features"""
        # Load data
        self.load_data()

        # Calculate importance scores using different methods
        anova_df = self.calculate_anova_importance()
        mi_df = self.calculate_mutual_info_importance()
        rf_df = self.calculate_random_forest_importance()
        svm_df = self.calculate_svm_importance()
        stability_df = self.calculate_stability_importance()

        # Combine methods with custom weights
        method_dfs = {
            'anova': anova_df,
            'mutual_info': mi_df,
            'random_forest': rf_df,
            'svm': svm_df,
            'stability': stability_df
        }

        # Weights reflect the relative importance of each method
        weights = {
            'anova': 0.2,
            'mutual_info': 0.2,
            'random_forest': 0.25,
            'svm': 0.15,
            'stability': 0.2
        }

        # Calculate ensemble importance
        ensemble_df = self.calculate_ensemble_importance(method_dfs, weights)

        # Plot top features for each method
        self.plot_feature_importance(anova_df, 'ANOVA F-Test Importance')
        self.plot_feature_importance(mi_df, 'Mutual Information Importance')
        self.plot_feature_importance(rf_df, 'Random Forest Importance')
        self.plot_feature_importance(svm_df, 'SVM Coefficient Importance')
        self.plot_feature_importance(stability_df, 'Feature Stability Importance')
        self.plot_feature_importance(ensemble_df, 'Ensemble Feature Importance')

        # Get top features from ensemble method
        top_features = ensemble_df['feature'].head(self.n_features).tolist()

        # Analyze channel distribution
        channel_analysis = self.analyze_channel_distributions(top_features)

        print("\nChannel distribution in top features:")
        for channel, count in channel_analysis['channel_counts'].items():
            print(f"  - {channel}: {count} features")

        print("\nFeature type distribution:")
        for feature_type, channels in channel_analysis['feature_types'].items():
            print(f"  - {feature_type}: {len(channels)} channels ({', '.join(channels)})")

        # Evaluate performance with selected features
        evaluation = self.evaluate_feature_subset(top_features)

        print("\nPerformance evaluation with selected features:")
        print(f"  - RF Accuracy: {evaluation['rf_accuracy_mean']:.4f} ± {evaluation['rf_accuracy_std']:.4f}")
        print(f"  - SVM Accuracy: {evaluation['svm_accuracy_mean']:.4f} ± {evaluation['svm_accuracy_std']:.4f}")
        print(f"  - SVM AUC: {evaluation['svm_auc_mean']:.4f} ± {evaluation['svm_auc_std']:.4f}")

        # Save top features info to CSV
        top_features_df = ensemble_df.head(self.n_features)
        features_info_file = f"{os.path.splitext(self.output_file)[0]}_info.csv"
        top_features_df.to_csv(features_info_file, index=False)
        print(f"\nTop {self.n_features} features info saved to '{features_info_file}'")

        # Save reduced dataset with only top features
        reduced_file = self.save_reduced_dataset(top_features)

        return {
            'top_features': top_features,
            'ensemble_df': ensemble_df,
            'channel_analysis': channel_analysis,
            'evaluation': evaluation,
            'method_dfs': method_dfs,
            'reduced_file': reduced_file
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='EEG Feature Selection Tool')
    parser.add_argument('--data_file', type=str,
                        default='data/merged_features/32_captures/combined_eeg_data_merged_features_imagery_task.csv',
                        help='Path to the CSV file containing combined EEG feature data')
    parser.add_argument('--n_features', type=int, default=30,
                        help='Number of top features to select (default: 30)')
    parser.add_argument('--cv_folds', type=int, default=5,
                        help='Number of cross-validation folds (default: 5)')
    parser.add_argument('--output_file', type=str, default='top_eeg_features.csv',
                        help='Output file for selected features (default: top_eeg_features.csv)')

    args = parser.parse_args()

    # Initialize and run the selector
    selector = EEGFeatureSelector(
        data_path=args.data_file,
        n_features=args.n_features,
        cv_folds=args.cv_folds,
        output_file=args.output_file
    )

    results = selector.run_complete_analysis()

    print(f"\nFeature selection complete. Selected {len(results['top_features'])} features.")
    print(f"Reduced dataset saved to: {results['reduced_file']}")