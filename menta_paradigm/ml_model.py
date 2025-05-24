import os
import time

import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, cross_val_score, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, classification_report
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import matplotlib.transforms as transforms



class EEGAnalyzer:
    def __init__(self,
                 features_file,
                 top_n_labels=2,
                 n_features_to_select=15,
                 channel_approach="pooled",  # Options: "pooled", "separate", "features"
                 cv_method="loo",  # Options: "loo", "kfold", "holdout"
                 kfold_splits=5,
                 test_size=0.2):  # For holdout validation
        # Configuration
        self.features_file = features_file
        self.top_n_labels = top_n_labels
        self.n_features_to_select = n_features_to_select
        self.channel_approach = channel_approach.lower()
        self.cv_method = cv_method.lower()
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

    def preprocess_data(self):
        print("\n---- PREPROCESSING DATA ----")
        # Standardize features
        self.X_scaled = self.scaler.fit_transform(self.X)
        self.X_scaled_df = pd.DataFrame(self.X_scaled, columns=self.feature_columns)
        print("Features standardized.")

    def feature_selection(self):
        print("\n---- FEATURE SELECTION ----")
        try:
            # Try to load external feature importance rankings (update the file path as needed)
            importance_df = pd.read_csv("data/merged_features/merge_run_1741541389/imagery_task_feature_ranking.csv")
            print(f"Loaded feature importance rankings for {len(importance_df)} features")
            importance_df = importance_df.sort_values(by='ensemble_score', ascending=False)
            top_features = importance_df.head(self.n_features_to_select)['feature'].tolist()
            print("Top features based on ensemble score:")
            for i, feature in enumerate(top_features):
                if feature in self.feature_columns:
                    score = importance_df[importance_df['feature'] == feature]['ensemble_score'].values[0]
                    print(f"  {i + 1}. {feature} (ensemble score: {score:.4f})")
                else:
                    print(f"  {i + 1}. {feature} - WARNING: Not found in current dataset")

            # Check which of the top features are available
            available_top_features = [f for f in top_features if f in self.feature_columns]
            if len(available_top_features) < self.n_features_to_select:
                print(
                    f"Warning: Only {len(available_top_features)} of the top {self.n_features_to_select} features are available.")
                missing_count = self.n_features_to_select - len(available_top_features)
                print(f"Supplementing with {missing_count} additional features using ANOVA F-test")
                # Use all features for ANOVA scoring
                selector_anova = SelectKBest(f_classif, k='all')
                selector_anova.fit(self.X_scaled, self.y)
                f_values = selector_anova.scores_
                feature_f_scores = pd.DataFrame({'feature': self.feature_columns, 'f_score': f_values})
                feature_f_scores = feature_f_scores[~feature_f_scores['feature'].isin(available_top_features)]
                additional_features = feature_f_scores.sort_values('f_score', ascending=False).head(missing_count)[
                    'feature'].tolist()
                available_top_features.extend(additional_features)
                print("Additional features selected via ANOVA:")
                for i, feature in enumerate(additional_features):
                    f_score = feature_f_scores[feature_f_scores['feature'] == feature]['f_score'].values[0]
                    print(f"  {len(top_features) + i + 1}. {feature} (F-score: {f_score:.4f})")
            self.selected_features = available_top_features
            self.X_selected = self.X_scaled_df[self.selected_features].values

        except Exception as e:
            print(f"Could not load feature importance file: {str(e)}")
            print("Falling back to standard ANOVA F-test feature selection")
            selector = SelectKBest(f_classif, k=self.n_features_to_select)
            self.X_selected = selector.fit_transform(self.X_scaled, self.y)
            selected_indices = selector.get_support(indices=True)
            self.selected_features = [self.feature_columns[i] for i in selected_indices]
            print("Selected features:")
            for i, feature in enumerate(self.selected_features):
                print(f"  {i + 1}. {feature}")

        # Create a dummy selector so that downstream calls work as expected
        self.selector = self.DummySelector(self.feature_columns, self.selected_features)

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

    def evaluate_label_combinations(self):
        print("\n---- EVALUATING LABEL COMBINATIONS ----")
        # Set up models
        models = {
            'RandomForest': RandomForestClassifier(n_estimators=50, max_depth=5,
                                                   class_weight='balanced', random_state=42),
            'SVM': SVC(kernel='rbf', C=1.0, gamma='scale',
                       class_weight='balanced', random_state=42, probability=True)
        }

        # Choose CV method based on parameter
        if self.cv_method == "loo":
            print("Using Leave-One-Out cross validation")
            cv = LeaveOneOut()
            cv_method = self._evaluate_with_cv
        elif self.cv_method == "kfold":
            print(f"Using Stratified K-Fold cross validation with {self.kfold_splits} splits")
            cv = StratifiedKFold(n_splits=self.kfold_splits)
            cv_method = self._evaluate_with_cv
        elif self.cv_method == "holdout":
            print(f"Using holdout validation with test_size={self.test_size}")
            cv = None
            cv_method = self._evaluate_with_holdout
        else:
            raise ValueError("Invalid cv_method provided. Choose 'loo', 'kfold', or 'holdout'.")

        # Iterate over all possible label combinations (pairs or triplets)
        for label_combo in combinations(self.unique_labels, self.top_n_labels):
            df_subset = self.df[self.df["label"].isin(label_combo)]
            X_subset = df_subset[self.feature_columns]
            X_subset_scaled = self.scaler.transform(X_subset)
            X_subset_selected = self.selector.transform(X_subset_scaled)
            y_subset = df_subset["label"]

            # Print sample counts for this combination
            combo_sample_count = df_subset.groupby("label").size()
            print(f"\nLabel combination {label_combo}: {len(df_subset)} samples")
            print(f"  Per label: {combo_sample_count.to_dict()}")

            # Evaluate with models
            results = cv_method(X_subset_selected, y_subset, models, cv)

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
        """Evaluate models with cross-validation"""
        results = {'model_metrics': {}, 'confusion_matrices': {}, 'misclassified_samples': {}}

        for name, model in models.items():
            y_true_all = []
            y_pred_all = []
            misclassified_samples = []  # NEW: Store misclassified sample info

            # Perform cross-validation
            for train_idx, test_idx in cv.split(X, y):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                # Get the original indices from y (they refer to df_subset)
                test_indices = y.iloc[test_idx].index.tolist()

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                y_true_all.extend(y_test.tolist())
                y_pred_all.extend(y_pred)

                # NEW: For each sample, if misclassified, record its index and labels.
                for idx, true_val, pred_val in zip(test_indices, y_test.tolist(), y_pred):
                    if true_val != pred_val:
                        misclassified_samples.append({
                            'index': idx,
                            'true_label': true_val,
                            'predicted_label': pred_val
                        })

            results['model_metrics'][name] = self._calculate_metrics(y_true_all, y_pred_all)
            results['confusion_matrices'][name] = confusion_matrix(y_true_all, y_pred_all)
            results['misclassified_samples'][name] = misclassified_samples

            print(f"\n{name} Classification Report:")
            print(classification_report(y_true_all, y_pred_all))

        return results

    def _evaluate_with_holdout(self, X, y, models, _):
        """Evaluate models with holdout validation"""
        results = {'model_metrics': {}, 'confusion_matrices': {}, 'misclassified_samples': {}}

        # Split data once
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=42, stratify=y)

        # Get the original indices for the test set
        test_indices = y_test.index.tolist()

        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

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
        """Export misclassified sample examples for all label combinations, merging with original features."""
        print("\n---- EXPORTING MISCLASSIFIED SAMPLES ----")
        for combo in self.detailed_results:
            combo_str = '_'.join(combo)
            misclassified_dict = self.detailed_results[combo].get('misclassified_samples', {})
            for model, misclassified in misclassified_dict.items():
                if misclassified:  # Only export if there are misclassified examples
                    mis_df = pd.DataFrame(misclassified)
                    # Merge with original features (using the index)
                    merged = pd.merge(mis_df, self.df.reset_index(), left_on='index', right_on='index', how='left')
                    filename = f"{self.run_directory}/{self.channel_approach}_misclassified_{combo_str}_{model}_{self.cv_method}.csv"
                    merged.to_csv(filename, index=False)
                    print(f"Exported misclassified samples for combination {combo_str} with model {model} to '{filename}'")

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

    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        self.load_data()
        self.preprocess_data()
        self.feature_selection()
        self.evaluate_label_combinations()
        self.visualize_confusion_matrix()
        self.visualize_pca()
        self.visualize_feature_importance()
        self.analyze_channel_distribution()
        self.visualize_metrics_comparison()  # New method
        self.export_results()  # New method

        print("\n---- ANALYSIS COMPLETE ----")
        print(f"Most separable label combination: {self.best_labels}")
        best_model = self.detailed_results[self.best_labels]['best_model']
        metrics = self.detailed_results[self.best_labels]['metrics'][best_model]

        print(f"Best model: {best_model}")
        print(f"Performance metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        print(f"\nTop {self.n_features_to_select} selected features:")
        for i, feature in enumerate(self.selected_features):
            print(f"  {i + 1}. {feature}")

# Example usage of the EEGAnalyzer class
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='EEG Data Analysis Tool')
    parser.add_argument('--features_file', type=str, default="data/final_sets/all_channels_binary/no_leak/final_final_set/6_sessions_fz.csv",
                        help='Path to the CSV file containing EEG features')
    parser.add_argument('--top_n_labels', type=int, default=6,
                        help='Number of labels to analyze (default: 2)')
    parser.add_argument('--n_features', type=int, default=30,
                        help='Number of top features to select (default: 10)')
    parser.add_argument('--channel_approach', type=str, default="pooled",
                        choices=["pooled", "separate", "features"],
                        help='How to handle channel data (default: pooled)')
    parser.add_argument('--cv_method', type=str, default="kfold",
                        choices=["loo", "kfold", "holdout"],
                        help='Cross-validation method (default: loo)')
    parser.add_argument('--kfold_splits', type=int, default=10,
                        help='Number of splits for k-fold cross-validation (default: 5)')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test set size for holdout validation (default: 0.2)')

    args = parser.parse_args()

    analyzer = EEGAnalyzer(
        features_file=args.features_file,
        top_n_labels=args.top_n_labels,
        n_features_to_select=args.n_features,
        channel_approach=args.channel_approach,
        cv_method=args.cv_method,
        kfold_splits=args.kfold_splits,
        test_size=args.test_size
    )

    analyzer.run_complete_analysis()