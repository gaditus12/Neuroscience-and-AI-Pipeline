import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import LeaveOneOut, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
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
                 channel_approach="pooled",   # Options: "pooled", "separate", "features"
                 cv_method="loo",             # Options: "loo", "kfold"
                 kfold_splits=5):
        # Configuration
        self.features_file = features_file
        self.top_n_labels = top_n_labels
        self.n_features_to_select = n_features_to_select
        self.channel_approach = channel_approach.lower()
        self.cv_method = cv_method.lower()
        self.kfold_splits = kfold_splits

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
        self.selector = None  # Dummy selector to be created after feature selection
        self.separability_scores = {}
        self.detailed_results = {}
        self.best_labels = None

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
                print("Error: This approach requires a 'session' column to identify unique recordings. Falling back to POOLED.")
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
                print(f"Warning: Only {len(available_top_features)} of the top {self.n_features_to_select} features are available.")
                missing_count = self.n_features_to_select - len(available_top_features)
                print(f"Supplementing with {missing_count} additional features using ANOVA F-test")
                # Use all features for ANOVA scoring
                selector_anova = SelectKBest(f_classif, k='all')
                selector_anova.fit(self.X_scaled, self.y)
                f_values = selector_anova.scores_
                feature_f_scores = pd.DataFrame({'feature': self.feature_columns, 'f_score': f_values})
                feature_f_scores = feature_f_scores[~feature_f_scores['feature'].isin(available_top_features)]
                additional_features = feature_f_scores.sort_values('f_score', ascending=False).head(missing_count)['feature'].tolist()
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
        # Choose CV method based on parameter
        if self.cv_method == "loo":
            cv = LeaveOneOut()
            print("Using Leave-One-Out cross validation")
        elif self.cv_method == "kfold":
            cv = StratifiedKFold(n_splits=self.kfold_splits)
            print(f"Using Stratified K-Fold cross validation with {self.kfold_splits} splits")
        else:
            raise ValueError("Invalid cv_method provided. Choose 'loo' or 'kfold'.")

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

            # Evaluate with two models
            models = {
                'RandomForest': RandomForestClassifier(n_estimators=50, max_depth=5,
                                                         class_weight='balanced', random_state=42),
                'SVM': SVC(kernel='rbf', C=1.0, gamma='scale',
                           class_weight='balanced', random_state=42)
            }
            model_scores = {}
            for name, model in models.items():
                cv_scores = cross_val_score(model, X_subset_selected, y_subset, cv=cv, scoring='accuracy')
                model_scores[name] = np.mean(cv_scores)
                print(f"  {name} accuracy: {model_scores[name]:.4f}")

            best_model = max(model_scores, key=model_scores.get)
            best_score = model_scores[best_model]
            self.separability_scores[label_combo] = best_score
            self.detailed_results[label_combo] = {
                'best_model': best_model,
                'scores': model_scores,
                'n_samples': len(y_subset),
                'sample_counts': combo_sample_count.to_dict()
            }
        self.best_labels = max(self.separability_scores, key=self.separability_scores.get)
        print(f"\nThe {self.top_n_labels} most diverse labels are: {self.best_labels}")
        print(f"Separation Score: {self.separability_scores[self.best_labels]:.4f}")
        print(f"Best model: {self.detailed_results[self.best_labels]['best_model']}")
        print(f"Total samples: {self.detailed_results[self.best_labels]['n_samples']}")

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

        plt.title("PCA Visualization of EEG Features\nChannel markers show distribution of readings", fontsize=16, pad=20)
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

        # Add text about the best label combination
        plt.figtext(0.5, 0.01,
                    f"Most Separable Labels: {', '.join(self.best_labels)}\n"
                    f"Separation Score: {self.separability_scores[self.best_labels]:.4f} using {self.detailed_results[self.best_labels]['best_model']}",
                    ha="center", fontsize=14,
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="black", alpha=0.8))
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        output_file = f"{self.channel_approach}_eeg_pooled_visualization_top{self.top_n_labels}_labels_top{self.n_features_to_select}_features_{self.cv_method}.png"
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
        importance_file = f"{self.channel_approach}_feature_importance_top{self.top_n_labels}_labels_top{self.n_features_to_select}_features_{self.cv_method}.png"
        plt.savefig(importance_file, dpi=300, bbox_inches='tight')
        print(f"Feature importance visualization saved as '{importance_file}'")
        plt.close()

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
            channel_file = "channel_distribution.png"
            plt.savefig(channel_file, dpi=300, bbox_inches='tight')
            print(f"Channel distribution saved as '{channel_file}'")
            plt.close()

    def run_all(self):
        self.load_data()
        self.preprocess_data()
        self.feature_selection()
        self.evaluate_label_combinations()
        self.visualize_pca()
        self.visualize_feature_importance()
        self.analyze_channel_distribution()


# Example usage:
if __name__ == "__main__":
    analyzer = EEGAnalyzer(
        features_file="data/merged_features/baseline_imagery_instruction_merge/merged_features_threezones.csv",
        top_n_labels=2,
        n_features_to_select=5,
        channel_approach="pooled",
        cv_method="loo",    # Change to "loo" for Leave-One-Out CV
        kfold_splits=10
    )
    analyzer.run_all()
