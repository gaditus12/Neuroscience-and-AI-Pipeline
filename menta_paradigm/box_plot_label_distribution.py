import pandas as pd
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import f_classif


class DataVisualizer:
    def __init__(self,
                 features_file,
                 feature_ranking_file=None,
                 top_n_features=10,
                 feature_output_file="feature_distributions.png",
                 label_output_file="label_distribution.png"):
        """
        Initialize with file paths and visualization parameters.

        :param features_file: Path to CSV file containing merged features (including a 'label' column)
        :param feature_ranking_file: Optional path to CSV file with a "feature" column for ranking.
                                     If None, features are ranked dynamically.
        :param top_n_features: Number of top features to visualize.
        :param feature_output_file: Filename for the feature distribution plot.
        :param label_output_file: Filename for the label distribution plot.
        """
        self.features_file = features_file
        self.feature_ranking_file = feature_ranking_file
        self.top_n_features = top_n_features
        self.feature_output_file = feature_output_file
        self.label_output_file = label_output_file

        self.df = None
        self.top_features = None

    def load_data(self):
        """Load the merged features dataset."""
        print(f"Loading data from {self.features_file}...")
        self.df = pd.read_csv(self.features_file)
        print(f"Data loaded with {len(self.df)} rows and columns: {list(self.df.columns)}")

    def rank_features(self):
        """
        Determine the top features.

        If a feature ranking file is provided, load the ranking and pick the top_n_features.
        Otherwise, dynamically rank features using an ANOVA F-test.
        """
        if self.feature_ranking_file:
            print(f"Loading feature ranking from {self.feature_ranking_file}...")
            df_importance = pd.read_csv(self.feature_ranking_file)
            self.top_features = df_importance["feature"].head(self.top_n_features).tolist()
            print("Top features from file:", self.top_features)
        else:
            print("No feature ranking file provided. Ranking features dynamically using ANOVA F-test...")
            # Exclude non-numeric columns from features.
            feature_columns = [col for col in self.df.columns if col not in ["label", "channel", "session"]]
            X = self.df[feature_columns]
            y = self.df["label"]
            # Encode the labels as integers using pandas factorize.
            y_encoded, uniques = pd.factorize(y)
            f_scores, p_values = f_classif(X, y_encoded)
            # Build a DataFrame of features and their F-scores.
            feature_scores = pd.DataFrame({
                "feature": feature_columns,
                "f_score": f_scores
            })
            feature_scores.sort_values("f_score", ascending=False, inplace=True)
            self.top_features = feature_scores["feature"].head(self.top_n_features).tolist()
            print("Top features determined dynamically:", self.top_features)

    def plot_label_distribution(self):
        """Plot and save the label distribution."""
        print("Plotting label distribution...")
        plt.figure(figsize=(10, 5))
        sns.countplot(x=self.df["label"], palette="viridis")
        plt.xticks(rotation=45)
        plt.xlabel("Label")
        plt.ylabel("Count")
        plt.title("Label Distribution in the Dataset")
        plt.tight_layout()
        plt.savefig(self.label_output_file, dpi=300)
        print(f"Label distribution plot saved as: {self.label_output_file}")
        plt.close()

    def plot_feature_distribution(self):
        """Plot and save the distribution of top features across labels."""
        print("Plotting feature distributions for top features...")
        plt.figure(figsize=(15, 10))
        for i, feature in enumerate(self.top_features, 1):
            plt.subplot((self.top_n_features + 1) // 2, 2, i)
            sns.boxplot(x=self.df["label"], y=self.df[feature],
                        hue=self.df["label"], palette="viridis", legend=False)
            plt.xticks(rotation=45)
            plt.xlabel("Label")
            plt.ylabel(feature)
            plt.title(f"Distribution of {feature} across Labels")
        plt.tight_layout()
        plt.savefig(self.feature_output_file, dpi=300)
        print(f"Feature distribution plot saved as: {self.feature_output_file}")
        plt.close()

    def save_top_features(self):
        with open (f"top_features_box_plots/{self.top_n_features}_features_{self.features_file.split('.csv')[0].split('/')[-1]+'.txt'}", 'w') as f:
            for i in self.top_features:
                f.write(f'{i}\n')

    def run_all(self):
        """Execute all steps: load data, rank features, and generate plots."""
        self.load_data()
        self.rank_features()
        self.plot_label_distribution()
        self.plot_feature_distribution()
        self.save_top_features()

# Example usage:
if __name__ == "__main__":
    features_file_dir = "data/final_sets/all_channels_binary"
    multiple=False
    if multiple:
        for i in range (1, 5):
            features_file=features_file_dir+ f"o1_sess_{i}.csv"
            top_n_features = 10
            visualizer = DataVisualizer(
                features_file=features_file,
                #feature_ranking_file="data/merged_features/merge_run_1741541389/imagery_task_feature_ranking.csv",
                feature_ranking_file=None,
                # Set to None for dynamic ranking
                top_n_features=top_n_features,
                feature_output_file=f"top_features_box_plots/{top_n_features}_feature_distributions_{features_file.split('.csv')[0].split('/')[-1]}.png",
                label_output_file=f"top_features_box_plots/{top_n_features}_label_distribution_{features_file.split('.csv')[0].split('/')[-1]}.png"
            )
            visualizer.run_all()
    else:
        features_file = features_file_dir + f"/o2_comBat.csv"
        visualizer = DataVisualizer(
            features_file=features_file,
            # feature_ranking_file="data/merged_features/merge_run_1741541389/imagery_task_feature_ranking.csv",
            feature_ranking_file= "ml_model_outputs/o2_comBat_loso_1.0k_run_1746011121/best_model_feature_importance.csv",
            # Set to None for dynamic ranking
            top_n_features=4,
            feature_output_file=f"top_features_box_plots/feature_distributions_{features_file.split('.csv')[0].split('/')[-1]}.png",
            label_output_file=f"top_features_box_plots/label_distribution_{features_file.split('.csv')[0].split('/')[-1]}.png"
        )
        visualizer.run_all()