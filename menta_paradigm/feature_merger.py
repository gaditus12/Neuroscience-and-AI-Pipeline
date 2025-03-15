import os
import pandas as pd
import numpy as np
import mne
import time
import argparse
from tqdm import tqdm
from scipy.stats import skew, kurtosis, entropy, f_oneway
from mne.time_frequency import psd_array_welch
import math
from pywt import wavedec
from feature_extractor import process_directory_structure
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------
# Feature Importance Calculator Class
# ---------------------------
class EEGFeatureImportance:
    def __init__(self, merged_features_dir):
        """
        Initializes the feature importance calculator.

        Args:
            merged_features_dir (str): Directory containing merged feature files
        """
        self.merged_features_dir = merged_features_dir
        self.results_dir = os.path.join(merged_features_dir, 'feature_importance')
        os.makedirs(self.results_dir, exist_ok=True)

    def load_merged_data(self, zone):
        """
        Load merged data for a specific zone.

        Args:
            zone (str): Zone name (imagery_task, instruction, baseline_post)

        Returns:
            pd.DataFrame: Loaded dataframe or None if file not found
        """
        filepath = os.path.join(self.merged_features_dir, f"merged_features_{zone}.csv")
        if os.path.exists(filepath):
            try:
                return pd.read_csv(filepath)
            except Exception as e:
                print(f"Error loading {filepath}: {str(e)}")
                return None
        else:
            print(f"File not found: {filepath}")
            return None

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

    def analyze_zone(self, zone):
        """
        Analyze feature importance for a specific zone.

        Args:
            zone (str): Zone name (imagery_task, instruction, baseline_post)

        Returns:
            dict: Dictionary with results for different methods
        """
        # Load merged data
        df = self.load_merged_data(zone)
        if df is None or len(df) == 0:
            print(f"No data available for zone: {zone}")
            return None

        print(f"\nAnalyzing feature importance for zone: {zone}")
        print(f"Dataset shape: {df.shape}")
        print(f"Number of labels: {df['label'].nunique()}")
        print(f"Labels: {df['label'].unique()}")

        # Calculate importance using different methods
        results = {}

        # ANOVA
        print("Calculating ANOVA importance...")
        anova_importance = self.calculate_anova_importance(df)
        results['anova'] = anova_importance

        # Mutual Information
        print("Calculating Mutual Information importance...")
        mi_importance = self.calculate_mutual_info_importance(df)
        results['mutual_info'] = mi_importance

        # Random Forest
        print("Calculating Random Forest importance...")
        rf_importance = self.calculate_random_forest_importance(df)
        results['random_forest'] = rf_importance

        # Ensemble method
        print("Calculating Ensemble importance...")
        ensemble_importance = self.calculate_ensemble_importance(
            anova_importance, mi_importance, rf_importance
        )
        results['ensemble'] = ensemble_importance

        return results

    def create_importance_files(self):
        """
        Create feature importance files for all zones.

        Returns:
            dict: Dictionary mapping zone types to output file paths
        """
        zones = ['imagery_task', 'instruction', 'baseline_post']
        output_files = {}

        for zone in zones:
            results = self.analyze_zone(zone)
            if results is None:
                continue

            # Save results for each method
            for method, importance_df in results.items():
                output_file = os.path.join(self.results_dir, f"{zone}_{method}_importance.csv")
                importance_df.to_csv(output_file, index=False)

                if method == 'ensemble':  # Save the ensemble result as the main ranking file
                    main_output_file = os.path.join(self.merged_features_dir, f"{zone}_feature_ranking.csv")
                    importance_df.to_csv(main_output_file, index=False)
                    output_files[zone] = main_output_file

                    # Create visualization
                    self.visualize_importance(importance_df, zone)

            print(f"Feature importance files for {zone} saved to {self.results_dir}")

        return output_files

    def visualize_importance(self, importance_df, zone, top_n=20):
        """
        Create visualization of feature importance.

        Args:
            importance_df (pd.DataFrame): Feature importance dataframe
            zone (str): Zone name
            top_n (int): Number of top features to display
        """
        # Get top N features
        top_features = importance_df.head(top_n)

        # Create figure
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=top_features, palette='viridis')
        plt.title(f'Top {top_n} Important Features for {zone}')
        plt.xlabel('Importance Score (0-100)')
        plt.ylabel('Feature')
        plt.tight_layout()

        # Save figure
        output_file = os.path.join(self.results_dir, f"{zone}_importance_plot.png")
        plt.savefig(output_file, dpi=300)
        plt.close()


# ---------------------------
# Function to create feature importance files
# ---------------------------
def create_feature_importance_files(merged_features_dir):
    """
    Create feature importance ranking files for all zones.

    Args:
        merged_features_dir (str): Directory containing merged feature files

    Returns:
        dict: Dictionary mapping zone types to output file paths
    """
    # Initialize the feature importance calculator
    importance_calculator = EEGFeatureImportance(merged_features_dir)

    # Create importance files
    output_files = importance_calculator.create_importance_files()

    return output_files

# ---------------------------
# Feature Merger Class
# ---------------------------
class EEGFeatureMerger:
    def __init__(self, feature_base_dir):
        """
        Initializes the feature merger.

        Args:
            feature_base_dir (str): Base directory containing feature files
        """
        self.feature_base_dir = feature_base_dir


    def _extract_metadata_from_filename(self, filename):
        """
        Extract label and zone information from filename.

        Args:
            filename (str): Feature filename (e.g., 'features_p_af_b_imagery_task_1740854836.csv')

        Returns:
            tuple: (label, zone) extracted from the filename
        """
        parts = filename.split('_')

        # Extract the label by finding the parts before zone identifiers
        label_parts = []
        for part in parts[1:]:  # Skip 'features_' prefix
            if part in ['imagery', 'baseline', 'instruction']:
                break
            label_parts.append(part)

        label = '_'.join(label_parts)

        # Determine the zone
        if 'imagery_task' in filename:
            zone = 'imagery_task'
        elif 'baseline_post' in filename:
            zone = 'baseline_post'
        elif 'instruction' in filename:
            zone = 'instruction'
        else:
            zone = 'unknown'

        return label, zone

    def scan_feature_files(self):
        """
        Scan all feature files in the base directory and organize them by zone.

        Returns:
            dict: Dictionary mapping zone types to lists of (file_path, label) tuples
        """
        zone_files = {
            'imagery_task': [],
            'baseline_post': [],
            'instruction': [],
            'unknown': []
        }

        # Walk through the directory structure
        for root, _, files in os.walk(self.feature_base_dir):
            for file in files:
                if file.startswith('features_') and file.endswith('.csv'):
                    filepath = os.path.join(root, file)

                    # Extract metadata
                    label, zone = self._extract_metadata_from_filename(file)

                    # Store file information
                    zone_files[zone].append((filepath, label))

        return zone_files

    def merge_features_by_zone(self, output_dir):
        """
        Merge features from all files by zone type.

        Args:
            output_dir (str): Directory where merged feature files will be saved

        Returns:
            dict: Dictionary mapping zone types to output file paths
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)

        # Scan feature files
        zone_files = self.scan_feature_files()

        # Merge features for each zone
        output_files = {}

        for zone, file_list in zone_files.items():
            if not file_list:  # Skip empty zones
                print(f"No files found for zone: {zone}")
                continue

            print(f"Merging {len(file_list)} files for zone: {zone}")

            # Initialize list to hold dataframes
            all_features = []

            # Process each file
            for filepath, label in tqdm(file_list, desc=f"Processing {zone} files"):
                try:
                    # Read feature file
                    df = pd.read_csv(filepath)

                    # Add label column
                    df['label'] = label

                    # Add session info (extracted from directory structure)
                    session_dir = os.path.basename(os.path.dirname(filepath))
                    df['session'] = session_dir

                    # Rename index column to channel if it exists
                    if 'Channel' in df.columns:
                        df = df.rename(columns={'Channel': 'channel'})
                    elif df.index.name == 'Channel':
                        df = df.reset_index().rename(columns={'Channel': 'channel'})

                    # Add to list
                    all_features.append(df)

                except Exception as e:
                    print(f"Error processing {filepath}: {str(e)}")

            if all_features:
                # Concatenate all dataframes
                merged_df = pd.concat(all_features, ignore_index=True)

                # Save to output file
                output_file = os.path.join(output_dir, f"merged_features_{zone}.csv")
                merged_df.to_csv(output_file, index=False)
                output_files[zone] = output_file

                print(f"Merged {zone} features saved to {output_file}")
                print(
                    f"Shape: {merged_df.shape}, Features: {len(merged_df.columns) - 3}")  # -3 for label, session, channel

                # Generate summary statistics
                print(f"Labels in dataset: {merged_df['label'].nunique()}")
                print(f"Channels in dataset: {merged_df['channel'].nunique()}")
                print(f"Sessions in dataset: {merged_df['session'].nunique()}")
            else:
                print(f"No valid features found for {zone}")

        return output_files

    def create_feature_summary(self, output_dir):
        """
        Create a summary CSV with feature statistics across all zones.

        Args:
            output_dir (str): Directory where summary file will be saved
        """
        # Scan feature files
        zone_files = self.scan_feature_files()

        # Flatten the list of files
        all_files = []
        for file_list in zone_files.values():
            all_files.extend(file_list)

        if not all_files:
            print("No feature files found for summary.")
            return None

        # Sample a file to get feature names
        sample_file = all_files[0][0]
        sample_df = pd.read_csv(sample_file)
        feature_names = [col for col in sample_df.columns if col != 'Channel' and col != 'Unnamed: 0']

        # Initialize summary DataFrame
        summary_data = {
            'feature': feature_names,
            'min': [float('inf')] * len(feature_names),
            'max': [float('-inf')] * len(feature_names),
            'mean': [0] * len(feature_names),
            'std': [0] * len(feature_names),
            'missing_pct': [0] * len(feature_names)
        }

        # Initialize counters
        total_rows = 0
        file_count = 0

        # Process each file
        for filepath, _ in tqdm(all_files, desc="Calculating feature statistics"):
            try:
                df = pd.read_csv(filepath)

                # Skip if no features found
                if not all(feat in df.columns for feat in feature_names):
                    missing_feats = [feat for feat in feature_names if feat not in df.columns]
                    print(f"Skipping {filepath}: Missing features {missing_feats}")
                    continue

                # Update statistics
                for i, feat in enumerate(feature_names):
                    values = df[feat].dropna().values
                    if len(values) > 0:
                        summary_data['min'][i] = min(summary_data['min'][i], np.min(values))
                        summary_data['max'][i] = max(summary_data['max'][i], np.max(values))
                        summary_data['mean'][i] += np.sum(values)
                        summary_data['missing_pct'][i] += df[feat].isna().sum()

                total_rows += len(df)
                file_count += 1

            except Exception as e:
                print(f"Error processing {filepath} for summary: {str(e)}")

        if file_count > 0:
            # Finalize statistics
            for i in range(len(feature_names)):
                summary_data['mean'][i] /= total_rows - summary_data['missing_pct'][i]
                summary_data['missing_pct'][i] = (summary_data['missing_pct'][i] / total_rows) * 100

            # Calculate standard deviation (requires a second pass)
            std_sums = [0] * len(feature_names)
            for filepath, _ in tqdm(all_files, desc="Calculating std deviation"):
                try:
                    df = pd.read_csv(filepath)
                    for i, feat in enumerate(feature_names):
                        if feat in df.columns:
                            values = df[feat].dropna().values
                            if len(values) > 0:
                                std_sums[i] += np.sum((values - summary_data['mean'][i]) ** 2)
                except Exception:
                    pass

            for i in range(len(feature_names)):
                summary_data['std'][i] = np.sqrt(
                    std_sums[i] / (total_rows - summary_data['missing_pct'][i] * total_rows / 100))

            # Create and save summary DataFrame
            summary_df = pd.DataFrame(summary_data)
            summary_file = os.path.join(output_dir, "feature_summary_stats.csv")
            summary_df.to_csv(summary_file, index=False)

            print(f"Feature summary statistics saved to {summary_file}")
            return summary_file

        else:
            print("No files processed successfully for summary.")
            return None



# ---------------------------
# Function to create merged feature files
# ---------------------------
def create_merged_feature_files(feature_base_dir, output_dir):
    """
    Create merged feature files for all zones.

    Args:
        feature_base_dir (str): Base directory containing feature files
        output_dir (str): Directory where merged feature files will be saved
    """
    # Initialize the feature merger
    merger = EEGFeatureMerger(feature_base_dir)

    # Merge features by zone
    output_files = merger.merge_features_by_zone(output_dir)

    # Create feature summary
    summary_file = merger.create_feature_summary(output_dir)

    return output_files, summary_file




# ---------------------------
# Update to main function
# ---------------------------
def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Extract EEG features, create merged feature files, and analyze feature importance.')
    parser.add_argument('--input_dir', type=str, default='data/processed',
                        help='Base input directory containing processed data folders')
    parser.add_argument('--output_dir', type=str, default='data/extracted_features',
                        help='Base output directory for feature files')
    parser.add_argument('--merged_dir', type=str, default=f'data/merged_features/merge_run_{int(time.time())}',
                        help='Directory for merged feature files')
    parser.add_argument('--sfreq', type=float, default=250.0,
                        help='Sampling frequency in Hz (default: 250.0)')
    parser.add_argument('--skip_extraction', action='store_true',
                        help='Skip feature extraction and only create merged files')
    parser.add_argument('--skip_merging', action='store_true',
                        help='Skip feature merging and only calculate feature importance')
    parser.add_argument('--importance_only', action='store_true',
                        help='Skip extraction and merging, only calculate feature importance')

    args = parser.parse_args()

    # skip extraction
    if False:
        # Process the directory structure (feature extraction)
        if not args.skip_extraction and not args.importance_only:
            start_time = time.time()
            process_directory_structure(args.input_dir, args.output_dir)
            extract_time = time.time()
            print(f"Feature extraction time: {extract_time - start_time:.2f} seconds")
        elif args.skip_extraction or args.importance_only:
            print("Skipping feature extraction.")

    # Create merged feature files
    if not args.skip_merging and not args.importance_only:
        merge_start_time = time.time()
        print("\nCreating merged feature files...")
        output_files, summary_file = create_merged_feature_files(args.output_dir, args.merged_dir)
        merge_end_time = time.time()

        print(f"Feature merging time: {merge_end_time - merge_start_time:.2f} seconds")

        # Print summary of created files
        print("\nCreated the following merged feature files:")
        for zone, filepath in output_files.items():
            print(f"- {zone}: {filepath}")

        if summary_file:
            print(f"- Summary statistics: {summary_file}")
    elif args.skip_merging or args.importance_only:
        print("Skipping feature merging.")

    # Calculate feature importance
    importance_start_time = time.time()
    print("\nCalculating feature importance rankings...")
    importance_files = create_feature_importance_files(args.merged_dir)
    importance_end_time = time.time()

    print(f"Feature importance calculation time: {importance_end_time - importance_start_time:.2f} seconds")

    # Print summary of created files
    print("\nCreated the following feature importance ranking files:")
    for zone, filepath in importance_files.items():
        print(f"- {zone}: {filepath}")

    # Print additional information about using the feature rankings
    print("\nFeature importance rankings are now available for interpretation!")
    print("These rankings can be used to:")
    print("1. Select the most discriminative features for classification")
    print("2. Reduce dimensionality by keeping only important features")
    print("3. Gain insights into which EEG features best differentiate between labels")
    print("\nExample usage with scikit-learn:")
    print("\n```python")
    print("import pandas as pd")
    print("from sklearn.ensemble import RandomForestClassifier")
    print("from sklearn.model_selection import train_test_split")
    print("")
    print("# Load merged features")
    print("df = pd.read_csv('data/merged_features/merged_features_imagery_task.csv')")
    print("")
    print("# Load feature rankings")
    print("rankings = pd.read_csv('data/merged_features/imagery_task_feature_ranking.csv')")
    print("")
    print("# Select top 20 features")
    print("top_features = rankings['feature'].head(20).tolist()")
    print("")
    print("# Prepare data with only top features")
    print("X = df[top_features]")
    print("y = df['label']")
    print("")
    print("# Split train/test")
    print("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)")
    print("")
    print("# Train model with reduced feature set")
    print("rf = RandomForestClassifier(n_estimators=100, random_state=42)")
    print("rf.fit(X_train, y_train)")
    print("```")


if __name__ == "__main__":
    main()