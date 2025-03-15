import os
import pandas as pd
import numpy as np
import mne
import time
import argparse
from tqdm import tqdm
from scipy.stats import skew, kurtosis, entropy
from mne.time_frequency import psd_array_welch
import math
from pywt import wavedec
from feature_extractor import *


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
        description='Extract EEG features from processed data files and create merged feature files.')
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

    args = parser.parse_args()

    # Process the directory structure (feature extraction)
    if not args.skip_extraction:
        start_time = time.time()
        process_directory_structure(args.input_dir, args.output_dir)
        extract_time = time.time()
        print(f"Feature extraction time: {extract_time - start_time:.2f} seconds")
    else:
        print("Skipping feature extraction.")

    # Create merged feature files
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

    # Generate machine learning ready files for each zone
    print("\nFeature files are now ready for machine learning!")
    print("You can use the merged files directly with scikit-learn, for example:")
    print("\n```python")
    print("import pandas as pd")
    print("from sklearn.ensemble import RandomForestClassifier")
    print("from sklearn.model_selection import train_test_split")
    print("from sklearn.metrics import classification_report")
    print("")
    print("# Load merged features")
    print("df = pd.read_csv('data/merged_features/merged_features_imagery_task.csv')")
    print("")
    print("# Prepare data")
    print("X = df.drop(['label', 'channel', 'session'], axis=1)")
    print("y = df['label']")
    print("")
    print("# Split train/test")
    print("X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)")
    print("")
    print("# Train RandomForest")
    print("rf = RandomForestClassifier(n_estimators=100, random_state=42)")
    print("rf.fit(X_train, y_train)")
    print("")
    print("# Evaluate")
    print("y_pred = rf.predict(X_test)")
    print("print(classification_report(y_test, y_pred))")
    print("```")


if __name__ == "__main__":
    main()