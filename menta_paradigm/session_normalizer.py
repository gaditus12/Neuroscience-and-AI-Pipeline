import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import RobustScaler
from typing import List, Optional, Union, Tuple

class EEGSessionNormalizer:
    def __init__(self, feature_columns: Optional[List[str]] = None,
                 outlier_threshold: float = 3.0,
                 output_range: Optional[Tuple[float, float]] = (-10, 10),
                 use_outlier_capping: bool = False):  # Set to False to disable outlier capping
        self.feature_columns = feature_columns
        self.outlier_threshold = outlier_threshold
        self.output_range = output_range
        self.use_outlier_capping = use_outlier_capping
        self.scalers = {}
        self.stats = {}

    def _identify_feature_columns(self, df: pd.DataFrame) -> List[str]:
        non_feature_cols = ['channel', 'label', 'session']
        return [col for col in df.columns if col not in non_feature_cols]

    def _convert_numeric_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df_converted = df.copy()
        for col in self.feature_columns:
            df_converted[col] = pd.to_numeric(df_converted[col], errors='coerce')
        return df_converted

    def _handle_outliers_fit(self, df: pd.DataFrame, group_key: str) -> pd.DataFrame:
        # This function is used only if use_outlier_capping is True.
        df_clean = df.copy()
        self.stats[group_key] = {}
        for col in self.feature_columns:
            if df[col].isna().all():
                continue
            df_clean[col] = df_clean[col].astype(float)
            # Impute NaNs with median
            median_val = df_clean[col].median()
            df_clean[col].fillna(median_val, inplace=True)
            q1 = df_clean[col].quantile(0.25)
            q3 = df_clean[col].quantile(0.75)
            iqr = q3 - q1
            if iqr == 0:
                continue
            lower_bound = q1 - (self.outlier_threshold * iqr)
            upper_bound = q3 + (self.outlier_threshold * iqr)
            self.stats[group_key][col] = {
                'median': median_val,
                'lower_bound': lower_bound,
                'upper_bound': upper_bound,
                'outliers_detected': ((df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)).sum()
            }
            df_clean.loc[df_clean[col] < lower_bound, col] = lower_bound
            df_clean.loc[df_clean[col] > upper_bound, col] = upper_bound
        return df_clean

    def _apply_outlier_capping(self, df: pd.DataFrame, group_key: str) -> pd.DataFrame:
        # This function is used only if use_outlier_capping is True.
        df_clean = df.copy()
        if group_key not in self.stats:
            return df_clean
        for col in self.feature_columns:
            if col not in self.stats[group_key]:
                continue
            stats = self.stats[group_key][col]
            lower = stats['lower_bound']
            upper = stats['upper_bound']
            # Impute NaNs with stored median
            df_clean[col].fillna(stats['median'], inplace=True)
            df_clean.loc[df_clean[col] < lower, col] = lower
            df_clean.loc[df_clean[col] > upper, col] = upper
        return df_clean

    def fit(self, df: pd.DataFrame) -> 'EEGSessionNormalizer':
        if self.feature_columns is None:
            self.feature_columns = self._identify_feature_columns(df)
        df = self._convert_numeric_columns(df)
        grouped = df.groupby(['session', 'channel'])
        for (session, channel), group_data in grouped:
            group_key = f"{session}_{channel}"
            if len(group_data) < 2:
                continue
            # Optionally apply outlier capping
            if self.use_outlier_capping:
                clean_data = self._handle_outliers_fit(group_data, group_key)
            else:
                clean_data = group_data.copy()
            scaler = RobustScaler()
            try:
                scaler.fit(clean_data[self.feature_columns])
                train_scaled = scaler.transform(clean_data[self.feature_columns])
                self.scalers[group_key] = {
                    'scaler': scaler,
                    'scaler_min': np.min(train_scaled, axis=0),
                    'scaler_max': np.max(train_scaled, axis=0)
                }
            except Exception as e:
                print(f"Error fitting {group_key}: {str(e)}")
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        result_df = self._convert_numeric_columns(df.copy())
        grouped = result_df.groupby(['session', 'channel'])
        for (session, channel), group_data in grouped:
            group_key = f"{session}_{channel}"
            if group_key not in self.scalers:
                continue
            indices = group_data.index
            if self.use_outlier_capping:
                clean_data = self._apply_outlier_capping(group_data, group_key)
            else:
                clean_data = group_data.copy()
            scaler_info = self.scalers[group_key]
            scaler = scaler_info['scaler']
            scaled_features = scaler.transform(clean_data[self.feature_columns])
            if self.output_range:
                min_val, max_val = self.output_range
                train_min = scaler_info['scaler_min']
                train_max = scaler_info['scaler_max']
                denominator = train_max - train_min
                denominator[denominator == 0] = 1.0
                scaled_features = (scaled_features - train_min) / denominator
                scaled_features = scaled_features * (max_val - min_val) + min_val
            result_df.loc[indices, self.feature_columns] = scaled_features
        return result_df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit the normalizer to the data and transform it.
        """
        return self.fit(df).transform(df)

    def summary(self) -> pd.DataFrame:
        """
        Generate a summary of the normalization statistics.
        """
        summary_rows = []
        for group_key, feature_stats in self.stats.items():
            if '_transform' in group_key:
                continue
            session, channel = group_key.split('_', 1)
            for feature, stats in feature_stats.items():
                row = {
                    'session': session,
                    'channel': channel,
                    'feature': feature,
                    **stats
                }
                summary_rows.append(row)
        return pd.DataFrame(summary_rows)


def process_eeg_data(csv_dir: str, output_dir: Optional[str] = None,
                     output_range: Optional[Tuple[float, float]] = None) -> tuple:
    """
    Process EEG data from CSV files applying robust session-based normalization.
    """
    input_path = f'data/merged_features/{csv_dir}'
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Directory {input_path} not found")

    csv_files = [f for f in os.listdir(input_path) if f.endswith('.csv')]
    if not csv_files:
        raise ValueError(f"No CSV files found in {input_path}")

    print(f"Found {len(csv_files)} CSV files in {input_path}")

    dfs = []
    for file in csv_files:
        file_path = os.path.join(input_path, file)
        print(f"Reading {file_path}")
        df = pd.read_csv(file_path)
        if 'channel' not in df.columns or 'session' not in df.columns:
            print(f"Warning: File {file} does not have required columns. Columns: {df.columns.tolist()}")
            continue
        dfs.append(df)

    if not dfs:
        raise ValueError("No valid data found in CSV files")

    combined_df = pd.concat(dfs, ignore_index=True)
    print(f"Combined data shape: {combined_df.shape}")
    print(f"Columns: {combined_df.columns.tolist()}")

    normalizer = EEGSessionNormalizer(output_range=output_range)
    if normalizer.feature_columns is None:
        normalizer.feature_columns = normalizer._identify_feature_columns(combined_df)
        print(f"Feature columns ({len(normalizer.feature_columns)}): {normalizer.feature_columns}")

    normalized_df = normalizer.fit_transform(combined_df)
    summary_df = normalizer.summary()

    if output_dir:
        output_path = f'data/normalized_features/{output_dir}'
        os.makedirs(output_path, exist_ok=True)
        normalized_file = os.path.join(output_path, 'normalized_data.csv')
        normalized_df.to_csv(normalized_file, index=False)
        summary_file = os.path.join(output_path, 'normalization_summary.csv')
        summary_df.to_csv(summary_file, index=False)
        print(f"Normalized data saved to {normalized_file}")
        print(f"Normalization summary saved to {summary_file}")

    return normalized_df, summary_df


if __name__ == "__main__":
    import argparse

    current = '12_sessions_merge_1743795360'
    parser = argparse.ArgumentParser(description='Normalize EEG features per session and channel.')
    parser.add_argument('--csv_dir', type=str, help='Directory containing CSV files in data/merged_features/', default=current)
    parser.add_argument('--output_dir', type=str, default=current,
                        help='Directory to save normalized data in data/normalized_features/')
    parser.add_argument('--min_value', type=float, default=-10,
                        help='Minimum value for output range scaling')
    parser.add_argument('--max_value', type=float, default=10,
                        help='Maximum value for output range scaling')

    args = parser.parse_args()
    output_range = None
    if args.min_value is not None and args.max_value is not None:
        output_range = (args.min_value, args.max_value)

    try:
        normalized_df, summary_df = process_eeg_data(
            csv_dir=args.csv_dir,
            output_dir=args.output_dir,
            output_range=output_range
        )
        print(f"Processing complete. Processed {len(normalized_df)} rows.")
    except Exception as e:
        print(f"Error processing data: {str(e)}")
        import traceback
        traceback.print_exc()
