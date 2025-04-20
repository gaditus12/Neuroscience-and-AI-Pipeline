import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def load_data(directory):
    """
    Load all CSV files from a directory and combine them into a single DataFrame.

    Args:
        directory (str): Path to directory containing CSV files

    Returns:
        pd.DataFrame: Combined DataFrame of all CSV files
    """
    all_data = []

    try:
        for file in os.listdir(directory):
            if file.endswith('.csv'):
                file_path = os.path.join(directory, file)
                logger.info(f"Loading file: {file_path}")
                df = pd.read_csv(file_path)
                all_data.append(df)

        if not all_data:
            raise ValueError(f"No CSV files found in {directory}")

        combined_df = pd.concat(all_data, ignore_index=True)
        logger.info(f"Successfully loaded {len(all_data)} files with {len(combined_df)} total rows")
        return combined_df

    except Exception as e:
        logger.error(f"Error loading data from {directory}: {e}")
        raise


def identify_feature_columns(df):
    """
    Identify feature columns (all columns except metadata columns).

    Args:
        df (pd.DataFrame): DataFrame containing EEG data

    Returns:
        list: List of feature column names
    """
    metadata_columns = ['session', 'channel', 'label']
    feature_columns = [col for col in df.columns if col not in metadata_columns]

    # Convert feature columns to numeric type
    for col in feature_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    logger.info(f"Identified {len(feature_columns)} feature columns")
    return feature_columns


# ------------------------------------------------------------------
# Helper : scale each (session, channel, feature) range to [-3, 3]
# ------------------------------------------------------------------
def range_scale_per_session_channel(df, feature_cols,
                                    target_min=-3.0, target_max=3.0):
    """
    Linearly map the values of EVERY (session, channel, feature) slice
    so that its local min -> target_min and its local max -> target_max.

    Constant / all‑NaN slices are left unchanged.

    Returns
    -------
    df     : the SAME dataframe, modified in‑place for speed
    stats  : dict with simple counters for logging
    """
    span = target_max - target_min
    stats = {'scaled_blocks': 0, 'constant_or_nan': 0}

    # loop over unique session‑channel combos
    for (sess, ch), idx in df.groupby(['session', 'channel']).groups.items():
        block       = df.loc[idx, feature_cols]
        blk_min     = block.min()
        blk_max     = block.max()
        denom       = blk_max - blk_min           # range per feature
        scale_mask  = denom.gt(0) & denom.notna() # features we CAN scale

        if scale_mask.any():
            df.loc[idx, scale_mask.index[scale_mask]] = (
                (block[scale_mask.index[scale_mask]] - blk_min[scale_mask]) /
                denom[scale_mask]                       # --> 0‑1
            ) * span + target_min                      # --> target range
            stats['scaled_blocks'] += scale_mask.sum()

        stats['constant_or_nan'] += (~scale_mask).sum()

    return df, stats


def normalize_data(imagery_df, baseline_df, feature_columns):
    """
    1.  Within‑session Z‑score normalisation w.r.t. baseline
    2.  Per‑(session, channel, feature) min‑max scaling to [-3, 3]
    """
    normalized_df = imagery_df.copy()

    # ------------------------------------------------------------------
    # Step‑1 : Z‑score using baseline stats (same logic you already had)
    # ------------------------------------------------------------------
    session_channels = imagery_df[['session', 'channel']].drop_duplicates().values

    stats = {
        'total_combinations': len(session_channels),
        'successful': 0,
        'skipped': 0,
        'zero_std': 0
    }

    for session, channel in session_channels:
        baseline_subset = baseline_df[(baseline_df['session'] == session) &
                                      (baseline_df['channel'] == channel)]

        if baseline_subset.empty:
            logger.warning(f"No baseline for session {session}, channel {channel}. Skipping.")
            stats['skipped'] += 1
            continue

        means = baseline_subset[feature_columns].mean()
        stds  = baseline_subset[feature_columns].std()

        idx = normalized_df[(normalized_df['session'] == session) &
                            (normalized_df['channel'] == channel)].index

        for feat in feature_columns:
            if stds[feat] == 0 or pd.isna(stds[feat]):
                # fall back to centring only
                normalized_df.loc[idx, feat] = normalized_df.loc[idx, feat] - means[feat]
                stats['zero_std'] += 1
            else:
                normalized_df.loc[idx, feat] = (
                    (normalized_df.loc[idx, feat] - means[feat]) / stds[feat]
                )

        stats['successful'] += 1
        if stats['successful'] % 10 == 0:
            logger.info(f"Processed {stats['successful']}/{stats['total_combinations']} combinations")

    logger.info(f"Z‑score complete: {stats['successful']} ok, "
                f"{stats['skipped']} skipped, {stats['zero_std']} zero‑std")

    # ------------------------------------------------------------------
    # Step‑2 : Range‑scale each block to [-3, 3]
    # ------------------------------------------------------------------
    logger.info("Scaling every (session, channel, feature) slice to [-3, 3] ...")
    normalized_df, scale_stats = range_scale_per_session_channel(
        normalized_df, feature_columns, target_min=-3.0, target_max=3.0)

    logger.info(f"Range scaling complete "
                f"(scaled features: {scale_stats['scaled_blocks']}, "
                f"constant/NaN features: {scale_stats['constant_or_nan']})")

    # merge stats for downstream saving
    stats['range_scaling'] = scale_stats
    return normalized_df, stats

def save_normalized_data(normalized_df, stats, output_dir=None):
    """
    Save normalized data and statistics to output directory.

    Args:
        normalized_df (pd.DataFrame): Normalized imagery data
        stats (dict): Normalization statistics
        output_dir (str, optional): Output directory. If None, creates a timestamped directory.

    Returns:
        str: Path to the output directory
    """
    # Create output directory with timestamp if not provided
    if output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"data/normalized_merges/normalization_{timestamp}"

    os.makedirs(output_dir, exist_ok=True)

    # Save normalized data
    output_file = os.path.join(output_dir, "normalized_imagery.csv")
    normalized_df.to_csv(output_file, index=False)
    logger.info(f"Saved normalized data to {output_file}")

    # Save normalization summary
    summary_file = os.path.join(output_dir, "normalization_summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Normalization Summary\n")
        f.write("====================\n\n")
        f.write(f"Total session-channel combinations: {stats['total_combinations']}\n")
        f.write(f"Successfully normalized: {stats['successful']}\n")
        f.write(f"Skipped (no baseline data): {stats['skipped']}\n")
        f.write(f"Features with zero std: {stats['zero_std']}\n")

    logger.info(f"Saved normalization summary to {summary_file}")
    return output_dir


def verify_data_integrity(imagery_df, baseline_df):
    """
    Verify integrity of the data and report any issues.

    Args:
        imagery_df (pd.DataFrame): DataFrame containing imagery task data
        baseline_df (pd.DataFrame): DataFrame containing baseline data

    Returns:
        bool: True if data integrity is verified, False otherwise
    """
    imagery_sessions = set(imagery_df['session'].unique())
    baseline_sessions = set(baseline_df['session'].unique())

    # Check for missing sessions
    missing_sessions = imagery_sessions - baseline_sessions
    if missing_sessions:
        logger.warning(f"Sessions in imagery data but not in baseline data: {missing_sessions}")

    # Check for missing channels within sessions
    issues_found = False
    for session in imagery_sessions.intersection(baseline_sessions):
        imagery_channels = set(imagery_df[imagery_df['session'] == session]['channel'].unique())
        baseline_channels = set(baseline_df[baseline_df['session'] == session]['channel'].unique())

        missing_channels = imagery_channels - baseline_channels
        if missing_channels:
            logger.warning(
                f"Session {session} has channels in imagery data but not in baseline data: {missing_channels}")
            issues_found = True

    # Check for NaN values in features
    nan_in_baseline = baseline_df.isna().sum().sum()
    if nan_in_baseline > 0:
        logger.warning(f"Found {nan_in_baseline} NaN values in baseline data")
        issues_found = True

    nan_in_imagery = imagery_df.isna().sum().sum()
    if nan_in_imagery > 0:
        logger.warning(f"Found {nan_in_imagery} NaN values in imagery data")
        issues_found = True

    if not issues_found:
        logger.info("Data integrity verified - no issues found")
    else:
        logger.warning("Data integrity issues detected - see warnings above")

    return not issues_found


def main():
    """
    Main function to execute the normalization process.
    """
    try:
        # 1. Define input and output paths
        imagery_dir = "data/merged_features/12_sessions_merge_1743795360/imagery"
        baseline_dir = "data/merged_features/12_sessions_merge_1743795360/baseline"

        # 2. Load data
        logger.info("Loading imagery task data...")
        imagery_df = load_data(imagery_dir)

        logger.info("Loading baseline data...")
        baseline_df = load_data(baseline_dir)

        # 3. Identify feature columns
        feature_columns = identify_feature_columns(imagery_df)

        # 4. Verify data integrity
        logger.info("Verifying data integrity...")
        verify_data_integrity(imagery_df, baseline_df)

        # 5. Perform normalization
        logger.info("Performing within-session Z-score normalization...")
        normalized_df, stats = normalize_data(imagery_df, baseline_df, feature_columns)

        # 6. Save results
        logger.info("Saving normalized data...")
        timestamp = datetime.now().strftime("%Y%m%d_%H")
        output_dir = f"data/normalized_merges/normalization_{timestamp}"
        save_normalized_data(normalized_df, stats, output_dir)

        logger.info("Normalization process completed successfully!")

    except Exception as e:
        logger.error(f"Error in normalization process: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()