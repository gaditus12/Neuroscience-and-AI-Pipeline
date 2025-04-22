import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import joblib

# Setup logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
try:
    from neuroHarmonize.harmonizationLearn import harmonizationLearn
    from neuroHarmonize.harmonizationApply import harmonizationApply
    COMBAT_AVAILABLE = True
except (ImportError, ModuleNotFoundError) as e:
    COMBAT_AVAILABLE = False
    logger.warning("neuroHarmonize + statsmodels not installed, "
                   "batch harmonisation will be skipped (%s).", e)


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


def apply_combat(df: pd.DataFrame,
                 feature_columns: list[str],
                 keep_label: bool = True) -> pd.DataFrame:
    """
    Run ComBat (neuroHarmonize) across *sessions* to remove scanner / session
    effects that remain after the Z‑score step.

    Parameters
    ----------
    df               : data to harmonise – **must already be z‑scored**
    feature_columns  : numeric feature names
    keep_label       : if True the ‘label’ column is passed to ComBat as a
                       biological covariate so the class signal is *preserved*.

    Returns
    -------
    DataFrame of identical shape/order, but with ComBat‑adjusted features.
    """
    if not COMBAT_AVAILABLE:
        logger.info("ComBat step skipped (neuroHarmonize unavailable).")
        return df.copy()

    # neuroHarmonize expects shape  (N_samples × N_features)
    data_matrix = df[feature_columns].values

    # --- covariate table --------------------------------------------------
    # The batch variable **must** be called “SITE”
    covars = pd.DataFrame({"SITE": df["session"].values})
    # supply label as an **integer code** so harmonizationLearn treats it as numeric

    if keep_label and "label" in df.columns:
        covars["LABEL"] = df["label"].astype("category").cat.codes
    # ----------------------------------------------------------------------

    logger.info("Running ComBat (neuroHarmonize)…")
    #
    # harmonizationLearn returns    model, bayes_data
    #              (N_samples × N_features)     ↑
    #
    _, bayes_data = harmonizationLearn(
        data=data_matrix,
        covars=covars,
        eb=True          # empirical‑Bayes (standard ComBat)
    )

    df_harmonised = df.copy()
    df_harmonised[feature_columns] = bayes_data     # ← NO .T !

    return df_harmonised

def normalize_data(imagery_df, baseline_df, feature_columns):
    """
    Perform within-session Z-score normalization using baseline statistics.

    Args:
        imagery_df (pd.DataFrame): DataFrame containing imagery task data
        baseline_df (pd.DataFrame): DataFrame containing baseline data
        feature_columns (list): List of feature column names

    Returns:
        pd.DataFrame: Normalized imagery data
    """
    normalized_df = imagery_df.copy()

    # Get unique session-channel combinations
    session_channels = imagery_df[['session', 'channel']].drop_duplicates().values

    # Track statistics
    normalization_stats = {
        'total_combinations': len(session_channels),
        'successful': 0,
        'skipped': 0,
        'zero_std': 0
    }

    for session, channel in session_channels:
        # Get baseline data for this session and channel
        baseline_subset = baseline_df[(baseline_df['session'] == session) &
                                      (baseline_df['channel'] == channel)]

        # Check if we have baseline data for this session and channel
        if len(baseline_subset) == 0:
            logger.warning(f"No baseline data found for session {session}, channel {channel}. Skipping normalization.")
            normalization_stats['skipped'] += 1
            continue

        # Calculate baseline statistics
        baseline_means = baseline_subset[feature_columns].mean()
        baseline_stds = baseline_subset[feature_columns].std()

        # Get imagery data for this session and channel
        imagery_indices = normalized_df[(normalized_df['session'] == session) &
                                        (normalized_df['channel'] == channel)].index

        # Apply normalization
        for feature in feature_columns:
            std_value = baseline_stds[feature]

            # Handle zero standard deviation
            if std_value == 0 or pd.isna(std_value):
                logger.warning(
                    f"Zero or NaN std for {feature} in session {session}, channel {channel}. Using mean without normalization.")
                normalized_df.loc[imagery_indices, feature] = normalized_df.loc[imagery_indices, feature] - \
                                                              baseline_means[feature]
                normalization_stats['zero_std'] += 1
            else:
                normalized_df.loc[imagery_indices, feature] = (
                            (normalized_df.loc[imagery_indices, feature] - baseline_means[feature]) /
                            std_value)

        normalization_stats['successful'] += 1

        # Log progress for every 10 combinations
        if normalization_stats['successful'] % 10 == 0:
            logger.info(
                f"Processed {normalization_stats['successful']}/{normalization_stats['total_combinations']} combinations")

    # Log normalization statistics
    logger.info(f"Normalization complete: {normalization_stats['successful']} successful, "
                f"{normalization_stats['skipped']} skipped, "
                f"{normalization_stats['zero_std']} features with zero std")

    # ----------------------  NEW  ----------------------
    # Optional: run ComBat on the already Z‑scored data
    normalized_df = apply_combat(normalized_df, feature_columns)
    # ---------------------------------------------------

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_file = f"combat_model_{ts}.pkl"
    normalized_df = apply_combat(
        normalized_df,
        feature_columns,
        keep_label = True,
                          )

    return normalized_df, normalization_stats


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
        imagery_dir = "data/merged_features/14_sessions_merge_1743884222/split_set/test"
        baseline_dir = "data/merged_features/14_sessions_merge_1743884222/split_set/baseline"

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