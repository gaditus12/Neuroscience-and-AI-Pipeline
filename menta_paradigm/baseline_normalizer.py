import os
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from scipy import signal

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


def apply_highpass_filter(df, feature_columns, sampling_rate=250, cutoff=0.5):
    """
    Apply high-pass filter to remove slow drifts which might be session-specific.

    Args:
        df (pd.DataFrame): DataFrame containing EEG data
        feature_columns (list): List of feature column names
        sampling_rate (int): Sampling rate of the EEG data in Hz
        cutoff (float): Cutoff frequency for high-pass filter in Hz

    Returns:
        pd.DataFrame: Filtered DataFrame
    """
    filtered_df = df.copy()

    # Design Butterworth high-pass filter
    nyquist = sampling_rate / 2
    normal_cutoff = cutoff / nyquist
    b, a = signal.butter(4, normal_cutoff, btype='high', analog=False)

    # Apply filter to each session-channel combination
    session_channels = df[['session', 'channel']].drop_duplicates().values

    for session, channel in session_channels:
        indices = filtered_df[(filtered_df['session'] == session) &
                              (filtered_df['channel'] == channel)].index

        for feature in feature_columns:
            # Get feature data for this session-channel
            data = filtered_df.loc[indices, feature].values

            # Apply filter if we have enough data points
            if len(data) > 10:  # Arbitrary threshold to avoid filtering very short signals
                try:
                    filtered_data = signal.filtfilt(b, a, data)
                    filtered_df.loc[indices, feature] = filtered_data
                except Exception as e:
                    logger.warning(f"Filtering failed for session {session}, channel {channel}, feature {feature}: {e}")

    logger.info(f"High-pass filtering applied to remove slow drifts (cutoff={cutoff}Hz)")
    return filtered_df


def covariance_normalization(imagery_df, baseline_df, feature_columns):
    """
    Normalize the data using covariance structure from baseline.
    This performs whitening transformation to decorrelate features based on baseline covariance.

    Args:
        imagery_df (pd.DataFrame): DataFrame containing imagery task data
        baseline_df (pd.DataFrame): DataFrame containing baseline data
        feature_columns (list): List of feature column names

    Returns:
        pd.DataFrame: Normalized imagery data
        dict: Normalization statistics
    """
    normalized_df = imagery_df.copy()

    # Track statistics
    normalization_stats = {
        'total_sessions': len(imagery_df['session'].unique()),
        'successful': 0,
        'skipped': 0,
        'ill_conditioned': 0
    }

    # Process each session
    for session in imagery_df['session'].unique():
        try:
            # Get baseline data for this session
            baseline_subset = baseline_df[baseline_df['session'] == session][feature_columns]

            # Check if we have enough baseline data
            if len(baseline_subset) < len(feature_columns):
                logger.warning(
                    f"Insufficient baseline data for session {session}. Falling back to Z-score normalization.")

                # Fall back to Z-score normalization by channel
                for channel in imagery_df[imagery_df['session'] == session]['channel'].unique():
                    baseline_channel = baseline_df[(baseline_df['session'] == session) &
                                                   (baseline_df['channel'] == channel)]

                    if len(baseline_channel) == 0:
                        continue

                    baseline_means = baseline_channel[feature_columns].mean()
                    baseline_stds = baseline_channel[feature_columns].std()

                    indices = normalized_df[(normalized_df['session'] == session) &
                                            (normalized_df['channel'] == channel)].index

                    for feature in feature_columns:
                        std_value = baseline_stds[feature]
                        if std_value > 0 and not pd.isna(std_value):
                            normalized_df.loc[indices, feature] = ((normalized_df.loc[indices, feature] -
                                                                    baseline_means[feature]) / std_value)
                        else:
                            normalized_df.loc[indices, feature] = normalized_df.loc[indices, feature] - baseline_means[
                                feature]

                normalization_stats['skipped'] += 1
                continue

            # Calculate covariance matrix from baseline
            baseline_mean = baseline_subset.mean()
            baseline_cov = np.cov(baseline_subset.T)

            # Check if covariance matrix is well-conditioned
            eigenvalues = np.linalg.eigvalsh(baseline_cov)
            condition_number = np.max(eigenvalues) / np.max(np.abs(eigenvalues[eigenvalues > 1e-10]))

            if condition_number > 1e6 or np.any(eigenvalues <= 0):
                logger.warning(f"Ill-conditioned covariance matrix for session {session}. Adding regularization.")
                # Add small regularization to make it positive definite
                baseline_cov += np.eye(baseline_cov.shape[0]) * (np.trace(baseline_cov) * 1e-3)
                normalization_stats['ill_conditioned'] += 1

            # Compute whitening matrix using eigendecomposition
            eigenvalues, eigenvectors = np.linalg.eigh(baseline_cov)
            eigenvalues = np.maximum(eigenvalues, 1e-10)  # Ensure all eigenvalues are positive
            whitening_matrix = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T

            # Get imagery data for this session
            imagery_indices = normalized_df[normalized_df['session'] == session].index

            # Center the data
            for feature in feature_columns:
                normalized_df.loc[imagery_indices, feature] = normalized_df.loc[imagery_indices, feature] - \
                                                              baseline_mean[feature]

            # Apply whitening transformation by channel
            for channel in imagery_df[imagery_df['session'] == session]['channel'].unique():
                channel_indices = normalized_df[(normalized_df['session'] == session) &
                                                (normalized_df['channel'] == channel)].index

                # Extract feature data for this channel
                channel_data = normalized_df.loc[channel_indices, feature_columns].values

                # Apply whitening transformation
                whitened_data = channel_data @ whitening_matrix

                # Update normalized DataFrame
                normalized_df.loc[channel_indices, feature_columns] = whitened_data

            normalization_stats['successful'] += 1
            logger.info(f"Covariance normalization applied for session {session}")

        except Exception as e:
            logger.error(f"Error in covariance normalization for session {session}: {e}")
            normalization_stats['skipped'] += 1

    logger.info(f"Covariance normalization complete: {normalization_stats['successful']} successful, "
                f"{normalization_stats['skipped']} skipped, "
                f"{normalization_stats['ill_conditioned']} required regularization")

    return normalized_df, normalization_stats


def normalize_data(imagery_df, baseline_df, feature_columns):
    """
    Perform enhanced normalization to reduce session-specific characteristics.

    Args:
        imagery_df (pd.DataFrame): DataFrame containing imagery task data
        baseline_df (pd.DataFrame): DataFrame containing baseline data
        feature_columns (list): List of feature column names

    Returns:
        pd.DataFrame: Normalized imagery data
        dict: Normalization statistics
    """
    # Step 1: Apply high-pass filtering to both imagery and baseline data
    logger.info("Applying high-pass filtering to remove slow drifts...")
    filtered_imagery = apply_highpass_filter(imagery_df, feature_columns)
    filtered_baseline = apply_highpass_filter(baseline_df, feature_columns)

    # Step 2: Apply covariance-based normalization
    logger.info("Applying covariance-based normalization...")
    normalized_df, cov_stats = covariance_normalization(filtered_imagery, filtered_baseline, feature_columns)

    # Combine statistics for reporting
    stats = {
        'highpass_filter_applied': True,
        'cutoff_frequency': 0.5,  # Hz
        'covariance_normalization': cov_stats
    }

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
        f.write("Advanced normalization techniques applied:\n")
        f.write("- High-pass filtering (0.5 Hz cutoff)\n")
        f.write("- Covariance-based normalization (whitening)\n\n")

        f.write("Covariance Normalization Statistics:\n")
        f.write(f"- Total sessions: {stats['covariance_normalization']['total_sessions']}\n")
        f.write(f"- Successfully normalized: {stats['covariance_normalization']['successful']}\n")
        f.write(f"- Skipped (insufficient data): {stats['covariance_normalization']['skipped']}\n")
        f.write(f"- Required regularization: {stats['covariance_normalization']['ill_conditioned']}\n")

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
    Main function to execute the enhanced normalization process.
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

        # 5. Perform enhanced normalization
        logger.info("Performing enhanced normalization to reduce session specificity...")
        normalized_df, stats = normalize_data(imagery_df, baseline_df, feature_columns)

        # 6. Save results
        logger.info("Saving normalized data...")
        timestamp = datetime.now().strftime("%Y%m%d_%H")
        output_dir = f"data/normalized_merges/normalization_{timestamp}"
        save_normalized_data(normalized_df, stats, output_dir)

        logger.info("Enhanced normalization process completed successfully!")

    except Exception as e:
        logger.error(f"Error in normalization process: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()