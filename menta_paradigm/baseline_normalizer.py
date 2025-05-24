import os
import argparse, os, sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import logging

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


# ----------------------------------------------------------------------
def _normalise_one_csv(csv_path: Path,
                       baseline_df: pd.DataFrame,
                       steps: str,
                       out_dir: Path) -> None:
    """
    Load *one* imagery CSV, run the requested normalisation
    pipeline(s), and save result(s) next to `out_dir`.

    Parameters
    ----------
    csv_path    : Path to imagery file
    baseline_df : full baseline table (already loaded once)
    steps       : 'z' | 'combat' | 'both'
    out_dir     : root output directory
    """
    logger.info(f"→  processing {csv_path.name}")
    img_df   = pd.read_csv(csv_path)
    feat_cols = identify_feature_columns(img_df)

    # --- Z-score------------------------------
    if steps in {"z", "both"}:
        z_df, stats = normalize_data(img_df, baseline_df, feat_cols,
                                     run_combat=False)
        z_out = out_dir / f"{csv_path.stem}_norm-Z.csv"
        z_df.to_csv(z_out, index=False)
        logger.info(f"   saved Z-scored   →  {z_out.name}")

    # --- ComBat -----------------------------------
    if steps in {"combat", "both"}:

        if steps == "both":
            # Z-scored already; now harmonise that
            combat_input = z_df
            suffix = "Z_ComBat"
        else:  # steps == "combat"  →  NO Z-score
            combat_input = img_df.copy()
            suffix = "ComBat"

        cb_df = apply_combat(combat_input, feat_cols, keep_label=False)
        cb_out = out_dir / f"{csv_path.stem}_norm-{suffix}.csv"
        cb_df.to_csv(cb_out, index=False)
        logger.info(f"   saved ComBat file →  {cb_out.name}")

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
                 keep_label: bool = False) -> pd.DataFrame:
    """
    Run ComBat (neuroHarmonize) across *sessions* to remove scanner / session
    effects that remain after the Z‑score step.

    Parameters
    ----------
    df               : data to harmonise
    feature_columns  : numeric feature names
    keep_label       : if True the ‘label’ column is passed to ComBat as a
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
        # commented out, should not be used due to model 'peeking' on labels
        # covars["LABEL"] = df["label"].astype("category").cat.codes
        ...
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


from datetime import datetime
import logging
from typing import List, Tuple, Dict

import pandas as pd

# Configure a module‑level logger
logger = logging.getLogger(__name__)


def normalize_data(
    imagery_df: pd.DataFrame,
    baseline_df: pd.DataFrame | None,
    feature_columns: List[str],
    run_combat: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Z‑score normalise *imagery_df* within each **session × channel**.

    If *baseline_df* contains rows for a given ``session`` & ``channel``
    those rows provide the mean and standard deviation.  Otherwise the
    imagery data for that combination is used (self‑normalisation).

    Parameters
    ----------
    imagery_df : pd.DataFrame
        DataFrame holding the task / imagery epochs.
    baseline_df : pd.DataFrame | None
        Baseline epochs.  Pass ``None`` (or an *empty* DataFrame) to force
        self‑normalisation everywhere.
    feature_columns : list[str]
        Column names that should be Z‑scored.
    run_combat : bool, default ``True``
        Apply ComBat after normalisation.

    Returns
    -------
    pd.DataFrame
        The normalised imagery dataframe.
    dict
        Statistics about the normalisation run.
    """

    # Copy input so that the original is not modified in‑place
    normalized_df = imagery_df.copy()

    # All unique (session, channel) pairs present in the imagery data
    session_channels = imagery_df[["session", "channel"]].drop_duplicates().values

    # Counters for logging/debugging
    normalization_stats = {
        "total_combinations": len(session_channels),
        "successful": 0,
        "baseline_normalised": 0,
        "self_normalised": 0,
        "zero_std": 0,
    }
    #todo revert this now does normalziation without baseline
    baseline_df = None
    for session, channel in session_channels:
        # Decide which dataframe should provide the statistics
        if baseline_df is not None and not baseline_df.empty:
            stats_subset = baseline_df[(baseline_df["session"] == session) &
                                        (baseline_df["channel"] == channel)]
        else:
            stats_subset = pd.DataFrame()  # guarantees *len==0* path below

        # Fallback to imagery data when no baseline rows are available
        if len(stats_subset) == 0:
            stats_subset = imagery_df[(imagery_df["session"] == session) &
                                       (imagery_df["channel"] == channel)]
            normalization_stats["self_normalised"] += 1
        else:
            normalization_stats["baseline_normalised"] += 1

        # Compute the mean/std that will be used for *this* session×channel
        means = stats_subset[feature_columns].mean()
        stds = stats_subset[feature_columns].std()

        # Indices in *normalized_df* belonging to this session×channel
        imagery_idx = normalized_df[(normalized_df["session"] == session) &
                                     (normalized_df["channel"] == channel)].index

        # Vectorised column‑wise transformation
        for feature in feature_columns:
            std_val = stds[feature]
            if std_val == 0 or pd.isna(std_val):
                logger.warning(
                    "Zero or NaN std for %s (session=%s, channel=%s). "
                    "Applying mean‑centering only.",
                    feature,
                    session,
                    channel,
                )
                normalized_df.loc[imagery_idx, feature] = (
                    normalized_df.loc[imagery_idx, feature] - means[feature]
                )
                normalization_stats["zero_std"] += 1
            else:
                normalized_df.loc[imagery_idx, feature] = (
                    normalized_df.loc[imagery_idx, feature] - means[feature]
                ) / std_val

        normalization_stats["successful"] += 1
        if normalization_stats["successful"] % 10 == 0:
            logger.info(
                "Processed %d/%d combinations",
                normalization_stats["successful"],
                normalization_stats["total_combinations"],
            )

    # Final summary
    logger.info(
        "Normalization complete: %d successful, %d via baseline, %d self‑normalised, %d zero‑std.",
        normalization_stats["successful"],
        normalization_stats["baseline_normalised"],
        normalization_stats["self_normalised"],
        normalization_stats["zero_std"],
    )

    # ─────────────────────────────── Optional ComBat ──────────────────────────────
    if not run_combat:
        return normalized_df, normalization_stats

    # ``apply_combat`` must be defined elsewhere
    normalized_df = apply_combat(normalized_df, feature_columns, keep_label=False)
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



# ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Z-score / ComBat EEG normaliser"
    )
    ap.add_argument("--imagery",
                    help="CSV file *or* directory of imagery CSVs", default='data/merged_features/final_set_1_30hz_SPI_with_feat_imp/trinary_classification', required=False)
    ap.add_argument("--baseline",
                    # default="data/merged_features/14_sessions_merge_1743884222/merged_features_baseline_post.csv",
                    default=None,
                    help="baseline CSV (merged over sessions)")
    ap.add_argument("--steps",
                    choices=["z", "combat", "both"],
                    default="combat",
                    help="which normalisation stages to run")
    ap.add_argument("--out",
                    default=None,
                    help="output directory (default: "
                         "data/normalized_merges/<timestamp>/)")
    args = ap.parse_args()

    # ----- paths -------------------------------------------------------
    imag_path = Path(args.imagery)
    if not imag_path.exists():
        logger.error("Imagery path not found: %s", imag_path)
        sys.exit(1)

    out_root = Path(args.out or
                    f"data/normalized_merges/normalization_{datetime.now():%Y%m%d_%H%M%S}")
    out_root.mkdir(parents=True, exist_ok=True)

    # ----- load baseline once -----------------------------------------
    logger.info("Loading baseline file: %s", args.baseline)
    try:
        baseline_df = pd.read_csv(args.baseline)
    except Exception as e:
        baseline_df = None
    # ----- iterate over imagery CSVs ----------------------------------
    if imag_path.is_file():
        _normalise_one_csv(imag_path, baseline_df, args.steps, out_root)
    else:  # directory
        csv_files = sorted(p for p in imag_path.glob("*.csv"))
        if not csv_files:
            logger.error("No CSVs found in directory %s", imag_path)
            sys.exit(1)

        for csv in csv_files:
            _normalise_one_csv(csv, baseline_df, args.steps, out_root)

    logger.info("🎉  Finished – results in %s", out_root)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()