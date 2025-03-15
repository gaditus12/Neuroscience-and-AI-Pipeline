import os
import pandas as pd
import numpy as np
import mne
import time
import matplotlib
import argparse
from tqdm import tqdm

# Use TkAgg backend if needed
matplotlib.use("TkAgg")


class EEGPreprocessor:
    def __init__(self, input_file, output_dir, sfreq=250.0, ch_names=None, ch_types=None):
        """
        Initializes the EEGPreprocessor.

        Args:
            input_file (str): Path to the CSV file containing raw EEG data.
            output_dir (str): Folder where processed files will be saved.
            sfreq (float): Sampling frequency (Hz). Default is 250 Hz.
            ch_names (list of str): List of channel names (default: Ch1-Ch8).
            ch_types (list of str): List of channel types (default: all "eeg").
        """
        self.input_file = input_file
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.sfreq = sfreq
        # Default channel names if not provided:
        self.ch_names = ch_names if ch_names is not None else [f"Ch{i + 1}" for i in range(8)]
        self.ch_types = ch_types if ch_types is not None else ["eeg"] * len(self.ch_names)
        self.raw = None

    def load_data(self):
        """
        Loads the CSV file into a pandas DataFrame and creates an MNE RawArray.

        Returns:
            raw (mne.io.RawArray): The raw EEG data.
        """
        try:
            df = pd.read_csv(self.input_file)
            # Assume the CSV has a column 'Timestamp' and 8 EEG channel columns
            if 'Timestamp' in df.columns:
                timestamps = df['Timestamp'].values
                eeg_data = df.drop(columns=['Timestamp']).values.T  # Transpose: shape becomes (n_channels, n_times)
            else:
                # Handle case where there might be no Timestamp column
                print(f"Warning: No Timestamp column found in {self.input_file}. Using all columns as EEG data.")
                eeg_data = df.values.T

            info = mne.create_info(ch_names=self.ch_names, sfreq=self.sfreq, ch_types=self.ch_types)
            self.raw = mne.io.RawArray(eeg_data, info)
            return self.raw
        except Exception as e:
            print(f"Error loading data from {self.input_file}: {str(e)}")
            return None

    def rename_channels(self, rename_dict):
        """
        Renames channels using the provided dictionary.

        Args:
            rename_dict (dict): Dictionary mapping current channel names to new names.
        """
        if self.raw is None:
            self.raw = self.load_data()
            if self.raw is None:
                return
        self.raw.rename_channels(rename_dict)

    def post_process(self, l_freq=1, h_freq=30, notch_freq=50):
        """
        Executes the standard preprocessing pipeline:
            1. Load data (if not already loaded)
            2. Apply a notch filter at notch_freq Hz.
            3. Apply a bandpass filter from l_freq to h_freq Hz.

        Args:
            l_freq (float): Lower cutoff frequency (Hz). Default is 1 Hz.
            h_freq (float): Upper cutoff frequency (Hz). Default is 30 Hz.
            notch_freq (float): Frequency for notch filtering (Hz). Default is 50 Hz.

        Returns:
            raw (mne.io.RawArray): The preprocessed raw EEG data.
        """
        if self.raw is None:
            self.raw = self.load_data()
            if self.raw is None:
                return None

        try:
            raw_notched = self.raw.copy().notch_filter(freqs=notch_freq, picks='eeg', verbose=False)
            raw_filtered = raw_notched.copy().filter(l_freq=l_freq, h_freq=h_freq, picks='eeg', verbose=False)
            self.raw = raw_filtered
            return self.raw
        except Exception as e:
            print(f"Error in post-processing {self.input_file}: {str(e)}")
            return None

    def save_processed_csv(self, output_filename=None):
        """
        Saves the processed data to a CSV file.
        The output CSV will have a "Timestamp" column followed by the EEG channels.
        It uses the current channel names from the Raw object (which reflects any renaming).

        Args:
            output_filename (str): Optional. The filename for the CSV file. If not provided,
                                   a default name will be used.
        """
        if self.raw is None:
            print(f"No processed raw data available to save for {self.input_file}.")
            return

        try:
            # Get the processed data: shape (n_channels, n_times)
            data = self.raw.get_data().T  # shape becomes (n_times, n_channels)
            n_samples = data.shape[0]
            # Generate timestamps assuming continuous sampling starting at 0.
            timestamps = np.linspace(0, (n_samples - 1) / self.sfreq, n_samples)
            # Use current channel names from the raw object
            current_ch_names = self.raw.info['ch_names']
            df = pd.DataFrame(data, columns=current_ch_names)
            df.insert(0, "Timestamp", timestamps)

            if output_filename is None:
                base_filename = os.path.basename(self.input_file)
                output_filename = os.path.join(self.output_dir, f"processed_{base_filename}")
            else:
                output_filename = os.path.join(self.output_dir, output_filename)

            os.makedirs(os.path.dirname(output_filename), exist_ok=True)
            df.to_csv(output_filename, index=False)
            print(f"Processed data saved to {output_filename}")
            return output_filename
        except Exception as e:
            print(f"Error saving processed data for {self.input_file}: {str(e)}")
            return None


def process_file(input_file, output_file, channel_map=None):
    """Process a single CSV file with EEG data."""

    # Skip if the output file already exists
    if os.path.exists(output_file):
        print(f"Output file {output_file} already exists. Skipping.")
        return True

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Set up the default channel mapping if none is provided
    if channel_map is None:
        channel_map = {
            'Ch1': 'O1',
            'Ch2': 'Oz',
            'Ch3': 'O2',
            'Ch4': 'PO3',
            'Ch5': 'PO4',
            'Ch6': 'TP7',
            'Ch7': 'TP8',
            'Ch8': 'Fz'
        }

    # Initialize the preprocessor
    preprocessor = EEGPreprocessor(input_file=input_file, output_dir=os.path.dirname(output_file), sfreq=250.0)

    # Load the data
    raw = preprocessor.load_data()
    if raw is None:
        return False

    # Rename channels
    preprocessor.rename_channels(channel_map)

    # Post-process: apply 50Hz notch, bandpass from 1 to 30 Hz
    processed_raw = preprocessor.post_process(l_freq=1, h_freq=30, notch_freq=50)
    if processed_raw is None:
        return False

    # Save the processed data to CSV with the specified output path
    basename = os.path.basename(input_file)
    result = preprocessor.save_processed_csv(basename)

    return result is not None


def process_directory_structure(input_base_dir, output_base_dir):
    """
    Process all CSV files in the directory structure, preserving the hierarchy.

    Args:
        input_base_dir (str): Base directory containing data folders
        output_base_dir (str): Base directory where processed data will be saved
    """
    # Count total files for progress tracking
    total_files = 0
    for root, _, files in os.walk(input_base_dir):
        total_files += sum(1 for f in files if f.endswith('.csv'))

    print(f"Found {total_files} CSV files to process")

    # Create counter for success/failure tracking
    success_count = 0
    failure_count = 0

    # Process all CSV files
    with tqdm(total=total_files, desc="Processing files") as pbar:
        for root, dirs, files in os.walk(input_base_dir):
            # Create corresponding output directory structure
            rel_path = os.path.relpath(root, input_base_dir)
            output_dir = os.path.join(output_base_dir, rel_path)
            os.makedirs(output_dir, exist_ok=True)

            # Process each CSV file in this directory
            for file in files:
                if file.endswith('.csv'):
                    input_file = os.path.join(root, file)
                    output_file = os.path.join(output_dir, file)

                    # Process the file
                    result = process_file(input_file, output_file)

                    if result:
                        success_count += 1
                    else:
                        failure_count += 1

                    pbar.update(1)

    print(f"\nProcessing complete!")
    print(f"Successfully processed: {success_count} files")
    print(f"Failed to process: {failure_count} files")

    return success_count, failure_count


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Process EEG data files while preserving directory structure.')
    parser.add_argument('--input_dir', type=str, default='data/captures',
                        help='Base input directory containing data folders')
    parser.add_argument('--output_dir', type=str, default='data/processed',
                        help='Base output directory for processed data')
    parser.add_argument('--plot', action='store_true',
                        help='Enable plotting of data (not recommended for batch processing)')

    args = parser.parse_args()

    # Process the directory structure
    start_time = time.time()
    process_directory_structure(args.input_dir, args.output_dir)
    end_time = time.time()

    print(f"Total processing time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    main()