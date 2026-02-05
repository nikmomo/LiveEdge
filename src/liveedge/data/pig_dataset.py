"""
CCI25 Pig Behavior Dataset Loader and Preprocessor.

Dataset: CCI25 pig behavior dataset
Sensor: MetaMotionC (Bosch BMI160 6-axis IMU)
Location: Pig ear tag
Original sampling rate: 50 Hz
Channels: 6-axis (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)

Behaviors:
- Active: Eating, Drinking, Walking, Interacting
- Inactive: Lying, Standing
- Other: Unknown, empty values
"""

import os
import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Literal

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt
from tqdm import tqdm


@dataclass
class DatasetInfo:
    """Information about the loaded dataset."""
    n_samples: int
    n_labeled_samples: int
    sampling_rate: float
    duration_hours: float
    n_pigs: int
    n_dates: int
    behavior_counts: dict
    behavior_percentages: dict
    files_loaded: list


@dataclass
class BMI160Spec:
    """BMI160 sensor specifications for energy modeling."""
    # Operating current at different ODRs (from datasheet)
    odr_current_ua = {
        50: 27,
        25: 17,
        12.5: 10,
        10: 10,  # Estimated
        5: 6,    # Estimated
    }
    voltage: float = 1.8  # Operating voltage
    sleep_current_ua: float = 3.0  # Sleep mode current


class CCI25PigDataset:
    """
    Loader and preprocessor for CCI25 pig behavior dataset.

    Attributes:
        data_dir: Path to the cleaned_raw data directory
        data: Combined DataFrame with all pig data
        info: DatasetInfo object with statistics
    """

    # Expected column names
    SENSOR_COLS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
    LABEL_COL = 'label'
    META_COLS = ['date', 'file', 'id', 'sequence']

    # Valid behaviors (active and inactive)
    ACTIVE_BEHAVIORS = ['Eating', 'Drinking', 'Walking', 'Interacting']
    INACTIVE_BEHAVIORS = ['Lying', 'Standing']
    VALID_BEHAVIORS = ACTIVE_BEHAVIORS + INACTIVE_BEHAVIORS

    def __init__(self, data_dir: str):
        """
        Initialize dataset loader.

        Args:
            data_dir: Path to cleaned_raw data directory containing CSV files
        """
        self.data_dir = Path(data_dir)
        self.data: Optional[pd.DataFrame] = None
        self.info: Optional[DatasetInfo] = None
        self._original_sampling_rate = 50.0

    def load(self, filter_labeled_only: bool = True, verbose: bool = True) -> pd.DataFrame:
        """
        Load all CSV files from the data directory.

        Args:
            filter_labeled_only: If True, keep only rows with valid behavior labels
            verbose: Print loading progress

        Returns:
            Combined DataFrame with all data
        """
        csv_files = sorted(self.data_dir.glob("*.csv"))

        if not csv_files:
            raise FileNotFoundError(f"No CSV files found in {self.data_dir}")

        if verbose:
            print(f"Found {len(csv_files)} CSV files in {self.data_dir}")

        dfs = []
        files_loaded = []

        for csv_file in tqdm(csv_files, desc="Loading files", disable=not verbose):
            try:
                df = pd.read_csv(csv_file)
                df['source_file'] = csv_file.name
                dfs.append(df)
                files_loaded.append(csv_file.name)
            except Exception as e:
                print(f"Warning: Could not load {csv_file.name}: {e}")

        # Combine all data
        self.data = pd.concat(dfs, ignore_index=True)

        if verbose:
            print(f"Total samples loaded: {len(self.data):,}")

        # Fix typo in raw data: "Laying" -> "Lying"
        laying_count = (self.data[self.LABEL_COL] == 'Laying').sum()
        if laying_count > 0:
            self.data[self.LABEL_COL] = self.data[self.LABEL_COL].replace('Laying', 'Lying')
            if verbose:
                print(f"Corrected label typo: 'Laying' -> 'Lying' ({laying_count:,} samples)")

        # Filter labeled data if requested
        if filter_labeled_only:
            original_count = len(self.data)
            self.data = self.data[self.data[self.LABEL_COL].isin(self.VALID_BEHAVIORS)]
            self.data = self.data.reset_index(drop=True)

            if verbose:
                print(f"Filtered to labeled samples: {len(self.data):,} "
                      f"({len(self.data)/original_count*100:.1f}%)")

        # Remove rows with NaN values in sensor columns
        nan_mask = self.data[self.SENSOR_COLS].isna().any(axis=1)
        nan_count = nan_mask.sum()
        if nan_count > 0:
            if verbose:
                print(f"Removing {nan_count} rows with NaN values")
            self.data = self.data[~nan_mask].reset_index(drop=True)

        # Compute statistics
        self._compute_info(files_loaded)

        return self.data

    def _compute_info(self, files_loaded: list):
        """Compute dataset statistics."""
        behavior_counts = self.data[self.LABEL_COL].value_counts().to_dict()
        total = sum(behavior_counts.values())
        behavior_percentages = {k: v/total*100 for k, v in behavior_counts.items()}

        self.info = DatasetInfo(
            n_samples=len(self.data),
            n_labeled_samples=len(self.data[self.data[self.LABEL_COL].isin(self.VALID_BEHAVIORS)]),
            sampling_rate=self._original_sampling_rate,
            duration_hours=len(self.data) / self._original_sampling_rate / 3600,
            n_pigs=self.data['id'].nunique() if 'id' in self.data.columns else 1,
            n_dates=self.data['date'].nunique() if 'date' in self.data.columns else 1,
            behavior_counts=behavior_counts,
            behavior_percentages=behavior_percentages,
            files_loaded=files_loaded,
        )

    def get_sensor_data(self, channels: Literal['6axis', 'acc_only'] = '6axis') -> np.ndarray:
        """
        Get sensor data as numpy array.

        Args:
            channels: '6axis' for all channels, 'acc_only' for accelerometer only

        Returns:
            numpy array of shape (n_samples, n_channels)
        """
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")

        if channels == '6axis':
            cols = self.SENSOR_COLS
        else:
            cols = self.SENSOR_COLS[:3]  # acc_x, acc_y, acc_z

        return self.data[cols].values

    def get_labels(self) -> np.ndarray:
        """Get behavior labels as numpy array."""
        if self.data is None:
            raise ValueError("Data not loaded. Call load() first.")
        return self.data[self.LABEL_COL].values

    def get_label_encoder(self) -> dict:
        """Get mapping from behavior names to integer labels."""
        behaviors = sorted(self.data[self.LABEL_COL].unique())
        return {b: i for i, b in enumerate(behaviors)}

    def get_encoded_labels(self) -> tuple[np.ndarray, dict]:
        """
        Get integer-encoded labels.

        Returns:
            Tuple of (encoded_labels, label_encoder)
        """
        encoder = self.get_label_encoder()
        labels = self.get_labels()
        encoded = np.array([encoder[l] for l in labels])
        return encoded, encoder

    def print_summary(self):
        """Print dataset summary."""
        if self.info is None:
            print("Dataset not loaded yet.")
            return

        print("\n" + "=" * 60)
        print("CCI25 Pig Behavior Dataset Summary")
        print("=" * 60)
        print(f"Total samples: {self.info.n_samples:,}")
        print(f"Labeled samples: {self.info.n_labeled_samples:,}")
        print(f"Sampling rate: {self.info.sampling_rate} Hz")
        print(f"Duration: {self.info.duration_hours:.2f} hours")
        print(f"Number of pigs: {self.info.n_pigs}")
        print(f"Number of dates: {self.info.n_dates}")
        print(f"Files loaded: {len(self.info.files_loaded)}")

        print("\nBehavior Distribution:")
        print("-" * 40)
        for behavior in sorted(self.info.behavior_counts.keys()):
            count = self.info.behavior_counts[behavior]
            pct = self.info.behavior_percentages[behavior]
            bar = "█" * int(pct / 2)
            print(f"  {behavior:<12} {count:>8,} ({pct:>5.1f}%) {bar}")
        print("=" * 60)


def downsample_with_antialiasing(
    data: np.ndarray,
    original_hz: float = 50.0,
    target_hz: float = 25.0,
) -> np.ndarray:
    """
    Downsample data using anti-aliasing filter.

    Uses a Butterworth lowpass filter to prevent aliasing before decimation.

    Args:
        data: numpy array of shape (n_samples, n_channels)
        original_hz: Original sampling rate
        target_hz: Target sampling rate

    Returns:
        Downsampled data
    """
    factor = int(original_hz / target_hz)

    if factor == 1:
        return data.copy()

    # Design lowpass filter
    nyq = original_hz / 2
    cutoff = target_hz / 2 * 0.9  # 10% margin to avoid aliasing
    b, a = butter(4, cutoff / nyq, btype='low')

    # Filter each channel
    filtered = np.zeros_like(data)
    for i in range(data.shape[1]):
        filtered[:, i] = filtfilt(b, a, data[:, i])

    # Decimate
    downsampled = filtered[::factor]

    return downsampled


def downsample_labels(labels: np.ndarray, factor: int) -> np.ndarray:
    """
    Downsample labels using majority voting.

    Args:
        labels: Array of labels (can be string or numeric)
        factor: Decimation factor

    Returns:
        Downsampled labels
    """
    if factor == 1:
        return labels.copy()

    n_new = len(labels) // factor
    new_labels = []

    for i in range(n_new):
        start = i * factor
        end = start + factor
        segment = labels[start:end]
        # Use Counter for string-compatible majority voting
        from collections import Counter
        counts = Counter(segment)
        majority = counts.most_common(1)[0][0]
        new_labels.append(majority)

    return np.array(new_labels)


def create_sliding_windows(
    data: np.ndarray,
    labels: np.ndarray,
    odr: float,
    window_sec: float = 1.5,
    overlap: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows from continuous data.

    Args:
        data: Sensor data of shape (n_samples, n_channels)
        labels: Labels of shape (n_samples,)
        odr: Sampling rate in Hz
        window_sec: Window length in seconds
        overlap: Overlap ratio (0-1)

    Returns:
        Tuple of (windows, window_labels)
        - windows: shape (n_windows, window_samples, n_channels)
        - window_labels: shape (n_windows,)
    """
    from collections import Counter

    window_samples = int(window_sec * odr)
    step_samples = int(window_samples * (1 - overlap))

    n_windows = (len(data) - window_samples) // step_samples + 1

    windows = np.zeros((n_windows, window_samples, data.shape[1]))
    window_labels = []

    for i in range(n_windows):
        start = i * step_samples
        end = start + window_samples

        windows[i] = data[start:end]

        # Majority vote for window label (string-compatible)
        segment_labels = labels[start:end]
        counts = Counter(segment_labels)
        majority = counts.most_common(1)[0][0]
        window_labels.append(majority)

    return windows, np.array(window_labels)


def extract_features(window: np.ndarray, odr: float) -> dict:
    """
    Extract time and frequency domain features from a window.

    Args:
        window: Window data of shape (window_samples, n_channels)
        odr: Sampling rate in Hz

    Returns:
        Dictionary of feature_name -> feature_value
    """
    from scipy.stats import skew, kurtosis
    from scipy.fft import rfft, rfftfreq

    features = {}
    channel_names = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'][:window.shape[1]]

    # Time domain features per channel
    for i, ch in enumerate(channel_names):
        signal = window[:, i]

        features[f'{ch}_mean'] = np.mean(signal)
        features[f'{ch}_std'] = np.std(signal)
        features[f'{ch}_min'] = np.min(signal)
        features[f'{ch}_max'] = np.max(signal)
        features[f'{ch}_range'] = np.ptp(signal)
        features[f'{ch}_rms'] = np.sqrt(np.mean(signal**2))
        features[f'{ch}_skew'] = skew(signal)
        features[f'{ch}_kurtosis'] = kurtosis(signal)

        # Zero crossing rate
        zcr = np.sum(np.diff(np.sign(signal)) != 0) / len(signal)
        features[f'{ch}_zcr'] = zcr

    # Acceleration magnitude features (first 3 channels)
    acc_mag = np.sqrt(np.sum(window[:, :3]**2, axis=1))
    features['acc_mag_mean'] = np.mean(acc_mag)
    features['acc_mag_std'] = np.std(acc_mag)
    features['acc_mag_max'] = np.max(acc_mag)
    features['acc_mag_range'] = np.ptp(acc_mag)

    # SMA (Signal Magnitude Area)
    features['sma'] = np.sum(np.abs(window[:, :3])) / len(window)

    # Frequency domain features (accelerometer channels)
    for i, ch in enumerate(['acc_x', 'acc_y', 'acc_z']):
        if i >= window.shape[1]:
            break
        signal = window[:, i]

        # FFT
        fft_vals = np.abs(rfft(signal))
        freqs = rfftfreq(len(signal), 1/odr)

        # Skip DC component
        fft_vals = fft_vals[1:]
        freqs = freqs[1:]

        if len(fft_vals) > 0:
            # Dominant frequency
            dom_idx = np.argmax(fft_vals)
            features[f'{ch}_dominant_freq'] = freqs[dom_idx]

            # Spectral energy
            features[f'{ch}_spectral_energy'] = np.sum(fft_vals**2)

            # Spectral entropy
            psd = fft_vals**2
            psd_norm = psd / (np.sum(psd) + 1e-10)
            features[f'{ch}_spectral_entropy'] = -np.sum(
                psd_norm * np.log2(psd_norm + 1e-10)
            )

            # Spectral centroid
            features[f'{ch}_spectral_centroid'] = np.sum(freqs * fft_vals) / (
                np.sum(fft_vals) + 1e-10
            )
        else:
            features[f'{ch}_dominant_freq'] = 0
            features[f'{ch}_spectral_energy'] = 0
            features[f'{ch}_spectral_entropy'] = 0
            features[f'{ch}_spectral_centroid'] = 0

    return features


def process_all_odrs(
    data: np.ndarray,
    labels: np.ndarray,
    output_dir: str,
    target_odrs: list[float] = [50, 25, 12.5, 10, 5],
    original_odr: float = 50.0,
    window_sec: float = 1.5,
    overlap: float = 0.5,
    verbose: bool = True,
) -> dict:
    """
    Process data for all target ODRs: downsample, create windows, extract features.

    Args:
        data: Sensor data of shape (n_samples, n_channels)
        labels: Labels of shape (n_samples,)
        output_dir: Directory to save processed data
        target_odrs: List of target sampling rates
        original_odr: Original sampling rate
        window_sec: Window length in seconds
        overlap: Window overlap ratio
        verbose: Print progress

    Returns:
        Dictionary with processing statistics
    """
    output_dir = Path(output_dir)
    stats = {}

    for odr in target_odrs:
        if verbose:
            print(f"\nProcessing {odr} Hz...")

        # Downsample
        if odr == original_odr:
            data_odr = data.copy()
            labels_odr = labels.copy()
        else:
            factor = int(original_odr / odr)
            data_odr = downsample_with_antialiasing(data, original_odr, odr)
            labels_odr = downsample_labels(labels, factor)

        if verbose:
            print(f"  Samples: {len(data_odr):,}")

        # Create windows
        windows, window_labels = create_sliding_windows(
            data_odr, labels_odr, odr, window_sec, overlap
        )

        if verbose:
            print(f"  Windows: {len(windows):,} (shape: {windows.shape})")

        # Extract features
        if verbose:
            print(f"  Extracting features...")

        feature_list = []
        for w in tqdm(windows, desc="  Features", disable=not verbose):
            features = extract_features(w, odr)
            feature_list.append(features)

        features_df = pd.DataFrame(feature_list)

        if verbose:
            print(f"  Features: {features_df.shape[1]}")

        # Save
        odr_str = str(odr).replace('.', '_')
        save_dir = output_dir / f"{odr_str}hz"
        save_dir.mkdir(parents=True, exist_ok=True)

        # Full 6-axis data
        np.save(save_dir / "windows.npy", windows)
        np.save(save_dir / "labels.npy", window_labels)
        features_df.to_csv(save_dir / "features.csv", index=False)

        # ACC-only data (first 3 channels)
        windows_acc = windows[:, :, :3]
        np.save(save_dir / "windows_acc.npy", windows_acc)

        # ACC-only features
        acc_feature_list = []
        for w in windows_acc:
            features = extract_features(w, odr)
            acc_feature_list.append(features)
        features_acc_df = pd.DataFrame(acc_feature_list)
        features_acc_df.to_csv(save_dir / "features_acc.csv", index=False)

        # Metadata
        meta = {
            'odr': odr,
            'window_sec': window_sec,
            'overlap': overlap,
            'n_samples': len(data_odr),
            'n_windows': len(windows),
            'window_samples': windows.shape[1],
            'n_channels': windows.shape[2],
            'n_features_6axis': features_df.shape[1],
            'n_features_acc': features_acc_df.shape[1],
        }
        with open(save_dir / "meta.json", 'w') as f:
            json.dump(meta, f, indent=2)

        stats[odr] = meta

        if verbose:
            print(f"  Saved to: {save_dir}")

    return stats
