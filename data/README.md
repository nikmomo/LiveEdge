# Dataset Metadata

## Overview

| Property | Value |
|----------|-------|
| Data source | Self-collected cattle IMU dataset |
| Collection period | Fall 2022 |
| Number of subjects (cattle) | Unknown (no subject_id column) |
| Total samples | 4,414,480 |
| Total duration (hours) | ~24.5 hours (at 50Hz) |
| Original sampling rate (Hz) | 50 (assumed) |
| Data status | Pre-normalized (mean=0, std=1) |

## File Structure

```
data/
├── raw/                              # Raw accelerometer data files
│   └── imu_cow_dataset_fall-2022.csv # Main dataset (783 MB, 4.4M rows)
├── processed/                        # Preprocessed and windowed data
└── splits/                           # Train/validation/test splits
```

## Data Format

### Raw Data File: `imu_cow_dataset_fall-2022.csv`

| Column | Type | Description | Range |
|--------|------|-------------|-------|
| ax | float64 | Accelerometer X-axis (normalized) | [-3.81, 3.28] |
| ay | float64 | Accelerometer Y-axis (normalized) | [-3.62, 3.89] |
| az | float64 | Accelerometer Z-axis (normalized) | [-3.62, 4.31] |
| gx | float64 | Gyroscope X-axis (normalized) | [-3.73, 3.75] |
| gy | float64 | Gyroscope Y-axis (normalized) | [-3.69, 3.63] |
| gz | float64 | Gyroscope Z-axis (normalized) | [-3.72, 3.76] |
| mx | float64 | Magnetometer X-axis (normalized) | [-3.02, 3.92] |
| my | float64 | Magnetometer Y-axis (normalized) | [-3.25, 2.77] |
| mz | float64 | Magnetometer Z-axis (normalized) | [-3.10, 3.45] |
| label | string | Behavior label | 14 categories |

### Sensor Units

All sensor values are **pre-normalized** with:
- Mean: ~0.0
- Standard deviation: ~1.0
- Original units unknown (likely g for accelerometer, deg/s for gyroscope)

### Axis Orientation

Unknown - assumed standard IMU orientation with sensor attached to cattle collar/ear tag.

## Behavior Labels

| Label | Description | Sample Count | Duration (h) | Percentage |
|-------|-------------|--------------|--------------|------------|
| chewing | Ruminating/jaw movements | 1,550,023 | 8.61 | 35.1% |
| standing | Standing stationary | 1,475,353 | 8.20 | 33.4% |
| grazing | Head-down grazing | 644,166 | 3.58 | 14.6% |
| lying | Lying down resting | 462,042 | 2.57 | 10.5% |
| walking | Locomotion | 105,052 | 0.58 | 2.4% |
| cow interactions | Social behavior | 39,832 | 0.22 | 0.9% |
| drinking | Water intake | 33,600 | 0.19 | 0.8% |
| grazing and walking | Mixed behavior | 31,003 | 0.17 | 0.7% |
| grazing and standing | Mixed behavior | 28,291 | 0.16 | 0.6% |
| grazing to walking or walking to grazing | Transition | 28,145 | 0.16 | 0.6% |
| licking | Self-grooming | 14,008 | 0.08 | 0.3% |
| lying to standing | Transition | 1,578 | 0.01 | 0.04% |
| staning to lying | Transition (typo in original) | 1,321 | 0.01 | 0.03% |
| running | High-speed locomotion | 66 | 0.00 | 0.001% |

**Class Imbalance Ratio**: 23,485:1 (chewing vs running)

### Behavior Grouping for Adaptive Sampling

| Cluster | Behaviors | Suggested Rate | Rationale |
|---------|-----------|----------------|-----------|
| INACTIVE | lying, lying to standing | 5-10 Hz | Low motion, low frequency content |
| RUMINATING | chewing | 10-15 Hz | Rhythmic jaw movements, mid-frequency |
| STATIONARY | standing, staning to lying | 10-15 Hz | Low motion, occasional weight shifts |
| FEEDING | grazing, drinking, licking | 15-25 Hz | Head movements, moderate activity |
| LOCOMOTION | walking, grazing and walking, transitions | 25-50 Hz | Higher frequency motion |
| HIGH_ACTIVITY | running, cow interactions | 50 Hz | Maximum motion capture |

## Data Quality

### Missing Values

| Column | Missing Count | Missing % |
|--------|---------------|-----------|
| All columns | 0 | 0.0% |

**Status**: No missing values detected.

### Duplicates

- Duplicate rows: 0

### Outliers (IQR 3x method)

| Column | Outlier Count | Outlier % |
|--------|---------------|-----------|
| gx | 173,467 | 3.93% |
| gy | 122,524 | 2.78% |
| gz | 151,848 | 3.44% |

**Note**: Gyroscope channels show elevated outlier rates, likely due to sharp rotational movements during active behaviors.

### Flat-line Detection

No significant flat-line periods detected (>5% threshold).

### Sensor Saturation

No saturation events detected at sensor limits.

## Summary Statistics

### Per-Axis Statistics (Full Dataset)

| Column | Mean | Std | Min | 25% | 50% | 75% | Max |
|--------|------|-----|-----|-----|-----|-----|-----|
| ax | 0.000 | 1.000 | -3.813 | -0.541 | 0.289 | 0.690 | 3.283 |
| ay | 0.000 | 1.000 | -3.619 | -0.598 | 0.136 | 0.700 | 3.890 |
| az | 0.000 | 1.000 | -3.621 | -0.572 | 0.160 | 0.681 | 4.312 |
| gx | 0.000 | 1.000 | -3.733 | -0.476 | 0.052 | 0.521 | 3.748 |
| gy | 0.000 | 1.000 | -3.692 | -0.535 | 0.003 | 0.542 | 3.632 |
| gz | 0.000 | 1.000 | -3.719 | -0.536 | 0.004 | 0.554 | 3.764 |
| mx | 0.000 | 1.000 | -3.017 | -0.706 | -0.043 | 0.686 | 3.921 |
| my | 0.000 | 1.000 | -3.251 | -0.673 | -0.024 | 0.650 | 2.769 |
| mz | 0.000 | 1.000 | -3.098 | -0.614 | -0.021 | 0.589 | 3.446 |

### Acceleration Magnitude by Behavior

| Behavior | Mean Magnitude | Std Magnitude |
|----------|----------------|---------------|
| running | 2.829 | 1.025 |
| grazing | 2.105 | 0.822 |
| walking | 2.001 | 0.876 |
| drinking | 1.914 | 0.628 |
| licking | 1.891 | 0.889 |
| grazing to walking or walking to grazing | 1.864 | 0.815 |
| grazing and standing | 1.807 | 0.722 |
| grazing and walking | 1.804 | 0.793 |
| cow interactions | 1.624 | 0.869 |
| staning to lying | 1.582 | 0.765 |
| standing | 1.512 | 0.681 |
| chewing | 1.385 | 0.682 |
| lying to standing | 1.323 | 0.734 |
| lying | 1.318 | 0.681 |

## Frequency Analysis

### Assumed Parameters

- Sampling Rate: 50 Hz
- Nyquist Frequency: 25 Hz

### Power Band Distribution by Behavior

| Behavior | Dominant (Hz) | Low (<2Hz) | Mid (2-10Hz) | High (>10Hz) |
|----------|---------------|------------|--------------|--------------|
| running | N/A | - | - | - |
| walking | 0.20 | 36.4% | 46.3% | 17.3% |
| grazing | 0.20 | 36.7% | 47.2% | 16.1% |
| grazing and walking | 0.20 | 48.8% | 41.5% | 9.7% |
| drinking | 0.20 | 57.5% | 39.2% | 3.3% |
| licking | 0.20 | 60.4% | 35.8% | 3.8% |
| cow interactions | 0.20 | 64.4% | 29.1% | 6.5% |
| grazing to walking | 0.20 | 58.3% | 35.1% | 6.6% |
| grazing and standing | 0.20 | 52.9% | 40.3% | 6.7% |
| chewing | 0.20 | 47.3% | 48.3% | 4.3% |
| standing | 0.20 | 74.0% | 24.0% | 2.0% |
| lying | 0.20 | 78.4% | 18.7% | 2.9% |
| lying to standing | 0.39 | 62.8% | 28.1% | 9.1% |
| staning to lying | 0.39 | 62.7% | 25.2% | 12.2% |

**Key Insight**: Low-frequency content (<2Hz) dominates inactive behaviors (lying, standing), while active behaviors (walking, grazing) have more mid-frequency content (2-10Hz). This supports adaptive sampling rate reduction for inactive states.

## Processed Data

Processed data is stored in NPZ format with the following arrays:

- `X`: Feature matrix or raw windows (n_samples, ...)
- `y`: Labels (n_samples,)
- `subject_ids`: Subject identifiers (n_samples,) - if available
- `timestamps`: Window start timestamps (n_samples,) - if available

## Preprocessing Pipeline

To preprocess raw data:

```bash
python scripts/preprocess_data.py data.raw_path=./data/raw
```

This will:
1. Load raw CSV files
2. Remove outliers (optional)
3. Create overlapping windows (1.5s, 50% overlap)
4. Extract features (if using traditional ML)
5. Save processed data to `data/processed/`
6. Create train/val/test splits in `data/splits/`

## Exploratory Analysis

Run the exploration script:

```bash
python scripts/explore_data.py --data_path data/raw/ --output_dir outputs/
```

Outputs:
- `outputs/exploration_results.json` - Raw analysis results
- `outputs/data_exploration_report.md` - Summary report
- `outputs/figures/exploration/` - Visualizations

## Known Issues

1. **No timestamp column**: Cannot verify actual sampling rate or temporal continuity
2. **No subject_id column**: Cannot analyze per-animal behavior patterns
3. **Typo in label**: "staning to lying" should be "standing to lying"
4. **Extreme class imbalance**: Running behavior has only 66 samples (0.001%)
5. **Transition labels**: Mixed behaviors may need special handling

## Recommendations

1. **Class Consolidation**: Consider merging rare classes (running, licking, transitions) into broader categories
2. **Oversampling**: Use SMOTE or similar techniques for minority classes
3. **Stratified Splits**: Ensure rare classes appear in all data splits
4. **Window Labeling**: For overlapping windows, use majority vote or weighted approach for mixed behaviors
