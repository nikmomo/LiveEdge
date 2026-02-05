# LiveEdge

Energy-efficient livestock behavior monitoring system using behavior-driven adaptive sampling.

## Overview

LiveEdge is an edge-deployed machine learning system that performs real-time **pig** behavior classification to dynamically adjust accelerometer sampling rates (6.25-25 Hz), achieving ~70% energy reduction while maintaining classification accuracy comparable to fixed-rate baselines.

### Key Features

- **Behavior Classification**: 5-class pig behavior classification (Lying, Eating, Standing, Walking, Drinking)
- **Adaptive Sampling**: 3-cluster ODR mapping with k-hysteresis FSM controller
- **Energy Optimization**: Cycle-accurate energy modeling validated via Renode (nRF52832 + BMI160)
- **24 Statistical Features**: MCU-friendly, ODR-invariant, FFT-free feature extraction

## Results

Results from cycle-accurate hardware simulation with compressed RF model (30 trees, depth 8) and k=3 hysteresis, using 5-fold temporal-order CV:

| Configuration | Behavior Acc | Cluster Acc | Energy (mJ/day) | Battery Life |
|--------------|--------------|-------------|-----------------|--------------|
| 50 Hz Baseline | 71.24% | 73.75% | 17,600 | 108 days |
| 25 Hz Baseline | 70.73% | 72.89% | 15,008 | 127 days |
| 12.5 Hz Baseline | 69.78% | 71.94% | 13,194 | 144 days |
| 6.25 Hz Baseline | 67.86% | 70.02% | 12,416 | 153 days |
| **LiveEdge (k=3)** | **70.09%** | **79.92%** | **5,266** | **361 days** |

LiveEdge achieves **3.34x battery life extension** (108 -> 361 days) with 70.1% total energy savings.

## Installation

```bash
# Clone the repository
git clone https://github.com/nikmomo/LiveEdge.git
cd LiveEdge

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/macOS
# or
.\venv\Scripts\activate  # Windows

# Install package
pip install -e ".[all]"
```

## Project Structure

```
LiveEdge/
├── src/liveedge/              # Core library (7 modules)
│   ├── data/                  # Data processing, 24-feature extraction
│   ├── clustering/            # Behavior clustering (STABLE/MODERATE/ACTIVE)
│   ├── models/                # RF, XGBoost, SVM, CNN1D, TCN classifiers
│   ├── sampling/              # FSM controller with k-hysteresis
│   ├── energy/                # Hardware power models (BMI160, nRF52832)
│   ├── evaluation/            # Classification and energy metrics
│   └── simulation/            # Streaming simulation environment
├── scripts/                   # Paper reproduction scripts
│   ├── verified_simulation.py          # Tables 4, 5: Main results & ablation
│   ├── dl_cluster_accuracy.py          # Table 7: DL classifier comparison
│   ├── cattle_generalization_study.py  # Table 8: Cross-species validation
│   ├── baseline_cluster_accuracy.py    # Baseline cluster accuracy analysis
│   ├── recalculate_energy_unified.py   # Unified energy model recalculation
│   ├── traditional_ml_comparison.py    # Traditional ML model comparison
│   ├── utils/
│   │   └── cluster_mapping.py          # Centralized cluster/energy config
│   └── figures/                        # Paper figure generation
│       ├── f1_vs_odr.py                # Table 1 / Figure: fmin determination
│       ├── pareto_curve.py             # Figure: Energy-accuracy Pareto
│       ├── odr_trajectory.py           # Figure: 24-hour ODR state changes
│       ├── dwell_time_dist.py          # Figure: Cluster dwell time distribution
│       ├── confusion_matrix.py         # Figure: 3x3 cluster confusion matrix
│       ├── k_sensitivity.py            # Figure: k-hysteresis sensitivity
│       ├── create_architecture.py      # Figure: System architecture diagram
│       └── generate_paper_figures_v2.py# Batch figure generation
├── tests/                     # Unit tests
├── data/                      # Dataset (not included, see below)
├── deprecated/                # Old/unused scripts
└── docs/paper/                # LaTeX source (not included)
```

## Cluster Mapping

Based on per-behavior fmin analysis with tau=5% F1 degradation threshold:

| Cluster | Behaviors | fmin | ODR | Dataset Proportion |
|---------|-----------|------|-----|-------------------|
| STABLE | Lying | 6.25 Hz | 6.25 Hz | ~35% |
| MODERATE | Eating | 12.5 Hz | 12.5 Hz | ~40% |
| ACTIVE | Standing, Walking, Drinking | 25 Hz | 25 Hz | ~25% |

Interacting is excluded (<2% of samples, requires 50 Hz).

## Experiment Scripts and Paper Tables

### Table 1: Minimum Sampling Frequency (fmin) by Behavior

**Script**: `scripts/figures/f1_vs_odr.py`

| Detail | Value |
|--------|-------|
| Dataset | Pig 50Hz, subsampled to 50k windows |
| CV | 5-fold shuffled (classifier evaluation) |
| Method | Train RF at each ODR (6.25-50Hz), measure per-behavior F1 |
| Features | 24 statistical features (inline extraction) |
| Output | `outputs/paper_revision/figures/f1_vs_odr.{pdf,png,json}` |

### Tables 4 & 5: Main Results and Ablation Study

**Script**: `scripts/verified_simulation.py`

| Detail | Value |
|--------|-------|
| Dataset | Full pig dataset (~257k windows after excluding Interacting) |
| CV | 5-fold temporal-order (no shuffle, for FSM validity) |
| Method | Train behavior + cluster RF classifiers, run LiveEdge FSM simulation |
| Features | 24 statistical features via `liveedge.data.extract_24_features_batch` |
| k values | Tests k=3 (default) |
| Output | `outputs/verified_simulation/verified_results.json` |

**Known code issues**:
- `T_INFERENCE = 0.020` (20ms) but paper uses 26ms from Renode
- Baseline energy hardcoded as 15,331 mJ/day; paper uses 17,600 mJ/day
- Uses full RF model (100 trees); paper hardware simulation uses compressed RF (30 trees, depth 8)

The correct energy parameters are derived in `scripts/recalculate_energy_unified.py` which takes Renode ground truth (108-day baseline = 17,600 mJ/day, 361-day LiveEdge = 5,266 mJ/day).

### Table 6: k-Sensitivity Analysis

**Script**: `deprecated/scripts/k_sensitivity_and_error_analysis.py` (not yet in active scripts/)

| Detail | Value |
|--------|-------|
| Dataset | 30k samples from pig dataset |
| CV | 5-fold temporal-order |
| Method | Test k=1,2,3,4,5; measure transitions, energy, detection delay |
| Status | Script exists in deprecated/, needs to be moved and updated |

### Table 7: Classifier Comparison (RF vs DL)

**Script**: `scripts/dl_cluster_accuracy.py`

| Detail | Value |
|--------|-------|
| Dataset | 50k samples from pig dataset |
| CV | 5-fold shuffled (standard classifier evaluation) |
| Models | RF+Features, 1D-CNN, LSTM, GRU, ResNet-1D, DeepConvLSTM, Transformer |
| Features | RF: 24 statistical; DL: raw accelerometer (75, 3) |
| GPU | Required for DL models |
| Output | `outputs/dl_comparison/dl_cluster_comparison.json` |

**Note**: `scripts/traditional_ml_comparison.py` provides additional MCU-compatible model comparison (Decision Tree, SVM, Logistic Regression) but currently uses deprecated RFE-20 features instead of 24 statistical features.

### Table 8: Cross-Species Validation (Cattle)

**Script**: `scripts/cattle_generalization_study.py`

| Detail | Value |
|--------|-------|
| Dataset | Cattle IMU data (`data/raw/imu_cow_dataset_fall-2022.csv`) |
| CV | 5-fold shuffled |
| Method | Map 14 cattle behaviors to 5-class + 3-cluster, evaluate RF |
| Output | `outputs/cattle_generalization/cattle_generalization_results.json` |

**Known code issue**: Uses `extract_rfe20_features()` (RFE-20, lines 137-184) instead of the unified 24 statistical features. Paper states all experiments use 24 features.

### Table 9: Related Work Comparison

**Script**: `deprecated/scripts/related_work_comparison.py` (not yet in active scripts/)

| Detail | Value |
|--------|-------|
| Methods | Fixed-rate, Variance-Adaptive, Variance-Skip, Confidence-Skip, LiveEdge |
| CV | 5-fold temporal-order |
| Status | Script exists in deprecated/, needs to be moved and updated |

### Supplementary Scripts

| Script | Purpose |
|--------|---------|
| `scripts/baseline_cluster_accuracy.py` | Cluster accuracy for all baselines (behavior-derived vs direct) |
| `scripts/recalculate_energy_unified.py` | Derives energy parameters from Renode ground truth |

### Figure Scripts

| Figure | Script | Description |
|--------|--------|-------------|
| F1 vs ODR | `scripts/figures/f1_vs_odr.py` | Per-behavior F1 score degradation curves |
| Pareto Curve | `scripts/figures/pareto_curve.py` | Energy-accuracy trade-off frontier |
| ODR Trajectory | `scripts/figures/odr_trajectory.py` | 24-hour ODR state change timeline |
| Dwell Time | `scripts/figures/dwell_time_dist.py` | Cluster dwell time histograms |
| Confusion Matrix | `scripts/figures/confusion_matrix.py` | 3x3 cluster confusion matrix |
| k-Sensitivity | `scripts/figures/k_sensitivity.py` | Multi-metric k analysis |
| Architecture | `scripts/figures/create_architecture.py` | System architecture diagram |
| Batch Generation | `scripts/figures/generate_paper_figures_v2.py` | All supplementary figures |

## Execution Order

```bash
# 1. Table 1: fmin determination
python scripts/figures/f1_vs_odr.py

# 2. Tables 4, 5: Main simulation results + ablation
python scripts/verified_simulation.py

# 3. Energy model recalculation (validates against Renode)
python scripts/recalculate_energy_unified.py

# 4. Baseline cluster accuracy analysis
python scripts/baseline_cluster_accuracy.py

# 5. Table 7: DL classifier comparison (requires GPU)
python scripts/dl_cluster_accuracy.py

# 6. Table 8: Cattle cross-species validation
python scripts/cattle_generalization_study.py

# 7. Generate paper figures
python scripts/figures/generate_paper_figures_v2.py
```

## Hardware and Energy Model

| Component | Parameter | Value | Source |
|-----------|-----------|-------|--------|
| IMU | BMI160 current | 7-27 uA (6.25-50Hz) | Datasheet |
| MCU | nRF52832 active | 2.1 mA | Datasheet |
| MCU | nRF52832 idle | 1.9 uA | Datasheet |
| MCU | T_inference | 26 ms | Renode cycle-accurate simulation |
| BLE | TX current | 5.3 mA | Datasheet |
| Battery | CR2032 | 220 mAh, 3V, 80% usable | Datasheet |
| Overhead | BLE + RTC + regulator | 589 mJ/day | Derived |

Energy formula:
```
E_total = E_IMU + E_MCU + E_overhead
E_IMU   = I_ODR x V x T_day
E_MCU   = n_inf x T_inf x P_active + T_idle x P_idle
Battery = (220 mAh x 3V x 3600s x 0.8) / E_daily
```

## Feature Set: 24 Statistical Features

```python
from liveedge.data import extract_24_features_batch

# Per-axis (X, Y, Z): mean, std, min, max, Q25, Q75, RMS  (7 x 3 = 21)
# Magnitude: mean, std, range                               (3)
# Total: 24 features
```

Design constraints:
1. **ODR-invariant**: Works at any rate (6.25-50Hz) without retraining
2. **FFT-free**: No frequency-domain features, minimal flash usage
3. **Fixed-point friendly**: Simple arithmetic for 16-bit embedded implementation

## Data Requirements

Place pig dataset in `data/processed/50hz/` with:
- `windows.npy`: Accelerometer windows (N, 75, 3) - 1.5s at 50Hz
- `labels.npy`: Behavior labels (N,) - string labels

For cattle validation, place in `data/raw/`:
- `imu_cow_dataset_fall-2022.csv`: Raw cattle IMU data

## Development

```bash
# Run tests
pytest tests/ -v --cov=liveedge

# Code formatting
black src/ tests/

# Linting
ruff src/ tests/ --fix

# Type checking
mypy src/
```

## Cross-Validation Methodology

Two CV approaches are used depending on evaluation purpose:

- **Shuffled CV** (Tables 1, 7, 8): Standard classifier evaluation. Used when measuring model capability in isolation.
- **Temporal-order CV** (Tables 4, 5, 6, 9): Required for FSM simulation. Consecutive test windows maintain temporal adjacency for valid k-hysteresis operation.

## License

MIT License

## Citation

```bibtex
@article{liveedge2026,
  title={LiveEdge: Behavior-Driven Cooperative Optimization for Energy-Efficient Livestock Monitoring},
  author={Zhang, Zhen and Cho, Jin-hee and Ha, Dong S. and Shin, Sook},
  journal={IEEE Journal on Selected Areas in Sensors},
  year={2026}
}
```
