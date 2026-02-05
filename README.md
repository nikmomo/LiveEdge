# LiveEdge

Energy-efficient livestock behavior monitoring system using behavior-driven adaptive sampling.

## Overview

LiveEdge is an edge-deployed machine learning system that performs real-time **pig** behavior classification to dynamically adjust accelerometer sampling rates (6.25-25 Hz), achieving ~70% energy reduction while maintaining classification accuracy comparable to fixed-rate baselines.

### Key Features

- **Behavior Classification**: 5-class pig behavior classification (Lying, Eating, Standing, Walking, Drinking)
- **Adaptive Sampling**: 3-cluster ODR mapping with k-hysteresis FSM controller
- **Energy Optimization**: Cycle-accurate energy modeling for nRF52832 + BMI160
- **24 Statistical Features**: MCU-friendly, ODR-invariant feature extraction

## Results

| Configuration | Behavior Acc | Cluster Acc | Energy (mJ/day) | Battery Life |
|--------------|--------------|-------------|-----------------|--------------|
| 50 Hz Baseline | 71.24% | 73.75% | 17,600 | 108 days |
| **LiveEdge (k=3)** | **70.09%** | **79.92%** | **5,266** | **361 days** |

LiveEdge achieves **3.34× battery life extension** with comparable accuracy.

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
├── src/liveedge/         # Core library
│   ├── data/             # Data processing, 24-feature extraction
│   ├── clustering/       # Behavior clustering (STABLE/MODERATE/ACTIVE)
│   ├── models/           # RF, XGBoost, SVM, CNN1D, TCN classifiers
│   ├── sampling/         # FSM controller with k-hysteresis
│   ├── energy/           # Hardware power models (BMI160, nRF52832)
│   ├── evaluation/       # Classification and energy metrics
│   └── simulation/       # Streaming simulation environment
├── scripts/              # Paper reproduction scripts
│   ├── verified_simulation.py      # Main results (Table 4, 5)
│   ├── dl_cluster_accuracy.py      # DL comparison (Table 7)
│   ├── cattle_generalization_study.py  # Cross-species (Table 8)
│   └── figures/          # Paper figure generation
├── tests/                # Unit tests
├── data/                 # Dataset (not included, see below)
└── docs/paper/           # LaTeX source (not included)
```

## Quick Start

### Reproduce Paper Results

```bash
# Main LiveEdge simulation (Tables 4, 5)
python scripts/verified_simulation.py

# DL classifier comparison (Table 7)
python scripts/dl_cluster_accuracy.py

# Cross-species validation (Table 8)
python scripts/cattle_generalization_study.py

# Generate all paper figures
python scripts/figures/generate_paper_figures_v2.py
```

### Data Requirements

Place pig dataset in `data/processed/50hz/` with structure:
- `X.npy`: Accelerometer windows (N, 75, 3) - 1.5s @ 50Hz
- `y.npy`: Behavior labels (N,)
- `pigs.npy`: Animal IDs (N,)

## Cluster Mapping

| Cluster | Behaviors | ODR | Rationale |
|---------|-----------|-----|-----------|
| STABLE | Lying | 6.25 Hz | Low-frequency stationary behavior |
| MODERATE | Eating | 12.5 Hz | Repetitive head movements |
| ACTIVE | Standing, Walking, Drinking | 25 Hz | Dynamic posture/locomotion |

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

## License

MIT License

## Citation

```bibtex
@article{liveedge2026,
  title={LiveEdge: Behavior-Driven Adaptive Sampling for Energy-Efficient Livestock Monitoring},
  author={...},
  journal={IEEE Journal on Selected Areas in Sensors},
  year={2026}
}
```
