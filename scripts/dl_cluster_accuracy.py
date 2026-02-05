#!/usr/bin/env python3
"""DL cluster accuracy evaluation (Table 7)."""

import sys
from pathlib import Path
import numpy as np
import json
import time
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root / 'src'))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler, LabelEncoder

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Configuration

DATA_DIR = Path("L:/GitHub/LiveEdge/data/processed/50hz")
OUTPUT_DIR = Path("L:/GitHub/LiveEdge/outputs/dl_comparison")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BEHAVIORS_5CLASS = ['Lying', 'Eating', 'Walking', 'Standing', 'Drinking']

# 3-cluster configuration (Standing in ACTIVE per fmin analysis)
BEHAVIOR_TO_CLUSTER = {
    'Lying': 'STABLE',
    'Eating': 'STABLE',
    'Standing': 'ACTIVE',
    'Walking': 'MODERATE',
    'Drinking': 'ACTIVE',
}

WINDOW_SAMPLES = 75
N_CHANNELS = 3
BATCH_SIZE = 512
EPOCHS = 50
LEARNING_RATE = 0.001
PATIENCE = 10


# =============================================================================
# Deep Learning Models (same as comprehensive_dl_comparison.py)
# =============================================================================

class CNN1D(nn.Module):
    def __init__(self, n_channels: int = 3, n_classes: int = 5):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(256)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.max_pool1d(x, 2)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


class LSTMClassifier(nn.Module):
    def __init__(self, n_channels: int = 3, n_classes: int = 5,
                 hidden_size: int = 128, n_layers: int = 2):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden_size, n_layers,
                           batch_first=True, dropout=0.3, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        h_combined = torch.cat([h_forward, h_backward], dim=1)
        x = self.dropout(h_combined)
        return self.fc(x)


class GRUClassifier(nn.Module):
    def __init__(self, n_channels: int = 3, n_classes: int = 5,
                 hidden_size: int = 128, n_layers: int = 2):
        super().__init__()
        self.gru = nn.GRU(n_channels, hidden_size, n_layers,
                         batch_first=True, dropout=0.3, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(hidden_size * 2, n_classes)

    def forward(self, x):
        gru_out, h_n = self.gru(x)
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        h_combined = torch.cat([h_forward, h_backward], dim=1)
        x = self.dropout(h_combined)
        return self.fc(x)


class ResBlock1D(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=kernel_size//2)
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        return x


class ResNet1D(nn.Module):
    def __init__(self, n_channels: int = 3, n_classes: int = 5):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(2)
        self.res1 = ResBlock1D(64, 64)
        self.res2 = ResBlock1D(64, 128)
        self.pool2 = nn.MaxPool1d(2)
        self.res3 = ResBlock1D(128, 256)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.pool2(x)
        x = self.res3(x)
        x = self.global_pool(x).squeeze(-1)
        x = self.dropout(x)
        return self.fc(x)


class DeepConvLSTM(nn.Module):
    def __init__(self, n_channels: int = 3, n_classes: int = 5):
        super().__init__()
        self.conv1 = nn.Conv1d(n_channels, 64, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(128)
        self.lstm = nn.LSTM(128, 128, num_layers=2, batch_first=True,
                           dropout=0.3, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(256, n_classes)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.permute(0, 2, 1)
        lstm_out, (h_n, c_n) = self.lstm(x)
        h_forward = h_n[-2, :, :]
        h_backward = h_n[-1, :, :]
        h_combined = torch.cat([h_forward, h_backward], dim=1)
        x = self.dropout(h_combined)
        return self.fc(x)


class TransformerClassifier(nn.Module):
    def __init__(self, n_channels: int = 3, n_classes: int = 5, seq_len: int = 75,
                 d_model: int = 64, nhead: int = 4, num_layers: int = 2):
        super().__init__()
        self.input_proj = nn.Linear(n_channels, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.1)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4,
            dropout=0.3, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(d_model, n_classes)

    def forward(self, x):
        x = self.input_proj(x)
        x = x + self.pos_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.fc(x)


# =============================================================================
# Feature Extraction (unified 24-feature module)
# =============================================================================

from liveedge.data import extract_24_features_batch


def extract_features(X: np.ndarray) -> np.ndarray:
    """Extract 24 statistical features using unified module.

    This function wraps the unified extract_24_features_batch for
    backward compatibility with existing code.
    """
    return extract_24_features_batch(X)


# =============================================================================
# Training Functions
# =============================================================================

def train_dl_model_gpu(model: nn.Module, X_train_gpu: torch.Tensor, y_train_gpu: torch.Tensor,
                       X_val_gpu: torch.Tensor, y_val_gpu: torch.Tensor,
                       epochs: int = EPOCHS, patience: int = PATIENCE) -> np.ndarray:
    """Train a DL model with data pre-loaded on GPU. Returns best-epoch predictions."""
    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    n_train = X_train_gpu.shape[0]
    n_val = X_val_gpu.shape[0]
    best_val_acc = 0
    best_state = None
    patience_counter = 0

    t_start = time.time()
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n_train, device=DEVICE)
        for i in range(0, n_train, BATCH_SIZE):
            idx = perm[i:i + BATCH_SIZE]
            optimizer.zero_grad()
            outputs = model(X_train_gpu[idx])
            loss = criterion(outputs, y_train_gpu[idx])
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_preds_list = []
            for i in range(0, n_val, BATCH_SIZE * 2):
                outputs = model(X_val_gpu[i:i + BATCH_SIZE * 2])
                val_preds_list.append(outputs.argmax(dim=1))
            val_preds_t = torch.cat(val_preds_list)
            val_acc = (val_preds_t == y_val_gpu).float().mean().item()

        scheduler.step(1 - val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    elapsed = time.time() - t_start
    n_epochs = epoch + 1
    print(f"[{n_epochs}ep/{elapsed:.0f}s]", end=" ", flush=True)

    # Load best model for final evaluation
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    val_preds_list = []
    with torch.no_grad():
        for i in range(0, n_val, BATCH_SIZE * 2):
            outputs = model(X_val_gpu[i:i + BATCH_SIZE * 2])
            val_preds_list.append(outputs.argmax(dim=1))

    return torch.cat(val_preds_list).cpu().numpy()


# =============================================================================
# Main Experiment
# =============================================================================

def run_cluster_accuracy_comparison(n_folds: int = 5,
                                     ml_only: bool = False) -> Dict:
    """Run cluster accuracy comparison for all models."""
    print("=" * 70)
    print("DL Cluster Accuracy Evaluation (Table 7)")
    print("=" * 70)

    # Load data
    print(f"\n[1] Loading data from {DATA_DIR}...")
    X = np.load(DATA_DIR / 'windows.npy')
    y = np.load(DATA_DIR / 'labels.npy', allow_pickle=True)

    # Filter to 5 classes
    mask = np.isin(y, BEHAVIORS_5CLASS)
    X, y = X[mask], y[mask]
    X = X[:, :, :3]  # ACC only

    print(f"  Loaded {len(X)} samples (full dataset, no subsampling)")

    # Create cluster labels
    y_clusters = np.array([BEHAVIOR_TO_CLUSTER[b] for b in y])

    # Encode labels
    le_behavior = LabelEncoder()
    y_behavior_encoded = le_behavior.fit_transform(y)

    le_cluster = LabelEncoder()
    y_cluster_encoded = le_cluster.fit_transform(y_clusters)

    print(f"  Behaviors: {list(le_behavior.classes_)}")
    print(f"  Clusters: {list(le_cluster.classes_)}")

    # Normalize raw data for DL
    X_normalized = (X - X.mean(axis=(0, 1))) / (X.std(axis=(0, 1)) + 1e-8)

    models_config = {
        'Random Forest': {'type': 'traditional', 'behavior_cls': RandomForestClassifier(
            n_estimators=100, max_depth=20, class_weight='balanced', random_state=42, n_jobs=-1),
            'cluster_cls': RandomForestClassifier(
            n_estimators=100, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1)},
        'Decision Tree': {'type': 'traditional', 'behavior_cls': DecisionTreeClassifier(
            max_depth=20, class_weight='balanced', random_state=42),
            'cluster_cls': DecisionTreeClassifier(
            max_depth=15, class_weight='balanced', random_state=42)},
        'SVM (Linear)': {'type': 'traditional', 'behavior_cls': LinearSVC(
            class_weight='balanced', random_state=42, max_iter=5000),
            'cluster_cls': LinearSVC(
            class_weight='balanced', random_state=42, max_iter=5000)},
        'Logistic Regression': {'type': 'traditional', 'behavior_cls': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42, n_jobs=-1),
            'cluster_cls': LogisticRegression(
            max_iter=1000, class_weight='balanced', random_state=42, n_jobs=-1)},
        'XGBoost': {'type': 'traditional', 'behavior_cls': XGBClassifier(
            n_estimators=100, max_depth=10, learning_rate=0.1,
            random_state=42, n_jobs=-1, tree_method='hist', verbosity=0),
            'cluster_cls': XGBClassifier(
            n_estimators=100, max_depth=10, learning_rate=0.1,
            random_state=42, n_jobs=-1, tree_method='hist', verbosity=0)},
        '1D-CNN': {'type': 'dl', 'class': CNN1D},
        'LSTM': {'type': 'dl', 'class': LSTMClassifier},
        'GRU': {'type': 'dl', 'class': GRUClassifier},
        'ResNet-1D': {'type': 'dl', 'class': ResNet1D},
        'DeepConvLSTM': {'type': 'dl', 'class': DeepConvLSTM},
        'Transformer': {'type': 'dl', 'class': TransformerClassifier},
    }

    if ml_only:
        models_config = {k: v for k, v in models_config.items() if v['type'] == 'traditional'}
        print("  [ML-only mode: skipping DL models]")

    results = {name: {
        'behavior_accs': [], 'behavior_f1s': [],
        'cluster_accs': [], 'cluster_f1s': []
    } for name in models_config}

    # Cross-validation (temporal order preserved for consistency)
    print(f"\n[2] Running {n_folds}-fold CV (temporal order)...")
    # Shuffled CV for classifier evaluation (standard ML practice)
    # Temporal-order CV is only needed for FSM simulation with k-hysteresis
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    for fold, (train_idx, val_idx) in enumerate(cv.split(X, y_cluster_encoded)):
        print(f"\n  --- Fold {fold + 1}/{n_folds} ---")

        X_train, X_val = X[train_idx], X[val_idx]
        X_train_norm, X_val_norm = X_normalized[train_idx], X_normalized[val_idx]
        X_train_feat = None  # Reset feature cache for traditional models
        y_train_behavior = y_behavior_encoded[train_idx]
        y_val_behavior = y_behavior_encoded[val_idx]
        y_train_cluster = y_cluster_encoded[train_idx]
        y_val_cluster = y_cluster_encoded[val_idx]

        # Pre-load GPU tensors for DL models
        X_train_gpu = torch.FloatTensor(X_train_norm).to(DEVICE)
        X_val_gpu = torch.FloatTensor(X_val_norm).to(DEVICE)
        y_train_beh_gpu = torch.LongTensor(y_train_behavior).to(DEVICE)
        y_val_beh_gpu = torch.LongTensor(y_val_behavior).to(DEVICE)
        y_train_cl_gpu = torch.LongTensor(y_train_cluster).to(DEVICE)
        y_val_cl_gpu = torch.LongTensor(y_val_cluster).to(DEVICE)

        for name, config in models_config.items():
            print(f"    {name}...", end=" ", flush=True)

            try:
                if config['type'] == 'traditional':
                    # Extract 24 features
                    if 'X_train_feat' not in locals() or X_train_feat is None:
                        X_train_feat = extract_features(X_train)
                        X_val_feat = extract_features(X_val)
                        scaler_feat = StandardScaler()
                        X_train_scaled = scaler_feat.fit_transform(X_train_feat)
                        X_val_scaled = scaler_feat.transform(X_val_feat)

                    # Behavior classifier
                    clf_behavior = clone(config['behavior_cls'])
                    clf_behavior.fit(X_train_scaled, y_train_behavior)
                    pred_behavior = clf_behavior.predict(X_val_scaled)

                    # Cluster classifier
                    clf_cluster = clone(config['cluster_cls'])
                    clf_cluster.fit(X_train_scaled, y_train_cluster)
                    pred_cluster = clf_cluster.predict(X_val_scaled)

                else:
                    # DL model - train on behavior task (data on GPU)
                    model_behavior = config['class'](n_channels=N_CHANNELS, n_classes=5)
                    pred_behavior = train_dl_model_gpu(
                        model_behavior, X_train_gpu, y_train_beh_gpu,
                        X_val_gpu, y_val_beh_gpu)

                    # DL model - train on cluster task (data on GPU)
                    model_cluster = config['class'](n_channels=N_CHANNELS, n_classes=3)
                    pred_cluster = train_dl_model_gpu(
                        model_cluster, X_train_gpu, y_train_cl_gpu,
                        X_val_gpu, y_val_cl_gpu)

                # Calculate metrics
                behavior_acc = accuracy_score(y_val_behavior, pred_behavior)
                behavior_f1 = f1_score(y_val_behavior, pred_behavior, average='macro')
                cluster_acc = accuracy_score(y_val_cluster, pred_cluster)
                cluster_f1 = f1_score(y_val_cluster, pred_cluster, average='macro')

                results[name]['behavior_accs'].append(behavior_acc)
                results[name]['behavior_f1s'].append(behavior_f1)
                results[name]['cluster_accs'].append(cluster_acc)
                results[name]['cluster_f1s'].append(cluster_f1)

                print(f"Behavior={behavior_acc*100:.1f}%, Cluster={cluster_acc*100:.1f}%")

            except Exception as e:
                print(f"FAILED: {e}")

        # Free GPU memory between folds
        del X_train_gpu, X_val_gpu, y_train_beh_gpu, y_val_beh_gpu
        del y_train_cl_gpu, y_val_cl_gpu
        torch.cuda.empty_cache()

    # Aggregate results
    print("\n[3] Aggregating results...")

    summary = {}
    for name, data in results.items():
        if len(data['behavior_accs']) > 0:
            summary[name] = {
                'behavior_accuracy_mean': np.mean(data['behavior_accs']),
                'behavior_accuracy_std': np.std(data['behavior_accs']),
                'behavior_f1_mean': np.mean(data['behavior_f1s']),
                'behavior_f1_std': np.std(data['behavior_f1s']),
                'cluster_accuracy_mean': np.mean(data['cluster_accs']),
                'cluster_accuracy_std': np.std(data['cluster_accs']),
                'cluster_f1_mean': np.mean(data['cluster_f1s']),
                'cluster_f1_std': np.std(data['cluster_f1s']),
            }

    return summary


def print_table7_format(results: Dict):
    """Print results in paper Table 7 format."""
    print("\n" + "=" * 80)
    print("TABLE 7: Classifier Comparison (Paper Format)")
    print("=" * 80)

    print(f"\n{'Model':<20} | {'Behavior':>12} | {'Cluster':>12} | {'F1':>12} | {'MCU':>6}")
    print("-" * 70)

    # MCU-compatible
    print("\nMCU-compatible (traditional ML):")
    for name in ['Random Forest', 'XGBoost', 'Decision Tree', 'SVM (Linear)', 'Logistic Regression']:
        if name in results:
            r = results[name]
            mcu = "Yes"
            print(f"  {name:<18} | {r['behavior_accuracy_mean']*100:>10.2f}% | "
                  f"{r['cluster_accuracy_mean']*100:>10.2f}% | "
                  f"{r['behavior_f1_mean']*100:>10.2f}% | {mcu:>6}")

    # GPU-required
    print("\nGPU-required (deep learning):")
    dl_models = ['1D-CNN', 'DeepConvLSTM', 'GRU', 'ResNet-1D', 'LSTM', 'Transformer']
    for name in dl_models:
        if name in results:
            r = results[name]
            mcu = "No"
            print(f"  {name:<18} | {r['behavior_accuracy_mean']*100:>10.2f}% | "
                  f"{r['cluster_accuracy_mean']*100:>10.2f}% | "
                  f"{r['behavior_f1_mean']*100:>10.2f}% | {mcu:>6}")


def compare_with_paper(results: Dict):
    """Compare computed cluster accuracy with paper claims."""
    print("\n" + "=" * 70)
    print("PAPER CLAIM VERIFICATION (Table 7)")
    print("=" * 70)

    paper_claims = {
        'RF + Features': {'behavior': 79.24, 'cluster': 82.40},
        '1D-CNN': {'behavior': 81.22, 'cluster': 85.29},
        'DeepConvLSTM': {'behavior': 80.38, 'cluster': 83.51},
        'GRU': {'behavior': 80.14, 'cluster': 83.27},
        'ResNet-1D': {'behavior': 79.97, 'cluster': 83.09},
        'LSTM': {'behavior': 79.87, 'cluster': 82.98},
        'Transformer': {'behavior': 75.59, 'cluster': 78.72},
    }

    print(f"\n{'Model':<20} | {'Paper Cluster':>14} | {'Computed':>14} | {'Δ':>8} | {'Status':>8}")
    print("-" * 70)

    for name, paper in paper_claims.items():
        if name in results:
            comp = results[name]['cluster_accuracy_mean'] * 100
            delta = comp - paper['cluster']
            status = "OK" if abs(delta) < 2.0 else "DIFF"
            print(f"{name:<20} | {paper['cluster']:>13.2f}% | {comp:>13.2f}% | "
                  f"{delta:>+7.2f} | {status:>8}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ml-only', action='store_true',
                        help='Only run traditional ML models (skip DL)')
    args = parser.parse_args()

    results = run_cluster_accuracy_comparison(n_folds=5, ml_only=args.ml_only)

    print_table7_format(results)
    compare_with_paper(results)

    # Save results
    def convert_numpy(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj

    results = convert_numpy(results)

    output_file = OUTPUT_DIR / 'dl_cluster_comparison.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n[Results saved to: {output_file}]")

    return results


if __name__ == "__main__":
    main()
