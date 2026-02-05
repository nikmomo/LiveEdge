#!/usr/bin/env python3
"""
Calculate cluster accuracy for all baselines.
This helps compare LiveEdge fairly against fixed-rate baselines.
"""

import sys
import numpy as np
from pathlib import Path

_script_dir = Path(__file__).parent
_project_root = _script_dir.parent
sys.path.insert(0, str(_project_root / 'src'))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Import unified 24-feature extraction
from liveedge.data import extract_24_features_batch

# Load data
DATA_DIR = Path("L:/GitHub/LiveEdge/data/processed/50hz")
X = np.load(DATA_DIR / 'windows.npy')[:, :, :3]
y = np.load(DATA_DIR / 'labels.npy', allow_pickle=True)

# Filter to 5 classes
valid_mask = np.array([b != 'Interacting' for b in y])
X, y = X[valid_mask], y[valid_mask]

# Corrected cluster mapping
BEHAVIOR_TO_CLUSTER = {
    'Lying': 'STABLE',
    'Eating': 'STABLE',
    'Standing': 'ACTIVE',
    'Walking': 'MODERATE',
    'Drinking': 'ACTIVE',
}


def extract_features(X):
    """Extract 24 statistical features using unified module."""
    return extract_24_features_batch(X)

def resample(X, odr, source=50):
    if odr >= source:
        return X
    factor = int(source / odr)
    return X[:, ::factor, :]

# Encode
le_behavior = LabelEncoder()
y_behavior = le_behavior.fit_transform(y)

y_cluster = np.array([BEHAVIOR_TO_CLUSTER[b] for b in y])
le_cluster = LabelEncoder()
y_cluster_encoded = le_cluster.fit_transform(y_cluster)

print('='*70)
print('BASELINE CLUSTER ACCURACY ANALYSIS')
print('='*70)
print(f'Cluster mapping: {BEHAVIOR_TO_CLUSTER}')
print(f'Clusters: {list(le_cluster.classes_)}')
print()

# 5-fold CV (temporal order preserved for FSM validity)
cv = StratifiedKFold(n_splits=5, shuffle=False)
results = {odr: {'behavior': [], 'cluster_from_behavior': [], 'cluster_direct': []}
           for odr in [50, 25, 12.5, 10, 6.25]}

for fold, (train_idx, test_idx) in enumerate(cv.split(X, y_cluster_encoded)):
    print(f"Fold {fold+1}/5...", end=" ", flush=True)
    X_train, X_test = X[train_idx], X[test_idx]
    y_train_behavior, y_test_behavior = y_behavior[train_idx], y_behavior[test_idx]
    y_train_cluster, y_test_cluster = y_cluster_encoded[train_idx], y_cluster_encoded[test_idx]
    y_test_behavior_str = le_behavior.inverse_transform(y_test_behavior)

    for odr in [50, 25, 12.5, 10, 6.25]:
        # Resample
        X_train_r = resample(X_train, odr)
        X_test_r = resample(X_test, odr)

        # Extract features
        feat_train = extract_features(X_train_r)
        feat_test = extract_features(X_test_r)

        scaler = StandardScaler()
        feat_train = scaler.fit_transform(feat_train)
        feat_test = scaler.transform(feat_test)

        # Train behavior classifier
        clf_behavior = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42, n_jobs=-1)
        clf_behavior.fit(feat_train, y_train_behavior)
        pred_behavior = clf_behavior.predict(feat_test)
        pred_behavior_str = le_behavior.inverse_transform(pred_behavior)

        # Behavior accuracy
        behavior_acc = accuracy_score(y_test_behavior, pred_behavior)
        results[odr]['behavior'].append(behavior_acc)

        # Cluster accuracy from behavior predictions
        pred_clusters_from_behavior = [BEHAVIOR_TO_CLUSTER[b] for b in pred_behavior_str]
        true_clusters = [BEHAVIOR_TO_CLUSTER[b] for b in y_test_behavior_str]
        cluster_acc_from_behavior = accuracy_score(true_clusters, pred_clusters_from_behavior)
        results[odr]['cluster_from_behavior'].append(cluster_acc_from_behavior)

        # Direct cluster classifier
        clf_cluster = RandomForestClassifier(n_estimators=100, max_depth=15, class_weight='balanced', random_state=42, n_jobs=-1)
        clf_cluster.fit(feat_train, y_train_cluster)
        pred_cluster = clf_cluster.predict(feat_test)
        cluster_acc_direct = accuracy_score(y_test_cluster, pred_cluster)
        results[odr]['cluster_direct'].append(cluster_acc_direct)

    print("Done")

print()
print("="*70)
print("RESULTS SUMMARY")
print("="*70)
print()
print(f"{'ODR':<10} | {'Behavior Acc':>14} | {'Cluster (from Beh)':>18} | {'Cluster (Direct)':>16}")
print('-'*65)
for odr in [50, 25, 12.5, 10, 6.25]:
    beh = np.mean(results[odr]['behavior']) * 100
    clust_beh = np.mean(results[odr]['cluster_from_behavior']) * 100
    clust_dir = np.mean(results[odr]['cluster_direct']) * 100
    print(f'{odr:>6} Hz | {beh:>13.2f}% | {clust_beh:>17.2f}% | {clust_dir:>15.2f}%')

print()
print("="*70)
print("KEY INSIGHTS")
print("="*70)
print()
print("1. Cluster accuracy is MUCH higher than behavior accuracy")
print("   (3-class is easier than 5-class)")
print()
print("2. 'Cluster from Behavior' = map behavior predictions to clusters")
print("   'Cluster Direct' = train a 3-class cluster classifier")
print()
print("3. For fair comparison with LiveEdge, use 'Cluster Direct' accuracy")
