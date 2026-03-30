"""
cluster_eval.py
Dunn Index 계산 모듈 (클러스터 평가지표)

Original : https://github.com/ (외부 참조 코드)
Refactored
  2026-03-30  scipy.spatial.distance 기반 벡터화, 미사용 함수 제거
"""

import numpy as np
from itertools import combinations
from scipy.spatial.distance import pdist, cdist


# --------------------------------------------------------------------------- #
#  Intra-cluster distance (클러스터 내부 거리)
# --------------------------------------------------------------------------- #

def _complete_diameter(X: np.ndarray) -> float:
    """클러스터 내 모든 점 쌍 거리의 최댓값."""
    if X.shape[0] < 2:
        return 0.0
    return pdist(X).max()


def _average_diameter(X: np.ndarray) -> float:
    """클러스터 내 모든 점 쌍 거리의 평균."""
    if X.shape[0] < 2:
        return 0.0
    return pdist(X).mean()


def _centroid_diameter(X: np.ndarray) -> float:
    """중심으로부터 각 점 거리의 평균 × 2."""
    center = X.mean(axis=0, keepdims=True)
    return float(2 * cdist(X, center).mean())


# --------------------------------------------------------------------------- #
#  Inter-cluster distance (클러스터 간 거리)
# --------------------------------------------------------------------------- #

def _single_linkage(X1: np.ndarray, X2: np.ndarray) -> float:
    return float(cdist(X1, X2).min())


def _complete_linkage(X1: np.ndarray, X2: np.ndarray) -> float:
    return float(cdist(X1, X2).max())


def _average_linkage(X1: np.ndarray, X2: np.ndarray) -> float:
    return float(cdist(X1, X2).mean())


def _centroid_linkage(X1: np.ndarray, X2: np.ndarray) -> float:
    return float(np.linalg.norm(X1.mean(axis=0) - X2.mean(axis=0)))


def _avg_centroid_linkage(X1: np.ndarray, X2: np.ndarray) -> float:
    c1 = X1.mean(axis=0, keepdims=True)
    c2 = X2.mean(axis=0, keepdims=True)
    return float(np.concatenate([cdist(X1, c2), cdist(X2, c1)]).mean())


# --------------------------------------------------------------------------- #
#  Dunn Index
# --------------------------------------------------------------------------- #

_INTRA = {
    "cmpl_dd":  _complete_diameter,
    "avdd":     _average_diameter,
    "cent_dd":  _centroid_diameter,
}

_INTER = {
    "sld":         _single_linkage,
    "cmpl_ld":     _complete_linkage,
    "avld":        _average_linkage,
    "cent_ld":     _centroid_linkage,
    "av_cent_ld":  _avg_centroid_linkage,
}


def get_dunn_index(X: np.ndarray, labels: np.ndarray,
                   intra_cluster_distance_type: str = "cmpl_dd",
                   inter_cluster_distance_type: str = "av_cent_ld") -> float:
    """
    Dunn Index = min(inter-cluster distance) / max(intra-cluster distance).

    Parameters
    ----------
    X      : (n_samples, n_features)
    labels : 클러스터 라벨 배열
    intra_cluster_distance_type : 'cmpl_dd' / 'avdd' / 'cent_dd'
    inter_cluster_distance_type : 'sld' / 'cmpl_ld' / 'avld' / 'cent_ld' / 'av_cent_ld'
    """
    intra_fn = _INTRA[intra_cluster_distance_type]
    inter_fn = _INTER[inter_cluster_distance_type]

    unique_labels = np.unique(labels)
    clusters = {lb: X[labels == lb] for lb in unique_labels}

    inter_dists = [
        inter_fn(clusters[i], clusters[j])
        for i, j in combinations(unique_labels, 2)
    ]

    intra_dists = [
        intra_fn(c) if c.shape[0] >= 2 else 0.0
        for c in clusters.values()
    ]

    return float(np.min(inter_dists) / np.max(intra_dists))
