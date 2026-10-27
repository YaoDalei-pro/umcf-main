"""Evaluation metrics of Section III-A, Eqs. (9)–(11).

Uniqueness : FDR, Rank-1, mean/min inter-class distance
Stability  : mean/max intra-class drift, geometric-perturbation drift
Robustness : R_noise = R1(SNR) / R1(ref), R_miss = FDR(rho) / FDR(0)
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch

from .distance import pairwise_squared_euclidean
from .fisher import scatter_traces


EPS = 1e-8


def fdr(z: torch.Tensor, y: torch.Tensor, n_classes: int, eps: float = EPS) -> torch.Tensor:
    """Eq. (9): FDR = tr(S_B) / (tr(S_W) + eps)."""
    tr_b, tr_w = scatter_traces(z, y, n_classes)
    return tr_b / (tr_w + eps)


def rank1_rate(
    z_query: torch.Tensor,
    y_query: torch.Tensor,
    z_gallery: torch.Tensor,
    y_gallery: torch.Tensor,
) -> torch.Tensor:
    """Eq. (10): R_1 = N_correct / N_query, nearest neighbor in Euclidean fingerprint space."""
    d2 = pairwise_squared_euclidean(z_query, z_gallery)
    pred = y_gallery[d2.argmin(dim=1)]
    return (pred == y_query).float().mean()


def class_means(z: torch.Tensor, y: torch.Tensor, n_classes: int) -> torch.Tensor:
    mus = []
    d = z.size(1)
    for c in range(n_classes):
        zc = z[y == c]
        if zc.size(0) == 0:
            mus.append(z.new_zeros(d))
        else:
            mus.append(zc.mean(dim=0))
    return torch.stack(mus, dim=0)


def inter_class_distance_matrix(z: torch.Tensor, y: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Euclidean distances among class-mean fingerprints, shape (C, C)."""
    mu = class_means(z, y, n_classes)
    return torch.cdist(mu, mu, p=2)


def inter_class_distances(z: torch.Tensor, y: torch.Tensor, n_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean and minimum Euclidean distances among class-mean fingerprints."""
    d = inter_class_distance_matrix(z, y, n_classes)
    mask = ~torch.eye(n_classes, dtype=torch.bool, device=z.device)
    vals = d[mask]
    return vals.mean(), vals.min()


def closest_class_pair(z: torch.Tensor, y: torch.Tensor, n_classes: int, names) -> Dict[str, object]:
    """Off-diagonal pair of class means with the smallest Euclidean separation."""
    d = inter_class_distance_matrix(z, y, n_classes)
    best = None
    for i in range(n_classes):
        for j in range(i + 1, n_classes):
            val = float(d[i, j].item())
            if best is None or val < best[0]:
                best = (val, names[i], names[j], i, j)
    return {
        "distance": best[0],
        "a": best[1],
        "b": best[2],
        "matrix": d.cpu().tolist(),
    }


def intra_class_drift(z: torch.Tensor, y: torch.Tensor, n_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mean and max intra-class Euclidean drift (sample to class mean)."""
    mu = class_means(z, y, n_classes)
    means, maxes = [], []
    for c in range(n_classes):
        zc = z[y == c]
        if zc.size(0) == 0:
            continue
        dist = torch.linalg.norm(zc - mu[c], dim=1)
        means.append(dist.mean())
        maxes.append(dist.max())
    mean_drift = torch.stack(means).mean()
    max_drift = torch.stack(maxes).max()
    return mean_drift, max_drift


def geometric_perturbation_drift(z: torch.Tensor, z_pert: torch.Tensor) -> torch.Tensor:
    """Mean ||z - z'||_2 after attitude / solar-phase jitter applied before GI."""
    return torch.linalg.norm(z - z_pert, dim=1).mean()


def uniqueness_metrics(
    z_query: torch.Tensor,
    y_query: torch.Tensor,
    z_gallery: torch.Tensor,
    y_gallery: torch.Tensor,
    n_classes: int,
    eps: float = EPS,
) -> Dict[str, float]:
    z_all = torch.cat([z_gallery, z_query], dim=0)
    y_all = torch.cat([y_gallery, y_query], dim=0)
    mean_d, min_d = inter_class_distances(z_query, y_query, n_classes)
    r1 = rank1_rate(z_query, y_query, z_gallery, y_gallery)
    return {
        "FDR": float(fdr(z_all, y_all, n_classes, eps).item()),
        "Rank-1": float(r1.item()),
        "mean_inter_class": float(mean_d.item()),
        "min_inter_class": float(min_d.item()),
    }


def stability_metrics(
    z: torch.Tensor,
    y: torch.Tensor,
    z_pert: torch.Tensor,
    n_classes: int,
) -> Dict[str, float]:
    mean_d, max_d = intra_class_drift(z, y, n_classes)
    geo = geometric_perturbation_drift(z, z_pert)
    return {
        "mean_intra_class_drift": float(mean_d.item()),
        "max_intra_class_drift": float(max_d.item()),
        "geometric_perturbation_drift": float(geo.item()),
    }


def noise_retention(r1_snr: float, r1_ref: float) -> float:
    """Eq. (11) left: R_noise = R_1(SNR) / R_1(ref)."""
    if r1_ref == 0.0:
        return 0.0
    return float(r1_snr / r1_ref)


def missing_retention(fdr_rho: float, fdr_0: float) -> float:
    """Eq. (11) right: R_miss = FDR(rho) / FDR(0). Equals 1 at rho=0."""
    if fdr_0 == 0.0:
        return 1.0
    return float(fdr_rho / fdr_0)
