"""Fisher discriminant layer weighting, Section II-C, Eqs. (6)–(7)."""

from __future__ import annotations

from typing import Dict, Tuple

import torch


def scatter_traces(z: torch.Tensor, y: torch.Tensor, n_classes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (tr(S_B), tr(S_W)) for a layer matrix z of shape (N, d).

    S_B = sum_c n_c (mu_c - mu)(mu_c - mu)^T
    S_W = sum_c sum_{i in c} (z_i - mu_c)(z_i - mu_c)^T
    so traces reduce to weighted squared Euclidean deviations.
    """
    mu = z.mean(dim=0)
    tr_b = z.new_zeros(())
    tr_w = z.new_zeros(())
    for c in range(n_classes):
        zc = z[y == c]
        n_c = zc.size(0)
        if n_c == 0:
            continue
        mu_c = zc.mean(dim=0)
        tr_b = tr_b + n_c * ((mu_c - mu) ** 2).sum()
        tr_w = tr_w + ((zc - mu_c) ** 2).sum()
    return tr_b, tr_w


def fisher_score(z: torch.Tensor, y: torch.Tensor, n_classes: int, eps: float) -> torch.Tensor:
    """Dimension-normalized Fisher score J_l, Eq. (6)."""
    d_l = max(z.size(1), 1)
    tr_b, tr_w = scatter_traces(z, y, n_classes)
    return (tr_b / (tr_w + eps)) / d_l


def fisher_weights(
    scores: Dict[str, torch.Tensor],
    tau: float,
) -> Dict[str, torch.Tensor]:
    """Softmax mapping w_l = exp(J_l / tau) / sum_k exp(J_k / tau), Eq. (7)."""
    order = ("s", "v", "f", "c")
    j = torch.stack([scores[k] for k in order])
    w = torch.softmax(j / tau, dim=0)
    return {k: w[i] for i, k in enumerate(order)}
