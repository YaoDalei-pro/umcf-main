"""Decomposable Euclidean distance, Section II-D, Eq. (8).

After square-root Fisher weighting, the unified fingerprint satisfies
    ||z^{(i)} - z^{(j)}||_2^2
        = sum_{l in {s,v,f,c}} w_l ||z_l^{(i)} - z_l^{(j)}||_2^2.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch


LAYER_SLICES = {
    "s": slice(0, 10),
    "v": slice(10, 20),
    "f": slice(20, 36),
    "c": slice(36, 56),
}


def split_layers(
    z: torch.Tensor,
    d_s: int = 10,
    d_v: int = 10,
    d_f: int = 16,
    d_c: int = 20,
) -> Dict[str, torch.Tensor]:
    i0, i1 = 0, d_s
    i2 = i1 + d_v
    i3 = i2 + d_f
    i4 = i3 + d_c
    return {"s": z[..., i0:i1], "v": z[..., i1:i2], "f": z[..., i2:i3], "c": z[..., i3:i4]}


def renormalized_layer_weights(
    weights: Dict[str, torch.Tensor],
    omit: Tuple[str, ...] = ("s",),
) -> Dict[str, torch.Tensor]:
    """Drop unavailable layers and renormalize the remaining Fisher weights.

    Discussion rule: when static coordinates are unknown, omit that layer.
    Zero-filling it against a library that still stores static coordinates is
    not the intended missing-data rule.
    """
    active = [k for k in ("s", "v", "f", "c") if k not in omit]
    total = sum(weights[k] for k in active)
    return {k: weights[k] / total for k in active}


def pairwise_squared_euclidean(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """D2[i, j] = ||a_i - b_j||^2, shapes (N, d) and (M, d) -> (N, M)."""
    return (a.pow(2).sum(dim=1, keepdim=True) + b.pow(2).sum(dim=1) - 2.0 * a @ b.T).clamp_min(0.0)


def decomposable_squared_distance(
    z_i: torch.Tensor,
    z_j: torch.Tensor,
    weights: Optional[Dict[str, torch.Tensor]] = None,
    d_s: int = 10,
    d_v: int = 10,
    d_f: int = 16,
    d_c: int = 20,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Return total D^2 and per-layer weighted contributions.

    z_i, z_j: (d,) or (N, d) fingerprints already containing sqrt(w_l) z_l.
    If weights is None, contributions are the squared diffs of the concatenated
    sub-vectors (already weighted). If weights is provided, z_* are interpreted
    as *unweighted* layer fingerprints and Eq. (8) is applied explicitly.
    """
    if z_i.dim() == 1:
        z_i = z_i.unsqueeze(0)
    if z_j.dim() == 1:
        z_j = z_j.unsqueeze(0)

    if weights is None:
        li = split_layers(z_i, d_s, d_v, d_f, d_c)
        lj = split_layers(z_j, d_s, d_v, d_f, d_c)
        contrib = {k: (li[k] - lj[k]).pow(2).sum(dim=-1) for k in li}
        total = sum(contrib.values())
        return total, contrib

    # Unweighted layers concatenated as [z_s, z_v, z_f, z_c]
    li = split_layers(z_i, d_s, d_v, d_f, d_c)
    lj = split_layers(z_j, d_s, d_v, d_f, d_c)
    contrib = {k: weights[k] * (li[k] - lj[k]).pow(2).sum(dim=-1) for k in li}
    total = sum(contrib.values())
    return total, contrib
