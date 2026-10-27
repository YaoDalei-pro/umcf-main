"""Observation-invariant extraction (GI), Section II-B and Fig. 1.

The fast-varying layer does not use raw amplitude sequences. After range
and brightness normalization, GI computes median, MAD, and selected
quantiles of the five multimodal responses, producing x_f in R^{16}.
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

# 16-D layout of x_f (paper Section II-B):
# optical     : median, MAD, q25, q75     -> 4
# radar       : median, MAD, q25, q75     -> 4
# infrared    : median, MAD               -> 2
# polarization: median, MAD               -> 2
# spectral    : median, MAD, q25, q75     -> 4
GI_LAYOUT: Dict[str, Tuple[int, ...]] = {
    "optical": (0, 1, 2, 3),
    "radar": (4, 5, 6, 7),
    "infrared": (8, 9),
    "polarization": (10, 11),
    "spectral": (12, 13, 14, 15),
}
MODALITY_NAMES = ("optical", "radar", "infrared", "polarization", "spectral")
RANGE_EXPONENTS = torch.tensor([2.0, 4.0, 2.0, 2.0, 2.0])  # optical r^2, radar r^4, ...


def _median(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    return x.median(dim=dim).values


def _mad(x: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    med = _median(x, dim=dim).unsqueeze(dim)
    return (x - med).abs().median(dim=dim).values.clamp_min(eps)


def _quantile(x: torch.Tensor, q: float, dim: int = -1) -> torch.Tensor:
    # torch.quantile needs float; keep shape (N,)
    return torch.quantile(x, q, dim=dim)


class ObservationInvariantExtractor(nn.Module):
    """GI: raw multimodal sequences (N, 5, T) -> x_f (N, 16)."""

    def __init__(self, eps: float = 1e-8, robust: bool = True):
        super().__init__()
        self.eps = eps
        self.robust = robust
        self.register_buffer("range_exp", RANGE_EXPONENTS.clone())

    def range_brightness_normalize(
        self,
        sequences: torch.Tensor,
        slant_range: torch.Tensor,
    ) -> torch.Tensor:
        """Range compensation then per-sequence brightness (median) normalization.

        sequences: (N, 5, T)
        slant_range: (N,) in km
        """
        r = slant_range.clamp_min(1.0).unsqueeze(1)  # (N, 1)
        scale = r.pow(self.range_exp.view(1, 5)).unsqueeze(-1)  # (N, 5, 1)
        x = sequences * scale
        if self.robust:
            bright = _median(x, dim=-1).unsqueeze(-1).clamp_min(self.eps)
        else:
            bright = x.mean(dim=-1, keepdim=True).clamp_min(self.eps)
        return x / bright

    def extract_stats(self, x: torch.Tensor) -> torch.Tensor:
        """x: (N, 5, T) normalized sequences -> (N, 16)."""
        opt, rad, ir, pol, spe = x[:, 0], x[:, 1], x[:, 2], x[:, 3], x[:, 4]
        if self.robust:
            feats = [
                _median(opt),
                _mad(opt, eps=self.eps),
                _quantile(opt, 0.25),
                _quantile(opt, 0.75),
                _median(rad),
                _mad(rad, eps=self.eps),
                _quantile(rad, 0.25),
                _quantile(rad, 0.75),
                _median(ir),
                _mad(ir, eps=self.eps),
                _median(pol),
                _mad(pol, eps=self.eps),
                _median(spe),
                _mad(spe, eps=self.eps),
                _quantile(spe, 0.25),
                _quantile(spe, 0.75),
            ]
        else:
            # Ablation without GI: non-robust moments of the same layout.
            feats = [
                opt.mean(-1),
                opt.std(-1).clamp_min(self.eps),
                opt.min(-1).values,
                opt.max(-1).values,
                rad.mean(-1),
                rad.std(-1).clamp_min(self.eps),
                rad.min(-1).values,
                rad.max(-1).values,
                ir.mean(-1),
                ir.std(-1).clamp_min(self.eps),
                pol.mean(-1),
                pol.std(-1).clamp_min(self.eps),
                spe.mean(-1),
                spe.std(-1).clamp_min(self.eps),
                spe.min(-1).values,
                spe.max(-1).values,
            ]
        return torch.stack(feats, dim=-1)

    def forward(
        self,
        sequences: torch.Tensor,
        slant_range: torch.Tensor,
    ) -> torch.Tensor:
        x = self.range_brightness_normalize(sequences, slant_range)
        return self.extract_stats(x)
