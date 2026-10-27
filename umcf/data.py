"""Semi-realistic multimodal dataset generation, loading, and degradation.

Protocol (Section III-A):
- 5 classes, 2000 samples each, train/test 7:3
- static parameters: class-conditional Table 2
- slowly varying: SGP4 ephemerides
- fast-varying: GI on range-normalized multimodal sequences
- noise: additive Gaussian on *standardized* features at a prescribed SNR
- missing: random feature-wise zero-fill, then coupling is recomputed
- geometric perturbation: attitude / solar-phase jitter *before* GI
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .config import UMCFConfig
from .gi import ObservationInvariantExtractor
from .orbit import propagate_slow_features
from .physics import (
    array_to_geometry,
    geometry_to_array,
    perturb_geometry,
    sample_geometry,
    sample_static,
    simulate_sequences,
)
from .prototypes import CLASS_NAMES, CLASS_TO_ID


@dataclass
class SplitData:
    x_s: torch.Tensor
    x_v: torch.Tensor
    x_f: torch.Tensor
    y: torch.Tensor
    sequences: torch.Tensor
    geometry: torch.Tensor
    slant_range: torch.Tensor


class UMCFDataset(Dataset):
    def __init__(self, split: SplitData):
        self.split = split

    def __len__(self) -> int:
        return int(self.split.y.size(0))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {
            "x_s": self.split.x_s[idx],
            "x_v": self.split.x_v[idx],
            "x_f": self.split.x_f[idx],
            "y": self.split.y[idx],
            "sequences": self.split.sequences[idx],
            "geometry": self.split.geometry[idx],
            "slant_range": self.split.slant_range[idx],
        }


def generate_dataset(
    cfg: UMCFConfig,
    seed: int,
    n_per_class: Optional[int] = None,
    use_gi: bool = True,
    verbose: bool = True,
) -> Dict[str, np.ndarray]:
    """Generate the full semi-realistic corpus (all classes, no split)."""
    rng = np.random.default_rng(seed)
    n_pc = int(n_per_class or cfg.data.n_per_class)
    t_len = cfg.data.sequence_length
    xs, xv, seqs, geos, ys, ranges = [], [], [], [], [], []
    gi = ObservationInvariantExtractor(eps=cfg.model.eps, robust=use_gi)

    for cid, name in enumerate(CLASS_NAMES):
        if verbose:
            print(f"[generate] class {name} ({cid + 1}/5), n={n_pc}")
        for _ in range(n_pc):
            x_s = sample_static(name, rng)
            x_v, aux = propagate_slow_features(name, rng)
            geo = sample_geometry(rng, aux["slant_range_km"], geom_jitter_deg=0.0)
            seq = simulate_sequences(x_s, name, geo, rng, t_len=t_len)
            xs.append(x_s)
            xv.append(x_v)
            seqs.append(seq)
            geos.append(geometry_to_array(geo))
            ys.append(cid)
            ranges.append(geo.slant_range_km)

    x_s = np.stack(xs).astype(np.float32)
    x_v = np.stack(xv).astype(np.float32)
    sequences = np.stack(seqs).astype(np.float32)
    geometry = np.stack(geos).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    slant = np.asarray(ranges, dtype=np.float32)

    with torch.no_grad():
        x_f = (
            gi(
                torch.from_numpy(sequences),
                torch.from_numpy(slant),
            )
            .cpu()
            .numpy()
            .astype(np.float32)
        )

    return {
        "x_s": x_s,
        "x_v": x_v,
        "x_f": x_f,
        "y": y,
        "sequences": sequences,
        "geometry": geometry,
        "slant_range": slant,
        "class_names": np.array(CLASS_NAMES),
        "seed": np.array([seed]),
        "use_gi": np.array([int(use_gi)]),
    }


def save_dataset(data: Dict[str, np.ndarray], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **data)


def load_dataset(path: str) -> Dict[str, np.ndarray]:
    raw = np.load(path, allow_pickle=True)
    return {k: raw[k] for k in raw.files}


def stratified_split(
    y: np.ndarray,
    train_ratio: float,
    seed: int,
    val_ratio_of_train: float = 0.1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return train, val, test index arrays. Val is inner 10% of the training split."""
    rng = np.random.default_rng(seed)
    train_idx, val_idx, test_idx = [], [], []
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_train_all = int(round(len(idx) * train_ratio))
        n_val = int(round(n_train_all * val_ratio_of_train))
        n_fit = n_train_all - n_val
        train_idx.append(idx[:n_fit])
        val_idx.append(idx[n_fit:n_train_all])
        test_idx.append(idx[n_train_all:])
    return (
        np.concatenate(train_idx),
        np.concatenate(val_idx),
        np.concatenate(test_idx),
    )


def clip_to_train_range(train: torch.Tensor, test: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Clip each test coordinate to the training min/max.

    Released-code check in RESULTS: a near-constant fast-layer coordinate must
    not dominate an out-of-range test sample. Training values are unchanged.
    """
    lo = train.min(dim=0).values
    hi = train.max(dim=0).values
    hi = torch.where(hi > lo, hi, lo)
    return train, torch.max(torch.min(test, hi), lo)


def pack_split(data: Dict[str, np.ndarray], idx: np.ndarray, device: str = "cpu") -> SplitData:
    def t(key, dtype=None):
        arr = data[key][idx]
        ten = torch.from_numpy(np.ascontiguousarray(arr))
        if dtype is not None:
            ten = ten.to(dtype)
        return ten.to(device)

    return SplitData(
        x_s=t("x_s", torch.float32),
        x_v=t("x_v", torch.float32),
        x_f=t("x_f", torch.float32),
        y=t("y", torch.long),
        sequences=t("sequences", torch.float32),
        geometry=t("geometry", torch.float32),
        slant_range=t("slant_range", torch.float32),
    )


def add_snr_noise_on_standardized(
    z: torch.Tensor,
    snr_db: float,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Additive Gaussian noise on standardized features (Section III-A).

    SNR_dB = 10 log10(P_signal / P_noise), P_signal = mean(z^2).
    """
    p_sig = z.pow(2).mean().clamp_min(1e-12)
    p_noise = p_sig / (10.0 ** (snr_db / 10.0))
    std = torch.sqrt(p_noise)
    noise = torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=rng) * std
    return z + noise


def apply_feature_missing(
    x_s: torch.Tensor,
    x_v: torch.Tensor,
    x_f: torch.Tensor,
    rho: float,
    generator: Optional[torch.Generator] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random feature-wise zero-fill of the concatenated (x_s, x_v, x_f)."""
    if rho <= 0.0:
        return x_s, x_v, x_f
    xs, xv, xf = x_s.clone(), x_v.clone(), x_f.clone()
    x = torch.cat([xs, xv, xf], dim=-1)
    mask = torch.rand(x.shape, device=x.device, generator=generator) < rho
    x = x.masked_fill(mask, 0.0)
    d_s, d_v = xs.size(1), xv.size(1)
    return x[:, :d_s], x[:, d_s : d_s + d_v], x[:, d_s + d_v :]


def recompute_xf_from_sequences(
    sequences: torch.Tensor,
    slant_range: torch.Tensor,
    use_gi: bool,
    eps: float,
) -> torch.Tensor:
    gi = ObservationInvariantExtractor(eps=eps, robust=use_gi).to(sequences.device)
    with torch.no_grad():
        return gi(sequences, slant_range)


def geometric_perturbed_xf(
    data: SplitData,
    class_names,
    rng: np.random.Generator,
    jitter_deg: float,
    t_len: int,
    use_gi: bool,
    eps: float,
    device: str,
) -> torch.Tensor:
    """Re-simulate sequences with attitude / solar-phase jitter, then GI."""
    n = data.y.size(0)
    seqs = []
    for i in range(n):
        x_s = data.x_s[i].cpu().numpy()
        geo = array_to_geometry(data.geometry[i].cpu().numpy())
        geo_p = perturb_geometry(geo, rng, jitter_deg)
        name = CLASS_NAMES[int(data.y[i].item())]
        seqs.append(simulate_sequences(x_s, name, geo_p, rng, t_len=t_len))
    sequences = torch.from_numpy(np.stack(seqs)).to(device=device, dtype=torch.float32)
    slant = data.slant_range.to(device)
    return recompute_xf_from_sequences(sequences, slant, use_gi=use_gi, eps=eps)
