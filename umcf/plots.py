"""Plot helpers for fingerprint matrices and robustness curves (Figs. 5–6 style)."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np


def _maybe_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def plot_class_fingerprints(z: np.ndarray, y: np.ndarray, class_names: List[str], path: str) -> None:
    """Fig. 5 style: four layer panels, rows are prototypes, blue/red are signed responses."""
    plt = _maybe_matplotlib()
    from .prototypes import PROTOTYPES

    layers = (
        ("Static physical", slice(0, 10)),
        ("Slowly varying", slice(10, 20)),
        ("Fast-varying", slice(20, 36)),
        ("Coupling", slice(36, 56)),
    )
    row_labels = []
    for name in class_names:
        meta = PROTOTYPES.get(name, {})
        norad = meta.get("norad", "")
        row_labels.append(f"{norad} {name}".strip())
    means = []
    for c in range(len(class_names)):
        sel = z[y == c]
        means.append(sel.mean(axis=0) if len(sel) else np.zeros(z.shape[1]))
    means = np.stack(means, axis=0)
    vmax = float(np.max(np.abs(means))) or 1.0
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.2))
    for ax, (title, sl) in zip(axes.ravel(), layers):
        block = means[:, sl]
        im = ax.imshow(block, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
        ax.set_title(title, fontsize=10)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.set_xlabel("dimension")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Mean UMCF fingerprints", fontsize=12)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_robustness(noise: Dict, missing: Dict, path: str) -> None:
    plt = _maybe_matplotlib()
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    snr = sorted(float(k) for k in noise)
    r1 = [100.0 * noise[str(s) if str(s) in noise else str(int(s))]["Rank-1"] for s in snr]
    # keys may be '20.0' or '20'
    def _get(d, k, field):
        for cand in (str(k), str(float(k)), str(int(k)) if float(k).is_integer() else str(k)):
            if cand in d:
                return d[cand][field]
        raise KeyError(k)

    r1 = [100.0 * _get(noise, s, "Rank-1") for s in snr]
    ax[0].plot(snr, r1, marker="o")
    ax[0].set_xlabel("SNR (dB)")
    ax[0].set_ylabel("Rank-1 (%)")
    ax[0].set_title("Noise robustness")
    ax[0].grid(True, alpha=0.3)

    rhos = sorted(float(k) for k in missing)
    rmiss = [100.0 * _get(missing, r, "R_miss") for r in rhos]
    ax[1].plot([100 * r for r in rhos], rmiss, marker="o")
    ax[1].set_xlabel("Missing ratio (%)")
    ax[1].set_ylabel("FDR retention (%)")
    ax[1].set_title("Modality-missing robustness")
    ax[1].grid(True, alpha=0.3)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160)
    plt.close(fig)
