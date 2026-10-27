"""Comparative methods in Section III-B / Table 4.

FC, LMNN, CMAF share the same standardized concatenated feature space
[z_s, z_v, z_f] (36-D). SMO uses optical GI dimensions only. SVM-RBF and
XGBoost classify that 36-D vector; DNNF learns a nonlinear embedding.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .distance import pairwise_squared_euclidean
from .metrics import uniqueness_metrics


def standardize_concat(x_s, x_v, x_f, mu, sigma, eps: float):
    x = torch.cat([x_s, x_v, x_f], dim=-1)
    return (x - mu) / (sigma + eps)


def fit_standardizer(x_s, x_v, x_f):
    x = torch.cat([x_s, x_v, x_f], dim=-1)
    mu = x.mean(0)
    sigma = x.std(0, unbiased=False).clamp_min(1e-12)
    return mu, sigma


class _MLP(nn.Module):
    def __init__(self, dims, n_classes: int):
        super().__init__()
        layers = []
        for a, b in zip(dims[:-1], dims[1:]):
            layers += [nn.Linear(a, b), nn.ReLU(inplace=True)]
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(dims[-1], n_classes)

    def forward(self, x, return_embed=False):
        h = self.backbone(x)
        if return_embed:
            return h, self.head(h)
        return self.head(h)


def _sgd_train(model, x, y, epochs, lr, batch_size, device):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = x.size(0)
    model.train()
    for _ in range(epochs):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, batch_size):
            idx = perm[i : i + batch_size]
            loss = F.cross_entropy(model(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()


class DNNF(nn.Module):
    """Deep nonlinear fusion: MLP on the 36-D concatenation."""

    def __init__(self, d_in: int, hidden, d_embed: int, n_classes: int):
        super().__init__()
        dims = [d_in] + list(hidden) + [d_embed]
        self.net = _MLP(dims, n_classes)

    def embed(self, x):
        h, _ = self.net(x, return_embed=True)
        return h

    def fit(self, x, y, epochs, lr, batch_size, device):
        _sgd_train(self.net, x, y, epochs, lr, batch_size, device)
        return self


class CMAF(nn.Module):
    """Cross-modal quality-aware fusion: per-layer gates on (z_s, z_v, z_f)."""

    def __init__(self, d_s=10, d_v=10, d_f=16, n_classes=5, d_embed=36):
        super().__init__()
        self.d_s, self.d_v, self.d_f = d_s, d_v, d_f
        self.gate = nn.Sequential(nn.Linear(d_s + d_v + d_f, 32), nn.ReLU(), nn.Linear(32, 3))
        self.proj = nn.Linear(d_s + d_v + d_f, d_embed)
        self.head = nn.Linear(d_embed, n_classes)

    def embed(self, x):
        zs, zv, zf = x[:, : self.d_s], x[:, self.d_s : self.d_s + self.d_v], x[:, self.d_s + self.d_v :]
        g = torch.softmax(self.gate(x), dim=-1)
        fused = torch.cat([g[:, 0:1] * zs, g[:, 1:2] * zv, g[:, 2:3] * zf], dim=-1)
        return self.proj(fused)

    def forward(self, x):
        return self.head(self.embed(x))

    def fit(self, x, y, epochs, lr, batch_size, device):
        _sgd_train(self, x, y, epochs, lr, batch_size, device)
        return self


class LMNNLinear(nn.Module):
    """Linear large-margin nearest-neighbor transform of the 36-D space."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.L = nn.Linear(d_in, d_out, bias=False)

    def embed(self, x):
        return self.L(x)

    def fit(self, x, y, epochs, lr, device, n_classes: int):
        opt = torch.optim.Adam(self.parameters(), lr=lr)
        n = x.size(0)
        self.train()
        for _ in range(epochs):
            z = self.embed(x)
            # pull to class mean, push from other means (simplified LMNN)
            loss = z.new_zeros(())
            for c in range(n_classes):
                zc = z[y == c]
                if zc.size(0) == 0:
                    continue
                mu = zc.mean(0)
                loss = loss + (zc - mu).pow(2).mean()
                others = z[y != c]
                if others.size(0):
                    dneg = (others - mu).pow(2).sum(-1)
                    loss = loss + F.relu(1.0 - dneg).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
        self.eval()
        return self


def fit_svm_rbf(x_np: np.ndarray, y_np: np.ndarray):
    from sklearn.svm import SVC

    clf = SVC(kernel="rbf", gamma="scale", probability=False)
    clf.fit(x_np, y_np)
    return clf


def fit_xgboost(x_np: np.ndarray, y_np: np.ndarray):
    try:
        from xgboost import XGBClassifier

        clf = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="multi:softprob",
            eval_metric="mlogloss",
            n_jobs=4,
            verbosity=0,
        )
        clf.fit(x_np, y_np)
        return clf
    except Exception:
        from sklearn.ensemble import GradientBoostingClassifier

        clf = GradientBoostingClassifier()
        clf.fit(x_np, y_np)
        return clf


@torch.no_grad()
def nn_uniqueness(z_q, y_q, z_g, y_g, n_classes, eps) -> Dict[str, float]:
    return uniqueness_metrics(z_q, y_q, z_g, y_g, n_classes, eps=eps)


def clf_uniqueness(pred_q, y_q, z_q, n_classes, eps) -> Dict[str, float]:
    """Rank-1 as closed-set accuracy; FDR/distances in the feature space used."""
    from .metrics import fdr, inter_class_distances

    r1 = float((pred_q == y_q).float().mean().item())
    mean_d, min_d = inter_class_distances(z_q, y_q, n_classes)
    return {
        "FDR": float(fdr(z_q, y_q, n_classes, eps).item()),
        "Rank-1": r1,
        "mean_inter_class": float(mean_d.item()),
        "min_inter_class": float(min_d.item()),
    }


def smo_features(x_f: torch.Tensor) -> torch.Tensor:
    """Single-modality optical: GI optical block, dims 0–3."""
    return x_f[:, :4]
