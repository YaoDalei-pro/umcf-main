"""UMCF model: Section II, Algorithms 1–2 and Eqs. (1)–(8).

UMCF is a freezeable deterministic mapping
    z = Phi(x; Theta) in R^{56},
    x = (x_s, x_v, x_f),
    Theta = {mu_l, sigma_l, mu_c, sigma_c, P, w_l}
estimated on the training set only (Algorithm 2) and frozen at test time.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .distance import decomposable_squared_distance, pairwise_squared_euclidean
from .fisher import fisher_score, fisher_weights


def _add_snr_noise(
    z: torch.Tensor, snr_db: float, generator: Optional[torch.Generator] = None
) -> torch.Tensor:
    p_sig = z.pow(2).mean().clamp_min(1e-12)
    p_noise = p_sig / (10.0 ** (snr_db / 10.0))
    std = torch.sqrt(p_noise)
    return z + torch.randn(z.shape, device=z.device, dtype=z.dtype, generator=generator) * std


class UMCF(nn.Module):
    """Unified Multimodal Coupled Fingerprint.

    Parameters
    ----------
    d_s, d_v, d_f, d_c : layer dimensions (paper: 10, 10, 16, 20)
    tau : Softmax temperature, default 1
    eps : numerical stabilizer, paper uses 1e-8
    use_ce : enable second-order coupling (CE module)
    use_fisher : enable Fisher layer weighting; otherwise equal weights
    """

    LAYER_NAMES = ("s", "v", "f", "c")

    def __init__(
        self,
        d_s: int = 10,
        d_v: int = 10,
        d_f: int = 16,
        d_c: int = 20,
        tau: float = 1.0,
        eps: float = 1e-8,
        use_ce: bool = True,
        use_fisher: bool = True,
        n_classes: int = 5,
    ):
        super().__init__()
        self.d_s = d_s
        self.d_v = d_v
        self.d_f = d_f
        self.d_c = d_c
        self.d_u = d_s * d_v + d_s * d_f + d_v * d_f  # 100+160+160=420
        self.d_z = d_s + d_v + d_f + d_c  # 56
        self.tau = tau
        self.eps = eps
        self.use_ce = use_ce
        self.use_fisher = use_fisher
        self.n_classes = n_classes

        self.register_buffer("mu_s", torch.zeros(d_s))
        self.register_buffer("sigma_s", torch.ones(d_s))
        self.register_buffer("mu_v", torch.zeros(d_v))
        self.register_buffer("sigma_v", torch.ones(d_v))
        self.register_buffer("mu_f", torch.zeros(d_f))
        self.register_buffer("sigma_f", torch.ones(d_f))
        self.register_buffer("mu_c", torch.zeros(self.d_u))
        self.register_buffer("sigma_c", torch.ones(self.d_u))
        self.register_buffer("P", torch.zeros(self.d_u, d_c))  # 420 x 20
        self.register_buffer("w_s", torch.tensor(0.25))
        self.register_buffer("w_v", torch.tensor(0.25))
        self.register_buffer("w_f", torch.tensor(0.25))
        self.register_buffer("w_c", torch.tensor(0.25))
        self.register_buffer("J_s", torch.tensor(0.0))
        self.register_buffer("J_v", torch.tensor(0.0))
        self.register_buffer("J_f", torch.tensor(0.0))
        self.register_buffer("J_c", torch.tensor(0.0))
        self.register_buffer("_fitted", torch.tensor(False))

    # ------------------------------------------------------------------
    # Eq. (3) and Algorithm 1
    # ------------------------------------------------------------------
    def _zscore(self, x: torch.Tensor, mu: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        return (x - mu) / (sigma + self.eps)

    def standardize_base(
        self, x_s: torch.Tensor, x_v: torch.Tensor, x_f: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        z_s = self._zscore(x_s, self.mu_s, self.sigma_s)
        z_v = self._zscore(x_v, self.mu_v, self.sigma_v)
        z_f = self._zscore(x_f, self.mu_f, self.sigma_f)
        return z_s, z_v, z_f

    @staticmethod
    def _outer_vec(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Row-major vec(a b^T). A zero-width factor yields an empty block."""
        n = a.size(0)
        width = a.size(-1) * b.size(-1)
        if a.size(-1) == 0 or b.size(-1) == 0:
            return a.new_zeros(n, width)
        return torch.einsum("ni,nj->nij", a, b).reshape(n, width)

    @staticmethod
    def second_order_coupling(
        z_s: torch.Tensor, z_v: torch.Tensor, z_f: torch.Tensor
    ) -> torch.Tensor:
        """u_c = [vec(z_s z_v^T); vec(z_s z_f^T); vec(z_v z_f^T)], Eq. (4)–(5).

        Batched outer products; vec is row-major flatten (consistent train/test).
        Full blocks are 10x10 + 10x16 + 10x16 = 420. A removed static layer
        (d_s = 0) keeps only vec(z_v z_f^T).
        """
        csv = UMCF._outer_vec(z_s, z_v)
        csf = UMCF._outer_vec(z_s, z_f)
        cvf = UMCF._outer_vec(z_v, z_f)
        return torch.cat([csv, csf, cvf], dim=-1)

    def construct_layers(
        self, x_s: torch.Tensor, x_v: torch.Tensor, x_f: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Algorithm 1: layered fingerprint construction."""
        z_s, z_v, z_f = self.standardize_base(x_s, x_v, x_f)
        if self.use_ce:
            u_c = self.second_order_coupling(z_s, z_v, z_f)
            u_tilde = self._zscore(u_c, self.mu_c, self.sigma_c)
            z_c = u_tilde @ self.P  # (N, 20) = P^T u  with P in R^{420 x 20}
        else:
            z_c = z_s.new_zeros(z_s.size(0), self.d_c)
        return {"s": z_s, "v": z_v, "f": z_f, "c": z_c}

    def assemble(self, layers: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Eq. (2): z = [sqrt(w_s) z_s; sqrt(w_v) z_v; sqrt(w_f) z_f; sqrt(w_c) z_c]."""
        ws = torch.sqrt(self.w_s.clamp_min(0.0))
        wv = torch.sqrt(self.w_v.clamp_min(0.0))
        wf = torch.sqrt(self.w_f.clamp_min(0.0))
        wc = torch.sqrt(self.w_c.clamp_min(0.0))
        return torch.cat(
            [ws * layers["s"], wv * layers["v"], wf * layers["f"], wc * layers["c"]],
            dim=-1,
        )

    def assemble_omitting(
        self, layers: Dict[str, torch.Tensor], omit: Tuple[str, ...] = ("s",)
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Omit unavailable layers and renormalize the remaining Fisher weights.

        Used when a query has no static coordinates. The same omission is applied
        to the gallery; the static block is not zero-filled.
        """
        from .distance import renormalized_layer_weights

        raw = {"s": self.w_s, "v": self.w_v, "f": self.w_f, "c": self.w_c}
        active = renormalized_layer_weights(raw, omit=omit)
        chunks = [
            torch.sqrt(active[k].clamp_min(0.0)) * layers[k]
            for k in self.LAYER_NAMES
            if k not in omit
        ]
        weights = {k: float(active[k].detach().cpu()) for k in active}
        return torch.cat(chunks, dim=-1), weights

    def _mask_standardized(
        self,
        z_s: torch.Tensor,
        z_v: torch.Tensor,
        z_f: torch.Tensor,
        rho: float,
        generator: Optional[torch.Generator],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Feature-wise zero-fill in standardized space, then coupling is recomputed."""
        z = torch.cat([z_s, z_v, z_f], dim=-1)
        mask = torch.rand(z.shape, device=z.device, generator=generator) < rho
        z = z.masked_fill(mask, 0.0)
        d_s, d_v = z_s.size(1), z_v.size(1)
        return z[:, :d_s], z[:, d_s : d_s + d_v], z[:, d_s + d_v :]

    def forward(
        self,
        x_s: torch.Tensor,
        x_v: torch.Tensor,
        x_f: torch.Tensor,
        return_layers: bool = False,
        snr_db: Optional[float] = None,
        missing_ratio: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ):
        """Frozen forward mapping.

        Degradation (Section III-A) is applied on *standardized* base layers:
        additive Gaussian noise at a prescribed SNR, and/or random feature-wise
        zero-fill. Coupling is then recomputed; Fisher weights stay frozen.
        """
        z_s, z_v, z_f = self.standardize_base(x_s, x_v, x_f)
        if missing_ratio and missing_ratio > 0.0:
            z_s, z_v, z_f = self._mask_standardized(z_s, z_v, z_f, float(missing_ratio), generator)
        if snr_db is not None:
            z_s = _add_snr_noise(z_s, snr_db, generator)
            z_v = _add_snr_noise(z_v, snr_db, generator)
            z_f = _add_snr_noise(z_f, snr_db, generator)
        if self.use_ce:
            u_c = self.second_order_coupling(z_s, z_v, z_f)
            u_tilde = self._zscore(u_c, self.mu_c, self.sigma_c)
            z_c = u_tilde @ self.P
        else:
            z_c = z_s.new_zeros(z_s.size(0), self.d_c)
        layers = {"s": z_s, "v": z_v, "f": z_f, "c": z_c}
        z = self.assemble(layers)
        if return_layers:
            return z, layers
        return z

    # ------------------------------------------------------------------
    # Algorithm 2 — training (estimate Theta on D_train, then freeze)
    # ------------------------------------------------------------------
    @staticmethod
    def _mean_std(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.size(-1) == 0:
            return x.new_zeros(0), x.new_ones(0)
        return x.mean(dim=0), x.std(dim=0, unbiased=False).clamp_min(1e-12)

    def _fit_pca(self, u_tilde: torch.Tensor, d_c: Optional[int] = None) -> torch.Tensor:
        """PCA projection P in R^{420 x d_c} from standardized coupling vectors."""
        d_c = int(d_c or self.d_c)
        # SVD of (N, 420); right singular vectors are principal axes
        # torch.linalg.svd(X, full_matrices=False) -> Vh shape (min(N,420), 420)
        n, d = u_tilde.shape
        if n < 2:
            return torch.eye(d, d_c, device=u_tilde.device, dtype=u_tilde.dtype)
        _u, _s, vh = torch.linalg.svd(u_tilde, full_matrices=False)
        components = vh[:d_c]  # (d_c, 420)
        if components.size(0) < d_c:
            pad = u_tilde.new_zeros(d_c - components.size(0), d)
            components = torch.cat([components, pad], dim=0)
        return components.T.contiguous()  # (420, d_c)

    @torch.no_grad()
    def fit(
        self,
        x_s: torch.Tensor,
        x_v: torch.Tensor,
        x_f: torch.Tensor,
        y: torch.Tensor,
        d_c: Optional[int] = None,
        tau: Optional[float] = None,
    ) -> "UMCF":
        """Training procedure (Algorithm 2). All statistics from D_train only."""
        if tau is not None:
            self.tau = float(tau)
        if d_c is not None and int(d_c) != self.d_c:
            self.d_c = int(d_c)
            self.d_z = self.d_s + self.d_v + self.d_f + self.d_c
            self.P = torch.zeros(self.d_u, self.d_c, device=x_s.device, dtype=x_s.dtype)

        y = y.long().view(-1)
        n_classes = int(max(self.n_classes, int(y.max().item()) + 1))
        self.n_classes = n_classes

        # 1. Estimate mu_l, sigma_l for l in {s, v, f}
        self.mu_s, self.sigma_s = self._mean_std(x_s)
        self.mu_v, self.sigma_v = self._mean_std(x_v)
        self.mu_f, self.sigma_f = self._mean_std(x_f)

        z_s, z_v, z_f = self.standardize_base(x_s, x_v, x_f)

        # 2–4. Coupling vector, standardize, PCA -> P
        if self.use_ce:
            u_c = self.second_order_coupling(z_s, z_v, z_f)
            self.mu_c, self.sigma_c = self._mean_std(u_c)
            u_tilde = self._zscore(u_c, self.mu_c, self.sigma_c)
            self.P = self._fit_pca(u_tilde, d_c=self.d_c)
            z_c = u_tilde @ self.P
        else:
            self.mu_c.zero_()
            self.sigma_c.fill_(1.0)
            self.P.zero_()
            z_c = z_s.new_zeros(z_s.size(0), self.d_c)

        # 5. Fisher scores J_l and Softmax weights w_l
        layers = {"s": z_s, "v": z_v, "f": z_f, "c": z_c}
        scores = {
            k: fisher_score(layers[k], y, n_classes, self.eps) for k in self.LAYER_NAMES
        }
        self.J_s, self.J_v, self.J_f, self.J_c = (
            scores["s"],
            scores["v"],
            scores["f"],
            scores["c"],
        )
        if self.use_fisher:
            w = fisher_weights(scores, self.tau)
        else:
            w = {k: z_s.new_tensor(0.25) for k in self.LAYER_NAMES}
        self.w_s, self.w_v, self.w_f, self.w_c = w["s"], w["v"], w["f"], w["c"]
        self._fitted.fill_(True)
        return self

    # ------------------------------------------------------------------
    # Matching
    # ------------------------------------------------------------------
    def layer_weights(self) -> Dict[str, float]:
        return {
            "s": float(self.w_s.item()),
            "v": float(self.w_v.item()),
            "f": float(self.w_f.item()),
            "c": float(self.w_c.item()),
        }

    def fisher_scores(self) -> Dict[str, float]:
        return {
            "s": float(self.J_s.item()),
            "v": float(self.J_v.item()),
            "f": float(self.J_f.item()),
            "c": float(self.J_c.item()),
        }

    def theta(self) -> Dict[str, torch.Tensor]:
        return {
            "mu_s": self.mu_s,
            "sigma_s": self.sigma_s,
            "mu_v": self.mu_v,
            "sigma_v": self.sigma_v,
            "mu_f": self.mu_f,
            "sigma_f": self.sigma_f,
            "mu_c": self.mu_c,
            "sigma_c": self.sigma_c,
            "P": self.P,
            "w_s": self.w_s,
            "w_v": self.w_v,
            "w_f": self.w_f,
            "w_c": self.w_c,
        }

    def attribute_match(
        self, z_i: torch.Tensor, z_j: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Eq. (8) attribution on already-assembled fingerprints."""
        return decomposable_squared_distance(
            z_i, z_j, d_s=self.d_s, d_v=self.d_v, d_f=self.d_f, d_c=self.d_c
        )

    def rank1_match(
        self,
        z_query: torch.Tensor,
        z_gallery: torch.Tensor,
        y_gallery: torch.Tensor,
    ) -> torch.Tensor:
        """Nearest-neighbor class labels in fingerprint space."""
        d2 = pairwise_squared_euclidean(z_query, z_gallery)
        nn = d2.argmin(dim=1)
        return y_gallery[nn]
