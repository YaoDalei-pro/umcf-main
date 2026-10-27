"""Unit tests that lock Section II equations and Section III metric definitions."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from umcf.distance import decomposable_squared_distance
from umcf.gi import ObservationInvariantExtractor
from umcf.metrics import fdr, missing_retention, noise_retention, rank1_rate
from umcf.model import UMCF


def test_coupling_dim_420():
    n, ds, dv, df = 8, 10, 10, 16
    z_s = torch.randn(n, ds)
    z_v = torch.randn(n, dv)
    z_f = torch.randn(n, df)
    u = UMCF.second_order_coupling(z_s, z_v, z_f)
    assert u.shape == (n, ds * dv + ds * df + dv * df)
    assert u.shape[-1] == 420


def test_fingerprint_dim_56_and_weight_sum():
    n, c = 40, 5
    x_s, x_v, x_f = torch.randn(n, 10), torch.randn(n, 10), torch.randn(n, 16)
    y = torch.arange(n) % c
    model = UMCF(n_classes=c)
    model.fit(x_s, x_v, x_f, y)
    z = model(x_s, x_v, x_f)
    assert z.shape == (n, 56)
    w = model.layer_weights()
    assert abs(sum(w.values()) - 1.0) < 1e-5
    assert all(v >= 0 for v in w.values())


def test_eq8_matches_euclidean():
    n, c = 30, 5
    x_s, x_v, x_f = torch.randn(n, 10), torch.randn(n, 10), torch.randn(n, 16)
    y = torch.arange(n) % c
    model = UMCF(n_classes=c)
    model.fit(x_s, x_v, x_f, y)
    z = model(x_s, x_v, x_f)
    d_eu = torch.linalg.norm(z[0] - z[1], dim=0) ** 2
    d_dec, parts = decomposable_squared_distance(z[0], z[1])
    assert torch.allclose(d_eu, d_dec.squeeze(), atol=1e-5)
    assert torch.allclose(d_dec.squeeze(), sum(parts.values()).squeeze(), atol=1e-5)


def test_theta_frozen_between_train_and_forward():
    n, c = 50, 5
    x_s, x_v, x_f = torch.randn(n, 10), torch.randn(n, 10), torch.randn(n, 16)
    y = torch.arange(n) % c
    model = UMCF(n_classes=c, tau=1.0)
    model.fit(x_s[:40], x_v[:40], x_f[:40], y[:40])
    P0 = model.P.clone()
    w0 = model.w_s.clone()
    _ = model(x_s[40:], x_v[40:], x_f[40:])
    assert torch.equal(P0, model.P)
    assert torch.equal(w0, model.w_s)


def test_gi_output_16():
    gi = ObservationInvariantExtractor()
    seq = torch.rand(6, 5, 64).abs() + 0.01
    r = torch.full((6,), 800.0)
    xf = gi(seq, r)
    assert xf.shape == (6, 16)
    assert torch.isfinite(xf).all()


def test_r_miss_is_one_at_zero():
    assert abs(missing_retention(0.358, 0.358) - 1.0) < 1e-8


def test_rank1_and_fdr_shapes():
    z_g = torch.randn(20, 56)
    y_g = torch.arange(20) % 5
    z_q = z_g[:5] + 0.01 * torch.randn(5, 56)
    y_q = y_g[:5]
    r1 = rank1_rate(z_q, y_q, z_g, y_g)
    score = fdr(z_g, y_g, 5)
    assert 0.0 <= float(r1) <= 1.0
    assert float(score) >= 0.0


def test_ablation_equal_weights_without_fisher():
    n, c = 30, 5
    x_s, x_v, x_f = torch.randn(n, 10), torch.randn(n, 10), torch.randn(n, 16)
    y = torch.arange(n) % c
    model = UMCF(n_classes=c, use_fisher=False)
    model.fit(x_s, x_v, x_f, y)
    w = model.layer_weights()
    for v in w.values():
        assert abs(v - 0.25) < 1e-6


def test_static_removed_refit_is_finite():
    n, c = 40, 5
    x_s = torch.zeros(n, 0)
    x_v, x_f = torch.randn(n, 10), torch.randn(n, 16)
    y = torch.arange(n) % c
    model = UMCF(d_s=0, d_v=10, d_f=16, d_c=8, n_classes=c)
    model.fit(x_s, x_v, x_f, y, d_c=8)
    z = model(x_s, x_v, x_f)
    assert z.shape == (n, 10 + 16 + 8)
    assert torch.isfinite(z).all()
    assert abs(sum(model.layer_weights().values()) - 1.0) < 1e-5


def test_clip_keeps_test_inside_train_range():
    from umcf.data import clip_to_train_range

    train = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
    test = torch.tensor([[-5.0, 9.0], [1.0, 2.0]])
    _, clipped = clip_to_train_range(train, test)
    assert float(clipped[:, 0].min()) >= 0.0
    assert float(clipped[:, 1].max()) <= 3.0


def test_eq6_is_dimension_normalized():
    from umcf.fisher import fisher_score, scatter_traces

    z = torch.tensor([[0.0, 0.0], [0.0, 0.0], [2.0, 0.0], [2.0, 0.0]])
    y = torch.tensor([0, 0, 1, 1])
    tr_b, tr_w = scatter_traces(z, y, 2)
    score = fisher_score(z, y, 2, eps=1e-8)
    assert torch.allclose(score, (tr_b / (tr_w + 1e-8)) / 2)


def test_missing_on_standardized_stays_finite():
    n, c = 40, 5
    x_s, x_v, x_f = torch.randn(n, 10), torch.randn(n, 10), torch.randn(n, 16)
    y = torch.arange(n) % c
    model = UMCF(n_classes=c)
    model.fit(x_s[:30], x_v[:30], x_f[:30], y[:30])
    z = model(x_s[30:], x_v[30:], x_f[30:], missing_ratio=0.6)
    assert torch.isfinite(z).all()
    assert z.abs().max() < 1e3


if __name__ == "__main__":
    tests = [
        test_coupling_dim_420,
        test_fingerprint_dim_56_and_weight_sum,
        test_eq8_matches_euclidean,
        test_theta_frozen_between_train_and_forward,
        test_gi_output_16,
        test_r_miss_is_one_at_zero,
        test_rank1_and_fdr_shapes,
        test_ablation_equal_weights_without_fisher,
        test_static_removed_refit_is_finite,
        test_clip_keeps_test_inside_train_range,
        test_eq6_is_dimension_normalized,
        test_missing_on_standardized_stays_finite,
    ]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print("all passed")
