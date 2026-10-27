"""Test / validation following Section III: uniqueness, stability, robustness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from .config import load_config
from .data import (
    geometric_perturbed_xf,
    load_dataset,
    pack_split,
)
from .metrics import (
    closest_class_pair,
    missing_retention,
    noise_retention,
    stability_metrics,
    uniqueness_metrics,
)
from .prototypes import CLASS_NAMES
from .model import UMCF
from .train import load_checkpoint


@torch.no_grad()
def encode_split(model: UMCF, split, snr_db: Optional[float], rho: float, g: torch.Generator):
    return model(
        split.x_s,
        split.x_v,
        split.x_f,
        snr_db=snr_db,
        missing_ratio=rho,
        generator=g,
    )


@torch.no_grad()
def evaluate_uniqueness(model: UMCF, train, test, n_classes, snr_db, rho, g, eps) -> Dict[str, float]:
    z_g = encode_split(model, train, snr_db, rho, g)
    z_q = encode_split(model, test, snr_db, rho, g)
    return uniqueness_metrics(z_q, test.y, z_g, train.y, n_classes, eps=eps)


@torch.no_grad()
def evaluate_stability(
    model: UMCF,
    test,
    n_classes,
    cfg,
    seed: int,
    snr_db: Optional[float] = None,
) -> Dict[str, float]:
    """Table 5. Jitter is applied before GI; SNR noise is not mixed into the drift.

    Mean/max intra-class drift are Euclidean distances to the class mean of the
    perturbed fingerprints. Geometric-perturbation drift is mean ||z - z'||_2.
    """
    z = model(test.x_s, test.x_v, test.x_f, snr_db=snr_db)
    rng = np.random.default_rng(seed + 23)
    x_f_pert = geometric_perturbed_xf(
        test,
        None,
        rng,
        jitter_deg=cfg.data.geom_jitter_deg,
        t_len=cfg.data.sequence_length,
        use_gi=cfg.model.use_gi,
        eps=cfg.model.eps,
        device=str(test.x_s.device),
    )
    z_pert = model(test.x_s, test.x_v, x_f_pert, snr_db=snr_db)
    # Intra-class statistics are taken on the perturbed fingerprints (Table 5).
    stats = stability_metrics(z_pert, test.y, z_pert, n_classes)
    from .metrics import geometric_perturbation_drift

    stats["geometric_perturbation_drift"] = float(
        geometric_perturbation_drift(z, z_pert).item()
    )
    return stats


@torch.no_grad()
def evaluate_robustness(model: UMCF, train, test, cfg, seed: int) -> Dict[str, dict]:
    n_classes = cfg.data.n_classes
    eps = cfg.model.eps
    device = train.y.device
    ref_db = cfg.eval.snr_ref_db

    g = torch.Generator(device=device)
    g.manual_seed(seed + 31)
    base = evaluate_uniqueness(model, train, test, n_classes, ref_db, 0.0, g, eps)
    r1_ref = base["Rank-1"]
    fdr0 = base["FDR"]

    noise_curve = {}
    for snr in cfg.data.snr_db_list:
        g.manual_seed(seed + 100 + int(snr))
        m = evaluate_uniqueness(model, train, test, n_classes, float(snr), 0.0, g, eps)
        noise_curve[str(snr)] = {
            **m,
            "R_noise": noise_retention(m["Rank-1"], r1_ref),
        }

    miss_curve = {}
    for rho in cfg.data.missing_ratios:
        if float(rho) == 0.0:
            miss_curve[str(rho)] = {**base, "R_miss": 1.0}
            continue
        g.manual_seed(seed + 200 + int(rho * 100))
        m = evaluate_uniqueness(
            model, train, test, n_classes, cfg.eval.missing_noise_db, float(rho), g, eps
        )
        miss_curve[str(rho)] = {
            **m,
            "R_miss": missing_retention(m["FDR"], fdr0),
        }
    return {"baseline": base, "noise": noise_curve, "missing": miss_curve}


def run_test(cfg, data, model: UMCF, meta: dict, seed: int, device: str) -> dict:
    idx_tr = np.array(meta["train_idx"])
    idx_te = np.array(meta["test_idx"])
    train = pack_split(data, idx_tr, device=device)
    test = pack_split(data, idx_te, device=device)
    n_classes = cfg.data.n_classes

    g = torch.Generator(device=device)
    g.manual_seed(seed)
    uniqueness = evaluate_uniqueness(
        model, train, test, n_classes, cfg.eval.snr_ref_db, 0.0, g, cfg.model.eps
    )
    z_q = encode_split(model, test, cfg.eval.snr_ref_db, 0.0, g)
    pair = closest_class_pair(z_q, test.y, n_classes, CLASS_NAMES)
    stability = evaluate_stability(model, test, n_classes, cfg, seed, snr_db=None)
    robustness = evaluate_robustness(model, train, test, cfg, seed)
    return {
        "uniqueness": uniqueness,
        "closest_class_pair": {k: pair[k] for k in ("distance", "a", "b")},
        "class_mean_distance": pair["matrix"],
        "stability": stability,
        "robustness": robustness,
        "weights": model.layer_weights(),
        "fisher_scores": model.fisher_scores(),
        "protocol": {
            "baseline": "SNR = 20 dB, no missing (Table 4)",
            "stability": "5 deg attitude and solar-phase jitter before GI (Table 5)",
            "ablation_conditions": {
                "Baseline": "20 dB, no missing",
                "Noise": "10 dB, no missing",
                "Missing": "20 dB, 30% missing",
            },
            "R_noise": "R1(SNR) / R1(20 dB)",
            "R_miss": "FDR(rho) / FDR(0); equals 1 at rho = 0",
        },
    }


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Evaluate UMCF (Section III metrics)")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--out", type=str, default="results/umcf_metrics.json")
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    device = args.device or cfg.resolved_device()
    data = load_dataset(args.data)
    model, meta = load_checkpoint(args.ckpt, device=device)
    seed = int(meta.get("seed", cfg.seed))
    report = run_test(cfg, data, model, meta, seed, device)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report["uniqueness"], indent=2))
    print("stability:", json.dumps(report["stability"], indent=2))
    print("Saved", args.out)


if __name__ == "__main__":
    main()
