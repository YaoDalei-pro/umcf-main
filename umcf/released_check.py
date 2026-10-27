"""Released-code check described in RESULTS.

Protocol string:
    20 repeats, seed 42+r, train-only fit, test clipped to training min/max.

This corpus is separate from Table 4. It reports FDR, Rank-1, mean/minimum
inter-class distance, Fisher weights, PCA cumulative variance, the tau/d_c
inner-validation sweep, and the three static-layer refits:
    static removed;
    equivalent size, projected area, and area-to-mass ratio only;
    slowly varying orbit block plus the eight optical and radar statistics.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .config import load_config
from .data import clip_to_train_range, generate_dataset, load_dataset, pack_split, save_dataset, stratified_split
from .metrics import class_means, fdr, uniqueness_metrics
from .model import UMCF
from .prototypes import CLASS_NAMES


def _mean_std(rows: List[dict], key: str) -> Dict[str, float]:
    vals = np.array([r[key] for r in rows], dtype=np.float64)
    std = float(vals.std(ddof=1)) if len(vals) > 1 else 0.0
    return {"mean": float(vals.mean()), "std": std}


def fit_eval(xs, xv, xf, y, xs2, xv2, xf2, y2, d_s: int, d_f: int, d_c: int = 20, tau: float = 1.0):
    model = UMCF(
        d_s=d_s,
        d_v=xv.size(1),
        d_f=d_f,
        d_c=d_c,
        tau=tau,
        eps=1e-8,
        use_ce=True,
        use_fisher=True,
        n_classes=5,
    )
    model.fit(xs, xv, xf, y, d_c=d_c, tau=tau)
    zg = model(xs, xv, xf)
    zq = model(xs2, xv2, xf2)
    met = uniqueness_metrics(zq, y2, zg, y, 5, eps=1e-8)
    met["zmax"] = float(zq.abs().max().item())
    return model, met, zq


def pca_cumvar(model: UMCF, xs, xv, xf, ks=(10, 20, 30)) -> Dict[str, float]:
    zs, zv, zf = model.standardize_base(xs, xv, xf)
    u = model.second_order_coupling(zs, zv, zf)
    ut = model._zscore(u, model.mu_c, model.sigma_c)
    s = torch.linalg.svdvals(ut)
    evr = s.pow(2)
    evr = evr / evr.sum().clamp_min(1e-12)
    cs = torch.cumsum(evr, 0)
    return {str(k): float(cs[k - 1].item()) for k in ks if k <= cs.numel()}


def run_released_check(data: dict, repeats: int = 20, seed0: int = 42, sweep: bool = True) -> dict:
    rows = {k: [] for k in ("full", "drop", "deg", "cat")}
    weights, cum, pairs = [], [], []
    sweep_rows = []

    if sweep:
        tr0, va0, _te0 = stratified_split(data["y"], 0.7, seed0, 0.1)
        fit0, val0 = pack_split(data, tr0, "cpu"), pack_split(data, va0, "cpu")
        _, xs_v = clip_to_train_range(fit0.x_s, val0.x_s)
        _, xv_v = clip_to_train_range(fit0.x_v, val0.x_v)
        _, xf_v = clip_to_train_range(fit0.x_f, val0.x_f)
        for tau in (0.5, 1.0, 2.0):
            for d_c in (10, 20, 30):
                model = UMCF(
                    d_s=10, d_v=10, d_f=16, d_c=d_c, tau=tau, eps=1e-8,
                    use_ce=True, use_fisher=True, n_classes=5,
                )
                model.fit(fit0.x_s, fit0.x_v, fit0.x_f, fit0.y, d_c=d_c, tau=tau)
                zv = model(xs_v, xv_v, xf_v)
                score = float(fdr(zv, val0.y, 5, 1e-8).item())
                sweep_rows.append({"tau": tau, "d_c": d_c, "val_FDR": score})
                print(f"[sweep] tau={tau} d_c={d_c} val_FDR={score:.4f}", flush=True)

    for r in range(repeats):
        seed = seed0 + r
        tr, _va, te = stratified_split(data["y"], 0.7, seed, 0.1)
        fit, test = pack_split(data, tr, "cpu"), pack_split(data, te, "cpu")
        xs, xs2 = clip_to_train_range(fit.x_s, test.x_s)
        xv, xv2 = clip_to_train_range(fit.x_v, test.x_v)
        xf, xf2 = clip_to_train_range(fit.x_f, test.x_f)
        y, y2 = fit.y, test.y

        model, met, zq = fit_eval(xs, xv, xf, y, xs2, xv2, xf2, y2, 10, 16)
        rows["full"].append(met)
        weights.append(model.layer_weights())
        cum.append(pca_cumvar(model, xs, xv, xf))
        pairs.append(class_means(zq, y2, 5).cpu())

        specs = {
            "drop": (xs[:, :0], xs2[:, :0], xv, xv2, xf, xf2, 0, 16),
            "deg": (xs[:, :3], xs2[:, :3], xv, xv2, xf, xf2, 3, 16),
            "cat": (xs[:, :0], xs2[:, :0], xv, xv2, xf[:, :8], xf2[:, :8], 0, 8),
        }
        for key, (a, b, c, d, e, f, ds, df) in specs.items():
            _, met2, _ = fit_eval(a, c, e, y, b, d, f, y2, ds, df)
            rows[key].append(met2)
        print(
            f"[rep {r + 1}/{repeats}] FDR={met['FDR']:.3f} "
            f"Rank-1={100 * met['Rank-1']:.2f}% "
            f"drop={rows['drop'][-1]['FDR']:.3f} "
            f"size={rows['deg'][-1]['FDR']:.3f} "
            f"catalog={rows['cat'][-1]['FDR']:.3f}",
            flush=True,
        )

    pair_stats = [torch.cdist(mu, mu, p=2) for mu in pairs]
    pair_mean = torch.stack(pair_stats).mean(0)
    best = None
    for i in range(5):
        for j in range(i + 1, 5):
            val = float(pair_mean[i, j].item())
            if best is None or val < best[0]:
                best = (val, CLASS_NAMES[i], CLASS_NAMES[j])

    full_keys = ("FDR", "Rank-1", "mean_inter_class", "min_inter_class", "zmax")
    part_keys = ("FDR", "Rank-1", "mean_inter_class", "min_inter_class")
    result = {
        "protocol": "20 repeats, seed 42+r, train-only fit, test clipped to training min/max",
        "note": "Released-code check only. Does not replace Table 4.",
        "full": {k: _mean_std(rows["full"], k) for k in full_keys},
        "static_removed": {k: _mean_std(rows["drop"], k) for k in part_keys},
        "static_size_area_amr": {k: _mean_std(rows["deg"], k) for k in part_keys},
        "catalog_orbit_optical_radar": {k: _mean_std(rows["cat"], k) for k in part_keys},
        "weights_mean": {k: float(np.mean([w[k] for w in weights])) for k in "svfc"},
        "weights_std": {k: float(np.std([w[k] for w in weights], ddof=1) if len(weights) > 1 else 0.0) for k in "svfc"},
        "pca_cumvar_mean": {k: float(np.mean([c[k] for c in cum])) for k in ("10", "20", "30") if k in cum[0]},
        "pca_cumvar_std": {
            k: float(np.std([c[k] for c in cum], ddof=1) if len(cum) > 1 else 0.0)
            for k in ("10", "20", "30") if k in cum[0]
        },
        "closest_pair": {"distance": best[0], "a": best[1], "b": best[2]},
        "pair_mean": np.round(pair_mean.numpy(), 4).tolist(),
        "class_names": list(CLASS_NAMES),
        "tau_dc_sweep_split0": sweep_rows,
        "per_repeat_full_zmax": [r["zmax"] for r in rows["full"]],
    }
    for name in ("static_removed", "static_size_area_amr", "catalog_orbit_optical_radar"):
        result[name]["FDR_ratio_vs_full"] = result[name]["FDR"]["mean"] / result["full"]["FDR"]["mean"]
        result[name]["Rank1_ratio_vs_full"] = result[name]["Rank-1"]["mean"] / result["full"]["Rank-1"]["mean"]
    return result


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="RESULTS released-code check (does not replace Table 4)")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--data", type=str, default=None, help="existing .npz; generated with seed 42 if omitted")
    p.add_argument("--out", type=str, default="results/released_check.json")
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--quick", action="store_true", help="tiny corpus, 1 repeat, skip the tau/d_c sweep")
    p.add_argument("--no-sweep", action="store_true")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    n_pc = 8 if args.quick else cfg.data.n_per_class
    repeats = 1 if args.quick else args.repeats
    if args.data:
        raw = load_dataset(args.data)
    else:
        raw = generate_dataset(cfg, seed=cfg.seed, n_per_class=n_pc, use_gi=True)
        cache = Path("data/processed/umcf_semireal_seed42.npz")
        if not args.quick:
            save_dataset(raw, str(cache))
            print("Saved", cache)
    report = run_released_check(raw, repeats=repeats, seed0=cfg.seed, sweep=not (args.quick or args.no_sweep))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("full", "weights_mean", "closest_pair")}, indent=2))
    print("Saved", out)


if __name__ == "__main__":
    main()
