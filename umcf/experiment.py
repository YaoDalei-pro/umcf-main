"""Full experimental protocol of Section III: repeats, uniqueness, ablation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from .baselines import (
    CMAF,
    DNNF,
    LMNNLinear,
    clf_uniqueness,
    fit_standardizer,
    fit_svm_rbf,
    fit_xgboost,
    nn_uniqueness,
    standardize_concat,
)
from .config import load_config
from .data import (
    generate_dataset,
    geometric_perturbed_xf,
    load_dataset,
    pack_split,
    save_dataset,
    stratified_split,
)
from .metrics import geometric_perturbation_drift, intra_class_drift
from .plots import plot_class_fingerprints, plot_robustness
from .test import evaluate_robustness, evaluate_stability, evaluate_uniqueness
from .train import save_checkpoint, train_one


def _mean_std(rows: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    keys = rows[0].keys()
    out = {}
    for k in keys:
        arr = np.array([r[k] for r in rows], dtype=np.float64)
        out[k] = {"mean": float(arr.mean()), "std": float(arr.std(ddof=1) if len(arr) > 1 else 0.0)}
    return out


def run_umcf_repeats(cfg, data, device: str, repeats: int, save_dir: Path) -> dict:
    uniq, stab = [], []
    last_model, last_meta, last_report = None, None, None
    for r in range(repeats):
        seed = cfg.seed + r
        torch.manual_seed(seed)
        model, meta = train_one(cfg, data, seed=seed, device=device)
        from .data import pack_split as _ps

        train = _ps(data, np.array(meta["train_idx"]), device=device)
        test = _ps(data, np.array(meta["test_idx"]), device=device)
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        u = evaluate_uniqueness(
            model, train, test, cfg.data.n_classes, cfg.eval.snr_ref_db, 0.0, g, cfg.model.eps
        )
        s = evaluate_stability(model, test, cfg.data.n_classes, cfg, seed, snr_db=None)
        uniq.append(u)
        stab.append(s)
        last_model, last_meta = model, meta
        print(f"[UMCF] repeat {r + 1}/{repeats}  FDR={u['FDR']:.4f}  Rank-1={100 * u['Rank-1']:.2f}%")
        if r == 0:
            last_report = evaluate_robustness(model, train, test, cfg, seed)
            z_te = model(test.x_s, test.x_v, test.x_f, snr_db=cfg.eval.snr_ref_db)
            plot_class_fingerprints(
                z_te.cpu().numpy(),
                test.y.cpu().numpy(),
                list(meta["class_names"]),
                str(save_dir / "figures" / "fig_fingerprints.png"),
            )
            plot_robustness(
                last_report["noise"],
                last_report["missing"],
                str(save_dir / "figures" / "fig_robustness.png"),
            )
            save_checkpoint(model, meta, str(save_dir / "umcf_repeat0.pt"))

    return {
        "uniqueness": _mean_std(uniq),
        "stability": _mean_std(stab),
        "robustness_repeat0": last_report,
        "weights_repeat0": last_model.layer_weights() if last_model is not None else {},
    }


def _noisy_concat(x_s, x_v, x_f, mu, sigma, eps, snr_db, rho, g):
    z = standardize_concat(x_s, x_v, x_f, mu, sigma, eps)
    if rho and rho > 0.0:
        mask = torch.rand(z.shape, device=z.device, generator=g) < rho
        z = z.masked_fill(mask, 0.0)
    if snr_db is not None:
        from .model import _add_snr_noise

        z = _add_snr_noise(z, snr_db, g)
    return z


def run_baselines_once(cfg, data, seed: int, device: str) -> dict:
    tr, va, te = stratified_split(data["y"], cfg.data.train_ratio, seed, cfg.data.val_ratio_of_train)
    train = pack_split(data, tr, device=device)
    test = pack_split(data, te, device=device)
    mu, sigma = fit_standardizer(train.x_s, train.x_v, train.x_f)
    g = torch.Generator(device=device)
    g.manual_seed(seed)
    snr = cfg.eval.snr_ref_db
    xtr = _noisy_concat(train.x_s, train.x_v, train.x_f, mu, sigma, cfg.model.eps, snr, 0.0, g)
    xte = _noisy_concat(test.x_s, test.x_v, test.x_f, mu, sigma, cfg.model.eps, snr, 0.0, g)
    ytr, yte = train.y, test.y
    n_c, eps = cfg.data.n_classes, cfg.model.eps
    out = {}

    out["FC"] = nn_uniqueness(xte, yte, xtr, ytr, n_c, eps)
    # SMO: optical GI block (dims 0–3 of the 16-D x_f, which occupy 20:24 of the 36-D concat)
    out["SMO"] = nn_uniqueness(xte[:, 20:24], yte, xtr[:, 20:24], ytr, n_c, eps)

    xtr_np, ytr_np = xtr.cpu().numpy(), ytr.cpu().numpy()
    xte_np, yte_np = xte.cpu().numpy(), yte.cpu().numpy()
    svm = fit_svm_rbf(xtr_np, ytr_np)
    pred = torch.from_numpy(svm.predict(xte_np)).to(yte)
    out["SVM-RBF"] = clf_uniqueness(pred, yte, xte, n_c, eps)

    xgb = fit_xgboost(xtr_np, ytr_np)
    pred = torch.from_numpy(np.asarray(xgb.predict(xte_np))).to(yte)
    out["XGBoost"] = clf_uniqueness(pred, yte, xte, n_c, eps)

    lmnn = LMNNLinear(xtr.size(1), cfg.baselines.lmnn_embed).to(device)
    lmnn.fit(xtr, ytr, cfg.baselines.lmnn_epochs, cfg.baselines.lmnn_lr, device, n_c)
    with torch.no_grad():
        out["LMNN"] = nn_uniqueness(lmnn.embed(xte), yte, lmnn.embed(xtr), ytr, n_c, eps)

    dnnf = DNNF(xtr.size(1), cfg.baselines.dnnf_hidden, cfg.baselines.dnnf_embed, n_c).to(device)
    dnnf.fit(xtr, ytr, cfg.baselines.dnnf_epochs, cfg.baselines.dnnf_lr, cfg.baselines.dnnf_batch_size, device)
    with torch.no_grad():
        out["DNNF"] = nn_uniqueness(dnnf.embed(xte), yte, dnnf.embed(xtr), ytr, n_c, eps)

    cmaf = CMAF(n_classes=n_c).to(device)
    cmaf.fit(xtr, ytr, cfg.baselines.cmaf_epochs, cfg.baselines.cmaf_lr, cfg.baselines.dnnf_batch_size, device)
    with torch.no_grad():
        out["CMAF"] = nn_uniqueness(cmaf.embed(xte), yte, cmaf.embed(xtr), ytr, n_c, eps)

    # Table 5: FC / LMNN / CMAF share the 36-D standardized space, so their
    # intra-class and geometric drifts are computed there. DNNF drift is taken
    # after the nonlinear map. No SNR noise is added on top of the 5 deg jitter.
    rng = np.random.default_rng(seed + 23)
    x_f_pert = geometric_perturbed_xf(
        test, None, rng, cfg.data.geom_jitter_deg, cfg.data.sequence_length,
        cfg.model.use_gi, cfg.model.eps, device,
    )
    g2 = torch.Generator(device=device)
    g2.manual_seed(seed + 23)
    clean = _noisy_concat(test.x_s, test.x_v, test.x_f, mu, sigma, eps, None, 0.0, g2)
    pert = _noisy_concat(test.x_s, test.x_v, x_f_pert, mu, sigma, eps, None, 0.0, g2)

    def _stab(z0, z1):
        mean_d, max_d = intra_class_drift(z1, yte, n_c)
        geo = geometric_perturbation_drift(z0, z1)
        return {
            "mean_intra_class_drift": float(mean_d.item()),
            "max_intra_class_drift": float(max_d.item()),
            "geometric_perturbation_drift": float(geo.item()),
        }

    with torch.no_grad():
        out["stability_feature_space"] = _stab(clean, pert)
        out["stability_SMO"] = _stab(clean[:, 20:24], pert[:, 20:24])
        out["stability_DNNF"] = _stab(dnnf.embed(clean), dnnf.embed(pert))
        out["stability_LMNN"] = _stab(lmnn.embed(clean), lmnn.embed(pert))
        out["stability_CMAF"] = _stab(cmaf.embed(clean), cmaf.embed(pert))
    return out


def run_ablation(cfg, data, device: str, seed: int) -> dict:
    """Table 6: GI / CE / Fisher combinations, FDR under baseline / noise / missing."""
    combos = [
        ("None", False, False, False),
        ("GI", True, False, False),
        ("GI+CE", True, True, False),
        ("UMCF", True, True, True),
    ]
    rows = {}
    for name, gi, ce, fisher in combos:
        cfg.model.use_ce = ce
        cfg.model.use_fisher = fisher
        # GI off: recompute x_f with non-robust stats
        data_use = dict(data)
        if not gi:
            from .data import recompute_xf_from_sequences

            xf = recompute_xf_from_sequences(
                torch.from_numpy(data["sequences"]),
                torch.from_numpy(data["slant_range"]),
                use_gi=False,
                eps=cfg.model.eps,
            )
            data_use["x_f"] = xf.cpu().numpy().astype(np.float32)
        model, meta = train_one(cfg, data_use, seed=seed, device=device, use_ce=ce, use_fisher=fisher)
        train = pack_split(data_use, np.array(meta["train_idx"]), device=device)
        test = pack_split(data_use, np.array(meta["test_idx"]), device=device)
        g = torch.Generator(device=device)
        g.manual_seed(seed)
        base = evaluate_uniqueness(
            model, train, test, cfg.data.n_classes, cfg.eval.snr_ref_db, 0.0, g, cfg.model.eps
        )
        g.manual_seed(seed + 1)
        noise = evaluate_uniqueness(
            model, train, test, cfg.data.n_classes, cfg.eval.ablation_noise_db, 0.0, g, cfg.model.eps
        )
        g.manual_seed(seed + 2)
        miss = evaluate_uniqueness(
            model,
            train,
            test,
            cfg.data.n_classes,
            cfg.eval.snr_ref_db,
            cfg.eval.ablation_missing_ratio,
            g,
            cfg.model.eps,
        )
        rows[name] = {"Baseline": base["FDR"], "Noise": noise["FDR"], "Missing": miss["FDR"]}
        print(f"[ablation] {name}: {rows[name]}")
    # restore
    cfg.model.use_ce = True
    cfg.model.use_fisher = True
    return rows


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Section III experimental protocol")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--data", type=str, default=None)
    p.add_argument("--generate", action="store_true")
    p.add_argument("--quick", action="store_true", help="small N and 1 repeat for a smoke test")
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--baselines", action="store_true")
    p.add_argument("--ablation", action="store_true")
    p.add_argument("--out", type=str, default="results")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    if args.quick:
        cfg.data.n_per_class = 40
        cfg.train.repeats = 1
        cfg.baselines.dnnf_epochs = 5
        cfg.baselines.cmaf_epochs = 5
        cfg.baselines.lmnn_epochs = 5
        cfg.data.snr_db_list = [10.0, 20.0]
        cfg.data.missing_ratios = [0.0, 0.3, 0.6]
    if args.repeats is not None:
        cfg.train.repeats = args.repeats
    if args.seed is not None:
        cfg.seed = args.seed
    device = args.device or cfg.resolved_device()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.generate or args.data is None:
        n_pc = cfg.data.n_per_class
        raw = generate_dataset(cfg, seed=cfg.seed, n_per_class=n_pc, use_gi=True)
        data_path = out_dir / "umcf_semireal.npz"
        save_dataset(raw, str(data_path))
        print("Wrote", data_path)
    else:
        raw = load_dataset(args.data)

    report = run_umcf_repeats(cfg, raw, device, cfg.train.repeats, out_dir)
    if args.baselines:
        report["baselines_repeat0"] = run_baselines_once(cfg, raw, cfg.seed, device)
        print("baselines:", json.dumps(report["baselines_repeat0"], indent=2))
    if args.ablation:
        report["ablation"] = run_ablation(cfg, raw, device, cfg.seed)

    path = out_dir / "experiment.json"
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Saved", path)
    print(json.dumps(report.get("uniqueness", {}), indent=2))


if __name__ == "__main__":
    main()
