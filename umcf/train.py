"""Training entry: Algorithm 2 (estimate Theta on D_train and freeze)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Tuple

import torch

from .config import UMCFConfig, load_config
from .data import load_dataset, pack_split, stratified_split
from .metrics import fdr
from .model import UMCF
from .prototypes import CLASS_NAMES


def build_model(cfg: UMCFConfig, device: str) -> UMCF:
    m = cfg.model
    model = UMCF(
        d_s=m.d_s,
        d_v=m.d_v,
        d_f=m.d_f,
        d_c=m.d_c,
        tau=m.tau,
        eps=m.eps,
        use_ce=m.use_ce,
        use_fisher=m.use_fisher,
        n_classes=cfg.data.n_classes,
    )
    return model.to(device)


def confirm_hyperparams(
    cfg: UMCFConfig,
    x_s, x_v, x_f, y,
    x_s_val, x_v_val, x_f_val, y_val,
    device: str,
) -> Tuple[float, int]:
    """Inner 10% of training is used only to confirm tau and d_c (Section III-A)."""
    best = (-1.0, cfg.model.tau, cfg.model.d_c)
    for tau in cfg.train.tau_grid:
        for d_c in cfg.train.dc_grid:
            model = UMCF(
                d_s=cfg.model.d_s,
                d_v=cfg.model.d_v,
                d_f=cfg.model.d_f,
                d_c=int(d_c),
                tau=float(tau),
                eps=cfg.model.eps,
                use_ce=cfg.model.use_ce,
                use_fisher=cfg.model.use_fisher,
                n_classes=cfg.data.n_classes,
            ).to(device)
            model.fit(x_s, x_v, x_f, y, d_c=int(d_c), tau=float(tau))
            z_val = model(x_s_val, x_v_val, x_f_val)
            score = float(fdr(z_val, y_val, cfg.data.n_classes, cfg.model.eps).item())
            if score > best[0]:
                best = (score, float(tau), int(d_c))
    return best[1], best[2]


def train_one(
    cfg: UMCFConfig,
    data,
    seed: int,
    device: str,
    use_gi: Optional[bool] = None,
    use_ce: Optional[bool] = None,
    use_fisher: Optional[bool] = None,
) -> Tuple[UMCF, dict]:
    if use_ce is not None:
        cfg.model.use_ce = use_ce
    if use_fisher is not None:
        cfg.model.use_fisher = use_fisher

    tr, va, te = stratified_split(
        data["y"], cfg.data.train_ratio, seed, cfg.data.val_ratio_of_train
    )
    train = pack_split(data, tr, device=device)
    val = pack_split(data, va, device=device)
    test = pack_split(data, te, device=device)

    tau, d_c = cfg.model.tau, cfg.model.d_c
    if cfg.train.confirm_hyperparams and val.y.numel() > 0:
        tau, d_c = confirm_hyperparams(
            cfg,
            train.x_s, train.x_v, train.x_f, train.y,
            val.x_s, val.x_v, val.x_f, val.y,
            device,
        )

    model = build_model(cfg, device)
    model.fit(train.x_s, train.x_v, train.x_f, train.y, d_c=d_c, tau=tau)
    meta = {
        "seed": seed,
        "tau": tau,
        "d_c": d_c,
        "n_train": int(train.y.numel()),
        "n_val": int(val.y.numel()),
        "n_test": int(test.y.numel()),
        "weights": model.layer_weights(),
        "fisher_scores": model.fisher_scores(),
        "class_names": list(CLASS_NAMES),
        "train_idx": tr.tolist(),
        "val_idx": va.tolist(),
        "test_idx": te.tolist(),
    }
    return model, meta


def save_checkpoint(model: UMCF, meta: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta, "cfg_model": {
        "d_s": model.d_s, "d_v": model.d_v, "d_f": model.d_f, "d_c": model.d_c,
        "tau": model.tau, "eps": model.eps, "use_ce": model.use_ce,
        "use_fisher": model.use_fisher, "n_classes": model.n_classes,
    }}, path)
    sidecar = Path(path).with_suffix(".json")
    dump = {k: v for k, v in meta.items() if k not in ("train_idx", "val_idx", "test_idx")}
    dump["n_train"] = meta["n_train"]
    dump["n_test"] = meta["n_test"]
    sidecar.write_text(json.dumps(dump, indent=2), encoding="utf-8")


def load_checkpoint(path: str, device: str = "cpu") -> Tuple[UMCF, dict]:
    blob = torch.load(path, map_location=device)
    kw = blob["cfg_model"]
    model = UMCF(**kw).to(device)
    model.load_state_dict(blob["state_dict"])
    model.eval()
    return model, blob.get("meta", {})


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Train UMCF (Algorithm 2)")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--data", type=str, required=True, help="path to .npz dataset")
    p.add_argument("--out", type=str, default="checkpoints/umcf.pt")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--device", type=str, default=None)
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg.seed)
    device = args.device or cfg.resolved_device()
    torch.manual_seed(seed)

    data = load_dataset(args.data)
    model, meta = train_one(cfg, data, seed=seed, device=device)
    save_checkpoint(model, meta, args.out)
    print("Saved", args.out)
    print("Fisher weights:", meta["weights"])
    print("Fisher scores J_l:", meta["fisher_scores"])


if __name__ == "__main__":
    main()
