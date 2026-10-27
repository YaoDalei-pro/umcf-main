"""Inference: frozen Phi(x; Theta) -> 56-D fingerprint and optional gallery match."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .config import load_config
from .data import load_dataset, pack_split
from .distance import pairwise_squared_euclidean
from .prototypes import CLASS_NAMES
from .train import load_checkpoint


@torch.no_grad()
def fingerprint(
    model,
    x_s: torch.Tensor,
    x_v: torch.Tensor,
    x_f: torch.Tensor,
    snr_db: Optional[float] = None,
):
    return model(x_s, x_v, x_f, return_layers=True, snr_db=snr_db)


@torch.no_grad()
def match_gallery(z_query: torch.Tensor, z_gallery: torch.Tensor, y_gallery: torch.Tensor, topk: int = 5):
    d2 = pairwise_squared_euclidean(z_query, z_gallery)
    val, idx = torch.topk(d2, k=min(topk, z_gallery.size(0)), dim=1, largest=False)
    pred = y_gallery[idx]
    return val.cpu().numpy(), pred.cpu().numpy(), idx.cpu().numpy()


def _tensor(arr, device):
    if isinstance(arr, torch.Tensor):
        return arr.to(device=device, dtype=torch.float32)
    a = np.asarray(arr, dtype=np.float32)
    if a.ndim == 1:
        a = a[None, :]
    return torch.from_numpy(a).to(device)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="UMCF inference (frozen Theta)")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--data", type=str, default=None, help="gallery dataset .npz")
    p.add_argument("--index", type=int, default=0, help="sample index in dataset")
    p.add_argument("--x-s", type=str, default=None, help="npy path to 10-D static vector")
    p.add_argument("--x-v", type=str, default=None)
    p.add_argument("--x-f", type=str, default=None)
    p.add_argument("--topk", type=int, default=5)
    p.add_argument(
        "--omit-static",
        action="store_true",
        help="drop the static layer and renormalize the remaining Fisher weights",
    )
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--out", type=str, default="results/infer.json")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    device = args.device or cfg.resolved_device()
    model, meta = load_checkpoint(args.ckpt, device=device)

    if args.x_s is not None:
        x_s = _tensor(np.load(args.x_s), device)
        x_v = _tensor(np.load(args.x_v), device)
        x_f = _tensor(np.load(args.x_f), device)
        y_true = None
    else:
        if args.data is None:
            raise SystemExit("Provide --data or --x-s/--x-v/--x-f")
        data = load_dataset(args.data)
        x_s = _tensor(data["x_s"][args.index], device)
        x_v = _tensor(data["x_v"][args.index], device)
        x_f = _tensor(data["x_f"][args.index], device)
        y_true = int(data["y"][args.index])

    z_full, layers = model(x_s, x_v, x_f, return_layers=True)
    if args.omit_static:
        z, w_used = model.assemble_omitting(layers, omit=("s",))
    else:
        z, w_used = z_full, model.layer_weights()
    out = {
        "z": z.squeeze(0).cpu().tolist(),
        "z_s": layers["s"].squeeze(0).cpu().tolist(),
        "z_v": layers["v"].squeeze(0).cpu().tolist(),
        "z_f": layers["f"].squeeze(0).cpu().tolist(),
        "z_c": layers["c"].squeeze(0).cpu().tolist(),
        "weights": w_used,
        "omit_static": bool(args.omit_static),
        "dim": int(z.size(-1)),
        "y_true": y_true,
        "y_true_name": None if y_true is None else CLASS_NAMES[y_true],
    }

    if args.data is not None and meta.get("train_idx") is not None:
        data = load_dataset(args.data)
        gallery = pack_split(data, np.array(meta["train_idx"]), device=device)
        _zg, g_layers = model(gallery.x_s, gallery.x_v, gallery.x_f, return_layers=True)
        if args.omit_static:
            z_g, _w = model.assemble_omitting(g_layers, omit=("s",))
        else:
            z_g = _zg
        dist, pred, idx = match_gallery(z, z_g, gallery.y, topk=args.topk)
        names = [CLASS_NAMES[int(c)] for c in pred[0]]
        out["match"] = {
            "topk_distance": dist[0].tolist(),
            "topk_class_id": pred[0].astype(int).tolist(),
            "topk_class_name": names,
            "topk_gallery_index": idx[0].astype(int).tolist(),
            "rank1_class": names[0],
        }
        z_nn = z_g[idx[0, 0] : idx[0, 0] + 1]
        if args.omit_static:
            d_v, d_f, d_c = model.d_v, model.d_f, model.d_c
            slices = {
                "v": slice(0, d_v),
                "f": slice(d_v, d_v + d_f),
                "c": slice(d_v + d_f, d_v + d_f + d_c),
            }
            parts = {k: (z[:, sl] - z_nn[:, sl]).pow(2).sum(-1) for k, sl in slices.items()}
            d2 = sum(parts.values())
        else:
            d2, parts = model.attribute_match(z, z_nn)
        out["match"]["layer_contributions"] = {k: float(v.squeeze().cpu()) for k, v in parts.items()}
        out["match"]["D2"] = float(d2.squeeze().cpu())

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("fingerprint dim =", out["dim"])
    if "match" in out:
        print("Rank-1:", out["match"]["rank1_class"], "D2=", out["match"]["D2"])
        print("layer contributions:", out["match"]["layer_contributions"])
    print("Saved", args.out)


if __name__ == "__main__":
    main()
