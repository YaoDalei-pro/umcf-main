"""CLI: generate the semi-realistic multimodal dataset (Section III-A)."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .data import generate_dataset, save_dataset


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Generate UMCF semi-realistic dataset")
    p.add_argument("--config", type=str, default=None)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--n-per-class", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--quick", action="store_true")
    args = p.parse_args(argv)

    cfg = load_config(args.config)
    seed = int(args.seed if args.seed is not None else cfg.seed)
    n_pc = int(args.n_per_class if args.n_per_class is not None else cfg.data.n_per_class)
    if args.quick:
        n_pc = min(n_pc, 40)
    out = args.out or str(Path(cfg.data.output_dir) / cfg.data.dataset_name)
    data = generate_dataset(cfg, seed=seed, n_per_class=n_pc, use_gi=True)
    save_dataset(data, out)
    print(f"Saved {out}  N={data['y'].shape[0]}  x_s={data['x_s'].shape}  x_f={data['x_f'].shape}")


if __name__ == "__main__":
    main()
