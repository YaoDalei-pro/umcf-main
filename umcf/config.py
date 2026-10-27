"""Configuration matching paper Table 3 and experimental protocol."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, List, Optional

import yaml


def _resolve_device(name: str) -> str:
    if name == "auto":
        try:
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    return name


@dataclass
class ModelConfig:
    d_s: int = 10
    d_v: int = 10
    d_f: int = 16
    d_c: int = 20
    d_u: int = 420
    d_z: int = 56
    tau: float = 1.0
    eps: float = 1e-8
    use_gi: bool = True
    use_ce: bool = True
    use_fisher: bool = True


@dataclass
class DataConfig:
    n_classes: int = 5
    n_per_class: int = 2000
    train_ratio: float = 0.7
    val_ratio_of_train: float = 0.1
    sequence_length: int = 64
    n_modalities: int = 5
    snr_db_baseline: float = 20.0
    snr_db_list: List[float] = field(
        default_factory=lambda: [0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0]
    )
    missing_ratios: List[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    )
    geom_jitter_deg: float = 5.0
    output_dir: str = "data/processed"
    dataset_name: str = "umcf_semireal.npz"


@dataclass
class TrainConfig:
    repeats: int = 20
    confirm_hyperparams: bool = False
    tau_grid: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    dc_grid: List[int] = field(default_factory=lambda: [10, 20, 30])
    save_dir: str = "checkpoints"


@dataclass
class EvalConfig:
    snr_ref_db: float = 20.0
    missing_noise_db: float = 20.0
    ablation_noise_db: float = 10.0
    ablation_missing_ratio: float = 0.3
    plot_dir: str = "results/figures"
    result_dir: str = "results"


@dataclass
class BaselineConfig:
    dnnf_epochs: int = 40
    dnnf_lr: float = 1e-3
    dnnf_hidden: List[int] = field(default_factory=lambda: [128, 64])
    dnnf_embed: int = 32
    dnnf_batch_size: int = 128
    lmnn_epochs: int = 30
    lmnn_lr: float = 1e-2
    lmnn_embed: int = 36
    cmaf_epochs: int = 40
    cmaf_lr: float = 1e-3


@dataclass
class UMCFConfig:
    seed: int = 42
    device: str = "auto"
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)

    def resolved_device(self) -> str:
        return _resolve_device(self.device)

    def to_dict(self) -> dict:
        return asdict(self)


def _merge(dc, raw: dict) -> None:
    for k, v in raw.items():
        if not hasattr(dc, k):
            continue
        cur = getattr(dc, k)
        if hasattr(cur, "__dataclass_fields__") and isinstance(v, dict):
            _merge(cur, v)
        else:
            setattr(dc, k, v)


def load_config(path: Optional[str] = None) -> UMCFConfig:
    cfg = UMCFConfig()
    if path is None:
        default = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
        path = str(default) if default.exists() else None
    if path is not None:
        with open(path, "r", encoding="utf-8") as f:
            raw: Any = yaml.safe_load(f) or {}
        _merge(cfg, raw)
    return cfg
