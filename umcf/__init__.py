"""Unified Multimodal Coupled Fingerprint (UMCF).

Implements the mathematical model in Section II of the paper and the
experimental protocol in Section III.
"""

from .config import UMCFConfig, load_config
from .model import UMCF
from .gi import ObservationInvariantExtractor
from .distance import decomposable_squared_distance

__all__ = [
    "UMCF",
    "UMCFConfig",
    "load_config",
    "ObservationInvariantExtractor",
    "decomposable_squared_distance",
]

__version__ = "1.0.0"
