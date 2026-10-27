"""Five NORAD/TLE orbital prototypes (paper Table 1) and static ranges (Table 2)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import math


CLASS_NAMES = ("RBF", "MLI", "SPF", "MSF", "CAF")
CLASS_TO_ID = {n: i for i, n in enumerate(CLASS_NAMES)}

# Table 1
PROTOTYPES = {
    "RBF": dict(norad=24877, orbit_class="LEO", object_name="DELTA 2 R/B", type="RBF"),
    "MLI": dict(norad=30670, orbit_class="SSO", object_name="FENGYUN 1C DEB", type="MLI"),
    "SPF": dict(norad=5, orbit_class="MEO", object_name="VANGUARD 1", type="SPF"),
    "MSF": dict(norad=7331, orbit_class="LEO", object_name="SL-12 R/B", type="MSF"),
    "CAF": dict(norad=33772, orbit_class="LEO", object_name="IRIDIUM 33 DEB", type="CAF"),
}

# Approximate Keplerian elements used to initialize SGP4 (public two-line
# mean elements of the named NORAD objects; used as orbital prototypes only).
# mean_motion: revolutions per day
ORBIT_ELEMENTS = {
    "RBF": dict(  # DELTA 2 R/B, sun-sync-ish LEO
        inclination_deg=97.40,
        raan_deg=80.0,
        eccentricity=0.0035,
        argp_deg=120.0,
        mean_anomaly_deg=0.0,
        mean_motion=14.80,
        bstar=1.0e-4,
    ),
    "MLI": dict(  # FENGYUN 1C DEB, SSO
        inclination_deg=98.80,
        raan_deg=40.0,
        eccentricity=0.0018,
        argp_deg=90.0,
        mean_anomaly_deg=40.0,
        mean_motion=14.12,
        bstar=2.0e-4,
    ),
    "SPF": dict(  # VANGUARD 1 (paper labels MEO)
        inclination_deg=34.25,
        raan_deg=10.0,
        eccentricity=0.184,
        argp_deg=130.0,
        mean_anomaly_deg=20.0,
        mean_motion=10.85,
        bstar=5.0e-5,
    ),
    "MSF": dict(  # SL-12 R/B prototype placed in LEO as in Table 1
        inclination_deg=64.80,
        raan_deg=150.0,
        eccentricity=0.0060,
        argp_deg=60.0,
        mean_anomaly_deg=80.0,
        mean_motion=14.05,
        bstar=8.0e-5,
    ),
    "CAF": dict(  # IRIDIUM 33 DEB, LEO
        inclination_deg=86.40,
        raan_deg=200.0,
        eccentricity=0.0008,
        argp_deg=30.0,
        mean_anomaly_deg=10.0,
        mean_motion=14.34,
        bstar=1.5e-4,
    ),
}

# Table 2: class-conditional static physical-layer ranges
# Keys follow the 10-D x_s layout in Section II-B.
STATIC_RANGES: Dict[str, Dict[str, Tuple[float, float]]] = {
    "RBF": {
        "size": (1.0, 8.0),
        "area": (0.5, 30.0),
        "amr": (0.005, 0.05),
        "complexity": (0.25, 0.65),
        "vis_ref": (0.20, 0.45),
        "ir_emi": (0.65, 0.85),
        "solar_abs": (0.55, 0.80),
        "roughness": (5.0, 50.0),
        "permittivity": (3.0, 8.0),
        "conductivity": (1e5, 1e7),
    },
    "MLI": {
        "size": (0.1, 2.0),
        "area": (0.01, 4.0),
        "amr": (0.10, 1.50),
        "complexity": (0.55, 0.95),
        "vis_ref": (0.45, 0.85),
        "ir_emi": (0.35, 0.75),
        "solar_abs": (0.15, 0.55),
        "roughness": (1.0, 20.0),
        "permittivity": (2.0, 4.0),
        "conductivity": (1e-2, 1e2),
    },
    "SPF": {
        "size": (0.3, 5.0),
        "area": (0.1, 25.0),
        "amr": (0.02, 0.3),
        "complexity": (0.60, 1.00),
        "vis_ref": (0.10, 0.35),
        "ir_emi": (0.70, 0.90),
        "solar_abs": (0.65, 0.92),
        "roughness": (2.0, 30.0),
        "permittivity": (4.0, 12.0),
        "conductivity": (1e3, 1e6),
    },
    "MSF": {
        "size": (0.05, 2.0),
        "area": (0.005, 3.0),
        "amr": (0.01, 0.20),
        "complexity": (0.35, 0.85),
        "vis_ref": (0.25, 0.60),
        "ir_emi": (0.20, 0.60),
        "solar_abs": (0.40, 0.75),
        "roughness": (10.0, 100.0),
        "permittivity": (1.5, 5.0),
        "conductivity": (1e6, 1e8),
    },
    "CAF": {
        "size": (0.1, 3.0),
        "area": (0.02, 8.0),
        "amr": (0.03, 0.60),
        "complexity": (0.55, 0.95),
        "vis_ref": (0.15, 0.55),
        "ir_emi": (0.55, 0.90),
        "solar_abs": (0.45, 0.85),
        "roughness": (5.0, 80.0),
        "permittivity": (2.5, 7.0),
        "conductivity": (1e0, 1e4),
    },
}

STATIC_KEYS = (
    "size",
    "area",
    "amr",
    "complexity",
    "vis_ref",
    "ir_emi",
    "solar_abs",
    "roughness",
    "permittivity",
    "conductivity",
)

# Protocol: ten slowly varying coordinates from the same SGP4 ephemeris.
SLOW_KEYS = (
    "da_dt",
    "de_dt",
    "di_dt",
    "dRAAN_dt",
    "dargp_dt",
    "dn_dt",
    "period",
    "nodal_drift",
    "perigee_drift",
    "aging",
)

# Fig. 4(b): qualitative type–modality coupling (darker = stronger designed coupling)
MODALITY_COUPLING = {
    "RBF": (0.60, 0.95, 0.55, 0.40, 0.50),
    "MLI": (0.90, 0.30, 0.85, 0.75, 0.60),
    "SPF": (0.85, 0.50, 0.70, 0.60, 0.85),
    "MSF": (0.50, 0.95, 0.40, 0.35, 0.40),
    "CAF": (0.70, 0.45, 0.75, 0.65, 0.90),
}


def mean_motion_rad_per_min(rev_per_day: float) -> float:
    return rev_per_day * 2.0 * math.pi / 1440.0


@dataclass
class SampleGeometry:
    slant_range_km: float
    solar_phase_deg: float
    att_roll_deg: float
    att_pitch_deg: float
    att_yaw_deg: float
    tumble_hz: float
    tumble_phase: float
