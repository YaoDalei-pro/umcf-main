"""Physics-driven multimodal response simulator (Section III-A, Fig. 4).

Produces optical, radar, infrared, polarization, and spectral sequences
from static physical parameters, orbital geometry, attitude tumbling, and
solar phase angle. Geometric perturbation is a small attitude / solar-phase
jitter applied *before* GI.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from .prototypes import MODALITY_COUPLING, STATIC_KEYS, STATIC_RANGES, SampleGeometry


def _uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(rng.uniform(lo, hi))


def _log_uniform(rng: np.random.Generator, lo: float, hi: float) -> float:
    return float(10 ** rng.uniform(np.log10(lo), np.log10(hi)))


def sample_static(class_name: str, rng: np.random.Generator) -> np.ndarray:
    """Sample 10-D x_s from Table 2 ranges (conductivity log-uniform)."""
    r = STATIC_RANGES[class_name]
    size = _uniform(rng, *r["size"])
    # Projected area correlated with size^2, then clipped to the table range
    area_nom = math_clip(0.6 * size ** 2 * _uniform(rng, 0.4, 1.4), *r["area"])
    amr = _uniform(rng, *r["amr"])
    complexity = _uniform(rng, *r["complexity"])
    vis_ref = _uniform(rng, *r["vis_ref"])
    ir_emi = _uniform(rng, *r["ir_emi"])
    solar_abs = _uniform(rng, *r["solar_abs"])
    roughness = _uniform(rng, *r["roughness"])
    permittivity = _uniform(rng, *r["permittivity"])
    conductivity_log10 = float(np.log10(_log_uniform(rng, *r["conductivity"])))
    x_s = np.array(
        [
            size,
            area_nom,
            amr,
            complexity,
            vis_ref,
            ir_emi,
            solar_abs,
            roughness,
            permittivity,
            conductivity_log10,
        ],
        dtype=np.float64,
    )
    return x_s


def math_clip(v: float, lo: float, hi: float) -> float:
    return float(min(max(v, lo), hi))


def sample_geometry(
    rng: np.random.Generator,
    slant_range_km: float,
    geom_jitter_deg: float = 0.0,
) -> SampleGeometry:
    return SampleGeometry(
        slant_range_km=float(slant_range_km),
        solar_phase_deg=float(rng.uniform(20.0, 140.0) + rng.normal(0.0, geom_jitter_deg)),
        att_roll_deg=float(rng.uniform(0.0, 360.0) + rng.normal(0.0, geom_jitter_deg)),
        att_pitch_deg=float(rng.uniform(-30.0, 30.0) + rng.normal(0.0, geom_jitter_deg)),
        att_yaw_deg=float(rng.uniform(0.0, 360.0) + rng.normal(0.0, geom_jitter_deg)),
        tumble_hz=float(rng.uniform(0.02, 0.35)),
        tumble_phase=float(rng.uniform(0.0, 2.0 * np.pi)),
    )


def perturb_geometry(geo: SampleGeometry, rng: np.random.Generator, jitter_deg: float) -> SampleGeometry:
    return SampleGeometry(
        slant_range_km=geo.slant_range_km,
        solar_phase_deg=geo.solar_phase_deg + rng.normal(0.0, jitter_deg),
        att_roll_deg=geo.att_roll_deg + rng.normal(0.0, jitter_deg),
        att_pitch_deg=geo.att_pitch_deg + rng.normal(0.0, jitter_deg),
        att_yaw_deg=geo.att_yaw_deg + rng.normal(0.0, jitter_deg),
        tumble_hz=geo.tumble_hz,
        tumble_phase=geo.tumble_phase,
    )


def simulate_sequences(
    x_s: np.ndarray,
    class_name: str,
    geo: SampleGeometry,
    rng: np.random.Generator,
    t_len: int = 64,
    sensor_noise: float = 0.02,
) -> np.ndarray:
    """Return sequences of shape (5, T): optical, radar, IR, polarization, spectral."""
    size, area, amr, complexity, vis_ref, ir_emi, solar_abs, roughness, eps_r, cond_log = x_s
    cond = 10.0 ** float(cond_log)
    coup = np.asarray(MODALITY_COUPLING[class_name], dtype=np.float64)
    t = np.linspace(0.0, 1.0, t_len)
    tumble = 2.0 * np.pi * geo.tumble_hz * 60.0 * t + geo.tumble_phase  # ~1 min arc
    att = (
        np.cos(np.deg2rad(geo.att_roll_deg) + tumble)
        * np.cos(np.deg2rad(geo.att_pitch_deg))
        * np.cos(np.deg2rad(geo.att_yaw_deg) + 0.35 * tumble)
    )
    # Solar phase function (simple illuminated-fraction + specular glint)
    alpha = np.deg2rad(geo.solar_phase_deg)
    phase = np.clip(np.cos(alpha), 0.0, 1.0) * (1.0 + 0.25 * att)
    glint = np.exp(-((np.sin(tumble + alpha)) ** 2) / (0.15 + 0.01 * roughness / 50.0))

    r_km = max(geo.slant_range_km, 1.0)
    # Optical light curve ~ reflectance * projected area * phase / r^2
    optical = coup[0] * vis_ref * area * (0.55 * phase + 0.20 * glint * (1.0 - complexity)) / (r_km ** 2)
    optical = optical * (1.0 + 0.15 * complexity * att)

    # Radar RCS ~ size^2 * Fresnel-like factor from permittivity/conductivity
    sigma_c = 1e7 / (cond + 1e-12)
    fresnel = ((eps_r - 1.0) ** 2 + (sigma_c) ** 0.25) / ((eps_r + 1.0) ** 2 + 1.0)
    rcs = coup[1] * (size ** 2) * fresnel * (0.6 + 0.4 * np.abs(att)) / (r_km ** 4)
    rcs = rcs * (1.0 + 0.1 * np.sin(2.0 * tumble))

    # Infrared ~ emissivity * solar absorptance * area * (thermal lag)
    thermal = 0.7 + 0.3 * np.sin(tumble * 0.5 + solar_abs)
    infrared = coup[2] * ir_emi * solar_abs * area * thermal / (r_km ** 2)

    # Polarization degree ~ roughness / complexity modulation
    pol = coup[3] * (0.15 + 0.7 * (roughness / 100.0) * complexity) * (0.5 + 0.5 * np.abs(np.sin(tumble + alpha)))
    pol = pol / max(r_km, 1.0) ** 2

    # Spectral band-integrated response (material-driven)
    spec_base = 0.4 * vis_ref + 0.3 * (eps_r / 12.0) + 0.3 * (1.0 - ir_emi)
    spectral = coup[4] * spec_base * area * (0.6 + 0.4 * phase) / (r_km ** 2)
    spectral = spectral * (1.0 + 0.12 * np.sin(3.0 * tumble))

    seq = np.stack([optical, rcs, infrared, pol, spectral], axis=0)
    seq = np.clip(seq, 0.0, None)
    seq = seq + rng.normal(0.0, sensor_noise, size=seq.shape) * (np.median(seq, axis=1, keepdims=True) + 1e-12)
    seq = np.clip(seq, 0.0, None)
    return seq.astype(np.float32)


def geometry_to_array(geo: SampleGeometry) -> np.ndarray:
    return np.array(
        [
            geo.slant_range_km,
            geo.solar_phase_deg,
            geo.att_roll_deg,
            geo.att_pitch_deg,
            geo.att_yaw_deg,
            geo.tumble_hz,
            geo.tumble_phase,
        ],
        dtype=np.float32,
    )


def array_to_geometry(a: np.ndarray) -> SampleGeometry:
    return SampleGeometry(
        slant_range_km=float(a[0]),
        solar_phase_deg=float(a[1]),
        att_roll_deg=float(a[2]),
        att_pitch_deg=float(a[3]),
        att_yaw_deg=float(a[4]),
        tumble_hz=float(a[5]),
        tumble_phase=float(a[6]),
    )
