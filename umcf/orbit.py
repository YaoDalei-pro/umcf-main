"""SGP4 propagation of NORAD/TLE prototypes and slowly varying features x_v.

x_v (10-D, Section II-B): classical-element rates, period, long-term drift,
and a scalar aging factor derived from the same ephemerides.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from .prototypes import ORBIT_ELEMENTS, mean_motion_rad_per_min

DEG2RAD = math.pi / 180.0


def _sgp4_satrec(class_name: str, rng: np.random.Generator, epoch_days: float = 24515.0):
    """Build a Satrec with small sample-wise element jitter around the prototype."""
    from sgp4.api import Satrec, WGS72

    el = ORBIT_ELEMENTS[class_name]
    # Small per-sample perturbation so x_v is not identical within a class
    inc = (el["inclination_deg"] + rng.normal(0.0, 0.05)) * DEG2RAD
    raan = (el["raan_deg"] + rng.normal(0.0, 1.5)) * DEG2RAD
    ecc = float(np.clip(el["eccentricity"] + rng.normal(0.0, 5e-5), 1e-6, 0.8))
    argp = (el["argp_deg"] + rng.normal(0.0, 1.5)) * DEG2RAD
    ma = (el["mean_anomaly_deg"] + rng.uniform(0.0, 360.0)) * DEG2RAD
    n = el["mean_motion"] + rng.normal(0.0, 0.002)
    no_kozai = mean_motion_rad_per_min(n)
    bstar = el["bstar"] * rng.uniform(0.5, 1.5)
    sat = Satrec()
    sat.sgp4init(
        WGS72,
        "i",
        0,
        epoch_days,
        bstar,
        0.0,
        0.0,
        ecc,
        argp,
        inc,
        ma,
        no_kozai,
        raan,
    )
    return sat, n, bstar


def _keplerian_fallback(class_name: str, rng: np.random.Generator) -> Dict[str, float]:
    """Two-body rates if sgp4 is unavailable (still class-conditional)."""
    el = ORBIT_ELEMENTS[class_name]
    n = el["mean_motion"] + rng.normal(0.0, 0.002)
    period_min = 1440.0 / max(n, 1e-6)
    j2_fac = 1e-3 * (n / 14.0) ** 2
    inc = el["inclination_deg"]
    return dict(
        da_dt=rng.normal(0.0, 0.02),
        de_dt=rng.normal(0.0, 2e-6),
        di_dt=rng.normal(0.0, 1e-4),
        dO_dt=-1.5 * j2_fac * math.cos(inc * DEG2RAD) + rng.normal(0.0, 1e-4),
        dw_dt=0.75 * j2_fac * (5.0 * math.cos(inc * DEG2RAD) ** 2 - 1.0)
        + rng.normal(0.0, 1e-4),
        dn_dt=rng.normal(-1e-6 * el["bstar"] * 1e4, 3e-7),
        period=period_min + rng.normal(0.0, 0.05),
        raan_drift=-1.5 * j2_fac * math.cos(inc * DEG2RAD),
        argp_drift=0.75 * j2_fac * (5.0 * math.cos(inc * DEG2RAD) ** 2 - 1.0),
        bstar=el["bstar"] * rng.uniform(0.5, 1.5),
        mean_motion=n,
        eccentricity=el["eccentricity"],
        inclination_deg=inc,
        slant_range_km=float(np.clip(rng.normal(800.0 if n > 12 else 8000.0, 80.0), 200.0, 40000.0)),
    )


def propagate_slow_features(
    class_name: str,
    rng: np.random.Generator,
    dt_min: float = 20.0,
    n_steps: int = 36,
    aging_base: Optional[float] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Return x_v (10,) and auxiliary orbit dict (including a representative range)."""
    try:
        sat, n0, bstar = _sgp4_satrec(class_name, rng)
        use_sgp4 = True
    except Exception:
        use_sgp4 = False
        sat, n0, bstar = None, None, None

    if not use_sgp4:
        aux = _keplerian_fallback(class_name, rng)
        aging = _aging_factor(class_name, rng, aux["bstar"], aging_base)
        x_v = np.array(
            [
                aux["da_dt"],
                aux["de_dt"],
                aux["di_dt"],
                aux["dO_dt"],
                aux["dw_dt"],
                aux["dn_dt"],
                aux["period"],
                aux["raan_drift"],
                aux["argp_drift"],
                aging,
            ],
            dtype=np.float64,
        )
        return x_v, aux

    from sgp4.api import jday

    # Sample an arc starting at a random day offset
    day0 = rng.uniform(0.0, 5.0)
    jd0, fr0 = jday(2024, 1, 1, 0, 0, 0.0)
    rs, vs = [], []
    for k in range(n_steps):
        dt_days = (day0 + k * dt_min / 1440.0)
        jd = jd0 + int(dt_days)
        fr = fr0 + (dt_days - int(dt_days))
        e, r, v = sat.sgp4(jd, fr)
        if e == 0:
            rs.append(r)
            vs.append(v)
    if len(rs) < 4:
        aux = _keplerian_fallback(class_name, rng)
        aging = _aging_factor(class_name, rng, bstar, aging_base)
        x_v = np.array(
            [
                aux["da_dt"],
                aux["de_dt"],
                aux["di_dt"],
                aux["dO_dt"],
                aux["dw_dt"],
                aux["dn_dt"],
                aux["period"],
                aux["raan_drift"],
                aux["argp_drift"],
                aging,
            ],
            dtype=np.float64,
        )
        return x_v, aux

    r = np.asarray(rs, dtype=np.float64)
    v = np.asarray(vs, dtype=np.float64)
    a_km, ecc, inc, raan, argp, n_rad_s = _rv_to_elements(r, v)
    t_min = np.arange(len(a_km)) * dt_min
    da_dt = _slope(t_min, a_km)
    de_dt = _slope(t_min, ecc)
    di_dt = _slope(t_min, inc)
    dO_dt = _slope(t_min, np.unwrap(raan))
    dw_dt = _slope(t_min, np.unwrap(argp))
    n_rev_day = n_rad_s * 86400.0 / (2.0 * math.pi)
    dn_dt = _slope(t_min, n_rev_day)
    period = float(np.median(2.0 * math.pi / np.clip(n_rad_s, 1e-8, None) / 60.0))
    # Long-term drifts: first vs last third of the arc
    k = max(len(a_km) // 3, 1)
    raan_drift = float((np.mean(np.unwrap(raan)[-k:]) - np.mean(np.unwrap(raan)[:k])) / (t_min[-1] - t_min[0] + 1e-6))
    argp_drift = float((np.mean(np.unwrap(argp)[-k:]) - np.mean(np.unwrap(argp)[:k])) / (t_min[-1] - t_min[0] + 1e-6))
    aging = _aging_factor(class_name, rng, bstar, aging_base)

    # Observer slant range: distance from a ground site at (R_E, 0, 0) approx
    site = np.array([6378.137, 0.0, 0.0])
    slant = float(np.median(np.linalg.norm(r - site, axis=1)))
    slant = float(np.clip(slant, 200.0, 50000.0))

    x_v = np.array(
        [da_dt, de_dt, di_dt, dO_dt, dw_dt, dn_dt, period, raan_drift, argp_drift, aging],
        dtype=np.float64,
    )
    aux = dict(
        slant_range_km=slant,
        mean_motion=float(np.median(n_rev_day)),
        eccentricity=float(np.median(ecc)),
        inclination_deg=float(np.median(inc) / DEG2RAD),
        bstar=bstar,
        period=period,
    )
    return x_v, aux


def _aging_factor(
    class_name: str, rng: np.random.Generator, bstar: float, aging_base: Optional[float]
) -> float:
    # CAF (aged surface) higher, RBF/MSF metal lower; tied to drag term bstar
    class_bias = {"RBF": 0.15, "MLI": 0.45, "SPF": 0.35, "MSF": 0.20, "CAF": 0.70}[class_name]
    if aging_base is None:
        aging_base = class_bias
    return float(np.clip(aging_base + 0.15 * math.log10(abs(bstar) * 1e4 + 1e-12) + rng.normal(0.0, 0.03), 0.0, 1.0))


def _slope(t: np.ndarray, y: np.ndarray) -> float:
    t = t - t.mean()
    denom = float((t ** 2).sum())
    if denom < 1e-12:
        return 0.0
    return float((t * (y - y.mean())).sum() / denom)


def _rv_to_elements(r: np.ndarray, v: np.ndarray, mu: float = 398600.4418):
    """r, v in km, km/s -> arrays of a, e, i, RAAN, argp, n."""
    h = np.cross(r, v)
    hn = np.linalg.norm(h, axis=1, keepdims=True)
    nvec = np.cross(np.array([0.0, 0.0, 1.0]), h)
    nn = np.linalg.norm(nvec, axis=1, keepdims=True) + 1e-12
    rnorm = np.linalg.norm(r, axis=1, keepdims=True)
    vnorm = np.linalg.norm(v, axis=1, keepdims=True)
    evec = np.cross(v, h) / mu - r / rnorm
    ecc = np.linalg.norm(evec, axis=1)
    energy = (vnorm[:, 0] ** 2) / 2.0 - mu / rnorm[:, 0]
    a = -mu / (2.0 * energy)
    inc = np.arccos(np.clip(h[:, 2] / (hn[:, 0] + 1e-12), -1.0, 1.0))
    raan = np.arctan2(nvec[:, 1], nvec[:, 0])
    argp = np.arccos(
        np.clip(((nvec * evec).sum(1) / ((nn[:, 0]) * (ecc + 1e-12))), -1.0, 1.0)
    )
    argp = np.where(evec[:, 2] < 0, 2.0 * math.pi - argp, argp)
    n = np.sqrt(mu / np.clip(np.abs(a) ** 3, 1e-8, None))
    return a, ecc, inc, raan, argp, n
