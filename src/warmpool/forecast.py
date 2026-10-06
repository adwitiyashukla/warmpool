from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from warmpool import analogs, dtw, stats
from warmpool.config import Config


@dataclass
class Setup:
    target: int
    years: np.ndarray
    predictor: np.ndarray
    x0: float
    weights: dict[str, np.ndarray] = field(default_factory=dict)
    ess: float = np.nan
    bandwidth: float = np.nan
    analog_years: list[int] = field(default_factory=list)


def setup(
    cfg: Config,
    series: pd.Series,
    winters: pd.DataFrame,
    target: int,
    years: np.ndarray,
    x0: float | None = None,
) -> Setup:
    f, a = cfg.forecast, cfg.analog
    info = winters.set_index("winter")
    years = np.array([y for y in years if y < target and y in info.index])
    x = info.loc[years, "predictor"].to_numpy(dtype=float)
    x0 = float(info.loc[target, "predictor"]) if x0 is None else float(x0)
    out = Setup(target=target, years=years, predictor=x, x0=x0)
    out.weights["climatology"] = np.full(years.size, 1.0 / years.size)
    w, h = stats.kernel_weights(x, x0, f.bandwidth, f.min_ess)
    out.weights["enso"], out.bandwidth, out.ess = w, h, stats.kish_ess(w)
    last = cfg.enso.predictor_month
    query = analogs.window(series, target, a.start_month, last)
    pool = {}
    for y in years:
        seq = analogs.window(series, int(y), a.start_month, last)
        if seq is not None:
            pool[int(y)] = seq
    found, _ = dtw.nearest(query, pool, a.band, a.top_k)
    out.analog_years = [y for y, _ in found]
    chosen = np.isin(years, out.analog_years).astype(float)
    out.weights["analog"] = chosen / chosen.sum()
    return out


def scenario_setup(
    cfg: Config, winters: pd.DataFrame, target: int, years: np.ndarray, djf: float
) -> Setup:
    info = winters.set_index("winter")
    years = np.array([y for y in years if y < target and y in info.index])
    values = info.loc[years, "djf"].to_numpy(dtype=float)
    keep = np.isfinite(values)
    years, values = years[keep], values[keep]
    out = Setup(target=target, years=years, predictor=values, x0=djf)
    w, h = stats.kernel_weights(values, djf, cfg.forecast.bandwidth, cfg.forecast.min_ess)
    out.weights["scenario"], out.bandwidth, out.ess = w, h, stats.kish_ess(w)
    return out


def predictive(
    method: str,
    s: Setup,
    members: np.ndarray,
    shift: float = 0.0,
    sigma: float = 0.0,
    gate: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, float]:
    if method == "gated":
        method = "enso" if abs(s.x0) >= gate else "climatology"
    if method == "linear":
        slope, intercept = np.polyfit(s.predictor, members, 1)
        resid = members - (intercept + slope * s.predictor)
        spread = float(np.sqrt(resid.var(ddof=2) + sigma**2))
        return np.array([intercept + slope * s.x0 + shift]), np.array([1.0]), spread
    return members + shift, s.weights[method], sigma


def summarize(
    mu: np.ndarray, w: np.ndarray, sigma: float, interval: float, observed: float | None = None
) -> dict:
    tail = (1.0 - interval) / 2.0
    lo, mid, hi = stats.mixture_quantiles(mu, w, sigma, (tail, 0.5, 1.0 - tail))
    out = {
        "mean": float(np.sum(mu * w / w.sum())),
        "lo": lo,
        "median": mid,
        "hi": hi,
        "prob_above": 1.0 - stats.mixture_cdf(mu, w, sigma, 0.0),
    }
    if observed is not None and np.isfinite(observed):
        out["observed"] = float(observed)
        out["crps"] = stats.crps_mixture(mu, w, sigma, observed)
        out["pit"] = stats.mixture_cdf(mu, w, sigma, observed)
        out["covered"] = bool(lo <= observed <= hi)
    return out
