from __future__ import annotations

import numpy as np
import pandas as pd

from warmpool import backtest, forecast
from warmpool.config import Config
from warmpool.load import SalesModel


def status(cfg: Config, ctx: dict, weekly: pd.DataFrame) -> pd.DataFrame:
    series = ctx["series"]
    last = series.dropna().index.max()
    same = series[series.index.month == last.month].dropna()
    rank = int((same > series[last]).sum()) + 1
    week = weekly.sort_values("week").iloc[-1]
    rows = [
        ("latest_season_center", str(last.date())),
        ("latest_index", f"{series[last]:.2f}"),
        ("rank_for_calendar_month", f"{rank} of {same.size}"),
        ("latest_week", str(pd.Timestamp(week["week"]).date())),
        ("latest_weekly_nino34_anomaly", f"{week['nino34_anom']:.1f}"),
        ("outlook_winter", f"{cfg.outlook.winter}-{str(cfg.outlook.winter + 1)[-2:]}"),
    ]
    return pd.DataFrame(rows, columns=["key", "value"])


def sample(
    mu: np.ndarray, w: np.ndarray, sigma: float, size: int, rng: np.random.Generator
) -> np.ndarray:
    picks = rng.choice(mu.size, size=size, p=w / w.sum())
    return mu[picks] + (rng.normal(0.0, sigma, size) if sigma > 0 else 0.0)


def price_outlook(cfg: Config, mixtures: dict, premium_table: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.prices.seed + 2)
    rows = []
    for market, part in premium_table.dropna(subset=["premium", "hdd_pct"]).groupby("market"):
        if len(part) < cfg.prices.min_winters:
            continue
        region = part["region"].iloc[0]
        slope, intercept = np.polyfit(part["hdd_pct"], part["premium"], 1)
        resid = part["premium"] - (intercept + slope * part["hdd_pct"])
        sd = float(resid.std(ddof=2))
        for (method, unit), (mu, w, sigma) in mixtures.items():
            if unit != region:
                continue
            hdd = sample(mu, w, sigma, 20000, rng)
            draws = intercept + slope * hdd + rng.normal(0.0, sd, hdd.size)
            lo, mid, hi = np.quantile(draws, [0.1, 0.5, 0.9])
            rows.append(
                {
                    "market": market,
                    "region": region,
                    "method": method,
                    "hdd_median": float(np.median(hdd)),
                    "premium_median": mid,
                    "premium_lo": lo,
                    "premium_hi": hi,
                    "prob_premium_negative": float(np.mean(draws < 0)),
                    "slope": slope,
                    "resid_sd": sd,
                    "winters": len(part),
                }
            )
    return pd.DataFrame(rows)


def run(
    cfg: Config,
    ctx: dict,
    unit_sales: pd.DataFrame,
    weekly: pd.DataFrame,
    premium_table: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    f = cfg.forecast
    target = cfg.outlook.winter
    if target not in set(ctx["winters"]["winter"]):
        raise ValueError(f"no ENSO predictor for winter {target} yet, check outlook.winter")
    panel = ctx["panel"]
    years, pct = backtest.hdd_percent(panel, cfg.climate.anomaly_half_window, target - 1)
    pos = {int(y): i for i, y in enumerate(years)}
    hist = np.array([y for y in years if y < target and np.isfinite(pct[pos[y]]).all()])
    s = forecast.setup(cfg, ctx["series"], ctx["winters"], target, hist)
    members = pct[[pos[int(y)] for y in s.years]]
    rows, mixtures = [], {}
    for method in f.methods:
        for j, unit in enumerate(panel.units):
            mix = forecast.predictive(method, s, members[:, j], gate=f.gate)
            mixtures[(method, unit)] = mix
            rows.append(
                {
                    "level": "climate",
                    "unit": unit,
                    "method": method,
                    **forecast.summarize(*mix, f.interval),
                    "ess": s.ess,
                    "bandwidth": s.bandwidth,
                }
            )
    model = SalesModel(cfg, panel, unit_sales)
    calib = []
    for j, unit in enumerate(panel.units):
        fit = model.fit(j, target)
        if fit is None:
            continue
        effects = model.member_effects(fit, j, target, s.years)
        shift, sigma = model.calibration(fit, j, target)
        calib.append(
            {
                "unit": unit,
                "shift": shift,
                "sigma": sigma,
                "billing_weight": fit.weight,
                "heating_pct_per_hdd": 100.0 * fit.beta[0],
                "cooling_pct_per_cdd": 100.0 * fit.beta[1],
            }
        )
        for method in f.methods:
            mix = forecast.predictive(method, s, effects, shift, sigma, f.gate)
            rows.append(
                {
                    "level": "sales",
                    "unit": unit,
                    "method": method,
                    **forecast.summarize(*mix, f.interval),
                    "ess": s.ess,
                    "bandwidth": s.bandwidth,
                }
            )
    scenarios = []
    for value in cfg.outlook.scenarios:
        sc = forecast.scenario_setup(cfg, ctx["winters"], target, hist, value)
        sm = pct[[pos[int(y)] for y in sc.years]]
        for j, unit in enumerate(panel.units):
            scenarios.append(
                {
                    "unit": unit,
                    "scenario_djf": value,
                    **forecast.summarize(sm[:, j], sc.weights["scenario"], 0.0, f.interval),
                    "ess": sc.ess,
                    "bandwidth": sc.bandwidth,
                }
            )
    labels = ctx["winters"].set_index("winter")["label"]
    us = panel.units.index("US")
    analog = pd.DataFrame(
        [
            {"winter": y, "label": labels[y], "us_hdd_pct": float(pct[pos[y], us])}
            for y in s.analog_years
        ]
    )
    return {
        "outlook": pd.DataFrame(rows),
        "outlook_scenario": pd.DataFrame(scenarios),
        "outlook_analog": analog,
        "outlook_model": pd.DataFrame(calib),
        "outlook_price": price_outlook(cfg, mixtures, premium_table),
        "outlook_status": status(cfg, ctx, weekly),
    }
