from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pandas as pd

from warmpool import climate, forecast, stats
from warmpool.config import Config
from warmpool.load import SalesModel


def hdd_percent(panel: climate.Panel, half: int, known_year: int) -> tuple[np.ndarray, np.ndarray]:
    known = panel.index(known_year)
    hdd = panel.values["hdd"]
    normal = stats.window_normal(hdd, half, known)
    anom = hdd - normal
    a = climate.winter_months(anom, (12, 1, 2)).sum(axis=1)
    n = climate.winter_months(normal, (12, 1, 2)).sum(axis=1)
    return panel.years[:-1], 100.0 * a / n


def climate_level(cfg: Config, ctx: dict, units: list[str]) -> pd.DataFrame:
    f = cfg.forecast
    panel = ctx["panel"].subset(units)
    half = cfg.climate.anomaly_half_window
    rows = []
    last = int(panel.years[-2])
    for target in range(f.first_climate_winter, last + 1):
        winters, pct = hdd_percent(panel, half, target - 1)
        pos = {int(y): i for i, y in enumerate(winters)}
        hist = np.array([y for y in winters if y < target and np.isfinite(pct[pos[y]]).all()])
        s = forecast.setup(cfg, ctx["series"], ctx["winters"], target, hist)
        members = pct[[pos[int(y)] for y in s.years]]
        observed = pct[pos[target]]
        for method in f.methods:
            for j, unit in enumerate(units):
                mu, w, sigma = forecast.predictive(method, s, members[:, j], gate=f.gate)
                row = forecast.summarize(mu, w, sigma, f.interval, observed[j])
                rows.append(
                    {"level": "climate", "unit": unit, "winter": target, "method": method, **row}
                )
    return pd.DataFrame(rows)


def sales_level(cfg: Config, ctx: dict, units: list[str], unit_sales: pd.DataFrame) -> pd.DataFrame:
    f = cfg.forecast
    panel = ctx["panel"].subset(units)
    model = SalesModel(cfg, panel, unit_sales)
    years_all = panel.years[:-1]
    rows = []
    last_target = int(panel.years[-1]) - 1
    for target in range(f.first_sales_winter, last_target + 1):
        hist = np.array([y for y in years_all if y < target])
        s = forecast.setup(cfg, ctx["series"], ctx["winters"], target, hist)
        normals = model.normals(target)
        for j, unit in enumerate(units):
            fit = model.fit(j, target)
            if fit is None:
                continue
            observed = model.observed(fit, j, target, normals)
            if not np.isfinite(observed):
                continue
            members = model.member_effects(fit, j, target, s.years)
            shift, sigma = model.calibration(fit, j, target)
            actual = model.member_effects(fit, j, target, np.array([target]))
            for method in f.methods + ("perfect_weather",):
                if method == "perfect_weather":
                    mu, w, spread = actual + shift, np.array([1.0]), sigma
                else:
                    mu, w, spread = forecast.predictive(method, s, members, shift, sigma, f.gate)
                row = forecast.summarize(mu, w, spread, f.interval, observed)
                rows.append(
                    {
                        "level": "sales",
                        "unit": unit,
                        "winter": target,
                        "method": method,
                        "billing_weight": fit.weight,
                        **row,
                    }
                )
    return pd.DataFrame(rows)


def scores(cfg: Config, detail: pd.DataFrame, winters: pd.DataFrame) -> pd.DataFrame:
    f = cfg.forecast
    rng = np.random.default_rng(f.seed)
    info = winters.set_index("winter")
    detail = detail.assign(signal=detail["winter"].map(info["predictor"]))
    edge = cfg.enso.threshold
    subsets = {
        "all": lambda d: d,
        "el_nino_at_issue": lambda d: d[d["signal"] >= edge],
        "strong_el_nino_at_issue": lambda d: d[d["signal"] >= f.gate],
        "la_nina_at_issue": lambda d: d[d["signal"] <= -edge],
        "neutral_at_issue": lambda d: d[d["signal"].abs() < edge],
    }
    rows = []
    for (level, unit), part in detail.groupby(["level", "unit"]):
        wide = part.pivot_table(index="winter", columns="method", values="crps")
        for name, pick in subsets.items():
            sub = pick(part)
            if sub.empty:
                continue
            chosen = wide.loc[sorted(sub["winter"].unique())]
            ref = chosen["climatology"].to_numpy()
            for method in [m for m in chosen.columns if m in f.methods or m == "perfect_weather"]:
                vals = chosen[method].to_numpy()
                mean = float(vals.mean())
                skill = 1.0 - mean / float(ref.mean())
                idx = rng.integers(0, vals.size, size=(f.bootstrap, vals.size))
                boot = 1.0 - vals[idx].mean(axis=1) / ref[idx].mean(axis=1)
                cover = sub.loc[sub["method"] == method, "covered"].mean()
                rows.append(
                    {
                        "level": level,
                        "unit": unit,
                        "subset": name,
                        "method": method,
                        "winters": int(vals.size),
                        "crps": mean,
                        "crpss": skill,
                        "crpss_lo": float(np.quantile(boot, 0.05)),
                        "crpss_hi": float(np.quantile(boot, 0.95)),
                        "coverage": float(cover),
                    }
                )
    return pd.DataFrame(rows)


def run(
    cfg: Config, ctx: dict, unit_sales: pd.DataFrame, log: Callable[[str], None] = print
) -> dict[str, pd.DataFrame]:
    units = [u for u in ctx["panel"].units]
    clim = climate_level(cfg, ctx, units)
    log(f"climate hindcast: {clim['winter'].nunique()} winters x {len(units)} units")
    sales = sales_level(cfg, ctx, units, unit_sales)
    log(f"sales hindcast: {sales['winter'].nunique()} winters x {sales['unit'].nunique()} units")
    detail = pd.concat([clim, sales], ignore_index=True)
    return {"backtest": detail, "backtest_score": scores(cfg, detail, ctx["winters"])}
