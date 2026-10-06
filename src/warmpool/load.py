from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from warmpool import climate, stats
from warmpool.config import Config


@dataclass
class Fit:
    weight: float
    beta: np.ndarray
    se: np.ndarray
    alpha: dict[int, float]
    gamma: np.ndarray
    sse: float
    r2: float
    n: int


def per_day(panel: climate.Panel, var: str) -> np.ndarray:
    days = np.array(
        [[pd.Timestamp(int(y), m, 1).days_in_month for m in range(1, 13)] for y in panel.years],
        dtype=float,
    )
    return panel.values[var] / days[:, :, None]


def billing(cube: np.ndarray, weight: float) -> np.ndarray:
    flat = cube.reshape((cube.shape[0] * cube.shape[1],) + cube.shape[2:])
    prev = np.concatenate([np.full((1,) + flat.shape[1:], np.nan), flat[:-1]])
    return (weight * prev + (1.0 - weight) * flat).reshape(cube.shape)


def sales_cube(unit_sales: pd.DataFrame, panel: climate.Panel, sector: str) -> np.ndarray:
    part = unit_sales[(unit_sales["sector"] == sector) & unit_sales["unit"].isin(panel.units)]
    cube = np.full((panel.years.size, 12, len(panel.units)), np.nan)
    pos = {u: i for i, u in enumerate(panel.units)}
    value = np.log(part["sales_mwh"].to_numpy() / part["days"].to_numpy())
    cube[
        part["year"].to_numpy() - panel.years[0],
        part["month"].to_numpy() - 1,
        part["unit"].map(pos).to_numpy(),
    ] = value
    return cube


def fit_unit(
    logy: np.ndarray,
    hdd: dict[float, np.ndarray],
    cdd: dict[float, np.ndarray],
    years: np.ndarray,
    mask: np.ndarray,
    hac_lags: int = 0,
) -> Fit:
    yrs = np.repeat(years, 12)[mask]
    months = np.tile(np.arange(12), years.size)[mask]
    y = logy.ravel()[mask]
    uniq = np.unique(yrs)
    year_dummies = (yrs[:, None] == uniq[None, :]).astype(float)
    month_dummies = (months[:, None] == np.arange(1, 12)[None, :]).astype(float)
    best = None
    for weight in hdd:
        h = hdd[weight].ravel()[mask]
        c = cdd[weight].ravel()[mask]
        x = np.column_stack([year_dummies, month_dummies, h, c])
        res = stats.ols_hac(x, y, hac_lags)
        sse = float(np.sum(res["resid"] ** 2))
        if best is None or sse < best[0] - 1e-12:
            best = (sse, weight, res)
    sse, weight, res = best
    beta = res["beta"]
    k = uniq.size
    return Fit(
        weight=weight,
        beta=beta[-2:],
        se=res["se"][-2:],
        alpha={int(yr): float(beta[i]) for i, yr in enumerate(uniq)},
        gamma=np.concatenate([[0.0], beta[k : k + 11]]),
        sse=sse,
        r2=res["r2"],
        n=int(mask.sum()),
    )


class SalesModel:
    def __init__(self, cfg: Config, panel: climate.Panel, unit_sales: pd.DataFrame) -> None:
        self.cfg = cfg
        self.panel = panel
        self.logy = sales_cube(unit_sales, panel, cfg.forecast.sector)
        self.hdd_pd = per_day(panel, "hdd")
        self.cdd_pd = per_day(panel, "cdd")
        weights = cfg.forecast.billing_weights
        self.hdd_bill = {w: billing(self.hdd_pd, w) for w in weights}
        self.cdd_bill = {w: billing(self.cdd_pd, w) for w in weights}

    def train_mask(self, unit: int, target: int) -> np.ndarray:
        f = self.cfg.forecast
        years = self.panel.years
        first = target - f.train_years + 1
        ym = (years[:, None] >= first) & (
            (years[:, None] < target)
            | ((years[:, None] == target) & (np.arange(1, 13)[None, :] <= f.data_through_month))
        )
        finite = np.isfinite(self.logy[:, :, unit])
        for w in self.cfg.forecast.billing_weights:
            finite &= np.isfinite(self.hdd_bill[w][:, :, unit])
        return (ym & finite).ravel()

    def fit(self, unit: int, target: int, hac_lags: int = 0) -> Fit | None:
        mask = self.train_mask(unit, target)
        if mask.sum() < 36:
            return None
        hdd = {w: v[:, :, unit] for w, v in self.hdd_bill.items()}
        cdd = {w: v[:, :, unit] for w, v in self.cdd_bill.items()}
        return fit_unit(self.logy[:, :, unit], hdd, cdd, self.panel.years, mask, hac_lags)

    def djf_rows(self, winter: int) -> list[tuple[int, int]]:
        i = self.panel.index(winter)
        return [(i, 11), (i + 1, 0), (i + 1, 1)]

    def anomaly(
        self, fit: Fit, unit: int, winter: int, level_year: int, hdd: np.ndarray, cdd: np.ndarray
    ) -> float:
        rows = self.djf_rows(winter)
        if rows[-1][0] >= self.panel.years.size or level_year not in fit.alpha:
            return np.nan
        values = []
        for yi, mi in rows:
            expected = (
                fit.alpha[level_year]
                + fit.gamma[mi]
                + fit.beta[0] * hdd[yi, mi]
                + fit.beta[1] * cdd[yi, mi]
            )
            values.append(self.logy[yi, mi, unit] - expected)
        return 100.0 * float(np.mean(values))

    def calibration(self, fit: Fit, unit: int, target: int) -> tuple[float, float]:
        hdd = self.hdd_bill[fit.weight][:, :, unit]
        cdd = self.cdd_bill[fit.weight][:, :, unit]
        errors = []
        for v in range(target - self.cfg.forecast.train_years, target):
            if v in fit.alpha and v + 1 in fit.alpha:
                e = self.anomaly(fit, unit, v, v, hdd, cdd)
                if np.isfinite(e):
                    errors.append(e)
        errors = np.array(errors)
        if errors.size < 3:
            return 0.0, float(np.sqrt(fit.sse / max(fit.n - 1, 1)) * 100.0)
        return float(errors.mean()), float(errors.std(ddof=1))

    def normals(self, target: int) -> tuple[dict[float, np.ndarray], dict[float, np.ndarray]]:
        half = self.cfg.climate.anomaly_half_window
        known = self.panel.index(target - 1)
        hn = stats.window_normal(self.hdd_pd, half, known)
        cn = stats.window_normal(self.cdd_pd, half, known)
        return (
            {w: billing(hn, w) for w in self.cfg.forecast.billing_weights},
            {w: billing(cn, w) for w in self.cfg.forecast.billing_weights},
        )

    def member_effects(self, fit: Fit, unit: int, target: int, years: np.ndarray) -> np.ndarray:
        half = self.cfg.climate.anomaly_half_window
        known = self.panel.index(target - 1)
        ha = billing(stats.window_anomaly(self.hdd_pd[:, :, unit], half, known), fit.weight)
        ca = billing(stats.window_anomaly(self.cdd_pd[:, :, unit], half, known), fit.weight)
        out = []
        for y in years:
            rows = self.djf_rows(int(y))
            dh = np.mean([ha[yi, mi] for yi, mi in rows])
            dc = np.mean([ca[yi, mi] for yi, mi in rows])
            out.append(100.0 * (fit.beta[0] * dh + fit.beta[1] * dc))
        return np.array(out)

    def observed(self, fit: Fit, unit: int, target: int, normals: tuple[dict, dict]) -> float:
        hn, cn = normals
        return self.anomaly(
            fit, unit, target, target, hn[fit.weight][:, :, unit], cn[fit.weight][:, :, unit]
        )


def coefficients(model: SalesModel, target: int) -> pd.DataFrame:
    rows = []
    for i, unit in enumerate(model.panel.units):
        fit = model.fit(i, target, hac_lags=model.cfg.forecast.hac_lags)
        if fit is None:
            continue
        rows.append(
            {
                "unit": unit,
                "billing_weight": fit.weight,
                "heating_pct_per_hdd": 100.0 * fit.beta[0],
                "heating_se": 100.0 * fit.se[0],
                "cooling_pct_per_cdd": 100.0 * fit.beta[1],
                "cooling_se": 100.0 * fit.se[1],
                "r2": fit.r2,
                "months": fit.n,
            }
        )
    return pd.DataFrame(rows)
