from __future__ import annotations

import numpy as np
import pandas as pd

from warmpool import climate, stats
from warmpool.config import Config


def _slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xc = x - x.mean()
    return (xc @ (y - y.mean(axis=0))) / (xc @ xc)


def mine(cfg: Config, panel: climate.Panel, series: pd.Series) -> pd.DataFrame:
    t = cfg.teleconnect
    rng = np.random.default_rng(t.seed)
    anoms = {v: climate.anomalies(panel, v, cfg.climate.anomaly_half_window) for v in t.variables}
    rows = []
    for month in range(1, 13):
        for lead in range(t.max_lead + 1):
            x = climate.enso_at(series, panel.years, month - lead)
            block = np.concatenate([anoms[v][:, month - 1, :] for v in t.variables], axis=1)
            ok = np.isfinite(x) & np.isfinite(block).all(axis=1)
            r, p = stats.spearman_field(x[ok], block[ok], t.surrogates, rng)
            slope = _slopes(x[ok], block[ok])
            for j, (var, unit) in enumerate((v, u) for v in t.variables for u in panel.units):
                rows.append((var, unit, month, lead, int(ok.sum()), r[j], slope[j], p[j]))
    out = pd.DataFrame(rows, columns=["variable", "state", "month", "lead", "n", "r", "slope", "p"])
    out["q"] = stats.qvalues(out["p"].to_numpy(), t.fdr_method)
    out["significant"] = out["q"] <= t.fdr_q
    return out


def summary(found: pd.DataFrame, q: float) -> pd.DataFrame:
    rows = []
    for (var, lead), part in found.groupby(["variable", "lead"]):
        hits = int(part["significant"].sum())
        rows.append(
            {
                "variable": var,
                "lead": lead,
                "tests": len(part),
                "discoveries": hits,
                "expected_false": round(hits * q, 1),
                "uncorrected_p05": int((part["p"] <= 0.05).sum()),
            }
        )
    return pd.DataFrame(rows)


def season_data(cfg: Config, panel: climate.Panel, series: pd.Series) -> dict:
    out = {}
    for var in cfg.teleconnect.variables:
        anom = climate.anomalies(panel, var, cfg.climate.anomaly_half_window)
        for season, months in climate.SEASON_MONTHS.items():
            years, values = climate.season_mean(anom, panel.years, season)
            x = climate.enso_at(series, years, months[1])
            ok = np.isfinite(x) & np.isfinite(values).all(axis=1)
            out[(var, season)] = (years[ok], x[ok], values[ok])
    return out


def fingerprint(
    cfg: Config, panel: climate.Panel, series: pd.Series, field: set[str]
) -> pd.DataFrame:
    t = cfg.teleconnect
    e = cfg.enso
    rng = np.random.default_rng(t.seed + 1)
    rows = []
    for (var, season), (_, xs, ys) in season_data(cfg, panel, series).items():
        r, p = stats.spearman_field(xs, ys, t.surrogates, rng)
        slope = _slopes(xs, ys)
        groups = {
            "el_nino": xs >= e.threshold,
            "la_nina": xs <= -e.threshold,
            "super": xs >= e.strength_edges[-1],
        }
        for j, unit in enumerate(panel.units):
            row = {
                "variable": var,
                "unit": unit,
                "season": season,
                "n": int(xs.size),
                "r": r[j],
                "p": p[j],
                "slope": slope[j],
            }
            for name, mask in groups.items():
                vals = ys[mask, j]
                lo, hi = stats.bootstrap_ci(vals, t.bootstrap, rng)
                row.update(
                    {
                        f"{name}_n": int(mask.sum()),
                        f"{name}_mean": float(vals.mean()) if vals.size else np.nan,
                        f"{name}_lo": lo,
                        f"{name}_hi": hi,
                    }
                )
            rows.append(row)
    out = pd.DataFrame(rows)
    out["q"] = np.nan
    for in_field in (True, False):
        mask = out["unit"].isin(field) == in_field
        if mask.any():
            out.loc[mask, "q"] = stats.qvalues(out.loc[mask, "p"].to_numpy(), t.fdr_method)
    out["significant"] = out["q"] <= t.fdr_q
    return out
