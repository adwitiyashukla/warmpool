from __future__ import annotations

import numpy as np
import pandas as pd

from warmpool import climate, stats
from warmpool.config import Config
from warmpool.rules import djf_percent


def _season(frame: pd.DataFrame, value: str, months: tuple[int, ...], winter: bool) -> pd.Series:
    part = frame[frame["month"].isin(months)].copy()
    part["season"] = np.where(winter & (part["month"] < 7), part["year"] - 1, part["year"])
    grouped = part.groupby("season")[value]
    return grouped.mean()[grouped.size() == len(months)]


def permutation_r(x: np.ndarray, y: np.ndarray, reps: int, rng: np.random.Generator) -> float:
    r = np.corrcoef(x, y)[0, 1]
    null = np.array([np.corrcoef(x, rng.permutation(y))[0, 1] for _ in range(reps)])
    return float((1 + np.sum(np.abs(null) >= abs(r) - 1e-12)) / (reps + 1))


def premiums(
    cfg: Config,
    price_month: pd.DataFrame,
    gas_month: pd.DataFrame,
    panel: climate.Panel,
    winters: pd.DataFrame,
) -> pd.DataFrame:
    pr = cfg.prices
    years, hdd = djf_percent(panel, "hdd", cfg.climate.anomaly_half_window)
    hdd_by = {u: pd.Series(hdd[:, i], index=years) for i, u in enumerate(panel.units)}
    gas = gas_month[gas_month["trading_days"] >= pr.gas_min_days][["year", "month", "henry_hub"]]
    series = {("HENRYHUB", "US"): gas.rename(columns={"henry_hub": "value"})}
    for hub, region in pr.hub_regions.items():
        part = price_month[price_month["hub"] == hub][["year", "month", "price"]]
        series[(hub, region)] = part.rename(columns={"price": "value"})
    rows = []
    info = winters.set_index("winter")
    for (name, region), frame in series.items():
        fall = _season(frame, "value", pr.fall_months, False)
        win = _season(frame, "value", pr.winter_months, True)
        both = pd.concat([fall.rename("fall"), win.rename("winter")], axis=1).dropna()
        for season, row in both.iterrows():
            if season not in info.index:
                continue
            rows.append(
                {
                    "market": name,
                    "region": region,
                    "winter": int(season),
                    "label": info.loc[season, "label"],
                    "fall": row["fall"],
                    "winter_price": row["winter"],
                    "premium": 100.0 * np.log(row["winter"] / row["fall"]),
                    "hdd_pct": float(hdd_by[region].get(season, np.nan)),
                    "predictor": info.loc[season, "predictor"],
                    "djf": info.loc[season, "djf"],
                    "phase": info.loc[season, "phase"],
                    "category": info.loc[season, "category"],
                }
            )
    return pd.DataFrame(rows)


def premium_summary(cfg: Config, table: pd.DataFrame) -> pd.DataFrame:
    pr = cfg.prices
    rng = np.random.default_rng(pr.seed)
    rows = []
    for market, part in table.groupby("market"):
        part = part.dropna(subset=["premium", "hdd_pct"])
        if len(part) < pr.min_winters:
            continue
        slope, intercept = np.polyfit(part["hdd_pct"], part["premium"], 1)
        resid = part["premium"] - (intercept + slope * part["hdd_pct"])
        row = {
            "market": market,
            "region": part["region"].iloc[0],
            "winters": len(part),
            "first": int(part["winter"].min()),
            "last": int(part["winter"].max()),
            "slope_per_hdd_pct": slope,
            "intercept": intercept,
            "resid_sd": float(resid.std(ddof=2)),
            "r": float(np.corrcoef(part["hdd_pct"], part["premium"])[0, 1]),
            "p": permutation_r(
                part["hdd_pct"].to_numpy(), part["premium"].to_numpy(), pr.permutations, rng
            ),
        }
        for phase in ("el_nino", "neutral", "la_nina"):
            vals = part.loc[part["phase"] == phase, "premium"].to_numpy()
            lo, hi = stats.bootstrap_ci(vals, 2000, rng)
            row.update(
                {
                    f"{phase}_n": int(vals.size),
                    f"{phase}_mean": float(vals.mean()) if vals.size else np.nan,
                    f"{phase}_lo": lo,
                    f"{phase}_hi": hi,
                }
            )
        strong = part[part["category"].isin(["strong_el_nino", "very_strong_el_nino"])]
        row["strong_el_nino_mean"] = float(strong["premium"].mean()) if len(strong) else np.nan
        row["strong_el_nino_winters"] = ", ".join(strong["label"])
        rows.append(row)
    return pd.DataFrame(rows)


def basin_precip(cfg: Config, panel: climate.Panel) -> pd.Series:
    pr = cfg.prices
    sub = panel.subset(list(pr.hydro_states))
    anom = climate.anomalies(sub, "pcpn", cfg.climate.anomaly_half_window)
    normal = sub.values["pcpn"] - anom
    a = climate.winter_months(anom, pr.hydro_precip_months).sum(axis=1)
    n = climate.winter_months(normal, pr.hydro_precip_months).sum(axis=1)
    return pd.Series((100.0 * a / n).mean(axis=1), index=sub.years[:-1])


def hub_spread(cfg: Config, price_month: pd.DataFrame) -> pd.DataFrame:
    pr = cfg.prices
    hub = price_month[price_month["hub"] == pr.hydro_hub].set_index(["year", "month"])["price"]
    ref = price_month[price_month["hub"] == pr.hydro_reference].set_index(["year", "month"])[
        "price"
    ]
    spread = (100.0 * np.log(hub / ref)).dropna().rename("spread").reset_index()
    spread["anomaly"] = spread["spread"] - spread.groupby("month")["spread"].transform("mean")
    return spread


def hydro(
    cfg: Config, price_month: pd.DataFrame, panel: climate.Panel, winters: pd.DataFrame
) -> pd.DataFrame:
    pr = cfg.prices
    rng = np.random.default_rng(pr.seed + 1)
    precip = basin_precip(cfg, panel)
    spread = hub_spread(cfg, price_month)
    info = winters.set_index("winter")
    rows = []
    for lag in range(1, pr.hydro_max_lag + 1):
        part = spread[spread["month"] == lag].copy()
        part["winter"] = part["year"] - 1
        part["precip"] = part["winter"].map(precip)
        part["djf"] = part["winter"].map(info["djf"])
        part = part.dropna(subset=["precip", "djf", "anomaly"])
        for driver in ("precip", "djf"):
            x, y = part[driver].to_numpy(), part["anomaly"].to_numpy()
            rows.append(
                {
                    "driver": driver,
                    "month": lag,
                    "winters": len(part),
                    "r": float(np.corrcoef(x, y)[0, 1]),
                    "slope": float(np.polyfit(x, y, 1)[0]),
                    "p": permutation_r(x, y, pr.permutations, rng),
                }
            )
    out = pd.DataFrame(rows)
    out["q"] = stats.bh_qvalues(out["p"].to_numpy())
    return out


def hydro_seasons(
    cfg: Config,
    price_month: pd.DataFrame,
    panel: climate.Panel,
    winters: pd.DataFrame,
    months: tuple[int, ...] = (4, 5, 6, 7),
) -> pd.DataFrame:
    precip = basin_precip(cfg, panel)
    spread = hub_spread(cfg, price_month)
    spring = spread[spread["month"].isin(months)].groupby("year")["spread"].agg(["mean", "size"])
    spring = spring[spring["size"] == len(months)]["mean"]
    info = winters.set_index("winter")
    frame = pd.DataFrame({"winter": spring.index - 1, "spring_spread": spring.to_numpy()})
    frame["label"] = frame["winter"].map(info["label"])
    frame["precip_pct"] = frame["winter"].map(precip)
    frame["djf"] = frame["winter"].map(info["djf"])
    frame["phase"] = frame["winter"].map(info["phase"])
    return frame.dropna(subset=["precip_pct"]).reset_index(drop=True)
