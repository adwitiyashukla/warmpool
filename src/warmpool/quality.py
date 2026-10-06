from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from warmpool.config import RETAIL_STATES, STATES, Config


class QualityError(RuntimeError):
    pass


def _check(name: str, passed: bool, detail: str, severity: str = "error") -> dict:
    return {"name": name, "severity": severity, "passed": bool(passed), "detail": detail}


def enso_checks(data: dict[str, pd.DataFrame]) -> list[dict]:
    season = data["enso_season"]
    roni = season.dropna(subset=["roni"]).sort_values("date")
    expected = pd.date_range(roni["date"].min(), roni["date"].max(), freq="MS")
    gaps = len(expected) - len(roni)
    both = season.dropna(subset=["roni", "oni"])
    strong = both[(both["roni"].abs() > 0.5) & (both["oni"].abs() > 0.5)]
    agree = float((np.sign(strong["roni"]) == np.sign(strong["oni"])).mean())
    cal = data["enso_calibration"].iloc[0]
    weeks = data["enso_week"]["week"].sort_values().diff().dt.days.dropna()
    return [
        _check("roni_continuous", gaps == 0, f"{len(roni)} seasons, {gaps} missing"),
        _check("roni_oni_sign_agreement", agree >= 0.95, f"{agree:.3f} of strong seasons agree"),
        _check(
            "hadisst_calibration_r",
            cal["r"] >= 0.9,
            f"r={cal['r']:.3f}, rmse={cal['rmse']:.3f} over {cal['n']} months",
        ),
        _check("weekly_spacing", bool((weeks == 7).all()), f"{len(weeks) + 1} weeks"),
    ]


def climate_checks(data: dict[str, pd.DataFrame]) -> list[dict]:
    clim = data["climate_month"]
    counts = clim.groupby("year").size()
    full = counts[counts.index < counts.index.max()]
    bad_years = full[full != len(STATES) * 12]
    ranges = (
        clim["tmp"].between(-30, 110)
        & (clim["hdd"] >= 0)
        & (clim["cdd"] >= 0)
        & (clim["pcpn"] >= 0)
    )
    jan = clim[clim["month"] == 1]
    corr = jan.groupby("state")[["tmp", "hdd"]].corr().unstack()[("tmp", "hdd")]
    nulls = int(clim[["tmp", "hdd", "cdd", "pcpn"]].isna().sum().sum())
    return [
        _check(
            "climate_complete_years",
            bad_years.empty and nulls == 0,
            f"{len(full)} full years x 48 states, {len(bad_years)} incomplete, {nulls} nulls",
        ),
        _check("climate_ranges", bool(ranges.all()), f"{int((~ranges).sum())} rows out of range"),
        _check(
            "hdd_tracks_temperature",
            bool((corr < -0.8).all()),
            f"weakest January tmp/hdd correlation {corr.max():.3f}",
        ),
    ]


def retail_checks(data: dict[str, pd.DataFrame]) -> list[dict]:
    retail = data["retail_month"]
    res = retail[retail["sector"] == "residential"]
    months = res.groupby(["year", "month"]).size()
    lower48 = set(RETAIL_STATES)
    present = set(res["state"])
    derived = res["revenue_k"] * 100.0 / res["sales_mwh"]
    close = ((derived - res["price_cents"]).abs() < 0.05).mean()
    pivot = retail.pivot_table(
        index=["state", "year", "month"], columns="sector", values="sales_mwh"
    )
    order = (pivot["total"] + 1e-6 >= pivot["residential"]).mean()
    return [
        _check("retail_states", lower48 <= present, f"{len(present)} states incl. DC, AK, HI"),
        _check(
            "retail_months_complete",
            bool((months == months.max()).all()),
            f"{len(months)} months with {months.max()} states each",
        ),
        _check("retail_price_consistent", close >= 0.999, f"{close:.4f} of rows match"),
        _check("retail_total_ge_residential", order >= 0.999, f"{order:.4f} of rows"),
        _check(
            "retail_positive_sales",
            bool((res["sales_mwh"] > 0).all()),
            f"{int((res['sales_mwh'] <= 0).sum())} non-positive residential rows",
        ),
    ]


def price_checks(data: dict[str, pd.DataFrame]) -> list[dict]:
    day = data["price_day"]
    inside = ((day["price"] >= day["low"] - 0.01) & (day["price"] <= day["high"] + 0.01)).mean()
    sane = day["price"].between(-100, 5000).mean()
    flipped = int((day["delivery_end"] < day["date"]).sum())
    gas = data["gas_day"]["henry_hub"]
    return [
        _check("wholesale_within_high_low", inside >= 0.999, f"{inside:.4f} of trading days"),
        _check("wholesale_price_range", sane == 1.0, f"{sane:.4f} within [-100, 5000] $/MWh"),
        _check(
            "wholesale_delivery_order",
            flipped == 0,
            f"{flipped} rows end before they start",
            severity="warn",
        ),
        _check(
            "henry_hub_range",
            bool(gas.between(0, 50).all()),
            f"min {gas.min():.2f}, max {gas.max():.2f} $/MMBtu",
        ),
    ]


def join_checks(cfg: Config, data: dict[str, pd.DataFrame]) -> list[dict]:
    states = data["state"]
    clim_states = set(data["climate_month"]["state"])
    unmatched = sorted(set(states["climate_state"]) - clim_states)
    weight = states["weight"].sum()
    covered = sorted(states["state"])
    return [
        _check("state_climate_join", not unmatched, f"unmatched {unmatched}"),
        _check(
            "region_partition",
            covered == sorted(RETAIL_STATES),
            f"{len(covered)} states in {len(cfg.regions)} regions",
        ),
        _check("weights_sum_to_one", abs(weight - 1) < 1e-9, f"sum {weight:.6f}"),
    ]


def manifest_checks(cfg: Config, data: dict[str, pd.DataFrame]) -> list[dict]:
    manifest = data["source_file"]
    if manifest.empty:
        return [
            _check(
                "raw_files_match_manifest",
                True,
                "no manifest, files placed by hand",
                severity="warn",
            )
        ]
    raw = cfg.path("raw")
    bad = []
    for row in manifest.itertuples():
        path = raw / row.relpath
        if not path.exists() or hashlib.sha256(path.read_bytes()).hexdigest() != row.sha256:
            bad.append(row.relpath)
    return [
        _check(
            "raw_files_match_manifest",
            not bad,
            f"{len(manifest) - len(bad)}/{len(manifest)} sha256 match",
            severity="warn",
        )
    ]


def run(cfg: Config, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    checks = (
        enso_checks(data)
        + climate_checks(data)
        + retail_checks(data)
        + price_checks(data)
        + join_checks(cfg, data)
        + manifest_checks(cfg, data)
    )
    return pd.DataFrame(checks)
