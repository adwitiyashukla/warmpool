from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import duckdb
import pandas as pd

from warmpool import enso, parse, quality
from warmpool.config import Config

GOLD_FILE = "warmpool.duckdb"


def gold_path(cfg: Config) -> Path:
    return cfg.path("gold") / GOLD_FILE


def _month_columns(frame: pd.DataFrame) -> pd.DataFrame:
    frame["date"] = pd.to_datetime(dict(year=frame["year"], month=frame["month"], day=1))
    frame["days"] = frame["date"].dt.days_in_month.astype(int)
    return frame


def parse_all(cfg: Config) -> dict[str, pd.DataFrame]:
    raw = cfg.path("raw")
    roni = parse.parse_roni(raw / "enso" / "RONI.ascii.txt")
    oni = parse.parse_oni(raw / "enso" / "oni.ascii.txt")
    season = roni.merge(
        oni[["year", "season", "oni", "oni_total"]], on=["year", "season"], how="outer"
    )
    season["month"] = season["season"].map(parse.CENTER_MONTH)
    season["date"] = pd.to_datetime(dict(year=season["year"], month=season["month"], day=1))
    retail_files = sorted((raw / "retail").glob("*.xlsx"))
    climate = parse.parse_climate(raw / "climdiv", cfg.sources.climdiv_elements)
    climate = climate[climate["year"] >= cfg.climate.first_year].reset_index(drop=True)
    return {
        "enso_season": season.sort_values("date").reset_index(drop=True),
        "enso_week": parse.parse_weekly(raw / "enso" / "wksst9120.for"),
        "nino34_month": parse.parse_psl(raw / "enso" / "nino34.long.anom.data"),
        "climate_month": _month_columns(climate),
        "retail_month": _month_columns(parse.parse_retail(retail_files)),
        "price_day": parse.parse_wholesale(raw / "wholesale"),
        "gas_day": parse.parse_henry_hub(raw / "gas" / "DHHNGSP.csv"),
    }


def state_table(cfg: Config, retail: pd.DataFrame) -> pd.DataFrame:
    first, last = cfg.climate.weight_years
    res = retail[(retail["sector"] == "residential") & retail["year"].between(first, last)]
    share = res.groupby("state")["sales_mwh"].sum()
    rows = []
    for region, states in cfg.regions.items():
        for st in states:
            rows.append(
                {
                    "state": st,
                    "region": region,
                    "climate_state": cfg.climate.dc_climate_state if st == "DC" else st,
                }
            )
    table = pd.DataFrame(rows)
    table["weight"] = table["state"].map(share).fillna(0.0)
    table["weight"] = table["weight"] / table["weight"].sum()
    return table.sort_values("state").reset_index(drop=True)


def units(cfg: Config, states: pd.DataFrame) -> pd.DataFrame:
    rows = [{"unit": st, "kind": "state", "state": st} for st in states["state"]]
    for region, members in cfg.regions.items():
        rows += [{"unit": region, "kind": "region", "state": st} for st in members]
    rows += [{"unit": "US", "kind": "nation", "state": st} for st in states["state"]]
    return pd.DataFrame(rows)


def unit_climate(
    climate: pd.DataFrame, states: pd.DataFrame, membership: pd.DataFrame
) -> pd.DataFrame:
    linked = membership.merge(states[["state", "climate_state", "weight"]], on="state")
    linked.loc[linked["kind"] == "state", "weight"] = 1.0
    merged = linked.merge(
        climate, left_on="climate_state", right_on="state", suffixes=("", "_clim")
    )
    cols = ["tmp", "hdd", "cdd", "pcpn"]
    for col in cols:
        merged[col] = merged[col] * merged["weight"]
    grouped = merged.groupby(["unit", "kind", "year", "month"], as_index=False)
    out = grouped[cols + ["weight"]].sum()
    for col in cols:
        out[col] = out[col] / out["weight"]
    return _month_columns(out.drop(columns="weight"))


def unit_sales(retail: pd.DataFrame, membership: pd.DataFrame) -> pd.DataFrame:
    merged = membership.merge(retail, on="state")
    cols = ["sales_mwh", "revenue_k", "customers"]
    out = merged.groupby(["unit", "kind", "sector", "year", "month"], as_index=False)[cols].sum(
        min_count=1
    )
    out["price_cents"] = out["revenue_k"] * 100.0 / out["sales_mwh"]
    return _month_columns(out)


def price_month(price_day: pd.DataFrame, gas_day: pd.DataFrame, min_days: int) -> pd.DataFrame:
    day = price_day.assign(
        year=price_day["delivery_start"].dt.year, month=price_day["delivery_start"].dt.month
    )
    power = day.groupby(["hub", "year", "month"], as_index=False).agg(
        price=("price", "mean"), days=("price", "size")
    )
    gas = gas_day.assign(year=gas_day["date"].dt.year, month=gas_day["date"].dt.month)
    gas = gas.groupby(["year", "month"], as_index=False)["henry_hub"].mean()
    out = power[power["days"] >= min_days].merge(gas, on=["year", "month"], how="left")
    out["heat_rate"] = out["price"] / out["henry_hub"]
    out["date"] = pd.to_datetime(dict(year=out["year"], month=out["month"], day=1))
    return out.rename(columns={"days": "trading_days"})


def gas_month(gas_day: pd.DataFrame) -> pd.DataFrame:
    gas = gas_day.assign(year=gas_day["date"].dt.year, month=gas_day["date"].dt.month)
    out = gas.groupby(["year", "month"], as_index=False).agg(
        henry_hub=("henry_hub", "mean"), trading_days=("henry_hub", "size")
    )
    out["date"] = pd.to_datetime(dict(year=out["year"], month=out["month"], day=1))
    return out


def manifest_table(cfg: Config) -> pd.DataFrame:
    path = cfg.path("raw") / "manifest.json"
    if not path.exists():
        return pd.DataFrame(columns=["relpath", "url", "bytes", "sha256", "fetched_utc"])
    data = json.loads(path.read_text())
    rows = [
        {"relpath": k, **{f: v.get(f) for f in ("url", "bytes", "sha256", "fetched_utc")}}
        for k, v in data.items()
    ]
    return pd.DataFrame(rows)


def tables(cfg: Config, frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    e = cfg.enso
    harmonized, cal = enso.harmonize(
        frames["enso_season"].dropna(subset=["roni"]),
        frames["nino34_month"],
        e.base_window_years,
        e.base_step_years,
        e.base_anchor_year,
    )
    states = state_table(cfg, frames["retail_month"])
    membership = units(cfg, states)
    out = dict(frames)
    out["price_day"] = frames["price_day"].rename(columns={"delivery_start": "date"})
    out["enso_month"] = harmonized
    out["enso_calibration"] = pd.DataFrame([cal.__dict__])
    out["state"] = states
    out["unit_member"] = membership
    out["unit_climate"] = unit_climate(frames["climate_month"], states, membership)
    out["unit_sales"] = unit_sales(frames["retail_month"], membership)
    out["price_month"] = price_month(frames["price_day"], frames["gas_day"], cfg.prices.min_days)
    out["gas_month"] = gas_month(frames["gas_day"])
    out["source_file"] = manifest_table(cfg)
    return out


def write_tables(con: duckdb.DuckDBPyConnection, data: dict[str, pd.DataFrame]) -> None:
    for name, frame in data.items():
        con.register("incoming", frame)
        con.execute(f"create or replace table {name} as select * from incoming")
        con.unregister("incoming")


def build(cfg: Config, log: Callable[[str], None] = print) -> Path:
    frames = parse_all(cfg)
    silver = cfg.path("silver")
    silver.mkdir(parents=True, exist_ok=True)
    for name, frame in frames.items():
        frame.to_parquet(silver / f"{name}.parquet", index=False)
    log(f"silver: {len(frames)} parquet tables in {silver}")
    data = tables(cfg, frames)
    checks = quality.run(cfg, data)
    data["quality"] = checks
    target = gold_path(cfg)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.with_name(target.name + ".building")
    staging.unlink(missing_ok=True)
    with duckdb.connect(str(staging)) as con:
        write_tables(con, data)
    staging.replace(target)
    failed = checks[(~checks["passed"]) & (checks["severity"] == "error")]
    warned = checks[(~checks["passed"]) & (checks["severity"] == "warn")]
    log(
        f"gold: {len(data)} tables in {target}; quality {int(checks['passed'].sum())}"
        f"/{len(checks)} passed, {len(warned)} warnings"
    )
    if len(failed):
        raise quality.QualityError("; ".join(f"{r.name}: {r.detail}" for r in failed.itertuples()))
    return target


def read(cfg: Config, query: str, params: list | None = None) -> pd.DataFrame:
    with duckdb.connect(str(gold_path(cfg)), read_only=True) as con:
        return con.execute(query, params or []).df()


def read_table(cfg: Config, name: str) -> pd.DataFrame:
    return read(cfg, f"select * from {name}")


def save(cfg: Config, data: dict[str, pd.DataFrame]) -> None:
    with duckdb.connect(str(gold_path(cfg))) as con:
        write_tables(con, data)
