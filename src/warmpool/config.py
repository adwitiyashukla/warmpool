from __future__ import annotations

import tomllib
import urllib.parse
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    pass


@dataclass(frozen=True)
class Paths:
    raw: str
    silver: str
    gold: str


@dataclass(frozen=True)
class Download:
    timeout_s: float
    retries: int
    backoff_s: float
    workers: int
    user_agent: str


@dataclass(frozen=True)
class Sources:
    roni: str
    oni: str
    weekly: str
    nino34_long: str
    climdiv_base: str
    climdiv_elements: tuple[str, ...]
    retail: str
    retail_history: str
    wholesale_index: str
    wholesale_pattern: str
    henry_hub: str


@dataclass(frozen=True)
class Enso:
    threshold: float
    min_months: int
    strength_edges: tuple[float, ...]
    base_window_years: int
    base_step_years: int
    base_anchor_year: int
    predictor_month: int


@dataclass(frozen=True)
class Climate:
    first_year: int
    anomaly_half_window: int
    weight_years: tuple[int, ...]
    dc_climate_state: str


@dataclass(frozen=True)
class Teleconnect:
    variables: tuple[str, ...]
    max_lead: int
    surrogates: int
    fdr_method: str
    fdr_q: float
    bootstrap: int
    seed: int


@dataclass(frozen=True)
class Cluster:
    variables: tuple[str, ...]
    k_min: int
    k_max: int
    bootstrap: int
    seed: int


@dataclass(frozen=True)
class Analog:
    start_month: int
    band: int
    top_k: int
    family_k_max: int


@dataclass(frozen=True)
class Rules:
    min_support: float
    max_len: int
    min_confidence: float
    min_lift: float
    precip_regions: tuple[str, ...]
    permutations: int
    fdr_q: float
    seed: int


@dataclass(frozen=True)
class Forecast:
    methods: tuple[str, ...]
    bandwidth: float
    min_ess: float
    gate: float
    first_climate_winter: int
    first_sales_winter: int
    sector: str
    train_years: int
    data_through_month: int
    billing_weights: tuple[float, ...]
    hac_lags: int
    interval: float
    bootstrap: int
    seed: int


@dataclass(frozen=True)
class Outlook:
    winter: int
    scenarios: tuple[float, ...]


@dataclass(frozen=True)
class Prices:
    min_days: int
    gas_min_days: int
    min_winters: int
    hub_regions: dict
    fall_months: tuple[int, ...]
    winter_months: tuple[int, ...]
    hydro_hub: str
    hydro_reference: str
    hydro_states: tuple[str, ...]
    hydro_precip_months: tuple[int, ...]
    hydro_max_lag: int
    permutations: int
    seed: int


@dataclass(frozen=True)
class Config:
    root: Path
    paths: Paths
    download: Download
    sources: Sources
    enso: Enso
    climate: Climate
    regions: dict
    teleconnect: Teleconnect
    cluster: Cluster
    analog: Analog
    rules: Rules
    forecast: Forecast
    outlook: Outlook
    prices: Prices

    def path(self, name: str) -> Path:
        return self.root / getattr(self.paths, name)


SECTIONS = {
    "paths": Paths,
    "download": Download,
    "sources": Sources,
    "enso": Enso,
    "climate": Climate,
    "regions": "dict[str, tuple[str, ...]]",
    "teleconnect": Teleconnect,
    "cluster": Cluster,
    "analog": Analog,
    "rules": Rules,
    "forecast": Forecast,
    "outlook": Outlook,
    "prices": Prices,
}

METHODS = ("climatology", "enso", "analog", "linear", "gated")
SECTORS = ("residential", "commercial", "industrial", "total")

VARIABLES = {"tmp", "hdd", "cdd", "pcpn"}

STATES = (
    "AL",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
)
RETAIL_STATES = STATES + ("DC",)

CLIMDIV_ELEMENTS = {"tmpc", "hddc", "cddc", "pcpn"}


def _coerce(value: Any, kind: str, where: str) -> Any:
    if kind == "str":
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{where} must be a non-empty string")
        return value
    if kind == "bool":
        if not isinstance(value, bool):
            raise ConfigError(f"{where} must be true or false")
        return value
    if kind == "int":
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where} must be an integer")
        return value
    if kind == "float":
        if isinstance(value, bool) or not isinstance(value, int | float):
            raise ConfigError(f"{where} must be a number")
        return float(value)
    if kind.startswith("tuple["):
        inner = kind[len("tuple[") : -len(", ...]")]
        if not isinstance(value, list) or not value:
            raise ConfigError(f"{where} must be a non-empty list")
        return tuple(_coerce(item, inner, f"{where}[{i}]") for i, item in enumerate(value))
    if kind == "dict":
        if not isinstance(value, dict) or not value:
            raise ConfigError(f"{where} must be a non-empty table")
        return {key: _coerce(item, "str", f"{where}.{key}") for key, item in value.items()}
    if kind == "dict[str, tuple[str, ...]]":
        if not isinstance(value, dict) or not value:
            raise ConfigError(f"{where} must be a non-empty table")
        return {
            key: _coerce(item, "tuple[str, ...]", f"{where}.{key}") for key, item in value.items()
        }
    raise ConfigError(f"{where} has an unsupported type {kind}")


def _section(cls: Any, raw: Any, name: str) -> Any:
    if isinstance(cls, str):
        return _coerce(raw, cls, name)
    if not isinstance(raw, dict):
        raise ConfigError(f"[{name}] must be a table")
    expected = {f.name for f in fields(cls)}
    unknown = sorted(set(raw) - expected)
    missing = sorted(expected - set(raw))
    if unknown:
        raise ConfigError(f"[{name}] has unknown keys {unknown}")
    if missing:
        raise ConfigError(f"[{name}] is missing keys {missing}")
    return cls(**{f.name: _coerce(raw[f.name], f.type, f"{name}.{f.name}") for f in fields(cls)})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ConfigError(message)


def _good_url(url: str) -> bool:
    parts = urllib.parse.urlsplit(url)
    if parts.scheme == "https" and parts.netloc:
        return True
    return parts.scheme == "http" and parts.hostname in {"127.0.0.1", "localhost"}


def _validate(cfg: Config) -> None:
    d = cfg.download
    _require(0 < d.timeout_s <= 600, "download.timeout_s must be in (0, 600]")
    _require(0 <= d.retries <= 10, "download.retries must be in [0, 10]")
    _require(0 <= d.backoff_s <= 60, "download.backoff_s must be in [0, 60]")
    _require(1 <= d.workers <= 16, "download.workers must be in [1, 16]")
    s = cfg.sources
    for key in (
        "roni",
        "oni",
        "weekly",
        "nino34_long",
        "climdiv_base",
        "retail",
        "retail_history",
        "wholesale_index",
        "henry_hub",
    ):
        _require(_good_url(getattr(s, key)), f"sources.{key} must be an https url")
    _require(s.climdiv_base.endswith("/"), "sources.climdiv_base must end with /")
    bad = sorted(set(s.climdiv_elements) - CLIMDIV_ELEMENTS)
    _require(not bad, f"sources.climdiv_elements has unknown elements {bad}")
    _require(
        len(set(s.climdiv_elements)) == len(s.climdiv_elements),
        "sources.climdiv_elements has duplicates",
    )
    names = [cfg.paths.raw, cfg.paths.silver, cfg.paths.gold]
    _require(len(set(names)) == len(names), "paths must be distinct")
    e = cfg.enso
    _require(0 < e.threshold < 3, "enso.threshold must be in (0, 3)")
    _require(3 <= e.min_months <= 12, "enso.min_months must be in [3, 12]")
    edges = e.strength_edges
    _require(
        len(edges) == 4 and all(a < b for a, b in zip(edges, edges[1:], strict=False)),
        "enso.strength_edges must be 4 increasing numbers",
    )
    _require(edges[0] == e.threshold, "enso.strength_edges must start at enso.threshold")
    _require(10 <= e.base_window_years <= 50, "enso.base_window_years must be in [10, 50]")
    _require(1 <= e.base_step_years <= 10, "enso.base_step_years must be in [1, 10]")
    _require(1850 <= e.base_anchor_year <= 2100, "enso.base_anchor_year must be in [1850, 2100]")
    _require(1 <= e.predictor_month <= 12, "enso.predictor_month must be a month number")
    c = cfg.climate
    _require(1895 <= c.first_year <= 1990, "climate.first_year must be in [1895, 1990]")
    _require(5 <= c.anomaly_half_window <= 30, "climate.anomaly_half_window must be in [5, 30]")
    _require(
        len(c.weight_years) == 2 and 1990 <= c.weight_years[0] <= c.weight_years[1],
        "climate.weight_years must be [first, last] with first >= 1990",
    )
    _require(c.dc_climate_state in STATES, "climate.dc_climate_state must be a state code")
    seen: dict[str, str] = {}
    for region, states in cfg.regions.items():
        _require(
            region.isupper() and region.replace("_", "").isalnum(),
            f"region name {region} must be upper case letters, digits or _",
        )
        for st in states:
            _require(st in RETAIL_STATES, f"regions.{region} has unknown state {st}")
            _require(st not in seen, f"state {st} is in both {seen.get(st)} and {region}")
            seen[st] = region
    missing_states = sorted(set(RETAIL_STATES) - set(seen))
    _require(not missing_states, f"regions do not cover {missing_states}")
    t = cfg.teleconnect
    _require(set(t.variables) <= VARIABLES, f"teleconnect.variables must be in {sorted(VARIABLES)}")
    _require(0 <= t.max_lead <= 12, "teleconnect.max_lead must be in [0, 12]")
    _require(99 <= t.surrogates <= 99999, "teleconnect.surrogates must be in [99, 99999]")
    _require(t.fdr_method in ("bh", "by"), "teleconnect.fdr_method must be bh or by")
    _require(0 < t.fdr_q < 0.5, "teleconnect.fdr_q must be in (0, 0.5)")
    _require(100 <= t.bootstrap <= 100000, "teleconnect.bootstrap must be in [100, 100000]")
    k = cfg.cluster
    _require(set(k.variables) <= VARIABLES, f"cluster.variables must be in {sorted(VARIABLES)}")
    _require(2 <= k.k_min <= k.k_max <= 20, "cluster needs 2 <= k_min <= k_max <= 20")
    _require(10 <= k.bootstrap <= 10000, "cluster.bootstrap must be in [10, 10000]")
    a = cfg.analog
    _require(
        1 <= a.start_month < cfg.enso.predictor_month,
        "analog.start_month must be before enso.predictor_month",
    )
    _require(0 <= a.band <= 6, "analog.band must be in [0, 6]")
    _require(1 <= a.top_k <= 30, "analog.top_k must be in [1, 30]")
    _require(2 <= a.family_k_max <= 12, "analog.family_k_max must be in [2, 12]")
    r = cfg.rules
    _require(0 < r.min_support < 0.5, "rules.min_support must be in (0, 0.5)")
    _require(2 <= r.max_len <= 6, "rules.max_len must be in [2, 6]")
    _require(0 < r.min_confidence <= 1, "rules.min_confidence must be in (0, 1]")
    _require(r.min_lift >= 1, "rules.min_lift must be at least 1")
    unknown_regions = sorted(set(r.precip_regions) - set(cfg.regions) - {"US"})
    _require(not unknown_regions, f"rules.precip_regions has unknown regions {unknown_regions}")
    _require(99 <= r.permutations <= 99999, "rules.permutations must be in [99, 99999]")
    _require(0 < r.fdr_q < 0.5, "rules.fdr_q must be in (0, 0.5)")
    f = cfg.forecast
    _require(f.methods[0] == "climatology", "forecast.methods must start with climatology")
    _require(set(f.methods) <= set(METHODS), f"forecast.methods must be in {list(METHODS)}")
    _require(0.05 <= f.bandwidth <= 3, "forecast.bandwidth must be in [0.05, 3]")
    _require(2 <= f.min_ess <= 100, "forecast.min_ess must be in [2, 100]")
    _require(0 < f.gate <= 3, "forecast.gate must be in (0, 3]")
    _require(
        c.first_year + 30 <= f.first_climate_winter <= 2020,
        "forecast.first_climate_winter must leave 30 years of history",
    )
    _require(
        1991 <= f.first_sales_winter <= 2020, "forecast.first_sales_winter must be in [1991, 2020]"
    )
    _require(f.sector in SECTORS, f"forecast.sector must be one of {list(SECTORS)}")
    _require(3 <= f.train_years <= 30, "forecast.train_years must be in [3, 30]")
    _require(1 <= f.data_through_month <= 11, "forecast.data_through_month must be in [1, 11]")
    _require(
        all(0 <= w < 1 for w in f.billing_weights), "forecast.billing_weights must be in [0, 1)"
    )
    _require(0 <= f.hac_lags <= 24, "forecast.hac_lags must be in [0, 24]")
    _require(0.5 <= f.interval < 1, "forecast.interval must be in [0.5, 1)")
    _require(100 <= f.bootstrap <= 100000, "forecast.bootstrap must be in [100, 100000]")
    o = cfg.outlook
    _require(f.first_sales_winter < o.winter <= 2100, "outlook.winter must be after the backtest")
    _require(all(-4 <= v <= 4 for v in o.scenarios), "outlook.scenarios must be in [-4, 4]")
    pr = cfg.prices
    _require(1 <= pr.min_days <= 23, "prices.min_days must be in [1, 23]")
    _require(1 <= pr.gas_min_days <= 23, "prices.gas_min_days must be in [1, 23]")
    _require(5 <= pr.min_winters <= 40, "prices.min_winters must be in [5, 40]")
    hubs = {"MIDC", "PALOVERDE", "SP15", "NP15", "MASSHUB", "PJMWEST", "INDIANA", "ERCOTNORTH"}
    _require(set(pr.hub_regions) <= hubs, f"prices.hub_regions keys must be in {sorted(hubs)}")
    bad_regions = sorted(set(pr.hub_regions.values()) - set(cfg.regions))
    _require(not bad_regions, f"prices.hub_regions has unknown regions {bad_regions}")
    for key in ("fall_months", "winter_months", "hydro_precip_months"):
        months = getattr(pr, key)
        _require(
            all(1 <= m <= 12 for m in months) and len(set(months)) == len(months),
            f"prices.{key} must be distinct month numbers",
        )
    _require(
        pr.hydro_hub in pr.hub_regions and pr.hydro_reference in pr.hub_regions,
        "prices.hydro_hub and prices.hydro_reference must be in prices.hub_regions",
    )
    _require(set(pr.hydro_states) <= set(STATES), "prices.hydro_states must be state codes")
    _require(1 <= pr.hydro_max_lag <= 12, "prices.hydro_max_lag must be in [1, 12]")
    _require(99 <= pr.permutations <= 99999, "prices.permutations must be in [99, 99999]")


def parse(raw: dict[str, Any], root: Path) -> Config:
    unknown = sorted(set(raw) - set(SECTIONS))
    missing = sorted(set(SECTIONS) - set(raw))
    if unknown:
        raise ConfigError(f"unknown sections {unknown}")
    if missing:
        raise ConfigError(f"missing sections {missing}")
    built = {name: _section(cls, raw[name], name) for name, cls in SECTIONS.items()}
    cfg = Config(root=root, **built)
    _validate(cfg)
    return cfg


def load(path: str | Path = "config.toml") -> Config:
    path = Path(path).resolve()
    if not path.is_file():
        raise ConfigError(f"config file not found: {path}")
    with path.open("rb") as fh:
        try:
            raw = tomllib.load(fh)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{path.name} is not valid toml: {exc}") from exc
    return parse(raw, path.parent)
