from __future__ import annotations

import copy

import pytest

from tests.conftest import ROOT
from warmpool.config import ConfigError, load, parse


def test_shipped_config_loads():
    cfg = load(ROOT / "config.toml")
    assert cfg.root == ROOT
    assert cfg.path("raw") == ROOT / "data" / "raw"


def _broken(base, section, key, value):
    raw = copy.deepcopy(base)
    if value is KeyError:
        del raw[section][key]
    else:
        raw[section][key] = value
    return raw


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("download", "retries", True, "integer"),
        ("download", "retries", 2.5, "integer"),
        ("download", "workers", 0, "workers"),
        ("download", "timeout_s", "fast", "number"),
        ("download", "timeout_s", -1, "timeout_s"),
        ("download", "user_agent", "  ", "non-empty"),
        ("download", "colour", 3, "unknown keys"),
        ("download", "retries", KeyError, "missing keys"),
        ("sources", "roni", "http://insecure.example", "https"),
        ("sources", "climdiv_base", "https://x.org/climdiv", "end with /"),
        ("sources", "climdiv_elements", ["tmpc", "snow"], "unknown elements"),
        ("sources", "climdiv_elements", ["tmpc", "tmpc"], "duplicates"),
        ("sources", "climdiv_elements", [], "non-empty list"),
        ("paths", "gold", "data/raw", "distinct"),
        ("enso", "strength_edges", [0.5, 1.5, 1.0, 2.0], "increasing"),
        ("enso", "strength_edges", [0.4, 1.0, 1.5, 2.0], "start at"),
        ("enso", "predictor_month", 13, "month number"),
        ("climate", "weight_years", [2024, 2015], "first, last"),
        ("climate", "dc_climate_state", "DC", "state code"),
        ("teleconnect", "fdr_method", "holm", "bh or by"),
        ("teleconnect", "variables", ["tmp", "snow"], "teleconnect.variables"),
        ("cluster", "k_min", 9, "k_min <= k_max"),
        ("analog", "start_month", 9, "before enso.predictor_month"),
        ("rules", "precip_regions", ["ATLANTIS"], "unknown regions"),
        ("rules", "min_lift", 0.5, "at least 1"),
        ("forecast", "methods", ["enso", "climatology"], "start with climatology"),
        ("forecast", "methods", ["climatology", "magic"], "forecast.methods"),
        ("forecast", "billing_weights", [0.5, 1.0], "billing_weights"),
        ("forecast", "sector", "farms", "forecast.sector"),
        ("outlook", "winter", 1990, "after the backtest"),
        ("prices", "hydro_hub", "NYC", "hydro_hub"),
        ("prices", "fall_months", [9, 9], "distinct month"),
    ],
)
def test_bad_values_are_rejected(base_raw, tmp_path, section, key, value, message):
    with pytest.raises(ConfigError, match=message):
        parse(_broken(base_raw, section, key, value), tmp_path)


def test_regions_must_partition_the_states(base_raw, tmp_path):
    raw = copy.deepcopy(base_raw)
    raw["regions"]["ERCOT"] = ["TX", "OK"]
    with pytest.raises(ConfigError, match="both"):
        parse(raw, tmp_path)
    raw = copy.deepcopy(base_raw)
    del raw["regions"]["CAISO"]
    with pytest.raises(ConfigError, match="do not cover"):
        parse(raw, tmp_path)
    raw = copy.deepcopy(base_raw)
    raw["prices"]["hub_regions"]["MIDC"] = "ATLANTIS"
    with pytest.raises(ConfigError, match="unknown regions"):
        parse(raw, tmp_path)


def test_unknown_and_missing_sections(base_raw, tmp_path):
    raw = copy.deepcopy(base_raw)
    raw["extra"] = {}
    with pytest.raises(ConfigError, match="unknown sections"):
        parse(raw, tmp_path)
    raw = copy.deepcopy(base_raw)
    del raw["paths"]
    with pytest.raises(ConfigError, match="missing sections"):
        parse(raw, tmp_path)


def test_missing_and_invalid_files(tmp_path):
    with pytest.raises(ConfigError, match="not found"):
        load(tmp_path / "nope.toml")
    bad = tmp_path / "bad.toml"
    bad.write_text("[paths\nraw = 1")
    with pytest.raises(ConfigError, match="not valid toml"):
        load(bad)
