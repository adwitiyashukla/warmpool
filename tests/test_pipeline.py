from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from warmpool import cli, parse, replica


@pytest.fixture
def gold(replica_gold):
    return replica_gold


def test_replica_round_trips_through_the_parsers(replica_root):
    root, world = replica_root
    raw = root / "data" / "raw"
    retail = parse.parse_retail(sorted((raw / "retail").glob("*.xlsx")))
    res = retail[retail["sector"] == "residential"].set_index(["state", "year", "month"])
    truth = replica.retail_frame(world)
    truth = truth.assign(year=truth["date"].dt.year, month=truth["date"].dt.month)
    truth = truth.set_index(["state", "year", "month"])
    joined = res.join(truth["res"], how="inner")
    assert len(joined) == len(truth)
    assert np.allclose(joined["sales_mwh"], joined["res"], rtol=1e-6)
    clim = parse.parse_climate(raw / "climdiv", ("tmpc", "hddc", "cddc", "pcpn"))
    assert len(clim) == len(world.climate)
    prices = parse.parse_wholesale(raw / "wholesale")
    assert set(prices["hub"]) == set(replica.HUB_NAMES)


def test_quality_checks_all_pass_on_the_replica(gold):
    _, _, tables = gold
    assert tables["quality"]["passed"].all()


def test_every_result_table_is_written(gold):
    _, _, tables = gold
    expected = {
        "enso_event",
        "enso_winter",
        "teleconnection",
        "fingerprint",
        "cluster_state",
        "analog",
        "event_family",
        "rule",
        "backtest",
        "backtest_score",
        "load_model",
        "price_premium_summary",
        "hydro_lag",
        "outlook",
        "outlook_scenario",
        "outlook_price",
        "run_info",
    }
    assert expected <= set(tables)
    assert not tables["outlook"].query("unit == 'US' and level == 'sales'").empty


def test_injected_load_sensitivities_are_recovered(gold):
    _, world, tables = gold
    model = tables["load_model"]
    states = model[model["unit"].isin(world.truth.heating)].copy()
    truth = states["unit"].map(world.truth.heating) * 100
    assert np.corrcoef(states["heating_pct_per_hdd"], truth)[0, 1] > 0.7
    assert (states["billing_weight"] == world.truth.billing_weight).mean() > 0.9


def test_injected_climate_structure_is_recovered(gold):
    _, world, tables = gold
    fp = tables["fingerprint"]
    djf = fp[(fp["variable"] == "tmp") & (fp["season"] == "DJF")].set_index("unit")
    strong = {s: v for s, v in world.truth.djf_tmp_per_c.items() if abs(v) >= 1}
    assert all(np.sign(djf.loc[s, "r"]) == np.sign(v) for s, v in strong.items())
    cl = tables["cluster_state"]
    groups = cl["state"].map(
        lambda s: "N" if s in replica.NORTH else "S" if s in replica.SOUTH else "O"
    )
    assert adjusted_rand_score(groups, cl["cluster"]) > 0.8


def test_injected_hydro_effect_is_recovered(gold):
    _, _, tables = gold
    lag = tables["hydro_lag"]
    spring = lag[(lag["driver"] == "precip") & lag["month"].between(4, 7)]
    assert (spring["r"] < -0.8).all()
    assert (spring["slope"].between(-1.2, -0.7)).all()


def test_gas_premium_tracks_heating_demand(gold):
    _, _, tables = gold
    gas = tables["price_premium_summary"].set_index("market").loc["HENRYHUB"]
    assert gas["slope_per_hdd_pct"] > 0.5
    assert gas["p"] < 0.05


def test_backtest_scores_are_relative_to_climatology(gold):
    _, _, tables = gold
    scores = tables["backtest_score"]
    clim = scores[scores["method"] == "climatology"]
    assert np.allclose(clim["crpss"], 0.0)
    perfect = scores[(scores["method"] == "perfect_weather") & (scores["subset"] == "all")]
    assert (perfect["crps"] > 0).all()


def test_cli_reports_config_errors(tmp_path, capsys):
    bad = tmp_path / "config.toml"
    bad.write_text("[paths]\nraw = 1\n")
    assert cli.main(["--config", str(bad), "build"]) == 2
    assert "config error" in capsys.readouterr().err


def test_outlook_is_a_proper_distribution(gold):
    _, _, tables = gold
    out = tables["outlook"]
    assert (out["lo"] <= out["median"]).all() and (out["median"] <= out["hi"]).all()
    assert out["prob_above"].between(0, 1).all()
    assert isinstance(tables["outlook_status"], pd.DataFrame)
