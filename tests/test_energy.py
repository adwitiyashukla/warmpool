from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from warmpool import forecast, load

RNG = np.random.default_rng(11)


def test_billing_mixes_previous_month_across_year_boundaries():
    cube = np.arange(24, dtype=float).reshape(2, 12, 1)
    out = load.billing(cube, 0.25)
    assert np.isnan(out[0, 0, 0])
    assert out[1, 0, 0] == pytest.approx(0.25 * 11 + 0.75 * 12)
    assert out[0, 5, 0] == pytest.approx(0.25 * 4 + 0.75 * 5)


def test_fit_unit_recovers_sensitivities_and_billing_weight():
    years = np.arange(2005, 2021)
    hdd = RNG.gamma(3.0, 6.0, size=(years.size, 12))
    cdd = RNG.gamma(2.0, 3.0, size=(years.size, 12))
    weights = (0.0, 0.25, 0.5, 0.75)
    hb = {w: load.billing(hdd[:, :, None], w)[:, :, 0] for w in weights}
    cb = {w: load.billing(cdd[:, :, None], w)[:, :, 0] for w in weights}
    level = np.repeat(np.linspace(10, 10.3, years.size), 12).reshape(years.size, 12)
    season = np.tile(0.05 * np.cos(np.arange(12)), (years.size, 1))
    logy = level + season + 0.012 * hb[0.5] + 0.007 * cb[0.5]
    logy = logy + RNG.normal(0, 0.004, logy.shape)
    mask = np.isfinite(hb[0.75]).ravel()
    fit = load.fit_unit(logy, hb, cb, years, mask, hac_lags=6)
    assert fit.weight == 0.5
    assert fit.beta[0] == pytest.approx(0.012, abs=6e-4)
    assert fit.beta[1] == pytest.approx(0.007, abs=6e-4)
    assert fit.r2 > 0.95


def _setup(members):
    years = np.arange(members.size)
    pred = np.linspace(-2, 2, members.size)
    s = forecast.Setup(target=999, years=years, predictor=pred, x0=1.8)
    s.weights["climatology"] = np.full(members.size, 1 / members.size)
    s.weights["enso"] = np.exp(-0.5 * ((pred - 1.8) / 0.3) ** 2)
    s.weights["enso"] /= s.weights["enso"].sum()
    s.weights["analog"] = (np.abs(pred - 1.8) < 0.3).astype(float)
    s.weights["analog"] /= s.weights["analog"].sum()
    return s


def test_gate_switches_between_climatology_and_enso():
    members = RNG.normal(size=100)
    s = _setup(members)
    mu, w, _ = forecast.predictive("gated", s, members, gate=1.0)
    assert np.allclose(w, s.weights["enso"])
    s.x0 = 0.3
    mu, w, _ = forecast.predictive("gated", s, members, gate=1.0)
    assert np.allclose(w, s.weights["climatology"])


def test_linear_method_shifts_with_the_predictor():
    pred = np.linspace(-2, 2, 200)
    members = 3.0 * pred + RNG.normal(0, 0.5, 200)
    s = _setup(members)
    s.predictor = pred
    mu, w, spread = forecast.predictive("linear", s, members, shift=1.0, sigma=0.0)
    assert mu[0] == pytest.approx(3.0 * 1.8 + 1.0, abs=0.3)
    assert spread == pytest.approx(0.5, abs=0.1)


def test_summary_is_internally_consistent():
    members = RNG.normal(size=80)
    s = _setup(members)
    for method in ("climatology", "enso", "analog", "linear"):
        mu, w, sigma = forecast.predictive(method, s, members, 0.2, 0.3)
        row = forecast.summarize(mu, w, sigma, 0.8, observed=0.5)
        assert row["lo"] <= row["median"] <= row["hi"]
        assert 0 <= row["pit"] <= 1 and 0 <= row["prob_above"] <= 1
        assert row["crps"] >= 0


def test_scenario_weights_follow_the_djf_value():
    winters = pd.DataFrame(
        {"winter": np.arange(1900, 2000), "djf": np.linspace(-2.5, 2.5, 100), "predictor": 0.0}
    )
    s = forecast.scenario_setup(cfg_stub(), winters, 2000, np.arange(1900, 2000), 2.4)
    top = s.years[np.argmax(s.weights["scenario"])]
    assert top >= 1995


def cfg_stub():
    class F:
        bandwidth = 0.3
        min_ess = 5

    class C:
        forecast = F()

    return C()
