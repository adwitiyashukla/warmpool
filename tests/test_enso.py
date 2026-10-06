from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from warmpool import enso

EDGES = (0.5, 1.0, 1.5, 2.0)


@pytest.mark.parametrize(
    ("year", "expected"),
    [
        (1956, (1941, 1970)),
        (1960, (1941, 1970)),
        (1961, (1946, 1975)),
        (2026, (1991, 2020)),
        (1871, (1870, 1899)),
    ],
)
def test_base_period_follows_the_cpc_rule(year, expected):
    assert enso.base_period(year, 1870, 2020, 30, 5, 1956) == expected


def _series(values, start="2000-01-01"):
    return pd.Series(values, index=pd.date_range(start, periods=len(values), freq="MS"))


def test_events_need_five_months_unless_ongoing():
    values = [0] * 3 + [0.6] * 4 + [0] * 3 + [0.7, 1.2, 2.3, 1.1, 0.6] + [0] * 2 + [-0.8] * 2
    events = enso.detect_events(_series(values), 0.5, 5, EDGES)
    assert events["event_id"].tolist() == ["EN2000", "LN2001"]
    el = events.iloc[0]
    assert (el["months"], el["peak"], el["strength"], el["status"]) == (
        5,
        2.3,
        "very_strong",
        "complete",
    )
    assert events.iloc[1]["status"] == "ongoing"


def test_strength_classes():
    assert [enso.strength(v, EDGES) for v in (0.5, 0.99, 1.0, 1.7, 2.0, 3.1, -1.6)] == [
        "weak",
        "weak",
        "moderate",
        "strong",
        "very_strong",
        "very_strong",
        "strong",
    ]


def test_harmonize_recovers_the_calibration():
    rng = np.random.default_rng(1)
    months = pd.date_range("1870-01-01", "2025-12-01", freq="MS")
    signal = np.sin(np.arange(months.size) / 7.0) + rng.normal(0, 0.1, months.size)
    nino = pd.DataFrame({"date": months, "nino34": signal})
    smooth = pd.Series(signal, index=months).rolling(3, center=True).mean()
    late = smooth[smooth.index >= "1950-01-01"].dropna()
    roni = pd.DataFrame({"date": late.index, "roni": 0.9 * late.to_numpy()})
    frame, cal = enso.harmonize(roni, nino, 30, 5, 1956)
    assert cal.r > 0.97
    assert cal.slope == pytest.approx(0.9, abs=0.08)
    assert (frame.loc[frame["date"] >= "1950-01-01", "source"] == "RONI").all()
    assert frame["date"].min() == pd.Timestamp("1870-02-01")


def test_winters_use_january_for_djf_and_flag_phase():
    values = np.zeros(36)
    values[6:20] = 1.6
    series = _series(values, "2001-01-01")
    events = enso.detect_events(series, 0.5, 5, EDGES)
    table = enso.winters(series, events, 8).set_index("winter")
    assert table.loc[2001, "label"] == "2001-02"
    assert table.loc[2001, "predictor"] == 1.6
    assert table.loc[2001, "category"] == "strong_el_nino"
    assert table.loc[2002, "phase"] == "neutral"
