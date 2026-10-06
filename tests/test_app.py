from __future__ import annotations

import os

import duckdb
import pytest
from streamlit.testing.v1 import AppTest

from tests.conftest import ROOT
from warmpool import space

APP = str(ROOT / "app" / "app.py")


def _run(db, monkeypatch):
    monkeypatch.setenv("WARMPOOL_DB", str(db))
    return AppTest.from_file(APP, default_timeout=120).run()


def test_dashboard_renders_every_tab(replica_gold, monkeypatch):
    cfg, _, _ = replica_gold
    at = _run(cfg.path("gold") / "warmpool.duckdb", monkeypatch)
    assert not at.exception
    assert [t.label for t in at.tabs] == [
        "Winter 2026-27",
        "The event",
        "Teleconnections",
        "Regions and rules",
        "Backtest",
        "Prices",
        "Data",
    ]
    assert at.metric[0].label == "ENSO index now"


@pytest.mark.parametrize("method", ["climatology", "enso", "analog", "linear", "gated"])
def test_every_outlook_method_renders(replica_gold, monkeypatch, method):
    cfg, _, _ = replica_gold
    at = _run(cfg.path("gold") / "warmpool.duckdb", monkeypatch)
    at.radio[0].set_value("climate").run()
    at.selectbox[0].set_value(method).run()
    assert not at.exception


def test_space_bundle_runs_the_dashboard(replica_gold, tmp_path, monkeypatch):
    cfg, _, _ = replica_gold
    out = space.build(cfg, tmp_path / "space", ROOT / "app" / "app.py")
    assert {p.name for p in out.iterdir()} >= {
        "app.py",
        "Dockerfile",
        "requirements.txt",
        "README.md",
        "warmpool.duckdb",
        ".streamlit",
    }
    with duckdb.connect(str(out / "warmpool.duckdb"), read_only=True) as con:
        names = {n for (n,) in con.execute("show tables").fetchall()}
    assert names == set(space.APP_TABLES)
    at = _run(out / "warmpool.duckdb", monkeypatch)
    assert not at.exception


def test_missing_warehouse_shows_a_clear_error(tmp_path, monkeypatch):
    at = _run(tmp_path / "nope.duckdb", monkeypatch)
    assert at.error and "No warehouse" in at.error[0].value
    assert os.environ["WARMPOOL_DB"].endswith("nope.duckdb")
