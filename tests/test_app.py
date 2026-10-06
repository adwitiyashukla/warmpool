from __future__ import annotations

import json
import os

import pytest
from plotly.offline.offline import get_plotlyjs_version
from streamlit.testing.v1 import AppTest

from tests.conftest import ROOT
from warmpool import space, views

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


def test_space_is_a_static_page_with_every_view(replica_gold, tmp_path):
    cfg, _, _ = replica_gold
    out = space.build(cfg, tmp_path / "space")
    assert {p.name for p in out.iterdir()} == {"index.html", "README.md"}
    assert "sdk: static" in (out / "README.md").read_text(encoding="utf-8")
    page = (out / "index.html").read_text(encoding="utf-8")
    assert page.isascii()
    assert f"plotly-{get_plotlyjs_version()}.min.js" in page
    start = page.index('id="data">') + len('id="data">')
    data = json.loads(page[start : page.index("</script>", start)])
    assert data["tabs"] == views.TABS
    assert len(data["outlook"]["views"]) == 2 * len(data["outlook"]["methods"])
    assert len(data["tele"]["fingerprint"]) == len(views.VARIABLE_NAME) * len(views.SEASONS)
    assert len(data["tele"]["grid"]) == len(views.VARIABLE_NAME) * len(data["tele"]["states"])
    maps = data["backtest"]["maps"]
    assert "sales|all|perfect_weather" in maps
    assert "climate|all|perfect_weather" not in maps
    figures = [*maps.values(), *data["tele"]["fingerprint"].values()]
    assert all(fig["data"] and fig["layout"]["height"] for fig in figures)


def test_missing_warehouse_shows_a_clear_error(tmp_path, monkeypatch):
    at = _run(tmp_path / "nope.duckdb", monkeypatch)
    assert at.error and "No warehouse" in at.error[0].value
    assert os.environ["WARMPOOL_DB"].endswith("nope.duckdb")
