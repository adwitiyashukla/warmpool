from __future__ import annotations

import copy
import tomllib
from pathlib import Path

import pytest

from warmpool import pipeline, replica, warehouse
from warmpool.config import parse

ROOT = Path(__file__).resolve().parents[1]

FAST = {
    "teleconnect": {"surrogates": 99, "bootstrap": 100, "max_lead": 2},
    "cluster": {"bootstrap": 20, "k_max": 5},
    "rules": {"permutations": 99},
    "forecast": {"first_climate_winter": 2000, "first_sales_winter": 2018, "bootstrap": 200},
    "prices": {"permutations": 99},
}


@pytest.fixture
def base_raw():
    with (ROOT / "config.toml").open("rb") as fh:
        return tomllib.load(fh)


@pytest.fixture
def make_config(base_raw):
    def build(root, **overrides):
        raw = copy.deepcopy(base_raw)
        for section, values in overrides.items():
            raw[section].update(values)
        return parse(raw, Path(root))

    return build


@pytest.fixture(scope="session")
def replica_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("replica")
    world = replica.write_all(root / "data" / "raw")
    return root, world


@pytest.fixture(scope="session")
def replica_gold(replica_root):
    import duckdb

    root, world = replica_root
    with (ROOT / "config.toml").open("rb") as fh:
        raw = tomllib.load(fh)
    for section, values in FAST.items():
        raw[section].update(copy.deepcopy(values))
    cfg = parse(raw, root)
    warehouse.build(cfg, log=lambda _: None)
    pipeline.analyze(cfg, log=lambda _: None)
    with duckdb.connect(str(warehouse.gold_path(cfg)), read_only=True) as con:
        tables = {
            name: con.execute(f"select * from {name}").df()
            for (name,) in con.execute("show tables").fetchall()
        }
    return cfg, world, tables
