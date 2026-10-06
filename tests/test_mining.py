from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from mlxtend.frequent_patterns import fpgrowth as mlx_fpgrowth
from mlxtend.preprocessing import TransactionEncoder

from warmpool import climate, dtw, rules, teleconnect
from warmpool.analogs import pam
from warmpool.fpgrowth import brute_force, fpgrowth

RNG = np.random.default_rng(7)


def full_dtw(a, b, band):
    n, m = len(a), len(b)
    width = max(band, abs(n - m))
    table = np.full((n + 1, m + 1), np.inf)
    table[0, 0] = 0.0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if abs(i - j) <= width:
                step = min(table[i - 1, j], table[i, j - 1], table[i - 1, j - 1])
                table[i, j] = (a[i - 1] - b[j - 1]) ** 2 + step
    return math.sqrt(table[n, m])


def test_dtw_matches_full_dynamic_programming_and_lower_bound():
    for _ in range(150):
        n, band = int(RNG.integers(2, 14)), int(RNG.integers(0, 4))
        a, b = RNG.normal(size=n), RNG.normal(size=n)
        d = dtw.dtw(a, b, band)
        assert d == pytest.approx(full_dtw(a, b, band))
        assert dtw.dtw(b, a, band) == pytest.approx(d)
        upper, lower = dtw.envelope(b, band)
        assert dtw.lb_keogh(a, upper, lower) <= d + 1e-12


def test_early_abandon_only_cuts_hopeless_pairs():
    a, b = np.zeros(8), np.ones(8)
    assert dtw.dtw(a, b, 2, cutoff=1.0) == math.inf
    assert dtw.dtw(a, b, 2, cutoff=10.0) == pytest.approx(math.sqrt(8))


def test_pruned_search_equals_exhaustive_search():
    pool = {k: RNG.normal(size=8).cumsum() for k in range(120)}
    query = RNG.normal(size=8).cumsum()
    found, pruned = dtw.nearest(query, pool, 2, 6)
    exact = sorted(((k, dtw.dtw(query, s, 2)) for k, s in pool.items()), key=lambda t: t[1])[:6]
    assert [k for k, _ in found] == [k for k, _ in exact]
    assert pruned > 0


def _baskets(n_tx, n_items):
    items = [f"i{k}" for k in range(n_items)]
    probs = RNG.uniform(0.1, 0.7, n_items)
    return [
        frozenset(i for i, p in zip(items, probs, strict=True) if RNG.random() < p)
        for _ in range(n_tx)
    ]


def test_fpgrowth_equals_brute_force_on_random_data():
    for _ in range(25):
        tx = _baskets(int(RNG.integers(5, 50)), int(RNG.integers(3, 10)))
        min_count, max_len = int(RNG.integers(1, 5)), int(RNG.integers(1, 5))
        assert fpgrowth(tx, min_count, max_len) == brute_force(tx, min_count, max_len)


def test_fpgrowth_equals_mlxtend():
    tx = _baskets(130, 14)
    enc = TransactionEncoder()
    onehot = pd.DataFrame(enc.fit(tx).transform(tx), columns=enc.columns_)
    ref = mlx_fpgrowth(onehot, min_support=3 / 130, use_colnames=True, max_len=4)
    ours = fpgrowth(tx, 3, 4)
    assert {
        frozenset(s): round(v * 130) for s, v in zip(ref["itemsets"], ref["support"], strict=True)
    } == ours


def test_closed_rules_keep_one_rule_per_hit_set():
    baskets = [
        frozenset(b.split())
        for b in ["enso=x a b c", "enso=x a b c", "enso=x a b", "enso=y a", "enso=y c", "b"]
    ]
    cands = [
        ("enso=x", frozenset(["a"]), 3, 1.0, 0.67, 1.5),
        ("enso=x", frozenset(["a", "b"]), 3, 1.0, 0.5, 2.0),
        ("enso=x", frozenset(["a", "c"]), 2, 0.67, 0.5, 1.33),
    ]
    kept = rules.closed(cands, baskets, len(baskets))
    assert {(r[0], tuple(sorted(r[1])), r[2]) for r in kept} == {
        ("enso=x", ("a", "b", "c"), 2),
        ("enso=x", ("a", "b"), 3),
    }
    assert all(r[5] == pytest.approx(2.0) for r in kept)


def test_pam_finds_the_two_obvious_groups():
    points = np.concatenate([RNG.normal(0, 0.1, 10), RNG.normal(5, 0.1, 10)])
    dist = np.abs(points[:, None] - points[None, :])
    medoids = pam(dist, 2, np.random.default_rng(0))
    labels = np.argmin(dist[:, medoids], axis=1)
    assert len(set(labels[:10])) == 1 and len(set(labels[10:])) == 1
    assert labels[0] != labels[-1]


def _panel(years, units, signal, planted):
    values = {v: RNG.normal(size=(years.size, 12, len(units))) for v in climate.VARIABLES}
    values["tmp"][:, 0, :planted] += 3.0 * signal[:, None]
    return climate.Panel(years=years, units=units, values=values)


def test_teleconnection_mining_finds_planted_links_and_controls_noise(make_config, tmp_path):
    cfg = make_config(tmp_path, teleconnect={"surrogates": 999, "max_lead": 1})
    years = np.arange(1900, 2000)
    months = pd.date_range("1899-01-01", "1999-12-01", freq="MS")
    series = pd.Series(RNG.normal(size=months.size), index=months)
    jan = climate.enso_at(series, years, 1)
    units = [f"S{i}" for i in range(20)]
    found = teleconnect.mine(cfg, _panel(years, units, jan, 10), series)
    planted = (
        found["state"].isin(units[:10])
        & (found["variable"] == "tmp")
        & (found["month"] == 1)
        & (found["lead"] == 0)
    )
    assert found.loc[planted, "significant"].all()
    false_hits = found.loc[~planted, "significant"].sum()
    assert false_hits <= 0.1 * found["significant"].sum() + 2
