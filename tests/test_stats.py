from __future__ import annotations

import numpy as np
import pytest
from scipy import integrate
from scipy import stats as sps
from statsmodels.regression.linear_model import OLS
from statsmodels.stats.multitest import multipletests

from warmpool import stats

RNG = np.random.default_rng(42)


def test_bh_and_by_match_statsmodels():
    p = RNG.uniform(size=300) ** 2
    assert np.allclose(stats.bh_qvalues(p), multipletests(p, method="fdr_bh")[1])
    assert np.allclose(stats.by_qvalues(p), multipletests(p, method="fdr_by")[1])


@pytest.mark.parametrize("sigma", [0.0, 0.35])
def test_crps_mixture_matches_numerical_integral(sigma):
    mu = RNG.normal(size=9)
    w = RNG.random(9)
    y = 0.4

    def cdf(x):
        return stats.mixture_cdf(mu, w, sigma, x)

    points = sorted(set(mu.tolist() + [y]))
    exact = integrate.quad(lambda x: (cdf(x) - (x >= y)) ** 2, -25, 25, points=points, limit=500)[0]
    assert stats.crps_mixture(mu, w, sigma, y) == pytest.approx(exact, abs=1e-6)


def test_crps_normal_closed_form_and_ensemble_identity():
    assert stats.crps_normal(0.5, 1.2, -0.3) == pytest.approx(
        stats.crps_mixture(np.array([0.5]), np.array([1.0]), 1.2, -0.3)
    )
    x = RNG.normal(size=60)
    ref = np.mean(np.abs(x - 0.2)) - 0.5 * np.mean(np.abs(x[:, None] - x[None, :]))
    assert stats.crps_mixture(x, np.ones(60), 0.0, 0.2) == pytest.approx(ref)


def test_mixture_quantiles_invert_the_cdf():
    mu, w = RNG.normal(size=30), RNG.random(30)
    for q, p in zip(
        stats.mixture_quantiles(mu, w, 0.5, (0.1, 0.5, 0.9)), (0.1, 0.5, 0.9), strict=True
    ):
        assert stats.mixture_cdf(mu, w, 0.5, q) == pytest.approx(p, abs=1e-9)


def test_window_normal_matches_a_plain_loop():
    v = RNG.normal(size=(40, 3))
    v[5, 1] = np.nan
    fast = stats.window_normal(v, 4, known=30)
    for t in range(40):
        for c in range(3):
            idx = [
                i
                for i in range(max(0, t - 4), min(39, t + 4) + 1)
                if i != t and i <= 30 and np.isfinite(v[i, c])
            ]
            if idx:
                assert fast[t, c] == pytest.approx(np.mean(v[idx, c]))
            else:
                assert np.isnan(fast[t, c])


def test_ebisuzaki_keeps_spectrum_and_mean():
    x = RNG.normal(size=64).cumsum()
    sur = stats.ebisuzaki(x, 5, RNG)
    amp = np.abs(np.fft.rfft(x - x.mean()))[1:-1]
    for row in sur:
        assert row.mean() == pytest.approx(x.mean())
        assert np.allclose(np.abs(np.fft.rfft(row - row.mean()))[1:-1], amp)


def test_spearman_field_matches_scipy_and_flags_signal():
    x = RNG.normal(size=120)
    y = np.column_stack([x + RNG.normal(scale=0.5, size=120), RNG.normal(size=120)])
    r, p = stats.spearman_field(x, y, 199, RNG)
    assert r[0] == pytest.approx(sps.spearmanr(x, y[:, 0])[0])
    assert p[0] < 1e-10
    assert p[1] > 0.01


def test_surrogate_p_values_are_uniform_under_the_null():
    x = RNG.normal(size=100)
    y = RNG.normal(size=(100, 2000))
    _, p = stats.spearman_field(x, y, 499, RNG)
    assert abs(np.mean(p <= 0.05) - 0.05) < 0.015
    assert abs(np.mean(p <= 0.5) - 0.5) < 0.04


def test_kernel_weights_reach_the_effective_size():
    x = np.linspace(-2, 2, 100)
    w, h = stats.kernel_weights(x, 1.9, 0.05, 12)
    assert stats.kish_ess(w) >= 12
    assert h > 0.05
    assert w.sum() == pytest.approx(1.0)


def test_ols_hac_matches_statsmodels():
    x = np.column_stack([np.ones(200), RNG.normal(size=200)])
    y = x @ np.array([1.0, 2.0]) + RNG.normal(size=200)
    ours = stats.ols_hac(x, y, 4)
    ref = OLS(y, x).fit(cov_type="HAC", cov_kwds={"maxlags": 4, "use_correction": False})
    assert np.allclose(ours["beta"], ref.params)
    assert np.allclose(ours["se"], ref.bse)
