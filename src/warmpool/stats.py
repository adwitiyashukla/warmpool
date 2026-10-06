from __future__ import annotations

import numpy as np
from scipy import stats as sps

SQRT_PI = np.sqrt(np.pi)


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = p.size
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(n)
    q[order] = np.minimum(ranked, 1.0)
    return q


def by_qvalues(p: np.ndarray) -> np.ndarray:
    n = np.asarray(p).size
    return np.minimum(bh_qvalues(p) * np.sum(1.0 / np.arange(1, n + 1)), 1.0)


def qvalues(p: np.ndarray, method: str) -> np.ndarray:
    return by_qvalues(p) if method == "by" else bh_qvalues(p)


def ebisuzaki(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    n = x.size
    spectrum = np.fft.rfft(x - x.mean())
    amplitude = np.abs(spectrum)
    phases = rng.uniform(0.0, 2.0 * np.pi, size=(k, amplitude.size))
    coeffs = amplitude * np.exp(1j * phases)
    coeffs[:, 0] = 0.0
    if n % 2 == 0:
        coeffs[:, -1] = np.sqrt(2.0) * amplitude[-1] * np.cos(phases[:, -1])
    return np.fft.irfft(coeffs, n=n) + x.mean()


def standardized_ranks(a: np.ndarray, axis: int = 0) -> np.ndarray:
    ranks = sps.rankdata(a, axis=axis)
    ranks = ranks - ranks.mean(axis=axis, keepdims=True)
    scale = ranks.std(axis=axis, keepdims=True)
    return np.divide(ranks, scale, out=np.zeros_like(ranks), where=scale > 0)


def spearman_field(
    x: np.ndarray, y: np.ndarray, surrogates: int, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    n = x.size
    zy = standardized_ranks(y, axis=0)
    r = (standardized_ranks(x)[None, :] @ zy / n).ravel()
    null = standardized_ranks(ebisuzaki(x, surrogates, rng), axis=1) @ zy / n
    spread = np.arctanh(np.clip(null, -0.999999, 0.999999)).std(axis=0)
    z = np.arctanh(np.clip(r, -0.999999, 0.999999)) / np.where(spread > 0, spread, np.inf)
    return r, 2.0 * sps.norm.sf(np.abs(z))


def _a(m: np.ndarray, var: float) -> np.ndarray:
    if var <= 0:
        return np.abs(m)
    s = np.sqrt(var)
    z = m / s
    return 2.0 * s * sps.norm.pdf(z) + m * (2.0 * sps.norm.cdf(z) - 1.0)


def crps_mixture(mu: np.ndarray, w: np.ndarray, sigma: float, y: float) -> float:
    mu = np.asarray(mu, dtype=float)
    w = np.asarray(w, dtype=float)
    w = w / w.sum()
    first = np.sum(w * _a(y - mu, sigma**2))
    second = np.sum(np.outer(w, w) * _a(mu[:, None] - mu[None, :], 2.0 * sigma**2))
    return float(first - 0.5 * second)


def crps_normal(mu: float, sigma: float, y: float) -> float:
    z = (y - mu) / sigma
    return float(
        sigma * (z * (2.0 * sps.norm.cdf(z) - 1.0) + 2.0 * sps.norm.pdf(z) - 1.0 / SQRT_PI)
    )


def mixture_cdf(mu: np.ndarray, w: np.ndarray, sigma: float, x: float) -> float:
    w = np.asarray(w, dtype=float) / np.sum(w)
    mu = np.asarray(mu, dtype=float)
    if sigma <= 0:
        return float(np.sum(w * (mu <= x)))
    return float(np.sum(w * sps.norm.cdf((x - mu) / sigma)))


def mixture_quantiles(
    mu: np.ndarray, w: np.ndarray, sigma: float, probs: tuple[float, ...]
) -> list[float]:
    mu = np.asarray(mu, dtype=float)
    w = np.asarray(w, dtype=float) / np.sum(w)
    target = np.asarray(probs, dtype=float)
    if sigma <= 0:
        order = np.argsort(mu)
        cum = np.cumsum(w[order])
        pick = np.searchsorted(cum, target - 1e-12, side="left")
        return [float(v) for v in mu[order][np.minimum(pick, mu.size - 1)]]
    lo = np.full(target.size, mu.min() - 8.0 * sigma)
    hi = np.full(target.size, mu.max() + 8.0 * sigma)
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        cdf = (w[:, None] * sps.norm.cdf((mid[None, :] - mu[:, None]) / sigma)).sum(axis=0)
        below = cdf < target
        lo = np.where(below, mid, lo)
        hi = np.where(below, hi, mid)
    return [float(v) for v in 0.5 * (lo + hi)]


def kish_ess(w: np.ndarray) -> float:
    w = np.asarray(w, dtype=float)
    return float(w.sum() ** 2 / np.sum(w**2))


def kernel_weights(
    x: np.ndarray,
    x0: float,
    bandwidth: float,
    min_ess: float,
    grow: float = 1.25,
    max_steps: int = 40,
) -> tuple[np.ndarray, float]:
    x = np.asarray(x, dtype=float)
    h = bandwidth
    for _ in range(max_steps):
        w = np.exp(-0.5 * ((x - x0) / h) ** 2)
        if w.sum() > 0 and kish_ess(w) >= min(min_ess, x.size):
            return w / w.sum(), h
        h *= grow
    return np.full(x.size, 1.0 / x.size), np.inf


def ols_hac(x: np.ndarray, y: np.ndarray, lags: int) -> dict:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    beta, *_ = np.linalg.lstsq(x, y, rcond=None)
    resid = y - x @ beta
    xtx_inv = np.linalg.pinv(x.T @ x)
    scores = x * resid[:, None]
    meat = scores.T @ scores
    for lag in range(1, lags + 1):
        weight = 1.0 - lag / (lags + 1.0)
        cross = scores[lag:].T @ scores[:-lag]
        meat += weight * (cross + cross.T)
    cov = xtx_inv @ meat @ xtx_inv
    total = np.sum((y - y.mean()) ** 2)
    return {
        "beta": beta,
        "se": np.sqrt(np.clip(np.diag(cov), 0, None)),
        "resid": resid,
        "r2": float(1.0 - np.sum(resid**2) / total) if total > 0 else 0.0,
    }


def bootstrap_ci(
    values: np.ndarray, reps: int, rng: np.random.Generator, level: float = 0.9
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size < 2:
        return (np.nan, np.nan)
    means = values[rng.integers(0, values.size, size=(reps, values.size))].mean(axis=1)
    tail = (1.0 - level) / 2.0
    return float(np.quantile(means, tail)), float(np.quantile(means, 1.0 - tail))


def window_normal(values: np.ndarray, half: int, known: int | None = None) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    count_t = v.shape[0]
    usable = np.isfinite(v)
    if known is not None:
        usable &= (np.arange(count_t) <= known).reshape((-1,) + (1,) * (v.ndim - 1))
    zeros = np.zeros((1,) + v.shape[1:])
    csum = np.concatenate([zeros, np.cumsum(np.where(usable, v, 0.0), axis=0)])
    ccnt = np.concatenate([zeros, np.cumsum(usable, axis=0)])
    idx = np.arange(count_t)
    lo = np.clip(idx - half, 0, count_t - 1)
    hi = np.clip(idx + half, 0, count_t - 1)
    total = csum[hi + 1] - csum[lo] - np.where(usable, v, 0.0)
    count = ccnt[hi + 1] - ccnt[lo] - usable
    return np.divide(total, count, out=np.full(v.shape, np.nan), where=count > 0)


def window_anomaly(values: np.ndarray, half: int, known: int | None = None) -> np.ndarray:
    return np.asarray(values, dtype=float) - window_normal(values, half, known)
