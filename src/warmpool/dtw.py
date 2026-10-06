from __future__ import annotations

import heapq
import math
from collections.abc import Hashable, Mapping

import numpy as np


def dtw(a: np.ndarray, b: np.ndarray, band: int, cutoff: float = math.inf) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    n, m = a.size, b.size
    width = max(band, abs(n - m))
    limit = cutoff * cutoff
    prev = np.full(m + 1, math.inf)
    prev[0] = 0.0
    for i in range(1, n + 1):
        cur = np.full(m + 1, math.inf)
        lo, hi = max(1, i - width), min(m, i + width)
        for j in range(lo, hi + 1):
            cost = (a[i - 1] - b[j - 1]) ** 2
            cur[j] = cost + min(prev[j], cur[j - 1], prev[j - 1])
        if cur[lo : hi + 1].min() > limit:
            return math.inf
        prev = cur
    return math.sqrt(prev[m])


def envelope(c: np.ndarray, band: int) -> tuple[np.ndarray, np.ndarray]:
    c = np.asarray(c, dtype=float)
    upper = np.array([c[max(0, i - band) : i + band + 1].max() for i in range(c.size)])
    lower = np.array([c[max(0, i - band) : i + band + 1].min() for i in range(c.size)])
    return upper, lower


def lb_keogh(q: np.ndarray, upper: np.ndarray, lower: np.ndarray) -> float:
    q = np.asarray(q, dtype=float)
    above = np.clip(q - upper, 0, None)
    below = np.clip(lower - q, 0, None)
    return float(math.sqrt(np.sum(above**2 + below**2)))


def nearest(
    query: np.ndarray, candidates: Mapping[Hashable, np.ndarray], band: int, k: int
) -> tuple[list[tuple[Hashable, float]], int]:
    bounds = []
    for key, seq in candidates.items():
        upper, lower = envelope(seq, band)
        bounds.append((lb_keogh(query, upper, lower), key))
    bounds.sort(key=lambda item: item[0])
    best: list[tuple[float, int, Hashable]] = []
    computed = 0
    for order, (bound, key) in enumerate(bounds):
        if len(best) == k and bound >= -best[0][0]:
            break
        cutoff = -best[0][0] if len(best) == k else math.inf
        dist = dtw(query, candidates[key], band, cutoff)
        computed += 1
        if len(best) < k:
            heapq.heappush(best, (-dist, -order, key))
        elif dist < -best[0][0]:
            heapq.heapreplace(best, (-dist, -order, key))
    ranked = sorted(((key, -neg) for neg, _, key in best), key=lambda item: item[1])
    return ranked, len(bounds) - computed


def distance_matrix(series: list[np.ndarray], band: int) -> np.ndarray:
    n = len(series)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            out[i, j] = out[j, i] = dtw(series[i], series[j], band)
    return out
