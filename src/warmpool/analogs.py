from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from warmpool import dtw


def window(series: pd.Series, year: int, first: int, last: int) -> np.ndarray | None:
    dates = pd.date_range(pd.Timestamp(year, first, 1), pd.Timestamp(year, last, 1), freq="MS")
    if not all(d in series.index for d in dates):
        return None
    values = series.loc[dates].to_numpy(dtype=float)
    return None if np.isnan(values).any() else values


def candidates(series: pd.Series, before: int, first: int, last: int) -> dict[int, np.ndarray]:
    out = {}
    for year in range(series.index.min().year, before):
        seq = window(series, year, first, last)
        if seq is not None:
            out[year] = seq
    return out


def analog_years(
    series: pd.Series, target: int, first: int, last: int, band: int, k: int
) -> tuple[list[tuple[int, float]], int]:
    query = window(series, target, first, last)
    if query is None:
        raise ValueError(f"no ENSO trajectory for {target} months {first}-{last}")
    return dtw.nearest(query, candidates(series, target, first, last), band, k)


def current_analogs(
    series: pd.Series, winters: pd.DataFrame, target: int, first: int, band: int, k: int
) -> pd.DataFrame:
    last = int(series[series.index.year == target].dropna().index.max().month)
    found, pruned = analog_years(series, target, first, last, band, k)
    query = window(series, target, first, last)
    info = winters.set_index("winter")
    rows = []
    for rank, (year, dist) in enumerate(found, start=1):
        seq = window(series, year, first, last)
        rows.append(
            {
                "rank": rank,
                "year": year,
                "label": info.loc[year, "label"],
                "dtw": dist,
                "euclidean": float(np.sqrt(np.sum((seq - query) ** 2))),
                "predictor": info.loc[year, "predictor"],
                "djf": info.loc[year, "djf"],
                "category": info.loc[year, "category"],
                "months": f"{first}-{last}",
                "pruned": pruned,
            }
        )
    return pd.DataFrame(rows)


def pam(dist: np.ndarray, k: int, rng: np.random.Generator, restarts: int = 20) -> np.ndarray:
    n = dist.shape[0]
    best_cost, best_medoids = np.inf, None
    for _ in range(restarts):
        medoids = np.sort(rng.choice(n, size=k, replace=False))
        for _ in range(100):
            labels = np.argmin(dist[:, medoids], axis=1)
            new = medoids.copy()
            for c in range(k):
                members = np.flatnonzero(labels == c)
                if members.size:
                    new[c] = members[np.argmin(dist[np.ix_(members, members)].sum(axis=1))]
            new = np.sort(new)
            if np.array_equal(new, medoids):
                break
            medoids = new
        cost = dist[:, medoids].min(axis=1).sum()
        if cost < best_cost - 1e-12:
            best_cost, best_medoids = cost, medoids
    return best_medoids


def families(
    series: pd.Series, events: pd.DataFrame, band: int, k_max: int, seed: int = 5
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, seqs = [], []
    for ev in events[
        (events["phase"] == "el_nino") & (events["status"] == "complete")
    ].itertuples():
        year0 = ev.peak_date.year if ev.peak_date.month >= 7 else ev.peak_date.year - 1
        dates = pd.date_range(pd.Timestamp(year0, 1, 1), periods=24, freq="MS")
        if not all(d in series.index for d in dates):
            continue
        seqs.append(series.loc[dates].to_numpy(dtype=float))
        rows.append(
            {"event_id": ev.event_id, "year0": year0, "peak": ev.peak, "strength": ev.strength}
        )
    table = pd.DataFrame(rows)
    dist = dtw.distance_matrix(seqs, band)
    rng = np.random.default_rng(seed)
    best = (-1.0, None, None)
    for k in range(2, min(k_max, len(seqs) - 1) + 1):
        medoids = pam(dist, k, rng)
        labels = np.argmin(dist[:, medoids], axis=1)
        if np.bincount(labels, minlength=k).min() < 3:
            continue
        s = float(silhouette_score(dist, labels, metric="precomputed"))
        if s > best[0]:
            best = (s, medoids, labels)
    score, medoids, labels = best
    if medoids is None:
        medoids = pam(dist, 2, rng)
        labels = np.argmin(dist[:, medoids], axis=1)
        score = float(silhouette_score(dist, labels, metric="precomputed"))
    table["raw"] = labels
    order = table.groupby("raw")["peak"].mean().sort_values(ascending=False).index
    names = {old: chr(ord("A") + i) for i, old in enumerate(order)}
    table["family"] = table["raw"].map(names)
    table["medoid"] = [i in set(medoids.tolist()) for i in range(len(table))]
    table["silhouette"] = score
    traj = pd.DataFrame(
        [
            {"event_id": eid, "step": step, "month": step % 12 + 1, "value": float(v)}
            for eid, seq in zip(table["event_id"], seqs, strict=True)
            for step, v in enumerate(seq)
        ]
    )
    return table.drop(columns="raw"), traj


def nearest_family(
    series: pd.Series,
    families_table: pd.DataFrame,
    traj: pd.DataFrame,
    year: int,
    first: int,
    band: int,
) -> pd.DataFrame:
    last = int(series[series.index.year == year].dropna().index.max().month)
    query = window(series, year, first, last)
    rows = []
    for row in families_table[families_table["medoid"]].itertuples():
        seq = traj[traj["event_id"] == row.event_id].sort_values("step")["value"].to_numpy()
        rows.append(
            {
                "family": row.family,
                "medoid": row.event_id,
                "dtw": dtw.dtw(query, seq[first - 1 : last], band),
            }
        )
    return pd.DataFrame(rows).sort_values("dtw").reset_index(drop=True)
