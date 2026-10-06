from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from sklearn.metrics import adjusted_rand_score, silhouette_score

from warmpool import stats
from warmpool.config import Config

SEASONS = ("DJF", "MAM", "JJA", "SON")


def ward_labels(features: np.ndarray, k: int) -> np.ndarray:
    return fcluster(linkage(features, method="ward"), t=k, criterion="maxclust")


def feature_matrix(
    seasonal: dict, variables: tuple[str, ...], units: list[str], rows: np.ndarray | None = None
) -> np.ndarray:
    cols = []
    for var in variables:
        for season in SEASONS:
            years, x, y = seasonal[(var, season)]
            if rows is not None:
                pos = {yr: i for i, yr in enumerate(years)}
                take = np.array([pos[yr] for yr in rows if yr in pos])
                x, y = x[take], y[take]
            zx = stats.standardized_ranks(x)
            zy = stats.standardized_ranks(y, axis=0)
            cols.append(zx @ zy / x.size)
    return np.column_stack(cols)[: len(units)]


def run(cfg: Config, seasonal: dict, states: list[str]) -> dict[str, pd.DataFrame]:
    c = cfg.cluster
    rng = np.random.default_rng(c.seed)
    feats = feature_matrix(seasonal, c.variables, states)
    common = sorted(
        set.intersection(*(set(seasonal[(v, s)][0]) for v in c.variables for s in SEASONS))
    )
    common = np.array(common)
    boots = [
        feature_matrix(
            seasonal, c.variables, states, rows=rng.choice(common, size=common.size, replace=True)
        )
        for _ in range(c.bootstrap)
    ]
    scores = []
    labels = {}
    for k in range(c.k_min, c.k_max + 1):
        labels[k] = ward_labels(feats, k)
        aris = [adjusted_rand_score(labels[k], ward_labels(boot, k)) for boot in boots]
        scores.append(
            {
                "k": k,
                "silhouette": float(silhouette_score(feats, labels[k])),
                "stability_ari": float(np.mean(aris)),
                "stability_ari_p10": float(np.quantile(aris, 0.1)),
            }
        )
    score = pd.DataFrame(scores)
    best = int(score.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"])
    names = [f"{v}_{s}" for v in c.variables for s in SEASONS]
    table = pd.DataFrame(feats, columns=names)
    table.insert(0, "state", states)
    table["raw_cluster"] = labels[best]
    lead = names[0]
    order = table.groupby("raw_cluster")[lead].mean().sort_values().index
    table["cluster"] = table["raw_cluster"].map({old: i + 1 for i, old in enumerate(order)})
    profile = table.groupby("cluster")[names].mean().reset_index()
    profile["states"] = (
        table.groupby("cluster")["state"].apply(lambda s: " ".join(sorted(s))).values
    )
    profile["name"] = [describe(row) for row in profile.to_dict("records")]
    table = table.merge(profile[["cluster", "name"]], on="cluster").drop(columns="raw_cluster")
    score["chosen"] = score["k"] == best
    return {
        "cluster_state": table.sort_values("state").reset_index(drop=True),
        "cluster_profile": profile,
        "cluster_score": score,
    }


def describe(row: dict) -> str:
    t = row.get("tmp_DJF", 0.0)
    p = row.get("pcpn_DJF", 0.0)
    temp = "warm" if t > 0.05 else "cool" if t < -0.05 else "mixed"
    wet = "wet" if p > 0.05 else "dry" if p < -0.05 else "near normal"
    return f"{temp} and {wet} El Nino winters"
