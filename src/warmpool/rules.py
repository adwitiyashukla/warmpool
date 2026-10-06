from __future__ import annotations

import math

import numpy as np
import pandas as pd

from warmpool import climate, stats
from warmpool.config import Config
from warmpool.fpgrowth import fpgrowth


def djf_percent(panel: climate.Panel, var: str, half: int) -> tuple[np.ndarray, np.ndarray]:
    anom = climate.anomalies(panel, var, half)
    normal = panel.values[var] - anom
    a = climate.winter_months(anom, (12, 1, 2)).sum(axis=1)
    n = climate.winter_months(normal, (12, 1, 2)).sum(axis=1)
    return panel.years[:-1], 100.0 * a / n


def transactions(
    cfg: Config, panel: climate.Panel, winters: pd.DataFrame, units: list[str]
) -> pd.DataFrame:
    half = cfg.climate.anomaly_half_window
    years, hdd = djf_percent(panel, "hdd", half)
    _, pcpn = djf_percent(panel, "pcpn", half)
    pos = {u: panel.units.index(u) for u in units}
    ok = np.isfinite(hdd[:, [pos[u] for u in units]]).all(axis=1)
    frame = pd.DataFrame({"winter": years[ok]})
    for unit in units:
        frame[f"hdd:{unit}"] = hdd[ok, pos[unit]]
    for unit in cfg.rules.precip_regions:
        frame[f"pcpn:{unit}"] = pcpn[ok, panel.units.index(unit)]
    frame = frame.merge(winters[["winter", "label", "phase", "category"]], on="winter")
    items = []
    for row in frame.to_dict("records"):
        basket = {f"enso={row['phase']}"}
        if row["phase"] != "neutral":
            basket.add(f"enso={row['category']}")
            if row["category"].startswith(("strong", "very_strong")):
                basket.add(f"enso={row['phase']}_strong_plus")
        items.append(basket)
    for unit in units:
        col = frame[f"hdd:{unit}"]
        lo, hi = col.quantile([1 / 3, 2 / 3])
        for basket, value in zip(items, col, strict=True):
            if value > hi:
                basket.add(f"cold:{unit}")
            elif value < lo:
                basket.add(f"mild:{unit}")
    for unit in cfg.rules.precip_regions:
        col = frame[f"pcpn:{unit}"]
        lo, hi = col.quantile([1 / 3, 2 / 3])
        for basket, value in zip(items, col, strict=True):
            if value > hi:
                basket.add(f"wet:{unit}")
            elif value < lo:
                basket.add(f"dry:{unit}")
    members: dict[str, frozenset] = {}
    for i, basket in enumerate(items):
        for item in basket:
            if item.startswith("enso="):
                members[item] = members.get(item, frozenset()) | {i}
    for alias in [i for i in members if i.endswith("_strong_plus")]:
        if any(members[alias] == members[o] for o in members if o != alias):
            for basket in items:
                basket.discard(alias)
    frame["items"] = [" ".join(sorted(b)) for b in items]
    return frame


def closed(candidates: list[tuple], baskets: list[frozenset], n: int) -> list[tuple]:
    groups: dict[tuple, frozenset] = {}
    for ante, cons, *_ in candidates:
        hits = tuple(i for i, b in enumerate(baskets) if ante in b and cons <= b)
        if (ante, hits) not in groups:
            common = frozenset.intersection(*(baskets[i] for i in hits))
            groups[(ante, hits)] = frozenset(i for i in common if not i.startswith("enso="))
    out = []
    for (ante, hits), closure in groups.items():
        ante_count = sum(1 for b in baskets if ante in b)
        base = sum(1 for b in baskets if closure <= b) / n
        conf = len(hits) / ante_count
        out.append((ante, closure, len(hits), conf, base, conf / base))
    return sorted(out, key=lambda c: (c[0], -c[5], sorted(c[1])))


def mine(cfg: Config, frame: pd.DataFrame) -> pd.DataFrame:
    r = cfg.rules
    baskets = [frozenset(s.split()) for s in frame["items"]]
    n = len(baskets)
    min_count = max(2, math.ceil(r.min_support * n))
    found = fpgrowth(baskets, min_count, r.max_len)
    candidates = []
    for itemset, count in found.items():
        enso_items = [i for i in itemset if i.startswith("enso=")]
        if len(itemset) < 2 or len(enso_items) != 1:
            continue
        antecedent = enso_items[0]
        consequent = itemset - {antecedent}
        conf = count / found[frozenset([antecedent])]
        base = found[consequent] / n
        lift = conf / base
        if conf >= r.min_confidence and lift >= r.min_lift:
            candidates.append((antecedent, consequent, count, conf, base, lift))
    if not candidates:
        return pd.DataFrame()
    candidates = closed(candidates, baskets, n)
    enso_names = sorted({c[0] for c in candidates})
    enso_matrix = np.array([[name in b for name in enso_names] for b in baskets], dtype=float)
    cons_matrix = np.array([[c[1] <= b for c in candidates] for b in baskets], dtype=float)
    a_idx = np.array([enso_names.index(c[0]) for c in candidates])
    observed = np.array([c[2] for c in candidates], dtype=float)
    rng = np.random.default_rng(r.seed)
    exceed = np.zeros(len(candidates))
    for _ in range(r.permutations):
        shuffled = enso_matrix[rng.permutation(n)]
        exceed += (shuffled[:, a_idx] * cons_matrix).sum(axis=0) >= observed - 1e-9
    p = (1.0 + exceed) / (r.permutations + 1.0)
    rows = []
    for (ante, cons, count, conf, base, lift), pv, col in zip(
        candidates, p, cons_matrix.T, strict=True
    ):
        hits = frame.loc[(col > 0) & enso_matrix[:, enso_names.index(ante)].astype(bool), "label"]
        rows.append(
            {
                "antecedent": ante,
                "consequent": " & ".join(sorted(cons)),
                "size": len(cons),
                "count": int(count),
                "antecedent_count": int(found[frozenset([ante])]),
                "confidence": conf,
                "base_rate": base,
                "lift": lift,
                "p": pv,
                "winters": ", ".join(hits),
            }
        )
    out = pd.DataFrame(rows)
    out["q"] = stats.bh_qvalues(out["p"].to_numpy())
    out["significant"] = out["q"] <= r.fdr_q
    return out.sort_values(["significant", "lift", "count"], ascending=[False, False, False])
