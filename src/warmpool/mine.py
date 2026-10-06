from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from warmpool import analogs, climate, cluster, enso, rules, teleconnect, warehouse
from warmpool.config import STATES, Config


def context(cfg: Config) -> dict:
    e = cfg.enso
    series = climate.enso_series(warehouse.read_table(cfg, "enso_month"))
    events = enso.detect_events(series, e.threshold, e.min_months, e.strength_edges)
    winters = enso.winters(series, events, e.predictor_month)
    units = list(STATES) + ["DC"] + list(cfg.regions) + ["US"]
    panel = climate.build_panel(
        warehouse.read_table(cfg, "unit_climate"), units, cfg.climate.first_year
    )
    return {"series": series, "events": events, "winters": winters, "panel": panel}


def run(cfg: Config, ctx: dict, log: Callable[[str], None] = print) -> dict[str, pd.DataFrame]:
    series, panel, winters = ctx["series"], ctx["panel"], ctx["winters"]
    field = panel.subset(list(STATES))
    out = {"enso_event": ctx["events"], "enso_winter": winters}
    tele = teleconnect.mine(cfg, field, series)
    out["teleconnection"] = tele
    out["teleconnection_summary"] = teleconnect.summary(tele, cfg.teleconnect.fdr_q)
    log(
        f"teleconnections: {len(tele):,} tests, {int(tele['significant'].sum()):,} discoveries "
        f"at FDR q={cfg.teleconnect.fdr_q}"
    )
    seasonal = teleconnect.season_data(cfg, panel, series)
    finger = teleconnect.fingerprint(cfg, panel, series, set(STATES))
    out["fingerprint"] = finger
    out.update(cluster.run(cfg, seasonal, list(STATES)))
    chosen = out["cluster_score"].query("chosen").iloc[0]
    log(
        f"clusters: k={int(chosen['k'])}, silhouette {chosen['silhouette']:.3f}, "
        f"bootstrap ARI {chosen['stability_ari']:.3f}"
    )
    a = cfg.analog
    target = int(series.index.max().year)
    out["analog"] = analogs.current_analogs(series, winters, target, a.start_month, a.band, a.top_k)
    out["event_family"], out["event_trajectory"] = analogs.families(
        series, ctx["events"], a.band, a.family_k_max
    )
    log(f"analogs for {target}: " + ", ".join(out["analog"]["label"].head(5)))
    regions = list(cfg.regions) + ["US"]
    baskets = rules.transactions(cfg, panel, winters, regions)
    out["rule_basket"] = baskets
    found = rules.mine(cfg, baskets)
    out["rule"] = found
    hits = int(found["significant"].sum()) if len(found) else 0
    log(f"rules: {len(found)} closed rules, {hits} significant at FDR q={cfg.rules.fdr_q}")
    return out
