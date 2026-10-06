from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

import duckdb
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.offline.offline import get_plotlyjs_version

from warmpool import views, warehouse
from warmpool.config import Config

TABLES = (
    "outlook_status",
    "outlook",
    "outlook_scenario",
    "outlook_price",
    "outlook_analog",
    "enso_month",
    "analog",
    "enso_event",
    "fingerprint",
    "teleconnection_summary",
    "teleconnection",
    "cluster_state",
    "cluster_score",
    "rule",
    "backtest_score",
    "price_premium",
    "price_premium_summary",
    "hydro_lag",
    "quality",
    "enso_calibration",
    "run_info",
    "source_file",
)

LEVELS = ("sales", "climate")

README = """---
title: warmpool
emoji: \U0001f30a
colorFrom: blue
colorTo: red
sdk: static
pinned: false
short_description: El Nino and US power demand, 1870 to the 2026-27 winter
---

Dashboard for the warmpool project: https://github.com/adwitiyashukla/warmpool
"""

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>warmpool</title>
<script src="__PLOTLY__"></script>
<style>
:root { --plane: #0d0d0d; --surface: #1a1a19; --grid: #2c2c2a; --line: #383835;
  --ink: #ffffff; --ink2: #c3c2b7; --muted: #898781; --blue: #3987e5; }
* { box-sizing: border-box; }
body { margin: 0; background: var(--plane); color: var(--ink2);
  font: 14px/1.5 system-ui, -apple-system, "Segoe UI", sans-serif; }
main { max-width: 1240px; margin: 0 auto; padding: 40px 24px 64px; }
h1 { color: var(--ink); font-size: 42px; margin: 0 0 4px; }
h3 { color: var(--ink); font-size: 15px; margin: 20px 0 8px; }
h3 span { color: var(--muted); font-weight: 400; }
.sub { color: var(--muted); margin: 0 0 24px; }
nav { display: flex; gap: 22px; border-bottom: 1px solid var(--line); margin-bottom: 20px;
  overflow-x: auto; }
nav button { background: none; border: 0; border-bottom: 2px solid transparent;
  color: var(--ink); padding: 8px 0; font: inherit; cursor: pointer; white-space: nowrap; }
nav button.on { color: var(--blue); border-bottom-color: var(--blue); }
section { display: none; }
section.on { display: block; }
.controls { display: flex; flex-wrap: wrap; gap: 28px; }
label { display: block; color: var(--ink); margin: 10px 0 6px; }
.radio { display: flex; gap: 20px; padding: 6px 0; }
.radio label { display: flex; gap: 6px; align-items: center; margin: 0; color: var(--ink2); }
select { background: var(--surface); color: var(--ink); border: 1px solid var(--line);
  border-radius: 6px; padding: 8px 10px; font: inherit; min-width: 260px; }
.tiles { display: grid; grid-template-columns: repeat(4, 1fr); gap: 24px; margin: 20px 0; }
.tiles.three { grid-template-columns: repeat(3, 1fr); }
.tile .label { color: var(--ink); }
.tile .value { color: var(--ink); font-size: 34px; line-height: 1.3; }
.tile .note { color: var(--muted); font-size: 13px; }
.chart { background: var(--surface); border-radius: 6px; margin: 12px 0; min-height: 320px; }
.row { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
.row.wide { grid-template-columns: 3fr 2fr; }
.row { align-items: start; }
.row > * { min-width: 0; }
.scroll { overflow: auto; max-height: 440px; border: 1px solid var(--grid); border-radius: 6px;
  margin: 8px 0; }
table.grid { border-collapse: collapse; width: 100%; font-size: 13px; }
table.grid th, table.grid td { padding: 6px 10px; border-bottom: 1px solid var(--grid);
  text-align: left; vertical-align: top; }
table.grid th { color: var(--muted); font-weight: 500; position: sticky; top: 0;
  background: var(--plane); white-space: nowrap; }
.swatch { display: inline-block; width: 12px; height: 12px; border-radius: 2px;
  margin-right: 6px; }
footer { color: var(--muted); margin-top: 40px; font-size: 13px; }
a { color: var(--blue); }
@media (max-width: 820px) {
  .row, .row.wide, .tiles, .tiles.three { grid-template-columns: 1fr; }
}
</style>
</head>
<body>
<main>
<h1>__TITLE__</h1>
<p class="sub">__SUBTITLE__</p>
<nav id="tabs"></nav>
<section>
<div class="controls">
<div><label>Measure</label><div class="radio" id="o-level"></div></div>
<div><label for="o-method">Forecast method</label><select id="o-method"></select></div>
</div>
<div class="tiles" id="o-tiles"></div>
<div class="chart" id="o-map"></div>
<div class="scroll" id="o-regions"></div>
<div class="row">
<div class="chart" id="o-scenario"></div>
<div>
<h3>Winter price premium outlook <span>(DJF average over Sep-Oct average)</span></h3>
<div class="scroll" id="o-price"></div>
<h3>DTW analog winters for 2026 <span>and what US heating demand did</span></h3>
<div class="scroll" id="o-analogs"></div>
</div>
</div>
</section>
<section>
<div class="chart" id="e-history"></div>
<div class="row wide">
<div class="chart" id="e-trajectories"></div>
<div><h3 id="e-title"></h3><div class="scroll" id="e-analogs"></div></div>
</div>
<h3>Strongest El Nino events since 1870</h3>
<div class="scroll" id="e-events"></div>
</section>
<section>
<div class="controls">
<div><label for="t-variable">Variable</label><select id="t-variable"></select></div>
<div><label for="t-season">Season</label><select id="t-season"></select></div>
</div>
<div class="chart" id="t-map"></div>
<div class="tiles three" id="t-counts"></div>
<div class="controls">
<div><label for="t-state">State for the month by lead grid</label>
<select id="t-state"></select></div>
</div>
<div class="chart" id="t-grid"></div>
<div class="scroll" id="t-summary"></div>
</section>
<section>
<div class="row wide">
<div class="chart" id="r-map"></div>
<div><h3>Clusters</h3><div id="r-groups"></div><div class="scroll" id="r-scores"></div></div>
</div>
<h3>Association rules mined with FP-growth <span>(permutation tested, BH FDR)</span></h3>
<div class="scroll" id="r-rules"></div>
</section>
<section>
<div class="controls">
<div><label>Level</label><div class="radio" id="b-level"></div></div>
<div><label for="b-subset">Winters</label><select id="b-subset"></select></div>
<div><label for="b-method">Method</label><select id="b-method"></select></div>
</div>
<div class="chart" id="b-map"></div>
<h3>CRPS skill vs climatology by region <span>(positive means the method helped)</span></h3>
<div class="scroll" id="b-regions"></div>
</section>
<section>
<div class="row">
<div class="chart" id="p-gas"></div>
<div><h3>Winter premium vs heating demand, by market</h3>
<div class="scroll" id="p-summary"></div></div>
</div>
<div class="chart" id="p-hydro"></div>
</section>
<section>
<h3>Quality checks run at every build</h3>
<div class="scroll" id="d-quality"></div>
<div class="row">
<div><h3>ENSO splice calibration <span>(HadISST to RONI)</span></h3>
<div class="scroll" id="d-calibration"></div></div>
<div><h3>Run info</h3><div class="scroll" id="d-run"></div></div>
</div>
<h3>Raw files and checksums</h3>
<div class="scroll" id="d-files"></div>
</section>
<footer>Static export of the warmpool dashboard, built from the run finished
<span id="built"></span>. Code:
<a href="https://github.com/adwitiyashukla/warmpool">github.com/adwitiyashukla/warmpool</a>
</footer>
</main>
<script type="application/json" id="data">__DATA__</script>
<script>
const D = JSON.parse(document.getElementById("data").textContent);
const CONFIG = {displaylogo: false, responsive: true};
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => (
  {"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));

function plot(id, fig) {
  Plotly.react($(id), fig.data, fig.layout, CONFIG);
}
function put(id, markup) {
  $(id).innerHTML = markup;
}
function options(id, items, chosen) {
  put(id, items.map(([v, t]) =>
    `<option value="${esc(v)}"${v === chosen ? " selected" : ""}>${esc(t)}</option>`).join(""));
}
function radios(id, items, chosen) {
  put(id, items.map(([v, t]) => `<label><input type="radio" name="${id}" value="${esc(v)}"`
    + `${v === chosen ? " checked" : ""}>${esc(t)}</label>`).join(""));
  document.querySelectorAll(`input[name="${id}"]`).forEach((r) =>
    r.addEventListener("change", () => PAGES[current]()));
}
function picked(id) {
  return document.querySelector(`input[name="${id}"]:checked`).value;
}
function tiles(id, items) {
  put(id, items.map((t) => `<div class="tile"><div class="label">${esc(t[0])}</div>`
    + `<div class="value">${esc(t[1])}</div>`
    + (t[2] ? `<div class="note">${esc(t[2])}</div>` : "") + "</div>").join(""));
}
function outlook() {
  const o = D.outlook;
  const method = $("o-method").value;
  const view = o.views[picked("o-level") + "|" + method];
  tiles("o-tiles", view.tiles);
  plot("o-map", view.map);
  put("o-regions", view.regions);
  put("o-price", o.price[method]);
  plot("o-scenario", o.scenario);
}
function event() {
  plot("e-history", D.event.history);
  plot("e-trajectories", D.event.trajectories);
}
function teleconnections() {
  const t = D.tele;
  const variable = $("t-variable").value;
  plot("t-map", t.fingerprint[variable + "|" + $("t-season").value]);
  plot("t-grid", t.grid[$("t-state").value + "|" + variable]);
}
function regions() {
  plot("r-map", D.regions.map);
}
function backtest() {
  const b = D.backtest;
  const level = picked("b-level");
  const methods = b.methods[level];
  const chosen = methods.some(([m]) => m === $("b-method").value) ? $("b-method").value : "enso";
  options("b-method", methods, chosen);
  plot("b-map", b.maps[level + "|" + $("b-subset").value + "|" + chosen]);
  put("b-regions", b.regions[level + "|" + $("b-subset").value]);
}
function prices() {
  plot("p-gas", D.prices.gas);
  plot("p-hydro", D.prices.hydro);
}
const PAGES = [outlook, event, teleconnections, regions, backtest, prices, () => {}];
let current = 0;
function show(i) {
  current = i;
  document.querySelectorAll("nav button").forEach((b, j) => b.classList.toggle("on", i === j));
  document.querySelectorAll("section").forEach((s, j) => s.classList.toggle("on", i === j));
  PAGES[i]();
}
put("tabs", D.tabs.map((t, i) => `<button type="button" data-i="${i}">${esc(t)}</button>`)
  .join(""));
document.querySelectorAll("nav button").forEach((b) =>
  b.addEventListener("click", () => show(Number(b.dataset.i))));
radios("o-level", D.outlook.levels, "sales");
options("o-method", D.outlook.methods, "gated");
put("o-analogs", D.outlook.analogs);
put("e-title", esc(D.event.title));
put("e-analogs", D.event.analogs);
put("e-events", D.event.events);
options("t-variable", D.tele.variables, "tmp");
options("t-season", D.tele.seasons.map((s) => [s, s]), "DJF");
options("t-state", D.tele.states.map((s) => [s, s]), "TX");
tiles("t-counts", D.tele.counts.map(([l, v]) => [l, v, ""]));
put("t-summary", D.tele.summary);
put("r-groups", D.regions.groups.map(([c, n, m]) => `<p><span class="swatch" `
  + `style="background:${esc(c)}"></span><b>${esc(n)}</b>: ${esc(m)}</p>`).join(""));
put("r-scores", D.regions.scores);
put("r-rules", D.regions.rules);
radios("b-level", D.backtest.levels, "climate");
options("b-subset", D.backtest.subsets, D.backtest.subsets[0][0]);
options("b-method", D.backtest.methods.climate, "enso");
put("p-summary", D.prices.summary);
["quality", "calibration", "run", "files"].forEach((k) => put("d-" + k, D.data[k]));
put("built", esc(D.built));
["o-method", "t-variable", "t-season", "t-state", "b-subset", "b-method"].forEach((id) =>
  $(id).addEventListener("change", () => PAGES[current]()));
show(0);
</script>
</body>
</html>
"""


def figure(fig: go.Figure) -> dict:
    fig.update_layout(template="none")
    return json.loads(pio.to_json(fig, validate=False, remove_uids=True))


def grid(frame: pd.DataFrame) -> str:
    return frame.to_html(index=False, border=0, classes="grid", na_rep="")


def read(cfg: Config) -> dict[str, pd.DataFrame]:
    with duckdb.connect(str(warehouse.gold_path(cfg)), read_only=True) as con:
        return {name: con.execute(f"select * from {name}").df() for name in TABLES}


def payload(t: dict[str, pd.DataFrame]) -> dict:
    status = t["outlook_status"].set_index("key")["value"]
    out = t["outlook"]
    methods = list(out["method"].unique())
    views_ = {}
    for level in LEVELS:
        for method in methods:
            tiles, fig, regions = views.outlook(out, status, level, method)
            views_[f"{level}|{method}"] = {
                "tiles": tiles,
                "map": figure(fig),
                "regions": grid(regions),
            }
    month = t["enso_month"].copy()
    month["date"] = pd.to_datetime(month["date"])
    title, analogs = views.analog_table(t["analog"])
    tele = t["teleconnection"]
    states = sorted(tele["state"].unique())
    variables = list(views.VARIABLE_NAME)
    score = t["backtest_score"]
    subsets = list(score["subset"].unique())
    backtest_levels = ("climate", "sales")
    cluster_map, groups = views.clusters(t["cluster_state"])
    return {
        "tabs": views.TABS,
        "built": pd.Timestamp(t["run_info"]["finished_utc"].iloc[0]).strftime("%Y-%m-%d %H:%M UTC"),
        "outlook": {
            "levels": [[lv, f"DJF {views.UNIT_LEVEL[lv]} vs normal"] for lv in LEVELS],
            "methods": [[m, views.METHOD_NAME.get(m, m)] for m in methods],
            "views": views_,
            "price": {m: grid(views.price_outlook(t["outlook_price"], m)) for m in methods},
            "scenario": figure(views.scenario(t["outlook_scenario"])),
            "analogs": grid(views.analog_outlook(t["outlook_analog"])),
        },
        "event": {
            "history": figure(views.enso_history(month)),
            "trajectories": figure(views.trajectories(month, t["analog"])),
            "title": title,
            "analogs": grid(analogs),
            "events": grid(views.top_el_ninos(t["enso_event"])),
        },
        "tele": {
            "variables": [[v, views.VARIABLE_NAME[v]] for v in variables],
            "seasons": views.SEASONS,
            "states": states,
            "fingerprint": {
                f"{v}|{s}": figure(views.fingerprint(t["fingerprint"], v, s))
                for v in variables
                for s in views.SEASONS
            },
            "grid": {
                f"{state}|{v}": figure(views.lead_grid(tele, state, v))
                for state in states
                for v in variables
            },
            "counts": views.field_counts(tele),
            "summary": grid(t["teleconnection_summary"]),
        },
        "regions": {
            "map": figure(cluster_map),
            "groups": groups,
            "scores": grid(views.rounded(t["cluster_score"], 3)),
            "rules": grid(views.rules_table(t["rule"])),
        },
        "backtest": {
            "levels": [[lv, views.UNIT_LEVEL[lv]] for lv in backtest_levels],
            "subsets": [[s, s.replace("_", " ")] for s in subsets],
            "methods": {
                lv: [[m, views.METHOD_NAME.get(m, m)] for m in views.backtest_methods(score, lv)]
                for lv in backtest_levels
            },
            "maps": {
                f"{lv}|{s}|{m}": figure(views.backtest_map(score, lv, s, m))
                for lv in backtest_levels
                for s in subsets
                for m in views.backtest_methods(score, lv)
            },
            "regions": {
                f"{lv}|{s}": grid(views.backtest_regions(score, lv, s).reset_index())
                for lv in backtest_levels
                for s in subsets
            },
        },
        "prices": {
            "gas": figure(views.gas_premium(t["price_premium"])),
            "summary": grid(views.premium_table(t["price_premium_summary"])),
            "hydro": figure(views.hydro(t["hydro_lag"])),
        },
        "data": {
            "quality": grid(views.quality_table(t["quality"])),
            "calibration": grid(views.rounded(t["enso_calibration"], 3)),
            "run": grid(t["run_info"]),
            "files": grid(t["source_file"]),
        },
    }


def page(data: dict) -> str:
    blob = json.dumps(data, separators=(",", ":")).replace("</", "<\\/")
    plotly = f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"
    return (
        PAGE.replace("__PLOTLY__", plotly)
        .replace("__TITLE__", html.escape(views.TITLE))
        .replace("__SUBTITLE__", html.escape(views.SUBTITLE))
        .replace("__DATA__", blob)
    )


def build(cfg: Config, out: Path) -> Path:
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    (out / "index.html").write_text(page(payload(read(cfg))), encoding="utf-8", newline="\n")
    (out / "README.md").write_text(README, encoding="utf-8", newline="\n")
    return out
