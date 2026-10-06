from __future__ import annotations

import os
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

DB = Path(os.environ.get("WARMPOOL_DB", "data/gold/warmpool.duckdb"))
SURFACE, PLANE, INK, INK2, MUTED = "#1a1a19", "#0d0d0d", "#ffffff", "#c3c2b7", "#898781"
GRID, BASELINE = "#2c2c2a", "#383835"
BLUE, ORANGE, AQUA, RED = "#3987e5", "#d95926", "#199e70", "#e66767"
SERIES = [BLUE, ORANGE, AQUA]
DIVERGING = [[0.0, BLUE], [0.5, BASELINE], [1.0, RED]]
PHASE_COLOR = {"la_nina": BLUE, "neutral": MUTED, "el_nino": RED}
PHASE_NAME = {"la_nina": "La Nina", "neutral": "Neutral", "el_nino": "El Nino"}
METHOD_NAME = {
    "climatology": "Climatology",
    "enso": "ENSO kernel",
    "analog": "DTW analogs",
    "linear": "Linear",
    "gated": "Gated ENSO",
    "perfect_weather": "Perfect weather",
}
UNIT_LEVEL = {"sales": "residential sales", "climate": "heating degree days"}


@st.cache_data(show_spinner=False)
def table(name: str) -> pd.DataFrame:
    with duckdb.connect(str(DB), read_only=True) as con:
        return con.execute(f"select * from {name}").df()


def style(fig: go.Figure, height: int = 420, title: str | None = None) -> go.Figure:
    fig.update_layout(
        paper_bgcolor=SURFACE,
        plot_bgcolor=SURFACE,
        height=height,
        font={
            "color": INK2,
            "family": "system-ui, -apple-system, Segoe UI, sans-serif",
            "size": 13,
        },
        title={"text": title, "font": {"color": INK, "size": 15}} if title else None,
        margin={"l": 50, "r": 20, "t": 56 if title else 24, "b": 44},
        hoverlabel={"bgcolor": GRID, "font": {"color": INK}},
        legend={"orientation": "h", "y": 1.08, "x": 0, "bgcolor": "rgba(0,0,0,0)"},
    )
    fig.update_xaxes(
        gridcolor=GRID, zerolinecolor=BASELINE, linecolor=BASELINE, tickfont={"color": MUTED}
    )
    fig.update_yaxes(
        gridcolor=GRID, zerolinecolor=BASELINE, linecolor=BASELINE, tickfont={"color": MUTED}
    )
    return fig


def us_map(
    frame: pd.DataFrame,
    value: str,
    label: str,
    hover: str,
    flip: bool = False,
    limit: float | None = None,
) -> go.Figure:
    span = limit or float(np.nanmax(np.abs(frame[value]))) or 1.0
    fig = go.Figure(
        go.Choropleth(
            locations=frame["state"],
            z=frame[value],
            locationmode="USA-states",
            colorscale=DIVERGING,
            reversescale=flip,
            zmid=0,
            zmin=-span,
            zmax=span,
            marker_line_color=SURFACE,
            marker_line_width=2,
            text=frame[hover],
            hovertemplate="%{location}<br>%{text}<extra></extra>",
            colorbar={
                "title": {"text": label, "font": {"color": INK2}},
                "tickfont": {"color": MUTED},
                "thickness": 12,
                "len": 0.8,
            },
        )
    )
    fig.update_geos(
        scope="usa",
        bgcolor=SURFACE,
        lakecolor=SURFACE,
        landcolor=GRID,
        showlakes=False,
        subunitcolor=SURFACE,
    )
    return style(fig, 430)


def tile(col, label: str, value: str, note: str) -> None:
    col.metric(label, value)
    col.caption(note)


def page_outlook() -> None:
    status = table("outlook_status").set_index("key")["value"]
    out = table("outlook")
    level = st.radio(
        "Measure",
        ["sales", "climate"],
        horizontal=True,
        format_func=lambda v: f"DJF {UNIT_LEVEL[v]} vs normal",
    )
    method = st.selectbox(
        "Forecast method",
        list(out["method"].unique()),
        index=list(out["method"].unique()).index("gated"),
        format_func=METHOD_NAME.get,
    )
    pick = out[(out["level"] == level) & (out["method"] == method)]
    us = pick[pick["unit"] == "US"].iloc[0]
    a, b, c, d = st.columns(4)
    tile(
        a,
        "ENSO index now",
        status["latest_index"],
        f"RONI, season centered {status['latest_season_center'][:7]}",
    )
    tile(
        b,
        "Weekly Nino 3.4",
        f"{float(status['latest_weekly_nino34_anomaly']):+.1f} C",
        f"week of {status['latest_week']}",
    )
    tile(
        c,
        "US winter outlook",
        f"{us['median']:+.1f}%",
        f"80% range {us['lo']:+.1f}% to {us['hi']:+.1f}%",
    )
    tile(
        d,
        "Chance above normal",
        f"{100 * us['prob_above']:.0f}%",
        f"effective sample {us['ess']:.0f} winters",
    )
    states = pick[(pick["unit"].str.len() == 2) & (pick["unit"] != "US")]
    states = states.rename(columns={"unit": "state"}).copy()
    states["text"] = [
        f"median {m:+.1f}%, 80% range {lo:+.1f} to {hi:+.1f}%, P(above) {100 * p:.0f}%"
        for m, lo, hi, p in zip(
            states["median"], states["lo"], states["hi"], states["prob_above"], strict=True
        )
    ]
    st.plotly_chart(us_map(states, "median", "% vs normal", "text"), width="stretch")
    regions = pick[~pick["unit"].str.len().eq(2) & (pick["unit"] != "US")]
    show = regions[["unit", "median", "lo", "hi", "prob_above"]].copy()
    show.columns = ["region", "median %", "low %", "high %", "P(above normal)"]
    st.dataframe(show.round(2), hide_index=True)
    left, right = st.columns(2)
    scen = table("outlook_scenario")
    scen = scen[scen["unit"] == "US"]
    fig = go.Figure(
        [
            go.Scatter(
                x=scen["scenario_djf"],
                y=scen["hi"],
                mode="lines",
                line={"width": 0},
                showlegend=False,
                hoverinfo="skip",
            ),
            go.Scatter(
                x=scen["scenario_djf"],
                y=scen["lo"],
                mode="lines",
                fill="tonexty",
                line={"width": 0},
                fillcolor="rgba(57,135,229,0.18)",
                name="80% range",
                hoverinfo="skip",
            ),
            go.Scatter(
                x=scen["scenario_djf"],
                y=scen["median"],
                mode="lines+markers",
                name="median",
                line={"color": BLUE, "width": 2},
                marker={"size": 8},
                hovertemplate="DJF index %{x}<br>median %{y:+.1f}%<extra></extra>",
            ),
        ]
    )
    fig.update_xaxes(title="if the DJF ENSO index reaches")
    fig.update_yaxes(title="US heating degree days vs normal (%)")
    left.plotly_chart(style(fig, 360, "Scenario: how strong the peak gets"), width="stretch")
    price = table("outlook_price")
    price = price[price["method"] == method][
        [
            "market",
            "region",
            "premium_median",
            "premium_lo",
            "premium_hi",
            "prob_premium_negative",
            "winters",
        ]
    ]
    price.columns = [
        "market",
        "region",
        "winter premium %",
        "low %",
        "high %",
        "P(below fall price)",
        "winters fit",
    ]
    right.markdown("**Winter price premium outlook** (DJF average over Sep-Oct average)")
    right.dataframe(price.round(2), hide_index=True)
    analog = table("outlook_analog")[["label", "us_hdd_pct"]]
    right.markdown("**DTW analog winters for 2026** and what US heating demand did")
    right.dataframe(
        analog.rename(columns={"label": "winter", "us_hdd_pct": "US HDD vs normal %"}).round(1),
        hide_index=True,
    )


def page_event() -> None:
    month = table("enso_month")
    month["date"] = pd.to_datetime(month["date"])
    fig = go.Figure(
        go.Scatter(
            x=month["date"],
            y=month["index"],
            mode="lines",
            line={"color": BLUE, "width": 1.5},
            name="ENSO index",
            hovertemplate="%{x|%b %Y}: %{y:.2f}<extra></extra>",
        )
    )
    for level, text in ((2.0, "very strong"), (-0.5, "")):
        fig.add_hline(
            y=level,
            line={"color": BASELINE, "dash": "dot"},
            annotation_text=text,
            annotation_font_color=MUTED,
        )
    fig.update_yaxes(title="RONI scale (C)")
    st.plotly_chart(
        style(fig, 360, "157 years of ENSO, HadISST before 1950 and RONI after"), width="stretch"
    )
    left, right = st.columns([3, 2])
    traj = []
    analog = table("analog")
    for year in [2026] + analog["year"].tolist():
        part = month[month["date"].dt.year == year]
        traj.append((year, part["date"].dt.month, part["index"]))
    fig = go.Figure()
    for year, months, values in traj[1:]:
        fig.add_trace(
            go.Scatter(
                x=months,
                y=values,
                mode="lines",
                name=str(year),
                line={"color": MUTED, "width": 1},
                hovertemplate=f"{year} month %{{x}}: %{{y:.2f}}<extra></extra>",
            )
        )
    year, months, values = traj[0]
    fig.add_trace(
        go.Scatter(
            x=months,
            y=values,
            mode="lines+markers",
            name="2026",
            line={"color": RED, "width": 3},
            marker={"size": 8},
        )
    )
    fig.update_xaxes(title="month", dtick=1)
    fig.update_yaxes(title="ENSO index")
    left.plotly_chart(
        style(fig, 380, "2026 so far (red) against its DTW analogs (gray)"), width="stretch"
    )
    show = analog[["rank", "label", "dtw", "euclidean", "predictor", "djf", "category"]]
    right.markdown(
        f"**Nearest past years by DTW** (pruned {int(analog['pruned'].iloc[0])} "
        "of the DTW runs with LB_Keogh)"
    )
    right.dataframe(show.round(2), hide_index=True)
    events = table("enso_event")
    el = events[events["phase"] == "el_nino"].sort_values("peak", ascending=False).head(12)
    el = el.assign(
        start=pd.to_datetime(el["start"]).dt.strftime("%Y-%m"),
        end=pd.to_datetime(el["end"]).dt.strftime("%Y-%m"),
        peak=el["peak"].round(2),
    )
    st.markdown("**Strongest El Nino events since 1870**")
    st.dataframe(
        el[["event_id", "start", "end", "peak", "months", "strength", "status"]], hide_index=True
    )


def page_teleconnections() -> None:
    fp = table("fingerprint")
    left, right = st.columns(2)
    var = left.selectbox(
        "Variable",
        ["tmp", "hdd", "cdd", "pcpn"],
        format_func={
            "tmp": "temperature",
            "hdd": "heating degree days",
            "cdd": "cooling degree days",
            "pcpn": "precipitation",
        }.get,
    )
    season = right.selectbox("Season", ["DJF", "MAM", "JJA", "SON"])
    pick = fp[
        (fp["variable"] == var)
        & (fp["season"] == season)
        & (fp["unit"].str.len() == 2)
        & (fp["unit"] != "DC")
        & (fp["unit"] != "US")
    ].rename(columns={"unit": "state"})
    pick = pick.copy()
    pick["text"] = [
        f"r {r:+.2f}, q {q:.3f}{' (significant)' if s else ''}"
        for r, q, s in zip(pick["r"], pick["q"], pick["significant"], strict=True)
    ]
    st.plotly_chart(us_map(pick, "r", "Spearman r", "text", limit=0.5), width="stretch")
    summary = table("teleconnection_summary")
    tele = table("teleconnection")
    a, b, c = st.columns(3)
    a.metric("Tests run", f"{len(tele):,}")
    b.metric("Discoveries at FDR q", f"{int(tele['significant'].sum()):,}")
    c.metric("Uncorrected p < 0.05", f"{int((tele['p'] < 0.05).sum()):,}")
    state = st.selectbox(
        "State for the month by lead grid",
        sorted(tele["state"].unique()),
        index=sorted(tele["state"].unique()).index("TX"),
    )
    grid = tele[(tele["state"] == state) & (tele["variable"] == var)]
    pivot = grid.pivot_table(index="lead", columns="month", values="r")
    sig = grid.pivot_table(index="lead", columns="month", values="significant")
    text = np.where(sig.to_numpy() > 0, "*", "")
    fig = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy(),
            x=pivot.columns,
            y=pivot.index,
            colorscale=DIVERGING,
            zmid=0,
            zmin=-0.5,
            zmax=0.5,
            text=text,
            texttemplate="%{text}",
            hovertemplate="month %{x}, lead %{y}: r %{z:.2f}<extra></extra>",
        )
    )
    fig.update_xaxes(title="target month", dtick=1)
    fig.update_yaxes(title="ENSO lead (months)", dtick=1)
    st.plotly_chart(
        style(fig, 330, f"{state}: ENSO link by month and lead (* = FDR discovery)"),
        width="stretch",
    )
    st.dataframe(summary, hide_index=True)


def page_regions() -> None:
    cl = table("cluster_state")
    score = table("cluster_score")
    left, right = st.columns([3, 2])
    names = cl.drop_duplicates("cluster").sort_values("cluster")
    fig = go.Figure()
    for i, row in enumerate(names.itertuples()):
        part = cl[cl["cluster"] == row.cluster]
        fig.add_trace(
            go.Choropleth(
                locations=part["state"],
                z=np.ones(len(part)),
                locationmode="USA-states",
                colorscale=[[0, SERIES[i % 3]], [1, SERIES[i % 3]]],
                showscale=False,
                name=row.name,
                marker_line_color=SURFACE,
                marker_line_width=2,
                hovertemplate="%{location}: " + row.name + "<extra></extra>",
            )
        )
    fig.update_geos(scope="usa", bgcolor=SURFACE, landcolor=GRID, subunitcolor=SURFACE)
    left.plotly_chart(style(fig, 400, "ENSO response regions (Ward clustering)"), width="stretch")
    right.markdown("**Clusters**")
    for i, row in enumerate(names.itertuples()):
        members = " ".join(sorted(cl.loc[cl["cluster"] == row.cluster, "state"]))
        swatch = (
            f"<span style='display:inline-block;width:12px;height:12px;border-radius:2px;"
            f"background:{SERIES[i % 3]};margin-right:6px'></span>"
        )
        right.markdown(f"{swatch}**{row.name}**: {members}", unsafe_allow_html=True)
    right.dataframe(score.round(3), hide_index=True)
    rules = table("rule")
    st.markdown("**Association rules mined with FP-growth** (permutation tested, BH FDR)")
    show = rules[
        [
            "antecedent",
            "consequent",
            "count",
            "antecedent_count",
            "confidence",
            "base_rate",
            "lift",
            "q",
            "winters",
        ]
    ]
    st.dataframe(show.round(3), hide_index=True)


def page_backtest() -> None:
    score = table("backtest_score")
    a, b, c = st.columns(3)
    level = a.radio(
        "Level", ["climate", "sales"], horizontal=True, format_func=lambda v: UNIT_LEVEL[v]
    )
    subsets = list(score["subset"].unique())
    subset = b.selectbox("Winters", subsets, format_func=lambda s: s.replace("_", " "))
    methods = [m for m in score["method"].unique() if m != "climatology"]
    method = c.selectbox(
        "Method", methods, index=methods.index("enso"), format_func=METHOD_NAME.get
    )
    pick = score[(score["level"] == level) & (score["subset"] == subset)]
    one = (
        pick[(pick["method"] == method) & (pick["unit"].str.len() == 2) & (pick["unit"] != "US")]
        .rename(columns={"unit": "state"})
        .copy()
    )
    one["text"] = [
        f"CRPSS {s:+.3f} (90% CI {lo:+.2f} to {hi:+.2f}), {n} winters"
        for s, lo, hi, n in zip(
            one["crpss"], one["crpss_lo"], one["crpss_hi"], one["winters"], strict=True
        )
    ]
    st.plotly_chart(
        us_map(one, "crpss", "skill vs climatology", "text", flip=True, limit=0.3), width="stretch"
    )
    wide = pick[~pick["unit"].str.len().eq(2) | (pick["unit"] == "US")]
    table_ = wide.pivot_table(index="unit", columns="method", values="crpss").round(3)
    table_.columns = [METHOD_NAME.get(c, c) for c in table_.columns]
    st.markdown("**CRPS skill vs climatology by region** (positive means the method helped)")
    st.dataframe(table_)


def page_prices() -> None:
    premium = table("price_premium")
    gas = premium[premium["market"] == "HENRYHUB"]
    left, right = st.columns(2)
    fig = go.Figure()
    for phase, color in PHASE_COLOR.items():
        part = gas[gas["phase"] == phase]
        fig.add_trace(
            go.Scatter(
                x=part["hdd_pct"],
                y=part["premium"],
                mode="markers",
                name=PHASE_NAME[phase],
                text=part["label"],
                marker={"color": color, "size": 10, "line": {"color": SURFACE, "width": 2}},
                hovertemplate="%{text}: HDD %{x:+.1f}%, premium %{y:+.1f}%<extra></extra>",
            )
        )
    fig.update_xaxes(title="US heating degree days vs normal (%)")
    fig.update_yaxes(title="Henry Hub winter premium (%)")
    left.plotly_chart(
        style(fig, 380, "Gas: colder winters, higher winter premium"), width="stretch"
    )
    summary = table("price_premium_summary")
    show = summary[
        [
            "market",
            "region",
            "winters",
            "slope_per_hdd_pct",
            "r",
            "p",
            "el_nino_mean",
            "neutral_mean",
            "la_nina_mean",
        ]
    ]
    right.markdown("**Winter premium vs heating demand, by market**")
    right.dataframe(show.round(3), hide_index=True)
    lag = table("hydro_lag")
    fig = go.Figure()
    for i, (driver, name) in enumerate(
        (("precip", "NW Oct-Mar precipitation"), ("djf", "DJF ENSO index"))
    ):
        part = lag[lag["driver"] == driver]
        fig.add_trace(
            go.Bar(
                x=part["month"],
                y=part["r"],
                name=name,
                marker_color=SERIES[i],
                hovertemplate="month %{x}: r %{y:.2f}<extra></extra>",
            )
        )
    fig.update_layout(barmode="group", bargap=0.3, barcornerradius=4)
    fig.update_xaxes(title="month of the following year", dtick=1)
    fig.update_yaxes(title="correlation r")
    st.plotly_chart(
        style(
            fig,
            360,
            "Mid-C minus Palo Verde spread: wet Northwest winters make Mid-C cheaper in spring",
        ),
        width="stretch",
    )


def page_data() -> None:
    quality = table("quality")
    quality["status"] = np.where(
        quality["passed"], "PASS", np.where(quality["severity"] == "warn", "WARN", "FAIL")
    )
    st.markdown("**Quality checks run at every build**")
    st.dataframe(quality[["status", "name", "detail"]], hide_index=True, width="stretch")
    left, right = st.columns(2)
    left.markdown("**ENSO splice calibration** (HadISST to RONI)")
    left.dataframe(table("enso_calibration").round(3), hide_index=True)
    right.markdown("**Run info**")
    right.dataframe(table("run_info"), hide_index=True)
    st.markdown("**Raw files and checksums**")
    st.dataframe(table("source_file"), hide_index=True)


def main() -> None:
    st.set_page_config(page_title="warmpool", layout="wide")
    if not DB.exists():
        st.error(f"No warehouse at {DB}. Run warmpool download and warmpool run first.")
        st.stop()
    st.title("warmpool")
    st.caption("El Nino and US power demand, from 1870 to the 2026-27 winter")
    tabs = st.tabs(
        [
            "Winter 2026-27",
            "The event",
            "Teleconnections",
            "Regions and rules",
            "Backtest",
            "Prices",
            "Data",
        ]
    )
    pages = [
        page_outlook,
        page_event,
        page_teleconnections,
        page_regions,
        page_backtest,
        page_prices,
        page_data,
    ]
    for tab, page in zip(tabs, pages, strict=True):
        with tab:
            page()


main()
