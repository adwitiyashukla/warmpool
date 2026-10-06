from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd
import streamlit as st

from warmpool import views

DB = Path(os.environ.get("WARMPOOL_DB", "data/gold/warmpool.duckdb"))


@st.cache_data(show_spinner=False)
def table(name: str) -> pd.DataFrame:
    with duckdb.connect(str(DB), read_only=True) as con:
        return con.execute(f"select * from {name}").df()


def chart(fig, where=st) -> None:
    where.plotly_chart(fig, width="stretch")


def page_outlook() -> None:
    status = table("outlook_status").set_index("key")["value"]
    out = table("outlook")
    level = st.radio(
        "Measure",
        ["sales", "climate"],
        horizontal=True,
        format_func=lambda v: f"DJF {views.UNIT_LEVEL[v]} vs normal",
    )
    methods = list(out["method"].unique())
    method = st.selectbox(
        "Forecast method",
        methods,
        index=methods.index("gated"),
        format_func=views.METHOD_NAME.get,
    )
    tiles, fig, regions = views.outlook(out, status, level, method)
    for col, (label, value, note) in zip(st.columns(4), tiles, strict=True):
        col.metric(label, value)
        col.caption(note)
    chart(fig)
    st.dataframe(regions, hide_index=True)
    left, right = st.columns(2)
    chart(views.scenario(table("outlook_scenario")), left)
    right.markdown("**Winter price premium outlook** (DJF average over Sep-Oct average)")
    right.dataframe(views.price_outlook(table("outlook_price"), method), hide_index=True)
    right.markdown("**DTW analog winters for 2026** and what US heating demand did")
    right.dataframe(views.analog_outlook(table("outlook_analog")), hide_index=True)


def page_event() -> None:
    month = table("enso_month")
    month["date"] = pd.to_datetime(month["date"])
    analog = table("analog")
    chart(views.enso_history(month))
    left, right = st.columns([3, 2])
    chart(views.trajectories(month, analog), left)
    title, show = views.analog_table(analog)
    right.markdown(f"**{title}**")
    right.dataframe(show, hide_index=True)
    st.markdown("**Strongest El Nino events since 1870**")
    st.dataframe(views.top_el_ninos(table("enso_event")), hide_index=True)


def page_teleconnections() -> None:
    left, right = st.columns(2)
    var = left.selectbox("Variable", list(views.VARIABLE_NAME), format_func=views.VARIABLE_NAME.get)
    season = right.selectbox("Season", views.SEASONS)
    chart(views.fingerprint(table("fingerprint"), var, season))
    tele = table("teleconnection")
    for col, (label, value) in zip(st.columns(3), views.field_counts(tele), strict=True):
        col.metric(label, value)
    states = sorted(tele["state"].unique())
    state = st.selectbox("State for the month by lead grid", states, index=states.index("TX"))
    chart(views.lead_grid(tele, state, var))
    st.dataframe(table("teleconnection_summary"), hide_index=True)


def page_regions() -> None:
    left, right = st.columns([3, 2])
    fig, groups = views.clusters(table("cluster_state"))
    chart(fig, left)
    right.markdown("**Clusters**")
    for color, name, members in groups:
        swatch = (
            f"<span style='display:inline-block;width:12px;height:12px;border-radius:2px;"
            f"background:{color};margin-right:6px'></span>"
        )
        right.markdown(f"{swatch}**{name}**: {members}", unsafe_allow_html=True)
    right.dataframe(views.rounded(table("cluster_score"), 3), hide_index=True)
    st.markdown("**Association rules mined with FP-growth** (permutation tested, BH FDR)")
    st.dataframe(views.rules_table(table("rule")), hide_index=True)


def page_backtest() -> None:
    score = table("backtest_score")
    a, b, c = st.columns(3)
    level = a.radio(
        "Level", ["climate", "sales"], horizontal=True, format_func=views.UNIT_LEVEL.get
    )
    subset = b.selectbox(
        "Winters", list(score["subset"].unique()), format_func=lambda s: s.replace("_", " ")
    )
    methods = views.backtest_methods(score, level)
    method = c.selectbox(
        "Method", methods, index=methods.index("enso"), format_func=views.METHOD_NAME.get
    )
    chart(views.backtest_map(score, level, subset, method))
    st.markdown("**CRPS skill vs climatology by region** (positive means the method helped)")
    st.dataframe(views.backtest_regions(score, level, subset))


def page_prices() -> None:
    left, right = st.columns(2)
    chart(views.gas_premium(table("price_premium")), left)
    right.markdown("**Winter premium vs heating demand, by market**")
    right.dataframe(views.premium_table(table("price_premium_summary")), hide_index=True)
    chart(views.hydro(table("hydro_lag")))


def page_data() -> None:
    st.markdown("**Quality checks run at every build**")
    st.dataframe(views.quality_table(table("quality")), hide_index=True, width="stretch")
    left, right = st.columns(2)
    left.markdown("**ENSO splice calibration** (HadISST to RONI)")
    left.dataframe(views.rounded(table("enso_calibration"), 3), hide_index=True)
    right.markdown("**Run info**")
    right.dataframe(table("run_info"), hide_index=True)
    st.markdown("**Raw files and checksums**")
    st.dataframe(table("source_file"), hide_index=True)


def main() -> None:
    st.set_page_config(page_title=views.TITLE, layout="wide")
    if not DB.exists():
        st.error(f"No warehouse at {DB}. Run warmpool download and warmpool run first.")
        st.stop()
    st.title(views.TITLE)
    st.caption(views.SUBTITLE)
    pages = [
        page_outlook,
        page_event,
        page_teleconnections,
        page_regions,
        page_backtest,
        page_prices,
        page_data,
    ]
    for tab, page in zip(st.tabs(views.TABS), pages, strict=True):
        with tab:
            page()


main()
