from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

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
VARIABLE_NAME = {
    "tmp": "temperature",
    "hdd": "heating degree days",
    "cdd": "cooling degree days",
    "pcpn": "precipitation",
}
SEASONS = ["DJF", "MAM", "JJA", "SON"]
TITLE = "warmpool"
SUBTITLE = "El Nino and US power demand, from 1870 to the 2026-27 winter"
TABS = [
    "Winter 2026-27",
    "The event",
    "Teleconnections",
    "Regions and rules",
    "Backtest",
    "Prices",
    "Data",
]


def rounded(frame: pd.DataFrame, digits: int) -> pd.DataFrame:
    out = frame.copy()
    cols = out.select_dtypes("float").columns
    out[cols] = out[cols].round(digits) + 0.0
    return out


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
            z=frame[value].round(4),
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


def outlook(
    out: pd.DataFrame, status: pd.Series, level: str, method: str
) -> tuple[list[tuple[str, str, str]], go.Figure, pd.DataFrame]:
    pick = out[(out["level"] == level) & (out["method"] == method)]
    us = pick[pick["unit"] == "US"].iloc[0]
    tiles = [
        (
            "ENSO index now",
            str(status["latest_index"]),
            f"RONI, season centered {status['latest_season_center'][:7]}",
        ),
        (
            "Weekly Nino 3.4",
            f"{float(status['latest_weekly_nino34_anomaly']):+.1f} C",
            f"week of {status['latest_week']}",
        ),
        (
            "US winter outlook",
            f"{us['median']:+.1f}%",
            f"80% range {us['lo']:+.1f}% to {us['hi']:+.1f}%",
        ),
        (
            "Chance above normal",
            f"{100 * us['prob_above']:.0f}%",
            f"effective sample {us['ess']:.0f} winters",
        ),
    ]
    states = pick[(pick["unit"].str.len() == 2) & (pick["unit"] != "US")]
    states = states.rename(columns={"unit": "state"}).copy()
    states["text"] = [
        f"median {m:+.1f}%, 80% range {lo:+.1f} to {hi:+.1f}%, P(above) {100 * p:.0f}%"
        for m, lo, hi, p in zip(
            states["median"], states["lo"], states["hi"], states["prob_above"], strict=True
        )
    ]
    fig = us_map(states, "median", "% vs normal", "text")
    regions = pick[~pick["unit"].str.len().eq(2) & (pick["unit"] != "US")]
    show = regions[["unit", "median", "lo", "hi", "prob_above"]].copy()
    show.columns = ["region", "median %", "low %", "high %", "P(above normal)"]
    return tiles, fig, rounded(show, 2)


def scenario(scen: pd.DataFrame) -> go.Figure:
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
    return style(fig, 360, "Scenario: how strong the peak gets")


def price_outlook(price: pd.DataFrame, method: str) -> pd.DataFrame:
    show = price[price["method"] == method][
        [
            "market",
            "region",
            "premium_median",
            "premium_lo",
            "premium_hi",
            "prob_premium_negative",
            "winters",
        ]
    ].copy()
    show.columns = [
        "market",
        "region",
        "winter premium %",
        "low %",
        "high %",
        "P(below fall price)",
        "winters fit",
    ]
    return rounded(show, 2)


def analog_outlook(analog: pd.DataFrame) -> pd.DataFrame:
    show = analog[["label", "us_hdd_pct"]]
    return rounded(show.rename(columns={"label": "winter", "us_hdd_pct": "US HDD vs normal %"}), 1)


def enso_history(month: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Scatter(
            x=month["date"],
            y=month["index"].round(3),
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
    return style(fig, 360, "157 years of ENSO, HadISST before 1950 and RONI after")


def trajectories(month: pd.DataFrame, analog: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for year in analog["year"].tolist():
        part = month[month["date"].dt.year == year]
        fig.add_trace(
            go.Scatter(
                x=part["date"].dt.month,
                y=part["index"].round(3),
                mode="lines",
                name=str(year),
                line={"color": MUTED, "width": 1},
                hovertemplate=f"{year} month %{{x}}: %{{y:.2f}}<extra></extra>",
            )
        )
    part = month[month["date"].dt.year == 2026]
    fig.add_trace(
        go.Scatter(
            x=part["date"].dt.month,
            y=part["index"].round(3),
            mode="lines+markers",
            name="2026",
            line={"color": RED, "width": 3},
            marker={"size": 8},
        )
    )
    fig.update_xaxes(title="month", dtick=1)
    fig.update_yaxes(title="ENSO index")
    return style(fig, 380, "2026 so far (red) against its DTW analogs (gray)")


def analog_table(analog: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    show = analog[["rank", "label", "dtw", "euclidean", "predictor", "djf", "category"]]
    title = (
        f"Nearest past years by DTW (pruned {int(analog['pruned'].iloc[0])} "
        "of the DTW runs with LB_Keogh)"
    )
    return title, rounded(show, 2)


def top_el_ninos(events: pd.DataFrame) -> pd.DataFrame:
    el = events[events["phase"] == "el_nino"].sort_values("peak", ascending=False).head(12)
    el = el.assign(
        start=pd.to_datetime(el["start"]).dt.strftime("%Y-%m"),
        end=pd.to_datetime(el["end"]).dt.strftime("%Y-%m"),
        peak=el["peak"].round(2),
    )
    return el[["event_id", "start", "end", "peak", "months", "strength", "status"]]


def fingerprint(fp: pd.DataFrame, var: str, season: str) -> go.Figure:
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
    return us_map(pick, "r", "Spearman r", "text", limit=0.5)


def field_counts(tele: pd.DataFrame) -> list[tuple[str, str]]:
    return [
        ("Tests run", f"{len(tele):,}"),
        ("Discoveries at FDR q", f"{int(tele['significant'].sum()):,}"),
        ("Uncorrected p < 0.05", f"{int((tele['p'] < 0.05).sum()):,}"),
    ]


def lead_grid(tele: pd.DataFrame, state: str, var: str) -> go.Figure:
    grid = tele[(tele["state"] == state) & (tele["variable"] == var)]
    pivot = grid.pivot_table(index="lead", columns="month", values="r")
    sig = grid.pivot_table(index="lead", columns="month", values="significant")
    text = np.where(sig.to_numpy() > 0, "*", "")
    fig = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy().round(3),
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
    return style(fig, 330, f"{state}: ENSO link by month and lead (* = FDR discovery)")


def clusters(cl: pd.DataFrame) -> tuple[go.Figure, list[tuple[str, str, str]]]:
    names = cl.drop_duplicates("cluster").sort_values("cluster")
    fig = go.Figure()
    groups = []
    for i, row in enumerate(names.itertuples()):
        part = cl[cl["cluster"] == row.cluster]
        color = SERIES[i % 3]
        fig.add_trace(
            go.Choropleth(
                locations=part["state"],
                z=np.ones(len(part)),
                locationmode="USA-states",
                colorscale=[[0, color], [1, color]],
                showscale=False,
                name=row.name,
                marker_line_color=SURFACE,
                marker_line_width=2,
                hovertemplate="%{location}: " + row.name + "<extra></extra>",
            )
        )
        groups.append((color, row.name, " ".join(sorted(part["state"]))))
    fig.update_geos(scope="usa", bgcolor=SURFACE, landcolor=GRID, subunitcolor=SURFACE)
    return style(fig, 400, "ENSO response regions (Ward clustering)"), groups


def rules_table(rules: pd.DataFrame) -> pd.DataFrame:
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
    return rounded(show, 3)


def backtest_methods(score: pd.DataFrame, level: str) -> list[str]:
    part = score[score["level"] == level]
    return [m for m in part["method"].unique() if m != "climatology"]


def backtest_map(score: pd.DataFrame, level: str, subset: str, method: str) -> go.Figure:
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
    return us_map(one, "crpss", "skill vs climatology", "text", flip=True, limit=0.3)


def backtest_regions(score: pd.DataFrame, level: str, subset: str) -> pd.DataFrame:
    pick = score[(score["level"] == level) & (score["subset"] == subset)]
    wide = pick[~pick["unit"].str.len().eq(2) | (pick["unit"] == "US")]
    table = rounded(wide.pivot_table(index="unit", columns="method", values="crpss"), 3)
    table.columns = [METHOD_NAME.get(c, c) for c in table.columns]
    return table


def gas_premium(premium: pd.DataFrame) -> go.Figure:
    gas = premium[premium["market"] == "HENRYHUB"]
    fig = go.Figure()
    for phase, color in PHASE_COLOR.items():
        part = gas[gas["phase"] == phase]
        fig.add_trace(
            go.Scatter(
                x=part["hdd_pct"].round(3),
                y=part["premium"].round(3),
                mode="markers",
                name=PHASE_NAME[phase],
                text=part["label"],
                marker={"color": color, "size": 10, "line": {"color": SURFACE, "width": 2}},
                hovertemplate="%{text}: HDD %{x:+.1f}%, premium %{y:+.1f}%<extra></extra>",
            )
        )
    fig.update_xaxes(title="US heating degree days vs normal (%)")
    fig.update_yaxes(title="Henry Hub winter premium (%)")
    return style(fig, 380, "Gas: colder winters, higher winter premium")


def premium_table(summary: pd.DataFrame) -> pd.DataFrame:
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
    return rounded(show, 3)


def hydro(lag: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    for i, (driver, name) in enumerate(
        (("precip", "NW Oct-Mar precipitation"), ("djf", "DJF ENSO index"))
    ):
        part = lag[lag["driver"] == driver]
        fig.add_trace(
            go.Bar(
                x=part["month"],
                y=part["r"].round(3),
                name=name,
                marker_color=SERIES[i],
                hovertemplate="month %{x}: r %{y:.2f}<extra></extra>",
            )
        )
    fig.update_layout(barmode="group", bargap=0.3, barcornerradius=4)
    fig.update_xaxes(title="month of the following year", dtick=1)
    fig.update_yaxes(title="correlation r")
    return style(
        fig,
        360,
        "Mid-C minus Palo Verde spread: wet Northwest winters make Mid-C cheaper in spring",
    )


def quality_table(quality: pd.DataFrame) -> pd.DataFrame:
    quality = quality.copy()
    quality["status"] = np.where(
        quality["passed"], "PASS", np.where(quality["severity"] == "warn", "WARN", "FAIL")
    )
    return quality[["status", "name", "detail"]]
