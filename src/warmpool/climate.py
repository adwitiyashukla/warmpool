from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from warmpool.stats import window_anomaly

VARIABLES = ("tmp", "hdd", "cdd", "pcpn")
SEASON_MONTHS = {"DJF": (12, 1, 2), "MAM": (3, 4, 5), "JJA": (6, 7, 8), "SON": (9, 10, 11)}


@dataclass
class Panel:
    years: np.ndarray
    units: list[str]
    values: dict[str, np.ndarray]

    def index(self, year: int) -> int:
        return int(year - self.years[0])

    def subset(self, units: list[str]) -> Panel:
        cols = [self.units.index(u) for u in units]
        return Panel(self.years, list(units), {k: v[:, :, cols] for k, v in self.values.items()})


def build_panel(unit_climate: pd.DataFrame, units: list[str], first_year: int) -> Panel:
    frame = unit_climate[unit_climate["unit"].isin(units) & (unit_climate["year"] >= first_year)]
    years = np.arange(first_year, int(frame["year"].max()) + 1)
    values = {}
    for var in VARIABLES:
        cube = np.full((years.size, 12, len(units)), np.nan)
        pos = {u: i for i, u in enumerate(units)}
        cube[
            frame["year"].to_numpy() - first_year,
            frame["month"].to_numpy() - 1,
            frame["unit"].map(pos).to_numpy(),
        ] = frame[var].to_numpy()
        values[var] = cube
    return Panel(years=years, units=list(units), values=values)


def anomalies(panel: Panel, var: str, half: int, known_year: int | None = None) -> np.ndarray:
    known = None if known_year is None else panel.index(known_year)
    return window_anomaly(panel.values[var], half, known)


def season_mean(cube: np.ndarray, years: np.ndarray, season: str) -> tuple[np.ndarray, np.ndarray]:
    months = SEASON_MONTHS[season]
    if season == "DJF":
        dec = cube[:-1, 11]
        jan = cube[1:, 0]
        feb = cube[1:, 1]
        return years[1:], (dec + jan + feb) / 3.0
    idx = [m - 1 for m in months]
    return years, cube[:, idx].mean(axis=1)


def winter_months(cube: np.ndarray, months: tuple[int, ...]) -> np.ndarray:
    parts = []
    for m in months:
        part = cube[:-1, m - 1] if m >= 7 else cube[1:, m - 1]
        parts.append(part)
    return np.stack(parts, axis=1)


def enso_series(enso_month: pd.DataFrame) -> pd.Series:
    series = enso_month.set_index("date")["index"].astype(float)
    series.index = pd.DatetimeIndex(series.index)
    return series.asfreq("MS")


def enso_at(series: pd.Series, years: np.ndarray, month: int) -> np.ndarray:
    out = np.full(years.size, np.nan)
    for i, year in enumerate(years):
        y, m = (year - 1, month + 12) if month < 1 else (year, month)
        key = pd.Timestamp(int(y), int(m), 1)
        if key in series.index:
            out[i] = series[key]
    return out
