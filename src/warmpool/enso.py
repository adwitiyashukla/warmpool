from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

STRENGTHS = ["weak", "moderate", "strong", "very_strong"]


@dataclass(frozen=True)
class Calibration:
    intercept: float
    slope: float
    r: float
    rmse: float
    n: int
    first: str
    last: str


def base_period(
    year: int, first: int, last_end: int, window: int, step: int, anchor: int
) -> tuple[int, int]:
    block = anchor + step * ((year - anchor) // step)
    start = block - window // 2
    end = start + window - 1
    if end > last_end:
        end = last_end
        start = end - window + 1
    if start < first:
        start = first
        end = start + window - 1
    return start, end


def sliding_anomaly(monthly: pd.Series, window: int, step: int, anchor: int) -> pd.Series:
    years = monthly.index.year
    months = monthly.index.month
    complete = [y for y in sorted(set(years)) if (years == y).sum() == 12]
    first, last = complete[0], complete[-1]
    last_end = (last // 10) * 10
    out = pd.Series(np.nan, index=monthly.index)
    periods = {y: base_period(y, first, last_end, window, step, anchor) for y in sorted(set(years))}
    for period in sorted(set(periods.values())):
        in_base = (years >= period[0]) & (years <= period[1])
        clim = monthly[in_base].groupby(months[in_base]).mean()
        target = np.isin(years, [y for y, p in periods.items() if p == period])
        out[target] = monthly[target].to_numpy() - clim.reindex(months[target]).to_numpy()
    return out


def running3(monthly: pd.Series) -> pd.Series:
    full = monthly.asfreq("MS")
    return full.rolling(3, center=True, min_periods=3).mean().dropna()


def harmonize(
    roni: pd.DataFrame, nino34: pd.DataFrame, window: int, step: int, anchor: int
) -> tuple[pd.DataFrame, Calibration]:
    long = nino34.set_index("date")["nino34"].asfreq("MS")
    long = running3(sliding_anomaly(long, window, step, anchor))
    official = roni.set_index("date")["roni"].asfreq("MS")
    both = pd.concat([official.rename("roni"), long.rename("long")], axis=1, sort=True).dropna()
    slope, intercept = np.polyfit(both["long"], both["roni"], 1)
    fitted = intercept + slope * both["long"]
    cal = Calibration(
        intercept=float(intercept),
        slope=float(slope),
        r=float(np.corrcoef(both["long"], both["roni"])[0, 1]),
        rmse=float(np.sqrt(np.mean((fitted - both["roni"]) ** 2))),
        n=len(both),
        first=str(both.index.min().date()),
        last=str(both.index.max().date()),
    )
    frame = pd.concat([official.rename("roni"), long.rename("long")], axis=1, sort=True)
    frame = frame[frame["roni"].notna() | frame["long"].notna()]
    frame["calibrated"] = intercept + slope * frame["long"]
    use_roni = frame["roni"].notna()
    frame["index"] = frame["roni"].where(use_roni, frame["calibrated"])
    frame["source"] = np.where(use_roni, "RONI", "HadISST")
    frame = frame.dropna(subset=["index"]).asfreq("MS")
    if frame["index"].isna().any():
        raise ValueError("harmonized ENSO index has gaps")
    frame.index.name = "date"
    return frame.reset_index(), cal


def strength(peak: float, edges: tuple[float, ...]) -> str:
    level = int(np.searchsorted(np.asarray(edges), abs(peak), side="right")) - 1
    return STRENGTHS[max(0, min(level, len(STRENGTHS) - 1))]


def detect_events(
    index: pd.Series, threshold: float, min_months: int, edges: tuple[float, ...]
) -> pd.DataFrame:
    values = index.to_numpy()
    dates = index.index
    rows = []
    for phase, sign in (("el_nino", 1.0), ("la_nina", -1.0)):
        hot = sign * values >= threshold
        i = 0
        while i < len(values):
            if not hot[i]:
                i += 1
                continue
            j = i
            while j + 1 < len(values) and hot[j + 1]:
                j += 1
            ongoing = j == len(values) - 1
            if j - i + 1 >= min_months or ongoing:
                run = values[i : j + 1]
                k = int(np.argmax(sign * run))
                rows.append(
                    {
                        "phase": phase,
                        "start": dates[i],
                        "end": dates[j],
                        "peak_date": dates[i + k],
                        "peak": float(run[k]),
                        "months": j - i + 1,
                        "intensity": float(np.abs(run).sum()),
                        "strength": strength(run[k], edges),
                        "status": "ongoing" if ongoing else "complete",
                    }
                )
            i = j + 1
    events = pd.DataFrame(rows).sort_values("start").reset_index(drop=True)
    events.insert(
        0,
        "event_id",
        [
            f"{'EN' if p == 'el_nino' else 'LN'}{s.year}"
            for p, s in zip(events["phase"], events["start"], strict=True)
        ],
    )
    return events


def winters(index: pd.Series, events: pd.DataFrame, predictor_month: int) -> pd.DataFrame:
    rows = []
    first, last = index.index.min(), index.index.max()
    for year in range(first.year, last.year + 1):
        jan = pd.Timestamp(year + 1, 1, 1)
        pred_date = pd.Timestamp(year, predictor_month, 1)
        if pred_date not in index.index:
            continue
        phase, label, event_id = "neutral", "neutral", ""
        when = jan if jan in index.index else pred_date
        for ev in events.itertuples():
            if ev.start <= when <= ev.end:
                phase, label, event_id = ev.phase, f"{ev.strength}_{ev.phase}", ev.event_id
        rows.append(
            {
                "winter": year,
                "label": f"{year}-{str(year + 1)[-2:]}",
                "predictor": float(index[pred_date]),
                "djf": float(index[jan]) if jan in index.index else np.nan,
                "phase": phase,
                "category": label,
                "event_id": event_id,
            }
        )
    return pd.DataFrame(rows)
