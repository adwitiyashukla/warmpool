from __future__ import annotations

import io
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

SEASONS = ["DJF", "JFM", "FMA", "MAM", "AMJ", "MJJ", "JJA", "JAS", "ASO", "SON", "OND", "NDJ"]
CENTER_MONTH = {season: i + 1 for i, season in enumerate(SEASONS)}

CLIMDIV_STATES = [
    "AL",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
]
CLIMDIV_CODE = {f"{i + 1:03d}": state for i, state in enumerate(CLIMDIV_STATES)}
ELEMENTS = {
    "tmpc": ("02", "tmp", -99.90),
    "pcpn": ("01", "pcpn", -9.99),
    "hddc": ("25", "hdd", -9999.0),
    "cddc": ("26", "cdd", -9999.0),
}

HUBS = [
    ("midc", "MIDC"),
    ("midcolumbia", "MIDC"),
    ("paloverde", "PALOVERDE"),
    ("sp15", "SP15"),
    ("np15", "NP15"),
    ("nepool", "MASSHUB"),
    ("masshub", "MASSHUB"),
    ("pjm", "PJMWEST"),
    ("indiana", "INDIANA"),
    ("ercotnorth", "ERCOTNORTH"),
]
WHOLESALE_COLUMNS = [
    ("price hub", "hub"),
    ("trade date", "trade_date"),
    ("delivery start", "delivery_start"),
    ("delivery end", "delivery_end"),
    ("high price", "high"),
    ("low price", "low"),
    ("wtd avg", "price"),
    ("daily volume", "volume_mwh"),
    ("number of trades", "trades"),
    ("number of counterparties", "counterparties"),
    ("number of companies", "counterparties"),
]
RETAIL_MEASURES = {
    "Revenue": "revenue_k",
    "Sales": "sales_mwh",
    "Customers": "customers",
    "Price": "price_cents",
}


class ParseError(ValueError):
    pass


def _lines(path: Path) -> list[str]:
    return Path(path).read_text(encoding="latin-1").splitlines()


def _season_frame(rows: list[tuple], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=columns)
    frame["month"] = frame["season"].map(CENTER_MONTH)
    frame["date"] = pd.to_datetime(dict(year=frame["year"], month=frame["month"], day=1))
    return frame


def parse_roni(path: Path) -> pd.DataFrame:
    lines = _lines(path)
    if not lines or lines[0].split()[:3] != ["SEAS", "YR", "ANOM"]:
        raise ParseError(f"{Path(path).name}: unexpected RONI header")
    rows = []
    for line in lines[1:]:
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 3 or parts[0] not in CENTER_MONTH:
            raise ParseError(f"{Path(path).name}: bad line {line!r}")
        rows.append((parts[0], int(parts[1]), float(parts[2])))
    return _season_frame(rows, ["season", "year", "roni"])


def parse_oni(path: Path) -> pd.DataFrame:
    lines = _lines(path)
    if not lines or lines[0].split()[:4] != ["SEAS", "YR", "TOTAL", "ANOM"]:
        raise ParseError(f"{Path(path).name}: unexpected ONI header")
    rows = []
    for line in lines[1:]:
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 4 or parts[0] not in CENTER_MONTH:
            raise ParseError(f"{Path(path).name}: bad line {line!r}")
        rows.append((parts[0], int(parts[1]), float(parts[2]), float(parts[3])))
    return _season_frame(rows, ["season", "year", "oni_total", "oni"])


def parse_weekly(path: Path) -> pd.DataFrame:
    rows = []
    for line in _lines(path):
        match = re.match(r"^\s*(\d{2}[A-Z]{3}\d{4})\s+(.*)$", line)
        if not match:
            continue
        numbers = re.findall(r"-?\d+\.\d+", match.group(2))
        if len(numbers) != 8:
            raise ParseError(f"{Path(path).name}: bad weekly line {line!r}")
        rows.append([pd.to_datetime(match.group(1), format="%d%b%Y")] + [float(x) for x in numbers])
    if not rows:
        raise ParseError(f"{Path(path).name}: no weekly rows")
    cols = ["week"]
    for region in ("nino12", "nino3", "nino34", "nino4"):
        cols += [f"{region}_sst", f"{region}_anom"]
    return pd.DataFrame(rows, columns=cols)


def parse_psl(path: Path) -> pd.DataFrame:
    lines = [line for line in _lines(path) if line.strip()]
    try:
        first, last = (int(x) for x in lines[0].split()[:2])
    except (ValueError, IndexError) as exc:
        raise ParseError(f"{Path(path).name}: bad PSL year range") from exc
    count = last - first + 1
    if len(lines) < count + 2:
        raise ParseError(f"{Path(path).name}: expected {count} year rows")
    missing = float(lines[count + 1].split()[0])
    rows = []
    for offset, line in enumerate(lines[1 : count + 1]):
        parts = line.split()
        year = int(parts[0])
        if year != first + offset or len(parts) != 13:
            raise ParseError(f"{Path(path).name}: bad row for {first + offset}")
        for month, value in enumerate(parts[1:], start=1):
            number = float(value)
            rows.append((year, month, np.nan if abs(number - missing) < 1e-6 else number))
    frame = pd.DataFrame(rows, columns=["year", "month", "nino34"])
    frame["date"] = pd.to_datetime(dict(year=frame["year"], month=frame["month"], day=1))
    return frame.dropna(subset=["nino34"]).reset_index(drop=True)


def parse_climdiv(path: Path, element: str) -> pd.DataFrame:
    code, name, sentinel = ELEMENTS[element]
    rows = []
    for line in _lines(path):
        parts = line.split()
        if not parts:
            continue
        key = parts[0]
        if len(key) != 10 or not key.isdigit() or len(parts) != 13:
            raise ParseError(f"{Path(path).name}: bad climdiv line {line[:40]!r}")
        if key[4:6] != code:
            raise ParseError(f"{Path(path).name}: element {key[4:6]} where {code} expected")
        state = CLIMDIV_CODE.get(key[:3])
        if state is None:
            continue
        year = int(key[6:10])
        for month, value in enumerate(parts[1:], start=1):
            number = float(value)
            if abs(number - sentinel) > 1e-6:
                rows.append((state, year, month, number))
    frame = pd.DataFrame(rows, columns=["state", "year", "month", name])
    if frame["state"].nunique() != len(CLIMDIV_STATES):
        raise ParseError(f"{Path(path).name}: expected 48 states, got {frame['state'].nunique()}")
    return frame


def latest_climdiv(folder: Path, element: str) -> Path:
    files = sorted(Path(folder).glob(f"climdiv-{element}st-v*-*"))
    files = [f for f in files if not f.name.endswith(".part")]
    if not files:
        raise ParseError(f"no climdiv {element} file in {folder}")
    return max(files, key=lambda f: f.name.rsplit("-", 1)[-1])


def parse_climate(folder: Path, elements: tuple[str, ...]) -> pd.DataFrame:
    frame = None
    for element in elements:
        part = parse_climdiv(latest_climdiv(folder, element), element)
        frame = (
            part if frame is None else frame.merge(part, on=["state", "year", "month"], how="outer")
        )
    return frame.sort_values(["state", "year", "month"]).reset_index(drop=True)


def _retail_sheet(path: Path) -> pd.DataFrame:
    book = pd.ExcelFile(path)
    names = [n for n in book.sheet_names if n.lower() in ("monthly-states", "monthly")]
    if not names:
        raise ParseError(f"{Path(path).name}: no monthly state sheet in {book.sheet_names}")
    sheet = book.parse(names[0], header=None, dtype=object)
    sectors = sheet.iloc[0].ffill()
    measures = sheet.iloc[1]
    head = [str(x).strip() for x in sheet.iloc[2, :4]]
    if head != ["Year", "Month", "State", "Data Status"]:
        raise ParseError(f"{Path(path).name}: unexpected header {head}")
    body = sheet.iloc[3:]
    body = body[pd.to_numeric(body.iloc[:, 0], errors="coerce").notna()]
    parts = []
    for sector in [s for s in sectors.iloc[4:].dropna().unique()]:
        cols = {
            RETAIL_MEASURES[str(measures[i]).strip()]: i
            for i in range(4, sheet.shape[1])
            if sectors[i] == sector and str(measures[i]).strip() in RETAIL_MEASURES
        }
        if len(cols) != 4:
            raise ParseError(f"{Path(path).name}: sector {sector} has columns {sorted(cols)}")
        part = pd.DataFrame(
            {
                "state": body.iloc[:, 2].astype(str).str.strip().to_numpy(),
                "year": pd.to_numeric(body.iloc[:, 0]).astype(int).to_numpy(),
                "month": pd.to_numeric(body.iloc[:, 1]).astype(int).to_numpy(),
                "status": body.iloc[:, 3].astype(str).str.strip().to_numpy(),
                "sector": str(sector).strip().lower(),
            }
        )
        for name, i in cols.items():
            part[name] = pd.to_numeric(
                body.iloc[:, i].replace(".", np.nan), errors="coerce"
            ).to_numpy()
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def parse_retail(paths: list[Path]) -> pd.DataFrame:
    frame = pd.concat([_retail_sheet(p) for p in paths], ignore_index=True)
    dupes = frame.duplicated(["state", "year", "month", "sector"]).sum()
    if dupes:
        raise ParseError(f"retail files overlap on {dupes} rows")
    return frame.sort_values(["sector", "state", "year", "month"]).reset_index(drop=True)


def canonical_hub(name: str) -> str | None:
    key = re.sub(r"[^a-z0-9]", "", str(name).lower())
    for prefix, hub in HUBS:
        if key.startswith(prefix) or prefix in key:
            return hub
    return None


def _wholesale_frame(raw: pd.DataFrame, source: str) -> pd.DataFrame:
    rename = {}
    for col in raw.columns:
        label = re.sub(r"\s+", " ", str(col)).strip().lower()
        for key, name in WHOLESALE_COLUMNS:
            if label.startswith(key) and name not in rename.values():
                rename[col] = name
                break
    missing = {"hub", "trade_date", "delivery_start", "delivery_end", "price"} - set(
        rename.values()
    )
    if missing:
        raise ParseError(f"{source}: missing columns {sorted(missing)}")
    frame = raw.rename(columns=rename)[list(rename.values())].copy()
    frame = frame[frame["hub"].notna()]
    frame["raw_hub"] = frame["hub"].astype(str).str.strip()
    frame["hub"] = frame["raw_hub"].map(canonical_hub)
    for col in ("trade_date", "delivery_start", "delivery_end"):
        frame[col] = pd.to_datetime(frame[col], errors="coerce")
    for col in ("high", "low", "price", "volume_mwh", "trades", "counterparties"):
        if col in frame:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")
        else:
            frame[col] = np.nan
    frame["source"] = source
    return frame.dropna(subset=["trade_date", "delivery_start", "price"])


def parse_wholesale(folder: Path) -> pd.DataFrame:
    parts = []
    for path in sorted(Path(folder).iterdir()):
        if path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                for member in archive.namelist():
                    if member.lower().endswith((".xls", ".xlsx")):
                        data = io.BytesIO(archive.read(member))
                        raw = pd.read_excel(data, sheet_name=0)
                        parts.append(_wholesale_frame(raw, f"{path.name}:{member}"))
        elif path.suffix.lower() in (".xls", ".xlsx"):
            parts.append(_wholesale_frame(pd.read_excel(path, sheet_name=0), path.name))
    if not parts:
        raise ParseError(f"no wholesale files in {folder}")
    frame = pd.concat(parts, ignore_index=True)
    unknown = sorted(frame.loc[frame["hub"].isna(), "raw_hub"].unique())
    if unknown:
        raise ParseError(f"unmapped wholesale hubs {unknown}")
    frame = frame.sort_values(["hub", "delivery_start", "trade_date"], kind="mergesort")
    frame = frame.drop_duplicates(["hub", "delivery_start"], keep="last")
    return frame.reset_index(drop=True)


def parse_henry_hub(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path, dtype=str)
    if raw.shape[1] != 2:
        raise ParseError(f"{Path(path).name}: expected two columns")
    frame = pd.DataFrame(
        {
            "date": pd.to_datetime(raw.iloc[:, 0], errors="coerce"),
            "henry_hub": pd.to_numeric(raw.iloc[:, 1].replace(".", np.nan), errors="coerce"),
        }
    )
    return frame.dropna().reset_index(drop=True)
