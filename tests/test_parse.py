from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest
from openpyxl import Workbook

from warmpool import parse


def test_roni_and_oni_map_seasons_to_center_months(tmp_path):
    roni = tmp_path / "RONI.ascii.txt"
    roni.write_text("SEAS   YR  ANOM\nDJF  1950 -1.19\nNDJ  2015  2.20\n")
    frame = parse.parse_roni(roni)
    assert frame["date"].dt.strftime("%Y-%m").tolist() == ["1950-01", "2015-12"]
    assert frame["roni"].tolist() == [-1.19, 2.2]
    oni = tmp_path / "oni.ascii.txt"
    oni.write_text(" SEAS  YR   TOTAL   ANOM\n  JAS 2026  29.12   2.16\n")
    assert parse.parse_oni(oni).loc[0, "month"] == 8


def test_bad_enso_headers_and_lines_raise(tmp_path):
    bad = tmp_path / "x.txt"
    bad.write_text("<html>error</html>\n")
    with pytest.raises(parse.ParseError):
        parse.parse_roni(bad)
    bad.write_text("SEAS   YR  ANOM\nXYZ  1950 -1.19\n")
    with pytest.raises(parse.ParseError, match="bad line"):
        parse.parse_roni(bad)


def test_weekly_handles_glued_negative_anomalies(tmp_path):
    path = tmp_path / "wk.for"
    path.write_text(
        " Weekly SST data starts week centered on 2Sept1981\n\n"
        "                Nino1+2      Nino3        Nino34        Nino4\n"
        " Week          SST SSTA     SST SSTA     SST SSTA     SST SSTA\n"
        " 02SEP1981     20.6-0.1     24.8-0.1     26.5-0.2     28.3-0.3\n"
        " 23SEP2026     25.4 4.7     28.8 3.9     29.7 3.1     29.8 1.1\n"
    )
    frame = parse.parse_weekly(path)
    assert frame["nino34_anom"].tolist() == [-0.2, 3.1]
    assert frame["nino12_anom"].tolist() == [-0.1, 4.7]
    assert frame["week"].iloc[1] == pd.Timestamp("2026-09-23")


def test_psl_reads_years_and_drops_missing(tmp_path):
    path = tmp_path / "nino34.long.anom.data"
    rows = [
        " 1870 1871",
        " 1870 " + " ".join(["  0.50"] * 12),
        " 1871 " + " ".join(["  1.00"] * 6 + [" -99.99"] * 6),
        "  -99.99",
        "  NINA34",
    ]
    path.write_text("\n".join(rows) + "\n")
    frame = parse.parse_psl(path)
    assert len(frame) == 18
    assert frame["date"].max() == pd.Timestamp("1871-06-01")


def test_climdiv_skips_regions_and_sentinels(tmp_path):
    lines = []
    for i in range(len(parse.CLIMDIV_STATES)):
        values = ["  43.10"] * 8 + [" -99.90"] * 4
        lines.append(f"{i + 1:03d}0022026" + "".join(values))
    lines.insert(0, "0010021895" + "  43.10" * 12)
    lines.append("1100022026" + "  50.00" * 12)
    lines.append("2010022026" + "  50.00" * 12)
    path = tmp_path / "climdiv-tmpcst-v1.0.0-20260904"
    path.write_text("\n".join(lines) + "\n")
    frame = parse.parse_climdiv(path, "tmpc")
    assert frame["state"].nunique() == 48
    assert frame[frame["year"] == 2026]["month"].max() == 8
    assert frame.loc[frame["state"] == "AL", "year"].min() == 1895


def test_climdiv_rejects_wrong_element(tmp_path):
    path = tmp_path / "climdiv-hddcst-v1.0.0-20260904"
    path.write_text("0010021895" + "  43.10" * 12 + "\n")
    with pytest.raises(parse.ParseError, match="element"):
        parse.parse_climdiv(path, "hddc")


def test_latest_climdiv_file_wins(tmp_path):
    for stamp in ("20260704", "20260904", "20260804"):
        (tmp_path / f"climdiv-tmpcst-v1.0.0-{stamp}").write_text("")
    (tmp_path / "climdiv-tmpcst-v1.0.0-20261004.part").write_text("")
    assert parse.latest_climdiv(tmp_path, "tmpc").name.endswith("20260904")


def _retail_book(path, sheet, sectors, rows):
    book = Workbook()
    ws = book.active
    ws.title = sheet
    top, mid, head = (
        ["", None, None, None],
        ["", None, None, None],
        ["Year", "Month", "State", "Data Status"],
    )
    for sector in sectors:
        top += [sector, None, None, None]
        mid += ["Revenue", "Sales", "Customers", "Price"]
        head += ["Thousand Dollars", "Megawatthours", "Count", "Cents/kWh"]
    for row in (top, mid, head, *rows, ["a note row"]):
        ws.append(row)
    book.save(path)


def test_retail_reads_both_layouts_and_dots(tmp_path):
    old = tmp_path / "HS861M_1990-2009.xlsx"
    _retail_book(
        old,
        "Monthly",
        ["RESIDENTIAL", "OTHER", "TOTAL"],
        [
            [
                2009,
                12,
                "AL",
                "Final",
                276749,
                2818685,
                ".",
                9.82,
                ".",
                ".",
                ".",
                ".",
                603260,
                6964800,
                2493148,
                8.66,
            ]
        ],
    )
    new = tmp_path / "sales_revenue.xlsx"
    _retail_book(
        new,
        "Monthly-States",
        ["RESIDENTIAL", "TOTAL"],
        [
            [
                2026,
                7,
                "AL",
                "Preliminary",
                610756.42,
                3724741.7,
                2455894,
                16.4,
                1216533,
                9136510.8,
                2862915,
                13.32,
            ]
        ],
    )
    frame = parse.parse_retail([old, new])
    res = frame[frame["sector"] == "residential"].set_index("year")
    assert np.isnan(res.loc[2009, "customers"])
    assert res.loc[2026, "customers"] == 2455894
    assert set(frame["sector"]) == {"residential", "other", "total"}
    assert frame[frame["sector"] == "other"]["sales_mwh"].isna().all()


def test_retail_overlap_is_an_error(tmp_path):
    rows = [[2010, 1, "AL", "Final", 1, 2, 3, 4]]
    a, b = tmp_path / "a.xlsx", tmp_path / "b.xlsx"
    _retail_book(a, "Monthly", ["RESIDENTIAL"], rows)
    _retail_book(b, "Monthly-States", ["RESIDENTIAL"], rows)
    with pytest.raises(parse.ParseError, match="overlap"):
        parse.parse_retail([a, b])


@pytest.mark.parametrize(
    ("name", "hub"),
    [
        ("Mid Columbia Peak", "MIDC"),
        ("Mid C Peak", "MIDC"),
        ("Palo Verde", "PALOVERDE"),
        ("SP 15", "SP15"),
        ("SP-15 Gen DA LMP Peak", "SP15"),
        ("NP 15 EZ Gen DA LMP Peak", "NP15"),
        ("NEPOOL", "MASSHUB"),
        ("Nepool MH Da Lmp ", "MASSHUB"),
        ("PJM-Wh Real Time Peak", "PJMWEST"),
        ("Indiana Rt Peak", "INDIANA"),
        ("ERCOT North 345KV Peak", "ERCOTNORTH"),
        ("Cinergy", None),
    ],
)
def test_hub_names_are_canonical(name, hub):
    assert parse.canonical_hub(name) == hub


def _price_rows(names):
    rows = [
        [
            "Price Hub",
            "Trade Date",
            "Delivery Start Date",
            "Delivery \nEnd Date",
            "High Price $/MWh",
            "Low Price $/MWh",
            "Wtd Avg Price $/MWh",
            "Change",
            "Daily Volume MWh",
            "Number of Trades",
            "Number of Companies",
        ]
    ]
    for i, name in enumerate(names):
        day = pd.Timestamp("2013-01-02") + pd.Timedelta(days=i)
        rows.append(
            [
                name,
                day,
                day + pd.Timedelta(days=1),
                day + pd.Timedelta(days=1),
                31,
                29,
                30 + i,
                "na",
                800,
                2,
                3,
            ]
        )
    return rows


def _book_bytes(rows):
    book = Workbook()
    for row in rows:
        book.active.append(row)
    buffer = io.BytesIO()
    book.save(buffer)
    return buffer.getvalue()


def test_wholesale_reads_zip_members_and_dedupes(tmp_path):
    with zipfile.ZipFile(tmp_path / "ice_electric-historical.zip", "w") as zf:
        zf.writestr("MID-C Hub.xlsx", _book_bytes(_price_rows(["Mid Columbia Peak"] * 3)))
    annual = _book_bytes(_price_rows(["Mid C Peak", "Palo Verde Peak"]))
    (tmp_path / "ice_electric-2014final.xlsx").write_bytes(annual)
    frame = parse.parse_wholesale(tmp_path)
    assert sorted(frame["hub"].unique()) == ["MIDC", "PALOVERDE"]
    assert (frame["hub"] == "MIDC").sum() == 3
    assert not frame.duplicated(["hub", "delivery_start"]).any()


def test_unmapped_hub_is_an_error(tmp_path):
    (tmp_path / "ice_electric-2014final.xlsx").write_bytes(_book_bytes(_price_rows(["Cinergy"])))
    with pytest.raises(parse.ParseError, match="unmapped"):
        parse.parse_wholesale(tmp_path)


def test_henry_hub_drops_blanks(tmp_path):
    path = tmp_path / "DHHNGSP.csv"
    path.write_text("observation_date,DHHNGSP\n1997-01-07,3.82\n1997-01-08,\n1997-01-09,.\n")
    frame = parse.parse_henry_hub(path)
    assert frame["henry_hub"].tolist() == [3.82]
