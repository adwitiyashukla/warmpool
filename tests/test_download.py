from __future__ import annotations

import hashlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from warmpool import download as dl
from warmpool.config import Download

PAYLOAD = bytes(range(256)) * 4000
XLSX = b"PK\x03\x04" + b"x" * 5000
SETTINGS = Download(timeout_s=5.0, retries=3, backoff_s=0.0, workers=2, user_agent="test")


class Handler(BaseHTTPRequestHandler):
    routes: dict = {}
    hits: dict = {}

    def log_message(self, *args):
        pass

    def do_GET(self):
        path = self.path
        Handler.hits[path] = Handler.hits.get(path, 0) + 1
        body, mode = Handler.routes.get(path, (None, "missing"))
        if mode == "missing":
            self.send_error(404)
            return
        start = 0
        rng = self.headers.get("Range")
        honour = mode != "ignore_range"
        if rng and honour:
            start = int(rng.split("=")[1].rstrip("-"))
            self.send_response(206)
        else:
            self.send_response(200)
        chunk = body[start:]
        self.send_header("Content-Length", str(len(chunk)))
        self.end_headers()
        if mode == "truncate_once" and Handler.hits[path] == 1:
            self.wfile.write(chunk[: len(chunk) // 3])
            self.wfile.flush()
            self.close_connection = True
            return
        self.wfile.write(chunk)


@pytest.fixture
def server():
    Handler.routes = {}
    Handler.hits = {}
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_plain_download_writes_file_and_manifest_entry(server, tmp_path):
    Handler.routes["/a.txt"] = (PAYLOAD, "ok")
    entry = dl.download(dl.Target("enso", f"{server}/a.txt", "a.txt"), tmp_path, SETTINGS)
    out = tmp_path / "enso" / "a.txt"
    assert out.read_bytes() == PAYLOAD
    assert entry["sha256"] == hashlib.sha256(PAYLOAD).hexdigest()
    assert entry["bytes"] == len(PAYLOAD)
    assert not (tmp_path / "enso" / "a.txt.part").exists()


def test_truncated_transfer_is_resumed_with_range(server, tmp_path):
    Handler.routes["/big.txt"] = (PAYLOAD, "truncate_once")
    dl.download(dl.Target("enso", f"{server}/big.txt", "big.txt"), tmp_path, SETTINGS)
    assert (tmp_path / "enso" / "big.txt").read_bytes() == PAYLOAD
    assert Handler.hits["/big.txt"] == 2


def test_server_ignoring_range_restarts_cleanly(server, tmp_path):
    Handler.routes["/c.txt"] = (PAYLOAD, "ignore_range")
    part = tmp_path / "enso" / "c.txt.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(PAYLOAD[:1000])
    dl.download(dl.Target("enso", f"{server}/c.txt", "c.txt"), tmp_path, SETTINGS)
    assert (tmp_path / "enso" / "c.txt").read_bytes() == PAYLOAD


def test_partial_file_is_resumed(server, tmp_path):
    Handler.routes["/d.txt"] = (PAYLOAD, "ok")
    part = tmp_path / "enso" / "d.txt.part"
    part.parent.mkdir(parents=True)
    part.write_bytes(PAYLOAD[:7777])
    dl.download(dl.Target("enso", f"{server}/d.txt", "d.txt"), tmp_path, SETTINGS)
    assert (tmp_path / "enso" / "d.txt").read_bytes() == PAYLOAD


def test_html_error_page_is_rejected_for_spreadsheets(server, tmp_path):
    Handler.routes["/x.xlsx"] = (b"<!DOCTYPE html><html>maintenance</html>", "ok")
    with pytest.raises(dl.DownloadError, match="not a xlsx"):
        dl.download(dl.Target("retail", f"{server}/x.xlsx", "x.xlsx"), tmp_path, SETTINGS)
    assert not list((tmp_path / "retail").iterdir())


def test_html_error_page_is_rejected_for_text(server, tmp_path):
    Handler.routes["/t.txt"] = (b"\n  <html><body>oops</body></html>", "ok")
    with pytest.raises(dl.DownloadError, match="html"):
        dl.download(dl.Target("enso", f"{server}/t.txt", "t.txt"), tmp_path, SETTINGS)


def test_not_found_is_not_retried(server, tmp_path):
    with pytest.raises(dl.DownloadError, match="404"):
        dl.download(dl.Target("enso", f"{server}/nope.txt", "nope.txt"), tmp_path, SETTINGS)
    assert Handler.hits["/nope.txt"] == 1


def test_valid_spreadsheet_passes_magic_check(server, tmp_path):
    Handler.routes["/s.xlsx"] = (XLSX, "ok")
    dl.download(dl.Target("retail", f"{server}/s.xlsx", "s.xlsx"), tmp_path, SETTINGS)
    assert (tmp_path / "retail" / "s.xlsx").read_bytes() == XLSX


def test_target_kind_follows_extension():
    assert dl.Target("w", "u", "a.xls").kind == "xls"
    assert dl.Target("w", "u", "a.XLSX").kind == "xlsx"
    assert dl.Target("w", "u", "b.zip").kind == "zip"
    assert dl.Target("c", "u", "climdiv-tmpcst-v1.0.0-20260904").kind == "text"


def test_wholesale_links_are_found_and_made_absolute():
    html = (
        '<a href="xls/archive/ice_electric-2014final.xls">a</a>'
        '<a href="xls/archive/ice_electric-2025final.xlsx">b</a>'
        '<a href="xls/ice_electric-2026.xlsx">c</a>'
        '<a href="xls/archive/ice_electric-historical.zip">d</a>'
        '<a href="other.xlsx">e</a><a href="xls/ice_electric-2026.xlsx">dup</a>'
    )
    links = dl.wholesale_links(html, "https://example.org/wholesale/", "ice_electric")
    assert links == [
        "https://example.org/wholesale/xls/archive/ice_electric-2014final.xls",
        "https://example.org/wholesale/xls/archive/ice_electric-2025final.xlsx",
        "https://example.org/wholesale/xls/archive/ice_electric-historical.zip",
        "https://example.org/wholesale/xls/ice_electric-2026.xlsx",
    ]


def test_run_plans_fetches_and_skips_present_files(server, tmp_path, make_config):
    files = {
        "/RONI.ascii.txt": b"SEAS   YR  ANOM\nDJF  1950 -1.19\n",
        "/oni.ascii.txt": b" SEAS  YR   TOTAL   ANOM\n  DJF 1950  25.01  -1.32\n",
        "/wksst9120.for": b" Weekly SST data\n",
        "/nino34.long.anom.data": b" 1870 1870\n 1870 " + b" -1.00" * 12 + b"\n",
        "/sales_revenue.xlsx": XLSX,
        "/HS861M%201990-2009.xlsx": XLSX,
        "/fred.csv": b"observation_date,DHHNGSP\n1997-01-07,3.82\n",
        "/climdiv/procdate.txt": b"20260904\n",
        "/climdiv/climdiv-tmpcst-v1.0.0-20260904": b"0010021895  44.00\n",
        "/wholesale/": b'<a href="xls/ice_electric-2026.xlsx">x</a>',
        "/wholesale/xls/ice_electric-2026.xlsx": XLSX,
    }
    for path, body in files.items():
        Handler.routes[path] = (body, "ok")
    cfg = make_config(
        tmp_path,
        sources={
            "roni": f"{server}/RONI.ascii.txt",
            "oni": f"{server}/oni.ascii.txt",
            "weekly": f"{server}/wksst9120.for",
            "nino34_long": f"{server}/nino34.long.anom.data",
            "climdiv_base": f"{server}/climdiv/",
            "climdiv_elements": ["tmpc"],
            "retail": f"{server}/sales_revenue.xlsx",
            "retail_history": f"{server}/HS861M%201990-2009.xlsx",
            "wholesale_index": f"{server}/wholesale/",
            "henry_hub": f"{server}/fred.csv",
        },
        download={"backoff_s": 0.0},
    )
    lines = []
    manifest = dl.run(cfg, log=lines.append)
    assert len(manifest) == 9
    assert "climdiv/climdiv-tmpcst-v1.0.0-20260904" in manifest
    saved = json.loads((tmp_path / "data" / "raw" / "manifest.json").read_text())
    assert saved == manifest
    dl.run(cfg, log=lines.append)
    assert "9 already present, 0 to fetch" in lines[-1]
    assert Handler.hits["/sales_revenue.xlsx"] == 1
    dl.run(cfg, refresh=True, only=("retail",), log=lines.append)
    assert Handler.hits["/sales_revenue.xlsx"] == 2
    assert Path(tmp_path / "data" / "raw" / "gas" / "DHHNGSP.csv").exists()
