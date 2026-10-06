from __future__ import annotations

import hashlib
import http.client
import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from warmpool.config import Config, Download

MAGIC = {"xlsx": b"PK\x03\x04", "zip": b"PK\x03\x04", "xls": b"\xd0\xcf\x11\xe0"}
NO_RETRY = {400, 401, 403, 404, 410}
CHUNK = 1 << 20


class DownloadError(RuntimeError):
    pass


@dataclass(frozen=True)
class Target:
    source: str
    url: str
    name: str

    @property
    def relpath(self) -> str:
        return f"{self.source}/{self.name}"

    @property
    def kind(self) -> str:
        suffix = Path(self.name).suffix.lower().lstrip(".")
        return suffix if suffix in MAGIC else "text"


Opener = Callable[..., object]


def _request(url: str, settings: Download, offset: int = 0) -> urllib.request.Request:
    headers = {"User-Agent": settings.user_agent}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    return urllib.request.Request(url, headers=headers)


def fetch_text(url: str, settings: Download, opener: Opener = urllib.request.urlopen) -> str:
    last: Exception | None = None
    for attempt in range(settings.retries + 1):
        try:
            with opener(_request(url, settings), timeout=settings.timeout_s) as resp:
                return resp.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code in NO_RETRY:
                break
        except (http.client.HTTPException, OSError) as exc:
            last = exc
        time.sleep(settings.backoff_s * 2**attempt)
    raise DownloadError(f"{url}: {last}")


def check_payload(head: bytes, kind: str) -> None:
    if not head:
        raise DownloadError("empty response")
    if kind in MAGIC:
        if not head.startswith(MAGIC[kind]):
            raise DownloadError(f"not a {kind} file")
        return
    start = head[:512].lstrip().lower()
    if start.startswith((b"<!doctype html", b"<html")):
        raise DownloadError("got an html page instead of data")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(CHUNK):
            digest.update(block)
    return digest.hexdigest()


def download(
    target: Target, raw: Path, settings: Download, opener: Opener = urllib.request.urlopen
) -> dict:
    dest = raw / target.relpath
    part = dest.with_name(dest.name + ".part")
    dest.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    started = time.monotonic()
    for attempt in range(settings.retries + 1):
        offset = part.stat().st_size if part.exists() else 0
        try:
            with opener(_request(target.url, settings, offset), timeout=settings.timeout_s) as resp:
                if offset and resp.status != 206:
                    offset = 0
                length = resp.headers.get("Content-Length")
                with part.open("ab" if offset else "wb") as fh:
                    while block := resp.read(CHUNK):
                        fh.write(block)
            size = part.stat().st_size
            if length is not None and size != offset + int(length):
                raise DownloadError(
                    f"truncated transfer, got {size} of {offset + int(length)} bytes"
                )
            with part.open("rb") as fh:
                check_payload(fh.read(4096), target.kind)
            part.replace(dest)
            return {
                "url": target.url,
                "bytes": dest.stat().st_size,
                "sha256": sha256(dest),
                "fetched_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "seconds": round(time.monotonic() - started, 2),
            }
        except urllib.error.HTTPError as exc:
            last = exc
            if exc.code == 416:
                part.unlink(missing_ok=True)
            elif exc.code in NO_RETRY:
                break
        except DownloadError as exc:
            last = exc
            if "truncated" not in str(exc):
                part.unlink(missing_ok=True)
        except (http.client.HTTPException, OSError) as exc:
            last = exc
        time.sleep(settings.backoff_s * 2**attempt)
    part.unlink(missing_ok=True)
    raise DownloadError(f"{target.url}: {last}")


def wholesale_links(html: str, index: str, pattern: str) -> list[str]:
    found = re.findall(
        r'href="([^"]*' + re.escape(pattern) + r'[^"]*\.(?:xlsx|xls|zip))"',
        html,
        flags=re.IGNORECASE,
    )
    return sorted({urllib.parse.urljoin(index, link) for link in found})


def plan(cfg: Config, opener: Opener = urllib.request.urlopen) -> list[Target]:
    s = cfg.sources
    targets = [
        Target("enso", s.roni, "RONI.ascii.txt"),
        Target("enso", s.oni, "oni.ascii.txt"),
        Target("enso", s.weekly, "wksst9120.for"),
        Target("enso", s.nino34_long, "nino34.long.anom.data"),
        Target("retail", s.retail, "sales_revenue.xlsx"),
        Target("retail", s.retail_history, "HS861M_1990-2009.xlsx"),
        Target("gas", s.henry_hub, "DHHNGSP.csv"),
    ]
    stamp = fetch_text(s.climdiv_base + "procdate.txt", cfg.download, opener).strip()
    if not re.fullmatch(r"\d{8}", stamp):
        raise DownloadError(f"unexpected climdiv procdate {stamp!r}")
    for element in s.climdiv_elements:
        name = f"climdiv-{element}st-v1.0.0-{stamp}"
        targets.append(Target("climdiv", s.climdiv_base + name, name))
    html = fetch_text(s.wholesale_index, cfg.download, opener)
    links = wholesale_links(html, s.wholesale_index, s.wholesale_pattern)
    if not links:
        raise DownloadError("no wholesale price files found on the EIA index page")
    targets += [Target("wholesale", link, link.rsplit("/", 1)[-1]) for link in links]
    return targets


def load_manifest(raw: Path) -> dict:
    path = raw / "manifest.json"
    return json.loads(path.read_text()) if path.exists() else {}


def save_manifest(raw: Path, manifest: dict) -> None:
    path = raw / "manifest.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(dict(sorted(manifest.items())), indent=1))
    tmp.replace(path)


def run(
    cfg: Config,
    refresh: bool = False,
    only: tuple[str, ...] = (),
    opener: Opener = urllib.request.urlopen,
    log: Callable[[str], None] = print,
) -> dict:
    raw = cfg.path("raw")
    raw.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest(raw)
    targets = [t for t in plan(cfg, opener) if not only or t.source in only]
    todo = [t for t in targets if refresh or not (raw / t.relpath).exists()]
    log(
        f"{len(targets)} files planned, {len(targets) - len(todo)} already present, "
        f"{len(todo)} to fetch"
    )
    failures = []

    def job(target: Target) -> tuple[Target, dict | Exception]:
        try:
            return target, download(target, raw, cfg.download, opener)
        except DownloadError as exc:
            return target, exc

    with ThreadPoolExecutor(max_workers=cfg.download.workers) as pool:
        for target, result in pool.map(job, todo):
            if isinstance(result, Exception):
                failures.append(f"{target.relpath}: {result}")
                log(f"FAILED {target.relpath}: {result}")
            else:
                manifest[target.relpath] = result
                log(f"ok {target.relpath} {result['bytes']:,} bytes in {result['seconds']} s")
    save_manifest(raw, manifest)
    if failures:
        raise DownloadError(f"{len(failures)} downloads failed: " + "; ".join(failures))
    return manifest
