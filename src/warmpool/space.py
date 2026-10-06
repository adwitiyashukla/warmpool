from __future__ import annotations

import shutil
from pathlib import Path

import duckdb

from warmpool import warehouse
from warmpool.config import Config

APP_TABLES = (
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

DOCKERFILE = """FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV WARMPOOL_DB=/app/warmpool.duckdb
EXPOSE 7860
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]
"""

REQUIREMENTS = """streamlit>=1.50
plotly>=5.20
duckdb>=1.0
pandas>=2.2
numpy>=1.26
"""

README = """---
title: warmpool
emoji: \U0001f30a
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
short_description: El Nino and US power demand, 1870 to the 2026-27 winter
---

Dashboard for the warmpool project: https://github.com/adwitiyashukla/warmpool
"""


def build(cfg: Config, out: Path, app: Path) -> Path:
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    (out / ".streamlit").mkdir(parents=True)
    shutil.copy2(app, out / "app.py")
    theme = app.parent.parent / ".streamlit" / "config.toml"
    if theme.exists():
        shutil.copy2(theme, out / ".streamlit" / "config.toml")
    (out / "Dockerfile").write_text(DOCKERFILE, newline="\n")
    (out / "requirements.txt").write_text(REQUIREMENTS, newline="\n")
    (out / "README.md").write_text(README, encoding="utf-8", newline="\n")
    target = out / "warmpool.duckdb"
    with duckdb.connect(str(target)) as con:
        con.execute(f"attach '{warehouse.gold_path(cfg).as_posix()}' as gold (read_only)")
        for name in APP_TABLES:
            con.execute(f"create table {name} as select * from gold.{name}")
        con.execute("detach gold")
    return out
