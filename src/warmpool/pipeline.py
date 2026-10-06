from __future__ import annotations

import hashlib
import platform
import time
from collections.abc import Callable
from datetime import UTC, datetime

import pandas as pd

from warmpool import __version__, backtest, load, mine, outlook, prices, warehouse
from warmpool.config import Config


def analyze(cfg: Config, log: Callable[[str], None] = print) -> dict[str, pd.DataFrame]:
    started = time.monotonic()
    ctx = mine.context(cfg)
    out = mine.run(cfg, ctx, log)
    unit_sales = warehouse.read_table(cfg, "unit_sales")
    out.update(backtest.run(cfg, ctx, unit_sales, log))
    model = load.SalesModel(cfg, ctx["panel"], unit_sales)
    out["load_model"] = load.coefficients(model, cfg.outlook.winter)
    price_month = warehouse.read_table(cfg, "price_month")
    gas_month = warehouse.read_table(cfg, "gas_month")
    premium = prices.premiums(cfg, price_month, gas_month, ctx["panel"], ctx["winters"])
    out["price_premium"] = premium
    out["price_premium_summary"] = prices.premium_summary(cfg, premium)
    out["hydro_lag"] = prices.hydro(cfg, price_month, ctx["panel"], ctx["winters"])
    out["hydro_season"] = prices.hydro_seasons(cfg, price_month, ctx["panel"], ctx["winters"])
    log("prices: winter premiums for " + ", ".join(out["price_premium_summary"]["market"]))
    out.update(outlook.run(cfg, ctx, unit_sales, warehouse.read_table(cfg, "enso_week"), premium))
    config_text = (
        (cfg.root / "config.toml").read_bytes() if (cfg.root / "config.toml").exists() else b""
    )
    out["run_info"] = pd.DataFrame(
        [
            {
                "version": __version__,
                "finished_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "seconds": round(time.monotonic() - started, 1),
                "python": platform.python_version(),
                "config_sha256": hashlib.sha256(config_text).hexdigest()[:12],
                "tables": len(out) + 1,
            }
        ]
    )
    warehouse.save(cfg, out)
    log(f"analysis: {len(out)} result tables saved to {warehouse.gold_path(cfg)}")
    return out
