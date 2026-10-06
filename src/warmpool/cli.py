from __future__ import annotations

import argparse
import sys
import time

from warmpool.config import ConfigError, load


def _download(cfg, args) -> None:
    from warmpool import download

    download.run(cfg, refresh=args.refresh, only=tuple(args.only or ()))


def _build(cfg, args) -> None:
    from warmpool import warehouse

    warehouse.build(cfg)


def _analyze(cfg, args) -> None:
    from warmpool import pipeline

    pipeline.analyze(cfg)


def _run(cfg, args) -> None:
    _build(cfg, args)
    _analyze(cfg, args)


def _replica(cfg, args) -> None:
    import shutil
    from pathlib import Path

    from warmpool import replica

    root = Path(args.root).resolve()
    if root == cfg.root:
        raise SystemExit("replica --root must be a separate folder, not the project folder")
    replica.write_all(root / cfg.paths.raw, seed=args.seed)
    shutil.copy2(cfg.root / "config.toml", root / "config.toml")
    print(f"synthetic replica written to {root}, run it with --config {root / 'config.toml'}")


def _space(cfg, args) -> None:
    from warmpool import space

    out = space.build(cfg, cfg.root / args.out)
    print(f"static site for the Space written to {out}")


COMMANDS = {
    "download": _download,
    "build": _build,
    "analyze": _analyze,
    "run": _run,
    "replica": _replica,
    "space": _space,
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="warmpool")
    parser.add_argument("--config", default="config.toml")
    sub = parser.add_subparsers(dest="command", required=True)
    dl = sub.add_parser("download", help="fetch the raw NOAA, EIA and FRED files")
    dl.add_argument("--refresh", action="store_true", help="fetch again even if present")
    dl.add_argument("--only", nargs="+", choices=["enso", "climdiv", "retail", "wholesale", "gas"])
    sub.add_parser("build", help="parse raw files into silver parquet and the gold DuckDB")
    sub.add_parser("analyze", help="mining, backtest, prices and the winter outlook")
    sub.add_parser("run", help="build and analyze in one go")
    rep = sub.add_parser("replica", help="write a synthetic copy of every raw file")
    rep.add_argument("--root", default="build/rehearsal")
    rep.add_argument("--seed", type=int, default=2026)
    sp = sub.add_parser("space", help="export the dashboard as a static Hugging Face Space")
    sp.add_argument("--out", default="build/space")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load(args.config)
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    started = time.monotonic()
    COMMANDS[args.command](cfg, args)
    print(f"{args.command} finished in {time.monotonic() - started:.1f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
