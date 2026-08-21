#!/usr/bin/env python3
"""Pull historical trade data for Polymarket markets — read-only, no auth needed.

Uses the public data-api (https://data-api.polymarket.com/trades), which
takes condition IDs directly (unlike the realtime websocket, which needs
per-outcome CLOB token IDs). Slugs are resolved to condition IDs via Gamma.

Note: the endpoint caps `offset` at 10,000, so for a very high-volume market
this will not retrieve its full trade history beyond ~10,000+limit trades —
narrow the --start/--end window and re-run per window if you need more.

Usage:
    python scripts/historical_trades.py --markets <condition_id_or_slug>[,...] \
        --start 2024-01-01 --end 2024-02-01 --out-dir data/history
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DATA_API_BASE = "https://data-api.polymarket.com"
GAMMA_BASE = "https://gamma-api.polymarket.com"
PAGE_LIMIT = 500
MAX_OFFSET = 10_000
REQUEST_DELAY_SECONDS = 0.2

FIELDS = [
    "proxyWallet", "side", "asset", "conditionId", "size", "price",
    "timestamp", "title", "slug", "eventSlug", "outcome", "outcomeIndex",
    "transactionHash",
]


def parse_time(value: str) -> int:
    if value.isdigit():
        return int(value)
    dt = datetime.fromisoformat(value).replace(tzinfo=timezone.utc)
    return int(dt.timestamp())


def resolve_condition_id(identifier: str) -> str:
    if identifier.startswith("0x") and len(identifier) == 66:
        return identifier
    resp = requests.get(f"{GAMMA_BASE}/markets/slug/{identifier}", timeout=15)
    resp.raise_for_status()
    market = resp.json()
    cond_id = market.get("conditionId")
    if not cond_id:
        raise ValueError(f"Could not resolve condition ID for slug '{identifier}'")
    return cond_id


def fetch_trades(condition_id: str, start: int | None, end: int | None, taker_only: bool):
    offset = 0
    while offset <= MAX_OFFSET:
        params = {
            "market": condition_id,
            "limit": PAGE_LIMIT,
            "offset": offset,
            "takerOnly": str(taker_only).lower(),
        }
        if start is not None:
            params["start"] = start
        if end is not None:
            params["end"] = end

        resp = requests.get(f"{DATA_API_BASE}/trades", params=params, timeout=30)
        resp.raise_for_status()
        page = resp.json()
        if not page:
            break
        for trade in page:
            yield trade
        if len(page) < PAGE_LIMIT:
            break
        offset += PAGE_LIMIT
        time.sleep(REQUEST_DELAY_SECONDS)

    if offset > MAX_OFFSET:
        print(
            f"  warning: hit offset cap ({MAX_OFFSET}) for {condition_id}; "
            "narrow --start/--end to fetch the rest",
            file=sys.stderr,
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--markets", required=True, help="Comma-separated condition IDs and/or market slugs")
    parser.add_argument("--start", help="ISO date/time or epoch seconds (inclusive)")
    parser.add_argument("--end", help="ISO date/time or epoch seconds (exclusive)")
    parser.add_argument("--taker-only", action="store_true", default=False,
                         help="Only return taker-side rows (data-api default is true; this script defaults to false to capture both sides)")
    parser.add_argument("--out-dir", default="data/history")
    parser.add_argument("--parquet", action="store_true", help="Also write a parquet file per market")
    args = parser.parse_args()

    start = parse_time(args.start) if args.start else None
    end = parse_time(args.end) if args.end else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    idents = [m.strip() for m in args.markets.split(",") if m.strip()]

    for ident in idents:
        cond_id = resolve_condition_id(ident)
        print(f"fetching trades for {ident} ({cond_id})")

        trades = list(fetch_trades(cond_id, start, end, args.taker_only))
        print(f"  got {len(trades)} trades")
        if not trades:
            continue

        import pandas as pd

        df = pd.DataFrame(trades)
        cols = [c for c in FIELDS if c in df.columns] + [c for c in df.columns if c not in FIELDS]
        df = df[cols]

        csv_path = out_dir / f"{cond_id}.csv"
        df.to_csv(csv_path, index=False)
        print(f"  wrote {csv_path}")

        if args.parquet:
            parquet_path = out_dir / f"{cond_id}.parquet"
            df.to_parquet(parquet_path, index=False)
            print(f"  wrote {parquet_path}")


if __name__ == "__main__":
    main()
