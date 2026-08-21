#!/usr/bin/env python3
"""Read-only logger for Polymarket's real-time market websocket.

Connects to wss://ws-subscriptions-clob.polymarket.com/ws/market, subscribes
to a list of markets (condition IDs, slugs, or raw CLOB asset/token IDs), and
logs to CSV:
  - quotes.csv: timestamp, market, asset_id, best_bid, best_ask, bid_size,
    ask_size, spread  (written whenever the top of book changes)
  - trades.csv: timestamp, asset_id, price, size, side  (every last_trade_price
    event, i.e. every fill reported on the tape)

This script places NO orders and touches NO wallet — it only reads public
market data.

If data/markets_config.json (built by resolving Gamma events) is present,
this also cross-checks logical consistency between related markets on every
price update -- e.g. CPI print buckets summing to 1, or a Fed meeting's hike
probability never exceeding the yearly hike market's price. Violations are
tracked as episodes (open -> persists -> resolves) and logged to
violations.csv via pmm_data.market_graph / pmm_data.violation_tracker. This
is pure logic on observed prices -- still no orders, no wallet.

Usage:
    python scripts/realtime_market_logger.py --markets <condition_id_or_slug>[,...] --out-dir data/live
    python scripts/realtime_market_logger.py --assets <token_id>[,...] --out-dir data/live

Stop with Ctrl+C. CSVs are flushed continuously and are safe to tail while
the script runs. Pass --parquet to also write a parquet snapshot on exit
(and every --parquet-interval seconds while running).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pmm_data.csv_logger import CsvLogger  # noqa: E402
from pmm_data.resolver import flatten_token_ids, resolve_token_ids  # noqa: E402
from pmm_data.market_graph import (  # noqa: E402
    DEFAULT_CONFIG_PATH,
    asset_to_market_map,
    build_constraint_set,
    load_config,
)
from pmm_data.violation_tracker import ViolationTracker  # noqa: E402

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
PING_INTERVAL_SECONDS = 5
RECONNECT_BACKOFF = [1, 2, 5, 10, 30]

QUOTE_FIELDS = [
    "timestamp",
    "recv_time",
    "market",
    "asset_id",
    "best_bid",
    "best_ask",
    "bid_size",
    "ask_size",
    "spread",
]
TRADE_FIELDS = [
    "timestamp",
    "recv_time",
    "market",
    "asset_id",
    "price",
    "size",
    "side",
]
VIOLATION_FIELDS = [
    "start_time_iso",
    "end_time_iso",
    "duration_seconds",
    "constraint_name",
    "constraint_type",
    "market_ids",
    "num_observations",
    "start_magnitude",
    "max_magnitude",
    "mean_magnitude",
    "end_magnitude",
    "resolved",
]


class OrderBook:
    """Tracks top-of-book per asset from book snapshots + price_change deltas."""

    def __init__(self):
        self.bids: dict[str, dict[str, float]] = {}
        self.asks: dict[str, dict[str, float]] = {}
        self.market_for_asset: dict[str, str] = {}

    def apply_snapshot(self, asset_id: str, market: str, bids: list, asks: list):
        self.market_for_asset[asset_id] = market
        self.bids[asset_id] = {b["price"]: float(b["size"]) for b in bids}
        self.asks[asset_id] = {a["price"]: float(a["size"]) for a in asks}

    def apply_price_change(self, asset_id: str, price: str, size: str, side: str):
        book = self.bids if side == "BUY" else self.asks
        levels = book.setdefault(asset_id, {})
        size_f = float(size)
        if size_f == 0:
            levels.pop(price, None)
        else:
            levels[price] = size_f

    def top_of_book(self, asset_id: str):
        bids = self.bids.get(asset_id, {})
        asks = self.asks.get(asset_id, {})
        if not bids or not asks:
            return None
        best_bid = max(bids, key=lambda p: float(p))
        best_ask = min(asks, key=lambda p: float(p))
        return {
            "best_bid": best_bid,
            "best_ask": best_ask,
            "bid_size": bids[best_bid],
            "ask_size": asks[best_ask],
            "spread": round(float(best_ask) - float(best_bid), 6),
        }


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def violation_record_to_row(record: dict) -> dict:
    return {
        "start_time_iso": datetime.fromtimestamp(record["start_time"], tz=timezone.utc).isoformat(),
        "end_time_iso": datetime.fromtimestamp(record["end_time"], tz=timezone.utc).isoformat(),
        "duration_seconds": record["duration_seconds"],
        "constraint_name": record["constraint_name"],
        "constraint_type": record["constraint_type"],
        "market_ids": "|".join(record["market_ids"]),
        "num_observations": record["num_observations"],
        "start_magnitude": record["start_magnitude"],
        "max_magnitude": record["max_magnitude"],
        "mean_magnitude": round(record["mean_magnitude"], 6),
        "end_magnitude": record["end_magnitude"],
        "resolved": record["resolved"],
    }


async def ping_loop(ws):
    try:
        while True:
            await asyncio.sleep(PING_INTERVAL_SECONDS)
            await ws.send("PING")
    except (asyncio.CancelledError, websockets.exceptions.ConnectionClosed):
        pass


def handle_event(
    event: dict,
    book: OrderBook,
    quote_log: CsvLogger,
    trade_log: CsvLogger,
    asset_to_market: dict[str, str] | None = None,
    current_prices: dict[str, float] | None = None,
    tracker: ViolationTracker | None = None,
):
    event_type = event.get("event_type")
    recv_time = now_iso()

    def maybe_update_graph(asset_id: str, top: dict, event_ts_ms) -> bool:
        if asset_to_market is None or asset_id not in asset_to_market:
            return False
        market_id = asset_to_market[asset_id]
        current_prices[market_id] = (float(top["best_bid"]) + float(top["best_ask"])) / 2
        return True

    def event_ts_seconds(event_ts_ms) -> float:
        try:
            return float(event_ts_ms) / 1000.0
        except (TypeError, ValueError):
            return time.time()

    if event_type == "book":
        asset_id = event["asset_id"]
        market = event.get("market", "")
        book.apply_snapshot(asset_id, market, event.get("bids", []), event.get("asks", []))
        top = book.top_of_book(asset_id)
        if top:
            quote_log.write({
                "timestamp": event.get("timestamp"),
                "recv_time": recv_time,
                "market": market,
                "asset_id": asset_id,
                **top,
            })
            if maybe_update_graph(asset_id, top, event.get("timestamp")) and tracker is not None:
                tracker.update(current_prices, event_ts_seconds(event.get("timestamp")))

    elif event_type == "price_change":
        market = event.get("market", "")
        graph_touched = False
        for change in event.get("price_changes", []):
            asset_id = change["asset_id"]
            book.market_for_asset.setdefault(asset_id, market)
            book.apply_price_change(asset_id, change["price"], change["size"], change["side"])
            top = book.top_of_book(asset_id)
            if top:
                quote_log.write({
                    "timestamp": event.get("timestamp"),
                    "recv_time": recv_time,
                    "market": market,
                    "asset_id": asset_id,
                    **top,
                })
                graph_touched = maybe_update_graph(asset_id, top, event.get("timestamp")) or graph_touched
        if graph_touched and tracker is not None:
            tracker.update(current_prices, event_ts_seconds(event.get("timestamp")))

    elif event_type == "last_trade_price":
        asset_id = event.get("asset_id")
        trade_log.write({
            "timestamp": event.get("timestamp"),
            "recv_time": recv_time,
            "market": book.market_for_asset.get(asset_id, ""),
            "asset_id": asset_id,
            "price": event.get("price"),
            "size": event.get("size"),
            "side": event.get("side"),
        })

    elif event_type == "best_bid_ask":
        # Redundant with our own book tracking, but log it for cross-checking
        # if custom_feature_enabled surfaces it without asset_id context.
        pass


async def run(asset_ids: list[str], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    quote_log = CsvLogger(out_dir / "quotes.csv", QUOTE_FIELDS)
    trade_log = CsvLogger(out_dir / "trades.csv", TRADE_FIELDS)
    book = OrderBook()

    asset_to_market: dict[str, str] | None = None
    current_prices: dict[str, float] = {}
    tracker: ViolationTracker | None = None
    violation_log: CsvLogger | None = None
    if DEFAULT_CONFIG_PATH.exists():
        config = load_config()
        asset_to_market = asset_to_market_map(config)
        constraint_set = build_constraint_set(config)
        violation_log = CsvLogger(out_dir / "violations.csv", VIOLATION_FIELDS)
        tracker = ViolationTracker(
            constraint_set,
            on_episode_closed=lambda record: violation_log.write(violation_record_to_row(record)),
        )
        print(
            f"[{now_iso()}] constraint checking enabled: "
            f"{len(constraint_set.negative_risk_groups)} negative-risk groups, "
            f"{len(constraint_set.monotone_constraints)} monotone constraints"
        )
    else:
        print(f"[{now_iso()}] no {DEFAULT_CONFIG_PATH} found; constraint checking disabled")

    stop = asyncio.Event()

    def _stop(*_):
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _stop)

    backoff_idx = 0
    try:
        while not stop.is_set():
            try:
                async with websockets.connect(WS_URL, ping_interval=20) as ws:
                    backoff_idx = 0
                    sub_msg = {
                        "assets_ids": asset_ids,
                        "type": "market",
                        "custom_feature_enabled": True,
                    }
                    await ws.send(json.dumps(sub_msg))
                    print(f"[{now_iso()}] subscribed to {len(asset_ids)} asset(s)")

                    pinger = asyncio.create_task(ping_loop(ws))
                    try:
                        while not stop.is_set():
                            recv_task = asyncio.create_task(ws.recv())
                            stop_task = asyncio.create_task(stop.wait())
                            done, pending = await asyncio.wait(
                                {recv_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
                            )
                            for p in pending:
                                p.cancel()
                            if stop_task in done:
                                break
                            raw = recv_task.result()
                            if raw in ("PONG", "PING"):
                                continue
                            try:
                                payload = json.loads(raw)
                            except json.JSONDecodeError:
                                continue
                            events = payload if isinstance(payload, list) else [payload]
                            for event in events:
                                if isinstance(event, dict):
                                    handle_event(
                                        event, book, quote_log, trade_log,
                                        asset_to_market, current_prices, tracker,
                                    )
                    finally:
                        pinger.cancel()
            except (websockets.exceptions.ConnectionClosed, OSError) as exc:
                if stop.is_set():
                    break
                delay = RECONNECT_BACKOFF[min(backoff_idx, len(RECONNECT_BACKOFF) - 1)]
                backoff_idx += 1
                print(f"[{now_iso()}] connection error ({exc}); reconnecting in {delay}s")
                await asyncio.sleep(delay)
    finally:
        if tracker is not None:
            tracker.flush(time.time())
        if violation_log is not None:
            violation_log.close()
        quote_log.close()
        trade_log.close()
        print(f"[{now_iso()}] stopped, logs closed")


def maybe_write_parquet(out_dir: Path):
    import pandas as pd

    for name in ("quotes", "trades"):
        csv_path = out_dir / f"{name}.csv"
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            df.to_parquet(out_dir / f"{name}.parquet", index=False)
            print(f"wrote {out_dir / f'{name}.parquet'} ({len(df)} rows)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--markets", help="Comma-separated condition IDs and/or market slugs")
    parser.add_argument("--assets", help="Comma-separated CLOB asset/token IDs (bypasses resolution)")
    parser.add_argument("--out-dir", default="data/live", help="Directory for quotes.csv / trades.csv")
    parser.add_argument("--parquet", action="store_true", help="Also write parquet snapshots on exit")
    args = parser.parse_args()

    if not args.markets and not args.assets:
        parser.error("provide --markets (condition IDs/slugs) or --assets (raw token IDs)")

    asset_ids: list[str] = []
    if args.assets:
        asset_ids.extend(a.strip() for a in args.assets.split(",") if a.strip())
    if args.markets:
        idents = [m.strip() for m in args.markets.split(",") if m.strip()]
        resolved = resolve_token_ids(idents)
        for ident, tokens in resolved.items():
            print(f"resolved {ident} -> {tokens}")
        asset_ids.extend(flatten_token_ids(resolved))

    asset_ids = list(dict.fromkeys(asset_ids))  # de-dupe, preserve order
    out_dir = Path(args.out_dir)

    try:
        asyncio.run(run(asset_ids, out_dir))
    except KeyboardInterrupt:
        pass

    if args.parquet:
        maybe_write_parquet(out_dir)


if __name__ == "__main__":
    main()
