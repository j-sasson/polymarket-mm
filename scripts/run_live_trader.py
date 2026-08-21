#!/usr/bin/env python3
"""Live (or paper) market-making trader for the tracked CPI/Fed markets.

Quotes both sides of each configured market using the same graph-fair-value
logic validated in the backtest (steps 2-4), skewed further by inventory,
submitted as GTD orders that expire before that market's configured
catalyst. Every order, fill, and cancellation is logged with a timestamp.
A kill switch cancels everything on any error, disconnect, or position/
exposure limit breach.

SAFETY DEFAULTS:
  - Runs in DRY RUN by default: intended orders are logged, nothing is ever
    submitted, no wallet/credentials are touched or required.
  - Going live requires BOTH the --live flag AND typing "yes" at an
    interactive confirmation prompt naming the wallet address that will
    trade -- so a saved/scripted invocation can't silently go live.
  - Requires POLYMARKET_PRIVATE_KEY in the environment for --live; never
    accept a private key as a CLI argument (shell history, process list).
  - On any error or disconnect while live, the kill switch cancels every
    order this process placed and the process EXITS -- it does not
    reconnect and resume trading unattended.

Usage (dry run, no credentials needed):
    python scripts/run_live_trader.py --catalysts catalysts.json --limits limits.json

Usage (live -- real orders, real funds):
    export POLYMARKET_PRIVATE_KEY=...   # never as a CLI flag
    python scripts/run_live_trader.py --catalysts catalysts.json --limits limits.json --live

catalysts.json: {"<condition_id>": "2026-09-16T18:00:00Z", ...}
limits.json:    {"<condition_id>": {"max_position_shares": 50, "max_exposure_usd": 100, "max_order_size_shares": 10}, ...}
"""
from __future__ import annotations

import argparse
import asyncio
import json
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

import websockets

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pmm_data.market_graph import asset_to_market_map, build_constraint_set, load_config  # noqa: E402
from pmm_data.trading.inventory import InventoryTracker  # noqa: E402
from pmm_data.trading.logging import TradeLogger, now_iso  # noqa: E402
from pmm_data.trading.order_manager import OrderManager, TradingConfig  # noqa: E402
from pmm_data.trading.risk_limits import KillSwitch, MarketLimits, RiskLimits  # noqa: E402

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"
PING_INTERVAL_SECONDS = 5


def load_catalysts(path: Path) -> dict[str, datetime]:
    raw = json.loads(path.read_text())
    return {k: datetime.fromisoformat(v.replace("Z", "+00:00")) for k, v in raw.items()}


def load_risk_limits(path: Path) -> RiskLimits:
    raw = json.loads(path.read_text())
    per_market = {
        k: MarketLimits(
            max_position_shares=v["max_position_shares"],
            max_exposure_usd=v["max_exposure_usd"],
            max_order_size_shares=v["max_order_size_shares"],
        )
        for k, v in raw.items()
    }
    return RiskLimits(per_market=per_market)


def confirm_live_or_exit(client) -> None:
    address = client.get_address()
    print("=" * 70)
    print("LIVE TRADING CONFIRMATION")
    print(f"Wallet address: {address}")
    print("This will submit real orders with real funds using the requirements")
    print("(risk limits, catalysts, half-spread, etc.) you configured.")
    print("=" * 70)
    reply = input('Type "yes" to proceed live, anything else to abort: ')
    if reply.strip().lower() != "yes":
        print("aborted -- not going live")
        sys.exit(1)


async def ping_loop(ws):
    try:
        while True:
            await asyncio.sleep(PING_INTERVAL_SECONDS)
            await ws.send("PING")
    except (asyncio.CancelledError, websockets.exceptions.ConnectionClosed):
        pass


async def run(manager: OrderManager, asset_to_market: dict[str, str], asset_ids: list[str], dry_run: bool):
    stop = asyncio.Event()

    def _stop(*_):
        stop.set()

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(sig, _stop)

    try:
        async with websockets.connect(WS_URL, ping_interval=20) as ws:
            await ws.send(json.dumps({"assets_ids": asset_ids, "type": "market", "custom_feature_enabled": True}))
            print(f"[{now_iso()}] subscribed to {len(asset_ids)} asset(s), dry_run={dry_run}")

            pinger = asyncio.create_task(ping_loop(ws))
            try:
                while not stop.is_set():
                    recv_task = asyncio.create_task(ws.recv())
                    stop_task = asyncio.create_task(stop.wait())
                    done, pending = await asyncio.wait({recv_task, stop_task}, return_when=asyncio.FIRST_COMPLETED)
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
                    now_dt = datetime.now(timezone.utc)
                    for event in events:
                        if not isinstance(event, dict):
                            continue
                        _handle_event(event, manager, asset_to_market, now_dt)
            finally:
                pinger.cancel()
    except (websockets.exceptions.ConnectionClosed, OSError) as exc:
        manager.logger.log_event("websocket_disconnect", str(exc))
        if not manager.kill_switch.triggered:
            manager.kill_switch.trigger(f"websocket disconnect: {exc}", manager.all_open_order_ids())
        print(f"[{now_iso()}] disconnected ({exc}); kill switch triggered, exiting (not auto-reconnecting while live)")
        raise
    finally:
        if not manager.kill_switch.triggered and not dry_run:
            manager.kill_switch.trigger("process shutting down", manager.all_open_order_ids())
        manager.logger.close()
        print(f"[{now_iso()}] stopped")


def _handle_event(event: dict, manager: OrderManager, asset_to_market: dict[str, str], now_dt: datetime):
    event_type = event.get("event_type")
    now_ts = now_dt.timestamp()

    def top_of_book(bids, asks):
        if not bids or not asks:
            return None, None
        best_bid = max((float(b["price"]), b["price"]) for b in bids)[1]
        best_ask = min((float(a["price"]), a["price"]) for a in asks)[1]
        return float(best_bid), float(best_ask)

    if event_type == "book":
        asset_id = event.get("asset_id")
        if asset_id not in asset_to_market:
            return
        best_bid, best_ask = top_of_book(event.get("bids", []), event.get("asks", []))
        if best_bid is not None:
            manager.on_quote_update(asset_to_market[asset_id], asset_id, best_bid, best_ask, now_ts, now_dt)

    elif event_type == "price_change":
        # price_change gives best_bid/best_ask directly per changed asset
        for change in event.get("price_changes", []):
            asset_id = change.get("asset_id")
            if asset_id not in asset_to_market:
                continue
            best_bid, best_ask = change.get("best_bid"), change.get("best_ask")
            if best_bid is not None and best_ask is not None:
                manager.on_quote_update(
                    asset_to_market[asset_id], asset_id, float(best_bid), float(best_ask), now_ts, now_dt,
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--catalysts", required=True, help="JSON file: condition_id -> ISO8601 catalyst time")
    parser.add_argument("--limits", required=True, help="JSON file: condition_id -> {max_position_shares, max_exposure_usd, max_order_size_shares}")
    parser.add_argument("--out-dir", default="data/trading")
    parser.add_argument("--live", action="store_true", help="submit real orders (default: dry run / paper mode)")
    parser.add_argument("--half-spread", type=float, default=0.01)
    parser.add_argument("--skew-aggressiveness", type=float, default=0.5)
    parser.add_argument("--taper-seconds", type=float, default=60.0)
    parser.add_argument("--inventory-skew-strength", type=float, default=0.5)
    parser.add_argument("--base-quote-size", type=float, default=5.0)
    parser.add_argument("--desired-order-lifetime-seconds", type=float, default=300.0)
    parser.add_argument("--pre-catalyst-buffer-seconds", type=float, default=120.0)
    parser.add_argument("--min-requote-delta", type=float, default=0.005)
    args = parser.parse_args()

    dry_run = not args.live

    graph_config = load_config()
    asset_to_market = asset_to_market_map(graph_config)
    constraint_set = build_constraint_set(graph_config)
    catalysts = load_catalysts(Path(args.catalysts))
    risk_limits = load_risk_limits(Path(args.limits))
    inventory = InventoryTracker()
    logger = TradeLogger(Path(args.out_dir))
    trading_config = TradingConfig(
        half_spread=args.half_spread,
        skew_aggressiveness=args.skew_aggressiveness,
        taper_seconds=args.taper_seconds,
        inventory_skew_strength=args.inventory_skew_strength,
        base_quote_size=args.base_quote_size,
        desired_order_lifetime_seconds=args.desired_order_lifetime_seconds,
        pre_catalyst_buffer_seconds=args.pre_catalyst_buffer_seconds,
        min_requote_delta=args.min_requote_delta,
    )

    client = None
    if not dry_run:
        from pmm_data.trading.auth import authenticate, build_client

        client = build_client()
        client = authenticate(client)
        confirm_live_or_exit(client)

    kill_switch = KillSwitch(client, logger, dry_run=dry_run)
    manager = OrderManager(client, constraint_set, risk_limits, inventory, logger, kill_switch, catalysts, trading_config, dry_run)

    asset_ids = [a for a in asset_to_market if catalysts.get(asset_to_market[a]) is not None]
    if not asset_ids:
        print("no subscribable assets: check that --catalysts covers markets in data/markets_config.json")
        return

    try:
        asyncio.run(run(manager, asset_to_market, asset_ids, dry_run))
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
