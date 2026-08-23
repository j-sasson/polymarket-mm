#!/usr/bin/env python3
"""One-shot snapshot poller, meant to be run repeatedly by a scheduler (e.g.
a GitHub Actions cron workflow) instead of a long-running websocket process.

Each invocation:
  1. Fetches current top-of-book for every tracked market via the public,
     unauthenticated CLOB REST API (GET /book) -- no websocket, no wallet.
  2. Appends a quotes.csv row per market.
  3. Runs the same constraint checker used by the live collector, with the
     violation tracker's open-episode state persisted to poll_state.json so
     episodes still span across separate invocations correctly. This is a
     midpoint-based statistical signal -- see violations.csv.
  3b. Separately runs pmm_data.executable_arbitrage against the real best
      bid/ask on each leg AND each token's real taker fee rate (GET
      /fee-rate) -- only fires when there's actual money on the table net
      of crossing the spread AND net of the fee you'd actually pay to do
      it. See arbitrage.csv.
  4. Fetches any new trades since the last poll (data-api, also public) and
     appends them to trades.csv.

This script does no git operations itself -- the calling workflow is
responsible for committing data/live/* afterward. It exits non-zero only on
a total failure (e.g. can't read config); a single market's book/trade fetch
failing is logged to stderr and skipped, not fatal, since one bad HTTP call
shouldn't lose the rest of the poll.

Trade-off vs. the websocket collector: resolution is capped at your polling
interval. A constraint violation that opens and resolves between two polls
is invisible, and every episode's duration/timing is quantized to however
often this runs -- there is no way around that without a persistent
connection somewhere.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pmm_data.csv_logger import CsvLogger  # noqa: E402
from pmm_data.executable_arbitrage import BookTop, check_executable_arbitrage  # noqa: E402
from pmm_data.market_graph import asset_to_market_map, build_constraint_set, load_config  # noqa: E402
from pmm_data.violation_tracker import ViolationTracker  # noqa: E402

CLOB_BOOK_URL = "https://clob.polymarket.com/book"
CLOB_FEE_RATE_URL = "https://clob.polymarket.com/fee-rate"
DATA_API_TRADES_URL = "https://data-api.polymarket.com/trades"
DEFAULT_TRADE_LOOKBACK_SECONDS = 1800  # first-ever run: how far back to backfill trades
DEFAULT_MIN_ARB_PROFIT = 0.0  # arbitrage's own net-of-fee floor now does the real filtering; this just excludes exact zero
FEE_RATE_FETCH_FAILURE_FALLBACK = 0.10  # conservative (highest real rate observed), NOT 0 -- a failed fetch must never
                                          # silently make a fee-losing trade look free

QUOTE_FIELDS = ["timestamp", "recv_time", "market", "asset_id", "best_bid", "best_ask", "bid_size", "ask_size", "spread"]
TRADE_FIELDS = ["timestamp", "recv_time", "market", "asset_id", "price", "size", "side"]
VIOLATION_FIELDS = [
    "start_time_iso", "end_time_iso", "duration_seconds", "constraint_name", "constraint_type",
    "market_ids", "num_observations", "start_magnitude", "max_magnitude", "mean_magnitude",
    "end_magnitude", "resolved",
]
ARBITRAGE_FIELDS = [
    "time_iso", "constraint_name", "constraint_type", "market_ids",
    "gross_profit_per_set", "fee_per_set", "profit_per_set", "max_size", "total_profit", "detail",
]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch_book(token_id: str) -> dict | None:
    resp = requests.get(CLOB_BOOK_URL, params={"token_id": token_id}, timeout=15)
    resp.raise_for_status()
    return resp.json()


def top_of_book(book: dict):
    bids, asks = book.get("bids", []), book.get("asks", [])
    if not bids or not asks:
        return None
    best_bid = max(bids, key=lambda b: float(b["price"]))
    best_ask = min(asks, key=lambda a: float(a["price"]))
    return best_bid["price"], best_bid["size"], best_ask["price"], best_ask["size"]


def fetch_taker_fee_rate(token_id: str) -> float:
    """GET /fee-rate returns {"base_fee": <bps>}. Observed to vary by
    market when checked against real data -- do not assume a fixed rate."""
    resp = requests.get(CLOB_FEE_RATE_URL, params={"token_id": token_id}, timeout=15)
    resp.raise_for_status()
    return resp.json().get("base_fee", 0) / 10000.0


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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="data/live")
    parser.add_argument("--min-arb-profit", type=float, default=DEFAULT_MIN_ARB_PROFIT,
                         help="minimum $ profit per complete set (before fees) to log as executable arbitrage")
    parser.add_argument("--min-arb-total-profit", type=float, default=0.0,
                         help="minimum total $ profit (per-set profit x depth-capped size) to log -- "
                              "0 logs everything that clears --min-arb-profit, however small in size")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    state_path = out_dir / "poll_state.json"

    config_data = load_config()
    asset_to_market = asset_to_market_map(config_data)
    constraint_set = build_constraint_set(config_data)

    quote_log = CsvLogger(out_dir / "quotes.csv", QUOTE_FIELDS)
    trade_log = CsvLogger(out_dir / "trades.csv", TRADE_FIELDS)
    violation_log = CsvLogger(out_dir / "violations.csv", VIOLATION_FIELDS)
    arbitrage_log = CsvLogger(out_dir / "arbitrage.csv", ARBITRAGE_FIELDS)

    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    tracker = ViolationTracker(
        constraint_set, on_episode_closed=lambda record: violation_log.write(violation_record_to_row(record)),
    )
    tracker.load_state(state.get("tracker_open_episodes", {}))
    last_trade_poll = state.get("last_trade_poll_epoch", int(time.time()) - DEFAULT_TRADE_LOOKBACK_SECONDS)

    now_ts = time.time()
    recv_time = now_iso()
    current_prices: dict[str, float] = {}
    books: dict[str, BookTop] = {}
    ok_count, err_count = 0, 0

    for token_id, market_id in asset_to_market.items():
        try:
            book = fetch_book(token_id)
            top = top_of_book(book)
            if top is None:
                continue
            best_bid, bid_size, best_ask, ask_size = top
            quote_log.write({
                "timestamp": book.get("timestamp", int(now_ts * 1000)),
                "recv_time": recv_time,
                "market": market_id,
                "asset_id": token_id,
                "best_bid": best_bid,
                "best_ask": best_ask,
                "bid_size": bid_size,
                "ask_size": ask_size,
                "spread": round(float(best_ask) - float(best_bid), 6),
            })
            current_prices[market_id] = (float(best_bid) + float(best_ask)) / 2
            try:
                fee_rate = fetch_taker_fee_rate(token_id)
            except Exception as fee_exc:
                print(f"warn: fee-rate fetch failed for asset {token_id} ({market_id}): {fee_exc}", file=sys.stderr)
                fee_rate = FEE_RATE_FETCH_FAILURE_FALLBACK
            books[market_id] = BookTop(
                float(best_bid), float(bid_size), float(best_ask), float(ask_size), taker_fee_rate=fee_rate,
            )
            ok_count += 1
        except Exception as exc:
            print(f"warn: book fetch failed for asset {token_id} ({market_id}): {exc}", file=sys.stderr)
            err_count += 1

    tracker.update(current_prices, now_ts)

    arbs = check_executable_arbitrage(
        constraint_set, books, min_profit=args.min_arb_profit, min_total_profit=args.min_arb_total_profit,
    )
    for arb in arbs:
        arbitrage_log.write({
            "time_iso": recv_time,
            "constraint_name": arb.constraint_name,
            "constraint_type": arb.constraint_type,
            "market_ids": "|".join(arb.market_ids),
            "gross_profit_per_set": arb.gross_profit_per_set,
            "fee_per_set": arb.fee_per_set,
            "profit_per_set": arb.profit_per_set,
            "max_size": arb.max_size,
            "total_profit": arb.total_profit,
            "detail": arb.detail,
        })

    trade_count = 0
    for market_id in set(asset_to_market.values()):
        try:
            resp = requests.get(
                DATA_API_TRADES_URL,
                params={"market": market_id, "start": int(last_trade_poll), "limit": 500, "takerOnly": "false"},
                timeout=20,
            )
            resp.raise_for_status()
            for trade in resp.json():
                trade_log.write({
                    "timestamp": trade.get("timestamp"),
                    "recv_time": recv_time,
                    "market": trade.get("conditionId", market_id),
                    "asset_id": trade.get("asset"),
                    "price": trade.get("price"),
                    "size": trade.get("size"),
                    "side": trade.get("side"),
                })
                trade_count += 1
        except Exception as exc:
            print(f"warn: trades fetch failed for market {market_id}: {exc}", file=sys.stderr)

    state = {
        "tracker_open_episodes": tracker.export_state(),
        "last_trade_poll_epoch": int(now_ts),
    }
    state_path.write_text(json.dumps(state, indent=2))

    quote_log.close()
    trade_log.close()
    violation_log.close()
    arbitrage_log.close()

    print(f"[{recv_time}] polled {ok_count} markets ok, {err_count} failed, {trade_count} new trades, "
          f"{len(tracker.export_state())} violation episode(s) currently open, {len(arbs)} executable arb(s) found")


if __name__ == "__main__":
    main()
