#!/usr/bin/env python3
"""Backtest baseline vs. graph-fair-value quoting on logged quotes.csv.

Replays the primary ("Yes") outcome quotes for every tracked market in
timestamp order, running both strategies against the identical historical
book, and reports fill rate / total simulated spread P&L / average P&L per
fill for each -- so you can see whether skewing toward the graph-implied
fair value during a constraint violation actually pays off, or whether it's
just giving up margin for nothing.

No orders are placed; this only replays already-logged data.

Usage:
    python scripts/run_backtest.py --live-dir data/live \
        --half-spread 0.01 --skew-aggressiveness 0.5 --taper-seconds 60 \
        --mark-horizon-seconds 30 --fill-base-rate 0.15 --seed 42
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pmm_data.backtest import BacktestConfig, run_backtest  # noqa: E402
from pmm_data.market_graph import asset_to_market_map, build_constraint_set, load_config  # noqa: E402


@dataclass
class Event:
    timestamp_s: float
    market_id: str
    best_bid: float
    best_ask: float


def load_events(quotes_path: Path, asset_to_market: dict[str, str]) -> list[Event]:
    df = pd.read_csv(quotes_path, usecols=["timestamp", "asset_id", "best_bid", "best_ask"])
    df["asset_id"] = df["asset_id"].astype(str)
    df = df[df["asset_id"].isin(asset_to_market)]
    df = df.sort_values("timestamp")
    df["market_id"] = df["asset_id"].map(asset_to_market)
    df["timestamp_s"] = df["timestamp"] / 1000.0
    return [
        Event(row.timestamp_s, row.market_id, row.best_bid, row.best_ask)
        for row in df.itertuples(index=False)
    ]


def print_report(results: dict, config: BacktestConfig):
    print("=" * 70)
    print("BACKTEST: baseline vs. graph-fair-value quoting")
    print("=" * 70)
    print(
        f"half_spread={config.half_spread}  skew_aggressiveness={config.skew_aggressiveness}  "
        f"taper_seconds={config.taper_seconds}  mark_horizon_seconds={config.mark_horizon_seconds}  "
        f"fill_base_rate={config.fill_base_rate}  seed={config.seed}"
    )
    print()
    header = f"{'strategy':<12} {'opportunities':>14} {'fills':>8} {'fill_rate':>10} {'unmarked':>9} {'total_pnl':>11} {'avg_pnl/fill':>13}"
    print(header)
    print("-" * len(header))
    for name, stats in results.items():
        print(
            f"{name:<12} {stats.quote_opportunities:>14} {stats.num_fills:>8} "
            f"{stats.fill_rate:>10.4f} {stats.num_unmarked:>9} "
            f"{stats.total_pnl:>11.4f} {stats.avg_pnl_per_fill:>13.5f}"
        )

    if results["baseline"].marked_fills and results["fair_value"].marked_fills:
        diff = results["fair_value"].avg_pnl_per_fill - results["baseline"].avg_pnl_per_fill
        print()
        print(f"fair_value avg P&L/fill - baseline avg P&L/fill = {diff:+.5f}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live-dir", default="data/live")
    parser.add_argument("--half-spread", type=float, default=0.01)
    parser.add_argument("--skew-aggressiveness", type=float, default=0.5,
                         help="0 = ignore fair value (identical to baseline), 1 = quote fully at fair value")
    parser.add_argument("--taper-seconds", type=float, default=60.0,
                         help="exponential time constant for skew decay once a violation resolves")
    parser.add_argument("--fill-base-rate", type=float, default=0.15,
                         help="fill probability per event when quoting exactly at top-of-book")
    parser.add_argument("--mark-horizon-seconds", type=float, default=30.0,
                         help="how far forward to mark a fill's P&L against that market's own future mid")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    live_dir = Path(args.live_dir)
    config_data = load_config()
    asset_to_market = asset_to_market_map(config_data)
    constraint_set = build_constraint_set(config_data)

    events = load_events(live_dir / "quotes.csv", asset_to_market)
    print(f"loaded {len(events)} quote events across {len(asset_to_market)} tracked markets")
    if not events:
        print("nothing to backtest yet")
        return

    config = BacktestConfig(
        half_spread=args.half_spread,
        skew_aggressiveness=args.skew_aggressiveness,
        taper_seconds=args.taper_seconds,
        fill_base_rate=args.fill_base_rate,
        mark_horizon_seconds=args.mark_horizon_seconds,
        seed=args.seed,
    )
    results = run_backtest(events, constraint_set, config)
    print_report(results, config)


if __name__ == "__main__":
    main()
