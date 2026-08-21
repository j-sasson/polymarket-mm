#!/usr/bin/env python3
"""Runs the backtest against logged data and dumps a JSON payload the HTML
report consumes: config, data window, per-strategy stats, equity curves, and
drawdown series. Kept separate from the HTML so the numbers are computed once
in plain Python (testable, inspectable) rather than recomputed in JS."""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pmm_data.backtest import BacktestConfig, run_backtest  # noqa: E402
from pmm_data.market_graph import asset_to_market_map, build_constraint_set, load_config  # noqa: E402
from pmm_data.performance import compute_performance  # noqa: E402
from scripts.run_backtest import load_events  # noqa: E402

MAX_PLOT_POINTS = 600


def downsample(points: list, max_points: int) -> list:
    """Uniform-stride downsample, but always keeps the first/last points and
    whichever points hit the series' min/max y-value -- so a stride that
    happens to skip the deepest drawdown or the peak doesn't misrepresent
    the chart relative to the stats computed from the full-resolution data."""
    if len(points) <= max_points:
        return points
    step = len(points) / max_points
    keep = {int(i * step) for i in range(max_points)}
    keep |= {0, len(points) - 1}
    keep.add(max(range(len(points)), key=lambda i: points[i][1]))
    keep.add(min(range(len(points)), key=lambda i: points[i][1]))
    return [points[i] for i in sorted(keep)]


STRATEGY_LABELS = {
    "baseline": "Baseline (symmetric around raw midpoint)",
    "fair_value": "Fair-value skew (no inventory)",
    "fair_value_inventory": "Fair-value skew + inventory management",
}


def main():
    live_dir = Path("data/live")
    graph_config = load_config()
    asset_to_market = asset_to_market_map(graph_config)
    constraint_set = build_constraint_set(graph_config)

    events = load_events(live_dir / "quotes.csv", asset_to_market)
    config = BacktestConfig()  # defaults, matching what's documented in the report
    results = run_backtest(events, constraint_set, config)

    t0 = events[0].timestamp_s
    t1 = events[-1].timestamp_s

    strategies = {}
    for name, stats in results.items():
        fills = [(f.fill_time, f.pnl) for f in stats.marked_fills]
        perf = compute_performance(fills, bucket_seconds=300.0)
        equity = [[round((t - t0) / 60, 2), round(v, 5)] for t, v in perf.equity_curve]
        peak = float("-inf")
        drawdown = []
        for t, v in equity:
            peak = max(peak, v)
            drawdown.append([t, round(v - peak, 5)])

        strategies[name] = {
            "label": STRATEGY_LABELS.get(name, name),
            "quote_opportunities": stats.quote_opportunities,
            "num_fills": stats.num_fills,
            "num_unmarked": stats.num_unmarked,
            "fill_rate": stats.fill_rate,
            "total_pnl": perf.total_pnl,
            "avg_pnl_per_fill": stats.avg_pnl_per_fill,
            "hit_rate": perf.hit_rate,
            "max_drawdown": perf.max_drawdown,
            "sharpe": perf.sharpe,
            "equity_curve": downsample(equity, MAX_PLOT_POINTS),
            "drawdown": downsample(drawdown, MAX_PLOT_POINTS),
        }

    payload = {
        "generated_at": t1,
        "window_minutes": round((t1 - t0) / 60, 1),
        "num_events": len(events),
        "num_markets": len({e.market_id for e in events}),
        "config": {
            "half_spread": config.half_spread,
            "skew_aggressiveness": config.skew_aggressiveness,
            "taper_seconds": config.taper_seconds,
            "fill_base_rate": config.fill_base_rate,
            "mark_horizon_seconds": config.mark_horizon_seconds,
            "inventory_skew_strength": config.inventory_skew_strength,
            "inventory_max_position": config.inventory_max_position,
            "seed": config.seed,
        },
        "strategies": strategies,
    }

    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/backtest_report_data.json")
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}")
    for name, s in strategies.items():
        print(f"  {name}: fills={s['num_fills']} total_pnl={s['total_pnl']:.4f} sharpe={s['sharpe']}")


if __name__ == "__main__":
    main()
