"""Risk-adjusted performance metrics computed from a strategy's marked fills:
equity curve, drawdown, Sharpe ratio, and hit rate. Pure functions over a
list of (time, pnl) pairs -- no dependency on the backtest engine or any
file format, so these are testable against hand-built fill sequences.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class PerformanceReport:
    num_fills: int
    total_pnl: float
    hit_rate: float          # fraction of marked fills with pnl > 0
    max_drawdown: float      # most negative peak-to-trough dip in cumulative pnl (<=0)
    sharpe: float | None     # None if too few buckets to estimate a variance
    equity_curve: list[tuple[float, float]]   # (time, cumulative_pnl), sorted by time
    bucket_returns: list[tuple[float, float]]  # (bucket_start_time, pnl_in_bucket)


def equity_curve(fills: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """`fills`: list of (time, pnl). Returns cumulative pnl over time, sorted."""
    ordered = sorted(fills, key=lambda f: f[0])
    curve = []
    running = 0.0
    for t, pnl in ordered:
        running += pnl
        curve.append((t, running))
    return curve


def max_drawdown(curve: list[tuple[float, float]]) -> float:
    if not curve:
        return 0.0
    peak = curve[0][1]
    worst = 0.0
    for _, value in curve:
        peak = max(peak, value)
        worst = min(worst, value - peak)
    return worst


def bucket_pnl(fills: list[tuple[float, float]], bucket_seconds: float) -> list[tuple[float, float]]:
    """Sums pnl into fixed-width time buckets, filling empty buckets with 0
    so the resulting return series has uniform spacing (required for a
    meaningful Sharpe ratio -- skipping empty buckets would understate
    variance)."""
    if not fills:
        return []
    ordered = sorted(fills, key=lambda f: f[0])
    start = ordered[0][0]
    buckets: dict[int, float] = {}
    for t, pnl in ordered:
        idx = int((t - start) // bucket_seconds)
        buckets[idx] = buckets.get(idx, 0.0) + pnl
    last_idx = max(buckets)
    return [(start + i * bucket_seconds, buckets.get(i, 0.0)) for i in range(last_idx + 1)]


def sharpe_ratio(bucket_returns: list[float], periods_per_year: float) -> float | None:
    n = len(bucket_returns)
    if n < 2:
        return None
    mean = sum(bucket_returns) / n
    variance = sum((r - mean) ** 2 for r in bucket_returns) / (n - 1)
    stdev = math.sqrt(variance)
    if stdev == 0:
        return None
    return (mean / stdev) * math.sqrt(periods_per_year)


def compute_performance(
    fills: list[tuple[float, float]],
    bucket_seconds: float = 300.0,
    periods_per_year: float | None = None,
) -> PerformanceReport:
    """`fills`: list of (time_seconds, pnl) for a strategy's MARKED fills only.

    periods_per_year defaults to the annualization factor implied by
    bucket_seconds (e.g. 5-minute buckets -> ~105,120 periods/year), matching
    the standard "annualize whatever frequency you sampled at" convention.
    """
    if periods_per_year is None:
        periods_per_year = (365.25 * 24 * 3600) / bucket_seconds

    curve = equity_curve(fills)
    buckets = bucket_pnl(fills, bucket_seconds)
    bucket_values = [pnl for _, pnl in buckets]
    sharpe = sharpe_ratio(bucket_values, periods_per_year)

    total_pnl = curve[-1][1] if curve else 0.0
    num_fills = len(fills)
    hits = sum(1 for _, pnl in fills if pnl > 0)
    hit_rate = hits / num_fills if num_fills else 0.0

    return PerformanceReport(
        num_fills=num_fills,
        total_pnl=total_pnl,
        hit_rate=hit_rate,
        max_drawdown=max_drawdown(curve),
        sharpe=sharpe,
        equity_curve=curve,
        bucket_returns=buckets,
    )
