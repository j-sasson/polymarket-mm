"""Event-driven backtest comparing three quoting strategies on the same
historical top-of-book series:

  - baseline: quote symmetrically around the raw book midpoint.
  - fair_value: quote around a price skewed toward the graph-implied fair
    value (pmm_data.fair_value) whenever this market is caught in an active
    constraint violation, tapering the skew back to 0 (pure midpoint) once
    the violation stops appearing.
  - fair_value_inventory: fair_value's skew, plus the SAME inventory-skew
    math the live OrderManager applies (pmm_data.trading.inventory) --
    leaning quotes away from an already-large position. This is the
    strategy that actually matches what would run live; fair_value alone
    isolates just the graph-fair-value effect for comparison. Note: only
    inventory's PRICE skew is modeled here, not its size-scaling -- fills in
    this backtest are unit-sized (fill-or-not), so there's no order size to
    scale down.

All three strategies see the exact same historical book at every timestep,
so any difference in outcome is attributable to the quoting logic, not the
data.

Fill model: since we only have top-of-book (not full L2 depth or a complete
tape), each strategy's resting quote at each book update gets an independent
per-side fill draw, with probability set by how aggressive that quote is
relative to the CURRENT best bid/ask (see `fill_probability`). Every fill is
treated as one unit of size -- this measures quote *placement* quality
(edge captured vs. the book), not position-sized dollar P&L.

P&L marking: a fill's P&L is NOT marked against the midpoint at the instant
of the fill. Marking a skewed bid against the mid it was skewed away from
would mechanically show a "loss" every time the strategy does exactly what
it's supposed to -- it says nothing about whether the skew was justified.
Instead, each fill is marked against that market's own midpoint
`mark_horizon_seconds` later, which is the actual question we care about:
did the market subsequently move toward the graph-implied fair value, making
the skewed fill pay off? Fills too close to the end of the available data to
reach their horizon are excluded from P&L stats (reported separately as
"unmarked") rather than silently approximated.
"""
from __future__ import annotations

import bisect
import math
import random
from dataclasses import dataclass, field

from pmm_data.constraints import ConstraintSet
from pmm_data.fair_value import graph_fair_value
from pmm_data.trading.inventory import InventoryTracker, inventory_skew


@dataclass
class BacktestConfig:
    half_spread: float = 0.01           # our quote sits this far from center on each side
    skew_aggressiveness: float = 0.5    # 0 = ignore fair value, 1 = quote fully at fair value
    taper_seconds: float = 60.0         # exponential time constant for skew decay after resolution
    fill_base_rate: float = 0.15        # fill probability per event when quoting exactly at top-of-book
    fill_prob_cap: float = 0.98
    mark_horizon_seconds: float = 30.0  # how far forward to mark a fill's P&L
    inventory_skew_strength: float = 0.5   # only affects the fair_value_inventory strategy
    inventory_max_position: float = 20.0   # in fill-units; position ratio is net_shares / this
    seed: int = 42


@dataclass
class Fill:
    market_id: str
    side: str  # "bid" | "ask"
    fill_time: float
    fill_price: float
    pnl: float | None = None  # filled in by _mark_fills; None if horizon unreachable


@dataclass
class StrategyStats:
    quote_opportunities: int = 0
    fills: list[Fill] = field(default_factory=list)

    @property
    def num_fills(self) -> int:
        return len(self.fills)

    @property
    def fill_rate(self) -> float:
        return self.num_fills / self.quote_opportunities if self.quote_opportunities else 0.0

    @property
    def marked_fills(self) -> list[Fill]:
        return [f for f in self.fills if f.pnl is not None]

    @property
    def num_unmarked(self) -> int:
        return self.num_fills - len(self.marked_fills)

    @property
    def total_pnl(self) -> float:
        return sum(f.pnl for f in self.marked_fills)

    @property
    def avg_pnl_per_fill(self) -> float:
        marked = self.marked_fills
        return sum(f.pnl for f in marked) / len(marked) if marked else 0.0


def fill_probability(edge: float, spread: float, base_rate: float, cap: float = 0.98) -> float:
    """Probability our resting quote gets hit this event.

    edge > 0 means our quote improves on (is more aggressive than) the
    current top-of-book on that side; edge = 0 means we're joining the
    existing best bid/ask; edge < 0 means we're sitting behind the book.
    Probability grows/decays exponentially with edge, scaled by the current
    spread (a wide spread means the same cent of edge is less significant).
    """
    scale = max(spread, 1e-4)
    exponent = min(edge / scale, 50.0)  # exp(50) already dwarfs any realistic cap; avoid OverflowError
    return min(base_rate * math.exp(exponent), cap)


def next_skew(prev_skew: float, dt: float, taper_seconds: float) -> float:
    """Exponential decay of an inactive skew back toward 0."""
    if taper_seconds <= 0:
        return 0.0
    decayed = prev_skew * math.exp(-dt / max(taper_seconds, 1e-6))
    return decayed if abs(decayed) > 1e-6 else 0.0


def _quote_and_fill(
    stats: StrategyStats,
    rng,
    config: BacktestConfig,
    center: float,
    best_bid: float,
    best_ask: float,
    spread: float,
    market_id: str,
    ts: float,
    on_fill=None,  # optional callback(side: "BUY"|"SELL") -- used to update inventory
):
    our_bid = max(0.0, min(1.0, center - config.half_spread))
    our_ask = max(0.0, min(1.0, center + config.half_spread))
    if our_bid >= our_ask:
        return  # degenerate quote (skew swamped the spread) -- no valid market to make

    stats.quote_opportunities += 1

    edge_bid = our_bid - best_bid
    if rng.random() < fill_probability(edge_bid, spread, config.fill_base_rate, config.fill_prob_cap):
        stats.fills.append(Fill(market_id, "bid", ts, our_bid))
        if on_fill:
            on_fill("BUY")

    edge_ask = best_ask - our_ask
    if rng.random() < fill_probability(edge_ask, spread, config.fill_base_rate, config.fill_prob_cap):
        stats.fills.append(Fill(market_id, "ask", ts, our_ask))
        if on_fill:
            on_fill("SELL")


def _mark_fills(stats: StrategyStats, mid_history: dict[str, tuple[list[float], list[float]]], horizon: float):
    for f in stats.fills:
        times, mids = mid_history[f.market_id]
        target = f.fill_time + horizon
        idx = bisect.bisect_left(times, target)
        if idx >= len(times):
            continue  # horizon extends past the data we have -- leave unmarked, don't guess
        future_mid = mids[idx]
        f.pnl = (future_mid - f.fill_price) if f.side == "bid" else (f.fill_price - future_mid)


def run_backtest(events, constraint_set: ConstraintSet, config: BacktestConfig, rng=None):
    """`events`: iterable of objects/rows with .timestamp_s, .market_id,
    .best_bid, .best_ask, sorted ascending by timestamp_s across ALL tracked
    markets (interleaved) -- required so constraint checks see a consistent
    cross-market snapshot at each step, same as the live tracker.

    Returns {"baseline": StrategyStats, "fair_value": StrategyStats,
    "fair_value_inventory": StrategyStats}.
    """
    rng = rng if rng is not None else random.Random(config.seed)
    current_prices: dict[str, float] = {}
    skew_state: dict[str, float] = {}
    last_update_ts: dict[str, float] = {}
    mid_history: dict[str, tuple[list[float], list[float]]] = {}
    stats = {"baseline": StrategyStats(), "fair_value": StrategyStats(), "fair_value_inventory": StrategyStats()}
    inventory = InventoryTracker()

    for row in events:
        ts = row.timestamp_s
        market_id = row.market_id
        best_bid, best_ask = row.best_bid, row.best_ask
        if best_bid is None or best_ask is None or best_ask <= best_bid:
            continue
        raw_mid = (best_bid + best_ask) / 2
        spread = best_ask - best_bid
        current_prices[market_id] = raw_mid

        times, mids = mid_history.setdefault(market_id, ([], []))
        times.append(ts)
        mids.append(raw_mid)

        violations = constraint_set.check(current_prices)
        active_markets = {m for v in violations for m in v.market_ids}

        dt = ts - last_update_ts.get(market_id, ts)
        last_update_ts[market_id] = ts

        if market_id in active_markets:
            fair_prices = graph_fair_value(current_prices, violations)
            skew_state[market_id] = config.skew_aggressiveness * (fair_prices[market_id] - raw_mid)
        else:
            skew_state[market_id] = next_skew(skew_state.get(market_id, 0.0), dt, config.taper_seconds)

        fair_value_center = raw_mid + skew_state[market_id]

        _quote_and_fill(stats["baseline"], rng, config, raw_mid, best_bid, best_ask, spread, market_id, ts)
        _quote_and_fill(stats["fair_value"], rng, config, fair_value_center, best_bid, best_ask, spread, market_id, ts)

        position_ratio = inventory.position_ratio(market_id, config.inventory_max_position)
        inventory_center = inventory_skew(
            fair_value_center, position_ratio, config.inventory_skew_strength, config.half_spread,
        )
        _quote_and_fill(
            stats["fair_value_inventory"], rng, config, inventory_center, best_bid, best_ask, spread, market_id, ts,
            on_fill=lambda side, mid=market_id: inventory.apply_fill(mid, side, 1.0),
        )

    for name in stats:
        _mark_fills(stats[name], mid_history, config.mark_horizon_seconds)

    return stats
