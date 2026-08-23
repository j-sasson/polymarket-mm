"""Detects EXECUTABLE arbitrage against real best bid/ask prices, NET OF
POLYMARKET'S TAKER FEE -- unlike pmm_data.constraints (which flags midpoint
inconsistencies as a monitoring signal, useful for tracking how often
markets look "off"), this only fires when there is real, tradeable, and
(as of this version) fee-adjusted money on the table.

Two negative-risk arbitrage directions:
  - mint_and_sell: mint the complete outcome set for $1 (Polymarket's
    split mechanism, not a CLOB trade -- no taker fee on this step), then
    sell every resulting token by crossing its best bid. Selling into an
    existing bid makes you the TAKER on all N legs, so N taker fees apply.
  - buy_the_set: buy every token by crossing its best ask (taker on all N
    legs), then redeem the winning outcome for $1 at resolution -- redemption
    is not a CLOB trade, so no fee on that step.

Monotonicity: if the subset's best bid exceeds the superset's best ask,
buying the superset at its ask while taking the opposite side of the subset
(economically: buying the subset's complementary NO token, since Polymarket
has no native short-selling) locks in the difference -- again taker fees on
both crossing legs.

Fee formula (docs.polymarket.com/trading/fees): for a leg traded at price p,
fee = size * fee_rate * p * (1 - p). Only takers pay; makers get rebates.
Capturing any of these arbs requires immediately crossing the resting book
to guarantee the risk-free lock-in, which makes you the taker on every leg
-- there's no way to structure this as a maker order and still call it
risk-free. `fee_rate` must be supplied per market in `BookTop` (query
GET /fee-rate per token_id; it varies by market, was NOT a fixed constant
when checked against real Polymarket data).

`profit_per_set` below is NET of this fee. `gross_profit_per_set` and
`fee_per_set` are broken out separately so a caller can see both. Still not
modeled: gas cost for the mint/split call itself (small on Polygon, but
non-zero), and slippage from walking past the top of book. `max_size` and
`total_profit` cap every opportunity at the THINNEST leg's depth.

None of this places an order or touches the NegRiskAdapter -- detection
only, same as pmm_data.constraints.
"""
from __future__ import annotations

from dataclasses import dataclass

from pmm_data.constraints import ConstraintSet


@dataclass(frozen=True)
class BookTop:
    best_bid: float
    bid_size: float
    best_ask: float
    ask_size: float
    taker_fee_rate: float = 0.0  # fraction (e.g. 0.10 for 10%), from GET /fee-rate for this token


@dataclass(frozen=True)
class ExecutableArbitrage:
    constraint_name: str
    constraint_type: str  # "mint_and_sell" | "buy_the_set" | "monotonicity"
    market_ids: tuple[str, ...]
    gross_profit_per_set: float  # $ before fees
    fee_per_set: float           # $ taker fees to cross every leg
    profit_per_set: float        # $ NET of fees -- what actually matters
    max_size: float              # complete sets tradeable, capped by the thinnest leg's depth
    total_profit: float          # profit_per_set * max_size, net of fees
    detail: str


def taker_fee(price: float, size: float, fee_rate: float) -> float:
    """fee = size * fee_rate * p * (1-p): symmetric, peaks at p=0.5, zero at
    the extremes. `size` is per-share here; pass 1.0 for a per-share rate."""
    return size * fee_rate * price * (1 - price)


def check_executable_arbitrage(
    constraint_set: ConstraintSet,
    books: dict[str, BookTop],
    min_profit: float = 0.0,
    min_total_profit: float = 0.0,
) -> list[ExecutableArbitrage]:
    results: list[ExecutableArbitrage] = []

    for group in constraint_set.negative_risk_groups:
        if any(m not in books for m in group.market_ids):
            continue

        bid_sum = sum(books[m].best_bid for m in group.market_ids)
        sell_fees = sum(taker_fee(books[m].best_bid, 1.0, books[m].taker_fee_rate) for m in group.market_ids)
        gross = round(bid_sum - 1.0, 6)
        net = round(gross - sell_fees, 6)
        if net > min_profit:
            max_size = min(books[m].bid_size for m in group.market_ids)
            total = round(net * max_size, 4)
            if total > min_total_profit:
                results.append(ExecutableArbitrage(
                    constraint_name=group.name,
                    constraint_type="mint_and_sell",
                    market_ids=group.market_ids,
                    gross_profit_per_set=gross,
                    fee_per_set=round(sell_fees, 6),
                    profit_per_set=net,
                    max_size=max_size,
                    total_profit=total,
                    detail=f"sum(best_bid)={bid_sum:.4f}: mint for $1, sell every leg at its bid "
                           f"(gross {gross:.4f}, taker fees {sell_fees:.4f}, net {net:.4f}/set, "
                           f"capped at {max_size:.2f} sets)",
                ))

        ask_sum = sum(books[m].best_ask for m in group.market_ids)
        buy_fees = sum(taker_fee(books[m].best_ask, 1.0, books[m].taker_fee_rate) for m in group.market_ids)
        gross = round(1.0 - ask_sum, 6)
        net = round(gross - buy_fees, 6)
        if net > min_profit:
            max_size = min(books[m].ask_size for m in group.market_ids)
            total = round(net * max_size, 4)
            if total > min_total_profit:
                results.append(ExecutableArbitrage(
                    constraint_name=group.name,
                    constraint_type="buy_the_set",
                    market_ids=group.market_ids,
                    gross_profit_per_set=gross,
                    fee_per_set=round(buy_fees, 6),
                    profit_per_set=net,
                    max_size=max_size,
                    total_profit=total,
                    detail=f"sum(best_ask)={ask_sum:.4f}: buy every leg at its ask, redeem for $1 "
                           f"(gross {gross:.4f}, taker fees {buy_fees:.4f}, net {net:.4f}/set, "
                           f"capped at {max_size:.2f} sets)",
                ))

    for mono in constraint_set.monotone_constraints:
        if mono.superset_id not in books or mono.subset_id not in books:
            continue
        superset, subset = books[mono.superset_id], books[mono.subset_id]
        gross = round(subset.best_bid - superset.best_ask, 6)
        fees = (
            taker_fee(superset.best_ask, 1.0, superset.taker_fee_rate)
            + taker_fee(subset.best_bid, 1.0, subset.taker_fee_rate)  # approximates the NO-leg fee
        )
        net = round(gross - fees, 6)
        if net > min_profit:
            max_size = min(subset.bid_size, superset.ask_size)
            total = round(net * max_size, 4)
            if total > min_total_profit:
                results.append(ExecutableArbitrage(
                    constraint_name=mono.name,
                    constraint_type="monotonicity",
                    market_ids=(mono.superset_id, mono.subset_id),
                    gross_profit_per_set=gross,
                    fee_per_set=round(fees, 6),
                    profit_per_set=net,
                    max_size=max_size,
                    total_profit=total,
                    detail=(
                        f"subset best_bid={subset.best_bid:.4f} > superset best_ask={superset.best_ask:.4f}: "
                        f"buy superset, take opposite side of subset "
                        f"(gross {gross:.4f}, taker fees ~{fees:.4f}, net {net:.4f}/set, "
                        f"capped at {max_size:.2f} sets)"
                    ),
                ))

    return results
