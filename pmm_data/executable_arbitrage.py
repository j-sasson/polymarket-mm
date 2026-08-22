"""Detects EXECUTABLE arbitrage against real best bid/ask prices -- unlike
pmm_data.constraints (which flags midpoint inconsistencies as a monitoring
signal, useful for tracking how often markets look "off"), this only fires
when there is real, tradeable money on the table net of crossing the spread.

Two negative-risk arbitrage directions:
  - mint_and_sell: mint the complete outcome set for $1 (Polymarket's
    split mechanism), sell every resulting token at its best bid.
    Profitable when sum(best_bid) > 1.
  - buy_the_set: buy every token at its best ask; exactly one outcome
    resolves, paying $1. Profitable when sum(best_ask) < 1.

Monotonicity: if the subset's best bid exceeds the superset's best ask,
buying the superset at its ask while taking the opposite side of the subset
(economically: buying the subset's complementary NO token, since Polymarket
has no native short-selling) locks in the difference risk-free.

A price crossing alone doesn't tell you the trade is worth anything -- the
best bid/ask might only be a few dollars deep. `max_size` and `total_profit`
cap every opportunity at the THINNEST leg's depth: a mint_and_sell needs one
share sold at each of N legs per complete set, so you can only ever trade as
many sets as the shallowest leg supports.

None of this places an order or touches the NegRiskAdapter -- it's detection
only, same as pmm_data.constraints. Fees, gas, and slippage past the top of
book are not modeled; `min_profit`/`min_total_profit` are floors, not
guarantees the trade clears them.
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


@dataclass(frozen=True)
class ExecutableArbitrage:
    constraint_name: str
    constraint_type: str  # "mint_and_sell" | "buy_the_set" | "monotonicity"
    market_ids: tuple[str, ...]
    profit_per_set: float  # $ profit per complete set traded, before fees
    max_size: float        # complete sets tradeable, capped by the thinnest leg's depth
    total_profit: float    # profit_per_set * max_size, before fees
    detail: str


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
        ask_sum = sum(books[m].best_ask for m in group.market_ids)

        if bid_sum - 1.0 > min_profit:
            per_set = round(bid_sum - 1.0, 6)
            max_size = min(books[m].bid_size for m in group.market_ids)
            total = round(per_set * max_size, 4)
            if total > min_total_profit:
                results.append(ExecutableArbitrage(
                    constraint_name=group.name,
                    constraint_type="mint_and_sell",
                    market_ids=group.market_ids,
                    profit_per_set=per_set,
                    max_size=max_size,
                    total_profit=total,
                    detail=f"sum(best_bid)={bid_sum:.4f}: mint the complete set for $1, sell every leg at its bid "
                           f"(capped at {max_size:.2f} sets by the thinnest leg)",
                ))
        if 1.0 - ask_sum > min_profit:
            per_set = round(1.0 - ask_sum, 6)
            max_size = min(books[m].ask_size for m in group.market_ids)
            total = round(per_set * max_size, 4)
            if total > min_total_profit:
                results.append(ExecutableArbitrage(
                    constraint_name=group.name,
                    constraint_type="buy_the_set",
                    market_ids=group.market_ids,
                    profit_per_set=per_set,
                    max_size=max_size,
                    total_profit=total,
                    detail=f"sum(best_ask)={ask_sum:.4f}: buy every leg at its ask for a guaranteed $1 payout "
                           f"(capped at {max_size:.2f} sets by the thinnest leg)",
                ))

    for mono in constraint_set.monotone_constraints:
        if mono.superset_id not in books or mono.subset_id not in books:
            continue
        superset, subset = books[mono.superset_id], books[mono.subset_id]
        profit = subset.best_bid - superset.best_ask
        if profit > min_profit:
            per_set = round(profit, 6)
            max_size = min(subset.bid_size, superset.ask_size)
            total = round(per_set * max_size, 4)
            if total > min_total_profit:
                results.append(ExecutableArbitrage(
                    constraint_name=mono.name,
                    constraint_type="monotonicity",
                    market_ids=(mono.superset_id, mono.subset_id),
                    profit_per_set=per_set,
                    max_size=max_size,
                    total_profit=total,
                    detail=(
                        f"subset best_bid={subset.best_bid:.4f} > superset best_ask={superset.best_ask:.4f}: "
                        f"buy the superset at its ask, take the opposite side of the subset (e.g. its NO token) "
                        f"(capped at {max_size:.2f} sets by the thinnest leg)"
                    ),
                ))

    return results
