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

None of this places an order -- it's detection only, same as
pmm_data.constraints. Fees are not modeled; a thin apparent profit can be
eaten entirely by Polymarket's fee schedule, so treat `min_profit` as a
floor, not a guarantee.
"""
from __future__ import annotations

from dataclasses import dataclass

from pmm_data.constraints import ConstraintSet


@dataclass(frozen=True)
class ExecutableArbitrage:
    constraint_name: str
    constraint_type: str  # "mint_and_sell" | "buy_the_set" | "monotonicity"
    market_ids: tuple[str, ...]
    profit_per_set: float  # $ profit per complete set traded, before fees
    detail: str


def check_executable_arbitrage(
    constraint_set: ConstraintSet,
    books: dict[str, tuple[float, float]],  # market_id -> (best_bid, best_ask)
    min_profit: float = 0.0,
) -> list[ExecutableArbitrage]:
    results: list[ExecutableArbitrage] = []

    for group in constraint_set.negative_risk_groups:
        if any(m not in books for m in group.market_ids):
            continue
        bid_sum = sum(books[m][0] for m in group.market_ids)
        ask_sum = sum(books[m][1] for m in group.market_ids)

        if bid_sum - 1.0 > min_profit:
            results.append(ExecutableArbitrage(
                constraint_name=group.name,
                constraint_type="mint_and_sell",
                market_ids=group.market_ids,
                profit_per_set=round(bid_sum - 1.0, 6),
                detail=f"sum(best_bid)={bid_sum:.4f}: mint the complete set for $1, sell every leg at its bid",
            ))
        if 1.0 - ask_sum > min_profit:
            results.append(ExecutableArbitrage(
                constraint_name=group.name,
                constraint_type="buy_the_set",
                market_ids=group.market_ids,
                profit_per_set=round(1.0 - ask_sum, 6),
                detail=f"sum(best_ask)={ask_sum:.4f}: buy every leg at its ask for a guaranteed $1 payout",
            ))

    for mono in constraint_set.monotone_constraints:
        if mono.superset_id not in books or mono.subset_id not in books:
            continue
        superset_ask = books[mono.superset_id][1]
        subset_bid = books[mono.subset_id][0]
        profit = subset_bid - superset_ask
        if profit > min_profit:
            results.append(ExecutableArbitrage(
                constraint_name=mono.name,
                constraint_type="monotonicity",
                market_ids=(mono.superset_id, mono.subset_id),
                profit_per_set=round(profit, 6),
                detail=(
                    f"subset best_bid={subset_bid:.4f} > superset best_ask={superset_ask:.4f}: "
                    "buy the superset at its ask, take the opposite side of the subset (e.g. its NO token)"
                ),
            ))

    return results
