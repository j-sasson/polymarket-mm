"""Hard pre-trade limit checks and the kill switch.

`check_order` is a gate, not advice: every order MUST pass it before being
constructed or submitted. It never resizes an order down to fit a limit --
an order that would breach a limit is rejected outright and logged, not
silently shrunk, so there's no hidden logic quietly changing what was
supposedly configured.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MarketLimits:
    max_position_shares: float
    max_exposure_usd: float
    max_order_size_shares: float


@dataclass
class RiskLimits:
    per_market: dict[str, MarketLimits] = field(default_factory=dict)

    def for_market(self, market_id: str) -> MarketLimits | None:
        return self.per_market.get(market_id)


def check_order(
    limits: MarketLimits,
    current_position_shares: float,
    price: float,
    proposed_size: float,
    side: str,
) -> tuple[bool, str]:
    if proposed_size <= 0:
        return False, "non-positive size"
    if proposed_size > limits.max_order_size_shares:
        return False, f"order size {proposed_size} exceeds max_order_size_shares={limits.max_order_size_shares}"

    resulting_position = current_position_shares + (proposed_size if side == "BUY" else -proposed_size)
    if abs(resulting_position) > limits.max_position_shares:
        return False, (
            f"resulting position {resulting_position:.4f} would exceed "
            f"max_position_shares={limits.max_position_shares}"
        )

    resulting_exposure = abs(resulting_position) * price
    if resulting_exposure > limits.max_exposure_usd:
        return False, (
            f"resulting exposure ${resulting_exposure:.2f} would exceed "
            f"max_exposure_usd=${limits.max_exposure_usd}"
        )

    return True, "ok"


def breach_reason(limits: MarketLimits, position_shares: float, mark_price: float) -> str | None:
    """Checks an EXISTING position against limits (e.g. after a fill, or
    because the mark price moved). Returns a reason string if breached,
    else None. Used to trigger the kill switch independent of any new order."""
    if abs(position_shares) > limits.max_position_shares:
        return f"position {position_shares:.4f} exceeds max_position_shares={limits.max_position_shares}"
    exposure = abs(position_shares) * mark_price
    if exposure > limits.max_exposure_usd:
        return f"exposure ${exposure:.2f} exceeds max_exposure_usd=${limits.max_exposure_usd}"
    return None


class KillSwitch:
    """Cancels every open order the moment something goes wrong. Trips on:
    any unhandled exception in the trading loop, a websocket disconnect, or
    a position/exposure limit breach detected on an existing position.

    In dry_run mode, `client` is never touched -- the switch just logs what
    it would have cancelled and clears local order tracking.
    """

    def __init__(self, client, logger, dry_run: bool):
        self.client = client
        self.logger = logger
        self.dry_run = dry_run
        self.triggered = False
        self.trigger_reason: str | None = None

    def trigger(self, reason: str, open_order_ids: list[str]) -> None:
        self.triggered = True
        self.trigger_reason = reason
        self.logger.log_event("kill_switch", reason)

        if self.dry_run:
            self.logger.log_cancel_all(reason=reason, dry_run=True, order_ids=open_order_ids)
            return

        try:
            response = self.client.cancel_all()
            self.logger.log_cancel_all(reason=reason, dry_run=False, order_ids=open_order_ids, response=str(response))
        except Exception as exc:  # cancellation failing is the worst case -- surface it loudly
            self.logger.log_event("kill_switch_cancel_all_FAILED", f"{reason} :: {exc}")
            raise
