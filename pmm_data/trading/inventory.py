"""Tracks net position per market and computes how much to skew price/size
to reduce inventory imbalance, per Polymarket's general inventory-management
guidance (lean your quotes away from your existing side to work back toward
flat). Pure bookkeeping and math -- no network calls.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Position:
    market_id: str
    net_shares: float = 0.0  # positive = long (net bought), negative = short (net sold)


class InventoryTracker:
    def __init__(self):
        self._positions: dict[str, Position] = {}

    def get(self, market_id: str) -> Position:
        return self._positions.setdefault(market_id, Position(market_id))

    def apply_fill(self, market_id: str, side: str, size: float) -> None:
        pos = self.get(market_id)
        pos.net_shares += size if side == "BUY" else -size

    def position_ratio(self, market_id: str, max_position: float) -> float:
        """Net position as a fraction of the configured cap, clipped to
        [-1, 1]. 0 = flat, +1 = at the long cap, -1 = at the short cap."""
        if max_position <= 0:
            return 0.0
        return max(-1.0, min(1.0, self.get(market_id).net_shares / max_position))


def inventory_skew(base_price: float, position_ratio: float, skew_strength: float, half_spread: float) -> float:
    """Shift the quote center away from the side that would grow an already
    large position. Long (positive ratio) -> lower center (less eager to
    keep buying, more eager to sell down); short -> higher center."""
    return base_price - skew_strength * position_ratio * half_spread


def inventory_size_scale(position_ratio: float, side: str) -> float:
    """Multiplier in [0, 1] applied to the base quote size on `side`.

    Long position (ratio > 0): BUY size shrinks toward 0 as we approach the
    cap (stop adding to an already-large long); SELL size stays full (we
    want to reduce, unrestricted). Short is the mirror image.
    """
    if side == "BUY":
        return max(0.0, min(1.0, 1.0 - position_ratio))
    return max(0.0, min(1.0, 1.0 + position_ratio))
