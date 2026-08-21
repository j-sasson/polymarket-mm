"""Computes GTD (Good Till Date) order expirations that land strictly before
a market's known catalyst (e.g. the CPI release or FOMC decision the
market resolves on), per docs.polymarket.com/trading/place-orders:

  - GTD orders expire 1 minute early as a built-in safety buffer.
  - The minimum expiration must be at least 3 minutes in the future, or the
    order is rejected.

So we need `catalyst_time - our own pre-catalyst buffer` to leave at least
`min_lifetime_seconds` of runway from now, or we skip quoting rather than
submit an order that's rejected or that lingers unexpectedly close to the
catalyst.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

GTD_EARLY_EXPIRY_BUFFER_SECONDS = 60  # Polymarket expires GTD orders 1 min early
POLYMARKET_MIN_LIFETIME_SECONDS = 180  # orders expiring sooner than this are rejected


def compute_gtd_expiration(
    now: datetime,
    catalyst_time: datetime,
    desired_lifetime_seconds: float,
    pre_catalyst_buffer_seconds: float = 120,
) -> int | None:
    """Returns a unix-seconds expiration timestamp, or None if there isn't
    enough runway before the catalyst to place a valid GTD order at all.

    `pre_catalyst_buffer_seconds` is how long before the catalyst we want
    the order to be fully dead (on top of Polymarket's own 60s early-expiry
    buffer), so we're never resting an order into the release itself.
    """
    latest_safe_expiration = catalyst_time - timedelta(
        seconds=pre_catalyst_buffer_seconds + GTD_EARLY_EXPIRY_BUFFER_SECONDS
    )
    runway_seconds = (latest_safe_expiration - now).total_seconds()
    if runway_seconds < POLYMARKET_MIN_LIFETIME_SECONDS:
        return None

    desired_expiration = now + timedelta(seconds=max(desired_lifetime_seconds, POLYMARKET_MIN_LIFETIME_SECONDS))
    expiration = min(desired_expiration, latest_safe_expiration)
    return int(expiration.timestamp())
