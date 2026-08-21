"""Orchestrates quoting: on every price update, recompute the graph-fair-
value + inventory-skewed target quotes (reusing the exact same constraint
and fair-value logic validated in steps 2-4), risk-check them, and either
log the intended order (dry_run) or sign + submit it as a GTD order that
expires before the market's configured catalyst.

Every code path that can go wrong -- a submission error, a cancel error, a
detected position/exposure breach -- trips the kill switch, which cancels
every order this manager believes is resting and halts further quoting.
After a trip, `on_quote_update` becomes a no-op; trading does not silently
resume on its own.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone

from pmm_data.backtest import next_skew
from pmm_data.constraints import ConstraintSet
from pmm_data.fair_value import graph_fair_value
from pmm_data.trading.gtd import compute_gtd_expiration
from pmm_data.trading.inventory import InventoryTracker, inventory_size_scale, inventory_skew
from pmm_data.trading.logging import TradeLogger
from pmm_data.trading.risk_limits import KillSwitch, RiskLimits, breach_reason, check_order


@dataclass
class TradingConfig:
    half_spread: float = 0.01
    skew_aggressiveness: float = 0.5
    taper_seconds: float = 60.0
    inventory_skew_strength: float = 0.5
    base_quote_size: float = 5.0
    desired_order_lifetime_seconds: float = 300.0
    pre_catalyst_buffer_seconds: float = 120.0
    min_requote_delta: float = 0.005  # don't cancel/replace for sub-half-cent moves


@dataclass
class RestingOrder:
    order_id: str
    market_id: str
    token_id: str
    side: str
    price: float
    size: float
    expiration_ts: int


class OrderManager:
    def __init__(
        self,
        client,  # py_clob_client.client.ClobClient, or None in pure dry-run
        constraint_set: ConstraintSet,
        risk_limits: RiskLimits,
        inventory: InventoryTracker,
        logger: TradeLogger,
        kill_switch: KillSwitch,
        catalysts: dict[str, datetime],
        config: TradingConfig,
        dry_run: bool,
    ):
        self.client = client
        self.constraint_set = constraint_set
        self.risk_limits = risk_limits
        self.inventory = inventory
        self.logger = logger
        self.kill_switch = kill_switch
        self.catalysts = catalysts
        self.config = config
        self.dry_run = dry_run

        self.current_prices: dict[str, float] = {}
        self.skew_state: dict[str, float] = {}
        self.last_update_ts: dict[str, float] = {}
        self.resting_orders: dict[str, list[RestingOrder]] = {}

    def all_open_order_ids(self) -> list[str]:
        return [o.order_id for orders in self.resting_orders.values() for o in orders if o.order_id]

    def on_quote_update(
        self, market_id: str, token_id: str, best_bid: float, best_ask: float, now_ts: float, now_dt: datetime,
    ) -> None:
        if self.kill_switch.triggered:
            return
        if best_bid is None or best_ask is None or best_ask <= best_bid:
            return

        raw_mid = (best_bid + best_ask) / 2
        self.current_prices[market_id] = raw_mid

        # Exposure can breach purely from the market moving against an
        # already-held position, with no new fill involved -- check on
        # every update, not only right after this manager's own fills.
        limits_now = self.risk_limits.for_market(market_id)
        if limits_now is not None:
            self._check_position_breach(market_id, raw_mid, limits_now)
            if self.kill_switch.triggered:
                return

        violations = self.constraint_set.check(self.current_prices)
        active_markets = {m for v in violations for m in v.market_ids}
        dt = now_ts - self.last_update_ts.get(market_id, now_ts)
        self.last_update_ts[market_id] = now_ts

        if market_id in active_markets:
            fair_prices = graph_fair_value(self.current_prices, violations)
            self.skew_state[market_id] = self.config.skew_aggressiveness * (fair_prices[market_id] - raw_mid)
        else:
            self.skew_state[market_id] = next_skew(self.skew_state.get(market_id, 0.0), dt, self.config.taper_seconds)

        limits = self.risk_limits.for_market(market_id)
        position_ratio = self.inventory.position_ratio(market_id, limits.max_position_shares) if limits else 0.0

        center = raw_mid + self.skew_state[market_id]
        center = inventory_skew(center, position_ratio, self.config.inventory_skew_strength, self.config.half_spread)

        bid_price = round(max(0.01, min(0.99, center - self.config.half_spread)), 4)
        ask_price = round(max(0.01, min(0.99, center + self.config.half_spread)), 4)
        if bid_price >= ask_price:
            self._cancel_market(market_id, reason="degenerate spread after skew")
            return

        catalyst_time = self.catalysts.get(market_id)
        if catalyst_time is None:
            self._cancel_market(market_id, reason="no catalyst configured for market")
            return
        expiration_ts = compute_gtd_expiration(
            now_dt, catalyst_time, self.config.desired_order_lifetime_seconds, self.config.pre_catalyst_buffer_seconds,
        )
        if expiration_ts is None:
            self._cancel_market(market_id, reason="catalyst too close/passed -- no safe GTD runway")
            return

        if limits is None:
            self._cancel_market(market_id, reason="no risk limits configured for market")
            return

        bid_size = round(self.config.base_quote_size * inventory_size_scale(position_ratio, "BUY"), 2)
        ask_size = round(self.config.base_quote_size * inventory_size_scale(position_ratio, "SELL"), 2)

        self._requote_side(market_id, token_id, "BUY", bid_price, bid_size, expiration_ts, limits)
        self._requote_side(market_id, token_id, "SELL", ask_price, ask_size, expiration_ts, limits)

    def _requote_side(
        self, market_id: str, token_id: str, side: str, price: float, size: float,
        expiration_ts: int, limits,
    ) -> None:
        existing = self._find_resting(market_id, side)
        if existing and abs(existing.price - price) < self.config.min_requote_delta and existing.size == size:
            return

        if existing:
            self._cancel_order(existing, reason="requoting")

        if size <= 0:
            return

        current_position = self.inventory.get(market_id).net_shares
        ok, reason = check_order(limits, current_position, price, size, side)
        if not ok:
            self.logger.log_order_attempt(
                self.dry_run, market_id, token_id, side, price, size, "GTD", None, "rejected_by_limits", reason=reason,
            )
            return

        expiration_iso = datetime.fromtimestamp(expiration_ts, tz=timezone.utc).isoformat()

        if self.dry_run:
            self.logger.log_order_attempt(
                self.dry_run, market_id, token_id, side, price, size, "GTD", expiration_iso, "dry_run_would_place",
            )
            return

        self._submit_live_order(market_id, token_id, side, price, size, expiration_ts, expiration_iso, limits)

    def _submit_live_order(
        self, market_id, token_id, side, price, size, expiration_ts, expiration_iso, limits,
    ) -> None:
        from py_clob_client.clob_types import OrderArgs, OrderType  # deferred: only needed when actually going live

        try:
            order_args = OrderArgs(token_id=token_id, price=price, size=size, side=side, expiration=expiration_ts)
            signed_order = self.client.create_order(order_args)
            response = self.client.post_order(signed_order, OrderType.GTD)
            order_id = response.get("orderID", "") if isinstance(response, dict) else ""
            status = response.get("status", "unknown") if isinstance(response, dict) else "unknown"
            self.logger.log_order_attempt(
                self.dry_run, market_id, token_id, side, price, size, "GTD", expiration_iso, status, order_id=order_id,
            )
            if order_id:
                self.resting_orders.setdefault(market_id, []).append(
                    RestingOrder(order_id, market_id, token_id, side, price, size, expiration_ts)
                )
            if status == "matched":
                self.inventory.apply_fill(market_id, side, size)
                self.logger.log_fill(market_id, token_id, side, price, size, order_id)
                self._check_position_breach(market_id, price, limits)
        except Exception as exc:
            self.logger.log_event("order_error", f"{market_id} {side}: {exc}")
            self.kill_switch.trigger(f"order placement error: {exc}", self.all_open_order_ids())
            self.resting_orders.clear()
            raise

    def _check_position_breach(self, market_id: str, mark_price: float, limits) -> None:
        position = self.inventory.get(market_id).net_shares
        reason = breach_reason(limits, position, mark_price)
        if reason:
            self.kill_switch.trigger(f"{market_id}: {reason}", self.all_open_order_ids())
            self.resting_orders.clear()

    def _find_resting(self, market_id: str, side: str) -> RestingOrder | None:
        for order in self.resting_orders.get(market_id, []):
            if order.side == side:
                return order
        return None

    def _cancel_order(self, order: RestingOrder, reason: str) -> None:
        orders_list = self.resting_orders.get(order.market_id, [])
        if order in orders_list:
            orders_list.remove(order)

        if self.dry_run:
            self.logger.log_cancel_all(reason=reason, dry_run=True, order_ids=[order.order_id])
            return
        try:
            self.client.cancel(order.order_id)
            self.logger.log_cancel_all(reason=reason, dry_run=False, order_ids=[order.order_id])
        except Exception as exc:
            self.logger.log_event("cancel_error", f"{order.order_id}: {exc}")
            self.kill_switch.trigger(f"cancel failure: {exc}", self.all_open_order_ids())
            raise

    def _cancel_market(self, market_id: str, reason: str) -> None:
        for order in list(self.resting_orders.get(market_id, [])):
            self._cancel_order(order, reason)
