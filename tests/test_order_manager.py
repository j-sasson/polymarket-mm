import sys
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pmm_data.constraints import ConstraintSet
from pmm_data.trading.inventory import InventoryTracker
from pmm_data.trading.order_manager import OrderManager, TradingConfig
from pmm_data.trading.risk_limits import KillSwitch, MarketLimits, RiskLimits


class FakeLogger:
    def __init__(self):
        self.order_attempts = []
        self.fills = []
        self.cancel_alls = []
        self.events = []

    def log_order_attempt(self, dry_run, market_id, token_id, side, price, size, order_type, expiration_iso, status, order_id="", reason=""):
        self.order_attempts.append(dict(
            dry_run=dry_run, market_id=market_id, token_id=token_id, side=side, price=price,
            size=size, order_type=order_type, expiration_iso=expiration_iso, status=status,
            order_id=order_id, reason=reason,
        ))

    def log_fill(self, market_id, token_id, side, price, size, order_id):
        self.fills.append(dict(market_id=market_id, token_id=token_id, side=side, price=price, size=size, order_id=order_id))

    def log_cancel_all(self, reason, dry_run, order_ids, response=""):
        self.cancel_alls.append(dict(reason=reason, dry_run=dry_run, order_ids=order_ids, response=response))

    def log_event(self, event_type, detail):
        self.events.append((event_type, detail))


class FakeClobClient:
    def __init__(self, fail_create_order=False, fail_post_order=False, matched=False):
        self.orders_posted = []
        self.cancelled = []
        self.cancel_all_called = False
        self.fail_create_order = fail_create_order
        self.fail_post_order = fail_post_order
        self.matched = matched
        self._next_id = 0

    def create_order(self, order_args):
        if self.fail_create_order:
            raise RuntimeError("signing failed")
        return order_args

    def post_order(self, signed_order, order_type):
        if self.fail_post_order:
            raise RuntimeError("submission failed")
        self._next_id += 1
        oid = f"order-{self._next_id}"
        self.orders_posted.append((signed_order, order_type, oid))
        return {"success": True, "orderID": oid, "status": "matched" if self.matched else "live"}

    def cancel(self, order_id):
        self.cancelled.append(order_id)
        return {"success": True}

    def cancel_all(self):
        self.cancel_all_called = True
        return {"canceled": list(self.cancelled)}


def make_manager(client, dry_run, matched=False, limits=None, catalyst_hours=24, base_quote_size=5.0):
    constraint_set = ConstraintSet()  # no groups/constraints -- pure baseline quoting for these tests
    risk_limits = RiskLimits(per_market={"m1": limits or MarketLimits(
        max_position_shares=100, max_exposure_usd=1000, max_order_size_shares=20,
    )})
    inventory = InventoryTracker()
    logger = FakeLogger()
    kill_switch = KillSwitch(client, logger, dry_run=dry_run)
    now = datetime(2026, 8, 21, 12, 0, 0, tzinfo=timezone.utc)
    catalysts = {"m1": now + timedelta(hours=catalyst_hours)}
    config = TradingConfig(half_spread=0.01, skew_aggressiveness=0.5, taper_seconds=60.0, base_quote_size=base_quote_size)
    manager = OrderManager(client, constraint_set, risk_limits, inventory, logger, kill_switch, catalysts, config, dry_run)
    return manager, logger, inventory, kill_switch, now


class TestDryRun(unittest.TestCase):
    def test_dry_run_logs_both_sides_without_touching_client(self):
        manager, logger, inventory, kill_switch, now = make_manager(client=None, dry_run=True)
        manager.on_quote_update("m1", "tok1", best_bid=0.39, best_ask=0.41, now_ts=0.0, now_dt=now)

        attempts = [a for a in logger.order_attempts if a["status"] == "dry_run_would_place"]
        self.assertEqual(len(attempts), 2)
        sides = {a["side"] for a in attempts}
        self.assertEqual(sides, {"BUY", "SELL"})
        self.assertEqual(manager.all_open_order_ids(), [])  # dry run never tracks real order ids

    def test_missing_catalyst_skips_quoting(self):
        manager, logger, inventory, kill_switch, now = make_manager(client=None, dry_run=True)
        del manager.catalysts["m1"]
        manager.on_quote_update("m1", "tok1", best_bid=0.39, best_ask=0.41, now_ts=0.0, now_dt=now)
        self.assertEqual(logger.order_attempts, [])

    def test_missing_limits_skips_quoting(self):
        manager, logger, inventory, kill_switch, now = make_manager(client=None, dry_run=True)
        manager.risk_limits.per_market.clear()
        manager.on_quote_update("m1", "tok1", best_bid=0.39, best_ask=0.41, now_ts=0.0, now_dt=now)
        self.assertEqual(logger.order_attempts, [])

    def test_order_size_exceeding_limit_rejected_not_shrunk(self):
        tight_limits = MarketLimits(max_position_shares=100, max_exposure_usd=1000, max_order_size_shares=2)
        manager, logger, inventory, kill_switch, now = make_manager(
            client=None, dry_run=True, limits=tight_limits, base_quote_size=5.0,
        )
        manager.on_quote_update("m1", "tok1", best_bid=0.39, best_ask=0.41, now_ts=0.0, now_dt=now)
        rejected = [a for a in logger.order_attempts if a["status"] == "rejected_by_limits"]
        self.assertEqual(len(rejected), 2)
        self.assertTrue(all("max_order_size_shares" in a["reason"] for a in rejected))


class TestLiveSubmission(unittest.TestCase):
    def test_successful_order_tracked_as_resting(self):
        client = FakeClobClient()
        manager, logger, inventory, kill_switch, now = make_manager(client=client, dry_run=False)
        manager.on_quote_update("m1", "tok1", best_bid=0.39, best_ask=0.41, now_ts=0.0, now_dt=now)

        self.assertEqual(len(client.orders_posted), 2)
        self.assertEqual(len(manager.all_open_order_ids()), 2)
        self.assertFalse(kill_switch.triggered)

    def test_matched_status_updates_inventory_and_logs_fill(self):
        client = FakeClobClient(matched=True)
        manager, logger, inventory, kill_switch, now = make_manager(client=client, dry_run=False)
        manager.on_quote_update("m1", "tok1", best_bid=0.39, best_ask=0.41, now_ts=0.0, now_dt=now)

        self.assertEqual(len(logger.fills), 2)  # both bid and ask "matched" immediately in this fake
        fill_sides = {f["side"] for f in logger.fills}
        self.assertEqual(fill_sides, {"BUY", "SELL"})
        # equal-size buy and sell both matching nets out to flat -- correct, not a bug
        self.assertEqual(inventory.get("m1").net_shares, 0)

    def test_create_order_exception_trips_kill_switch_and_reraises(self):
        client = FakeClobClient(fail_create_order=True)
        manager, logger, inventory, kill_switch, now = make_manager(client=client, dry_run=False)
        with self.assertRaises(RuntimeError):
            manager.on_quote_update("m1", "tok1", best_bid=0.39, best_ask=0.41, now_ts=0.0, now_dt=now)

        self.assertTrue(kill_switch.triggered)
        self.assertTrue(client.cancel_all_called)
        self.assertEqual(manager.resting_orders, {})

    def test_after_kill_switch_further_updates_are_noop(self):
        client = FakeClobClient(fail_create_order=True)
        manager, logger, inventory, kill_switch, now = make_manager(client=client, dry_run=False)
        with self.assertRaises(RuntimeError):
            manager.on_quote_update("m1", "tok1", best_bid=0.39, best_ask=0.41, now_ts=0.0, now_dt=now)

        posted_before = len(client.orders_posted)
        # trading should be halted now -- no further attempts, even on a fresh clean update
        manager.on_quote_update("m1", "tok1", best_bid=0.40, best_ask=0.42, now_ts=10.0, now_dt=now)
        self.assertEqual(len(client.orders_posted), posted_before)

    def test_position_breach_from_price_move_trips_kill_switch_without_new_order(self):
        # This is the scenario the pre-trade check_order gate CAN'T catch:
        # an existing, already-within-limits position whose dollar exposure
        # breaches purely because the market moved, with no new fill at all.
        client = FakeClobClient()
        tight_limits = MarketLimits(max_position_shares=100, max_exposure_usd=10, max_order_size_shares=20)
        manager, logger, inventory, kill_switch, now = make_manager(
            client=client, dry_run=False, limits=tight_limits,
        )
        inventory.apply_fill("m1", "BUY", 15)  # pre-existing position from an earlier fill
        manager.on_quote_update("m1", "tok1", best_bid=0.39, best_ask=0.41, now_ts=0.0, now_dt=now)
        self.assertFalse(kill_switch.triggered)  # exposure = 15*0.40 = 6.0, within the $10 cap

        manager.on_quote_update("m1", "tok1", best_bid=0.89, best_ask=0.91, now_ts=1.0, now_dt=now)
        self.assertTrue(kill_switch.triggered)  # exposure = 15*0.90 = 13.5, breaches the $10 cap
        self.assertTrue(client.cancel_all_called)

    def test_requote_skipped_for_near_identical_price(self):
        client = FakeClobClient()
        manager, logger, inventory, kill_switch, now = make_manager(client=client, dry_run=False)
        manager.on_quote_update("m1", "tok1", best_bid=0.39, best_ask=0.41, now_ts=0.0, now_dt=now)
        first_count = len(client.orders_posted)

        # essentially unchanged book -- should not cancel/replace
        manager.on_quote_update("m1", "tok1", best_bid=0.3901, best_ask=0.4101, now_ts=1.0, now_dt=now)
        self.assertEqual(len(client.orders_posted), first_count)
        self.assertEqual(len(client.cancelled), 0)

    def test_requote_replaces_on_meaningful_price_move(self):
        client = FakeClobClient()
        manager, logger, inventory, kill_switch, now = make_manager(client=client, dry_run=False)
        manager.on_quote_update("m1", "tok1", best_bid=0.39, best_ask=0.41, now_ts=0.0, now_dt=now)
        first_count = len(client.orders_posted)

        manager.on_quote_update("m1", "tok1", best_bid=0.44, best_ask=0.46, now_ts=1.0, now_dt=now)
        self.assertGreater(len(client.orders_posted), first_count)
        self.assertEqual(len(client.cancelled), 2)  # old bid + ask cancelled before replacement


if __name__ == "__main__":
    unittest.main()
