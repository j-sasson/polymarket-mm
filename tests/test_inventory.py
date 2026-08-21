import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pmm_data.trading.inventory import InventoryTracker, inventory_size_scale, inventory_skew


class TestInventoryTracker(unittest.TestCase):
    def test_fills_accumulate_signed_position(self):
        inv = InventoryTracker()
        inv.apply_fill("m1", "BUY", 10)
        inv.apply_fill("m1", "SELL", 4)
        self.assertEqual(inv.get("m1").net_shares, 6)

    def test_position_ratio_clipped(self):
        inv = InventoryTracker()
        inv.apply_fill("m1", "BUY", 150)
        self.assertEqual(inv.position_ratio("m1", max_position=100), 1.0)

    def test_position_ratio_zero_cap_is_zero(self):
        inv = InventoryTracker()
        inv.apply_fill("m1", "BUY", 10)
        self.assertEqual(inv.position_ratio("m1", max_position=0), 0.0)


class TestInventorySkew(unittest.TestCase):
    def test_flat_position_no_skew(self):
        self.assertEqual(inventory_skew(0.50, position_ratio=0.0, skew_strength=0.5, half_spread=0.01), 0.50)

    def test_long_position_skews_center_down(self):
        result = inventory_skew(0.50, position_ratio=1.0, skew_strength=0.5, half_spread=0.01)
        self.assertLess(result, 0.50)

    def test_short_position_skews_center_up(self):
        result = inventory_skew(0.50, position_ratio=-1.0, skew_strength=0.5, half_spread=0.01)
        self.assertGreater(result, 0.50)


class TestInventorySizeScale(unittest.TestCase):
    def test_flat_full_size_both_sides(self):
        self.assertEqual(inventory_size_scale(0.0, "BUY"), 1.0)
        self.assertEqual(inventory_size_scale(0.0, "SELL"), 1.0)

    def test_at_long_cap_buy_shrinks_to_zero_sell_stays_full(self):
        self.assertEqual(inventory_size_scale(1.0, "BUY"), 0.0)
        self.assertEqual(inventory_size_scale(1.0, "SELL"), 1.0)

    def test_at_short_cap_sell_shrinks_to_zero_buy_stays_full(self):
        self.assertEqual(inventory_size_scale(-1.0, "SELL"), 0.0)
        self.assertEqual(inventory_size_scale(-1.0, "BUY"), 1.0)


if __name__ == "__main__":
    unittest.main()
