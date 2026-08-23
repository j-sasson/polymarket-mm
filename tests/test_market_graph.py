import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pmm_data.market_graph import (
    build_constraint_set,
    build_yes_no_constraint_set,
    load_config,
    yes_no_token_ids,
)


class TestYesNoConstraintSet(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def test_one_group_per_tracked_market(self):
        cs = build_yes_no_constraint_set(self.config)
        num_markets = sum(len(e["markets"]) for e in self.config)
        self.assertEqual(len(cs.negative_risk_groups), num_markets)
        self.assertEqual(cs.monotone_constraints, [])  # this graph only ever has YES/NO pairs

    def test_group_ids_are_synthetic_not_real_condition_ids(self):
        cs = build_yes_no_constraint_set(self.config)
        group = cs.negative_risk_groups[0]
        self.assertEqual(len(group.market_ids), 2)
        self.assertTrue(group.market_ids[0].endswith("__YES"))
        self.assertTrue(group.market_ids[1].endswith("__NO"))

    def test_yes_no_token_ids_match_config(self):
        mapping = yes_no_token_ids(self.config)
        first_market = self.config[0]["markets"][0]
        yes_tok, no_tok = mapping[first_market["condition_id"]]
        self.assertEqual(yes_tok, first_market["clob_token_ids"][0])
        self.assertEqual(no_tok, first_market["clob_token_ids"][1])

    def test_does_not_interfere_with_the_cross_market_constraint_set(self):
        # the two constraint sets use disjoint id spaces (real condition_ids
        # vs synthetic __YES/__NO ids) and must not be built from each other
        cross_market_cs = build_constraint_set(self.config)
        yes_no_cs = build_yes_no_constraint_set(self.config)
        cross_market_ids = {m for g in cross_market_cs.negative_risk_groups for m in g.market_ids}
        yes_no_ids = {m for g in yes_no_cs.negative_risk_groups for m in g.market_ids}
        self.assertEqual(cross_market_ids & yes_no_ids, set())


if __name__ == "__main__":
    unittest.main()
