import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pmm_data.constraints import ConstraintSet, MonotoneConstraint, NegativeRiskGroup
from pmm_data.violation_tracker import ViolationTracker


class TestViolationTracker(unittest.TestCase):
    def setUp(self):
        self.group = NegativeRiskGroup(name="cpi_buckets", market_ids=("a", "b"), tolerance=0.02)
        self.constraints = ConstraintSet(negative_risk_groups=[self.group])
        self.closed_episodes = []
        self.tracker = ViolationTracker(self.constraints, on_episode_closed=self.closed_episodes.append)

    def test_no_violation_emits_nothing(self):
        self.tracker.update({"a": 0.4, "b": 0.6}, now_ts=0.0)
        self.assertEqual(self.closed_episodes, [])

    def test_violation_persists_then_resolves_reports_duration_and_magnitude(self):
        # t=0: consistent
        self.tracker.update({"a": 0.4, "b": 0.6}, now_ts=0.0)
        # t=10: violation appears (sums to 1.20)
        self.tracker.update({"a": 0.5, "b": 0.7}, now_ts=10.0)
        # t=20: still violated, bigger (sums to 1.30)
        self.tracker.update({"a": 0.55, "b": 0.75}, now_ts=20.0)
        # t=35: still violated, smaller (sums to 1.10)
        self.tracker.update({"a": 0.45, "b": 0.65}, now_ts=35.0)
        # t=40: resolved (sums to 1.00)
        self.tracker.update({"a": 0.4, "b": 0.6}, now_ts=40.0)

        self.assertEqual(len(self.closed_episodes), 1)
        ep = self.closed_episodes[0]
        self.assertEqual(ep["constraint_name"], "cpi_buckets")
        self.assertEqual(ep["resolved"], True)
        self.assertAlmostEqual(ep["start_time"], 10.0)
        self.assertAlmostEqual(ep["end_time"], 40.0)  # resolution confirmed at the first clean observation
        self.assertAlmostEqual(ep["duration_seconds"], 30.0)
        self.assertEqual(ep["num_observations"], 3)
        self.assertAlmostEqual(ep["start_magnitude"], 0.20, places=4)
        self.assertAlmostEqual(ep["max_magnitude"], 0.30, places=4)
        self.assertAlmostEqual(ep["end_magnitude"], 0.10, places=4)

    def test_flush_closes_open_episode_as_unresolved(self):
        self.tracker.update({"a": 0.5, "b": 0.7}, now_ts=0.0)  # violation opens
        self.tracker.flush(now_ts=100.0)  # process stops mid-violation

        self.assertEqual(len(self.closed_episodes), 1)
        ep = self.closed_episodes[0]
        self.assertEqual(ep["resolved"], False)
        self.assertAlmostEqual(ep["duration_seconds"], 100.0)

    def test_multiple_separate_episodes_tracked_independently(self):
        self.tracker.update({"a": 0.5, "b": 0.7}, now_ts=0.0)   # episode 1 opens
        self.tracker.update({"a": 0.4, "b": 0.6}, now_ts=5.0)   # episode 1 resolves
        self.tracker.update({"a": 0.3, "b": 0.55}, now_ts=20.0)  # episode 2 opens (sums to 0.85)
        self.tracker.update({"a": 0.4, "b": 0.6}, now_ts=30.0)   # episode 2 resolves

        self.assertEqual(len(self.closed_episodes), 2)
        self.assertAlmostEqual(self.closed_episodes[0]["duration_seconds"], 5.0)
        self.assertAlmostEqual(self.closed_episodes[1]["duration_seconds"], 10.0)

    def test_export_load_state_roundtrip_continues_episode_across_process_restart(self):
        self.tracker.update({"a": 0.5, "b": 0.7}, now_ts=0.0)  # violation opens
        self.tracker.update({"a": 0.55, "b": 0.75}, now_ts=10.0)  # persists, bigger

        state = self.tracker.export_state()

        # simulate a fresh process: new tracker, restore state, continue
        closed = []
        fresh_tracker = ViolationTracker(self.constraints, on_episode_closed=closed.append)
        fresh_tracker.load_state(state)
        fresh_tracker.update({"a": 0.4, "b": 0.6}, now_ts=100.0)  # resolves

        self.assertEqual(len(closed), 1)
        ep = closed[0]
        self.assertEqual(ep["start_time"], 0.0)  # preserved from before the restart
        self.assertEqual(ep["end_time"], 100.0)
        self.assertEqual(ep["num_observations"], 2)  # both pre-restart observations carried over
        self.assertAlmostEqual(ep["max_magnitude"], 0.30, places=4)

    def test_load_state_replaces_existing_open_episodes(self):
        self.tracker.update({"a": 0.5, "b": 0.7}, now_ts=0.0)
        self.tracker.load_state({})  # empty state wipes it
        closed = []
        self.tracker.on_episode_closed = closed.append
        self.tracker.update({"a": 0.4, "b": 0.6}, now_ts=5.0)
        self.assertEqual(closed, [])  # nothing to resolve -- the open episode was wiped

    def test_monotone_violation_tracked(self):
        mono = MonotoneConstraint(name="june_ge_march", superset_id="june", subset_id="march")
        constraints = ConstraintSet(monotone_constraints=[mono])
        closed = []
        tracker = ViolationTracker(constraints, on_episode_closed=closed.append)

        tracker.update({"march": 0.60, "june": 0.45}, now_ts=0.0)   # violation: march > june
        tracker.update({"march": 0.60, "june": 0.45}, now_ts=5.0)   # still violated
        tracker.update({"march": 0.30, "june": 0.55}, now_ts=8.0)   # resolved

        self.assertEqual(len(closed), 1)
        ep = closed[0]
        self.assertEqual(ep["constraint_type"], "monotonicity")
        self.assertEqual(ep["market_ids"], ("june", "march"))
        self.assertAlmostEqual(ep["duration_seconds"], 8.0)
        self.assertEqual(ep["num_observations"], 2)


if __name__ == "__main__":
    unittest.main()
