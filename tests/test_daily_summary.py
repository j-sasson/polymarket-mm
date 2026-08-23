import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.daily_summary import build_summary


class TestBuildSummary(unittest.TestCase):
    def test_summary_with_no_data_files_does_not_crash(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = build_summary(Path(tmp))
            self.assertIn("No quotes.csv found yet", summary)
            self.assertIn("None completed yet", summary)
            self.assertIn("None yet", summary)

    def test_summary_with_synthetic_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            live_dir = Path(tmp)
            (live_dir / "quotes.csv").write_text(
                "timestamp,recv_time,market,asset_id,best_bid,best_ask,bid_size,ask_size,spread\n"
                "1000,x,m,a,0.4,0.5,10,10,0.1\n"
                "3601000,x,m,a,0.4,0.5,10,10,0.1\n"  # 1 hour later
            )
            (live_dir / "trades.csv").write_text(
                "timestamp,recv_time,market,asset_id,price,size,side\n1000,x,m,a,0.4,10,BUY\n"
            )
            (live_dir / "violations.csv").write_text(
                "start_time_iso,end_time_iso,duration_seconds,constraint_name,constraint_type,market_ids,"
                "num_observations,start_magnitude,max_magnitude,mean_magnitude,end_magnitude,resolved\n"
                "2026-01-01T00:00:00+00:00,2026-01-01T01:00:00+00:00,3600.0,test-group,negative_risk,a|b,2,"
                "0.03,0.04,0.035,0.03,True\n"
            )
            (live_dir / "arbitrage.csv").write_text(
                "time_iso,constraint_name,constraint_type,market_ids,profit_per_set,max_size,total_profit,detail\n"
                "2026-01-01T00:00:00+00:00,test-group,mint_and_sell,a|b|c|d|e,0.012,220.0,2.64,detail text\n"
            )

            summary = build_summary(live_dir)

            self.assertIn("1.0 hours", summary)
            self.assertIn("Trades logged: 1", summary)
            self.assertIn("Completed episodes: 1 (1 resolved, 0 censored)", summary)
            self.assertIn("test-group: 1 episode(s)", summary)
            self.assertIn("Hits logged: 1", summary)
            self.assertIn("$2.64", summary)
            self.assertIn("avg depth 220 shares", summary)


if __name__ == "__main__":
    unittest.main()
