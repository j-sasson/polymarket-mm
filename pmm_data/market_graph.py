"""Builds the real ConstraintSet for our tracked CPI/Fed markets from
data/markets_config.json (produced when the events were resolved via Gamma).

Market ids used throughout are condition IDs (matching the `market` column
logged in quotes.csv). Each market's "Yes" outcome (clob_token_ids[0], per
Polymarket's ["Yes","No"] outcome convention) is the price the constraint
logic reasons about -- e.g. "P(this CPI bucket is the actual print)".
"""
from __future__ import annotations

import json
from pathlib import Path

from pmm_data.constraints import ConstraintSet, MonotoneConstraint, NegativeRiskGroup

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "data" / "markets_config.json"

# "Fed rate hike in 2026?" is a logical superset of any single 2026 meeting
# having a rate increase -- if September hikes, a hike already happened in
# 2026, regardless of what October/December do. January 2027 is excluded:
# that meeting falls outside "in 2026".
FED_YEARLY_HIKE_EVENT_ID = "101936"
FED_MEETING_EVENT_IDS_IN_2026 = ("481717", "606422", "770450")  # Sep/Oct/Dec 2026


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> list[dict]:
    return json.loads(Path(path).read_text())


def asset_to_market_map(config: list[dict]) -> dict[str, str]:
    """Yes-outcome CLOB token id -> condition_id. No-outcome tokens are
    intentionally excluded; the checker only needs one price per market."""
    return {m["clob_token_ids"][0]: m["condition_id"] for e in config for m in e["markets"]}


def market_question_lookup(config: list[dict]) -> dict[str, str]:
    return {m["condition_id"]: m["question"] for e in config for m in e["markets"]}


def build_constraint_set(config: list[dict]) -> ConstraintSet:
    groups = []
    for event in config:
        market_ids = tuple(m["condition_id"] for m in event["markets"])
        if len(market_ids) >= 2:
            groups.append(NegativeRiskGroup(name=event["event_slug"], market_ids=market_ids))

    monotone = []
    hike_2026_event = next(e for e in config if e["event_id"] == FED_YEARLY_HIKE_EVENT_ID)
    hike_2026_market = hike_2026_event["markets"][0]["condition_id"]
    for event in config:
        if event["event_id"] not in FED_MEETING_EVENT_IDS_IN_2026:
            continue
        for m in event["markets"]:
            if "increase" in m["question"].lower():
                monotone.append(MonotoneConstraint(
                    name=f"{event['event_slug']}__implies__fed_hike_2026",
                    superset_id=hike_2026_market,
                    subset_id=m["condition_id"],
                ))

    return ConstraintSet(negative_risk_groups=groups, monotone_constraints=monotone)
