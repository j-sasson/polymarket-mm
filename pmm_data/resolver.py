"""Resolve human-facing Polymarket identifiers (condition IDs / slugs) into the
CLOB token ("asset") IDs that the real-time websocket actually subscribes on.

Polymarket has two different ID spaces that are easy to confuse:
  - "market" / condition ID: a 0x-prefixed 64-hex-char id identifying the market
    as a whole (both outcomes).
  - "asset" / CLOB token ID: a large numeric string identifying ONE outcome
    token (e.g. the "Yes" side). The realtime market websocket subscribes by
    asset_ids, not condition IDs.

Gamma API (https://gamma-api.polymarket.com) exposes both, so given a
condition ID or a market slug we can look up its clobTokenIds.
"""
from __future__ import annotations

import json
from typing import Iterable

import requests

GAMMA_BASE = "https://gamma-api.polymarket.com"


def _looks_like_condition_id(identifier: str) -> bool:
    return identifier.startswith("0x") and len(identifier) == 66


def _looks_like_token_id(identifier: str) -> bool:
    return identifier.isdigit() and len(identifier) > 20


def resolve_token_ids(identifiers: Iterable[str]) -> dict[str, list[str]]:
    """Map each input identifier to its list of CLOB token IDs.

    Accepts condition IDs (0x...), market slugs, or already-resolved token
    IDs (passed through unchanged, mapped to themselves).

    Returns: {identifier: [token_id, ...]}
    """
    result: dict[str, list[str]] = {}
    condition_ids = []
    slugs = []

    for ident in identifiers:
        ident = ident.strip()
        if not ident:
            continue
        if _looks_like_token_id(ident):
            result[ident] = [ident]
        elif _looks_like_condition_id(ident):
            condition_ids.append(ident)
        else:
            slugs.append(ident)

    if condition_ids:
        # Gamma requires repeated `condition_ids=` params, not a comma-joined
        # string (a comma-joined value silently matches zero markets), and
        # silently caps how many repeated params it honors per request
        # (observed: 31 requested -> only 20 returned), so batch conservatively.
        BATCH_SIZE = 20
        for batch_start in range(0, len(condition_ids), BATCH_SIZE):
            batch = condition_ids[batch_start:batch_start + BATCH_SIZE]
            resp = requests.get(
                f"{GAMMA_BASE}/markets",
                params={"condition_ids": batch},
                timeout=15,
            )
            resp.raise_for_status()
            for market in resp.json():
                cond_id = market.get("conditionId")
                token_ids = _parse_clob_token_ids(market)
                if cond_id and token_ids:
                    result[cond_id] = token_ids

    for slug in slugs:
        resp = requests.get(f"{GAMMA_BASE}/markets/slug/{slug}", timeout=15)
        if resp.status_code == 404:
            continue
        resp.raise_for_status()
        market = resp.json()
        token_ids = _parse_clob_token_ids(market)
        if token_ids:
            result[slug] = token_ids

    missing = set(identifiers) - set(result.keys())
    if missing:
        raise ValueError(
            f"Could not resolve token IDs for: {sorted(missing)}. "
            "Double-check the condition ID / slug is correct and the market exists."
        )
    return result


def _parse_clob_token_ids(market: dict) -> list[str]:
    raw = market.get("clobTokenIds")
    if not raw:
        return []
    if isinstance(raw, str):
        return json.loads(raw)
    return raw


def flatten_token_ids(resolved: dict[str, list[str]]) -> list[str]:
    seen = []
    for token_ids in resolved.values():
        for t in token_ids:
            if t not in seen:
                seen.append(t)
    return seen
