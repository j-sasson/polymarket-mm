"""Structured, timestamped logs of every order attempt, fill, cancellation,
and trading-loop event (rejections, errors, kill-switch trips) -- separate
CSVs so each is easy to review or analyze independently later."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pmm_data.csv_logger import CsvLogger

ORDER_FIELDS = [
    "time_iso", "dry_run", "market_id", "token_id", "side", "price", "size",
    "order_type", "expiration_iso", "status", "order_id", "reason",
]
FILL_FIELDS = ["time_iso", "market_id", "token_id", "side", "price", "size", "order_id"]
CANCEL_FIELDS = ["time_iso", "dry_run", "reason", "order_ids", "response"]
EVENT_FIELDS = ["time_iso", "event_type", "detail"]


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class TradeLogger:
    def __init__(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.orders = CsvLogger(out_dir / "orders.csv", ORDER_FIELDS)
        self.fills = CsvLogger(out_dir / "fills.csv", FILL_FIELDS)
        self.cancels = CsvLogger(out_dir / "cancels.csv", CANCEL_FIELDS)
        self.events = CsvLogger(out_dir / "trading_events.csv", EVENT_FIELDS)

    def log_order_attempt(
        self, dry_run: bool, market_id: str, token_id: str, side: str, price: float, size: float,
        order_type: str, expiration_iso: str | None, status: str, order_id: str = "", reason: str = "",
    ):
        self.orders.write({
            "time_iso": now_iso(), "dry_run": dry_run, "market_id": market_id, "token_id": token_id,
            "side": side, "price": price, "size": size, "order_type": order_type,
            "expiration_iso": expiration_iso or "", "status": status, "order_id": order_id, "reason": reason,
        })

    def log_fill(self, market_id: str, token_id: str, side: str, price: float, size: float, order_id: str):
        self.fills.write({
            "time_iso": now_iso(), "market_id": market_id, "token_id": token_id,
            "side": side, "price": price, "size": size, "order_id": order_id,
        })

    def log_cancel_all(self, reason: str, dry_run: bool, order_ids: list[str], response: str = ""):
        self.cancels.write({
            "time_iso": now_iso(), "dry_run": dry_run, "reason": reason,
            "order_ids": "|".join(order_ids), "response": response,
        })

    def log_event(self, event_type: str, detail: str):
        self.events.write({"time_iso": now_iso(), "event_type": event_type, "detail": detail})

    def close(self):
        self.orders.close()
        self.fills.close()
        self.cancels.close()
        self.events.close()
