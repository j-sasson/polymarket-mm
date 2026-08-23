#!/usr/bin/env python3
"""Builds a daily summary of the live poller's findings and emails it via
Gmail SMTP using an app password.

Summary generation (`build_summary`) is decoupled from sending, so it can
be tested and previewed with `--dry-run` (prints instead of sending; no
credentials needed for that path at all).

Credentials: reads GMAIL_ADDRESS and GMAIL_APP_PASSWORD from the
environment. Never pass a password as a CLI argument, and never hardcode
one here -- in the GitHub Actions workflow these come from repository
secrets, injected as env vars only for the run.
"""
from __future__ import annotations

import argparse
import os
import smtplib
import sys
from datetime import datetime, timezone
from email.mime.text import MIMEText
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pmm_data.market_graph import load_config, market_question_lookup  # noqa: E402

SMTP_HOST = "smtp.gmail.com"
SMTP_PORT = 587


def _fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    minutes = seconds / 60
    if minutes < 60:
        return f"{minutes:.0f}m"
    hours = minutes / 60
    return f"{hours:.1f}h"


def build_summary(live_dir: Path) -> str:
    config = load_config()
    questions = market_question_lookup(config)
    lines: list[str] = []

    quotes_path = live_dir / "quotes.csv"
    if quotes_path.exists():
        q = pd.read_csv(quotes_path, usecols=["timestamp"])
        span_hours = (q["timestamp"].max() - q["timestamp"].min()) / 1000 / 3600
        lines.append(f"Collection window: {span_hours:.1f} hours ({span_hours/24:.2f} days), {len(q):,} quote rows")
    else:
        lines.append("No quotes.csv found yet.")

    trades_path = live_dir / "trades.csv"
    if trades_path.exists():
        t = pd.read_csv(trades_path)
        lines.append(f"Trades logged: {len(t):,}")

    lines.append("")
    lines.append("=== Constraint violations (midpoint-based, statistical signal) ===")
    violations_path = live_dir / "violations.csv"
    if violations_path.exists() and violations_path.stat().st_size > 0:
        v = pd.read_csv(violations_path)
        if len(v):
            resolved = v[v["resolved"] == True]  # noqa: E712
            lines.append(f"Completed episodes: {len(v)} ({len(resolved)} resolved, {len(v)-len(resolved)} censored)")
            if len(resolved):
                lines.append(
                    f"Duration: mean={_fmt_duration(resolved['duration_seconds'].mean())}, "
                    f"median={_fmt_duration(resolved['duration_seconds'].median())}, "
                    f"max={_fmt_duration(resolved['duration_seconds'].max())}"
                )
            by_constraint = v.groupby("constraint_name").size().sort_values(ascending=False)
            for name, count in by_constraint.items():
                lines.append(f"  {name}: {count} episode(s)")
        else:
            lines.append("None completed yet.")
    else:
        lines.append("None completed yet.")

    lines.append("")
    lines.append("=== Executable arbitrage (real bid/ask, size-capped, NET of Polymarket's real taker fee) ===")
    arb_path = live_dir / "arbitrage.csv"
    if arb_path.exists() and arb_path.stat().st_size > 0:
        a = pd.read_csv(arb_path)
        if len(a):
            lines.append(f"Hits logged: {len(a)}")
            lines.append(f"Total notional profit if all captured: ${a['total_profit'].sum():.2f}")
            lines.append(
                f"Per-hit: mean=${a['total_profit'].mean():.2f}, "
                f"median=${a['total_profit'].median():.2f}, "
                f"max=${a['total_profit'].max():.2f}"
            )
            by_constraint = a.groupby("constraint_name").agg(
                hits=("total_profit", "size"), total=("total_profit", "sum"), avg_size=("max_size", "mean"),
            ).sort_values("total", ascending=False)
            for name, row in by_constraint.iterrows():
                lines.append(f"  {name}: {int(row['hits'])} hit(s), ${row['total']:.2f} total, avg depth {row['avg_size']:.0f} shares")
        else:
            lines.append("None yet.")
    else:
        lines.append("None yet (or size tracking not deployed at start of window).")

    lines.append("")
    lines.append(f"Generated {datetime.now(timezone.utc).isoformat()}")
    lines.append("Repo: https://github.com/j-sasson/polymarket-mm")

    return "\n".join(lines)


def send_email(subject: str, body: str, to_addr: str) -> None:
    address = os.environ.get("GMAIL_ADDRESS")
    app_password = os.environ.get("GMAIL_APP_PASSWORD")
    if not address or not app_password:
        raise RuntimeError(
            "GMAIL_ADDRESS and GMAIL_APP_PASSWORD must be set in the environment "
            "(GitHub Actions: repository secrets, injected as env vars for this step only)."
        )

    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = address
    msg["To"] = to_addr

    with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
        server.starttls()
        server.login(address, app_password)
        server.sendmail(address, [to_addr], msg.as_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live-dir", default="data/live")
    parser.add_argument("--to", default=os.environ.get("GMAIL_ADDRESS", ""), help="recipient address")
    parser.add_argument("--dry-run", action="store_true", help="print the summary instead of sending it")
    args = parser.parse_args()

    summary = build_summary(Path(args.live_dir))
    subject = f"Polymarket bot daily check-in -- {datetime.now(timezone.utc).strftime('%Y-%m-%d')}"

    if args.dry_run:
        print(f"Subject: {subject}\n")
        print(summary)
        return

    if not args.to:
        raise SystemExit("--to is required when not using --dry-run (or set GMAIL_ADDRESS)")

    send_email(subject, summary, args.to)
    print(f"sent to {args.to}")


if __name__ == "__main__":
    main()
