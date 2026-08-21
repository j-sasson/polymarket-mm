#!/usr/bin/env python3
"""Summarize data/live/violations.csv into frequency / size / duration stats.

Answers: how often do constraint violations occur, how large are they
typically, and how long do they last on average -- overall and broken down
per constraint.

Usage:
    python scripts/summarize_violations.py --live-dir data/live
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def load_collection_window_hours(live_dir: Path) -> float | None:
    quotes_path = live_dir / "quotes.csv"
    if not quotes_path.exists():
        return None
    ts = pd.read_csv(quotes_path, usecols=["timestamp"])["timestamp"]
    if ts.empty:
        return None
    span_ms = ts.max() - ts.min()
    return span_ms / 1000 / 3600


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    resolved = df[df["resolved"] == True]  # noqa: E712
    agg = df.groupby(group_cols).agg(
        episodes=("duration_seconds", "size"),
        resolved_episodes=("resolved", "sum"),
    )
    dur_stats = resolved.groupby(group_cols)["duration_seconds"].agg(
        mean_duration_s="mean", median_duration_s="median", max_duration_s="max",
    )
    mag_stats = df.groupby(group_cols)["max_magnitude"].agg(
        mean_magnitude="mean", median_magnitude="median", max_magnitude="max",
    )
    out = agg.join(dur_stats).join(mag_stats)
    return out.reset_index()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--live-dir", default="data/live")
    parser.add_argument("--out-csv", help="Optional path to also write the per-constraint summary as CSV")
    args = parser.parse_args()

    live_dir = Path(args.live_dir)
    violations_path = live_dir / "violations.csv"
    if not violations_path.exists():
        print(f"no violations.csv found at {violations_path}")
        return

    df = pd.read_csv(violations_path)
    if df.empty:
        print(f"{violations_path} has no violation episodes yet.")
        return

    hours = load_collection_window_hours(live_dir)

    n_resolved = int((df["resolved"] == True).sum())  # noqa: E712
    n_censored = len(df) - n_resolved

    print("=" * 70)
    print("VIOLATION SUMMARY")
    print("=" * 70)
    if hours:
        print(f"Collection window: {hours:.1f} hours ({hours / 24:.2f} days)")
    print(f"Total episodes logged: {len(df)} ({n_resolved} resolved, {n_censored} censored/ongoing at a restart)")
    if hours and hours > 0:
        print(f"Overall frequency: {len(df) / hours:.3f} episodes/hour  ({len(df) / hours * 24:.2f} episodes/day)")

    resolved = df[df["resolved"] == True]  # noqa: E712
    if not resolved.empty:
        print()
        print("Duration (resolved episodes only, seconds):")
        print(f"  mean={resolved['duration_seconds'].mean():.2f}  "
              f"median={resolved['duration_seconds'].median():.2f}  "
              f"max={resolved['duration_seconds'].max():.2f}")
    print()
    print("Magnitude (price units, 0-1 scale):")
    print(f"  mean_of_max={df['max_magnitude'].mean():.4f}  "
          f"median_of_max={df['max_magnitude'].median():.4f}  "
          f"max={df['max_magnitude'].max():.4f}")

    print()
    print("-" * 70)
    print("By constraint type")
    print("-" * 70)
    by_type = summarize(df, ["constraint_type"])
    print(by_type.to_string(index=False))

    print()
    print("-" * 70)
    print("By individual constraint")
    print("-" * 70)
    by_constraint = summarize(df, ["constraint_name", "constraint_type"])
    by_constraint = by_constraint.sort_values("episodes", ascending=False)
    print(by_constraint.to_string(index=False))

    if args.out_csv:
        by_constraint.to_csv(args.out_csv, index=False)
        print(f"\nwrote {args.out_csv}")


if __name__ == "__main__":
    main()
