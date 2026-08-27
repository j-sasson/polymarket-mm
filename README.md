# polymarket-mm

A market-making / arbitrage-detection research project for [Polymarket](https://polymarket.com), focused on **logically related markets** — CPI print buckets, Fed rate-decision buckets, and the cumulative/yearly markets that bound them.

The core idea: many Polymarket events aren't independent. CPI print buckets are mutually exclusive and must sum to 100%. A Fed meeting's rate-decision buckets work the same way. A "did the Fed hike at all in 2026" market must always be priced at least as high as any single meeting's hike probability, since it's a logical superset. When live prices violate these relationships, that's either a genuine mispricing or just bid-ask spread noise on a thin market — this project is built to tell the two apart, size the real opportunity (if any), and log everything for review.

**Status: research/observation only. No live capital has been deployed.** Live-trading infrastructure exists (see below) but defaults to dry-run and requires an explicit flag plus a typed confirmation to submit a real order — nothing in this repo does that on its own.

## What it does

- **Collects live market data** for a tracked set of CPI/Fed markets every 15 minutes via a scheduled GitHub Actions workflow (no server required) — [`scripts/poll_snapshot.py`](scripts/poll_snapshot.py), committing results straight back into `data/live/`.
- **Detects two kinds of inconsistency**, kept deliberately separate:
  - *Statistical* — [`pmm_data/constraints.py`](pmm_data/constraints.py): midpoint-based violations of negative-risk groups (sum to 1) and monotone relationships (superset ≥ subset). Useful as a monitoring signal even when it isn't tradeable.
  - *Executable* — [`pmm_data/executable_arbitrage.py`](pmm_data/executable_arbitrage.py): the same relationships checked against real best bid/ask, sized by the thinnest leg's actual order-book depth, and **net of Polymarket's real per-market taker fee** (fetched live, not assumed). Only this one claims real money is on the table.
- **Graph-theory extensions** on top of the same constraint set:
  - [`pmm_data/market_graph.py`](pmm_data/market_graph.py) also checks every market's own YES/NO pair as a trivial 2-outcome group — the simplest possible arbitrage check, and one nothing else here originally covered.
  - [`pmm_data/difference_graph.py`](pmm_data/difference_graph.py): a Floyd-Warshall closure over the monotone relationships (the same difference-constraint-system technique behind FX triangular-arbitrage detection), finding bounds implied by *chains* of constraints that were never directly hand-coded.
- **A backtest** ([`pmm_data/backtest.py`](pmm_data/backtest.py), [`pmm_data/performance.py`](pmm_data/performance.py)) comparing a baseline symmetric-quoting strategy against a graph-fair-value-skew strategy and an inventory-aware variant, with proper equity curve / Sharpe / max drawdown / hit-rate reporting — and a fill model and P&L-marking scheme designed specifically to avoid the two ways this kind of backtest usually lies to you (see [Notable findings](#notable-findings)).
- **Live-trading infrastructure** ([`pmm_data/trading/`](pmm_data/trading)) built on Polymarket's official [`py-clob-client`](https://github.com/Polymarket/py-clob-client) for wallet auth and order signing: inventory-aware position sizing, hard per-market exposure/size limits enforced *before* order construction (never shrunk after the fact), GTD orders that expire before a market's catalyst, full order/fill/cancel audit logging, and a kill switch that cancels everything on any error, disconnect, or limit breach — including one caused purely by a price move against an existing position, not just a new fill.
- **A daily email summary** via GitHub Actions + Gmail SMTP (app-password auth, stored as a repo secret, never in code) — [`scripts/daily_summary.py`](scripts/daily_summary.py).

## Notable findings

A few things surfaced along the way that shaped the design:

- **A real 12.9%-looking "arbitrage" turned out to be spread noise, not free money.** Summing raw midpoints across a 10-bucket CPI group showed a large violation; summing actual best bids showed only a razor-thin 0.6% edge, and best asks were nowhere close from the other side. This is why the executable-arbitrage checker uses real bid/ask depth, not midpoints.
- **A real, recurring, size-capped arbitrage ($1–3/hit, dozens of times over two days) turned out to be fee-negative.** Polymarket's taker fee (`fee = size × rate × p × (1-p)`, live rate fetched per market, observed up to 10% on some Fed markets) exceeded the entire edge once applied correctly — explaining why it had sat unclaimed on a liquid public exchange for 8+ hours. This is why `executable_arbitrage.py` reports gross, fee, and net profit separately, and why the net number is what actually gates whether something gets logged as a finding.
- **A naive backtest of the fair-value-skew strategy would have been systematically biased against itself.** Marking a fill's P&L against the midpoint at the *instant* of the fill mechanically penalizes any skewed quote, regardless of whether the skew was justified — so fills are marked against that market's own midpoint some seconds later instead, actually testing whether the market moved the way the skew predicted.
- **The current constraint set is mostly a star, not a chain** — several relationships point into one shared "yearly hike" node, so the difference-graph mostly reduces to what direct pairwise checks already find. One real chain exists (`meeting bucket → hike-by-<month> → hike-by-<next-month> → yearly`), added specifically to give the graph-theory layer something genuine to traverse rather than claim a result it didn't have.

## Repository layout

```
pmm_data/                 core library
  constraints.py          negative-risk groups + monotone constraints (statistical)
  executable_arbitrage.py fee-aware, depth-capped, real arbitrage detection
  market_graph.py         builds the real constraint set from data/markets_config.json
  difference_graph.py     Floyd-Warshall closure over monotone relationships
  fair_value.py           graph-implied fair value for the backtest's skew strategy
  violation_tracker.py    tracks violations as open/persist/resolve episodes
  backtest.py             baseline vs. fair-value-skew vs. inventory-aware strategies
  performance.py          equity curve, Sharpe, max drawdown, hit rate
  resolver.py             Gamma API helpers (condition IDs, slugs -> CLOB token IDs)
  trading/                live order placement, inventory, risk limits, kill switch (dry-run by default)
scripts/                  CLI entry points (see each file's docstring)
tests/                    121 tests, all against hand-built scenarios or real API responses
data/                     tracked market config + live-collected data (committed by the poller)
.github/workflows/        scheduled poller (15 min) and daily email summary
```

## Running it

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests

# one-off snapshot poll (what the scheduled workflow runs)
python scripts/poll_snapshot.py --out-dir data/live

# backtest the strategies against collected data
python scripts/run_backtest.py --live-dir data/live
```

Live trading requires `POLYMARKET_PRIVATE_KEY` in the environment (never as a CLI argument) and the explicit `--live` flag on `scripts/run_live_trader.py`, which then asks for a typed confirmation naming the wallet before doing anything. Omit `--live` and it only logs intended orders.

## Disclaimer

Research/educational project. Not financial advice. No live trading has occurred; findings above describe *detected* pricing behavior, not realized returns.
