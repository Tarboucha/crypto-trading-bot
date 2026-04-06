"""Select diverse market regime periods for RL training data.

Analyzes historical candle data, identifies market regimes, and selects
representative periods so the RL agent trains on varied conditions.

Usage:
    # Analyze and show regime stats
    python scripts/select_rl_periods.py --pair ETH --analyze

    # Export selected periods to a config file
    python scripts/select_rl_periods.py --pair ETH --export

    # Use custom periods (edit PERIODS dict below)
    python scripts/select_rl_periods.py --pair ETH --export --custom
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
PREPARED_DIR = PROJECT_ROOT / "data" / "ml" / "kronos" / "prepared"
OUTPUT_DIR = PROJECT_ROOT / "data" / "ml" / "rl"

# ──────────────────────────────────────────────
# Hand-picked periods covering diverse regimes
# ──────────────────────────────────────────────

# Quick test: 3 months covering different regimes (for pipeline testing)
PERIODS_TEST = {
    "test_crash": {
        "start": "2022-06-01",
        "end": "2022-07-01",
        "regime": "bear market",
        "description": "1 month bear (Luna aftermath) — test downtrend",
    },
    "test_trend": {
        "start": "2024-02-01",
        "end": "2024-03-01",
        "regime": "uptrend",
        "description": "1 month ETF rally — test uptrend",
    },
    "test_chop": {
        "start": "2023-02-01",
        "end": "2023-03-01",
        "regime": "low volatility",
        "description": "1 month sideways — test range-bound",
    },
}

PERIODS = {
    "covid_crash_recovery": {
        "start": "2020-03-01",
        "end": "2020-06-01",
        "regime": "crash + V-recovery",
        "description": "COVID crash (-60%), extreme volatility, sharp bounce",
    },
    "bull_top_may_crash": {
        "start": "2021-04-01",
        "end": "2021-07-01",
        "regime": "euphoria + crash",
        "description": "ATH euphoria, Elon/China FUD, -50% crash in May",
    },
    "bear_luna_3ac": {
        "start": "2022-05-01",
        "end": "2022-08-01",
        "regime": "bear market",
        "description": "Luna collapse, 3AC liquidation, sustained downtrend",
    },
    "low_vol_recovery": {
        "start": "2023-01-01",
        "end": "2023-04-01",
        "regime": "low volatility",
        "description": "Choppy, range-bound, slow recovery after FTX",
    },
    "etf_rally": {
        "start": "2024-01-01",
        "end": "2024-04-01",
        "regime": "strong uptrend",
        "description": "BTC ETF approval, institutional inflows, clean trend",
    },
    "recent_market": {
        "start": "2024-10-01",
        "end": "2025-02-01",
        "regime": "current conditions",
        "description": "Most recent data, current market microstructure",
    },
}

# Validation period (not for training)
VAL_PERIOD = {
    "validation": {
        "start": "2025-02-01",
        "end": "2025-07-01",
        "regime": "validation",
        "description": "Walk-forward validation — never used for RL training",
    },
}


def load_train_data(pair: str) -> pd.DataFrame:
    """Load the training CSV (test set already excluded)."""
    path = PREPARED_DIR / f"{pair}_5m_train.csv"
    df = pd.read_csv(path)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    return df


def extract_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    """Extract a date range from the data."""
    mask = (df["timestamps"] >= start) & (df["timestamps"] < end)
    return df[mask].copy()


def compute_regime_stats(period_df: pd.DataFrame) -> dict:
    """Compute key statistics for a period."""
    closes = period_df["close"].values
    returns = np.diff(closes) / closes[:-1]

    total_return = (closes[-1] - closes[0]) / closes[0]
    volatility = np.std(returns) * np.sqrt(288)  # annualized (288 5m candles/day)
    max_drawdown = 0.0
    peak = closes[0]
    for c in closes:
        peak = max(peak, c)
        dd = (c - peak) / peak
        max_drawdown = min(max_drawdown, dd)

    # Trend strength: fraction of days with positive returns
    daily_closes = period_df.set_index("timestamps")["close"].resample("1D").last().dropna()
    daily_returns = daily_closes.pct_change().dropna()
    trend_strength = (daily_returns > 0).mean()

    return {
        "candles": len(period_df),
        "days": (period_df["timestamps"].iloc[-1] - period_df["timestamps"].iloc[0]).days,
        "total_return": total_return,
        "annualized_vol": volatility,
        "max_drawdown": max_drawdown,
        "trend_strength": trend_strength,
        "mean_volume": period_df["volume"].mean(),
    }


def analyze(pair: str):
    """Analyze all periods and print regime statistics."""
    df = load_train_data(pair)
    print(f"\nFull dataset: {len(df):,} candles ({df['timestamps'].iloc[0]} → {df['timestamps'].iloc[-1]})")

    all_periods = {**PERIODS, **VAL_PERIOD}
    total_candles = 0

    print(f"\n{'Period':<25} {'Regime':<20} {'Candles':>8} {'Days':>5} {'Return':>8} {'Vol':>8} {'MaxDD':>8} {'Trend':>6}")
    print("-" * 105)

    for name, config in all_periods.items():
        period_df = extract_period(df, config["start"], config["end"])
        if len(period_df) == 0:
            print(f"{name:<25} {'NO DATA':<20}")
            continue

        stats = compute_regime_stats(period_df)
        is_val = name == "validation"
        marker = " (VAL)" if is_val else ""

        print(f"{name:<25} {config['regime']:<20} {stats['candles']:>8,} {stats['days']:>5} "
              f"{stats['total_return']:>+7.1%} {stats['annualized_vol']:>7.1%} "
              f"{stats['max_drawdown']:>+7.1%} {stats['trend_strength']:>5.1%}{marker}")

        if not is_val:
            total_candles += stats["candles"]

    print("-" * 105)
    print(f"{'TOTAL TRAINING':<25} {'':20} {total_candles:>8,}")
    print(f"\nSubsampled every 6th candle: ~{total_candles // 6:,} MC steps to pre-compute")


def export(pair: str, subsample: int = 6):
    """Export selected periods as a config JSON for the pre-compute script."""
    df = load_train_data(pair)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    periods_out = {}
    for name, config in PERIODS.items():
        period_df = extract_period(df, config["start"], config["end"])
        if len(period_df) == 0:
            print(f"WARNING: {name} has no data, skipping")
            continue

        stats = compute_regime_stats(period_df)
        periods_out[name] = {
            **config,
            "candles": stats["candles"],
            "days": stats["days"],
            "total_return": round(stats["total_return"], 4),
            "annualized_vol": round(stats["annualized_vol"], 4),
            "max_drawdown": round(stats["max_drawdown"], 4),
        }

    # Also export validation period separately
    val_df = extract_period(df, VAL_PERIOD["validation"]["start"], VAL_PERIOD["validation"]["end"])
    val_config = {
        **VAL_PERIOD["validation"],
        "candles": len(val_df),
    }

    output = {
        "pair": pair,
        "subsample_every": subsample,
        "training_periods": periods_out,
        "validation_period": val_config,
        "total_training_candles": sum(p["candles"] for p in periods_out.values()),
        "total_mc_steps_approx": sum(p["candles"] for p in periods_out.values()) // subsample,
    }

    out_path = OUTPUT_DIR / f"{pair}_rl_periods.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nExported to {out_path}")
    print(f"Training periods: {len(periods_out)}")
    print(f"Total candles: {output['total_training_candles']:,}")
    print(f"MC steps to pre-compute (~): {output['total_mc_steps_approx']:,}")


def main():
    parser = argparse.ArgumentParser(description="Select market regime periods for RL training")
    parser.add_argument("--pair", type=str, default="ETH", help="Pair (ETH or BTC)")
    parser.add_argument("--analyze", action="store_true", help="Show regime statistics")
    parser.add_argument("--export", action="store_true", help="Export periods config for pre-compute")
    parser.add_argument("--subsample", type=int, default=6, help="Subsample every N candles (default: 6)")
    parser.add_argument("--test", action="store_true", help="Use small 3-month test periods for pipeline testing")
    args = parser.parse_args()

    # Swap periods if --test
    if args.test:
        global PERIODS
        PERIODS = PERIODS_TEST
        print("*** USING TEST PERIODS (3 × 1 month) ***\n")

    if not args.analyze and not args.export:
        args.analyze = True

    if args.analyze:
        analyze(args.pair)

    if args.export:
        export(args.pair, args.subsample)


if __name__ == "__main__":
    main()
