"""Prepare Binance CSV data for Kronos fine-tuning.

Reads monthly Binance CSVs, converts to Kronos format, and splits into
train and test files (test physically separated to prevent leakage).

Usage:
    python scripts/prepare_kronos_data.py --pair ETH
    python scripts/prepare_kronos_data.py --pair BTC
    python scripts/prepare_kronos_data.py --pair ETH --test-start 2025-07
"""
import argparse
import os
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
BINANCE_DIR = DATA_DIR / "binance"
OUTPUT_DIR = DATA_DIR / "ml" / "kronos" / "prepared"


def load_binance_pair(pair: str, timeframe: str = "5m") -> pd.DataFrame:
    """Load all monthly CSVs for a pair into one DataFrame."""
    pair_dir = BINANCE_DIR / f"{pair}USDT" / timeframe
    if not pair_dir.exists():
        raise FileNotFoundError(f"No data found at {pair_dir}")

    csv_files = sorted(pair_dir.glob(f"*.csv"))
    print(f"Found {len(csv_files)} files in {pair_dir}")

    dfs = []
    for f in csv_files:
        df = pd.read_csv(f)
        dfs.append(df)

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    print(f"Total candles after dedup: {len(combined):,}")
    return combined


def convert_to_kronos_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convert our Binance CSV format to Kronos expected format.

    Our format:    timestamp (unix ms), open, high, low, close, volume
    Kronos format: timestamps (datetime), open, close, high, low, volume, amount
    """
    out = pd.DataFrame()
    out["timestamps"] = pd.to_datetime(df["timestamp"], unit="ms").dt.strftime("%Y/%m/%d %H:%M")
    out["open"] = df["open"]
    out["close"] = df["close"]
    out["high"] = df["high"]
    out["low"] = df["low"]
    out["volume"] = df["volume"]
    out["amount"] = 0
    return out


def main():
    parser = argparse.ArgumentParser(description="Prepare Binance data for Kronos fine-tuning")
    parser.add_argument("--pair", type=str, required=True, help="Pair to prepare (ETH or BTC)")
    parser.add_argument("--timeframe", type=str, default="5m", help="Candle timeframe (default: 5m)")
    parser.add_argument("--test-start", type=str, default="2025-07",
                        help="Test set start month (YYYY-MM). Data from this point is saved separately. Default: 2025-07")
    args = parser.parse_args()

    pair = args.pair.upper()
    print(f"\n{'=' * 60}")
    print(f"Preparing {pair} {args.timeframe} data for Kronos")
    print(f"{'=' * 60}\n")

    # Load and convert
    df = load_binance_pair(pair, args.timeframe)
    kronos_df = convert_to_kronos_format(df)

    # Split by test date
    test_start_ts = pd.Timestamp(args.test_start)
    timestamps = pd.to_datetime(kronos_df["timestamps"], format="%Y/%m/%d %H:%M")

    train_mask = timestamps < test_start_ts
    test_mask = timestamps >= test_start_ts

    train_df = kronos_df[train_mask].reset_index(drop=True)
    test_df = kronos_df[test_mask].reset_index(drop=True)

    print(f"Train set: {len(train_df):,} candles ({timestamps[train_mask].min()} → {timestamps[train_mask].max()})")
    print(f"Test set:  {len(test_df):,} candles ({timestamps[test_mask].min()} → {timestamps[test_mask].max()})")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = OUTPUT_DIR / f"{pair}_{args.timeframe}_train.csv"
    test_path = OUTPUT_DIR / f"{pair}_{args.timeframe}_test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"\nSaved:")
    print(f"  Train: {train_path}")
    print(f"  Test:  {test_path}")
    print()


if __name__ == "__main__":
    main()
