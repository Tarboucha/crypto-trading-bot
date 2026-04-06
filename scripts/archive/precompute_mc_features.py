"""Pre-compute Kronos MC features for RL training.

Reads the period config from select_rl_periods.py, slides a 512-candle window
through each period, runs N MC samples at each step, extracts features,
and saves everything to a parquet file.

Supports resume: if interrupted, re-run and it will skip already-computed steps.

Usage:
    python scripts/precompute_mc_features.py --pair ETH --samples 5
    python scripts/precompute_mc_features.py --pair ETH --samples 5 --pretrained  # baseline comparison
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent.parent
KRONOS_PATH = PROJECT_ROOT / "third_party" / "kronos"
sys.path.insert(0, str(KRONOS_PATH))

from model import Kronos, KronosTokenizer, KronosPredictor

PREPARED_DIR = PROJECT_ROOT / "data" / "ml" / "kronos" / "prepared"
RL_DIR = PROJECT_ROOT / "data" / "ml" / "rl"
CONTEXT_LEN = 512
PRED_LEN = 10


def load_models(pair: str, use_pretrained: bool = False, device: str = "cuda"):
    """Load tokenizer + predictor."""
    if use_pretrained:
        tok_path = str(PROJECT_ROOT / "data/ml/pretrained/Kronos-Tokenizer-base")
        pred_path = str(PROJECT_ROOT / "data/ml/pretrained/Kronos-base")
        print("Loading PRE-TRAINED models")
    else:
        base = PROJECT_ROOT / "data/ml/kronos/finetuned" / f"{pair.lower()}_5m"
        tok_path = str(base / "tokenizer" / "best_model")
        pred_path = str(base / "predictor" / "best_model")
        print(f"Loading FINE-TUNED models from {base}")

    tokenizer = KronosTokenizer.from_pretrained(tok_path).to(device).eval()
    model = Kronos.from_pretrained(pred_path).to(device).eval()
    predictor = KronosPredictor(model, tokenizer, max_context=CONTEXT_LEN)
    return predictor


def load_full_data(pair: str) -> pd.DataFrame:
    """Load full training CSV."""
    path = PREPARED_DIR / f"{pair}_5m_train.csv"
    df = pd.read_csv(path)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    return df


def extract_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (df["timestamps"] >= start) & (df["timestamps"] < end)
    return df[mask].reset_index(drop=True)


def mc_sample_features(predictor, context_df: pd.DataFrame, actual_next: pd.DataFrame,
                       n_samples: int, temperature: float, top_p: float) -> dict:
    """Run N MC samples and extract features for one timestep."""
    x_df = context_df[["open", "close", "high", "low", "volume", "amount"]]
    x_timestamps = context_df["timestamps"]
    current_close = context_df["close"].iloc[-1]

    last_ts = x_timestamps.iloc[-1]
    y_timestamps = pd.Series(pd.date_range(
        start=last_ts + pd.Timedelta(minutes=5), periods=PRED_LEN, freq="5min"
    ))

    final_returns = []
    optimal_long_returns = []
    optimal_short_returns = []
    mae_longs = []
    mae_shorts = []

    for _ in range(n_samples):
        pred_df = predictor.predict(
            df=x_df, x_timestamp=x_timestamps, y_timestamp=y_timestamps,
            pred_len=PRED_LEN, T=temperature, top_p=top_p, sample_count=1, verbose=False,
        )
        closes = pred_df["close"].values
        highs = pred_df["high"].values
        lows = pred_df["low"].values

        final_returns.append((closes[-1] - current_close) / current_close)
        optimal_long_returns.append((max(closes) - current_close) / current_close)
        optimal_short_returns.append((current_close - min(closes)) / current_close)
        mae_longs.append((min(lows) - current_close) / current_close)
        mae_shorts.append((max(highs) - current_close) / current_close)

    final_returns = np.array(final_returns)

    # Actual future returns (for reward computation during RL training)
    actual_closes = actual_next["close"].values
    actual_returns = [(c - current_close) / current_close for c in actual_closes]

    features = {
        "timestamp": context_df["timestamps"].iloc[-1],
        "close": current_close,
        # MC consensus
        "p_long": float(np.mean(final_returns > 0)),
        "p_short": float(np.mean(final_returns < 0)),
        "mu_return": float(np.mean(final_returns)),
        "sigma_return": float(np.std(final_returns)),
        "mu_opt_long": float(np.mean(optimal_long_returns)),
        "mu_opt_short": float(np.mean(optimal_short_returns)),
        # MC risk
        "worst_mae_long": float(min(mae_longs)),
        "worst_mae_short": float(max(mae_shorts)),
        "p_sl_long_2pct": float(np.mean(np.array(mae_longs) < -0.02)),
        "p_sl_short_2pct": float(np.mean(np.array(mae_shorts) > 0.02)),
        # Path agreement
        "avg_agreement": float(np.mean([abs(np.mean(np.sign(final_returns)))])),
    }

    # Add actual returns for each horizon step
    for k in range(PRED_LEN):
        if k < len(actual_returns):
            features[f"actual_return_{k+1}"] = actual_returns[k]
            features[f"actual_close_{k+1}"] = float(actual_closes[k])
        else:
            features[f"actual_return_{k+1}"] = np.nan
            features[f"actual_close_{k+1}"] = np.nan

    return features


def precompute_period(predictor, df: pd.DataFrame, period_name: str,
                      subsample: int, n_samples: int, temperature: float,
                      top_p: float, output_path: Path, existing_timestamps: set):
    """Pre-compute MC features for one period."""
    n_candles = len(df)
    n_steps = (n_candles - CONTEXT_LEN - PRED_LEN) // subsample
    print(f"\n  Period: {period_name}")
    print(f"  Candles: {n_candles:,}, Steps: {n_steps:,} (subsample every {subsample})")

    results = []
    skipped = 0
    t_start = time.time()

    for step_i in range(n_steps):
        idx = step_i * subsample
        context = df.iloc[idx : idx + CONTEXT_LEN]
        actual_next = df.iloc[idx + CONTEXT_LEN : idx + CONTEXT_LEN + PRED_LEN]

        ts = context["timestamps"].iloc[-1]

        # Skip if already computed (resume support)
        if ts in existing_timestamps:
            skipped += 1
            continue

        if len(actual_next) < PRED_LEN:
            break

        features = mc_sample_features(
            predictor, context, actual_next,
            n_samples=n_samples, temperature=temperature, top_p=top_p,
        )
        features["period"] = period_name
        results.append(features)

        # Progress logging
        done = step_i + 1
        if done % 100 == 0 or done == n_steps:
            elapsed = time.time() - t_start
            speed = (done - skipped) / elapsed if elapsed > 0 else 0
            eta = (n_steps - done) / speed / 3600 if speed > 0 else 0
            print(f"    [{done}/{n_steps}] {speed:.1f} steps/s, ETA: {eta:.1f}h")

    if skipped > 0:
        print(f"    Skipped {skipped} already-computed steps")

    return results


def main():
    parser = argparse.ArgumentParser(description="Pre-compute Kronos MC features for RL")
    parser.add_argument("--pair", type=str, default="ETH")
    parser.add_argument("--samples", type=int, default=5, help="MC samples per step (default: 5)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--include-val", action="store_true", help="Also pre-compute validation period")
    args = parser.parse_args()

    pair = args.pair.upper()

    # Load periods config
    periods_path = RL_DIR / f"{pair}_rl_periods.json"
    if not periods_path.exists():
        print(f"ERROR: Run select_rl_periods.py --export first. Missing: {periods_path}")
        sys.exit(1)

    with open(periods_path) as f:
        periods_config = json.load(f)

    subsample = periods_config["subsample_every"]

    # Load model and data
    predictor = load_models(pair, use_pretrained=args.pretrained, device=args.device)
    full_df = load_full_data(pair)

    # Output path
    suffix = "_pretrained" if args.pretrained else ""
    output_path = RL_DIR / f"{pair}_mc_features{suffix}.parquet"

    # Load existing results for resume
    existing_timestamps = set()
    existing_df = None
    if output_path.exists():
        existing_df = pd.read_parquet(output_path)
        existing_timestamps = set(pd.to_datetime(existing_df["timestamp"]))
        print(f"\nResuming: {len(existing_timestamps):,} steps already computed")

    # Process each training period
    all_results = []
    periods = periods_config["training_periods"]

    if args.include_val:
        periods["validation"] = periods_config["validation_period"]

    total_start = time.time()

    for name, config in periods.items():
        period_df = extract_period(full_df, config["start"], config["end"])
        if len(period_df) < CONTEXT_LEN + PRED_LEN:
            print(f"\n  WARNING: {name} has too few candles ({len(period_df)}), skipping")
            continue

        results = precompute_period(
            predictor, period_df, name,
            subsample=subsample, n_samples=args.samples,
            temperature=args.temperature, top_p=args.top_p,
            output_path=output_path, existing_timestamps=existing_timestamps,
        )
        all_results.extend(results)

        # Save after each period (incremental)
        if all_results:
            new_df = pd.DataFrame(all_results)
            if existing_df is not None:
                combined = pd.concat([existing_df, new_df], ignore_index=True)
            else:
                combined = new_df
            combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
            combined.to_parquet(output_path, index=False)
            existing_df = combined
            existing_timestamps = set(pd.to_datetime(combined["timestamp"]))
            print(f"    Saved: {len(combined):,} total steps → {output_path}")

    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"Done! Total time: {total_time / 3600:.1f}h")
    print(f"Output: {output_path}")
    if existing_df is not None:
        print(f"Total steps: {len(existing_df):,}")
        print(f"Periods: {existing_df['period'].nunique()}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
