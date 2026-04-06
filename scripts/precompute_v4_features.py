"""Pre-compute V4 direct logit features at 5min resolution.

One Kronos forward pass per candle (no MC sampling). Extracts logit-based
conviction/direction features + candle-derived features.

Supports resume: skips timestamps already present in the output parquet.
Supports time-boxing: --max-hours to stop after N hours.

Usage:
    python scripts/precompute_v4_features.py --pair ETH --max-hours 5
    python scripts/precompute_v4_features.py --pair ETH --max-hours 5 --periods recent_market
    python scripts/precompute_v4_features.py --pair ETH --max-hours 0  # unlimited
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).parent.parent
KRONOS_PATH = PROJECT_ROOT / "third_party" / "kronos"
sys.path.insert(0, str(KRONOS_PATH))

from model import Kronos, KronosTokenizer
from model.kronos_cached import kronos_prefill

PREPARED_DIR = PROJECT_ROOT / "data" / "ml" / "kronos" / "prepared"
RL_DIR = PROJECT_ROOT / "data" / "ml" / "rl"
CONTEXT_LEN = 512
PRED_LEN = 10


# ──────────────────────────────────────────────
# Model loading
# ──────────────────────────────────────────────

def load_models(pair: str, device: str = "cuda"):
    base = PROJECT_ROOT / "data/ml/kronos/finetuned" / f"{pair.lower()}_5m"
    tok_path = str(base / "tokenizer" / "best_model")
    pred_path = str(base / "predictor" / "best_model")
    print(f"Loading models from {base}")

    tokenizer = KronosTokenizer.from_pretrained(tok_path).to(device).eval()
    model = Kronos.from_pretrained(pred_path).to(device).eval()
    return tokenizer, model


# ──────────────────────────────────────────────
# Token-to-price mapping (built once)
# ──────────────────────────────────────────────

def build_token_price_map(tokenizer, device):
    """Decode all 1024 tokens per head to get price values."""
    with torch.no_grad():
        s1_ids = torch.arange(1024, device=device).unsqueeze(0)
        s2_zeros = torch.zeros_like(s1_ids)
        s1_decoded = tokenizer.decode((s1_ids, s2_zeros), half=True).squeeze(0).cpu().numpy()

        s1_zeros = torch.zeros_like(s1_ids)
        s2_ids = torch.arange(1024, device=device).unsqueeze(0)
        s2_decoded = tokenizer.decode((s1_zeros, s2_ids), half=True).squeeze(0).cpu().numpy()

    # Close price is column 3
    return s1_decoded[:, 3], s2_decoded[:, 3]


# ──────────────────────────────────────────────
# Feature extraction (1 forward pass)
# ──────────────────────────────────────────────

def extract_kronos_features(
    tokenizer, model, context_df, s1_close_map, s2_close_map, device,
):
    """Run one Kronos forward pass and extract logit-based features."""
    price_cols = ["open", "close", "high", "low", "volume", "amount"]
    time_cols = ["minute", "hour", "weekday", "day", "month"]

    x_vals = context_df[price_cols].values.astype(np.float32)
    x_mean = x_vals.mean(axis=0)
    x_std = x_vals.std(axis=0)
    x_norm = (x_vals - x_mean) / (x_std + 1e-5)
    x_norm = np.clip(x_norm, -5, 5)

    stamps = context_df[time_cols].values.astype(np.float32)

    x_tensor = torch.from_numpy(x_norm[np.newaxis]).to(device)
    stamp_tensor = torch.from_numpy(stamps[np.newaxis]).to(device)

    with torch.no_grad():
        tokens = tokenizer.encode(x_tensor, half=True)
        s1_logits, context, _ = kronos_prefill(
            model, tokens[0], tokens[1], stamp=stamp_tensor,
        )

        # Get s2 logits (conditioned on argmax s1)
        s1_probs = F.softmax(s1_logits[:, -1, :], dim=-1).squeeze(0)
        s1_argmax = s1_probs.argmax().unsqueeze(0).unsqueeze(0)  # [1, 1]
        s2_logits = model.decode_s2(context, s1_argmax)
        s2_probs = F.softmax(s2_logits[:, -1, :], dim=-1).squeeze(0)

    s1_probs_np = s1_probs.cpu().numpy()
    s2_probs_np = s2_probs.cpu().numpy()

    # ── Kronos logit features ──

    # Conviction
    conviction_s1 = float(s1_probs_np.max())
    conviction_s2 = float(s2_probs_np.max())

    # Entropy
    entropy_s1 = float(-(s1_probs_np * np.log(s1_probs_np + 1e-10)).sum())
    entropy_s2 = float(-(s2_probs_np * np.log(s2_probs_np + 1e-10)).sum())

    # Direction: decoded price of argmax token (normalized space)
    s1_argmax_idx = int(s1_probs_np.argmax())
    s2_argmax_idx = int(s2_probs_np.argmax())
    direction_s1 = float(s1_close_map[s1_argmax_idx])
    direction_s2 = float(s2_close_map[s2_argmax_idx])

    # Top-3 direction agreement
    s1_top3 = np.argsort(s1_probs_np)[-3:]
    s1_top3_prices = s1_close_map[s1_top3]
    current_close_norm = x_norm[-1, 1]  # close is index 1 in normalized
    s1_top3_returns = s1_top3_prices - current_close_norm
    top3_agree = float(np.all(np.sign(s1_top3_returns) == np.sign(s1_top3_returns[0])))

    features = {
        "conviction_s1": conviction_s1,
        "conviction_s2": conviction_s2,
        "entropy_s1": entropy_s1,
        "entropy_s2": entropy_s2,
        "direction_s1": direction_s1,
        "direction_s2": direction_s2,
        "top3_direction_agree": top3_agree,
    }

    return features


def extract_candle_features(context_df):
    """Extract candle-derived features from the context window."""
    closes = context_df["close"].values
    highs = context_df["high"].values
    lows = context_df["low"].values
    volumes = context_df["volume"].values

    current_close = closes[-1]

    # Returns at different horizons
    def safe_return(n):
        if len(closes) > n:
            return float((closes[-1] - closes[-1 - n]) / (closes[-1 - n] + 1e-8))
        return 0.0

    # Volatility (std of log returns)
    log_returns = np.diff(np.log(closes + 1e-8))

    def rolling_vol(n):
        if len(log_returns) >= n:
            return float(np.std(log_returns[-n:]))
        return float(np.std(log_returns))

    # Volume ratio
    if len(volumes) > 12:
        vol_ratio = float(volumes[-1] / (np.mean(volumes[-12:]) + 1e-8))
    else:
        vol_ratio = 1.0

    # Intrabar range
    high_low_range = float((highs[-1] - lows[-1]) / (current_close + 1e-8))

    return {
        "return_1": safe_return(1),
        "return_6": safe_return(6),
        "return_12": safe_return(12),
        "volatility_12": rolling_vol(12),
        "volatility_72": rolling_vol(72),
        "volume_ratio": vol_ratio,
        "high_low_range": high_low_range,
    }


def extract_actual_returns(context_df, actual_next_df):
    """Extract ground truth future returns for RL reward."""
    current_close = context_df["close"].iloc[-1]
    features = {}

    actual_closes = actual_next_df["close"].values
    for k in range(PRED_LEN):
        if k < len(actual_closes):
            features[f"actual_return_{k+1}"] = float(
                (actual_closes[k] - current_close) / (current_close + 1e-8)
            )
            features[f"actual_close_{k+1}"] = float(actual_closes[k])
        else:
            features[f"actual_return_{k+1}"] = np.nan
            features[f"actual_close_{k+1}"] = np.nan

    return features


# ──────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────

def load_full_data(pair: str) -> pd.DataFrame:
    path = PREPARED_DIR / f"{pair}_5m_train.csv"
    df = pd.read_csv(path)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df["minute"] = df["timestamps"].dt.minute
    df["hour"] = df["timestamps"].dt.hour
    df["weekday"] = df["timestamps"].dt.weekday
    df["day"] = df["timestamps"].dt.day
    df["month"] = df["timestamps"].dt.month
    return df


def extract_period(df, start, end):
    mask = (df["timestamps"] >= start) & (df["timestamps"] < end)
    return df[mask].reset_index(drop=True)


# ──────────────────────────────────────────────
# Checkpoint save
# ──────────────────────────────────────────────

def save_checkpoint(results, existing_df, output_path):
    new_df = pd.DataFrame(results)
    if existing_df is not None:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    combined.to_parquet(output_path, index=False)
    timestamps = set(pd.to_datetime(combined["timestamp"]))
    return combined, timestamps


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Pre-compute V4 direct logit features at 5min resolution")
    parser.add_argument("--pair", type=str, default="ETH")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--periods", nargs="+", default=None)
    parser.add_argument("--include-val", action="store_true")
    parser.add_argument("--test", action="store_true", help="Run on test set instead of training periods")
    parser.add_argument("--max-hours", type=float, default=5.0)
    parser.add_argument("--save-every", type=int, default=500)
    args = parser.parse_args()

    pair = args.pair.upper()
    max_seconds = args.max_hours * 3600 if args.max_hours > 0 else float("inf")

    tokenizer, model = load_models(pair, device=args.device)

    print("Building token → price map...")
    s1_close_map, s2_close_map = build_token_price_map(tokenizer, args.device)

    # ── Test mode: run on test CSV directly ──
    if args.test:
        test_path = PREPARED_DIR / f"{pair}_5m_test.csv"
        if not test_path.exists():
            print(f"ERROR: Missing {test_path}")
            sys.exit(1)

        test_df = pd.read_csv(test_path)
        test_df["timestamps"] = pd.to_datetime(test_df["timestamps"])
        test_df["minute"] = test_df["timestamps"].dt.minute
        test_df["hour"] = test_df["timestamps"].dt.hour
        test_df["weekday"] = test_df["timestamps"].dt.weekday
        test_df["day"] = test_df["timestamps"].dt.day
        test_df["month"] = test_df["timestamps"].dt.month

        output_path = RL_DIR / f"{pair}_v4_features_test.parquet"

        existing_timestamps = set()
        existing_df = None
        if output_path.exists():
            existing_df = pd.read_parquet(output_path)
            existing_timestamps = set(pd.to_datetime(existing_df["timestamp"]))
            print(f"\nResuming: {len(existing_timestamps):,} steps already computed")

        n_steps = len(test_df) - CONTEXT_LEN - PRED_LEN
        print(f"\nTest set: {len(test_df):,} candles, {n_steps:,} steps")

        total_start = time.time()
        results = []
        total_new = 0

        for step_i in range(n_steps):
            elapsed = time.time() - total_start
            if elapsed >= max_seconds:
                print(f"\n  Time limit reached. Saving.")
                break

            context = test_df.iloc[step_i : step_i + CONTEXT_LEN]
            actual_next = test_df.iloc[step_i + CONTEXT_LEN : step_i + CONTEXT_LEN + PRED_LEN]
            ts = context["timestamps"].iloc[-1]

            if ts in existing_timestamps:
                continue
            if len(actual_next) < PRED_LEN:
                break

            kronos_feats = extract_kronos_features(
                tokenizer, model, context, s1_close_map, s2_close_map, args.device,
            )
            candle_feats = extract_candle_features(context)
            actual_feats = extract_actual_returns(context, actual_next)

            row = {
                "timestamp": ts,
                "close": float(context["close"].iloc[-1]),
                **kronos_feats,
                **candle_feats,
                **actual_feats,
                "period": "test",
            }
            results.append(row)
            total_new += 1

            if len(results) >= args.save_every:
                existing_df, existing_timestamps = save_checkpoint(results, existing_df, output_path)
                print(f"  Checkpoint: {len(existing_df):,} total steps saved")
                results = []

            if total_new > 0 and total_new % 200 == 0:
                speed = total_new / (time.time() - total_start)
                eta = (n_steps - step_i) / speed / 3600 if speed > 0 else 0
                print(f"  [{step_i + 1}/{n_steps}] new={total_new} {speed:.1f} steps/s | ETA: {eta:.1f}h")

        if results:
            existing_df, existing_timestamps = save_checkpoint(results, existing_df, output_path)

        elapsed = time.time() - total_start
        print(f"\nDone. {total_new:,} steps in {elapsed / 60:.1f} min")
        print(f"Output: {output_path}")
        return

    # ── Training mode ──
    periods_path = RL_DIR / f"{pair}_rl_periods.json"
    if not periods_path.exists():
        print(f"ERROR: Missing {periods_path}")
        sys.exit(1)

    with open(periods_path) as f:
        periods_config = json.load(f)

    full_df = load_full_data(pair)

    output_path = RL_DIR / f"{pair}_v4_features.parquet"

    # Resume support
    existing_timestamps = set()
    existing_df = None
    if output_path.exists():
        existing_df = pd.read_parquet(output_path)
        existing_timestamps = set(pd.to_datetime(existing_df["timestamp"]))
        print(f"\nResuming: {len(existing_timestamps):,} steps already computed")

    # Select periods
    periods = dict(periods_config["training_periods"])
    if args.include_val:
        periods["validation"] = periods_config["validation_period"]

    if args.periods:
        periods = {k: v for k, v in periods.items() if k in args.periods}
        if not periods:
            print(f"ERROR: None of {args.periods} found in config")
            sys.exit(1)

    total_start = time.time()
    total_new = 0
    time_exceeded = False

    for name, config in periods.items():
        if time_exceeded:
            break

        period_df = extract_period(full_df, config["start"], config["end"])
        if len(period_df) < CONTEXT_LEN + PRED_LEN:
            print(f"\n  WARNING: {name} too few candles ({len(period_df)}), skipping")
            continue

        n_candles = len(period_df)
        n_steps = n_candles - CONTEXT_LEN - PRED_LEN
        print(f"\n  Period: {name}")
        print(f"  Candles: {n_candles:,}, Steps: {n_steps:,}")

        results = []
        skipped = 0
        period_start = time.time()

        for step_i in range(n_steps):
            elapsed_total = time.time() - total_start
            if elapsed_total >= max_seconds:
                print(f"\n  Time limit reached ({args.max_hours}h). Saving and stopping.")
                time_exceeded = True
                break

            context = period_df.iloc[step_i : step_i + CONTEXT_LEN]
            actual_next = period_df.iloc[step_i + CONTEXT_LEN : step_i + CONTEXT_LEN + PRED_LEN]

            ts = context["timestamps"].iloc[-1]

            if ts in existing_timestamps:
                skipped += 1
                continue

            if len(actual_next) < PRED_LEN:
                break

            # Extract all features
            kronos_feats = extract_kronos_features(
                tokenizer, model, context, s1_close_map, s2_close_map, args.device,
            )
            candle_feats = extract_candle_features(context)
            actual_feats = extract_actual_returns(context, actual_next)

            row = {
                "timestamp": ts,
                "close": float(context["close"].iloc[-1]),
                **kronos_feats,
                **candle_feats,
                **actual_feats,
                "period": name,
            }
            results.append(row)
            total_new += 1

            # Checkpoint
            if len(results) >= args.save_every:
                existing_df, existing_timestamps = save_checkpoint(
                    results, existing_df, output_path,
                )
                print(f"    Checkpoint: {len(existing_df):,} total steps saved")
                results = []

            # Progress
            if total_new > 0 and total_new % 200 == 0:
                elapsed = time.time() - period_start
                speed = (step_i + 1 - skipped) / elapsed if elapsed > 0 else 0
                remaining = n_steps - step_i - 1
                eta = remaining / speed / 3600 if speed > 0 else 0
                time_left = (max_seconds - elapsed_total) / 3600
                print(f"    [{step_i + 1}/{n_steps}] new={total_new} skipped={skipped} "
                      f"{speed:.1f} steps/s | ETA: {eta:.1f}h | time left: {time_left:.1f}h")

        # Save remaining
        if results:
            existing_df, existing_timestamps = save_checkpoint(
                results, existing_df, output_path,
            )

        period_elapsed = time.time() - period_start
        print(f"    {name} done: {period_elapsed / 3600:.1f}h, new={total_new}, skipped={skipped}")

    total_elapsed = time.time() - total_start
    print(f"\nDone. Total: {total_new:,} new steps in {total_elapsed / 3600:.1f}h")
    print(f"Output: {output_path}")
    if existing_df is not None:
        print(f"Total rows: {len(existing_df):,}")
        print(f"Features: {[c for c in existing_df.columns if c not in ['timestamp', 'close', 'period']]}")


if __name__ == "__main__":
    main()
