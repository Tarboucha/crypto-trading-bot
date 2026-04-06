"""Pre-compute Kronos MC features at native 5min resolution (subsample=1).

Same logic as precompute_mc_features.py but without subsampling, producing
~6x more data for training the RL agent on 5min candles directly.

Supports resume: skips timestamps already present in the output parquet.
Supports time-boxing: --max-hours to stop after N hours (saves and exits cleanly).

Usage:
    # Run a 5-hour chunk (resume-safe, re-run same command to continue)
    python scripts/precompute_mc_features_5m.py --pair ETH --samples 5 --max-hours 5

    # Run specific periods only
    python scripts/precompute_mc_features_5m.py --pair ETH --samples 5 --max-hours 5 --periods recent_market

    # Include validation
    python scripts/precompute_mc_features_5m.py --pair ETH --samples 5 --max-hours 5 --include-val
"""
import argparse
import json
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


def load_models(pair: str, device: str = "cuda"):
    base = PROJECT_ROOT / "data/ml/kronos/finetuned" / f"{pair.lower()}_5m"
    tok_path = str(base / "tokenizer" / "best_model")
    pred_path = str(base / "predictor" / "best_model")
    print(f"Loading FINE-TUNED models from {base}")

    tokenizer = KronosTokenizer.from_pretrained(tok_path).to(device).eval()
    model = Kronos.from_pretrained(pred_path).to(device).eval()
    predictor = KronosPredictor(model, tokenizer, max_context=CONTEXT_LEN)
    return predictor


def load_full_data(pair: str) -> pd.DataFrame:
    path = PREPARED_DIR / f"{pair}_5m_train.csv"
    df = pd.read_csv(path)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    return df


def extract_period(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    mask = (df["timestamps"] >= start) & (df["timestamps"] < end)
    return df[mask].reset_index(drop=True)


def mc_sample_features(predictor, context_df: pd.DataFrame, actual_next: pd.DataFrame,
                       n_samples: int, temperature: float, top_p: float) -> dict:
    """Run N MC samples and extract 28 MC features for one timestep.

    Features are organized in tiers:
      - Original 11: direction, returns, extremes, SL probability, agreement
      - Tier 1 (6): distribution shape — sharpe, skew, kurtosis, tails
      - Tier 2 (5): path-walking — first passage, mid-return, reversal, time-to-exit
      - Tier 3 (6): robust risk — CVaR, bimodality, edge
    """
    from scipy import stats as sp_stats

    x_df = context_df[["open", "close", "high", "low", "volume", "amount"]]
    x_timestamps = context_df["timestamps"]
    current_close = context_df["close"].iloc[-1]

    last_ts = x_timestamps.iloc[-1]
    y_timestamps = pd.Series(pd.date_range(
        start=last_ts + pd.Timedelta(minutes=5), periods=PRED_LEN, freq="5min"
    ))

    # Collect per-path data
    final_returns = []
    optimal_long_returns = []
    optimal_short_returns = []
    mae_longs = []
    mae_shorts = []
    mid_returns = []              # return at candle 5 (midpoint)
    path_closes_all = []          # per-path close arrays (for reversal/monotonicity)
    path_highs_all = []           # per-path high arrays (for first-passage)
    path_lows_all = []            # per-path low arrays (for first-passage)

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

        # Mid-return (candle 5, index 4)
        mid_idx = min(4, len(closes) - 1)
        mid_returns.append((closes[mid_idx] - current_close) / current_close)

        path_closes_all.append(closes)
        path_highs_all.append(highs)
        path_lows_all.append(lows)

    final_returns = np.array(final_returns)
    mae_longs = np.array(mae_longs)
    mae_shorts = np.array(mae_shorts)
    n = len(final_returns)

    # ══════════════════════════════════════════════
    # ORIGINAL 11 FEATURES
    # ══════════════════════════════════════════════

    mu_return = float(np.mean(final_returns))
    sigma_return = float(np.std(final_returns))
    mu_opt_long = float(np.mean(optimal_long_returns))
    mu_opt_short = float(np.mean(optimal_short_returns))
    worst_mae_long = float(np.min(mae_longs))
    worst_mae_short = float(np.max(mae_shorts))

    features = {
        "timestamp": context_df["timestamps"].iloc[-1],
        "close": current_close,
        "p_long": float(np.mean(final_returns > 0)),
        "p_short": float(np.mean(final_returns < 0)),
        "mu_return": mu_return,
        "sigma_return": sigma_return,
        "mu_opt_long": mu_opt_long,
        "mu_opt_short": mu_opt_short,
        "worst_mae_long": worst_mae_long,
        "worst_mae_short": worst_mae_short,
        "p_sl_long_2pct": float(np.mean(mae_longs < -0.02)),
        "p_sl_short_2pct": float(np.mean(mae_shorts > 0.02)),
        "avg_agreement": float(abs(np.mean(np.sign(final_returns)))),
    }

    # ══════════════════════════════════════════════
    # TIER 1: DISTRIBUTION SHAPE (6 features)
    # ══════════════════════════════════════════════

    features["predicted_sharpe"] = mu_return / (sigma_return + 1e-8)

    if n >= 3:
        features["skew_return"] = float(sp_stats.skew(final_returns))
        features["kurt_return"] = float(sp_stats.kurtosis(final_returns))
    else:
        features["skew_return"] = 0.0
        features["kurt_return"] = 0.0

    p5 = float(np.percentile(final_returns, 5))
    p95 = float(np.percentile(final_returns, 95))
    features["p5_return"] = p5
    features["p95_return"] = p95
    features["tail_ratio"] = abs(p95) / (abs(p5) + 1e-8)

    # ══════════════════════════════════════════════
    # TIER 2: PATH-WALKING (5 features)
    # ══════════════════════════════════════════════

    # First-passage: does TP (+2%) hit before SL (-2%) for long?
    tp_before_sl_long = 0
    tp_before_sl_short = 0
    time_to_exit_steps = []

    for i in range(n):
        highs = path_highs_all[i]
        lows = path_lows_all[i]
        tp_long_price = current_close * 1.02
        sl_long_price = current_close * 0.98
        tp_short_price = current_close * 0.98
        sl_short_price = current_close * 1.02

        # Long first-passage
        long_resolved = False
        for k in range(len(highs)):
            if highs[k] >= tp_long_price:
                tp_before_sl_long += 1
                long_resolved = True
                break
            if lows[k] <= sl_long_price:
                long_resolved = True
                break

        # Short first-passage
        for k in range(len(lows)):
            if lows[k] <= tp_short_price:
                tp_before_sl_short += 1
                break
            if highs[k] >= sl_short_price:
                break

        # Time to exit (either TP or SL for long)
        exit_candle = len(highs)  # default: never exits
        for k in range(len(highs)):
            if highs[k] >= tp_long_price or lows[k] <= sl_long_price:
                exit_candle = k + 1
                break
        time_to_exit_steps.append(exit_candle)

    features["p_tp_before_sl_long"] = tp_before_sl_long / n
    features["p_tp_before_sl_short"] = tp_before_sl_short / n
    features["mean_time_to_exit"] = float(np.mean(time_to_exit_steps)) / PRED_LEN

    # Mid-return (path shape)
    features["mu_return_mid"] = float(np.mean(mid_returns))

    # Reversal rate: fraction of paths where direction at midpoint differs from final
    mid_signs = np.sign(mid_returns)
    final_signs = np.sign(final_returns)
    reversals = np.sum((mid_signs != final_signs) & (mid_signs != 0) & (final_signs != 0))
    features["path_reversal_rate"] = float(reversals / max(1, np.sum((mid_signs != 0) & (final_signs != 0))))

    # ══════════════════════════════════════════════
    # TIER 3: ROBUST RISK (6 features)
    # ══════════════════════════════════════════════

    # CVaR (expected shortfall) — mean of worst 5% (at least 1 path)
    n_tail = max(1, int(0.05 * n))
    sorted_mae_long = np.sort(mae_longs)
    sorted_mae_short = np.sort(mae_shorts)[::-1]
    features["cvar_long_5pct"] = float(np.mean(sorted_mae_long[:n_tail]))
    features["cvar_short_5pct"] = float(np.mean(sorted_mae_short[:n_tail]))

    # Bimodality score (Sarle's coefficient)
    skew = features["skew_return"]
    kurt = features["kurt_return"]
    features["bimodality_score"] = (skew ** 2 + 1) / (kurt + 3) if (kurt + 3) != 0 else 0.0

    # Edge: reward-to-risk ratio
    features["edge_long"] = mu_opt_long - abs(worst_mae_long)
    features["edge_short"] = mu_opt_short - abs(worst_mae_short)

    # ══════════════════════════════════════════════
    # ACTUAL FUTURE RETURNS (for RL reward)
    # ══════════════════════════════════════════════

    actual_closes = actual_next["close"].values
    actual_returns = [(c - current_close) / current_close for c in actual_closes]

    for k in range(PRED_LEN):
        if k < len(actual_returns):
            features[f"actual_return_{k+1}"] = actual_returns[k]
            features[f"actual_close_{k+1}"] = float(actual_closes[k])
        else:
            features[f"actual_return_{k+1}"] = np.nan
            features[f"actual_close_{k+1}"] = np.nan

    return features


def save_checkpoint(results: list, existing_df: pd.DataFrame | None,
                    output_path: Path) -> tuple[pd.DataFrame, set]:
    """Save new results merged with existing data. Returns updated df and timestamps."""
    new_df = pd.DataFrame(results)
    if existing_df is not None:
        combined = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        combined = new_df
    combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    combined.to_parquet(output_path, index=False)
    timestamps = set(pd.to_datetime(combined["timestamp"]))
    return combined, timestamps


def main():
    parser = argparse.ArgumentParser(description="Pre-compute Kronos MC features at 5min resolution")
    parser.add_argument("--pair", type=str, default="ETH")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--include-val", action="store_true", help="Include validation period")
    parser.add_argument("--periods", nargs="+", default=None,
                        help="Only compute specific periods (e.g., --periods recent_market etf_rally)")
    parser.add_argument("--max-hours", type=float, default=5.0,
                        help="Stop after N hours and save (default: 5). Use 0 for unlimited.")
    parser.add_argument("--save-every", type=int, default=500,
                        help="Save checkpoint every N new steps (default: 500)")
    args = parser.parse_args()

    pair = args.pair.upper()
    max_seconds = args.max_hours * 3600 if args.max_hours > 0 else float("inf")

    periods_path = RL_DIR / f"{pair}_rl_periods.json"
    if not periods_path.exists():
        print(f"ERROR: Missing {periods_path}")
        sys.exit(1)

    with open(periods_path) as f:
        periods_config = json.load(f)

    predictor = load_models(pair, device=args.device)
    full_df = load_full_data(pair)

    output_path = RL_DIR / f"{pair}_mc_features_5m.parquet"

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
        print(f"  Candles: {n_candles:,}, Steps: {n_steps:,} (every 5min candle)")

        results = []
        skipped = 0
        period_start = time.time()

        for step_i in range(n_steps):
            # Time check
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

            features = mc_sample_features(
                predictor, context, actual_next,
                n_samples=args.samples, temperature=args.temperature, top_p=args.top_p,
            )
            features["period"] = name
            results.append(features)
            total_new += 1

            # Periodic checkpoint save
            if len(results) >= args.save_every:
                existing_df, existing_timestamps = save_checkpoint(
                    results, existing_df, output_path,
                )
                print(f"    Checkpoint: {len(existing_df):,} total steps saved")
                results = []

            # Progress logging
            computed = total_new
            if computed > 0 and computed % 200 == 0:
                elapsed = time.time() - period_start
                speed = (step_i + 1 - skipped) / elapsed if elapsed > 0 else 0
                remaining_period = n_steps - step_i - 1
                eta_period = remaining_period / speed / 3600 if speed > 0 else 0
                time_left = (max_seconds - elapsed_total) / 3600
                print(f"    [{step_i + 1}/{n_steps}] new={computed} skipped={skipped} "
                      f"{speed:.1f} steps/s | period ETA: {eta_period:.1f}h | "
                      f"time left: {time_left:.1f}h")

        # Save remaining results for this period
        if results:
            existing_df, existing_timestamps = save_checkpoint(
                results, existing_df, output_path,
            )

        period_elapsed = time.time() - period_start
        print(f"    {name} done: {period_elapsed / 3600:.1f}h, skipped={skipped}")
        if existing_df is not None:
            print(f"    Total saved: {len(existing_df):,} steps")

    total_time = time.time() - total_start
    print(f"\n{'=' * 60}")
    print(f"{'STOPPED (time limit)' if time_exceeded else 'COMPLETED'}")
    print(f"Run time: {total_time / 3600:.1f}h")
    print(f"New steps this run: {total_new:,}")
    print(f"Output: {output_path}")
    if existing_df is not None:
        print(f"Total steps in file: {len(existing_df):,}")
        for p, count in existing_df["period"].value_counts().items():
            print(f"  {p}: {count:,}")
    if time_exceeded:
        print(f"\nRe-run the same command to continue from where you left off.")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
