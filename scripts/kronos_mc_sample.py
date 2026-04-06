"""Monte Carlo sampling from Kronos — generate N predicted futures and analyze.

Usage:
    python scripts/kronos_mc_sample.py --pair ETH --samples 10
    python scripts/kronos_mc_sample.py --pair ETH --samples 10 --pretrained  # use original model for comparison
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent.parent
KRONOS_PATH = PROJECT_ROOT / "third_party" / "kronos"
sys.path.insert(0, str(KRONOS_PATH))

from model import Kronos, KronosTokenizer, KronosPredictor


# ──────────────────────────────────────────────
# 1. Load models
# ──────────────────────────────────────────────

def load_models(pair: str, use_pretrained: bool = False, device: str = "cuda"):
    """Load tokenizer + predictor, either fine-tuned or pre-trained."""
    if use_pretrained:
        tok_path = str(PROJECT_ROOT / "data/ml/pretrained/Kronos-Tokenizer-base")
        pred_path = str(PROJECT_ROOT / "data/ml/pretrained/Kronos-base")
        print(f"Loading PRE-TRAINED models")
    else:
        base = PROJECT_ROOT / "data/ml/kronos/finetuned" / f"{pair.lower()}_5m"
        tok_path = str(base / "tokenizer" / "best_model")
        pred_path = str(base / "predictor" / "best_model")
        print(f"Loading FINE-TUNED models from {base}")

    tokenizer = KronosTokenizer.from_pretrained(tok_path).to(device).eval()
    model = Kronos.from_pretrained(pred_path).to(device).eval()
    predictor = KronosPredictor(model, tokenizer, max_context=512)

    print(f"  Tokenizer: {tok_path}")
    print(f"  Predictor: {pred_path}")
    return predictor


# ──────────────────────────────────────────────
# 2. Load recent candle data (context window)
# ──────────────────────────────────────────────

def load_context(pair: str, context_len: int = 512):
    """Load the last context_len candles from the test set as context."""
    test_path = PROJECT_ROOT / f"data/ml/kronos/prepared/{pair}_5m_test.csv"
    df = pd.read_csv(test_path)
    df["timestamps"] = pd.to_datetime(df["timestamps"])

    # Take a window from the middle of the test set (so we have actuals to compare)
    start_idx = len(df) // 2
    context = df.iloc[start_idx : start_idx + context_len].copy()
    actuals = df.iloc[start_idx + context_len : start_idx + context_len + 10].copy()

    print(f"\n  Context: {context['timestamps'].iloc[0]} → {context['timestamps'].iloc[-1]}")
    print(f"  Predicting: next 10 candles after {context['timestamps'].iloc[-1]}")
    print(f"  Actuals available: {len(actuals)} candles")

    return context, actuals


# ──────────────────────────────────────────────
# 3. Monte Carlo sampling — the core
# ──────────────────────────────────────────────

def mc_sample(predictor, context_df, n_samples: int = 10, pred_len: int = 10,
              temperature: float = 1.0, top_p: float = 0.9):
    """Sample N predicted paths from Kronos.

    Each sample uses different randomness in the autoregressive decoding,
    producing a different plausible future. The collection of paths
    IS our uncertainty estimate.
    """
    x_df = context_df[["open", "close", "high", "low", "volume", "amount"]]
    x_timestamps = context_df["timestamps"]

    # Generate future timestamps (5-min intervals)
    last_ts = x_timestamps.iloc[-1]
    y_timestamps = pd.Series(pd.date_range(start=last_ts + pd.Timedelta(minutes=5),
                                           periods=pred_len, freq="5min"))

    paths = []
    for i in range(n_samples):
        pred_df = predictor.predict(
            df=x_df,
            x_timestamp=x_timestamps,
            y_timestamp=y_timestamps,
            pred_len=pred_len,
            T=temperature,        # higher = more diverse paths
            top_p=top_p,          # nucleus sampling
            sample_count=1,       # 1 sample per call (we loop ourselves for diversity)
            verbose=False,
        )
        paths.append(pred_df)
        print(f"  Sample {i+1}/{n_samples}: final close = {pred_df['close'].iloc[-1]:.2f}")

    return paths, y_timestamps


# ──────────────────────────────────────────────
# 4. Analyze MC paths — extract trading features
# ──────────────────────────────────────────────

def analyze_paths(paths, current_close: float, actuals_df=None):
    """Extract consensus metrics, risk metrics, and path agreement from MC samples."""
    n = len(paths)

    # Collect final returns and per-candle returns for each path
    final_returns = []
    optimal_long_returns = []
    optimal_short_returns = []
    mae_longs = []
    mae_shorts = []

    for path_df in paths:
        closes = path_df["close"].values
        highs = path_df["high"].values
        lows = path_df["low"].values

        # Returns relative to current price
        final_ret = (closes[-1] - current_close) / current_close
        opt_long = (max(closes) - current_close) / current_close
        opt_short = (current_close - min(closes)) / current_close
        mae_long = (min(lows) - current_close) / current_close      # worst dip for long
        mae_short = (max(highs) - current_close) / current_close    # worst spike for short

        final_returns.append(final_ret)
        optimal_long_returns.append(opt_long)
        optimal_short_returns.append(opt_short)
        mae_longs.append(mae_long)
        mae_shorts.append(mae_short)

    final_returns = np.array(final_returns)
    optimal_long_returns = np.array(optimal_long_returns)
    optimal_short_returns = np.array(optimal_short_returns)

    # ── Consensus metrics ──
    p_long = np.mean(final_returns > 0)         # fraction of paths predicting up
    p_short = np.mean(final_returns < 0)         # fraction predicting down
    mu_return = np.mean(final_returns)            # mean predicted return
    sigma_return = np.std(final_returns)           # uncertainty (MC spread)
    mu_opt_long = np.mean(optimal_long_returns)    # mean best long exit
    mu_opt_short = np.mean(optimal_short_returns)  # mean best short exit

    # ── Risk metrics ──
    worst_mae_long = min(mae_longs)               # worst drawdown across all paths
    worst_mae_short = max(mae_shorts)              # worst adverse move for short
    p_sl_long = np.mean(np.array(mae_longs) < -0.02)   # prob of hitting 2% SL
    p_sl_short = np.mean(np.array(mae_shorts) > 0.02)

    # ── Path agreement ──
    # Do paths agree on direction at each candle?
    direction_agreement = []
    for k in range(len(paths[0])):
        directions = [1 if p["close"].iloc[k] > current_close else -1 for p in paths]
        agreement = abs(sum(directions)) / n   # 1.0 = all agree, 0.0 = split
        direction_agreement.append(agreement)
    avg_agreement = np.mean(direction_agreement)

    print("\n" + "=" * 60)
    print("MONTE CARLO ANALYSIS")
    print("=" * 60)
    print(f"  Current close:       {current_close:.2f}")
    print(f"  Samples:             {n}")
    print()
    print(f"  ── Consensus ──")
    print(f"  P(long profitable):  {p_long:.0%}")
    print(f"  P(short profitable): {p_short:.0%}")
    print(f"  Mean final return:   {mu_return:+.4%}")
    print(f"  Return std (σ):      {sigma_return:.4%}  ← MC uncertainty")
    print(f"  Mean optimal long:   {mu_opt_long:+.4%}")
    print(f"  Mean optimal short:  {mu_opt_short:+.4%}")
    print()
    print(f"  ── Risk ──")
    print(f"  Worst MAE (long):    {worst_mae_long:+.4%}")
    print(f"  Worst MAE (short):   {worst_mae_short:+.4%}")
    print(f"  P(hit 2% SL long):   {p_sl_long:.0%}")
    print(f"  P(hit 2% SL short):  {p_sl_short:.0%}")
    print()
    print(f"  ── Path Agreement ──")
    print(f"  Avg agreement:       {avg_agreement:.2f}  (1.0 = all paths agree)")
    print()

    # ── Compare to actuals if available ──
    if actuals_df is not None and len(actuals_df) > 0:
        actual_final = actuals_df["close"].iloc[-1]
        actual_return = (actual_final - current_close) / current_close
        actual_direction = "UP" if actual_return > 0 else "DOWN"
        predicted_direction = "UP" if mu_return > 0 else "DOWN"
        correct = actual_direction == predicted_direction

        print(f"  ── vs Actuals ──")
        print(f"  Actual return:       {actual_return:+.4%} ({actual_direction})")
        print(f"  Predicted direction: {predicted_direction}  {'✓' if correct else '✗'}")
        print(f"  Prediction error:    {abs(mu_return - actual_return):.4%}")
        print()

    # Return features as dict (this is what the RL agent would consume)
    return {
        "p_long": p_long,
        "p_short": p_short,
        "mu_return": mu_return,
        "sigma_return": sigma_return,
        "mu_opt_long": mu_opt_long,
        "mu_opt_short": mu_opt_short,
        "worst_mae_long": worst_mae_long,
        "worst_mae_short": worst_mae_short,
        "p_sl_long": p_sl_long,
        "p_sl_short": p_sl_short,
        "avg_agreement": avg_agreement,
    }


# ──────────────────────────────────────────────
# 5. Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Kronos Monte Carlo Sampling")
    parser.add_argument("--pair", type=str, default="ETH", help="Pair (ETH or BTC)")
    parser.add_argument("--samples", type=int, default=10, help="Number of MC samples")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.9, help="Nucleus sampling threshold")
    parser.add_argument("--pretrained", action="store_true", help="Use pre-trained model instead of fine-tuned")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    args = parser.parse_args()

    # Load
    predictor = load_models(args.pair, use_pretrained=args.pretrained, device=args.device)
    context_df, actuals_df = load_context(args.pair)

    current_close = context_df["close"].iloc[-1]

    # Sample N paths
    print(f"\nSampling {args.samples} paths (T={args.temperature}, top_p={args.top_p})...\n")
    paths, timestamps = mc_sample(
        predictor, context_df,
        n_samples=args.samples,
        temperature=args.temperature,
        top_p=args.top_p,
    )

    # Analyze
    features = analyze_paths(paths, current_close, actuals_df)

    # This features dict is what the RL agent would receive as state
    print("  ── RL State Vector ──")
    for k, v in features.items():
        print(f"  {k:20s}: {v:.4f}")


if __name__ == "__main__":
    main()
