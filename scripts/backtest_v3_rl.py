"""
Proper backtest of V3 RL conviction strategy using precomputed MC features.

Tests two things:
  1. Full V3: RL model with all 27 features (MC + candle)
  2. Candle-only baseline: same RL model but MC features zeroed out

If V3 performs the same with MC zeroed → Kronos contributed nothing.

Usage:
    python scripts/backtest_v3_rl.py
    python scripts/backtest_v3_rl.py --periods validation test_crash test_trend test_chop
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from ownbot.strategy.feature_builder import FeatureBuilder, MC_FEATURE_COLS


def load_rl_model(model_path, device="cpu"):
    """Load the V3 RecurrentPPO model."""
    from sb3_contrib import RecurrentPPO
    model = RecurrentPPO.load(model_path, device=device)
    return model


def load_mc_data(periods=None):
    """Load precomputed MC features."""
    df = pd.read_parquet(PROJECT_ROOT / "data/ml/rl/ETH_mc_features.parquet")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if periods:
        df = df[df["period"].isin(periods)]
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def build_candle_df_from_mc(mc_row, history_closes):
    """Build a minimal candle DataFrame for FeatureBuilder.

    The FeatureBuilder needs a DataFrame with 'close' column
    to compute ret_1, ret_6, ret_24, vol_12, vol_48.
    """
    closes = np.array(history_closes)
    df = pd.DataFrame({"close": closes})
    return df


def run_backtest(model, mc_df, zero_mc=False, fee_pct=0.0007,
                 entry_threshold=0.6, exit_threshold=0.3):
    """Run the RL model on precomputed features and simulate trading."""
    fb = FeatureBuilder()

    # LSTM hidden state
    lstm_states = None
    episode_starts = np.array([True])

    # Position tracking
    position = 0  # 0=flat, 1=long, -1=short
    entry_price = 0
    trades = []
    equity_curve = [10000.0]
    balance = 10000.0

    # Build close price history for candle features
    close_history = []

    for i in range(len(mc_df)):
        row = mc_df.iloc[i]
        close = row["close"]
        close_history.append(close)

        # Build MC features dict
        mc_features = {}
        for col in MC_FEATURE_COLS:
            if col in row.index:
                mc_features[col] = 0.0 if zero_mc else float(row[col])
            else:
                mc_features[col] = 0.0

        # Update feature builder with MC data
        fb.update_mc(mc_features)

        if not fb.is_warm or len(close_history) < 50:
            equity_curve.append(balance)
            continue

        # Build candle DataFrame
        candle_df = pd.DataFrame({"close": close_history[-512:]})

        # Build 27-feature observation
        obs = fb.build(mc_features, candle_df)

        # Run RL model
        action, lstm_states = model.predict(
            obs.reshape(1, -1),
            state=lstm_states,
            episode_start=episode_starts,
            deterministic=True,
        )
        episode_starts = np.array([False])

        conviction = float(action[0])
        fb.prev_action = conviction

        # Trading logic (mirrors SignalInterpreter)
        actual_return = row.get("actual_return_1", 0)
        if pd.isna(actual_return):
            actual_return = 0

        if position == 0:
            # Entry
            if conviction > entry_threshold:
                position = 1
                entry_price = close
                balance -= balance * fee_pct  # entry fee
            elif conviction < -entry_threshold:
                position = -1
                entry_price = close
                balance -= balance * fee_pct
        else:
            # Exit conditions
            should_exit = False

            # Conviction faded
            if abs(conviction) < exit_threshold:
                should_exit = True
            # Conviction flipped direction
            elif (position == 1 and conviction < -exit_threshold):
                should_exit = True
            elif (position == -1 and conviction > exit_threshold):
                should_exit = True

            if should_exit:
                # Calculate PnL
                if position == 1:
                    pnl_pct = (close - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - close) / entry_price

                balance *= (1 + pnl_pct)
                balance -= balance * fee_pct  # exit fee

                trades.append({
                    "entry_price": entry_price,
                    "exit_price": close,
                    "direction": "long" if position == 1 else "short",
                    "pnl_pct": pnl_pct * 100,
                    "timestamp": row["timestamp"],
                })

                # Check for immediate re-entry in opposite direction
                if conviction > entry_threshold:
                    position = 1
                    entry_price = close
                    balance -= balance * fee_pct
                elif conviction < -entry_threshold:
                    position = -1
                    entry_price = close
                    balance -= balance * fee_pct
                else:
                    position = 0

        equity_curve.append(balance)

    # Close any open position at the end
    if position != 0:
        close = mc_df.iloc[-1]["close"]
        if position == 1:
            pnl_pct = (close - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - close) / entry_price
        balance *= (1 + pnl_pct)
        trades.append({
            "entry_price": entry_price,
            "exit_price": close,
            "direction": "long" if position == 1 else "short",
            "pnl_pct": pnl_pct * 100,
            "timestamp": mc_df.iloc[-1]["timestamp"],
        })

    return trades, equity_curve, balance


def print_results(label, trades, equity_curve, initial=10000):
    final = equity_curve[-1]
    total_return = (final / initial - 1) * 100

    if not trades:
        print(f"\n{label}: NO TRADES")
        return

    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df["pnl_pct"] > 0]
    losses = trades_df[trades_df["pnl_pct"] <= 0]

    # Sharpe (approximate)
    returns = np.diff(equity_curve) / equity_curve[:-1]
    returns = returns[returns != 0]  # remove zero returns
    sharpe = returns.mean() / returns.std() * np.sqrt(48 * 365) if len(returns) > 1 and returns.std() > 0 else 0

    # Max drawdown
    peak = np.maximum.accumulate(equity_curve)
    dd = (np.array(equity_curve) - peak) / peak
    max_dd = dd.min() * 100

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Total return:  {total_return:+.1f}%")
    print(f"  Trades:        {len(trades)}")
    print(f"  Win rate:      {len(wins)/len(trades)*100:.1f}% ({len(wins)}W / {len(losses)}L)")
    print(f"  Avg win:       {wins['pnl_pct'].mean():+.2f}%" if len(wins) else "  Avg win:       N/A")
    print(f"  Avg loss:      {losses['pnl_pct'].mean():+.2f}%" if len(losses) else "  Avg loss:      N/A")
    print(f"  Best trade:    {trades_df['pnl_pct'].max():+.2f}%")
    print(f"  Worst trade:   {trades_df['pnl_pct'].min():+.2f}%")
    print(f"  Sharpe:        {sharpe:.2f}")
    print(f"  Max drawdown:  {max_dd:.1f}%")
    print(f"  Profit factor: {wins['pnl_pct'].sum() / abs(losses['pnl_pct'].sum()):.2f}" if len(losses) and losses['pnl_pct'].sum() != 0 else "  Profit factor: inf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--periods", nargs="+", default=None,
                        help="Periods to test (default: all)")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--entry", type=float, default=0.6)
    parser.add_argument("--exit", type=float, default=0.3)
    args = parser.parse_args()

    model_path = PROJECT_ROOT / "data/ml/rl/models/conviction_lstm128x1_mlp64_linear/best_model/best_model.zip"
    print(f"Loading RL model from {model_path}...")
    model = load_rl_model(str(model_path), device=args.device)

    print("Loading precomputed MC features...")
    mc_df = load_mc_data(periods=args.periods)
    print(f"  {len(mc_df)} rows, periods: {mc_df['period'].unique().tolist()}")

    # Test 1: Full V3 (MC + candle features)
    print("\nRunning V3 FULL (MC + candle features)...")
    trades_full, eq_full, bal_full = run_backtest(
        model, mc_df, zero_mc=False,
        entry_threshold=args.entry, exit_threshold=args.exit,
    )
    print_results("V3 FULL (MC + candle)", trades_full, eq_full)

    # Test 2: Candle-only (MC features zeroed out)
    print("\nRunning V3 CANDLE-ONLY (MC zeroed)...")
    trades_candle, eq_candle, bal_candle = run_backtest(
        model, mc_df, zero_mc=True,
        entry_threshold=args.entry, exit_threshold=args.exit,
    )
    print_results("V3 CANDLE-ONLY (MC zeroed)", trades_candle, eq_candle)

    # Test 3: Buy and hold comparison
    first_close = mc_df.iloc[50]["close"]
    last_close = mc_df.iloc[-1]["close"]
    bh_return = (last_close / first_close - 1) * 100
    print(f"\n{'='*60}")
    print(f"  BUY & HOLD: {bh_return:+.1f}% ({first_close:.0f} → {last_close:.0f})")
    print(f"{'='*60}")

    # Verdict
    print(f"\n{'='*60}")
    print(f"  VERDICT: Does Kronos MC add value?")
    print(f"{'='*60}")
    full_ret = (bal_full / 10000 - 1) * 100
    candle_ret = (bal_candle / 10000 - 1) * 100
    diff = full_ret - candle_ret
    print(f"  Full V3:      {full_ret:+.1f}%")
    print(f"  Candle-only:  {candle_ret:+.1f}%")
    print(f"  MC added:     {diff:+.1f}pp")
    if abs(diff) < 5:
        print(f"  → MC features contributed NOTHING meaningful")
    elif diff > 5:
        print(f"  → MC features HELPED (+{diff:.1f}pp)")
    else:
        print(f"  → MC features HURT ({diff:.1f}pp)")


if __name__ == "__main__":
    main()
