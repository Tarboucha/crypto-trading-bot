"""Train an RL agent on pre-computed Kronos MC features (v3 — RecurrentPPO with LSTM).

Uses RecurrentPPO from sb3-contrib with LSTM policy.
The agent sees sequences of observations and can detect temporal patterns
in MC predictions and market state.

Usage:
    # Train for 1M steps
    python scripts/train_rl_agent.py --steps 1000000

    # Evaluate
    python scripts/train_rl_agent.py --eval

    # Baselines
    python scripts/train_rl_agent.py --baselines
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mllab.rl.trading_env import make_env_from_parquet, OBS_DIM


def train(parquet_path: str, total_steps: int, save_dir: str, eval_freq: int = 10_000):
    """Train RecurrentPPO agent with LSTM policy."""
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor

    os.makedirs(save_dir, exist_ok=True)

    print("Creating training environment...")
    train_env = Monitor(make_env_from_parquet(parquet_path, episode_length=500))
    eval_env = Monitor(make_env_from_parquet(parquet_path, episode_length=500))

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, "best_model"),
        log_path=os.path.join(save_dir, "eval_logs"),
        eval_freq=eval_freq,
        n_eval_episodes=10,
        deterministic=False,
        verbose=1,
    )

    model = RecurrentPPO(
        "MlpLstmPolicy",
        train_env,
        n_steps=500,                # full episode per rollout
        batch_size=250,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        device="auto",
        policy_kwargs=dict(
            lstm_hidden_size=128,
            n_lstm_layers=1,
            shared_lstm=True,       # single LSTM shared between policy and value
            enable_critic_lstm=False,
            net_arch=dict(pi=[64], vf=[64]),
        ),
    )

    print(f"\nTraining RecurrentPPO (LSTM) for {total_steps:,} steps...")
    print(f"  Save dir: {save_dir}")
    print(f"  Eval every: {eval_freq:,} steps")
    print(f"  Action space: continuous [-3.0, +3.0]")
    print(f"  Observation: {OBS_DIM} features")
    print(f"  Episode length: 500 steps")
    print(f"  Policy: LSTM(128) → MLP(64)")
    print()

    model.learn(total_timesteps=total_steps, callback=eval_callback, progress_bar=True)

    final_path = os.path.join(save_dir, "final_model")
    model.save(final_path)
    print(f"\nFinal model saved to {final_path}")

    return model


def evaluate(model_path: str, parquet_path: str, n_episodes: int = 10, base_seed: int = 42):
    """Evaluate a trained agent."""
    from sb3_contrib import RecurrentPPO

    print(f"Loading model from {model_path}")
    model = RecurrentPPO.load(model_path, device="cpu")

    env = make_env_from_parquet(parquet_path, episode_length=500)

    all_rewards = []
    all_balances = []
    all_trades = []
    all_max_dd = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 1000 + base_seed)
        total_reward = 0
        trades = 0
        prev_exposure = 0.0
        max_dd = 0.0

        # LSTM needs hidden state
        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)

        while True:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_start, deterministic=True
            )
            episode_start = np.zeros((1,), dtype=bool)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            if abs(info["exposure"] - prev_exposure) > 0.3:
                trades += 1
            prev_exposure = info["exposure"]
            max_dd = max(max_dd, info["drawdown"])

            if terminated or truncated:
                break

        final_balance = info["balance"]
        pnl_pct = (final_balance - 10_000) / 10_000 * 100

        all_rewards.append(total_reward)
        all_balances.append(final_balance)
        all_trades.append(trades)
        all_max_dd.append(max_dd)

        print(f"  Episode {ep + 1:>2}: reward={total_reward:>7.2f}, "
              f"P&L={pnl_pct:>+6.2f}%, trades={trades:>3}, "
              f"max_dd={max_dd:.2%}, final_exp={info['exposure']:+.1f}")

    print(f"\n{'=' * 60}")
    print(f"EVALUATION SUMMARY ({n_episodes} episodes)")
    print(f"{'=' * 60}")
    print(f"  Mean reward:    {np.mean(all_rewards):.2f} ± {np.std(all_rewards):.2f}")
    print(f"  Mean P&L:       {np.mean([(b - 10_000) / 100 for b in all_balances]):+.2f}%")
    print(f"  Mean trades:    {np.mean(all_trades):.1f}")
    print(f"  Mean max DD:    {np.mean(all_max_dd):.2%}")
    print(f"  Win rate:       {np.mean([b > 10_000 for b in all_balances]):.0%}")
    print(f"{'=' * 60}")


def evaluate_baselines(parquet_path: str, n_episodes: int = 10, base_seed: int = 42):
    """Evaluate simple baselines."""
    env = make_env_from_parquet(parquet_path, episode_length=500)

    strategies = {
        "always_flat": 0.0,
        "always_long_1x": 1.0,
        "always_short_1x": -1.0,
        "always_long_2x": 2.0,
    }

    print(f"\n{'=' * 60}")
    print("BASELINES")
    print(f"{'=' * 60}")

    for name, fixed_exposure in strategies.items():
        rewards = []
        balances = []
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=ep * 1000 + base_seed)
            total_reward = 0
            while True:
                action = np.array([fixed_exposure], dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if terminated or truncated:
                    break
            rewards.append(total_reward)
            balances.append(info["balance"])

        mean_pnl = np.mean([(b - 10_000) / 100 for b in balances])
        mean_reward = np.mean(rewards)
        print(f"  {name:<20}: reward={mean_reward:>7.2f}, P&L={mean_pnl:>+6.2f}%")

    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate RL trading agent (v3 — LSTM)")
    parser.add_argument("--pair", type=str, default="ETH")
    parser.add_argument("--steps", type=int, default=1_000_000, help="Training steps (default: 1M)")
    parser.add_argument("--eval", action="store_true", help="Evaluate instead of train")
    parser.add_argument("--model", type=str, default=None, help="Model path for evaluation")
    parser.add_argument("--baselines", action="store_true", help="Run baseline comparison")
    parser.add_argument("--eval-freq", type=int, default=10_000, help="Eval frequency during training")
    parser.add_argument("--seed", type=int, default=42, help="Base seed for eval/baselines (default: 42)")
    parser.add_argument("--episodes", type=int, default=10, help="Number of eval episodes (default: 10)")
    parser.add_argument("--val-only", action="store_true", help="Eval only on validation period (unseen data)")
    args = parser.parse_args()

    pair = args.pair.upper()
    parquet_path = str(PROJECT_ROOT / f"data/ml/rl/{pair}_mc_features.parquet")
    save_dir = str(PROJECT_ROOT / "data/ml/rl/models/lstm_ppo_trading")

    if not os.path.exists(parquet_path):
        print(f"ERROR: Pre-computed features not found: {parquet_path}")
        print("Run precompute_mc_features.py first")
        sys.exit(1)

    # Filter to validation period if requested
    if args.val_only:
        import pandas as pd
        df = pd.read_parquet(parquet_path)
        val_df = df[df["period"] == "validation"]
        if len(val_df) == 0:
            print("ERROR: No validation data found. Run precompute with --include-val first")
            sys.exit(1)
        val_path = str(PROJECT_ROOT / f"data/ml/rl/{pair}_mc_features_val.parquet")
        val_df.to_parquet(val_path, index=False)
        parquet_path = val_path
        print(f"Using VALIDATION data only: {len(val_df):,} steps")
        print(f"Period: {val_df['timestamp'].min()} → {val_df['timestamp'].max()}\n")

    if args.baselines:
        evaluate_baselines(parquet_path, n_episodes=args.episodes, base_seed=args.seed)
    elif args.eval:
        model_path = args.model or os.path.join(save_dir, "final_model.zip")
        if not os.path.exists(model_path):
            print(f"ERROR: Model not found: {model_path}")
            sys.exit(1)
        evaluate(model_path, parquet_path, n_episodes=args.episodes, base_seed=args.seed)
    else:
        train(parquet_path, args.steps, save_dir, eval_freq=args.eval_freq)


if __name__ == "__main__":
    main()
