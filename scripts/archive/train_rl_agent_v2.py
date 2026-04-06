"""Train an RL agent on pre-computed Kronos MC features (v2 — SAC).

Uses SAC (Soft Actor-Critic) with continuous action space.
The agent outputs a signed exposure float in [-3, +3].

Usage:
    # Train for 500K steps
    python scripts/train_rl_agent.py --steps 500000

    # Evaluate a trained agent
    python scripts/train_rl_agent.py --eval

    # Run baselines
    python scripts/train_rl_agent.py --baselines
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mllab.rl.trading_env_v2 import make_env_from_parquet


def train(parquet_path: str, total_steps: int, save_dir: str, eval_freq: int = 10_000):
    """Train SAC agent."""
    from stable_baselines3 import SAC
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
        deterministic=False,       # allow random episode starts for diverse eval
        verbose=1,
    )

    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=100_000,        # replay buffer
        batch_size=256,
        tau=0.005,                  # soft target update
        gamma=0.99,                 # discount factor
        learning_starts=1000,       # collect random data before training
        train_freq=1,               # update every step
        gradient_steps=1,
        ent_coef="auto",            # auto-tune entropy (key SAC feature)
        verbose=0,
        device="auto",  # uses CUDA if available, falls back to CPU
        policy_kwargs=dict(
            net_arch=dict(pi=[128, 128], qf=[128, 128]),
        ),
    )

    print(f"\nTraining SAC for {total_steps:,} steps...")
    print(f"  Save dir: {save_dir}")
    print(f"  Eval every: {eval_freq:,} steps")
    print(f"  Action space: continuous [-3.0, +3.0]")
    print(f"  Episode length: 500 steps")
    print(f"  Network: 128x128 (policy + Q-function)")
    print()

    model.learn(total_timesteps=total_steps, callback=eval_callback, progress_bar=True)

    final_path = os.path.join(save_dir, "final_model")
    model.save(final_path)
    print(f"\nFinal model saved to {final_path}")

    return model


def evaluate(model_path: str, parquet_path: str, n_episodes: int = 10):
    """Evaluate a trained agent and print statistics."""
    from stable_baselines3 import SAC

    print(f"Loading model from {model_path}")
    model = SAC.load(model_path, device="cpu")

    env = make_env_from_parquet(parquet_path, episode_length=500)

    all_rewards = []
    all_balances = []
    all_trades = []
    all_max_dd = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 1000 + 42)  # different seed per episode
        total_reward = 0
        trades = 0
        prev_exposure = 0.0
        max_dd = 0.0

        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

            # Count trades: exposure changed significantly
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

        print(f"  Episode {ep+1:>2}: reward={total_reward:>7.2f}, "
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


def evaluate_baselines(parquet_path: str, n_episodes: int = 10):
    """Evaluate simple baselines for comparison."""
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
            obs, _ = env.reset(seed=ep * 42 + 7)
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
        print(f"  {name:<20}: reward={np.mean(rewards):>7.2f}, P&L={mean_pnl:>+6.2f}%")

    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate RL trading agent (v2 — SAC)")
    parser.add_argument("--pair", type=str, default="ETH")
    parser.add_argument("--steps", type=int, default=500_000, help="Training steps (default: 500K)")
    parser.add_argument("--eval", action="store_true", help="Evaluate instead of train")
    parser.add_argument("--model", type=str, default=None, help="Model path for evaluation")
    parser.add_argument("--baselines", action="store_true", help="Run baseline comparison")
    parser.add_argument("--eval-freq", type=int, default=10_000, help="Eval frequency during training")
    args = parser.parse_args()

    pair = args.pair.upper()
    parquet_path = str(PROJECT_ROOT / f"data/ml/rl/{pair}_mc_features.parquet")
    save_dir = str(PROJECT_ROOT / "data/ml/rl/models/sac_trading")

    if not os.path.exists(parquet_path):
        print(f"ERROR: Pre-computed features not found: {parquet_path}")
        print("Run precompute_mc_features.py first")
        sys.exit(1)

    if args.baselines:
        evaluate_baselines(parquet_path)
    elif args.eval:
        model_path = args.model or os.path.join(save_dir, "best_model", "best_model.zip")
        if not os.path.exists(model_path):
            print(f"ERROR: Model not found: {model_path}")
            sys.exit(1)
        evaluate(model_path, parquet_path)
    else:
        train(parquet_path, args.steps, save_dir, eval_freq=args.eval_freq)


if __name__ == "__main__":
    main()
