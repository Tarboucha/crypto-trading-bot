"""Train an RL agent on pre-computed Kronos MC features (v3 — conviction signal).

Uses RecurrentPPO from sb3-contrib with LSTM policy.
The agent outputs conviction [-1, +1], not position size.
Reward measures signal quality, not portfolio P&L.

Usage:
    # Train for 1M steps (pnl + uncertainty only — recommended start)
    python scripts/train_rl_agent_v3.py --steps 1000000

    # Train with all reward components enabled
    python scripts/train_rl_agent_v3.py --steps 1000000 --full-reward

    # Evaluate
    python scripts/train_rl_agent_v3.py --eval

    # Baselines
    python scripts/train_rl_agent_v3.py --baselines
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mllab.rl.trading_env_v3 import make_env_from_parquet, OBS_DIM


def train(parquet_path: str, total_steps: int, save_dir: str,
          eval_freq: int = 10_000, reward_components: list[str] | None = None,
          val_parquet: str | None = None,
          pnl_mode: str = "linear", lambda_wrong: float = 1.0,
          lstm_hidden: int = 128, lstm_layers: int = 1,
          mlp_hidden: int = 64, lr: float = 3e-4):
    """Train RecurrentPPO agent with LSTM policy on conviction env."""
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor

    os.makedirs(save_dir, exist_ok=True)

    # All reward components enabled by default.
    # If --rewards is specified, only those are active.
    all_components = {"cal", "unc", "cost", "shape", "rest"}
    env_kwargs = {"pnl_mode": pnl_mode, "lambda_wrong": lambda_wrong}
    if reward_components is not None:
        selected = set(reward_components)
        for comp in all_components:
            if comp not in selected:
                env_kwargs[f"lambda_{comp}"] = 0.0

    print("Creating training environment...")
    train_env = Monitor(make_env_from_parquet(parquet_path, episode_length=500, **env_kwargs))

    # Use validation data for eval if available, otherwise use training data
    eval_parquet = val_parquet if val_parquet and os.path.exists(val_parquet) else parquet_path
    eval_env = Monitor(make_env_from_parquet(eval_parquet, episode_length=500, **env_kwargs))

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
        n_steps=500,
        batch_size=250,
        n_epochs=10,
        learning_rate=lr,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        device="auto",
        policy_kwargs=dict(
            lstm_hidden_size=lstm_hidden,
            n_lstm_layers=lstm_layers,
            shared_lstm=True,
            enable_critic_lstm=False,
            net_arch=dict(pi=[mlp_hidden], vf=[mlp_hidden]),
        ),
    )

    if reward_components is not None:
        active = ["pnl"] + reward_components
    else:
        active = ["pnl", "cal", "unc", "cost", "shape", "rest"]
    reward_mode = ", ".join(active)

    print(f"\nTraining RecurrentPPO (LSTM) — conviction signal")
    print(f"  Steps: {total_steps:,}")
    print(f"  Save dir: {save_dir}")
    print(f"  Eval every: {eval_freq:,} steps")
    print(f"  Eval data: {'validation set' if val_parquet else 'training set'}")
    print(f"  Action space: conviction [-1, +1]")
    print(f"  Observation: {OBS_DIM} features")
    print(f"  Episode length: 500 steps")
    print(f"  Policy: LSTM({lstm_hidden}x{lstm_layers}) -> MLP({mlp_hidden})")
    print(f"  Learning rate: {lr}")
    print(f"  Reward components: {reward_mode}")
    print(f"  PnL mode: {pnl_mode}" + (f" (lambda_wrong={lambda_wrong})" if pnl_mode == "asymmetric" else ""))
    print()

    model.learn(total_timesteps=total_steps, callback=eval_callback, progress_bar=True)

    final_path = os.path.join(save_dir, "final_model")
    model.save(final_path)
    print(f"\nFinal model saved to {final_path}")

    return model


def evaluate(model_path: str, parquet_path: str, n_episodes: int = 10, base_seed: int = 42):
    """Evaluate a trained conviction agent."""
    from sb3_contrib import RecurrentPPO

    print(f"Loading model from {model_path}")
    model = RecurrentPPO.load(model_path, device="cpu")

    env = make_env_from_parquet(parquet_path, episode_length=500)

    all_rewards = []
    all_directional_pnl = []
    all_mean_conviction = []
    all_flat_pct = []
    all_direction_changes = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 1000 + base_seed)
        total_reward = 0
        total_pnl = 0.0
        convictions = []
        n_flat = 0
        direction_changes = 0
        prev_dir = 0

        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)
        steps = 0

        while True:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_start, deterministic=True
            )
            episode_start = np.zeros((1,), dtype=bool)

            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            total_pnl += info["r_pnl"]
            convictions.append(abs(info["conviction"]))
            steps += 1

            if info["conviction"] == 0.0:
                n_flat += 1

            cur_dir = info["direction"]
            if cur_dir != prev_dir and prev_dir != 0 and cur_dir != 0:
                direction_changes += 1
            prev_dir = cur_dir

            if terminated or truncated:
                break

        all_rewards.append(total_reward)
        all_directional_pnl.append(total_pnl)
        all_mean_conviction.append(np.mean(convictions))
        all_flat_pct.append(n_flat / steps * 100)
        all_direction_changes.append(direction_changes)

        print(f"  Episode {ep + 1:>2}: reward={total_reward:>8.2f}, "
              f"dir_pnl={total_pnl:>+8.2f}, "
              f"avg_conv={np.mean(convictions):.2f}, "
              f"flat={n_flat / steps:.0%}, "
              f"flips={direction_changes}")

    print(f"\n{'=' * 65}")
    print(f"EVALUATION SUMMARY ({n_episodes} episodes)")
    print(f"{'=' * 65}")
    print(f"  Mean reward:          {np.mean(all_rewards):>8.2f} +/- {np.std(all_rewards):.2f}")
    print(f"  Mean directional PnL: {np.mean(all_directional_pnl):>+8.2f}")
    print(f"  Mean conviction:      {np.mean(all_mean_conviction):>8.2f}")
    print(f"  Mean flat %:          {np.mean(all_flat_pct):>7.1f}%")
    print(f"  Mean direction flips: {np.mean(all_direction_changes):>7.1f}")
    print(f"{'=' * 65}")


def evaluate_baselines(parquet_path: str, n_episodes: int = 10, base_seed: int = 42):
    """Evaluate simple conviction baselines."""
    env = make_env_from_parquet(parquet_path, episode_length=500)

    strategies = {
        "always_flat":       0.0,
        "always_long_0.5":   0.5,
        "always_long_1.0":   1.0,
        "always_short_0.5": -0.5,
        "always_short_1.0": -1.0,
    }

    print(f"\n{'=' * 65}")
    print("BASELINES (fixed conviction)")
    print(f"{'=' * 65}")

    for name, fixed_action in strategies.items():
        rewards = []
        pnls = []
        for ep in range(n_episodes):
            obs, _ = env.reset(seed=ep * 1000 + base_seed)
            total_reward = 0
            total_pnl = 0.0
            while True:
                action = np.array([fixed_action], dtype=np.float32)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                total_pnl += info["r_pnl"]
                if terminated or truncated:
                    break
            rewards.append(total_reward)
            pnls.append(total_pnl)

        print(f"  {name:<22}: reward={np.mean(rewards):>8.2f}, dir_pnl={np.mean(pnls):>+8.2f}")

    print(f"{'=' * 65}")


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate RL conviction agent (v3)")
    parser.add_argument("--pair", type=str, default="ETH")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--val-only", action="store_true",
                        help="Eval only on validation period")
    parser.add_argument("--test-only", action="store_true",
                        help="Eval only on test periods (test_crash, test_trend, test_chop)")
    parser.add_argument("--data", type=str, default=None,
                        help="Custom parquet path (overrides --pair)")
    parser.add_argument("--rewards", type=str, nargs="+", default=None,
                        choices=["cal", "unc", "cost", "shape", "rest"],
                        help="Select specific reward components (default: all enabled). "
                             "PnL is always active.")
    parser.add_argument("--pnl-mode", type=str, default="linear",
                        choices=["linear", "asymmetric"],
                        help="PnL reward mode. 'linear': a*r. "
                             "'asymmetric': a*r with extra penalty for wrong direction.")
    parser.add_argument("--lambda-wrong", type=float, default=1.0,
                        help="Extra penalty multiplier for wrong direction (asymmetric mode)")
    parser.add_argument("--lstm-hidden", type=int, default=128,
                        help="LSTM hidden size (default: 128)")
    parser.add_argument("--lstm-layers", type=int, default=1,
                        help="Number of LSTM layers (default: 1)")
    parser.add_argument("--mlp-hidden", type=int, default=64,
                        help="MLP hidden layer size after LSTM (default: 64)")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate (default: 3e-4)")
    args = parser.parse_args()

    pair = args.pair.upper()
    parquet_path = args.data or str(PROJECT_ROOT / f"data/ml/rl/{pair}_mc_features.parquet")
    val_parquet = str(PROJECT_ROOT / f"data/ml/rl/{pair}_mc_features_val.parquet")
    test_parquet = str(PROJECT_ROOT / f"data/ml/rl/{pair}_mc_features_test.parquet")
    # Build save dir from config to avoid overwriting across experiments
    run_name = (f"conviction_lstm{args.lstm_hidden}x{args.lstm_layers}"
                f"_mlp{args.mlp_hidden}"
                f"_{args.pnl_mode}")
    if args.pnl_mode == "asymmetric":
        run_name += f"_lw{args.lambda_wrong}"
    if args.lr != 3e-4:
        run_name += f"_lr{args.lr}"
    save_dir = str(PROJECT_ROOT / f"data/ml/rl/models/{run_name}")

    if not os.path.exists(parquet_path):
        print(f"ERROR: Pre-computed features not found: {parquet_path}")
        print("Run precompute_mc_features.py first")
        sys.exit(1)

    if args.val_only:
        import pandas as pd
        df = pd.read_parquet(parquet_path)
        val_df = df[df["period"] == "validation"]
        if len(val_df) == 0:
            print("ERROR: No validation data found. Run precompute with --include-val first")
            sys.exit(1)
        val_df.to_parquet(val_parquet, index=False)
        parquet_path = val_parquet
        print(f"Using VALIDATION data only: {len(val_df):,} steps")
        print(f"Period: {val_df['timestamp'].min()} -> {val_df['timestamp'].max()}\n")

    if args.test_only:
        import pandas as pd
        df = pd.read_parquet(args.data or str(PROJECT_ROOT / f"data/ml/rl/{pair}_mc_features.parquet"))
        test_df = df[df["period"].str.startswith("test_")]
        if len(test_df) == 0:
            print("ERROR: No test data found (test_crash, test_trend, test_chop)")
            sys.exit(1)
        test_df.to_parquet(test_parquet, index=False)
        parquet_path = test_parquet
        print(f"Using TEST data only: {len(test_df):,} steps")
        periods = test_df.groupby("period").agg(
            rows=("timestamp", "count"),
            start=("timestamp", "min"),
            end=("timestamp", "max"),
        )
        for p, row in periods.iterrows():
            print(f"  {p}: {row['rows']} rows, {row['start']} -> {row['end']}")
        print()

    if args.baselines:
        evaluate_baselines(parquet_path, n_episodes=args.episodes, base_seed=args.seed)
    elif args.eval:
        model_path = args.model or os.path.join(save_dir, "final_model.zip")
        if not os.path.exists(model_path):
            print(f"ERROR: Model not found: {model_path}")
            sys.exit(1)
        evaluate(model_path, parquet_path, n_episodes=args.episodes, base_seed=args.seed)
    else:
        train(parquet_path, args.steps, save_dir, eval_freq=args.eval_freq,
              reward_components=args.rewards, val_parquet=val_parquet,
              pnl_mode=args.pnl_mode, lambda_wrong=args.lambda_wrong,
              lstm_hidden=args.lstm_hidden, lstm_layers=args.lstm_layers,
              mlp_hidden=args.mlp_hidden, lr=args.lr)


if __name__ == "__main__":
    main()
