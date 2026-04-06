"""Train RL agent on V4 direct logit features.

Uses RecurrentPPO from sb3-contrib with LSTM policy.
Agent outputs conviction [-1, +1] based on 14 features
(7 Kronos logit + 7 candle).

Usage:
    # Train
    python scripts/train_rl_v4.py --steps 1000000

    # Train with custom hyperparams
    python scripts/train_rl_v4.py --steps 2000000 --lstm-hidden 128 --lr 1e-4

    # Evaluate
    python scripts/train_rl_v4.py --eval

    # Evaluate on test set
    python scripts/train_rl_v4.py --eval --test
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch as th

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from mllab.rl.trading_env_v4 import make_env_from_parquet, OBS_DIM

RL_DIR = PROJECT_ROOT / "data" / "ml" / "rl"
SAVE_DIR = PROJECT_ROOT / "data" / "ml" / "rl" / "v4_agent"


def train(args):
    from sb3_contrib import RecurrentPPO
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor

    pair = args.pair.upper()
    train_path = RL_DIR / f"{pair}_v4_features.parquet"
    val_path = RL_DIR / f"{pair}_v4_features_test.parquet"

    if not train_path.exists():
        print(f"ERROR: Missing {train_path}")
        print("Run: python scripts/precompute_v4_features.py --pair ETH")
        sys.exit(1)

    run_name = (
        f"v4_{pair.lower()}_lstm{args.lstm_hidden}x{args.lstm_layers}"
        f"_mlp{args.mlp_hidden}_{args.pnl_mode}"
        f"_lr{args.lr}_steps{args.steps // 1000}k"
    )
    save_dir = str(SAVE_DIR / run_name)
    os.makedirs(save_dir, exist_ok=True)

    env_kwargs = {
        "pnl_mode": args.pnl_mode,
        "lambda_wrong": args.lambda_wrong,
        "episode_length": args.episode_length,
    }

    print("Creating training environment...")
    train_env = Monitor(make_env_from_parquet(str(train_path), **env_kwargs))

    # Validation
    eval_path = str(val_path) if val_path.exists() else str(train_path)
    eval_env = Monitor(make_env_from_parquet(eval_path, **env_kwargs))

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(save_dir, "best_model"),
        log_path=os.path.join(save_dir, "eval_logs"),
        eval_freq=args.eval_freq,
        n_eval_episodes=10,
        deterministic=False,
        verbose=1,
    )

    model = RecurrentPPO(
        "MlpLstmPolicy",
        train_env,
        n_steps=args.episode_length,
        batch_size=args.episode_length // 2,
        n_epochs=10,
        learning_rate=args.lr,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=0,
        device="auto",
        policy_kwargs=dict(
            lstm_hidden_size=args.lstm_hidden,
            n_lstm_layers=args.lstm_layers,
            shared_lstm=True,
            enable_critic_lstm=False,
            net_arch=dict(pi=[args.mlp_hidden, args.mlp_hidden], vf=[args.mlp_hidden, args.mlp_hidden]),
            activation_fn=th.nn.ReLU,
        ),
    )

    print(f"\nTraining RecurrentPPO (LSTM) — V4 direct logit features")
    print(f"  Steps: {args.steps:,}")
    print(f"  Save dir: {save_dir}")
    print(f"  Eval every: {args.eval_freq:,} steps")
    print(f"  Eval data: {'test set' if val_path.exists() else 'training set'}")
    print(f"  Observation: {OBS_DIM} features")
    print(f"  Episode length: {args.episode_length}")
    print(f"  Policy: LSTM({args.lstm_hidden}x{args.lstm_layers}) -> MLP({args.mlp_hidden}x2, relu)")
    print(f"  Learning rate: {args.lr}")
    print(f"  PnL mode: {args.pnl_mode} (lambda_wrong={args.lambda_wrong})")
    print()

    model.learn(total_timesteps=args.steps, callback=eval_callback, progress_bar=True)

    final_path = os.path.join(save_dir, "final_model")
    model.save(final_path)
    print(f"\nFinal model saved to {final_path}")


def evaluate(args):
    from sb3_contrib import RecurrentPPO

    pair = args.pair.upper()
    if args.model_path:
        model_path = args.model_path
    else:
        model_path = os.path.join(str(SAVE_DIR), "best_model", "best_model.zip")
        if not os.path.exists(model_path):
            model_path = os.path.join(str(SAVE_DIR), "final_model.zip")

    if not os.path.exists(model_path):
        print(f"ERROR: No model found at {model_path}")
        sys.exit(1)

    if args.test:
        data_path = RL_DIR / f"{pair}_v4_features_test.parquet"
    else:
        data_path = RL_DIR / f"{pair}_v4_features.parquet"

    if not data_path.exists():
        print(f"ERROR: Missing {data_path}")
        sys.exit(1)

    print(f"Loading model from {model_path}")
    model = RecurrentPPO.load(model_path, device="cpu")

    env = make_env_from_parquet(str(data_path), episode_length=args.episode_length)

    n_episodes = args.n_eval_episodes
    all_rewards = []
    all_pnl = []
    all_conviction = []
    all_flat_pct = []
    all_flips = []

    for ep in range(n_episodes):
        obs, _ = env.reset(seed=ep * 1000 + 42)
        total_reward = 0
        total_pnl = 0.0
        convictions = []
        n_flat = 0
        n_flips = 0
        prev_dir = 0

        lstm_states = None
        episode_start = np.ones((1,), dtype=bool)
        steps = 0

        while True:
            action, lstm_states = model.predict(
                obs, state=lstm_states, episode_start=episode_start, deterministic=True,
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
                n_flips += 1
            prev_dir = cur_dir

            if terminated or truncated:
                break

        all_rewards.append(total_reward)
        all_pnl.append(total_pnl)
        all_conviction.append(np.mean(convictions))
        all_flat_pct.append(n_flat / steps * 100)
        all_flips.append(n_flips)

        print(f"  Episode {ep + 1:>2}: reward={total_reward:>8.2f}, "
              f"pnl={total_pnl:>+8.2f}, "
              f"avg_conv={np.mean(convictions):.2f}, "
              f"flat={n_flat / steps:.0%}, "
              f"flips={n_flips}")

    print(f"\n{'=' * 65}")
    print(f"EVALUATION SUMMARY ({n_episodes} episodes, {'test' if args.test else 'train'})")
    print(f"{'=' * 65}")
    print(f"  Mean reward:     {np.mean(all_rewards):>8.2f} ± {np.std(all_rewards):.2f}")
    print(f"  Mean PnL:        {np.mean(all_pnl):>+8.2f}")
    print(f"  Mean conviction: {np.mean(all_conviction):>8.2f}")
    print(f"  Mean flat %:     {np.mean(all_flat_pct):>7.1f}%")
    print(f"  Mean flips:      {np.mean(all_flips):>7.1f}")
    print(f"{'=' * 65}")


def main():
    parser = argparse.ArgumentParser(description="Train/evaluate V4 RL agent")
    parser.add_argument("--pair", type=str, default="ETH")
    parser.add_argument("--eval", action="store_true", help="Evaluate instead of train")
    parser.add_argument("--test", action="store_true", help="Evaluate on test set")
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--episode-length", type=int, default=500)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--n-eval-episodes", type=int, default=10)
    parser.add_argument("--lstm-hidden", type=int, default=128)
    parser.add_argument("--lstm-layers", type=int, default=1)
    parser.add_argument("--mlp-hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--pnl-mode", type=str, default="linear",
                        choices=["linear", "asymmetric"])
    parser.add_argument("--lambda-wrong", type=float, default=1.5)
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to model .zip for evaluation")
    args = parser.parse_args()

    if args.eval:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
