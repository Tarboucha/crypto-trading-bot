"""
Analyze Kronos 10-candle trajectory predictions.

Instead of measuring next-1 token direction (which doesn't work),
measure direction/return accuracy over the full 10-step (50-min) horizon.

For each sample:
1. Run Kronos autoregressive for 10 steps
2. Decode the full predicted trajectory
3. Compare predicted cumulative return at T+1..T+10 vs actual
4. Measure where the directional signal kicks in

Usage:
    python scripts/analyze_kronos_trajectory.py --n-samples 500 --device cuda
    python scripts/analyze_kronos_trajectory.py --n-samples 500 --mc 5 --device cuda
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PROJECT_ROOT = Path(__file__).parent.parent
KRONOS_PATH = PROJECT_ROOT / "third_party" / "kronos"
sys.path.insert(0, str(KRONOS_PATH))

from model import Kronos, KronosTokenizer, KronosPredictor


PRED_LEN = 10  # 10 candles = 50 min at 5-min resolution
PRICE_COLS = ["open", "high", "low", "close", "volume", "amount"]
TIME_COLS = ["minute", "hour", "weekday", "day", "month"]


def load_predictor(device):
    base = PROJECT_ROOT / "data/ml/kronos/finetuned/eth_5m"
    tok_path = str(base / "tokenizer" / "best_model")
    pred_path = str(base / "predictor" / "best_model")

    tokenizer = KronosTokenizer.from_pretrained(tok_path).to(device).eval()
    model = Kronos.from_pretrained(pred_path).to(device).eval()
    predictor = KronosPredictor(model, tokenizer, max_context=512)
    return predictor, tokenizer, model


def load_test_windows(n_samples=500, context_len=512):
    """Load test data windows that include context + 10 future candles."""
    test_path = PROJECT_ROOT / "data/ml/kronos/prepared/ETH_5m_test.csv"
    df = pd.read_csv(test_path)
    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df["minute"] = df["timestamps"].dt.minute
    df["hour"] = df["timestamps"].dt.hour
    df["weekday"] = df["timestamps"].dt.weekday
    df["day"] = df["timestamps"].dt.day
    df["month"] = df["timestamps"].dt.month

    # Need context_len + PRED_LEN candles per window
    window = context_len + PRED_LEN
    max_start = len(df) - window
    n_samples = min(n_samples, max_start)
    starts = np.linspace(0, max_start - 1, n_samples, dtype=int)

    samples = []
    for start in starts:
        chunk = df.iloc[start:start + window]

        # Context: first context_len candles
        context = chunk.iloc[:context_len]
        # Future: next PRED_LEN candles (ground truth)
        future = chunk.iloc[context_len:context_len + PRED_LEN]

        # Raw prices for actual returns
        context_close = context["close"].values[-1]
        future_closes = future["close"].values
        actual_cum_returns = (future_closes - context_close) / context_close

        # Actual direction at each horizon
        actual_dirs = np.sign(actual_cum_returns)

        # Context volatility
        closes = context["close"].values
        rets = np.diff(closes) / closes[:-1]
        context_vol = np.std(rets) if len(rets) > 1 else 0

        samples.append({
            "context_df": context,
            "future_df": future,
            "context_close": context_close,
            "future_closes": future_closes,
            "actual_cum_returns": actual_cum_returns,
            "actual_dirs": actual_dirs,
            "timestamp": future["timestamps"].iloc[0],
            "context_volatility": context_vol,
        })

    print(f"Loaded {len(samples)} windows from {samples[0]['timestamp']} to {samples[-1]['timestamp']}")
    return samples


def run_trajectory_analysis(predictor, samples, device, n_mc=1, temperature=1.0,
                            top_p=0.9):
    """Run Kronos for 10 steps on each sample and measure trajectory accuracy."""
    records = []

    for i, sample in enumerate(samples):
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(samples)}")

        context = sample["context_df"]
        future = sample["future_df"]

        x_df = context[PRICE_COLS]
        x_timestamps = pd.to_datetime(context["timestamps"])
        y_timestamps = pd.to_datetime(future["timestamps"])

        # Run Kronos prediction (n_mc samples, 10 steps each)
        all_pred_closes = []

        for mc in range(n_mc):
            try:
                pred_df = predictor.predict(
                    df=x_df,
                    x_timestamp=x_timestamps,
                    y_timestamp=y_timestamps,
                    pred_len=PRED_LEN,
                    T=temperature,
                    top_k=0,
                    top_p=top_p,
                    sample_count=1,
                    verbose=False,
                )
                pred_closes = pred_df["close"].values  # [10]
                all_pred_closes.append(pred_closes)
            except Exception as e:
                if mc == 0:
                    import traceback
                    print(f"  Sample {i} failed: {e}")
                    traceback.print_exc()
                break

        if len(all_pred_closes) == 0:
            if i == 0:
                print("  WARNING: First sample failed. Check predictor API.")
            continue

        all_pred_closes = np.array(all_pred_closes)  # [n_mc, 10]
        context_close = sample["context_close"]

        # MC-averaged predicted closes
        mean_pred_closes = all_pred_closes.mean(axis=0)  # [10]
        pred_cum_returns = (mean_pred_closes - context_close) / context_close
        pred_dirs = np.sign(pred_cum_returns)

        # Per-MC sample returns (for sigma)
        mc_final_returns = (all_pred_closes[:, -1] - context_close) / context_close
        sigma_return = mc_final_returns.std() if n_mc > 1 else 0.0
        mu_return = mc_final_returns.mean()
        p_up = (mc_final_returns > 0).mean()

        # Actual
        actual_cum_returns = sample["actual_cum_returns"]
        actual_dirs = sample["actual_dirs"]

        rec = {
            "timestamp": sample["timestamp"],
            "context_close": context_close,
            "context_volatility": sample["context_volatility"],
            "mu_return": mu_return,
            "sigma_return": sigma_return,
            "p_up": p_up,
            "n_mc": len(all_pred_closes),
        }

        # Per-horizon metrics (T+1 through T+10)
        for h in range(PRED_LEN):
            rec[f"actual_ret_{h+1}"] = actual_cum_returns[h]
            rec[f"actual_dir_{h+1}"] = actual_dirs[h]
            rec[f"pred_ret_{h+1}"] = pred_cum_returns[h]
            rec[f"pred_dir_{h+1}"] = pred_dirs[h]
            rec[f"dir_correct_{h+1}"] = (pred_dirs[h] == actual_dirs[h])
            rec[f"ret_error_{h+1}"] = abs(pred_cum_returns[h] - actual_cum_returns[h])

        records.append(rec)

    return pd.DataFrame(records)


def print_results(df):
    print()
    print("=" * 75)
    print("KRONOS TRAJECTORY ANALYSIS — 10-STEP (50 MIN) HORIZON")
    print("=" * 75)
    print(f"Samples: {len(df)}, MC runs: {df['n_mc'].iloc[0]}")

    # Direction accuracy by horizon
    print(f"\n{'─' * 75}")
    print(f"  Direction accuracy by horizon (cumulative return vs entry)")
    print(f"{'─' * 75}")
    print(f"  {'Horizon':>8s}  {'Dir Acc':>7s}  {'Edge':>6s}  {'Mean|ActRet|':>12s}  "
          f"{'Mean|PredRet|':>13s}  {'RetCorr':>7s}")
    print(f"  {'─'*8}  {'─'*7}  {'─'*6}  {'─'*12}  {'─'*13}  {'─'*7}")

    for h in range(1, PRED_LEN + 1):
        dir_acc = df[f"dir_correct_{h}"].mean()
        baseline = max((df[f"actual_dir_{h}"] == 1).mean(),
                       (df[f"actual_dir_{h}"] == -1).mean())
        edge = dir_acc - baseline
        mean_act = df[f"actual_ret_{h}"].abs().mean()
        mean_pred = df[f"pred_ret_{h}"].abs().mean()

        # Correlation between predicted and actual return
        corr = df[f"pred_ret_{h}"].corr(df[f"actual_ret_{h}"])

        mins = h * 5
        print(f"  T+{h:<2d} ({mins:>2d}m)  {dir_acc*100:6.1f}%  {edge*100:+5.1f}pp  "
              f"{mean_act*100:11.4f}%  {mean_pred*100:12.4f}%  {corr:+6.3f}")

    # MC features (if n_mc > 1)
    if df["n_mc"].iloc[0] > 1:
        print(f"\n{'─' * 75}")
        print(f"  MC features (T+10 final return)")
        print(f"{'─' * 75}")
        print(f"  mu_return mean:    {df['mu_return'].mean()*100:.4f}%")
        print(f"  sigma_return mean: {df['sigma_return'].mean()*100:.4f}%")
        print(f"  p_up mean:         {df['p_up'].mean():.3f}")

        # Does p_up predict direction?
        actual_dir_10 = df["actual_dir_10"]
        p_up_pred_dir = np.where(df["p_up"] > 0.5, 1, -1)
        p_up_acc = (p_up_pred_dir == actual_dir_10).mean()
        print(f"  p_up direction acc (T+10): {p_up_acc*100:.1f}%")

        # Does mu_return predict direction?
        mu_pred_dir = np.sign(df["mu_return"])
        mu_acc = (mu_pred_dir == actual_dir_10).mean()
        print(f"  mu_return direction acc (T+10): {mu_acc*100:.1f}%")

        # Correlation of mu_return with actual
        corr_mu = df["mu_return"].corr(df["actual_ret_10"])
        print(f"  Correlation(mu_return, actual_ret_10): {corr_mu:+.4f}")


def generate_pdf(df, pdf_path):
    with PdfPages(pdf_path) as pdf:
        # ── Page 1: Direction accuracy by horizon ──
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        fig.suptitle("Kronos Trajectory: Direction Accuracy by Horizon", fontsize=14, fontweight="bold")

        horizons = list(range(1, PRED_LEN + 1))
        mins = [h * 5 for h in horizons]

        dir_accs = [df[f"dir_correct_{h}"].mean() * 100 for h in horizons]
        baselines = [max((df[f"actual_dir_{h}"] == 1).mean(),
                        (df[f"actual_dir_{h}"] == -1).mean()) * 100 for h in horizons]
        edges = [d - b for d, b in zip(dir_accs, baselines)]

        ax = axes[0]
        ax.plot(mins, dir_accs, "o-", color="#2ecc71", lw=2, ms=6, label="Direction accuracy")
        ax.plot(mins, baselines, "s--", color="gray", lw=1, ms=4, label="Baseline (majority)")
        ax.axhline(50, color="red", ls=":", lw=0.5)
        ax.set_xlabel("Horizon (minutes)")
        ax.set_ylabel("Accuracy (%)")
        ax.set_title("Direction Accuracy")
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xticks(mins)

        ax = axes[1]
        colors = ["#2ecc71" if e > 0 else "#e74c3c" for e in edges]
        ax.bar(mins, edges, width=3, color=colors)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("Horizon (minutes)")
        ax.set_ylabel("Edge (pp)")
        ax.set_title("Edge over Baseline")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(mins)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 2: Return correlation by horizon ──
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        fig.suptitle("Predicted vs Actual Return Correlation", fontsize=14, fontweight="bold")

        corrs = [df[f"pred_ret_{h}"].corr(df[f"actual_ret_{h}"]) for h in horizons]

        ax = axes[0]
        colors = ["#2ecc71" if c > 0 else "#e74c3c" for c in corrs]
        ax.bar(mins, corrs, width=3, color=colors)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("Horizon (minutes)")
        ax.set_ylabel("Pearson correlation")
        ax.set_title("Return Correlation (pred vs actual)")
        ax.grid(True, alpha=0.3)
        ax.set_xticks(mins)

        # Scatter at T+10
        ax = axes[1]
        ax.scatter(df["pred_ret_10"] * 100, df["actual_ret_10"] * 100,
                   alpha=0.3, s=10, color="#3498db")
        ax.axhline(0, color="gray", ls="--", lw=0.5)
        ax.axvline(0, color="gray", ls="--", lw=0.5)
        corr_10 = df["pred_ret_10"].corr(df["actual_ret_10"])
        ax.set_xlabel("Predicted return T+10 (%)")
        ax.set_ylabel("Actual return T+10 (%)")
        ax.set_title(f"T+10 (50min): r={corr_10:.3f}")
        ax.grid(True, alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        pdf.savefig(fig)
        plt.close(fig)

        # ── Page 3: MC features (if available) ──
        if df["n_mc"].iloc[0] > 1:
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            fig.suptitle("MC Trajectory Features", fontsize=14, fontweight="bold")

            # mu_return vs actual
            ax = axes[0, 0]
            ax.scatter(df["mu_return"] * 100, df["actual_ret_10"] * 100,
                       alpha=0.3, s=10, color="#2ecc71")
            corr_mu = df["mu_return"].corr(df["actual_ret_10"])
            ax.set_xlabel("MC mu_return (%)")
            ax.set_ylabel("Actual return T+10 (%)")
            ax.set_title(f"mu_return vs actual (r={corr_mu:.3f})")
            ax.axhline(0, color="gray", ls="--", lw=0.5)
            ax.axvline(0, color="gray", ls="--", lw=0.5)
            ax.grid(True, alpha=0.3)

            # p_up distribution
            ax = axes[0, 1]
            up_mask = df["actual_dir_10"] == 1
            ax.hist(df.loc[up_mask, "p_up"], bins=20, alpha=0.6, color="#2ecc71",
                    label="Actually up", density=True)
            ax.hist(df.loc[~up_mask, "p_up"], bins=20, alpha=0.6, color="#e74c3c",
                    label="Actually down", density=True)
            ax.set_xlabel("MC p_up")
            ax.set_ylabel("Density")
            ax.set_title("p_up when price actually goes up vs down")
            ax.legend()

            # sigma_return vs actual |return|
            ax = axes[1, 0]
            ax.scatter(df["sigma_return"] * 100, df["actual_ret_10"].abs() * 100,
                       alpha=0.3, s=10, color="#e67e22")
            corr_sig = df["sigma_return"].corr(df["actual_ret_10"].abs())
            ax.set_xlabel("MC sigma_return (%)")
            ax.set_ylabel("|Actual return| T+10 (%)")
            ax.set_title(f"sigma vs |actual| (r={corr_sig:.3f})")
            ax.grid(True, alpha=0.3)

            # Direction accuracy by p_up confidence
            ax = axes[1, 1]
            try:
                df["_pup_q"] = pd.qcut(df["p_up"].clip(0.01, 0.99), 5,
                                        labels=False, duplicates="drop")
                grp = df.groupby("_pup_q").agg(
                    p_up=("p_up", "mean"),
                    dir_acc=("dir_correct_10", "mean"),
                    n=("dir_correct_10", "count"),
                ).reset_index()
                ax.plot(grp["p_up"], grp["dir_acc"] * 100, "o-", color="#9b59b6",
                        ms=6, lw=2)
                ax.axhline(50, color="gray", ls="--", lw=0.5)
                ax.set_xlabel("MC p_up (quintile mean)")
                ax.set_ylabel("Direction accuracy T+10 (%)")
                ax.set_title("Direction acc by p_up confidence")
                ax.grid(True, alpha=0.3)
                df.drop(columns="_pup_q", inplace=True)
            except Exception:
                ax.text(0.5, 0.5, "Not enough data", ha="center", va="center",
                        transform=ax.transAxes)

            fig.tight_layout(rect=[0, 0, 1, 0.93])
            pdf.savefig(fig)
            plt.close(fig)

        # ── Page 4: Predicted trajectory examples ──
        fig, axes = plt.subplots(3, 2, figsize=(11, 10))
        fig.suptitle("Example Trajectories: Predicted vs Actual", fontsize=14, fontweight="bold")

        # Pick examples: best, worst, median by T+10 return error
        df_sorted = df.sort_values(f"ret_error_10")
        indices = [
            df_sorted.index[0],                          # best
            df_sorted.index[len(df_sorted) // 4],        # Q1
            df_sorted.index[len(df_sorted) // 2],        # median
            df_sorted.index[3 * len(df_sorted) // 4],    # Q3
            df_sorted.index[-1],                         # worst
            df_sorted.index[len(df_sorted) // 3],        # another
        ]

        for ax, idx in zip(axes.flat, indices):
            row = df.loc[idx]
            actual = [row[f"actual_ret_{h+1}"] * 100 for h in range(PRED_LEN)]
            pred = [row[f"pred_ret_{h+1}"] * 100 for h in range(PRED_LEN)]
            ax.plot(mins, actual, "o-", color="#2ecc71", lw=2, ms=4, label="Actual")
            ax.plot(mins, pred, "s--", color="#e74c3c", lw=1.5, ms=4, label="Predicted")
            ax.axhline(0, color="gray", ls=":", lw=0.5)
            err = row[f"ret_error_10"]
            ax.set_title(f"{row['timestamp']:%Y-%m-%d %H:%M} | err={err*100:.3f}%", fontsize=9)
            ax.set_xlabel("Min ahead")
            ax.set_ylabel("Cum return (%)")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.95])
        pdf.savefig(fig)
        plt.close(fig)

    print(f"PDF saved to {pdf_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=500)
    parser.add_argument("--mc", type=int, default=1,
                        help="MC samples per prediction (1=argmax, 5=MC averaging)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Samples: {args.n_samples}, MC: {args.mc}, T: {args.temperature}")

    print("\nLoading predictor...")
    predictor, tokenizer, model = load_predictor(args.device)

    print("\nLoading test windows...")
    samples = load_test_windows(n_samples=args.n_samples)

    print("\nRunning trajectory analysis...")
    df = run_trajectory_analysis(
        predictor, samples, args.device,
        n_mc=args.mc, temperature=args.temperature, top_p=args.top_p,
    )

    # Save raw data
    out_dir = PROJECT_ROOT / "data/ml/kronos/analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    if len(df) == 0:
        print("\nERROR: All predictions failed. No results to show.")
        return

    df.to_parquet(out_dir / "trajectory_analysis.parquet")
    print(f"Saved {len(df)} records to {out_dir / 'trajectory_analysis.parquet'}")

    print_results(df)
    generate_pdf(df, out_dir / "kronos_trajectory_analysis.pdf")


if __name__ == "__main__":
    main()
