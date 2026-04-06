"""
Analyze Kronos logit distribution: how is probability mass distributed
relative to the correct token IN PRICE SPACE?

Token indices in BSQ are NOT ordered by price — token 42 and 43 can map to
completely different prices. So we decode all 1024 tokens to their price values
and measure probability concentration in actual price distance.

Usage:
    python scripts/analyze_kronos_logits.py
    python scripts/analyze_kronos_logits.py --n-samples 200 --device cpu
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
KRONOS_PATH = PROJECT_ROOT / "third_party" / "kronos"
sys.path.insert(0, str(KRONOS_PATH))

from model import Kronos, KronosTokenizer


def load_models(device):
    base = PROJECT_ROOT / "data/ml/kronos/finetuned/eth_5m"
    tok_path = str(base / "tokenizer" / "best_model")
    pred_path = str(base / "predictor" / "best_model")

    tokenizer = KronosTokenizer.from_pretrained(tok_path).to(device).eval()
    model = Kronos.from_pretrained(pred_path).to(device).eval()
    return tokenizer, model


def build_token_price_map(tokenizer, device):
    """
    Decode every possible (s1, s2) joint token to its price-space value.

    Since there are 1024 s1 × 1024 s2 = 1M combinations, we can't decode all.
    Instead, decode each s1 token (with a fixed s2=0) and each s2 token
    (with a fixed s1=0) to understand the price mapping per head.

    Returns:
        s1_prices: [1024, 6] — decoded OHLCV+amount for each s1 token (s2=0)
        s2_prices: [1024, 6] — decoded OHLCV+amount for each s2 token (s1=0)
    """
    print("  Building token → price mapping (decoding all 1024 tokens per head)...")

    with torch.no_grad():
        # Decode s1 tokens with s2=0
        s1_ids = torch.arange(1024, device=device).unsqueeze(0)  # [1, 1024]
        s2_zeros = torch.zeros_like(s1_ids)
        s1_decoded = tokenizer.decode((s1_ids, s2_zeros), half=True)  # [1, 1024, 6]
        s1_prices = s1_decoded.squeeze(0).cpu().numpy()  # [1024, 6]

        # Decode s2 tokens with s1=0
        s1_zeros = torch.zeros_like(s1_ids)
        s2_ids = torch.arange(1024, device=device).unsqueeze(0)
        s2_decoded = tokenizer.decode((s1_zeros, s2_ids), half=True)  # [1, 1024, 6]
        s2_prices = s2_decoded.squeeze(0).cpu().numpy()  # [1024, 6]

    # Use close price (index 3) as the primary price dimension
    print(f"    S1 close price range: [{s1_prices[:, 3].min():.4f}, {s1_prices[:, 3].max():.4f}]")
    print(f"    S2 close price range: [{s2_prices[:, 3].min():.4f}, {s2_prices[:, 3].max():.4f}]")
    print(f"    S1 unique close values: {len(np.unique(s1_prices[:, 3].round(6)))}")
    print(f"    S2 unique close values: {len(np.unique(s2_prices[:, 3].round(6)))}")

    return s1_prices, s2_prices


def load_test_data(tokenizer, device, context_len=512, n_samples=200):
    """Load test data and tokenize it."""
    test_path = PROJECT_ROOT / "data/ml/kronos/prepared/ETH_5m_test.csv"
    df = pd.read_csv(test_path)

    price_cols = ["open", "high", "low", "close", "volume", "amount"]
    time_cols = ["minute", "hour", "weekday", "day", "month"]

    df["timestamps"] = pd.to_datetime(df["timestamps"])
    df["minute"] = df["timestamps"].dt.minute
    df["hour"] = df["timestamps"].dt.hour
    df["weekday"] = df["timestamps"].dt.weekday
    df["day"] = df["timestamps"].dt.day
    df["month"] = df["timestamps"].dt.month

    window = context_len + 1
    max_start = len(df) - window
    n_samples = min(n_samples, max_start)

    np.random.seed(42)
    starts = np.random.choice(max_start, size=n_samples, replace=False)
    starts.sort()

    samples = []
    for start in starts:
        chunk = df.iloc[start:start + window]
        x_vals = chunk[price_cols].values.astype(np.float32)

        # Normalize (same as KronosPredictor.predict)
        x_mean = x_vals[:-1].mean(axis=0)
        x_std = x_vals[:-1].std(axis=0)
        x_norm = (x_vals - x_mean) / (x_std + 1e-5)
        x_norm = np.clip(x_norm, -5, 5)

        stamps = chunk[time_cols].values.astype(np.float32)

        samples.append({
            "x": torch.from_numpy(x_norm).unsqueeze(0).to(device),
            "stamps": torch.from_numpy(stamps).unsqueeze(0).to(device),
            "x_mean": x_mean,
            "x_std": x_std,
        })

    return samples


def analyze_logit_distribution(tokenizer, model, samples, s1_prices, s2_prices, device):
    """
    Run inference and analyze how probability is distributed
    relative to the correct token in PRICE SPACE.
    """
    # Close price column index
    CLOSE_IDX = 3

    # Price values for each token (close dimension, normalized space)
    s1_close = s1_prices[:, CLOSE_IDX]  # [1024]
    s2_close = s2_prices[:, CLOSE_IDX]  # [1024]

    stats = {
        # S1 head
        "s1_correct_prob": [],
        "s1_top1_is_correct": [],
        "s1_correct_rank": [],
        "s1_prob_within_01pct": [],    # P(tokens within 0.1% price of correct)
        "s1_prob_within_05pct": [],    # P(tokens within 0.5% price)
        "s1_prob_within_1pct": [],     # P(tokens within 1% price)
        "s1_prob_within_2pct": [],     # P(tokens within 2% price)
        "s1_prob_within_5pct": [],     # P(tokens within 5% price)
        "s1_weighted_price_error": [], # Probability-weighted absolute price error
        "s1_top10_price_spread": [],   # Price spread of top 10 tokens
        # S2 head
        "s2_correct_prob": [],
        "s2_top1_is_correct": [],
        "s2_correct_rank": [],
        "s2_prob_within_01pct": [],
        "s2_prob_within_05pct": [],
        "s2_prob_within_1pct": [],
        "s2_prob_within_2pct": [],
        "s2_prob_within_5pct": [],
        "s2_weighted_price_error": [],
        "s2_top10_price_spread": [],
    }

    print(f"\nAnalyzing {len(samples)} samples...")

    with torch.no_grad():
        for i, sample in enumerate(samples):
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(samples)}")

            x = sample["x"]
            stamps = sample["stamps"]

            # Tokenize full sequence
            tokens = tokenizer.encode(x, half=True)
            s1_tokens = tokens[0]
            s2_tokens = tokens[1]

            # Input: all but last. Target: last.
            input_s1 = s1_tokens[:, :-1]
            input_s2 = s2_tokens[:, :-1]
            target_s1 = s1_tokens[:, -1].item()
            target_s2 = s2_tokens[:, -1].item()
            input_stamps = stamps[:, :-1, :]

            # Forward pass
            s1_logits, s2_logits = model(
                input_s1, input_s2, stamp=input_stamps,
                use_teacher_forcing=True, s1_targets=input_s1,
            )

            s1_probs = F.softmax(s1_logits[:, -1, :], dim=-1).squeeze(0).cpu().numpy()
            s2_probs = F.softmax(s2_logits[:, -1, :], dim=-1).squeeze(0).cpu().numpy()

            for prefix, probs, target, token_prices in [
                ("s1", s1_probs, target_s1, s1_close),
                ("s2", s2_probs, target_s2, s2_close),
            ]:
                correct_price = token_prices[target]

                # Basic stats
                stats[f"{prefix}_correct_prob"].append(probs[target])
                stats[f"{prefix}_top1_is_correct"].append(int(np.argmax(probs) == target))

                rank = (probs > probs[target]).sum()
                stats[f"{prefix}_correct_rank"].append(rank)

                # Price-space distance for every token relative to correct
                # Use relative distance: |price_i - price_correct| / |price_correct|
                if abs(correct_price) > 1e-8:
                    rel_distances = np.abs(token_prices - correct_price) / abs(correct_price)
                else:
                    # Near-zero normalized price, use absolute distance
                    rel_distances = np.abs(token_prices - correct_price)

                # Probability within X% price of correct token
                for threshold, label in [(0.001, "01pct"), (0.005, "05pct"),
                                         (0.01, "1pct"), (0.02, "2pct"), (0.05, "5pct")]:
                    prob_within = probs[rel_distances <= threshold].sum()
                    stats[f"{prefix}_prob_within_{label}"].append(prob_within)

                # Probability-weighted price error
                weighted_error = (probs * rel_distances).sum()
                stats[f"{prefix}_weighted_price_error"].append(weighted_error)

                # Top-10 token price spread
                top10_idx = np.argsort(probs)[-10:]
                top10_prices = token_prices[top10_idx]
                price_spread = top10_prices.max() - top10_prices.min()
                stats[f"{prefix}_top10_price_spread"].append(price_spread)

    return stats


def print_results(stats):
    print()
    print("=" * 70)
    print("KRONOS LOGIT DISTRIBUTION ANALYSIS (PRICE SPACE)")
    print("=" * 70)

    for prefix, head_name in [("s1", "S1 Head (coarse)"), ("s2", "S2 Head (fine, conditioned on s1)")]:
        print(f"\n{'─' * 70}")
        print(f"  {head_name}")
        print(f"{'─' * 70}")

        correct_probs = np.array(stats[f"{prefix}_correct_prob"])
        print(f"\n  P(exact correct token):")
        print(f"    Mean:   {correct_probs.mean():.4f}  ({correct_probs.mean()*100:.1f}%)")
        print(f"    Median: {np.median(correct_probs):.4f}  ({np.median(correct_probs)*100:.1f}%)")

        top1_acc = np.mean(stats[f"{prefix}_top1_is_correct"])
        print(f"\n  Top-1 accuracy: {top1_acc*100:.1f}%")

        ranks = np.array(stats[f"{prefix}_correct_rank"])
        print(f"\n  Rank of correct token:")
        print(f"    Mean: {ranks.mean():.1f},  Median: {np.median(ranks):.1f}")
        print(f"    In top 5: {(ranks < 5).mean()*100:.1f}%")
        print(f"    In top 10: {(ranks < 10).mean()*100:.1f}%")

        print(f"\n  *** Probability concentration in PRICE SPACE ***")
        print(f"  (What % of probability mass maps to tokens within X% price of correct)")
        print(f"  ┌──────────────────┬────────────┬────────────┐")
        print(f"  │ Price Threshold  │ Mean P     │ Median P   │")
        print(f"  ├──────────────────┼────────────┼────────────┤")
        for label, name in [("01pct", "±0.1%"), ("05pct", "±0.5%"),
                            ("1pct", "±1.0%"), ("2pct", "±2.0%"), ("5pct", "±5.0%")]:
            vals = np.array(stats[f"{prefix}_prob_within_{label}"])
            print(f"  │ {name:<16} │ {vals.mean():.4f}     │ {np.median(vals):.4f}     │")
        print(f"  └──────────────────┴────────────┴────────────┘")

        w_errors = np.array(stats[f"{prefix}_weighted_price_error"])
        print(f"\n  Probability-weighted price error:")
        print(f"    Mean:   {w_errors.mean():.4f}  ({w_errors.mean()*100:.2f}%)")
        print(f"    Median: {np.median(w_errors):.4f}  ({np.median(w_errors)*100:.2f}%)")

        spreads = np.array(stats[f"{prefix}_top10_price_spread"])
        print(f"\n  Top-10 tokens price spread (normalized):")
        print(f"    Mean:   {spreads.mean():.4f}")
        print(f"    Median: {np.median(spreads):.4f}")
        print(f"    (smaller = top predictions cluster tightly)")

    # Interpretation
    print(f"\n{'=' * 70}")
    print("INTERPRETATION")
    print(f"{'=' * 70}")

    s1_within_1 = np.mean(stats["s1_prob_within_1pct"])
    s1_within_5 = np.mean(stats["s1_prob_within_5pct"])
    s2_within_1 = np.mean(stats["s2_prob_within_1pct"])
    s2_within_5 = np.mean(stats["s2_prob_within_5pct"])

    print(f"\n  S1: P(within ±1% price) = {s1_within_1:.1%},  P(within ±5%) = {s1_within_5:.1%}")
    print(f"  S2: P(within ±1% price) = {s2_within_1:.1%},  P(within ±5%) = {s2_within_5:.1%}")
    print()

    avg_within_1 = (s1_within_1 + s2_within_1) / 2
    avg_within_5 = (s1_within_5 + s2_within_5) / 2

    if avg_within_1 > 0.5:
        print("  EXCELLENT: >50% of probability mass is within ±1% of correct price.")
        print("  The model is highly concentrated around the right answer.")
        print("  Beam search will capture the vast majority of useful probability.")
        print("  The 8.5% on the exact token is misleading — the model is much better")
        print("  than CE loss suggests when measured in price space.")
    elif avg_within_5 > 0.5:
        print("  GOOD: >50% of probability mass is within ±5% of correct price.")
        print("  The model knows the approximate price range. Beam search beams")
        print("  will cluster around the right neighborhood.")
        print("  For 5-min candles (typical move <1%), ±5% is broad but useful.")
    elif avg_within_5 > 0.3:
        print("  MODERATE: 30-50% of probability is within ±5% of correct price.")
        print("  The model has some concentration but significant dispersion.")
        print("  Beam search helps but won't fully solve the uncertainty.")
    else:
        print("  WEAK: <30% of probability within ±5% of correct price.")
        print("  The token-to-price mapping in BSQ scatters probability widely.")
        print("  The model's predictions are genuinely uncertain in price space.")
        print("  Consider: is BSQ the right tokenization for this task?")

    print()
    print("  NOTE: These results are in normalized price space.")
    print("  A typical 5-min ETH candle moves <0.5%, so ±1% covers ~2 candles of movement.")
    print("  If most probability is within ±1%, the model is effectively 'right'")
    print("  for trading purposes even when it misses the exact token.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=200)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Samples: {args.n_samples}")

    print("\nLoading models...")
    tokenizer, model = load_models(args.device)

    print("\nBuilding token-to-price mapping...")
    s1_prices, s2_prices = build_token_price_map(tokenizer, args.device)

    print("\nLoading test data...")
    samples = load_test_data(tokenizer, device=args.device, n_samples=args.n_samples)

    stats = analyze_logit_distribution(tokenizer, model, samples, s1_prices, s2_prices, args.device)
    print_results(stats)


if __name__ == "__main__":
    main()
