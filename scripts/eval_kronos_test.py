"""
Evaluate Kronos predictor CE loss on the test set.

Usage:
    python scripts/eval_kronos_test.py
    python scripts/eval_kronos_test.py --device cpu
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

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
    print(f"Loaded from {base}")
    return tokenizer, model


def load_test_data(context_len=512):
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

    # Create sliding windows of context_len + 1
    window = context_len + 1
    n_windows = len(df) - window + 1
    print(f"Test set: {len(df)} candles → {n_windows} windows of {window}")

    x_list = []
    stamp_list = []

    for start in range(0, n_windows, context_len):  # non-overlapping for speed
        if start + window > len(df):
            break
        chunk = df.iloc[start:start + window]
        x_vals = chunk[price_cols].values.astype(np.float32)

        # Normalize
        x_mean = x_vals[:-1].mean(axis=0)
        x_std = x_vals[:-1].std(axis=0)
        x_norm = (x_vals - x_mean) / (x_std + 1e-5)
        x_norm = np.clip(x_norm, -5, 5)

        stamps = chunk[time_cols].values.astype(np.float32)

        x_list.append(x_norm)
        stamp_list.append(stamps)

    x_tensor = torch.from_numpy(np.stack(x_list))
    stamp_tensor = torch.from_numpy(np.stack(stamp_list))
    print(f"Prepared {len(x_list)} non-overlapping windows")

    return x_tensor, stamp_tensor


def evaluate(tokenizer, model, x_data, stamp_data, device, batch_size=8):
    dataset = TensorDataset(x_data, stamp_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    total_loss = 0
    total_s1_loss = 0
    total_s2_loss = 0
    n_batches = 0

    with torch.no_grad():
        for batch_x, batch_stamp in loader:
            batch_x = batch_x.to(device)
            batch_stamp = batch_stamp.to(device)

            # Tokenize
            token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)

            # Input: all but last, Target: shifted by 1
            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_target_0 = token_seq_0[:, 1:]
            token_target_1 = token_seq_1[:, 1:]

            # Forward
            s1_logits, s2_logits = model(
                token_in[0], token_in[1],
                stamp=batch_stamp[:, :-1, :],
                use_teacher_forcing=True,
                s1_targets=token_in[0],
            )

            # Loss
            ce_s1 = F.cross_entropy(s1_logits.reshape(-1, s1_logits.size(-1)), token_target_0.reshape(-1))
            ce_s2 = F.cross_entropy(s2_logits.reshape(-1, s2_logits.size(-1)), token_target_1.reshape(-1))
            loss = (ce_s1 + ce_s2) / 2

            total_loss += loss.item()
            total_s1_loss += ce_s1.item()
            total_s2_loss += ce_s2.item()
            n_batches += 1

            if n_batches % 10 == 0:
                print(f"  Batch {n_batches}/{len(loader)}")

    avg_loss = total_loss / n_batches
    avg_s1 = total_s1_loss / n_batches
    avg_s2 = total_s2_loss / n_batches

    print(f"\n{'=' * 50}")
    print(f"TEST SET RESULTS")
    print(f"{'=' * 50}")
    print(f"  CE Loss:  {avg_loss:.4f}  (s1: {avg_s1:.4f}, s2: {avg_s2:.4f})")
    print(f"  Perplexity (joint): {np.exp(avg_s1 + avg_s2):.1f}")
    print(f"  P(correct s1): {np.exp(-avg_s1)*100:.1f}%")
    print(f"  P(correct s2): {np.exp(-avg_s2)*100:.1f}%")
    print(f"{'=' * 50}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    print(f"Device: {args.device}\n")

    tokenizer, model = load_models(args.device)
    x_data, stamp_data = load_test_data()
    evaluate(tokenizer, model, x_data, stamp_data, args.device, args.batch_size)


if __name__ == "__main__":
    main()
