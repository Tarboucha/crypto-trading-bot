"""
Deep analysis of Kronos prediction errors:

1. When wrong (67% s1, 47% s2), is the remaining probability:
   - Concentrated on a few wrong tokens (confident but wrong)?
   - Scattered uniformly (genuinely uncertain)?

2. Are accurate predictions clustered in time (regime-dependent)
   or randomly distributed?

Usage:
    python scripts/analyze_kronos_errors.py --n-samples 5000 --device cuda
    python scripts/analyze_kronos_errors.py --n-samples 5000 --device cuda --include-train
"""
import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

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
    with torch.no_grad():
        s1_ids = torch.arange(1024, device=device).unsqueeze(0)
        s2_zeros = torch.zeros_like(s1_ids)
        s1_decoded = tokenizer.decode((s1_ids, s2_zeros), half=True)
        s1_prices = s1_decoded.squeeze(0).cpu().numpy()

        s1_zeros = torch.zeros_like(s1_ids)
        s2_ids = torch.arange(1024, device=device).unsqueeze(0)
        s2_decoded = tokenizer.decode((s1_zeros, s2_ids), half=True)
        s2_prices = s2_decoded.squeeze(0).cpu().numpy()
    return s1_prices, s2_prices


def load_data_from_csv(csv_path, device, context_len=512, n_samples=5000, label="unknown"):
    """Load data from a single CSV, return samples WITH timestamps for temporal analysis."""
    df = pd.read_csv(csv_path)

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

    # Use evenly spaced samples to cover the full time range
    n_samples = min(n_samples, max_start)
    starts = np.linspace(0, max_start - 1, n_samples, dtype=int)

    samples = []
    for start in starts:
        chunk = df.iloc[start:start + window]
        x_vals = chunk[price_cols].values.astype(np.float32)

        x_mean = x_vals[:-1].mean(axis=0)
        x_std = x_vals[:-1].std(axis=0)
        x_norm = (x_vals - x_mean) / (x_std + 1e-5)
        x_norm = np.clip(x_norm, -5, 5)

        stamps = chunk[time_cols].values.astype(np.float32)

        # The target candle is the last one — get its timestamp and raw price
        target_ts = chunk["timestamps"].iloc[-1]
        target_close = chunk["close"].iloc[-1]
        prev_close = chunk["close"].iloc[-2]
        actual_return = (target_close - prev_close) / prev_close

        # Volatility of context window
        closes = chunk["close"].values[:-1]
        returns = np.diff(closes) / closes[:-1]
        context_volatility = np.std(returns) if len(returns) > 1 else 0

        samples.append({
            "x": torch.from_numpy(x_norm).unsqueeze(0).to(device),
            "stamps": torch.from_numpy(stamps).unsqueeze(0).to(device),
            "timestamp": target_ts,
            "actual_return": actual_return,
            "context_volatility": context_volatility,
            "close_price": target_close,
            "dataset": label,
        })

    print(f"  [{label}] {len(samples)} samples from {samples[0]['timestamp']} to {samples[-1]['timestamp']}")
    return samples


def load_all_data(device, n_samples_test=5000, n_samples_train=0, context_len=512):
    """Load test data, and optionally train data for cross-regime diversity."""
    samples = []

    test_path = PROJECT_ROOT / "data/ml/kronos/prepared/ETH_5m_test.csv"
    print(f"\nLoading test data ({test_path.name})...")
    samples.extend(load_data_from_csv(test_path, device, context_len, n_samples_test, "test"))

    if n_samples_train > 0:
        train_path = PROJECT_ROOT / "data/ml/kronos/prepared/ETH_5m_train.csv"
        if train_path.exists():
            print(f"Loading train data ({train_path.name}) — NOTE: model was trained on this, "
                  f"expect higher accuracy...")
            samples.extend(load_data_from_csv(train_path, device, context_len, n_samples_train, "train"))

    print(f"\nTotal: {len(samples)} samples")
    return samples


def analyze(tokenizer, model, samples, s1_prices, s2_prices, device):
    CLOSE_IDX = 3
    s1_close = s1_prices[:, CLOSE_IDX]
    s2_close = s2_prices[:, CLOSE_IDX]

    records = []

    with torch.no_grad():
        for i, sample in enumerate(samples):
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(samples)}")

            x = sample["x"]
            stamps = sample["stamps"]

            tokens = tokenizer.encode(x, half=True)
            s1_tokens = tokens[0]
            s2_tokens = tokens[1]

            input_s1 = s1_tokens[:, :-1]
            input_s2 = s2_tokens[:, :-1]
            target_s1 = s1_tokens[:, -1].item()
            target_s2 = s2_tokens[:, -1].item()

            # Previous candle tokens (second to last) — our reference point
            prev_s1 = s1_tokens[:, -2].item()
            prev_s2 = s2_tokens[:, -2].item()

            input_stamps = stamps[:, :-1, :]

            s1_logits, s2_logits = model(
                input_s1, input_s2, stamp=input_stamps,
                use_teacher_forcing=True, s1_targets=input_s1,
            )

            s1_probs = F.softmax(s1_logits[:, -1, :], dim=-1).squeeze(0).cpu().numpy()
            s2_probs = F.softmax(s2_logits[:, -1, :], dim=-1).squeeze(0).cpu().numpy()

            # ── Decode actual direction from raw prices ──
            actual_dir = 1 if sample["actual_return"] > 0 else (-1 if sample["actual_return"] < 0 else 0)

            # ── Decode predicted direction via FULL SEQUENCE decode ──
            pred_s1 = int(np.argmax(s1_probs))
            pred_s2 = int(np.argmax(s2_probs))

            # Build predicted sequence: replace last token with prediction
            # s1_tokens/s2_tokens are [1, 513] — the full encoded sequence
            pred_seq_s1 = s1_tokens.clone()  # [1, 513]
            pred_seq_s2 = s2_tokens.clone()  # [1, 513]
            pred_seq_s1[:, -1] = pred_s1
            pred_seq_s2[:, -1] = pred_s2

            # Decode full sequence with predicted last token
            pred_decoded_full = tokenizer.decode(
                (pred_seq_s1, pred_seq_s2), half=True
            )  # [1, 513, 6]

            # Also decode ground truth sequence for sanity check
            gt_decoded_full = tokenizer.decode(
                (s1_tokens, s2_tokens), half=True
            )  # [1, 513, 6]

            # Extract close prices at last two positions
            pred_close_last = pred_decoded_full[0, -1, CLOSE_IDX].item()
            pred_close_prev = pred_decoded_full[0, -2, CLOSE_IDX].item()
            gt_close_last = gt_decoded_full[0, -1, CLOSE_IDX].item()
            gt_close_prev = gt_decoded_full[0, -2, CLOSE_IDX].item()

            # Direction from full-sequence decode
            pred_dir = 1 if pred_close_last > pred_close_prev else (
                -1 if pred_close_last < pred_close_prev else 0)

            # Sanity: does GT decode give correct direction?
            gt_dir = 1 if gt_close_last > gt_close_prev else (
                -1 if gt_close_last < gt_close_prev else 0)

            pred_return = pred_close_last - pred_close_prev

            rec = {
                "timestamp": sample["timestamp"],
                "actual_return": sample["actual_return"],
                "actual_dir": actual_dir,
                "context_volatility": sample["context_volatility"],
                "close_price": sample["close_price"],
                "dataset": sample.get("dataset", "unknown"),
                "direction_correct": (pred_dir == actual_dir),
                "pred_dir": pred_dir,
                "gt_decode_dir_correct": (gt_dir == actual_dir),  # sanity check
                "pred_decoded_close": pred_close_last,
                "prev_decoded_close": pred_close_prev,
                "pred_return": pred_return,
            }

            for prefix, probs, target, token_prices in [
                ("s1", s1_probs, target_s1, s1_close),
                ("s2", s2_probs, target_s2, s2_close),
            ]:
                top1 = np.argmax(probs)
                correct = (top1 == target)
                correct_prob = probs[target]
                top1_prob = probs[top1]

                # Entropy of distribution
                entropy = -np.sum(probs * np.log(probs + 1e-10))

                # How concentrated is the distribution?
                sorted_probs = np.sort(probs)[::-1]
                top5_mass = sorted_probs[:5].sum()
                top10_mass = sorted_probs[:10].sum()
                top50_mass = sorted_probs[:50].sum()

                # Effective number of tokens (exp of entropy)
                effective_n = np.exp(entropy)

                # Price error of top-1 prediction
                correct_price = token_prices[target]
                top1_price = token_prices[top1]
                if abs(correct_price) > 1e-8:
                    top1_price_error = abs(top1_price - correct_price) / abs(correct_price)
                else:
                    top1_price_error = abs(top1_price - correct_price)

                # Per-head direction: compare decoded price of predicted token
                # vs decoded price of previous candle's token, using per-head
                # price maps (approximate — joint decode above is the real one)
                prev_tok = prev_s1 if prefix == "s1" else prev_s2
                prev_price_head = token_prices[prev_tok]
                top1_head_dir = 1 if top1_price > prev_price_head else (
                    -1 if top1_price < prev_price_head else 0)
                head_direction_correct = (top1_head_dir == actual_dir)

                # Probability mass on correct direction (per-head, relative to prev token)
                if actual_dir > 0:
                    prob_correct_dir = probs[token_prices > prev_price_head].sum()
                elif actual_dir < 0:
                    prob_correct_dir = probs[token_prices < prev_price_head].sum()
                else:
                    prob_correct_dir = 0.5

                rec[f"{prefix}_correct"] = correct
                rec[f"{prefix}_correct_prob"] = correct_prob
                rec[f"{prefix}_top1_prob"] = top1_prob
                rec[f"{prefix}_entropy"] = entropy
                rec[f"{prefix}_top5_mass"] = top5_mass
                rec[f"{prefix}_top10_mass"] = top10_mass
                rec[f"{prefix}_top50_mass"] = top50_mass
                rec[f"{prefix}_effective_n"] = effective_n
                rec[f"{prefix}_top1_price_error"] = top1_price_error
                rec[f"{prefix}_direction_correct"] = head_direction_correct
                rec[f"{prefix}_prob_correct_dir"] = prob_correct_dir

            records.append(rec)

    return pd.DataFrame(records)


def print_error_analysis(df):
    print()
    print("=" * 75)
    print("PART 1: WHEN KRONOS IS WRONG — CONCENTRATED OR SCATTERED?")
    print("=" * 75)

    for prefix, head in [("s1", "S1 (coarse)"), ("s2", "S2 (fine)")]:
        correct_mask = df[f"{prefix}_correct"]
        wrong_mask = ~correct_mask
        n_correct = correct_mask.sum()
        n_wrong = wrong_mask.sum()

        print(f"\n{'─' * 75}")
        print(f"  {head}: {n_correct} correct ({n_correct/len(df)*100:.1f}%), "
              f"{n_wrong} wrong ({n_wrong/len(df)*100:.1f}%)")
        print(f"{'─' * 75}")

        # Compare distributions when correct vs wrong
        print(f"\n  Distribution shape — CORRECT predictions:")
        c = df[correct_mask]
        print(f"    Top-1 prob:    mean={c[f'{prefix}_top1_prob'].mean():.4f}  "
              f"median={c[f'{prefix}_top1_prob'].median():.4f}")
        print(f"    Top-5 mass:    mean={c[f'{prefix}_top5_mass'].mean():.4f}")
        print(f"    Top-10 mass:   mean={c[f'{prefix}_top10_mass'].mean():.4f}")
        print(f"    Top-50 mass:   mean={c[f'{prefix}_top50_mass'].mean():.4f}")
        print(f"    Entropy:       mean={c[f'{prefix}_entropy'].mean():.3f}")
        print(f"    Effective N:   mean={c[f'{prefix}_effective_n'].mean():.1f}")

        print(f"\n  Distribution shape — WRONG predictions:")
        w = df[wrong_mask]
        print(f"    Top-1 prob:    mean={w[f'{prefix}_top1_prob'].mean():.4f}  "
              f"median={w[f'{prefix}_top1_prob'].median():.4f}")
        print(f"    Top-5 mass:    mean={w[f'{prefix}_top5_mass'].mean():.4f}")
        print(f"    Top-10 mass:   mean={w[f'{prefix}_top10_mass'].mean():.4f}")
        print(f"    Top-50 mass:   mean={w[f'{prefix}_top50_mass'].mean():.4f}")
        print(f"    Entropy:       mean={w[f'{prefix}_entropy'].mean():.3f}")
        print(f"    Effective N:   mean={w[f'{prefix}_effective_n'].mean():.1f}")

        # Classify wrong predictions: confident-wrong vs uncertain-wrong
        print(f"\n  Breakdown of WRONG predictions by confidence:")
        # High confidence wrong: top1 prob > median of correct predictions
        median_correct_prob = c[f"{prefix}_top1_prob"].median()
        confident_wrong = w[w[f"{prefix}_top1_prob"] >= median_correct_prob]
        uncertain_wrong = w[w[f"{prefix}_top1_prob"] < median_correct_prob]

        print(f"    Threshold (median correct top1 prob): {median_correct_prob:.4f}")
        print(f"    Confident-wrong (top1 >= threshold): {len(confident_wrong)} "
              f"({len(confident_wrong)/len(w)*100:.1f}%)")
        print(f"    Uncertain-wrong (top1 < threshold):  {len(uncertain_wrong)} "
              f"({len(uncertain_wrong)/len(w)*100:.1f}%)")

        if len(confident_wrong) > 0:
            print(f"\n    Confident-wrong details:")
            print(f"      Mean top1 prob:        {confident_wrong[f'{prefix}_top1_prob'].mean():.4f}")
            print(f"      Mean price error:      {confident_wrong[f'{prefix}_top1_price_error'].mean():.4f} "
                  f"({confident_wrong[f'{prefix}_top1_price_error'].mean()*100:.2f}%)")
            print(f"      Direction still right:  {confident_wrong[f'{prefix}_direction_correct'].mean()*100:.1f}%")

        if len(uncertain_wrong) > 0:
            print(f"\n    Uncertain-wrong details:")
            print(f"      Mean top1 prob:        {uncertain_wrong[f'{prefix}_top1_prob'].mean():.4f}")
            print(f"      Mean price error:      {uncertain_wrong[f'{prefix}_top1_price_error'].mean():.4f} "
                  f"({uncertain_wrong[f'{prefix}_top1_price_error'].mean()*100:.2f}%)")
            print(f"      Direction still right:  {uncertain_wrong[f'{prefix}_direction_correct'].mean()*100:.1f}%")

        # Direction accuracy overall
        print(f"\n  Direction accuracy (up/down, regardless of exact token):")
        print(f"    Overall:  {df[f'{prefix}_direction_correct'].mean()*100:.1f}%")
        print(f"    Correct:  {c[f'{prefix}_direction_correct'].mean()*100:.1f}%")
        print(f"    Wrong:    {w[f'{prefix}_direction_correct'].mean()*100:.1f}%")

        # Probability mass on correct direction
        print(f"\n  P(mass on correct direction):")
        print(f"    Overall:  {df[f'{prefix}_prob_correct_dir'].mean()*100:.1f}%")
        print(f"    Correct:  {c[f'{prefix}_prob_correct_dir'].mean()*100:.1f}%")
        print(f"    Wrong:    {w[f'{prefix}_prob_correct_dir'].mean()*100:.1f}%")


def print_temporal_analysis(df):
    print()
    print("=" * 75)
    print("PART 2: ARE ACCURATE PREDICTIONS CLUSTERED IN TIME OR RANDOM?")
    print("=" * 75)

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour

    # Assign market regime labels
    def assign_regime(ts):
        if ts.year == 2020:
            return "2020 Covid crash/recovery"
        elif ts.year == 2021:
            return "2021 Bull top / May crash"
        elif ts.year == 2022:
            return "2022 Bear (Luna/3AC)"
        elif ts.year == 2023:
            return "2023 Low-vol recovery"
        elif ts.year == 2024:
            return "2024 ETF rally"
        elif ts.year >= 2025:
            return "2025-26 Recent market"
        return "unknown"

    df["regime"] = df["timestamp"].apply(assign_regime)

    # Per-dataset breakdown if we have both train and test
    datasets = df["dataset"].unique()
    if len(datasets) > 1:
        print(f"\n  Per-dataset overview:")
        for ds in sorted(datasets):
            sub = df[df["dataset"] == ds]
            for prefix, head in [("s1", "S1"), ("s2", "S2")]:
                acc = sub[f"{prefix}_correct"].mean()
                dacc = sub[f"{prefix}_direction_correct"].mean()
                print(f"    [{ds:5s}] {head}: token_acc={acc*100:.1f}%, "
                      f"dir_acc={dacc*100:.1f}%, n={len(sub)}")

    for prefix, head in [("s1", "S1 (coarse)"), ("s2", "S2 (fine)")]:
        print(f"\n{'─' * 75}")
        print(f"  {head}")
        print(f"{'─' * 75}")

        # ── 1. Rolling accuracy at trading-relevant timescales ──
        # Sort by time for rolling windows
        df_sorted = df.sort_values("timestamp").reset_index(drop=True)

        # Rolling windows: 6 samples ~ 30min (if evenly spaced at ~5min),
        # but with n_samples spread across data, compute actual time spans
        for window_name, window_size in [("~30min (6 pts)", 6),
                                          ("~2h (24 pts)", 24),
                                          ("~6h (72 pts)", 72),
                                          ("~1d (288 pts)", 288)]:
            if window_size > len(df_sorted) // 3:
                continue
            rolling_acc = df_sorted[f"{prefix}_correct"].rolling(window_size, min_periods=window_size).mean()
            rolling_dir = df_sorted["direction_correct"].rolling(window_size, min_periods=window_size).mean()
            valid = rolling_acc.dropna()
            if len(valid) < 10:
                continue
            print(f"\n  Rolling {window_name} token accuracy distribution:")
            pcts = valid.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
            print(f"    P5={pcts.iloc[0]*100:.1f}%  P25={pcts.iloc[1]*100:.1f}%  "
                  f"median={pcts.iloc[2]*100:.1f}%  P75={pcts.iloc[3]*100:.1f}%  "
                  f"P95={pcts.iloc[4]*100:.1f}%")
            valid_dir = rolling_dir.dropna()
            if len(valid_dir) > 0:
                pcts_d = valid_dir.quantile([0.05, 0.25, 0.5, 0.75, 0.95])
                print(f"    Direction: P5={pcts_d.iloc[0]*100:.1f}%  "
                      f"P25={pcts_d.iloc[1]*100:.1f}%  median={pcts_d.iloc[2]*100:.1f}%  "
                      f"P75={pcts_d.iloc[3]*100:.1f}%  P95={pcts_d.iloc[4]*100:.1f}%")
            # Are there sustained good/bad windows?
            hot_pct = (valid > valid.mean() + valid.std()).mean()
            cold_pct = (valid < valid.mean() - valid.std()).mean()
            print(f"    Hot windows (>1σ above mean): {hot_pct*100:.1f}%  "
                  f"Cold windows (<1σ below mean): {cold_pct*100:.1f}%")

        # ── 2. Daily accuracy (the natural trading day view) ──
        df_sorted["date"] = df_sorted["timestamp"].dt.date
        daily = df_sorted.groupby("date").agg(
            n=(f"{prefix}_correct", "count"),
            token_acc=(f"{prefix}_correct", "mean"),
            dir_acc=(f"{prefix}_direction_correct", "mean"),
            mean_ent=(f"{prefix}_entropy", "mean"),
            mean_vol=("context_volatility", "mean"),
        ).reset_index()
        # Only show days with enough samples
        daily = daily[daily["n"] >= 5]

        if len(daily) > 0:
            print(f"\n  Daily accuracy (days with >=5 samples, showing best/worst/typical):")
            daily_sorted = daily.sort_values("token_acc")
            n_show = min(5, len(daily_sorted))

            print(f"    WORST {n_show} days:")
            for _, row in daily_sorted.head(n_show).iterrows():
                print(f"      {row['date']}: tok={row['token_acc']*100:5.1f}%  "
                      f"dir={row['dir_acc']*100:5.1f}%  ent={row['mean_ent']:.2f}  "
                      f"vol={row['mean_vol']:.5f}  n={int(row['n'])}")

            print(f"    BEST {n_show} days:")
            for _, row in daily_sorted.tail(n_show).iterrows():
                print(f"      {row['date']}: tok={row['token_acc']*100:5.1f}%  "
                      f"dir={row['dir_acc']*100:5.1f}%  ent={row['mean_ent']:.2f}  "
                      f"vol={row['mean_vol']:.5f}  n={int(row['n'])}")

            print(f"\n    Daily accuracy stats (n={len(daily)} days):")
            print(f"      Token: mean={daily['token_acc'].mean()*100:.1f}%  "
                  f"std={daily['token_acc'].std()*100:.1f}%  "
                  f"min={daily['token_acc'].min()*100:.1f}%  "
                  f"max={daily['token_acc'].max()*100:.1f}%")
            print(f"      Dir:   mean={daily['dir_acc'].mean()*100:.1f}%  "
                  f"std={daily['dir_acc'].std()*100:.1f}%  "
                  f"min={daily['dir_acc'].min()*100:.1f}%  "
                  f"max={daily['dir_acc'].max()*100:.1f}%")

        # ── 3. 4-hour session blocks (trading sessions) ──
        df_sorted["session_4h"] = (df_sorted["hour"] // 4) * 4
        print(f"\n  Accuracy by 4h session:")
        session_stats = df_sorted.groupby("session_4h").agg(
            token_acc=(f"{prefix}_correct", "mean"),
            dir_acc=(f"{prefix}_direction_correct", "mean"),
            mean_ent=(f"{prefix}_entropy", "mean"),
            n=(f"{prefix}_correct", "count"),
        ).reset_index()
        for _, row in session_stats.iterrows():
            h = int(row["session_4h"])
            tok_bar = "█" * int(row["token_acc"] * 40)
            dir_bar = "▓" * int(row["dir_acc"] * 40)
            print(f"    {h:02d}:00-{h+4:02d}:00  tok={row['token_acc']*100:5.1f}%  "
                  f"dir={row['dir_acc']*100:5.1f}%  ent={row['mean_ent']:.2f}  "
                  f"n={int(row['n']):4d}")
            print(f"               tok: {tok_bar}")
            print(f"               dir: {dir_bar}")

        # ── 4. Regime-level accuracy ──
        print(f"\n  Accuracy by market regime:")
        regime_stats = df.groupby("regime").agg(
            n=(f"{prefix}_correct", "count"),
            token_acc=(f"{prefix}_correct", "mean"),
            dir_acc=(f"{prefix}_direction_correct", "mean"),
            mean_ent=(f"{prefix}_entropy", "mean"),
            mean_vol=("context_volatility", "mean"),
            prob_dir=(f"{prefix}_prob_correct_dir", "mean"),
        ).reset_index()
        for _, row in regime_stats.iterrows():
            print(f"    {row['regime']:30s}: tok={row['token_acc']*100:5.1f}%  "
                  f"dir={row['dir_acc']*100:5.1f}%  "
                  f"P(dir)={row['prob_dir']*100:5.1f}%  "
                  f"ent={row['mean_ent']:.2f}  vol={row['mean_vol']:.5f}  "
                  f"n={int(row['n']):4d}")

        # ── 5. Streaks (test set, chronological) ──
        test_df = df[df["dataset"] == "test"].sort_values("timestamp")
        if len(test_df) > 10:
            print(f"\n  Streak analysis (test set, chronological):")
            correct_arr = test_df[f"{prefix}_correct"].values
            streaks_correct = []
            streaks_wrong = []
            current_val = correct_arr[0]
            current_len = 1
            for i in range(1, len(correct_arr)):
                if correct_arr[i] == current_val:
                    current_len += 1
                else:
                    if current_val:
                        streaks_correct.append(current_len)
                    else:
                        streaks_wrong.append(current_len)
                    current_val = correct_arr[i]
                    current_len = 1
            if current_val:
                streaks_correct.append(current_len)
            else:
                streaks_wrong.append(current_len)

            if streaks_correct:
                print(f"    Correct streaks: max={max(streaks_correct)}, "
                      f"mean={np.mean(streaks_correct):.1f}, "
                      f"median={np.median(streaks_correct):.1f}")
            if streaks_wrong:
                print(f"    Wrong streaks:   max={max(streaks_wrong)}, "
                      f"mean={np.mean(streaks_wrong):.1f}, "
                      f"median={np.median(streaks_wrong):.1f}")

            p = test_df[f"{prefix}_correct"].mean()
            expected_correct_streak = 1 / (1 - p) if p < 1 else float('inf')
            expected_wrong_streak = 1 / p if p > 0 else float('inf')
            print(f"    Expected (if random, p={p:.2f}): "
                  f"correct={expected_correct_streak:.1f}, "
                  f"wrong={expected_wrong_streak:.1f}")
            if streaks_correct and streaks_wrong:
                print(f"    Actual/Expected ratio: "
                      f"correct={np.mean(streaks_correct)/expected_correct_streak:.2f}x, "
                      f"wrong={np.mean(streaks_wrong)/expected_wrong_streak:.2f}x")

        # ── 6. Autocorrelation at trading-relevant lags ──
        if len(test_df) > 50:
            print(f"\n  Autocorrelation (test set):")
            correct_series = test_df[f"{prefix}_correct"].astype(float).values
            var_c = correct_series.var()
            if var_c > 0:
                # With evenly-spaced samples, lag meaning depends on spacing
                spacing_min = (test_df["timestamp"].diff().median().total_seconds() / 60)
                print(f"    (sample spacing ~{spacing_min:.0f} min)")
                for lag in [1, 2, 3, 6, 12, 24, 48]:
                    if lag < len(correct_series) - 1:
                        autocorr = np.corrcoef(correct_series[:-lag], correct_series[lag:])[0, 1]
                        approx_time = lag * spacing_min
                        print(f"    Lag {lag:3d} (~{approx_time:.0f}min): {autocorr:+.4f}")

        # ── 7. Accuracy vs volatility (deciles) ──
        print(f"\n  Accuracy vs context volatility (deciles):")
        try:
            df["vol_decile"] = pd.qcut(df["context_volatility"], 10, labels=False, duplicates="drop")
            vol_stats = df.groupby("vol_decile").agg(
                vol_min=("context_volatility", "min"),
                vol_max=("context_volatility", "max"),
                token_acc=(f"{prefix}_correct", "mean"),
                dir_acc=(f"{prefix}_direction_correct", "mean"),
                mean_ent=(f"{prefix}_entropy", "mean"),
                n=(f"{prefix}_correct", "count"),
            ).reset_index()
            for _, row in vol_stats.iterrows():
                print(f"    vol [{row['vol_min']:.5f}-{row['vol_max']:.5f}]: "
                      f"tok={row['token_acc']*100:5.1f}%  dir={row['dir_acc']*100:5.1f}%  "
                      f"ent={row['mean_ent']:.2f}  n={int(row['n']):4d}")
        except Exception:
            print("    (not enough data for decile split)")

        # ── 8. Accuracy vs return magnitude (deciles) ──
        print(f"\n  Accuracy vs actual return magnitude (deciles):")
        try:
            df["ret_decile"] = pd.qcut(df["actual_return"].abs(), 10, labels=False, duplicates="drop")
            ret_stats = df.groupby("ret_decile").agg(
                ret_min=("actual_return", lambda x: x.abs().min()),
                ret_max=("actual_return", lambda x: x.abs().max()),
                token_acc=(f"{prefix}_correct", "mean"),
                dir_acc=(f"{prefix}_direction_correct", "mean"),
                n=(f"{prefix}_correct", "count"),
            ).reset_index()
            for _, row in ret_stats.iterrows():
                print(f"    |ret| [{row['ret_min']*100:.4f}%-{row['ret_max']*100:.4f}%]: "
                      f"tok={row['token_acc']*100:5.1f}%  dir={row['dir_acc']*100:5.1f}%  "
                      f"n={int(row['n']):4d}")
        except Exception:
            print("    (not enough data for decile split)")

        # ── 9. Accuracy vs entropy (is Kronos's own confidence reliable?) ──
        print(f"\n  Accuracy vs own entropy (deciles — is entropy a reliable confidence signal?):")
        try:
            df["ent_decile"] = pd.qcut(df[f"{prefix}_entropy"], 10, labels=False, duplicates="drop")
            ent_stats = df.groupby("ent_decile").agg(
                ent_min=(f"{prefix}_entropy", "min"),
                ent_max=(f"{prefix}_entropy", "max"),
                token_acc=(f"{prefix}_correct", "mean"),
                dir_acc=(f"{prefix}_direction_correct", "mean"),
                n=(f"{prefix}_correct", "count"),
            ).reset_index()
            for _, row in ent_stats.iterrows():
                bar = "█" * int(row["token_acc"] * 40)
                print(f"    ent [{row['ent_min']:.2f}-{row['ent_max']:.2f}]: "
                      f"tok={row['token_acc']*100:5.1f}%  dir={row['dir_acc']*100:5.1f}%  "
                      f"n={int(row['n']):4d}  {bar}")
        except Exception:
            print("    (not enough data for decile split)")


def print_effective_n_strategy(df):
    """Analyze effective N as a trading filter — threshold sweep."""
    print()
    print("=" * 75)
    print("PART 3: EFFECTIVE N AS TRADING FILTER")
    print("=" * 75)

    for prefix, head in [("s1", "S1 (coarse)"), ("s2", "S2 (fine)")]:
        print(f"\n{'─' * 75}")
        print(f"  {head}")
        print(f"{'─' * 75}")

        eff_n = df[f"{prefix}_effective_n"]
        tok_correct = df[f"{prefix}_correct"]
        dir_correct = df[f"{prefix}_direction_correct"]
        joint_dir = df["direction_correct"]

        # Distribution overview
        print(f"\n  Effective N distribution:")
        for p in [5, 10, 25, 50, 75, 90, 95]:
            print(f"    P{p}: {eff_n.quantile(p/100):.1f}")

        # Threshold sweep: if we only trade when eff_n < threshold
        print(f"\n  Threshold sweep — only trade when effective_n < threshold:")
        print(f"  {'Threshold':>10s}  {'Coverage':>8s}  {'Token Acc':>9s}  "
              f"{'Head Dir':>8s}  {'Joint Dir':>9s}  {'Samples':>8s}")
        print(f"  {'─'*10}  {'─'*8}  {'─'*9}  {'─'*8}  {'─'*9}  {'─'*8}")

        thresholds = [3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 30, 50, 1000]
        for t in thresholds:
            mask = eff_n < t
            n_pass = mask.sum()
            if n_pass < 10:
                continue
            coverage = n_pass / len(df) * 100
            tok_acc = tok_correct[mask].mean() * 100
            h_dir = dir_correct[mask].mean() * 100
            j_dir = joint_dir[mask].mean() * 100
            label = f"<{t}" if t < 1000 else "ALL"
            print(f"  {label:>10s}  {coverage:7.1f}%  {tok_acc:8.1f}%  "
                  f"{h_dir:7.1f}%  {j_dir:8.1f}%  {n_pass:>8d}")

        # Combined filter: s2_effective_n AND s1_effective_n
        print(f"\n  Cross-head filter — require BOTH heads low effective N:")
        if prefix == "s2":  # only print once
            s1_eff = df["s1_effective_n"]
            s2_eff = df["s2_effective_n"]
            print(f"  {'S1 < ':>6s}  {'S2 < ':>6s}  {'Coverage':>8s}  {'Token S1':>8s}  "
                  f"{'Token S2':>8s}  {'Joint Dir':>9s}  {'Samples':>8s}")
            print(f"  {'─'*6}  {'─'*6}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*9}  {'─'*8}")

            for s1_t in [8, 10, 15, 20]:
                for s2_t in [4, 5, 6, 8, 10]:
                    mask = (s1_eff < s1_t) & (s2_eff < s2_t)
                    n_pass = mask.sum()
                    if n_pass < 10:
                        continue
                    coverage = n_pass / len(df) * 100
                    s1_acc = df["s1_correct"][mask].mean() * 100
                    s2_acc = df["s2_correct"][mask].mean() * 100
                    j_dir = joint_dir[mask].mean() * 100
                    print(f"  {s1_t:>5d}  {s2_t:>5d}  {coverage:7.1f}%  {s1_acc:7.1f}%  "
                          f"{s2_acc:7.1f}%  {j_dir:8.1f}%  {n_pass:>8d}")

        # Effective N vs return profitability
        # If we trade in the predicted direction, what's the expected return?
        print(f"\n  Expected return when trading predicted direction (by eff_n bucket):")
        try:
            df["_eff_n_bucket"] = pd.qcut(eff_n, 5, labels=False, duplicates="drop")
            bucket_stats = df.groupby("_eff_n_bucket").agg(
                eff_n_min=(f"{prefix}_effective_n", "min"),
                eff_n_max=(f"{prefix}_effective_n", "max"),
                mean_abs_ret=("actual_return", lambda x: x.abs().mean()),
                dir_acc=(f"{prefix}_direction_correct", "mean"),
                joint_dir_acc=("direction_correct", "mean"),
                n=(f"{prefix}_correct", "count"),
            ).reset_index()

            for _, row in bucket_stats.iterrows():
                # If direction accuracy is d, expected return per trade:
                # E[r] = d * mean_abs_ret - (1-d) * mean_abs_ret = (2d - 1) * mean_abs_ret
                edge = (2 * row["joint_dir_acc"] - 1) * row["mean_abs_ret"] * 100
                print(f"    eff_n [{row['eff_n_min']:5.1f}-{row['eff_n_max']:5.1f}]: "
                      f"dir={row['joint_dir_acc']*100:5.1f}%  "
                      f"mean|ret|={row['mean_abs_ret']*100:.4f}%  "
                      f"E[edge]={edge:+.4f}%/trade  n={int(row['n'])}")
            df.drop(columns="_eff_n_bucket", inplace=True)
        except Exception:
            print("    (not enough data)")

    # Correlation between effective_n and other features
    print(f"\n{'─' * 75}")
    print(f"  Correlation matrix (effective_n vs other signals)")
    print(f"{'─' * 75}")
    corr_cols = ["s1_effective_n", "s2_effective_n", "s1_entropy", "s2_entropy",
                 "s1_top1_prob", "s2_top1_prob", "context_volatility"]
    corr_labels = ["s1_eff_n", "s2_eff_n", "s1_ent", "s2_ent",
                   "s1_top1", "s2_top1", "vol"]
    corr_matrix = df[corr_cols].corr()
    print(f"\n  {'':>10s}  ", "  ".join(f"{l:>8s}" for l in corr_labels))
    for i, label in enumerate(corr_labels):
        vals = "  ".join(f"{corr_matrix.iloc[i, j]:+8.3f}" for j in range(len(corr_labels)))
        print(f"  {label:>10s}  {vals}")


def print_summary(df):
    print()
    print("=" * 75)
    print("SUMMARY")
    print("=" * 75)

    # Direction accuracy
    dir_acc = df["direction_correct"].mean()
    gt_sanity = df["gt_decode_dir_correct"].mean()

    up_pct = (df["actual_dir"] == 1).mean()
    down_pct = (df["actual_dir"] == -1).mean()
    baseline = max(up_pct, down_pct)

    print(f"\n  Direction (full-sequence decode, pred tokens): {dir_acc*100:.1f}%")
    print(f"  GT sanity (full-sequence decode, real tokens):  {gt_sanity*100:.1f}%")
    print(f"  Baseline (always predict majority):             {baseline*100:.1f}% "
          f"(up={up_pct*100:.1f}%, down={down_pct*100:.1f}%)")
    print(f"  Edge: {(dir_acc - baseline)*100:+.1f}pp")
    if gt_sanity < 0.9:
        print(f"  WARNING: GT decode only {gt_sanity*100:.1f}% — tokenizer loses direction info!")

    for prefix, head in [("s1", "S1"), ("s2", "S2")]:
        c_mask = df[f"{prefix}_correct"]
        w_mask = ~c_mask
        c = df[c_mask]
        w = df[w_mask]

        print(f"\n  {head}:")
        print(f"    When CORRECT: high conviction (top1={c[f'{prefix}_top1_prob'].mean():.3f}), "
              f"low entropy ({c[f'{prefix}_entropy'].mean():.2f})")
        print(f"    When WRONG:   conviction={w[f'{prefix}_top1_prob'].mean():.3f}, "
              f"entropy={w[f'{prefix}_entropy'].mean():.2f}")

        ratio = w[f"{prefix}_effective_n"].mean() / max(c[f"{prefix}_effective_n"].mean(), 1e-8)
        print(f"    Effective tokens: correct={c[f'{prefix}_effective_n'].mean():.0f}, "
              f"wrong={w[f'{prefix}_effective_n'].mean():.0f} ({ratio:.1f}x more scattered)")

        dir_all = df[f"{prefix}_direction_correct"].mean()
        dir_wrong = w[f"{prefix}_direction_correct"].mean()
        print(f"    Per-head direction accuracy: {dir_all*100:.1f}%")
        print(f"    Even when wrong on exact token, direction correct: {dir_wrong*100:.1f}%")


def generate_pdf_report(df, pdf_path):
    """Generate a multi-page PDF report with charts."""
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour

    def assign_regime(ts):
        if ts.year == 2020:
            return "2020 Covid"
        elif ts.year == 2021:
            return "2021 Bull/Crash"
        elif ts.year == 2022:
            return "2022 Bear"
        elif ts.year == 2023:
            return "2023 Low-vol"
        elif ts.year == 2024:
            return "2024 ETF rally"
        elif ts.year >= 2025:
            return "2025-26 Recent"
        return "unknown"

    df["regime"] = df["timestamp"].apply(assign_regime)

    COLORS = {"correct": "#2ecc71", "wrong": "#e74c3c", "direction": "#3498db",
              "token": "#e67e22", "s1": "#9b59b6", "s2": "#1abc9c",
              "confident_wrong": "#c0392b", "uncertain_wrong": "#95a5a6"}

    with PdfPages(pdf_path) as pdf:
        # ─────────────────────────────────────────────────────────
        # PAGE 1: Title + Overview
        # ─────────────────────────────────────────────────────────
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor("white")
        ax = fig.add_subplot(111)
        ax.axis("off")

        title_text = "Kronos Error Analysis Report"
        subtitle = f"ETH 5m | {len(df)} samples | {df['timestamp'].min():%Y-%m-%d} to {df['timestamp'].max():%Y-%m-%d}"

        ax.text(0.5, 0.85, title_text, transform=ax.transAxes, fontsize=24,
                fontweight="bold", ha="center", va="top")
        ax.text(0.5, 0.78, subtitle, transform=ax.transAxes, fontsize=12,
                ha="center", va="top", color="gray")

        # Overview table
        overview_lines = []

        # Direction accuracy + baseline
        dir_acc = df["direction_correct"].mean() * 100
        gt_sanity = df["gt_decode_dir_correct"].mean() * 100
        up_pct = (df["actual_dir"] == 1).mean() * 100
        down_pct = (df["actual_dir"] == -1).mean() * 100
        baseline = max(up_pct, down_pct)
        overview_lines.append(f"DIRECTION (full-sequence decode)")
        overview_lines.append(f"  Predicted tokens: {dir_acc:.1f}%  |  Edge: {dir_acc - baseline:+.1f}pp")
        overview_lines.append(f"  GT tokens sanity: {gt_sanity:.1f}%  (should be ~100% if decode preserves dir)")
        overview_lines.append(f"  Baseline (majority): {baseline:.1f}%  "
                              f"(up={up_pct:.1f}%, down={down_pct:.1f}%)")
        overview_lines.append("")

        for prefix, head in [("S1", "S1 (coarse)"), ("S2", "S2 (fine)")]:
            p = prefix.lower()
            tok_acc = df[f"{p}_correct"].mean() * 100
            dir_acc = df[f"{p}_direction_correct"].mean() * 100
            c = df[df[f"{p}_correct"]]
            w = df[~df[f"{p}_correct"]]
            overview_lines.append(f"{head}")
            overview_lines.append(f"  Token accuracy: {tok_acc:.1f}%  |  Per-head direction: {dir_acc:.1f}%")
            overview_lines.append(f"  When correct: top1_prob={c[f'{p}_top1_prob'].mean():.3f}, "
                                  f"entropy={c[f'{p}_entropy'].mean():.2f}, "
                                  f"eff_N={c[f'{p}_effective_n'].mean():.0f}")
            overview_lines.append(f"  When wrong:   top1_prob={w[f'{p}_top1_prob'].mean():.3f}, "
                                  f"entropy={w[f'{p}_entropy'].mean():.2f}, "
                                  f"eff_N={w[f'{p}_effective_n'].mean():.0f}")
            overview_lines.append(f"  Direction correct even when token wrong: "
                                  f"{w[f'{p}_direction_correct'].mean()*100:.1f}%")
            overview_lines.append("")

        ax.text(0.1, 0.68, "\n".join(overview_lines), transform=ax.transAxes,
                fontsize=9.5, fontfamily="monospace", va="top")
        pdf.savefig(fig)
        plt.close(fig)

        # ─────────────────────────────────────────────────────────
        # PAGE 2: Distribution shape — correct vs wrong
        # ─────────────────────────────────────────────────────────
        fig, axes = plt.subplots(2, 3, figsize=(11, 8.5))
        fig.suptitle("Part 1: When Wrong — Concentrated or Scattered?", fontsize=14, fontweight="bold")

        for row, (prefix, head) in enumerate([("s1", "S1 (coarse)"), ("s2", "S2 (fine)")]):
            c = df[df[f"{prefix}_correct"]]
            w = df[~df[f"{prefix}_correct"]]

            # Top-1 probability histogram
            ax = axes[row, 0]
            ax.hist(c[f"{prefix}_top1_prob"], bins=40, alpha=0.7, color=COLORS["correct"],
                    label=f"Correct (n={len(c)})", density=True)
            ax.hist(w[f"{prefix}_top1_prob"], bins=40, alpha=0.7, color=COLORS["wrong"],
                    label=f"Wrong (n={len(w)})", density=True)
            ax.set_xlabel("Top-1 probability")
            ax.set_ylabel("Density")
            ax.set_title(f"{head} — Top-1 Prob")
            ax.legend(fontsize=7)

            # Effective N histogram
            ax = axes[row, 1]
            max_n = min(100, int(df[f"{prefix}_effective_n"].quantile(0.99)))
            ax.hist(c[f"{prefix}_effective_n"].clip(upper=max_n), bins=40, alpha=0.7,
                    color=COLORS["correct"], label="Correct", density=True)
            ax.hist(w[f"{prefix}_effective_n"].clip(upper=max_n), bins=40, alpha=0.7,
                    color=COLORS["wrong"], label="Wrong", density=True)
            ax.set_xlabel("Effective N tokens")
            ax.set_title(f"{head} — Effective N")
            ax.legend(fontsize=7)

            # Entropy histogram
            ax = axes[row, 2]
            ax.hist(c[f"{prefix}_entropy"], bins=40, alpha=0.7, color=COLORS["correct"],
                    label="Correct", density=True)
            ax.hist(w[f"{prefix}_entropy"], bins=40, alpha=0.7, color=COLORS["wrong"],
                    label="Wrong", density=True)
            ax.set_xlabel("Entropy")
            ax.set_title(f"{head} — Entropy")
            ax.legend(fontsize=7)

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig)
        plt.close(fig)

        # ─────────────────────────────────────────────────────────
        # PAGE 3: Confident-wrong vs uncertain-wrong breakdown
        # ─────────────────────────────────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle("Wrong Predictions Breakdown: Confident vs Uncertain", fontsize=14, fontweight="bold")

        for row, (prefix, head) in enumerate([("s1", "S1 (coarse)"), ("s2", "S2 (fine)")]):
            c = df[df[f"{prefix}_correct"]]
            w = df[~df[f"{prefix}_correct"]]
            threshold = c[f"{prefix}_top1_prob"].median()
            conf_wrong = w[w[f"{prefix}_top1_prob"] >= threshold]
            unc_wrong = w[w[f"{prefix}_top1_prob"] < threshold]

            # Pie chart: correct / confident-wrong / uncertain-wrong
            ax = axes[row, 0]
            sizes = [len(c), len(conf_wrong), len(unc_wrong)]
            labels = [f"Correct\n({len(c)}, {len(c)/len(df)*100:.0f}%)",
                      f"Confident-wrong\n({len(conf_wrong)}, {len(conf_wrong)/len(df)*100:.0f}%)",
                      f"Uncertain-wrong\n({len(unc_wrong)}, {len(unc_wrong)/len(df)*100:.0f}%)"]
            colors = [COLORS["correct"], COLORS["confident_wrong"], COLORS["uncertain_wrong"]]
            ax.pie(sizes, labels=labels, colors=colors, autopct="", startangle=90, textprops={"fontsize": 8})
            ax.set_title(f"{head} — Prediction Categories")

            # Direction accuracy per category
            ax = axes[row, 1]
            cats = ["Correct", "Conf-Wrong", "Unc-Wrong"]
            dir_accs = [
                c[f"{prefix}_direction_correct"].mean() * 100,
                conf_wrong[f"{prefix}_direction_correct"].mean() * 100 if len(conf_wrong) > 0 else 0,
                unc_wrong[f"{prefix}_direction_correct"].mean() * 100 if len(unc_wrong) > 0 else 0,
            ]
            prob_dirs = [
                c[f"{prefix}_prob_correct_dir"].mean() * 100,
                conf_wrong[f"{prefix}_prob_correct_dir"].mean() * 100 if len(conf_wrong) > 0 else 0,
                unc_wrong[f"{prefix}_prob_correct_dir"].mean() * 100 if len(unc_wrong) > 0 else 0,
            ]
            x = np.arange(len(cats))
            ax.bar(x - 0.15, dir_accs, 0.3, label="Direction Acc", color=COLORS["direction"])
            ax.bar(x + 0.15, prob_dirs, 0.3, label="P(correct dir)", color=COLORS["token"])
            ax.set_xticks(x)
            ax.set_xticklabels(cats, fontsize=9)
            ax.set_ylabel("%")
            ax.set_ylim(0, 105)
            ax.axhline(50, color="gray", ls="--", lw=0.8, label="Random (50%)")
            ax.legend(fontsize=7)
            ax.set_title(f"{head} — Direction Accuracy by Category")

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig)
        plt.close(fig)

        # ─────────────────────────────────────────────────────────
        # PAGE 4: Rolling accuracy over time
        # ─────────────────────────────────────────────────────────
        df_sorted = df.sort_values("timestamp").reset_index(drop=True)

        for prefix, head in [("s1", "S1 (coarse)"), ("s2", "S2 (fine)")]:
            fig, axes = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True)
            fig.suptitle(f"Temporal Analysis — {head}", fontsize=14, fontweight="bold")

            timestamps = df_sorted["timestamp"]

            # Rolling 6h token accuracy
            for ax_idx, (win_name, win_size) in enumerate([
                ("Rolling 30min", 6), ("Rolling 2h", 24), ("Rolling 6h", 72)
            ]):
                if win_size > len(df_sorted) // 3:
                    continue
                ax = axes[ax_idx]
                roll_tok = df_sorted[f"{prefix}_correct"].rolling(win_size, min_periods=win_size).mean()
                roll_dir = df_sorted["direction_correct"].rolling(win_size, min_periods=win_size).mean()

                ax.plot(timestamps, roll_tok * 100, color=COLORS["token"], lw=0.8,
                        alpha=0.8, label="Token accuracy")
                ax.plot(timestamps, roll_dir * 100, color=COLORS["direction"], lw=0.8,
                        alpha=0.8, label="Direction accuracy")
                ax.axhline(df[f"{prefix}_correct"].mean() * 100, color=COLORS["token"],
                           ls="--", lw=0.6, alpha=0.5)
                ax.axhline(df[f"{prefix}_direction_correct"].mean() * 100, color=COLORS["direction"],
                           ls="--", lw=0.6, alpha=0.5)
                ax.set_ylabel(f"{win_name} (%)")
                ax.set_ylim(0, 105)
                ax.legend(fontsize=7, loc="upper right")
                ax.grid(True, alpha=0.3)

            axes[-1].set_xlabel("Time")
            fig.autofmt_xdate(rotation=30)
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            pdf.savefig(fig)
            plt.close(fig)

        # ─────────────────────────────────────────────────────────
        # PAGE 6: Daily accuracy heatmap-style
        # ─────────────────────────────────────────────────────────
        df_sorted["date"] = df_sorted["timestamp"].dt.date
        daily = df_sorted.groupby("date").agg(
            s1_tok=("s1_correct", "mean"),
            s2_tok=("s2_correct", "mean"),
            dir_acc=("direction_correct", "mean"),
            vol=("context_volatility", "mean"),
            n=("s1_correct", "count"),
        ).reset_index()
        daily = daily[daily["n"] >= 3]

        if len(daily) > 5:
            fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
            fig.suptitle("Daily Accuracy (Direction (pred_s1, pred_s2))", fontsize=14, fontweight="bold")

            dates = pd.to_datetime(daily["date"])
            for ax, prefix, head in [(axes[0], "s1", "S1"), (axes[1], "s2", "S2")]:
                ax.bar(dates, daily[f"{prefix}_tok"] * 100, width=0.8, alpha=0.6,
                       color=COLORS["token"], label=f"{head} token acc")
                ax.plot(dates, daily["dir_acc"] * 100, color=COLORS["direction"],
                        lw=1.2, label="Directionection")
                ax.axhline(50, color="gray", ls="--", lw=0.5)
                ax.axhline(daily[f"{prefix}_tok"].mean() * 100, color=COLORS["token"],
                           ls="--", lw=0.8, alpha=0.5)
                ax.set_ylabel(f"{head} accuracy (%)")
                ax.set_ylim(0, 105)
                ax.legend(fontsize=8)
                ax.grid(True, alpha=0.3)

            axes[-1].set_xlabel("Date")
            fig.autofmt_xdate(rotation=30)
            fig.tight_layout(rect=[0, 0, 1, 0.94])
            pdf.savefig(fig)
            plt.close(fig)

        # ─────────────────────────────────────────────────────────
        # PAGE 7: 4h sessions + hour of day
        # ─────────────────────────────────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle("Accuracy by Time of Day", fontsize=14, fontweight="bold")

        for col, (prefix, head) in enumerate([("s1", "S1"), ("s2", "S2")]):
            # Hour of day
            ax = axes[0, col]
            hourly = df.groupby("hour").agg(
                tok=(f"{prefix}_correct", "mean"),
                s2_dir=("direction_correct", "mean"),
            ).reset_index()
            ax.bar(hourly["hour"] - 0.15, hourly["tok"] * 100, 0.3, color=COLORS["token"],
                   label=f"{head} token")
            ax.bar(hourly["hour"] + 0.15, hourly["s2_dir"] * 100, 0.3, color=COLORS["direction"],
                   label="Direction")
            ax.set_xlabel("Hour (UTC)")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title(f"{head} Token + Direction — By Hour")
            ax.set_ylim(0, 105)
            ax.axhline(50, color="gray", ls="--", lw=0.5)
            ax.legend(fontsize=7)

            # 4h session
            ax = axes[1, col]
            df["session_4h"] = (df["hour"] // 4) * 4
            session = df.groupby("session_4h").agg(
                tok=(f"{prefix}_correct", "mean"),
                s2_dir=("direction_correct", "mean"),
                n=(f"{prefix}_correct", "count"),
            ).reset_index()
            labels_4h = [f"{int(h):02d}-{int(h)+4:02d}" for h in session["session_4h"]]
            x = np.arange(len(labels_4h))
            ax.bar(x - 0.15, session["tok"] * 100, 0.3, color=COLORS["token"],
                   label=f"{head} token")
            ax.bar(x + 0.15, session["s2_dir"] * 100, 0.3, color=COLORS["direction"],
                   label="Direction")
            ax.set_xticks(x)
            ax.set_xticklabels(labels_4h)
            ax.set_xlabel("4h Session (UTC)")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title(f"{head} Token + Direction — By 4h Session")
            ax.set_ylim(0, 105)
            ax.axhline(50, color="gray", ls="--", lw=0.5)
            ax.legend(fontsize=7)

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig)
        plt.close(fig)

        # ─────────────────────────────────────────────────────────
        # PAGE 8: Regime comparison
        # ─────────────────────────────────────────────────────────
        regimes = sorted(df["regime"].unique())
        if len(regimes) > 1:
            fig, axes = plt.subplots(1, 2, figsize=(11, 6))
            fig.suptitle("Accuracy by Market Regime", fontsize=14, fontweight="bold")

            for ax, (prefix, head) in zip(axes, [("s1", "S1"), ("s2", "S2")]):
                regime_data = df.groupby("regime").agg(
                    tok=(f"{prefix}_correct", "mean"),
                    dir=(f"{prefix}_direction_correct", "mean"),
                    n=(f"{prefix}_correct", "count"),
                ).reindex(regimes)
                x = np.arange(len(regimes))
                ax.barh(x - 0.15, regime_data["tok"] * 100, 0.3, color=COLORS["token"], label="Token")
                ax.barh(x + 0.15, regime_data["dir"] * 100, 0.3, color=COLORS["direction"], label="Direction")
                ax.set_yticks(x)
                ax.set_yticklabels(regimes, fontsize=8)
                ax.set_xlabel("Accuracy (%)")
                ax.set_title(head)
                ax.set_xlim(0, 105)
                ax.axvline(50, color="gray", ls="--", lw=0.5)
                ax.legend(fontsize=7)
                # Annotate sample counts
                for i, n in enumerate(regime_data["n"]):
                    ax.text(2, i, f"n={int(n)}", va="center", fontsize=7, color="gray")

            fig.tight_layout(rect=[0, 0, 1, 0.93])
            pdf.savefig(fig)
            plt.close(fig)

        # ─────────────────────────────────────────────────────────
        # PAGE 9: Accuracy vs Entropy / Volatility / Return magnitude
        # ─────────────────────────────────────────────────────────
        fig, axes = plt.subplots(2, 4, figsize=(14, 8.5))
        fig.suptitle("Accuracy vs Market Conditions & Model Confidence", fontsize=14, fontweight="bold")

        for row, (prefix, head) in enumerate([("s1", "S1"), ("s2", "S2")]):
            # vs Entropy (deciles)
            ax = axes[row, 0]
            try:
                df["_ent_d"] = pd.qcut(df[f"{prefix}_entropy"], 10, labels=False, duplicates="drop")
                grp = df.groupby("_ent_d").agg(
                    ent=(f"{prefix}_entropy", "mean"),
                    tok=(f"{prefix}_correct", "mean"),
                    s2_dir=("direction_correct", "mean"),
                ).reset_index()
                ax.plot(grp["ent"], grp["tok"] * 100, "o-", color=COLORS["token"], label="Token", ms=4)
                ax.plot(grp["ent"], grp["s2_dir"] * 100, "s-", color=COLORS["direction"],
                        label="Direction", ms=4)
                ax.set_xlabel(f"{head} Entropy")
                ax.set_ylabel("Accuracy (%)")
                ax.set_title(f"{head} Entropy — vs Accuracy")
                ax.axhline(50, color="gray", ls="--", lw=0.5)
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
            except Exception:
                ax.text(0.5, 0.5, "Not enough data", ha="center", va="center", transform=ax.transAxes)

            # vs Volatility (deciles)
            ax = axes[row, 1]
            try:
                df["_vol_d"] = pd.qcut(df["context_volatility"], 10, labels=False, duplicates="drop")
                grp = df.groupby("_vol_d").agg(
                    vol=("context_volatility", "mean"),
                    tok=(f"{prefix}_correct", "mean"),
                    s2_dir=("direction_correct", "mean"),
                ).reset_index()
                ax.plot(grp["vol"] * 100, grp["tok"] * 100, "o-", color=COLORS["token"], label="Token", ms=4)
                ax.plot(grp["vol"] * 100, grp["s2_dir"] * 100, "s-", color=COLORS["direction"],
                        label="Direction", ms=4)
                ax.set_xlabel("Context Volatility (%)")
                ax.set_ylabel("Accuracy (%)")
                ax.set_title(f"{head} — vs Volatility")
                ax.axhline(50, color="gray", ls="--", lw=0.5)
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
            except Exception:
                ax.text(0.5, 0.5, "Not enough data", ha="center", va="center", transform=ax.transAxes)

            # vs Return magnitude (deciles)
            ax = axes[row, 2]
            try:
                df["_ret_d"] = pd.qcut(df["actual_return"].abs(), 10, labels=False, duplicates="drop")
                grp = df.groupby("_ret_d").agg(
                    ret=("actual_return", lambda x: x.abs().mean()),
                    tok=(f"{prefix}_correct", "mean"),
                    s2_dir=("direction_correct", "mean"),
                ).reset_index()
                ax.plot(grp["ret"] * 100, grp["tok"] * 100, "o-", color=COLORS["token"], label="Token", ms=4)
                ax.plot(grp["ret"] * 100, grp["s2_dir"] * 100, "s-", color=COLORS["direction"],
                        label="Direction", ms=4)
                ax.set_xlabel("|Actual Return| (%)")
                ax.set_ylabel("Accuracy (%)")
                ax.set_title(f"{head} — vs Return Size")
                ax.axhline(50, color="gray", ls="--", lw=0.5)
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
            except Exception:
                ax.text(0.5, 0.5, "Not enough data", ha="center", va="center", transform=ax.transAxes)

            # vs Predicted return magnitude (deciles)
            ax = axes[row, 3]
            try:
                pred_ret_abs = df["pred_return"].abs()
                df["_pred_d"] = pd.qcut(pred_ret_abs, 10, labels=False, duplicates="drop")
                grp = df.groupby("_pred_d").agg(
                    pred_ret=("pred_return", lambda x: x.abs().mean()),
                    tok=(f"{prefix}_correct", "mean"),
                    dir_acc=("direction_correct", "mean"),
                ).reset_index()
                ax.plot(grp["pred_ret"], grp["tok"] * 100, "o-", color=COLORS["token"], label="Token", ms=4)
                ax.plot(grp["pred_ret"], grp["dir_acc"] * 100, "s-", color=COLORS["direction"],
                        label="Direction", ms=4)
                ax.set_xlabel("|Predicted Return| (norm)")
                ax.set_ylabel("Accuracy (%)")
                ax.set_title(f"{head} — vs Predicted Move")
                ax.axhline(50, color="gray", ls="--", lw=0.5)
                ax.legend(fontsize=7)
                ax.grid(True, alpha=0.3)
            except Exception:
                ax.text(0.5, 0.5, "Not enough data", ha="center", va="center", transform=ax.transAxes)

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig)
        plt.close(fig)

        # ─────────────────────────────────────────────────────────
        # PAGE 10: Effective N — Threshold sweep (the money chart)
        # ─────────────────────────────────────────────────────────
        fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
        fig.suptitle("Effective N as Trading Filter — Threshold Sweep",
                     fontsize=14, fontweight="bold")

        for col, (prefix, head) in enumerate([("s1", "S1"), ("s2", "S2")]):
            eff_n = df[f"{prefix}_effective_n"]
            tok_correct = df[f"{prefix}_correct"]
            dir_correct = df["direction_correct"]

            thresholds = np.arange(2, 51, 1)
            coverages, tok_accs, dir_accs, edges = [], [], [], []

            for t in thresholds:
                mask = eff_n < t
                n_pass = mask.sum()
                if n_pass < 10:
                    coverages.append(0)
                    tok_accs.append(0)
                    dir_accs.append(0)
                    edges.append(0)
                    continue
                coverages.append(n_pass / len(df) * 100)
                tok_accs.append(tok_correct[mask].mean() * 100)
                d = dir_correct[mask].mean()
                dir_accs.append(d * 100)
                mean_ret = df["actual_return"][mask].abs().mean()
                edges.append((2 * d - 1) * mean_ret * 10000)  # in bps

            # Accuracy vs threshold
            ax = axes[0, col]
            ax.plot(thresholds, tok_accs, "o-", color=COLORS["token"], ms=2, lw=1, label="Token acc")
            ax.plot(thresholds, dir_accs, "^-", color=COLORS["direction"], ms=2, lw=1,
                    label="Direction")
            ax.set_xlabel(f"{head} Effective N threshold (<)")
            ax.set_ylabel("Accuracy (%)")
            ax.set_title(f"{head} — Accuracy vs Eff N Filter")
            ax.axhline(50, color="gray", ls="--", lw=0.5, label="Random")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

            # Coverage + Edge vs threshold (dual axis)
            ax = axes[1, col]
            ax.plot(thresholds, coverages, "o-", color=COLORS["correct"], ms=2, lw=1, label="Coverage (%)")
            ax.set_xlabel(f"{head} Effective N threshold (<)")
            ax.set_ylabel("Coverage (%)", color=COLORS["correct"])
            ax.set_ylim(0, 105)
            ax.grid(True, alpha=0.3)

            ax2 = ax.twinx()
            ax2.plot(thresholds, edges, "s-", color=COLORS["wrong"], ms=2, lw=1, label="Edge (bps/trade)")
            ax2.set_ylabel("Expected edge (bps/trade)", color=COLORS["wrong"])
            ax2.axhline(0, color="gray", ls="--", lw=0.5)

            # Combined legend
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc="center right")
            ax.set_title(f"{head} — Coverage & Edge vs Eff N Filter")

        fig.tight_layout(rect=[0, 0, 1, 0.94])
        pdf.savefig(fig)
        plt.close(fig)

        # ─────────────────────────────────────────────────────────
        # PAGE 11: Cross-head Effective N heatmap
        # ─────────────────────────────────────────────────────────
        fig, axes = plt.subplots(1, 3, figsize=(11, 5))
        fig.suptitle("Cross-Head Filter: Direction Accuracy by (S1 eff_n, S2 eff_n)",
                     fontsize=13, fontweight="bold")

        s1_bins = [0, 5, 8, 12, 20, 999]
        s2_bins = [0, 3, 5, 7, 10, 999]
        s1_labels = ["<5", "5-8", "8-12", "12-20", "20+"]
        s2_labels = ["<3", "3-5", "5-7", "7-10", "10+"]

        df["_s1_bin"] = pd.cut(df["s1_effective_n"], bins=s1_bins, labels=s1_labels)
        df["_s2_bin"] = pd.cut(df["s2_effective_n"], bins=s2_bins, labels=s2_labels)

        # Direction accuracy heatmap
        pivot_dir = df.pivot_table(values="direction_correct", index="_s2_bin",
                                   columns="_s1_bin", aggfunc="mean") * 100
        pivot_n = df.pivot_table(values="direction_correct", index="_s2_bin",
                                 columns="_s1_bin", aggfunc="count")
        # Edge heatmap
        pivot_ret = df.pivot_table(values="actual_return", index="_s2_bin",
                                   columns="_s1_bin", aggfunc=lambda x: x.abs().mean())

        for ax_idx, (pivot, title, fmt, cmap) in enumerate([
            (pivot_dir, "Joint Direction Acc (%)", ".1f", "RdYlGn"),
            (pivot_n, "Sample Count", ".0f", "Blues"),
        ]):
            ax = axes[ax_idx]
            im = ax.imshow(pivot.values, cmap=cmap, aspect="auto")
            ax.set_xticks(range(len(s1_labels)))
            ax.set_xticklabels(s1_labels, fontsize=8)
            ax.set_yticks(range(len(s2_labels)))
            ax.set_yticklabels(s2_labels, fontsize=8)
            ax.set_xlabel("S1 Effective N")
            ax.set_ylabel("S2 Effective N")
            ax.set_title(title, fontsize=10)
            # Annotate cells
            for i in range(len(s2_labels)):
                for j in range(len(s1_labels)):
                    try:
                        val = pivot.values[i, j]
                        if not np.isnan(val):
                            ax.text(j, i, f"{val:{fmt}}", ha="center", va="center", fontsize=7,
                                    color="white" if val > pivot.values[~np.isnan(pivot.values)].mean() else "black")
                    except (IndexError, TypeError):
                        pass
            plt.colorbar(im, ax=ax, shrink=0.8)

        # Edge in bps
        ax = axes[2]
        # Compute edge per cell: (2*dir_acc - 1) * mean_abs_ret * 10000
        edge_pivot = (2 * pivot_dir / 100 - 1) * pivot_ret * 10000
        im = ax.imshow(edge_pivot.values, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(len(s1_labels)))
        ax.set_xticklabels(s1_labels, fontsize=8)
        ax.set_yticks(range(len(s2_labels)))
        ax.set_yticklabels(s2_labels, fontsize=8)
        ax.set_xlabel("S1 Effective N")
        ax.set_ylabel("S2 Effective N")
        ax.set_title("Expected Edge (bps/trade)", fontsize=10)
        for i in range(len(s2_labels)):
            for j in range(len(s1_labels)):
                try:
                    val = edge_pivot.values[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7,
                                color="white" if val < 0 else "black")
                except (IndexError, TypeError):
                    pass
        plt.colorbar(im, ax=ax, shrink=0.8)

        df.drop(columns=["_s1_bin", "_s2_bin"], inplace=True)

        fig.tight_layout(rect=[0, 0, 1, 0.92])
        pdf.savefig(fig)
        plt.close(fig)

        # ─────────────────────────────────────────────────────────
        # PAGE 12: Autocorrelation + Streaks
        # ─────────────────────────────────────────────────────────
        test_df = df[df["dataset"] == "test"].sort_values("timestamp").reset_index(drop=True)
        if len(test_df) > 50:
            fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
            fig.suptitle("Temporal Clustering: Autocorrelation & Streaks", fontsize=14, fontweight="bold")

            spacing_min = test_df["timestamp"].diff().median().total_seconds() / 60

            for col, (prefix, head) in enumerate([("s1", "S1"), ("s2", "S2")]):
                # Autocorrelation
                ax = axes[0, col]
                correct_series = test_df[f"{prefix}_correct"].astype(float).values
                lags = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 36, 48]
                lags = [l for l in lags if l < len(correct_series) - 1]
                autocorrs = [np.corrcoef(correct_series[:-l], correct_series[l:])[0, 1] for l in lags]
                lag_times = [l * spacing_min for l in lags]

                ax.bar(range(len(lags)), autocorrs, color=COLORS["s1"] if prefix == "s1" else COLORS["s2"])
                ax.set_xticks(range(len(lags)))
                ax.set_xticklabels([f"{int(t)}m" for t in lag_times], rotation=45, fontsize=7)
                ax.axhline(0, color="black", lw=0.5)
                # 95% confidence band for white noise
                ci = 1.96 / np.sqrt(len(correct_series))
                ax.axhline(ci, color="gray", ls="--", lw=0.5)
                ax.axhline(-ci, color="gray", ls="--", lw=0.5)
                ax.set_ylabel("Autocorrelation")
                ax.set_title(f"{head} — Autocorrelation")
                ax.set_xlabel("Lag")

                # Streak histogram
                ax = axes[1, col]
                correct_arr = test_df[f"{prefix}_correct"].values
                streaks_c, streaks_w = [], []
                cur_val, cur_len = correct_arr[0], 1
                for i in range(1, len(correct_arr)):
                    if correct_arr[i] == cur_val:
                        cur_len += 1
                    else:
                        (streaks_c if cur_val else streaks_w).append(cur_len)
                        cur_val, cur_len = correct_arr[i], 1
                (streaks_c if cur_val else streaks_w).append(cur_len)

                max_streak = max(max(streaks_c, default=1), max(streaks_w, default=1))
                bins = np.arange(0.5, min(max_streak + 1.5, 25), 1)
                ax.hist(streaks_c, bins=bins, alpha=0.7, color=COLORS["correct"],
                        label=f"Correct (n={len(streaks_c)})")
                ax.hist(streaks_w, bins=bins, alpha=0.7, color=COLORS["wrong"],
                        label=f"Wrong (n={len(streaks_w)})")
                ax.set_xlabel("Streak length")
                ax.set_ylabel("Count")
                ax.set_title(f"{head} — Streak Distribution")
                ax.legend(fontsize=7)

            fig.tight_layout(rect=[0, 0, 1, 0.94])
            pdf.savefig(fig)
            plt.close(fig)

    # Clean up temp columns
    for col in ["_ent_d", "_vol_d", "_ret_d", "regime", "hour", "session_4h"]:
        if col in df.columns:
            df.drop(columns=col, inplace=True)

    print(f"PDF report saved to {pdf_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=5000,
                        help="Samples from test set (default: 5000)")
    parser.add_argument("--include-train", type=int, default=0, metavar="N",
                        help="Also sample N points from train set for cross-regime diversity")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    print(f"Device: {args.device}")
    print(f"Test samples: {args.n_samples}")
    if args.include_train:
        print(f"Train samples: {args.include_train} (NOTE: model trained on this data)")

    print("\nLoading models...")
    tokenizer, model = load_models(args.device)

    print("\nBuilding token-to-price mapping...")
    s1_prices, s2_prices = build_token_price_map(tokenizer, args.device)

    samples = load_all_data(
        args.device,
        n_samples_test=args.n_samples,
        n_samples_train=args.include_train,
    )

    print("\nRunning analysis...")
    df = analyze(tokenizer, model, samples, s1_prices, s2_prices, args.device)

    # Save raw results
    out_path = PROJECT_ROOT / "data/ml/kronos/analysis/error_analysis.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    print(f"\nSaved raw results to {out_path}")

    print_error_analysis(df)
    print_temporal_analysis(df)
    print_effective_n_strategy(df)
    print_summary(df)

    # Generate PDF report
    pdf_path = PROJECT_ROOT / "data/ml/kronos/analysis/kronos_error_analysis.pdf"
    generate_pdf_report(df, pdf_path)


if __name__ == "__main__":
    main()
