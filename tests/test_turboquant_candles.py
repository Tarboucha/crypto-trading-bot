"""
Compare decoded candle predictions with and without TurboQuant.

Runs full autoregressive inference (10 steps), decodes tokens back to
OHLCV candles, and compares the actual predicted prices.

Usage:
    python tests/test_turboquant_candles.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).parent.parent
KRONOS_PATH = PROJECT_ROOT / "third_party" / "kronos"
sys.path.insert(0, str(KRONOS_PATH))

from model import Kronos, KronosTokenizer, KronosPredictor
from model.kronos import auto_regressive_inference
from model.kronos_cached import auto_regressive_inference_cached

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_models():
    base = PROJECT_ROOT / "data/ml/kronos/finetuned/eth_5m"
    tokenizer = KronosTokenizer.from_pretrained(str(base / "tokenizer" / "best_model")).to(device).eval()
    model = Kronos.from_pretrained(str(base / "predictor" / "best_model")).to(device).eval()
    return tokenizer, model


def load_real_context():
    """Load real ETH candles as context."""
    test_path = PROJECT_ROOT / "data/ml/kronos/prepared/ETH_5m_test.csv"
    df = pd.read_csv(test_path)
    df["timestamps"] = pd.to_datetime(df["timestamps"])

    price_cols = ["open", "high", "low", "close", "volume", "amount"]
    time_cols = ["minute", "hour", "weekday", "day", "month"]

    df["minute"] = df["timestamps"].dt.minute
    df["hour"] = df["timestamps"].dt.hour
    df["weekday"] = df["timestamps"].dt.weekday
    df["day"] = df["timestamps"].dt.day
    df["month"] = df["timestamps"].dt.month

    # Take a window from the middle
    start = len(df) // 2
    context = df.iloc[start:start + 512]

    x = context[price_cols].values.astype(np.float32)
    x_stamp = context[time_cols].values.astype(np.float32)

    # Normalize
    x_mean = x.mean(axis=0)
    x_std = x.std(axis=0)
    x_norm = (x - x_mean) / (x_std + 1e-5)
    x_norm = np.clip(x_norm, -5, 5)

    # Future timestamps
    last_ts = context["timestamps"].iloc[-1]
    y_times = pd.date_range(start=last_ts + pd.Timedelta(minutes=5), periods=10, freq="5min")
    y_time_df = pd.DataFrame({
        "minute": y_times.minute,
        "hour": y_times.hour,
        "weekday": y_times.weekday,
        "day": y_times.day,
        "month": y_times.month,
    })
    y_stamp = y_time_df.values.astype(np.float32)

    return x_norm, x_stamp, y_stamp, x_mean, x_std


def run_inference(tokenizer, model, x_norm, x_stamp, y_stamp, x_mean, x_std, use_tq=False, tq_config=None):
    """Run autoregressive inference and decode to price candles."""
    from turboquant import TurboQuantKVCache
    from model.kronos_cached import (
        kronos_prefill, kronos_decode_step, cached_block_forward,
    )
    from model.kronos import sample_from_logits

    x_tensor = torch.from_numpy(x_norm[np.newaxis]).to(device)
    x_stamp_tensor = torch.from_numpy(x_stamp[np.newaxis]).to(device)
    y_stamp_tensor = torch.from_numpy(y_stamp[np.newaxis]).to(device)

    with torch.no_grad():
        x_tensor = torch.clip(x_tensor, -5, 5)
        full_stamp = torch.cat([x_stamp_tensor, y_stamp_tensor], dim=1)

        x_token = tokenizer.encode(x_tensor, half=True)
        initial_seq_len = x_tensor.size(1)
        pred_len = 10

        # Prefill
        s1_logits, context, past_kvs = kronos_prefill(
            model, x_token[0], x_token[1], stamp=full_stamp[:, :initial_seq_len],
        )

        # Optionally compress KV with TurboQuant
        if use_tq and tq_config:
            lossy_kvs = []
            for K, V in past_kvs:
                B, H, S, D = K.shape
                tq = TurboQuantKVCache(
                    head_dim=D, **tq_config,
                )
                tq.key_quantizer.to(device)
                tq.value_quantizer.to(device)
                K_flat = K.reshape(-1, D)
                V_flat = V.reshape(-1, D)
                K_lossy = tq.key_quantizer.dequantize(tq.key_quantizer.quantize(K_flat)).reshape(B, H, S, D)
                V_lossy = tq.value_quantizer.dequantize(tq.value_quantizer.quantize(V_flat)).reshape(B, H, S, D)
                lossy_kvs.append((K_lossy, V_lossy))
            past_kvs = lossy_kvs

        generated_pre = []
        generated_post = []
        context_outputs = [context]

        for i in range(pred_len):
            s1_logits_last = s1_logits[:, -1, :]
            sample_pre = sample_from_logits(s1_logits_last, temperature=1.0, top_k=0, top_p=0.99)

            full_context = torch.cat(context_outputs, dim=1)
            s2_logits = model.decode_s2(full_context, sample_pre)
            s2_logits_last = s2_logits[:, -1, :]
            sample_post = sample_from_logits(s2_logits_last, temperature=1.0, top_k=0, top_p=0.99)

            generated_pre.append(sample_pre.squeeze(-1))
            generated_post.append(sample_post.squeeze(-1))

            if i < pred_len - 1:
                pos = initial_seq_len + i
                next_stamp = full_stamp[:, pos:pos + 1, :]
                s1_logits, step_context, past_kvs = kronos_decode_step(
                    model, sample_pre, sample_post, next_stamp, past_kvs, position_offset=pos,
                )
                context_outputs.append(step_context)

        # Decode tokens to candles
        gen_pre = torch.stack(generated_pre, dim=1)
        gen_post = torch.stack(generated_post, dim=1)
        full_pre = torch.cat([x_token[0], gen_pre], dim=1)
        full_post = torch.cat([x_token[1], gen_post], dim=1)

        total_seq_len = initial_seq_len + pred_len
        context_start = max(0, total_seq_len - 512)
        input_tokens = [
            full_pre[:, context_start:total_seq_len].contiguous(),
            full_post[:, context_start:total_seq_len].contiguous(),
        ]
        z = tokenizer.decode(input_tokens, half=True)
        preds = z[:, -pred_len:, :].cpu().numpy().squeeze(0)

        # Denormalize
        preds = preds * (x_std + 1e-5) + x_mean

    return preds


def main():
    print(f"Device: {device}\n")
    print("Loading models...")
    tokenizer, model = load_models()

    print("Loading real ETH context (512 candles)...")
    x_norm, x_stamp, y_stamp, x_mean, x_std = load_real_context()

    price_cols = ["open", "high", "low", "close", "volume", "amount"]
    n_runs = 5

    configs = [
        ("No TurboQuant", False, None),
        ("K4/V4 res=128", True, dict(bit_width=4, residual_length=128)),
        ("K4/V2 res=128", True, dict(key_bit_width=4, value_bit_width=2, residual_length=128)),
    ]

    print(f"\nRunning {n_runs} predictions per config (same seed each time)...\n")

    for config_name, use_tq, tq_config in configs:
        print(f"{'=' * 70}")
        print(f"  {config_name}")
        print(f"{'=' * 70}")

        all_preds = []
        for run in range(n_runs):
            torch.manual_seed(run * 1000)
            preds = run_inference(tokenizer, model, x_norm, x_stamp, y_stamp, x_mean, x_std, use_tq, tq_config)
            all_preds.append(preds)

        # Show mean prediction across runs
        mean_preds = np.mean(all_preds, axis=0)
        print(f"\n  Mean predicted candles (close price, {n_runs} runs):")
        for step in range(10):
            close = mean_preds[step, 3]  # close is index 3
            print(f"    Step {step+1}: close = {close:.2f}")
        print()

    # Now compare: for each seed, how different are the predictions?
    print(f"\n{'=' * 70}")
    print(f"  DIRECT COMPARISON (same seed, with vs without TQ)")
    print(f"{'=' * 70}\n")

    for seed in range(3):
        torch.manual_seed(seed * 1000)
        ref_preds = run_inference(tokenizer, model, x_norm, x_stamp, y_stamp, x_mean, x_std, False, None)

        for config_name, use_tq, tq_config in configs[1:]:
            torch.manual_seed(seed * 1000)
            tq_preds = run_inference(tokenizer, model, x_norm, x_stamp, y_stamp, x_mean, x_std, use_tq, tq_config)

            close_ref = ref_preds[:, 3]
            close_tq = tq_preds[:, 3]
            abs_diff = np.abs(close_ref - close_tq)
            pct_diff = abs_diff / (np.abs(close_ref) + 1e-8) * 100

            print(f"  Seed {seed}, {config_name}:")
            print(f"    {'Step':>6s} | {'Ref close':>12s} | {'TQ close':>12s} | {'Abs diff':>10s} | {'Pct diff':>10s}")
            print(f"    {'-'*6} | {'-'*12} | {'-'*12} | {'-'*10} | {'-'*10}")
            for step in range(10):
                print(f"    {step+1:>6d} | {close_ref[step]:>12.2f} | {close_tq[step]:>12.2f} | {abs_diff[step]:>10.2f} | {pct_diff[step]:>9.4f}%")
            print(f"    {'':>6s} | {'':>12s} | {'':>12s} | avg={np.mean(abs_diff):>.2f} | avg={np.mean(pct_diff):>.4f}%")
            print()


def benchmark_speed_memory(tokenizer, model, x_norm, x_stamp, y_stamp, x_mean, x_std):
    """Compare inference speed and memory with/without TurboQuant."""
    import time

    configs = [
        ("No TurboQuant", False, None),
        ("K4/V4 res=128", True, dict(bit_width=4, residual_length=128)),
        ("K4/V2 res=128", True, dict(key_bit_width=4, value_bit_width=2, residual_length=128)),
    ]

    print(f"\n{'=' * 70}")
    print(f"  INFERENCE SPEED & MEMORY BENCHMARK")
    print(f"{'=' * 70}\n")

    n_runs = 5
    warmup = 2

    for config_name, use_tq, tq_config in configs:
        # Warmup
        for _ in range(warmup):
            torch.manual_seed(0)
            run_inference(tokenizer, model, x_norm, x_stamp, y_stamp, x_mean, x_std, use_tq, tq_config)

        # Memory
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

        # Timed runs
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        times = []
        for i in range(n_runs):
            torch.manual_seed(i * 1000)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            run_inference(tokenizer, model, x_norm, x_stamp, y_stamp, x_mean, x_std, use_tq, tq_config)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append(t1 - t0)

        mean_t = np.mean(times)
        std_t = np.std(times)

        if torch.cuda.is_available():
            peak_mb = torch.cuda.max_memory_allocated() / 1024**2
        else:
            peak_mb = 0

        print(f"  {config_name}:")
        print(f"    Time:   {mean_t:.4f} ± {std_t:.4f} s")
        print(f"    Memory: {peak_mb:.0f} MB peak GPU")
        print()


if __name__ == "__main__":
    main()

    print("\nLoading for benchmark...")
    tokenizer, model = load_models()
    x_norm, x_stamp, y_stamp, x_mean, x_std = load_real_context()
    benchmark_speed_memory(tokenizer, model, x_norm, x_stamp, y_stamp, x_mean, x_std)
