"""
Benchmark KV-cached inference vs original inference.

Usage:
    python tests/test_kv_cache_speed.py
    python tests/test_kv_cache_speed.py --n-runs 20 --device cpu
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
KRONOS_PATH = PROJECT_ROOT / "third_party" / "kronos"
sys.path.insert(0, str(KRONOS_PATH))

from model import Kronos, KronosTokenizer
from model.kronos import auto_regressive_inference
from model.kronos_cached import auto_regressive_inference_cached


def load_models(device):
    base = PROJECT_ROOT / "data/ml/kronos/finetuned/eth_5m"
    tok_path = str(base / "tokenizer" / "best_model")
    pred_path = str(base / "predictor" / "best_model")

    tokenizer = KronosTokenizer.from_pretrained(tok_path).to(device).eval()
    model = Kronos.from_pretrained(pred_path).to(device).eval()
    return tokenizer, model


def make_input(device, context_len=512):
    """Create a realistic dummy input."""
    torch.manual_seed(42)
    x = torch.randn(1, context_len, 6, device=device) * 0.5

    stamps = torch.zeros(1, context_len + 10, 5, device=device)
    for t in range(context_len + 10):
        stamps[0, t, 0] = (t * 5) % 60
        stamps[0, t, 1] = (t * 5 // 60) % 24
        stamps[0, t, 2] = (t // 288) % 7
        stamps[0, t, 3] = (t // 288) % 31 + 1
        stamps[0, t, 4] = 1

    x_stamp = stamps[:, :context_len, :]
    y_stamp = stamps[:, context_len:context_len + 10, :]
    return x, x_stamp, y_stamp


def benchmark(func, tokenizer, model, x, x_stamp, y_stamp, n_runs, sample_count, warmup=2):
    """Time a function over n_runs, returning list of times."""
    kwargs = dict(
        tokenizer=tokenizer, model=model, x=x,
        x_stamp=x_stamp, y_stamp=y_stamp,
        max_context=512, pred_len=10, clip=5,
        T=1.0, top_k=0, top_p=0.99,
        sample_count=sample_count, verbose=False,
    )

    # Warmup (not counted)
    for _ in range(warmup):
        func(**kwargs)

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    times = []
    for i in range(n_runs):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        func(**kwargs)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    return times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-runs", type=int, default=10)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    print(f"Device: {device}")
    print(f"Runs: {args.n_runs}")

    print("\nLoading models...")
    tokenizer, model = load_models(device)

    print("Preparing input (512 context tokens, 10 prediction steps)...\n")
    x, x_stamp, y_stamp = make_input(device)

    for sample_count in [1, 5]:
        print(f"{'=' * 60}")
        print(f"  sample_count = {sample_count}")
        print(f"{'=' * 60}")

        print(f"\n  Benchmarking ORIGINAL (no KV cache)...")
        times_orig = benchmark(
            auto_regressive_inference, tokenizer, model,
            x, x_stamp, y_stamp, args.n_runs, sample_count,
        )

        print(f"  Benchmarking CACHED (KV cache)...")
        times_cached = benchmark(
            auto_regressive_inference_cached, tokenizer, model,
            x, x_stamp, y_stamp, args.n_runs, sample_count,
        )

        orig_mean = np.mean(times_orig)
        orig_std = np.std(times_orig)
        cached_mean = np.mean(times_cached)
        cached_std = np.std(times_cached)
        speedup = orig_mean / cached_mean

        print(f"\n  Results (sample_count={sample_count}):")
        print(f"  ┌────────────┬────────────┬────────────┐")
        print(f"  │            │ Mean (s)   │ Std (s)    │")
        print(f"  ├────────────┼────────────┼────────────┤")
        print(f"  │ Original   │ {orig_mean:>10.4f} │ {orig_std:>10.4f} │")
        print(f"  │ Cached     │ {cached_mean:>10.4f} │ {cached_std:>10.4f} │")
        print(f"  └────────────┴────────────┴────────────┘")
        print(f"  Speedup: {speedup:.2f}x")
        print(f"  Time saved per inference: {(orig_mean - cached_mean)*1000:.1f} ms")
        print()

    # Memory usage
    if torch.cuda.is_available():
        print(f"{'=' * 60}")
        print(f"  GPU Memory")
        print(f"{'=' * 60}")
        torch.cuda.reset_peak_memory_stats()

        # Original
        auto_regressive_inference(
            tokenizer=tokenizer, model=model, x=x,
            x_stamp=x_stamp, y_stamp=y_stamp,
            max_context=512, pred_len=10, clip=5,
            T=1.0, top_k=0, top_p=0.99,
            sample_count=5, verbose=False,
        )
        orig_peak = torch.cuda.max_memory_allocated() / 1024**2
        torch.cuda.reset_peak_memory_stats()

        # Cached
        auto_regressive_inference_cached(
            tokenizer=tokenizer, model=model, x=x,
            x_stamp=x_stamp, y_stamp=y_stamp,
            max_context=512, pred_len=10, clip=5,
            T=1.0, top_k=0, top_p=0.99,
            sample_count=5, verbose=False,
        )
        cached_peak = torch.cuda.max_memory_allocated() / 1024**2

        print(f"\n  Peak GPU memory (sample_count=5):")
        print(f"    Original: {orig_peak:.0f} MB")
        print(f"    Cached:   {cached_peak:.0f} MB")
        print(f"    Delta:    {cached_peak - orig_peak:+.0f} MB (KV cache overhead)")
        print()


if __name__ == "__main__":
    main()
