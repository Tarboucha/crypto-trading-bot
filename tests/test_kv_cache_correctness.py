"""
Numerical correctness test: KV-cached inference vs original inference.

Verifies that auto_regressive_inference_cached produces identical logits
and predictions as the original auto_regressive_inference.

Usage:
    python tests/test_kv_cache_correctness.py
    python tests/test_kv_cache_correctness.py --device cpu
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).parent.parent
KRONOS_PATH = PROJECT_ROOT / "third_party" / "kronos"
sys.path.insert(0, str(KRONOS_PATH))

from model import Kronos, KronosTokenizer, KronosPredictor
from model.kronos import auto_regressive_inference
from model.kronos_cached import (
    auto_regressive_inference_cached,
    apply_rotary_with_offset,
    cached_self_attn_forward,
    cached_block_forward,
    kronos_prefill,
    kronos_decode_step,
)


def load_models(device: str):
    """Load fine-tuned models."""
    base = PROJECT_ROOT / "data/ml/kronos/finetuned/eth_5m"
    tok_path = str(base / "tokenizer" / "best_model")
    pred_path = str(base / "predictor" / "best_model")

    tokenizer = KronosTokenizer.from_pretrained(tok_path).to(device).eval()
    model = Kronos.from_pretrained(pred_path).to(device).eval()
    return tokenizer, model


def make_dummy_input(tokenizer, model, device, seq_len=64):
    """Create a reproducible dummy input for testing."""
    torch.manual_seed(42)
    # Random normalized input (simulating clipped, normalized candle data)
    x = torch.randn(1, seq_len, 6, device=device) * 0.5

    # Encode to tokens
    with torch.no_grad():
        x_token = tokenizer.encode(x, half=True)

    # Dummy timestamps (5 temporal features: minute, hour, weekday, day, month)
    stamps = torch.zeros(1, seq_len + 10, 5, device=device)
    for t in range(seq_len + 10):
        stamps[0, t, 0] = (t * 5) % 60       # minute
        stamps[0, t, 1] = (t * 5 // 60) % 24  # hour
        stamps[0, t, 2] = (t // 288) % 7      # weekday
        stamps[0, t, 3] = (t // 288) % 31 + 1 # day
        stamps[0, t, 4] = 1                    # month

    return x, x_token, stamps


# ──────────────────────────────────────────────
# Test 1: RoPE with offset produces same results
# ──────────────────────────────────────────────
def test_rope_offset(model, device):
    """Verify that apply_rotary_with_offset(offset=0) == original RoPE."""
    print("Test 1: RoPE with offset=0 matches original ... ", end="", flush=True)

    rotary = model.transformer[0].self_attn.rotary

    torch.manual_seed(123)
    q = torch.randn(1, 16, 32, 52, device=device)  # [batch, heads, seq, head_dim]
    k = torch.randn(1, 16, 32, 52, device=device)

    # Original
    q_orig, k_orig = rotary(q, k)

    # Cached with offset=0
    q_cached, k_cached = apply_rotary_with_offset(rotary, q, k, position_offset=0)

    q_diff = (q_orig - q_cached).abs().max().item()
    k_diff = (k_orig - k_cached).abs().max().item()

    assert q_diff < 1e-5, f"Q mismatch: {q_diff}"
    assert k_diff < 1e-5, f"K mismatch: {k_diff}"
    print(f"PASS (max diff: q={q_diff:.2e}, k={k_diff:.2e})")


# ──────────────────────────────────────────────
# Test 2: RoPE offset consistency
# ──────────────────────────────────────────────
def test_rope_offset_consistency(model, device):
    """Verify that processing [full seq] == [first part] + [second part with offset]."""
    print("Test 2: RoPE offset consistency (split vs full) ... ", end="", flush=True)

    rotary = model.transformer[0].self_attn.rotary

    torch.manual_seed(456)
    seq_len = 32
    split_at = 20

    q_full = torch.randn(1, 16, seq_len, 52, device=device)
    k_full = torch.randn(1, 16, seq_len, 52, device=device)

    # Full sequence
    q_rot_full, k_rot_full = apply_rotary_with_offset(rotary, q_full, k_full, position_offset=0)

    # Split: first part
    q1, k1 = apply_rotary_with_offset(rotary, q_full[:, :, :split_at], k_full[:, :, :split_at], position_offset=0)
    # Split: second part with offset
    q2, k2 = apply_rotary_with_offset(rotary, q_full[:, :, split_at:], k_full[:, :, split_at:], position_offset=split_at)

    q_combined = torch.cat([q1, q2], dim=2)
    k_combined = torch.cat([k1, k2], dim=2)

    q_diff = (q_rot_full - q_combined).abs().max().item()
    k_diff = (k_rot_full - k_combined).abs().max().item()

    assert q_diff < 1e-5, f"Q mismatch: {q_diff}"
    assert k_diff < 1e-5, f"K mismatch: {k_diff}"
    print(f"PASS (max diff: q={q_diff:.2e}, k={k_diff:.2e})")


# ──────────────────────────────────────────────
# Test 3: Cached attention prefill matches original
# ──────────────────────────────────────────────
def test_attention_prefill(model, device):
    """Verify that cached_self_attn_forward(past_kv=None) == original forward."""
    print("Test 3: Cached attention prefill matches original ... ", end="", flush=True)

    attn = model.transformer[0].self_attn

    torch.manual_seed(789)
    x = torch.randn(1, 32, model.d_model, device=device)

    # Original
    out_orig = attn(x, key_padding_mask=None)

    # Cached prefill (past_kv=None, offset=0)
    out_cached, present_kv = cached_self_attn_forward(attn, x, past_kv=None, position_offset=0)

    diff = (out_orig - out_cached).abs().max().item()
    assert diff < 1e-4, f"Output mismatch: {diff}"

    # Check cache shapes
    K, V = present_kv
    assert K.shape == (1, 16, 32, 52), f"K shape wrong: {K.shape}"
    assert V.shape == (1, 16, 32, 52), f"V shape wrong: {V.shape}"

    print(f"PASS (max diff: {diff:.2e}, cache K: {K.shape}, V: {V.shape})")


# ──────────────────────────────────────────────
# Test 4: Prefill + decode_step matches full forward
# ──────────────────────────────────────────────
def test_prefill_plus_decode(model, device):
    """
    Verify that prefill(tokens[:N]) + decode_step(token[N])
    produces the same output as a full forward(tokens[:N+1]).
    """
    print("Test 4: Prefill + decode_step matches full forward ... ", end="", flush=True)

    torch.manual_seed(101)
    seq_len = 32

    s1_ids = torch.randint(0, model.s1_vocab_size, (1, seq_len + 1), device=device)
    s2_ids = torch.randint(0, 2 ** model.s2_bits, (1, seq_len + 1), device=device)
    # Realistic timestamps: [minute, hour, weekday, day, month]
    stamps = torch.zeros(1, seq_len + 1, 5, device=device)
    for t in range(seq_len + 1):
        stamps[0, t, 0] = (t * 5) % 60        # minute
        stamps[0, t, 1] = (t * 5 // 60) % 24  # hour
        stamps[0, t, 2] = (t // 288) % 7       # weekday
        stamps[0, t, 3] = (t // 288) % 31 + 1  # day
        stamps[0, t, 4] = 1                     # month

    # Full forward for N+1 tokens
    s1_logits_full, context_full = model.decode_s1(
        s1_ids, s2_ids, stamp=stamps,
    )

    # Prefill for N tokens
    s1_logits_pre, context_pre, past_kvs = kronos_prefill(
        model, s1_ids[:, :seq_len], s2_ids[:, :seq_len], stamp=stamps[:, :seq_len],
    )

    # Decode step for token N
    s1_logits_step, context_step, _ = kronos_decode_step(
        model,
        s1_ids[:, seq_len:seq_len + 1],
        s2_ids[:, seq_len:seq_len + 1],
        stamp=stamps[:, seq_len:seq_len + 1],
        past_kvs=past_kvs,
        position_offset=seq_len,
    )

    # Compare last-position logits (this is what we sample from)
    logits_full_last = s1_logits_full[:, -1, :]
    logits_step_last = s1_logits_step[:, -1, :]

    diff = (logits_full_last - logits_step_last).abs().max().item()
    rel_diff = diff / (logits_full_last.abs().max().item() + 1e-8)

    # Also compare prefill portion
    prefill_diff = (s1_logits_full[:, :seq_len, :] - s1_logits_pre).abs().max().item()

    assert diff < 1e-2, f"Logits mismatch at last position: abs={diff:.4e}, rel={rel_diff:.4e}"
    assert prefill_diff < 1e-4, f"Prefill logits mismatch: {prefill_diff:.4e}"

    print(f"PASS (prefill diff: {prefill_diff:.2e}, decode step diff: {diff:.2e}, rel: {rel_diff:.2e})")


# ──────────────────────────────────────────────
# Test 5: Multi-step decode matches full recompute
# ──────────────────────────────────────────────
def test_multistep_decode(model, device):
    """
    Verify that prefill + N decode_steps matches running full forward
    at each step (like the original auto_regressive_inference does).
    """
    print("Test 5: Multi-step decode matches full recompute ... ", end="", flush=True)

    torch.manual_seed(202)
    context_len = 32
    n_steps = 5

    # Pre-generate all tokens (no sampling, just checking the forward pass)
    total_len = context_len + n_steps
    all_s1 = torch.randint(0, model.s1_vocab_size, (1, total_len), device=device)
    all_s2 = torch.randint(0, 2 ** model.s2_bits, (1, total_len), device=device)
    # Realistic timestamps
    all_stamps = torch.zeros(1, total_len, 5, device=device)
    for t in range(total_len):
        all_stamps[0, t, 0] = (t * 5) % 60
        all_stamps[0, t, 1] = (t * 5 // 60) % 24
        all_stamps[0, t, 2] = (t // 288) % 7
        all_stamps[0, t, 3] = (t // 288) % 31 + 1
        all_stamps[0, t, 4] = 1

    # Cached path: prefill + incremental steps
    _, _, past_kvs = kronos_prefill(
        model, all_s1[:, :context_len], all_s2[:, :context_len],
        stamp=all_stamps[:, :context_len],
    )

    cached_logits = []
    for step in range(n_steps):
        pos = context_len + step
        s1_logits, _, past_kvs = kronos_decode_step(
            model,
            all_s1[:, pos:pos + 1],
            all_s2[:, pos:pos + 1],
            stamp=all_stamps[:, pos:pos + 1],
            past_kvs=past_kvs,
            position_offset=pos,
        )
        cached_logits.append(s1_logits[:, -1, :])

    # Full recompute path: at each step, run the full sequence
    full_logits = []
    for step in range(n_steps):
        pos = context_len + step + 1  # include the new token
        s1_logits_full, _ = model.decode_s1(
            all_s1[:, :pos], all_s2[:, :pos], stamp=all_stamps[:, :pos],
        )
        full_logits.append(s1_logits_full[:, -1, :])

    max_diffs = []
    for step in range(n_steps):
        diff = (cached_logits[step] - full_logits[step]).abs().max().item()
        max_diffs.append(diff)

    worst = max(max_diffs)
    assert worst < 1e-2, f"Multi-step mismatch: {max_diffs}"
    print(f"PASS (per-step max diffs: {['%.2e' % d for d in max_diffs]})")


# ──────────────────────────────────────────────
# Test 6: Full end-to-end with real model
# ──────────────────────────────────────────────
def test_end_to_end(tokenizer, model, device):
    """
    Compare auto_regressive_inference vs auto_regressive_inference_cached
    with the same random seed. Predictions should be identical.
    """
    print("Test 6: End-to-end inference (cached vs original) ... ", end="", flush=True)

    x, x_token, stamps = make_dummy_input(tokenizer, model, device, seq_len=64)

    x_stamp = stamps[:, :64, :]
    y_stamp = stamps[:, 64:74, :]

    kwargs = dict(
        tokenizer=tokenizer,
        model=model,
        x=x,
        x_stamp=x_stamp,
        y_stamp=y_stamp,
        max_context=512,
        pred_len=10,
        clip=5,
        T=1.0,
        top_k=0,
        top_p=0.99,
        sample_count=1,  # single sample for deterministic comparison
        verbose=False,
    )

    # Run original with fixed seed
    torch.manual_seed(999)
    preds_orig = auto_regressive_inference(**kwargs)

    # Run cached with same seed
    torch.manual_seed(999)
    preds_cached = auto_regressive_inference_cached(**kwargs)

    diff = np.abs(preds_orig - preds_cached).max()
    rel_diff = diff / (np.abs(preds_orig).max() + 1e-8)

    assert diff < 1e-2, f"End-to-end mismatch: abs={diff:.4e}, rel={rel_diff:.4e}"
    print(f"PASS (max abs diff: {diff:.2e}, rel: {rel_diff:.2e})")
    print(f"       Original output range: [{preds_orig.min():.4f}, {preds_orig.max():.4f}]")
    print(f"       Cached output range:   [{preds_cached.min():.4f}, {preds_cached.max():.4f}]")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    print(f"Device: {device}\n")

    print("Loading models ...")
    tokenizer, model = load_models(device)
    print(f"  Predictor: {sum(p.numel() for p in model.parameters()):,} params")
    print(f"  Tokenizer: {sum(p.numel() for p in tokenizer.parameters()):,} params")
    print(f"  s1_bits: {model.s1_bits}, s2_bits: {model.s2_bits}, vocab_s1: {model.s1_vocab_size}")
    print(f"  d_model: {model.d_model}, n_heads: {model.n_heads}, n_layers: {model.n_layers}")
    print()

    print("=" * 60)
    print("NUMERICAL CORRECTNESS TESTS")
    print("=" * 60)

    test_rope_offset(model, device)
    test_rope_offset_consistency(model, device)
    test_attention_prefill(model, device)
    test_prefill_plus_decode(model, device)
    test_multistep_decode(model, device)
    test_end_to_end(tokenizer, model, device)

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
