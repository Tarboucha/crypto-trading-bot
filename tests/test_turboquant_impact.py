"""
Measure TurboQuant's actual impact on Kronos predictions.

Not attention fidelity — the real question: does the final token prediction change?

Usage:
    python tests/test_turboquant_impact.py
"""
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from turboquant import TurboQuantKVCache

PROJECT_ROOT = Path(__file__).parent.parent
KRONOS_PATH = PROJECT_ROOT / "third_party" / "kronos"
sys.path.insert(0, str(KRONOS_PATH))

from model import Kronos, KronosTokenizer
from model.kronos_cached import (
    kronos_prefill,
    kronos_decode_step,
    apply_rotary_with_offset,
    cached_self_attn_forward,
    cached_block_forward,
)
from model.kronos import sample_from_logits

device = "cuda" if torch.cuda.is_available() else "cpu"


def load_models():
    base = PROJECT_ROOT / "data/ml/kronos/finetuned/eth_5m"
    tokenizer = KronosTokenizer.from_pretrained(str(base / "tokenizer" / "best_model")).to(device).eval()
    model = Kronos.from_pretrained(str(base / "predictor" / "best_model")).to(device).eval()
    return tokenizer, model


def make_tq_cache(bit_width=4, key_bit_width=None, value_bit_width=None, residual_length=128):
    """Create TurboQuant cache for one layer, moved to device."""
    kbw = key_bit_width or bit_width
    vbw = value_bit_width or bit_width
    cache = TurboQuantKVCache(
        head_dim=52, bit_width=bit_width,
        key_bit_width=kbw, value_bit_width=vbw,
        residual_length=residual_length,
    )
    cache.key_quantizer.to(device)
    cache.value_quantizer.to(device)
    return cache


def compress_kvs(past_kvs, tq_caches):
    """Compress and decompress all layer KV caches through TurboQuant (lossy round-trip)."""
    lossy = []
    for (K, V), tq in zip(past_kvs, tq_caches):
        B, H, S, D = K.shape
        # Quantize and dequantize keys
        K_flat = K.reshape(-1, D)
        K_quant = tq.key_quantizer.quantize(K_flat)
        K_lossy = tq.key_quantizer.dequantize(K_quant).reshape(B, H, S, D)
        # Quantize and dequantize values
        V_flat = V.reshape(-1, D)
        V_quant = tq.value_quantizer.quantize(V_flat)
        V_lossy = tq.value_quantizer.dequantize(V_quant).reshape(B, H, S, D)
        lossy.append((K_lossy, V_lossy))
    return lossy


def test_prediction_impact(tokenizer, model, n_samples=50):
    """Compare predictions with and without TurboQuant."""

    configs = [
        ("K4/V4 res=128", dict(bit_width=4, residual_length=128)),
        ("K4/V2 res=128", dict(key_bit_width=4, value_bit_width=2, residual_length=128)),
        ("K8/V4 res=128", dict(key_bit_width=8, value_bit_width=4, residual_length=128)),
    ]

    torch.manual_seed(42)
    x = torch.randn(1, 512, 6, device=device) * 0.5

    with torch.no_grad():
        tokens = tokenizer.encode(x, half=True)

    stamps = torch.zeros(1, 512, 5, device=device)
    for t in range(512):
        stamps[0, t, 0] = (t * 5) % 60
        stamps[0, t, 1] = (t * 5 // 60) % 24
        stamps[0, t, 2] = (t // 288) % 7
        stamps[0, t, 3] = (t // 288) % 31 + 1
        stamps[0, t, 4] = 1

    with torch.no_grad():
        # Reference: no TurboQuant
        s1_logits_ref, context_ref, past_kvs_ref = kronos_prefill(
            model, tokens[0], tokens[1], stamp=stamps,
        )
        ref_probs = F.softmax(s1_logits_ref[:, -1, :], dim=-1).squeeze(0).cpu().numpy()
        ref_argmax = ref_probs.argmax()
        ref_top5 = np.argsort(ref_probs)[-5:][::-1]
        ref_entropy = -(ref_probs * np.log(ref_probs + 1e-10)).sum()

        print(f"  Reference (no TQ):")
        print(f"    Argmax token: {ref_argmax} (prob: {ref_probs[ref_argmax]:.4f})")
        print(f"    Top-5: {ref_top5} (probs: {[f'{ref_probs[t]:.4f}' for t in ref_top5]})")
        print(f"    Entropy: {ref_entropy:.4f}")
        print()

        for name, cfg in configs:
            # Create TQ caches for all 12 layers
            n_layers = len(past_kvs_ref)
            tq_caches = [make_tq_cache(**cfg) for _ in range(n_layers)]

            # Compress and decompress KV
            lossy_kvs = compress_kvs(past_kvs_ref, tq_caches)

            # Re-run the final norm + head with lossy KV
            # We need to recompute from the lossy KV through the model
            # Simpler: compare logits from prefill with lossy vs original KV
            # The logits come from the context (transformer output), not directly from KV
            # So we need to measure: how much does lossy KV change the transformer output?

            # Run transformer with lossy KV (just one decode step to see the effect)
            # We'll recompute attention for the last token using lossy cache
            x_embed = model.embedding([tokens[0], tokens[1]])
            x_embed = x_embed + model.time_emb(stamps)
            x_embed = model.token_drop(x_embed)

            # Run through all layers with original vs lossy KV
            x_orig = x_embed.clone()
            x_lossy = x_embed.clone()

            for i, layer in enumerate(model.transformer):
                # Original
                x_orig, _ = cached_block_forward(layer, x_orig, past_kv=None, position_offset=0)
                # We can't easily inject lossy KV into the block forward
                # Instead, let's just compare the prefill logits impact

            # Simpler approach: compare raw logit differences
            # Rerun prefill but inject noise equivalent to TQ compression error
            # Actually simplest: just measure KV error and project to logit space

            # Most direct: compare softmax at last position
            # We already have ref logits. Get TQ logits by:
            # 1. Compress KV from prefill
            # 2. Use lossy KV to recompute attention for last position only

            # Let's just compare the logits directly by running the full forward
            # with a hack: replace past_kvs with lossy versions and do a decode step

            # Use the original prefill context but with lossy KV for a decode step
            s1_logits_tq, _, _ = kronos_decode_step(
                model,
                tokens[0][:, -1:], tokens[1][:, -1:],
                stamp=stamps[:, -1:, :],
                past_kvs=lossy_kvs,
                position_offset=511,
            )

            # Also run reference decode step for fair comparison
            s1_logits_ref_step, _, _ = kronos_decode_step(
                model,
                tokens[0][:, -1:], tokens[1][:, -1:],
                stamp=stamps[:, -1:, :],
                past_kvs=past_kvs_ref,
                position_offset=511,
            )

            ref_step_probs = F.softmax(s1_logits_ref_step[:, -1, :], dim=-1).squeeze(0).cpu().numpy()
            tq_probs = F.softmax(s1_logits_tq[:, -1, :], dim=-1).squeeze(0).cpu().numpy()

            ref_step_argmax = ref_step_probs.argmax()
            tq_argmax = tq_probs.argmax()
            same_argmax = ref_step_argmax == tq_argmax

            # Top-5 overlap
            ref_top5 = set(np.argsort(ref_step_probs)[-5:])
            tq_top5 = set(np.argsort(tq_probs)[-5:])
            top5_overlap = len(ref_top5 & tq_top5)

            # KL divergence
            kl_div = (ref_step_probs * np.log((ref_step_probs + 1e-10) / (tq_probs + 1e-10))).sum()

            # Max probability difference
            prob_diff = np.abs(ref_step_probs - tq_probs)
            max_prob_diff = prob_diff.max()
            mean_prob_diff = prob_diff.mean()

            # Cosine similarity of probability distributions
            cosine = np.dot(ref_step_probs, tq_probs) / (np.linalg.norm(ref_step_probs) * np.linalg.norm(tq_probs) + 1e-10)

            print(f"  {name}:")
            print(f"    Same argmax: {same_argmax} (ref={ref_step_argmax}, tq={tq_argmax})")
            print(f"    Top-5 overlap: {top5_overlap}/5")
            print(f"    KL divergence: {kl_div:.6f}")
            print(f"    Prob cosine sim: {cosine:.6f}")
            print(f"    Max prob diff: {max_prob_diff:.6f}")
            print(f"    Mean prob diff: {mean_prob_diff:.6f}")
            print(f"    Ref top-1 prob: {ref_step_probs[ref_step_argmax]:.4f}, TQ top-1 prob: {tq_probs[tq_argmax]:.4f}")
            print()


def main():
    print(f"Device: {device}\n")
    print("Loading models...")
    tokenizer, model = load_models()

    print("\n" + "=" * 60)
    print("TURBOQUANT IMPACT ON TOKEN PREDICTION")
    print("=" * 60 + "\n")

    test_prediction_impact(tokenizer, model)


if __name__ == "__main__":
    main()
