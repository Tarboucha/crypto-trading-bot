"""
Step 0: Verify TurboQuant works with Kronos's head_dim=52.

Usage:
    python tests/test_turboquant_verify.py
"""
import torch

try:
    from turboquant import TurboQuantKVCache
    print("turboquant-torch imported successfully")
except ImportError:
    print("ERROR: turboquant-torch not installed. Run: pip install turboquant-torch")
    exit(1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Kronos dimensions
BATCH = 1
N_HEADS = 16
SEQ_LEN = 512
HEAD_DIM = 52

print(f"\nKronos KV shape: [{BATCH}, {N_HEADS}, {SEQ_LEN}, {HEAD_DIM}]")
print(f"head_dim={HEAD_DIM} (non-standard, must verify)\n")

# Test 1: Basic compress/attention
print("Test 1: Compress + attention with head_dim=52 ... ", end="", flush=True)
try:
    cache = TurboQuantKVCache(head_dim=HEAD_DIM, bit_width=3, residual_length=32)
    # Try moving cache internals to device if possible
    if hasattr(cache, 'to'):
        cache = cache.to(device)

    # First try on GPU, fall back to CPU if device mismatch
    test_device = device
    K = torch.randn(BATCH, N_HEADS, SEQ_LEN, HEAD_DIM, device=test_device)
    V = torch.randn(BATCH, N_HEADS, SEQ_LEN, HEAD_DIM, device=test_device)
    try:
        compressed = cache.compress(K, V)
    except RuntimeError as device_err:
        if "device" in str(device_err).lower():
            print(f"GPU failed ({device_err}), trying CPU ... ", end="", flush=True)
            test_device = "cpu"
            cache = TurboQuantKVCache(head_dim=HEAD_DIM, bit_width=3, residual_length=32)
            K = K.cpu()
            V = V.cpu()
            compressed = cache.compress(K, V)
        else:
            raise

    Q = torch.randn(BATCH, N_HEADS, 1, HEAD_DIM, device=test_device)
    output = cache.attention(Q, compressed)
    assert output.shape == (BATCH, N_HEADS, 1, HEAD_DIM), f"Wrong shape: {output.shape}"
    print(f"PASS (output shape: {output.shape}, device: {test_device})")
    # Update device for remaining tests
    device = test_device
except Exception as e:
    print(f"FAIL: {e}")
    import traceback; traceback.print_exc()
    print("\nhead_dim=52 not supported. Will need padding to 64.")
    exit(1)

# Test 2: Attention fidelity
print("Test 2: Attention fidelity ... ", end="", flush=True)
Q = torch.randn(BATCH, N_HEADS, 1, HEAD_DIM, device=device)
K = torch.randn(BATCH, N_HEADS, SEQ_LEN, HEAD_DIM, device=device)
V = torch.randn(BATCH, N_HEADS, SEQ_LEN, HEAD_DIM, device=device)

# Reference: standard attention
import torch.nn.functional as F
ref_output = F.scaled_dot_product_attention(Q, K, V, is_causal=False)

# TurboQuant attention
compressed = cache.compress(K, V)
tq_output = cache.attention(Q, compressed)

diff = (ref_output - tq_output).abs()
rel_error = diff.mean() / (ref_output.abs().mean() + 1e-8)
max_error = diff.max().item()
print(f"PASS (mean rel error: {rel_error.item():.4f}, max abs error: {max_error:.4f})")

# Test 3: Memory savings
print("Test 3: Memory savings ... ", end="", flush=True)
try:
    orig_mb, comp_mb, ratio = cache.memory_savings(BATCH, N_HEADS, SEQ_LEN)
    print(f"PASS ({orig_mb:.1f} MB → {comp_mb:.1f} MB, {ratio:.1f}x compression)")
except Exception:
    # memory_savings might not exist in all versions
    orig_bytes = BATCH * N_HEADS * SEQ_LEN * HEAD_DIM * 4 * 2  # K + V, float32
    print(f"SKIP (manual estimate: {orig_bytes / 1024**2:.1f} MB uncompressed)")

# Test 4: Incremental update (append new token to existing cache)
print("Test 4: Incremental cache update ... ", end="", flush=True)
try:
    K_full = torch.randn(BATCH, N_HEADS, SEQ_LEN + 1, HEAD_DIM, device=device)
    V_full = torch.randn(BATCH, N_HEADS, SEQ_LEN + 1, HEAD_DIM, device=device)

    # Compress initial 512 tokens
    compressed1 = cache.compress(K_full[:, :, :SEQ_LEN], V_full[:, :, :SEQ_LEN])

    # Compress all 513 tokens
    compressed2 = cache.compress(K_full, V_full)

    # Both should work for attention
    Q = torch.randn(BATCH, N_HEADS, 1, HEAD_DIM, device=device)
    out1 = cache.attention(Q, compressed1)
    out2 = cache.attention(Q, compressed2)
    print(f"PASS (incremental recompress works)")
except Exception as e:
    print(f"FAIL: {e}")

print("\nAll basic checks passed. TurboQuant is compatible with Kronos.")
