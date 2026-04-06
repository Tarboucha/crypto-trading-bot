"""
Test TurboQuant fidelity with different configurations and real Kronos tensors.

Usage:
    python tests/test_turboquant_fidelity.py
"""
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from turboquant import TurboQuantKVCache

PROJECT_ROOT = Path(__file__).parent.parent
KRONOS_PATH = PROJECT_ROOT / "third_party" / "kronos"
sys.path.insert(0, str(KRONOS_PATH))

device = "cuda" if torch.cuda.is_available() else "cpu"

HEAD_DIM = 52
BATCH = 1
N_HEADS = 16
SEQ_LEN = 512


def make_cache(bit_width, residual_length):
    """Create a TurboQuantKVCache and move it to the correct device."""
    cache = TurboQuantKVCache(head_dim=HEAD_DIM, bit_width=bit_width, residual_length=residual_length)
    # Move internal quantizers to device
    cache.key_quantizer.to(device)
    cache.value_quantizer.to(device)
    return cache


def test_config(bit_width, residual_length, K, V, Q, label=""):
    """Test a specific TurboQuant configuration."""
    cache = make_cache(bit_width, residual_length)

    # Reference attention
    ref = F.scaled_dot_product_attention(Q, K, V, is_causal=False)

    # TurboQuant attention
    compressed = cache.compress(K, V)
    tq = cache.attention(Q, compressed)

    diff = (ref - tq).abs()
    mean_rel = (diff.mean() / (ref.abs().mean() + 1e-8)).item()
    max_rel = (diff.max() / (ref.abs().max() + 1e-8)).item()
    cosine_sim = F.cosine_similarity(ref.flatten(), tq.flatten(), dim=0).item()

    print(f"  {label:40s} | {mean_rel:>10.4f} | {max_rel:>10.4f} | {cosine_sim:>12.6f}")
    return mean_rel, cosine_sim


print(f"Device: {device}")
print(f"KV shape: [{BATCH}, {N_HEADS}, {SEQ_LEN}, {HEAD_DIM}]\n")

# Test 1: Random tensors, sweep configurations
print("=" * 90)
print("Test 1: Random tensors — sweep bit_width and residual_length")
print("=" * 90)

torch.manual_seed(42)
K = torch.randn(BATCH, N_HEADS, SEQ_LEN, HEAD_DIM, device=device)
V = torch.randn(BATCH, N_HEADS, SEQ_LEN, HEAD_DIM, device=device)
Q = torch.randn(BATCH, N_HEADS, 1, HEAD_DIM, device=device)

print(f"  {'Config':40s} | {'mean_rel':>10s} | {'max_rel':>10s} | {'cosine_sim':>12s}")
print(f"  {'-'*40} | {'-'*10} | {'-'*10} | {'-'*12}")

for bw in [2, 3, 4, 8]:
    for rl in [0, 32, 128, 256]:
        test_config(bw, rl, K, V, Q, label=f"bit_width={bw}, residual={rl}")
    print()

# Test 2: Real Kronos KV tensors
print("=" * 90)
print("Test 2: Real Kronos KV tensors")
print("=" * 90)

from model import Kronos, KronosTokenizer
from model.kronos_cached import kronos_prefill

base = PROJECT_ROOT / "data/ml/kronos/finetuned/eth_5m"
tokenizer = KronosTokenizer.from_pretrained(str(base / "tokenizer" / "best_model")).to(device).eval()
model = Kronos.from_pretrained(str(base / "predictor" / "best_model")).to(device).eval()

# Create input from real data distribution
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
    _, context, past_kvs = kronos_prefill(model, tokens[0], tokens[1], stamp=stamps)

# Use KV from layer 0
real_K, real_V = past_kvs[0]
real_Q = torch.randn(BATCH, N_HEADS, 1, HEAD_DIM, device=device)

print(f"\n  Real K stats: mean={real_K.mean():.4f}, std={real_K.std():.4f}, range=[{real_K.min():.4f}, {real_K.max():.4f}]")
print(f"  Real V stats: mean={real_V.mean():.4f}, std={real_V.std():.4f}, range=[{real_V.min():.4f}, {real_V.max():.4f}]")
print()

print(f"  {'Config':40s} | {'mean_rel':>10s} | {'max_rel':>10s} | {'cosine_sim':>12s}")
print(f"  {'-'*40} | {'-'*10} | {'-'*10} | {'-'*12}")

for bw in [2, 3, 4, 8]:
    for rl in [0, 32, 128, 256]:
        test_config(bw, rl, real_K, real_V, real_Q, label=f"bit_width={bw}, residual={rl}")
    print()

# Test 3: Fidelity per layer
print("=" * 90)
print("Test 3: Fidelity per layer (bit_width=4, residual=128)")
print("=" * 90)

print(f"\n  {'Layer':40s} | {'mean_rel':>10s} | {'max_rel':>10s} | {'cosine_sim':>12s}")
print(f"  {'-'*40} | {'-'*10} | {'-'*10} | {'-'*12}")

for i, (layer_K, layer_V) in enumerate(past_kvs):
    test_config(4, 128, layer_K, layer_V, real_Q, label=f"Layer {i}")
