"""
Local smoke test for the broccoli/gradboard API surface this repo depends on.

Purpose: surface *all* broccoli/gradboard breakages in one run (each call site is
isolated in its own try/except), so the migration loop is edit -> rerun -> read.

    .venv/bin/python scripts/broccoli_smoke.py

It exercises the exact kwargs used at the real call sites (sender.py,
receiver.py, vision.py, builder.py, train.py) with tiny dims on CPU. It does NOT
need data, SLURM, or CUDA.
"""
import sys, traceback
import torch
import torch.nn as nn

sys.path.insert(0, "code")

results = []  # (name, ok, err)

def check(name, fn):
    try:
        out = fn()
        results.append((name, True, None))
        return out
    except Exception:
        results.append((name, False, traceback.format_exc(limit=2)))
        return None

import broccoli
import broccoli.transformer, broccoli.vit, broccoli.activation
import gradboard.cycles
from gradboard.scheduler import PASS
from gradboard.optimiser import get_optimiser

D, H, L, SEQ = 32, 4, 2, 8

# --- vision.py: ViT2 (the heaviest kwarg surface) ---
def build_vit2():
    from models.backbone.vision import ViT2
    # d_model/heads kept generous: the new broccoli rotary embedding needs a
    # head dim large enough to rotate all positions (tiny dims fail in rope).
    m = ViT2(n_feats=(3, 64, 64), d_model=64, layers=L, heads=2, utility_tokens=1)
    x = torch.randn(2, 3, 64, 64)
    return m(x)
check("vision.ViT2 construct+forward", build_vit2)

# --- sender.py: broccoli.transformer.TransformerEncoder ---
def sender_encoder():
    return broccoli.transformer.TransformerEncoder(
        SEQ, D, L, H,
        absolute_position_embedding=True, relative_position_embedding=True,
        source_size=(SEQ,), ff_ratio=2,
        activation=broccoli.activation.SwiGLU, activation_kwargs=None,
        ff_dropout=0., msa_dropout=0., stochastic_depth=0.2, causal=False,
        bos_tokens=1, return_bos_tokens=False,
        pre_norm=False, post_norm=True, msa_scaling="d",
    )
check("sender.TransformerEncoder", sender_encoder)

# --- receiver.py: TransformerEncoder ---
def receiver_encoder():
    return broccoli.transformer.TransformerEncoder(
        SEQ, D, L, H,
        absolute_position_embedding=True, relative_position_embedding=True,
        source_size=(SEQ,), ff_ratio=2,
        activation=broccoli.activation.SwiGLU, stochastic_depth=0.1, causal=True,
        bos_tokens=1, return_bos_tokens=False,
        pre_norm=False, post_norm=True, msa_scaling="d",
    )
check("receiver.TransformerEncoder", receiver_encoder)

# --- sender/receiver: broccoli.transformer.MHAttention ---
def mhattention():
    return broccoli.transformer.MHAttention(
        D, H, dropout=0., causal=False, seq_len=SEQ, scaling="d",
    )
check("MHAttention", mhattention)

# --- sender.py: broccoli.vit.SequencePool ---
check("vit.SequencePool", lambda: broccoli.vit.SequencePool(D))

# --- activation: SwiGLU is now a factory function ---
check("activation.SwiGLU()", lambda: broccoli.activation.SwiGLU())

# --- builder.py: gradboard.optimiser.get_optimiser ---
def optimiser():
    m = nn.Linear(D, D)
    return get_optimiser(m, D, lr=1e-3, weight_decay=0.01)
check("get_optimiser(model, base_model_embedding_size, ...)", optimiser)

# --- train.py: gradboard cycles + PASS ---
def scheduler():
    warm = gradboard.cycles.Cycle(gradboard.cycles.ascent, 1000, 1, 16)
    main = gradboard.cycles.Cycle("cosine", 1000, 4, 16)
    seq = gradboard.cycles.CycleSequence([warm, main])
    m = nn.Linear(D, D)
    opt = torch.optim.AdamW(m.parameters())
    return PASS(seq, m, opt, scaler=torch.amp.GradScaler())
check("gradboard cycles + PASS", scheduler)

print("\n" + "=" * 70)
for name, ok, err in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        last = err.strip().splitlines()[-1]
        print(f"        {last}")
n_fail = sum(1 for _, ok, _ in results if not ok)
print("=" * 70)
print(f"{len(results)-n_fail}/{len(results)} passed")
