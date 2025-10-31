# NPNet-V Weak Pairs Generator

Generate semantic-enhanced noise pairs for text-to-video diffusion training.

## Quick Start

### 1. Generate Weak Pairs
```bash
python Work/make_weak_pairs.py \
```

Output: `weak_pairs/*.pt` files

### 2. Generate Videos

**Modify** `scripts/evaluation/funcs.py`:
```python
def batch_ddim_sampling(..., x_T=None, **kwargs):  # Add x_T=None
    for _ in range(n_samples):
        if x_T is None:
            x_T_sample = torch.randn(noise_shape, device=model.device)
        else:
            x_T_sample = x_T.to(model.device)
        samples, _ = ddim_sampler.sample(..., x_T=x_T_sample, ...)
```

**Run**:
```bash
python Work/inference_from_weak_pairs.py \
```

Output: `x_T/` vs `x_T_target/` videos

## What's Inside

Each `.pt` file contains:
```python
{
    "x_T": tensor,           # Original noise
    "x_T_target": tensor,    # x_T + λ·(ε_cond - ε_uncond)
    "prompt": str,
    "meta": {...}
}
```

---

Based on VideoCrafter2