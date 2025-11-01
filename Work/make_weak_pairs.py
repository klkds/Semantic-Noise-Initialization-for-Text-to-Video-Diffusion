# make_weak_pairs.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse, torch
from typing import List
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from utils.utils import instantiate_from_config
from scripts.evaluation.funcs import load_model_checkpoint, load_prompts
try:
    import xformers.ops

    def safe_memory_efficient_attention(q, k, v, attn_bias=None, op=None):
        """
        Fallback attention: pure PyTorch, no custom CUDA kernel.
        q,k,v: [B*h, N, d]
        return: [B*h, N, d]
        """
        # q, k, v come in already batched per-head in xformers style
        # we'll just do scaled dot-product attention manually.
        d = q.shape[-1]
        scale = (d ** -0.5)

        # [B*h, N, N]
        attn = torch.bmm(q * scale, k.transpose(1, 2))

        # no attn_bias / mask for now; if their code passes mask it should've
        # been applied before calling xformers in this repo
        attn = torch.softmax(attn, dim=-1)

        # [B*h, N, d]
        out = torch.bmm(attn, v)
        return out

    # patch it
    xformers.ops.memory_efficient_attention = safe_memory_efficient_attention
    print("[Patch] Using safe_memory_efficient_attention (xformers disabled).")
except Exception as e:
    print("[Patch] Could not patch xformers, continuing without:", e)
# ================================================================

def read_prompts(path: str) -> List[str]:
    try:
        return load_prompts(path)
    except:
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        return lines


@torch.no_grad()
def build_weak_pairs(
    model,
    prompts: List[str],
    outdir: str,
    lambda_sem: float = 0.5,
    t_mode: str = "high",
    t_fixed: int = None,
    batch_size: int = 10,
    height: int = 512,
    width: int = 512,
    frames: int = -1,
):
    os.makedirs(outdir, exist_ok=True)

    assert (height % 16 == 0) and (width % 16 == 0)
    h, w = height // 8, width // 8
    T = model.temporal_length if frames < 0 else frames
    C = model.channels
    
    device = next(model.parameters()).device
    num_timesteps = getattr(model, "num_timesteps", 1000)
    
    print(f"Latent shape: [{C}, {T}, {h}, {w}]")

    def choose_t(bs: int):
        if t_mode == "high":
            return torch.full((bs,), int(0.9 * num_timesteps), dtype=torch.long, device=device)
        elif t_mode == "uniform":
            return torch.randint(int(0.5*num_timesteps), num_timesteps, (bs,), device=device)
        elif t_mode == "fixed":
            assert t_fixed is not None
            return torch.full((bs,), int(t_fixed), dtype=torch.long, device=device)

    n = len(prompts)
    n_rounds = (n + batch_size - 1) // batch_size
    
    print(f"\nGenerating {n} pairs in {n_rounds} batches...")
    
    for idx in range(n_rounds):
        idx_s = idx * batch_size
        idx_e = min(idx_s + batch_size, n)
        batch_prompts = prompts[idx_s:idx_e]
        bs = len(batch_prompts)
        
        print(f"  Batch {idx+1}/{n_rounds}: prompts {idx_s}-{idx_e-1}")

        # Noise sampling
        noise_shape = [bs, C, T, h, w]
        x_T = torch.randn(noise_shape, device=device)
        t = choose_t(bs)

        # ===== Optimization =====
        # Calculate conditional eps
        text_emb = model.get_learned_conditioning(batch_prompts)
        eps_cond = model.apply_model(x_T, t, text_emb).cpu()
        del text_emb  # Release GPU memory
        
        # Calculate unconditional eps
        empty_emb = model.get_learned_conditioning([""] * bs)
        eps_uncond = model.apply_model(x_T, t, empty_emb).cpu()
        del empty_emb  
        
        torch.cuda.empty_cache()
        
        # The difference
        d = eps_cond - eps_uncond
        d = d.to(device)  
        
        x_T_target = x_T + lambda_sem * d

        # Save
        for i in range(bs):
            file_idx = idx_s + i
            torch.save(
                {
                    "x_T": x_T[i].half().cpu(),
                    "x_T_target": x_T_target[i].half().cpu(),
                    "prompt": batch_prompts[i],
                    "meta": {
                        "t": int(t[i].item()),
                        "lambda_sem": float(lambda_sem),
                        "latent_shape": [C, T, h, w],
                    },
                },
                os.path.join(outdir, f"{file_idx:06d}.pt"),
            )
        
        print(f"    ✓ Saved {bs} pairs")

    print(f"\n✓ All {n} pairs saved to {outdir}/")


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20230211)
    parser.add_argument("--config", type=str, default="configs/inference_t2v_512_v2.0.yaml")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/base_512_v2/model.ckpt")
    parser.add_argument("--prompt_file", type=str, default="Work/prompts/initial.txt")
    parser.add_argument("--outdir", type=str, default="Work/outputs/weak_pairs")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--frames", type=int, default=-1)
    parser.add_argument("--lambda_sem", type=float, default=0.5)
    parser.add_argument("--t_mode", type=str, default="high", choices=["high", "uniform", "fixed"])
    parser.add_argument("--t_fixed", type=int, default=None)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--save_intermediate", action="store_true",
                        help="Save eps_cond and eps_uncond for debugging")
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    seed_everything(args.seed)
    
    print("=" * 60)
    print("NPNet-V Weak Pairs Generation (Two-Stage in One File)")
    print("=" * 60)
    
    print("\n[1/3] Loading model...")
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_config).cuda()
    
    assert os.path.exists(args.ckpt_path)
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()
    print(f"  ✓ Model loaded")
    
    print("\n[2/3] Loading prompts...")
    prompts = read_prompts(args.prompt_file)
    print(f"  ✓ Loaded {len(prompts)} prompts")
    
    print("\n[3/3] Generating weak pairs...")
    build_weak_pairs(
        model=model,
        prompts=prompts,
        outdir=args.outdir,
        lambda_sem=args.lambda_sem,
        t_mode=args.t_mode,
        t_fixed=args.t_fixed,
        batch_size=args.bs,
        height=args.height,
        width=args.width,
        frames=args.frames,
    )
    
    print("\n" + "=" * 60)
    print("✓ Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()