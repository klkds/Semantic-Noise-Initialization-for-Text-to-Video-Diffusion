# make_golden_pairs.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse, torch, numpy as np
from typing import List
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything

from utils.utils import instantiate_from_config
from scripts.evaluation.funcs import load_model_checkpoint, load_prompts
from lvdm.models.samplers.ddim import DDIMSampler


def read_prompts(path: str) -> List[str]:
    try:
        return load_prompts(path)
    except:
        with open(path, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        return lines


@torch.no_grad()
def ddim_inversion_manual(sampler, x0, cond, t_enc, unconditional_guidance_scale=1.0, unconditional_conditioning=None):
    """
    Manual DDIM inversion: x_0 → x_T
    Iteratively add noise following the reverse diffusion process
    """
    device = x0.device
    batch_size = x0.shape[0]
    is_video = (x0.dim() == 5)
    
    # Use stochastic_encode for a simpler approach
    # This adds noise to x0 to approximate x_T
    t_tensor = torch.full((batch_size,), t_enc - 1, device=device, dtype=torch.long)
    
    # Simple approach: use stochastic_encode
    x_T = sampler.stochastic_encode(x0, t_tensor, use_original_steps=False, noise=None)
    
    return x_T


@torch.no_grad()
def build_golden_pairs(
    model,
    prompts: List[str],
    outdir: str,
    K_steps: int = 10,
    cfg_forward: float = 7.5,
    cfg_backward: float = 1.0,
    batch_size: int = 1,  
    height: int = 320,
    width: int = 512,
    frames: int = -1,
    seed: int = 20230211,
):
    """
    Generate Golden Noise pairs using K-step DDIM forward-backward
    
    Process:
    1. Sample x_T ~ N(0, I)  (random Gaussian noise)
    2. Forward K steps: x_T → x_0 (generate video, with CFG)
    3. Backward: x_0 → x'_T (DDIM inversion via stochastic_encode)
    4. Save (x_T, x'_T) as Golden pair
    """
    torch.manual_seed(seed)
    os.makedirs(outdir, exist_ok=True)

    # Infer latent shape
    assert (height % 16 == 0) and (width % 16 == 0)
    h, w = height // 8, width // 8
    T = model.temporal_length if frames < 0 else frames
    C = model.channels
    
    device = next(model.parameters()).device
    
    print(f"Latent shape: [{C}, {T}, {h}, {w}]")
    print(f"Device: {device}")
    print(f"K_steps: {K_steps}, CFG forward: {cfg_forward}, CFG backward: {cfg_backward}")

    # Initialize DDIM sampler
    ddim_sampler = DDIMSampler(model)
    
    n = len(prompts)
    print(f"\nGenerating {n} Golden pairs...")
    
    for idx in range(0, n, batch_size):
        batch_prompts = prompts[idx:min(idx+batch_size, n)]
        bs = len(batch_prompts)
        
        print(f"  [{idx+1}/{n}] Processing: {batch_prompts[0][:50]}...")

        # ===== Step 1: Sample random Gaussian noise x_T =====
        noise_shape = [bs, C, T, h, w]
        x_T = torch.randn(noise_shape, device=device)

        # ===== Step 2: Forward K steps (x_T → x_0) =====
        # Prepare conditioning
        text_emb = model.get_learned_conditioning(batch_prompts)
        cond = {"c_crossattn": [text_emb]}
        
        # Prepare unconditional conditioning for CFG
        if cfg_forward > 1.0:
            uc_emb = model.get_learned_conditioning([""] * bs)
            uc = {"c_crossattn": [uc_emb]}
        else:
            uc = None
        
        # Forward sampling with CFG
        print(f"    Forward sampling {K_steps} steps...")
        samples, intermediates = ddim_sampler.sample(
            S=K_steps,
            conditioning=cond,
            batch_size=bs,
            shape=[C, T, h, w],
            verbose=False,
            unconditional_guidance_scale=cfg_forward,
            unconditional_conditioning=uc,
            eta=1.0,  # stochastic sampling
            x_T=x_T,
        )
        # samples: [bs, C, T, h, w] (latent x_0)
        
        print(f"    samples shape: {samples.shape}")
        
        # ===== Step 3: Backward (x_0 → x'_T) via stochastic_encode =====
        print(f"    Backward inversion...")
        
        # Use stochastic_encode to add noise back
        # t_enc corresponds to the number of steps we want to invert
        t_enc_tensor = torch.full((bs,), K_steps - 1, device=device, dtype=torch.long)
        
        x_T_golden = ddim_sampler.stochastic_encode(
            samples, 
            t_enc_tensor, 
            use_original_steps=False,
            noise=None  # Will use random noise
        )
        # x_T_golden: [bs, C, T, h, w] (inverted noise)
        
        print(f"    x_T_golden shape: {x_T_golden.shape}")

        # ===== Step 4: Save Golden pairs =====
        for i in range(bs):
            file_idx = idx + i
            
            # Calculate difference for verification
            diff_norm = torch.norm(x_T[i] - x_T_golden[i]).item()
            
            torch.save(
                {
                    "x_T": x_T[i].half().cpu(),           # Original Gaussian noise
                    "x_T_target": x_T_golden[i].half().cpu(),  # Golden Noise
                    "prompt": batch_prompts[i],
                    "meta": {
                        "K_steps": K_steps,
                        "cfg_forward": cfg_forward,
                        "cfg_backward": cfg_backward,
                        "latent_shape": [C, T, h, w],
                        "height": height,
                        "width": width,
                        "method": "golden_noise",
                        "diff_norm": diff_norm,  # For debugging
                    },
                },
                os.path.join(outdir, f"{file_idx:06d}.pt"),
            )
        
        print(f"    ✓ Saved {bs} Golden pairs (diff_norm: {diff_norm:.4f})")
        
        # Clear cache
        torch.cuda.empty_cache()

    print(f"\n✓ All {n} Golden pairs saved to {outdir}/")


def get_parser():
    parser = argparse.ArgumentParser(description="Generate Golden Noise pairs for NPNet-V")
    
    # Model
    parser.add_argument("--seed", type=int, default=20230211)
    parser.add_argument("--config", type=str, default="configs/inference_t2v_512_v2.0.yaml")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/base_512_v2/model.ckpt")
    
    # Data
    parser.add_argument("--prompt_file", type=str, default="Work/prompts/initial.txt")
    parser.add_argument("--outdir", type=str, default="Work/outputs/golden_pairs")
    
    # Generation params
    parser.add_argument("--height", type=int, default=320)
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--frames", type=int, default=-1)
    
    # Golden Noise specific
    parser.add_argument("--K_steps", type=int, default=10,
                        help="Number of DDIM steps for forward-backward (1-50)")
    parser.add_argument("--cfg_forward", type=float, default=7.5,
                        help="CFG scale for forward sampling (x_T → x_0)")
    parser.add_argument("--cfg_backward", type=float, default=1.0,
                        help="CFG scale for backward (not used in stochastic_encode)")
    parser.add_argument("--bs", type=int, default=1,
                        help="Batch size (recommend 1 for Golden Noise)")
    
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    seed_everything(args.seed)
    
    print("=" * 60)
    print("Golden Noise Pairs Generation")
    print("=" * 60)
    
    # Load model
    print("\n[1/3] Loading model...")
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_config).cuda()
    
    assert os.path.exists(args.ckpt_path), f"Checkpoint not found: {args.ckpt_path}"
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()
    print(f"  ✓ Model loaded")
    
    # Load prompts
    print("\n[2/3] Loading prompts...")
    assert os.path.exists(args.prompt_file), f"Prompt file not found: {args.prompt_file}"
    prompts = read_prompts(args.prompt_file)
    print(f"  ✓ Loaded {len(prompts)} prompts")
    
    # Generate Golden pairs
    print("\n[3/3] Generating Golden pairs...")
    build_golden_pairs(
        model=model,
        prompts=prompts,
        outdir=args.outdir,
        K_steps=args.K_steps,
        cfg_forward=args.cfg_forward,
        cfg_backward=args.cfg_backward,
        batch_size=args.bs,
        height=args.height,
        width=args.width,
        frames=args.frames,
        seed=args.seed,
    )
    
    print("\n" + "=" * 60)
    print("✓ Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()