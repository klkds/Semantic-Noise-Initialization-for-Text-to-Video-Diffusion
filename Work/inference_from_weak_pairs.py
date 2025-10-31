# inference_from_weak_pairs.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse, os, sys, glob
import datetime, time
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm
import torch
from pytorch_lightning import seed_everything

from scripts.evaluation.funcs import load_model_checkpoint, save_videos, batch_ddim_sampling
from utils.utils import instantiate_from_config


def load_weak_pair(pt_path):
    """Load weak pairs"""
    data = torch.load(pt_path, map_location="cpu")
    return data


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=20230211)
    parser.add_argument("--config", type=str, default="configs/inference_t2v_512_v2.0.yaml")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/base_512_v2/model.ckpt")
    parser.add_argument("--weak_pairs_dir", type=str, default="Work/outputs/weak_pairs", 
                        help="Directory containing .pt files")
    parser.add_argument("--savedir", type=str, default="Work/outputs/weak_pair_videos")
    parser.add_argument("--savefps", type=int, default=10)
    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--ddim_eta", type=float, default=1.0)
    parser.add_argument("--unconditional_guidance_scale", type=float, default=7.5)
    parser.add_argument("--n_samples", type=int, default=1)
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    
    seed_everything(args.seed)
    
    print("=" * 60)
    print("Generate Videos from Weak Pairs")
    print("=" * 60)
    
    # ===== 1. Load the model =====
    print("\n[1/3] Loading model...")
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_config).cuda()
    
    assert os.path.exists(args.ckpt_path)
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()
    print(f"  ✓ Model loaded")
    
    # ===== 2. Load the weak pairs =====
    print("\n[2/3] Loading weak pairs...")
    pt_files = sorted(glob.glob(os.path.join(args.weak_pairs_dir, "*.pt")))
    print(f"  ✓ Found {len(pt_files)} weak pairs")
    
    # Output directories
    os.makedirs(args.savedir, exist_ok=True)
    os.makedirs(os.path.join(args.savedir, "x_T"), exist_ok=True)
    os.makedirs(os.path.join(args.savedir, "x_T_target"), exist_ok=True)
    
    # ===== 3. Generating videos =====
    print("\n[3/3] Generating videos...")
    start_time = time.time()
    
    for idx, pt_path in enumerate(tqdm(pt_files, desc="Processing")):
        # Load data
        data = load_weak_pair(pt_path)
        x_T = data["x_T"].float().unsqueeze(0)          # [1, C, T, H, W]
        x_T_target = data["x_T_target"].float().unsqueeze(0)
        prompt = data["prompt"]
        meta = data["meta"]
        
        batch_size = 1
        channels, frames, h, w = x_T.shape[1:]
        noise_shape = [batch_size, channels, frames, h, w]
        
        text_emb = model.get_learned_conditioning([prompt])
        cond = {"c_crossattn": [text_emb]}
        
        # ===== Video Generation, reverse process =====
        samples_xT = batch_ddim_sampling(
            model, 
            cond, 
            noise_shape, 
            n_samples=args.n_samples,
            ddim_steps=args.ddim_steps, 
            ddim_eta=args.ddim_eta,
            cfg_scale=args.unconditional_guidance_scale,
            x_T=x_T, 
        )
        
        samples_xT_target = batch_ddim_sampling(
            model, 
            cond, 
            noise_shape, 
            n_samples=args.n_samples,
            ddim_steps=args.ddim_steps, 
            ddim_eta=args.ddim_eta,
            cfg_scale=args.unconditional_guidance_scale,
            x_T=x_T_target, 
        )
        
        filename = f"{idx:06d}"
        save_videos(samples_xT, os.path.join(args.savedir, "x_T"), 
                    [filename], fps=args.savefps)
        save_videos(samples_xT_target, os.path.join(args.savedir, "x_T_target"), 
                    [filename], fps=args.savefps)
        
        with open(os.path.join(args.savedir, f"{filename}.txt"), "w") as f:
            f.write(f"Prompt: {prompt}\n")
            f.write(f"Metadata: {meta}\n")
    
    elapsed = time.time() - start_time
    print(f"\n✓ Generated {len(pt_files) * 2} videos in {elapsed:.2f}s")
    print(f"  Saved to: {args.savedir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()