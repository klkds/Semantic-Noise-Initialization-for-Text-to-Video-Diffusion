# inference_from_golden_pairs.py
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse, glob
import time
from omegaconf import OmegaConf
from tqdm import tqdm
import torch
from pytorch_lightning import seed_everything

from scripts.evaluation.funcs import load_model_checkpoint, save_videos, batch_ddim_sampling
from utils.utils import instantiate_from_config


# ==== xformers monkeypatch，避免 A800 + 新 CUDA 下 cutlass kernel 崩 ====
try:
    import xformers.ops
    def safe_memory_efficient_attention(q, k, v, attn_bias=None, op=None):
        d = q.shape[-1]
        scale = (d ** -0.5)
        attn = torch.bmm(q * scale, k.transpose(1, 2))
        attn = torch.softmax(attn, dim=-1)
        out = torch.bmm(attn, v)
        return out
    xformers.ops.memory_efficient_attention = safe_memory_efficient_attention
    print("[Patch] Using safe_memory_efficient_attention (xformers disabled).")
except Exception as e:
    print("[Patch] Could not patch xformers, continuing without:", e)
# ==================================================================


def load_pair(pt_path):
    """Load .pt file (weak or golden pairs) into CPU tensors + metadata dict"""
    data = torch.load(pt_path, map_location="cpu")
    return data


def get_parser():
    parser = argparse.ArgumentParser(description="Generate videos from weak/golden noise pairs")
    
    parser.add_argument("--seed", type=int, default=20230211)
    parser.add_argument("--config", type=str, default="configs/inference_t2v_512_v2.0.yaml")
    parser.add_argument("--ckpt_path", type=str, default="checkpoints/base_512_v2/model.ckpt")
    
    # ===== 关键修改：统一为 pairs_dir =====
    parser.add_argument("--pairs_dir", type=str, default="Work/outputs/weak_pairs",
                        help="Directory containing .pt files (weak_pairs or golden_pairs)")
    parser.add_argument("--savedir", type=str, default="Work/outputs/pair_videos")
    parser.add_argument("--savefps", type=int, default=10)

    parser.add_argument("--ddim_steps", type=int, default=50)
    parser.add_argument("--ddim_eta", type=float, default=1.0)
    parser.add_argument("--unconditional_guidance_scale", type=float, default=7.5)

    parser.add_argument("--n_samples", type=int, default=1)
    parser.add_argument("--runtime_bs", type=int, default=3,
                        help="Batch size for inference")

    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()
    seed_everything(args.seed)

    # ===== 修改：检测是 weak 还是 golden =====
    pair_type = "golden" if "golden" in args.pairs_dir else "weak"
    
    print("=" * 60)
    print(f"Generate Videos from {pair_type.upper()} Pairs (batched)")
    print("=" * 60)

    # ===== 1. Load the model =====
    print("\n[1/3] Loading model...")
    config = OmegaConf.load(args.config)
    model_config = config.pop("model", OmegaConf.create())
    model = instantiate_from_config(model_config).cuda()

    assert os.path.exists(args.ckpt_path), f"ckpt not found: {args.ckpt_path}"
    model = load_model_checkpoint(model, args.ckpt_path)
    model.eval()
    device = next(model.parameters()).device
    print(f"  ✓ Model loaded on {device}")

    # ===== 2. Load the pairs list =====
    print(f"\n[2/3] Loading {pair_type} pairs...")
    pt_files = sorted(glob.glob(os.path.join(args.pairs_dir, "*.pt")))
    total_files = len(pt_files)
    print(f"  ✓ Found {total_files} {pair_type} pairs in {args.pairs_dir}")

    # Output directories
    os.makedirs(args.savedir, exist_ok=True)
    os.makedirs(os.path.join(args.savedir, "x_T"), exist_ok=True)
    os.makedirs(os.path.join(args.savedir, "x_T_target"), exist_ok=True)

    # ===== 3. Batched generation =====
    print("\n[3/3] Generating videos in batches of", args.runtime_bs)
    start_time = time.time()

    runtime_bs = args.runtime_bs
    pbar_all = tqdm(total=total_files, desc="Total progress", position=0)

    for start_idx in range(0, total_files, runtime_bs):
        batch_paths = pt_files[start_idx : start_idx + runtime_bs]
        cur_bs = len(batch_paths)

        batch_desc = f"[Batch {start_idx}-{start_idx+cur_bs-1}] loading data"
        pbar_batch = tqdm(total=4, desc=batch_desc, position=1, leave=False)

        # -------------------------------------------------
        # Step 0. 读这一批的数据
        xT_list = []
        xTt_list = []
        prompts_batch = []
        metas_batch = []

        for p in batch_paths:
            data = load_pair(p)  # ← 改名为 load_pair

            # [C,T,H,W] -> [1,C,T,H,W]
            x_T = data["x_T"].float().unsqueeze(0).to(device)
            x_T_target = data["x_T_target"].float().unsqueeze(0).to(device)

            xT_list.append(x_T)
            xTt_list.append(x_T_target)

            prompts_batch.append(data["prompt"])
            metas_batch.append(data["meta"])

        x_T_batch  = torch.cat(xT_list,  dim=0)    # [cur_bs,C,T,H,W]
        x_Tt_batch = torch.cat(xTt_list, dim=0)    # [cur_bs,C,T,H,W]

        channels, frames, h, w = x_T_batch.shape[1:]
        noise_shape = [cur_bs, channels, frames, h, w]

        # 文本条件
        text_emb = model.get_learned_conditioning(prompts_batch).to(device)
        cond = {"c_crossattn": [text_emb]}

        pbar_batch.update(1)
        pbar_batch.set_description(f"[Batch {start_idx}-{start_idx+cur_bs-1}] sampling x_T")

        # -------------------------------------------------
        # Step 1. baseline: x_T
        samples_xT = batch_ddim_sampling(
            model,
            cond,
            noise_shape,
            n_samples=args.n_samples,
            ddim_steps=args.ddim_steps,
            ddim_eta=args.ddim_eta,
            cfg_scale=args.unconditional_guidance_scale,
            x_T=x_T_batch,
        )

        pbar_batch.update(1)
        pbar_batch.set_description(f"[Batch {start_idx}-{start_idx+cur_bs-1}] sampling x_T_target")

        # -------------------------------------------------
        # Step 2. enhanced: x_T_target
        samples_xT_target = batch_ddim_sampling(
            model,
            cond,
            noise_shape,
            n_samples=args.n_samples,
            ddim_steps=args.ddim_steps,
            ddim_eta=args.ddim_eta,
            cfg_scale=args.unconditional_guidance_scale,
            x_T=x_Tt_batch,
        )

        pbar_batch.update(1)
        pbar_batch.set_description(f"[Batch {start_idx}-{start_idx+cur_bs-1}] saving")

        # -------------------------------------------------
        # Step 3. 保存结果
        filenames = [f"{start_idx + j:06d}" for j in range(cur_bs)]

        save_videos(
            samples_xT,
            os.path.join(args.savedir, "x_T"),
            filenames,
            fps=args.savefps
        )
        save_videos(
            samples_xT_target,
            os.path.join(args.savedir, "x_T_target"),
            filenames,
            fps=args.savefps
        )

        for fname, prompt, meta in zip(filenames, prompts_batch, metas_batch):
            with open(os.path.join(args.savedir, f"{fname}.txt"), "w") as f:
                f.write(f"Prompt: {prompt}\n")
                f.write(f"Metadata: {meta}\n")

        pbar_batch.update(1)
        pbar_batch.close()
        pbar_all.update(cur_bs)

    pbar_all.close()

    elapsed = time.time() - start_time
    print(f"\n✓ Generated {total_files * 2} videos in {elapsed:.2f}s")
    print(f"  Saved to: {args.savedir}/")
    print("=" * 60)


if __name__ == "__main__":
    main()