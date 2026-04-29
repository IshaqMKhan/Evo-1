# visualize_samples.py
# Run from Evo_1/ directory:
#   python visualize_samples.py --dataset_config_path dataset/config.yaml --n_samples 10

import sys, os, argparse, yaml, random
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

sys.path.append(os.path.abspath("."))
from dataset.lerobot_dataset_pretrain_mp import LeRobotDataset

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

def tensor_to_displayable(img_tensor):
    """Convert normalised (3,H,W) bfloat16 tensor to numpy (H,W,3) for imshow."""
    img = img_tensor.float()
    img = img * IMAGENET_STD + IMAGENET_MEAN   # undo ImageNet normalisation
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()        # (H,W,3)

def visualize_samples(dataset, indices, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    for sample_idx in indices:
        sample     = dataset[sample_idx]
        print(f"pkl file : {dataset.data[sample_idx]}")
        images     = sample["images"]       # (max_views, 3, H, W)  original single-frame
        image_mask = sample["image_mask"]   # (max_views,)
        state      = sample["state"]        # (max_state_dim,)
        state_mask = sample["state_mask"]   # (max_state_dim,)
        action     = sample["action"]       # (horizon, max_action_dim)
        action_mask= sample["action_mask"]  # (horizon, max_action_dim)
        prompt     = sample["prompt"]
        emb_id     = sample["embodiment_id"].item()

        # how many real camera views
        n_views = int(image_mask.sum().item())

        # how many valid state dims
        n_state_dims = int(state_mask[0].sum().item()) if state_mask.dim() > 1 \
                       else int(state_mask.sum().item())

        # how many valid action dims
        n_action_dims = int(action_mask[0].sum().item())

        # ── print to terminal ──────────────────────────────────────
        print(f"\n{'='*60}")
        print(f"Sample index : {sample_idx}")
        print(f"Prompt       : '{prompt}'")
        print(f"Embodiment id: {emb_id}")
        print(f"images shape : {tuple(images.shape)}  "
              f"(max_views={images.shape[0]}, real views={n_views})")
        print(f"state shape  : {tuple(state.shape)}  "
              f"(max_state_dim={state.shape[-1]}, real dims={n_state_dims})")
        print(f"action shape : {tuple(action.shape)}  "
              f"(horizon={action.shape[0]}, real action_dim={n_action_dims})")
        print(f"State values (real dims only):")

        state_f32 = state.float()
        if state_f32.dim() == 1:
            print(f"  t=0 : {state_f32[:n_state_dims].tolist()}")
        else:
            for t in range(state_f32.shape[0]):
                print(f"  t={t} : {state_f32[t, :n_state_dims].tolist()}")

        print(f"Action values (first 3 steps, real dims only):")
        action_f32 = action.float()
        for t in range(min(3, action_f32.shape[0])):
            print(f"  step {t} : {action_f32[t, :n_action_dims].tolist()}")

        # ── matplotlib figure ──────────────────────────────────────
        fig, axes = plt.subplots(
            1, max(n_views, 1),
            figsize=(5 * max(n_views, 1), 5),
            squeeze=False
        )
        fig.suptitle(
            f"Sample {sample_idx}  |  prompt: '{prompt[:70]}'\n"
            f"state (real {n_state_dims} dims): "
            f"{state.float().view(-1)[:n_state_dims].tolist()}\n"
            f"images: {tuple(images.shape)}   "
            f"state: {tuple(state.shape)}   "
            f"action: {tuple(action.shape)}",
            fontsize=9, y=1.02
        )

        for v in range(n_views):
            img_np = tensor_to_displayable(images[v])   # real view
            axes[0][v].imshow(img_np)
            axes[0][v].set_title(f"view {v}", fontsize=9)
            axes[0][v].axis("off")
        for v in range(n_views, axes.shape[1]):
            axes[0][v].axis("off")

        plt.tight_layout()
        save_path = os.path.join(save_dir, f"sample_{sample_idx:05d}.png")
        plt.savefig(save_path, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved → {save_path}")

    print(f"\nAll done. Images saved to: {save_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_config_path", type=str, required=True,
                        help="path to dataset/config.yaml")
    parser.add_argument("--n_samples", type=int, default=10,
                        help="how many samples to visualize")
    parser.add_argument("--indices", type=int, nargs="+", default=None,
                        help="specific indices to visualize, e.g. --indices 0 1 5 100 "
                             "if not set, uses 0,1,2,...,n_samples-1")
    parser.add_argument("--random", action="store_true",
                        help="pick indices randomly instead of sequentially")
    parser.add_argument("--save_dir", type=str, default="./visualization_output")
    parser.add_argument("--horizon", type=int, default=50)
    args = parser.parse_args()

    # load dataset config
    with open(args.dataset_config_path, "r") as f:
        dataset_config = yaml.safe_load(f)

    print("Loading dataset...")
    dataset = LeRobotDataset(
        config=dataset_config,
        image_size=448,
        action_horizon=args.horizon,
        use_augmentation=False    # no augmentation so you see the real frames
    )
    print(f"Dataset loaded: {len(dataset)} total samples")

    # decide which indices to visualize
    if args.indices is not None:
        indices = args.indices
    elif args.random:
        indices = random.sample(range(len(dataset)), args.n_samples)
        print(f"Random indices picked: {indices}")
    else:
        indices = list(range(args.n_samples))

    visualize_samples(dataset, indices, args.save_dir)