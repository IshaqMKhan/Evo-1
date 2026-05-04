# visualize_dataloader.py
# shows exactly what the model sees during training - one real minibatch
# run from Evo_1/ directory:
# python visualize_dataloader.py --dataset_config_path dataset/config.yaml

import os
import sys
import yaml
import argparse
import random
import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath("."))
from lerobot_dataset_pretrain_mp import LeRobotDataset
from torch.utils.data import DataLoader


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def tensor_to_displayable(img_tensor):
    img = img_tensor.float()
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = img.clamp(0, 1)
    return img.permute(1, 2, 0).numpy()


def custom_collate_fn(batch):
    prompts        = [item["prompt"]        for item in batch]
    images         = [item["images"]        for item in batch]
    states         = torch.stack([item["state"]         for item in batch], dim=0)
    actions        = torch.stack([item["action"]        for item in batch], dim=0)
    action_mask    = torch.stack([item["action_mask"]   for item in batch], dim=0)
    image_masks    = torch.stack([item["image_mask"]    for item in batch], dim=0)
    state_mask     = torch.stack([item["state_mask"]    for item in batch], dim=0)
    embodiment_ids = torch.stack([item["embodiment_id"] for item in batch], dim=0)
    return {
        "prompts":        prompts,
        "images":         images,
        "states":         states,
        "actions":        actions,
        "action_mask":    action_mask,
        "state_mask":     state_mask,
        "image_masks":    image_masks,
        "embodiment_ids": embodiment_ids,
    }


def visualize_minibatch(batch, dataset, save_dir, batch_idx=0):
    os.makedirs(save_dir, exist_ok=True)

    prompts        = batch["prompts"]         # list of 16 strings
    images_batch   = batch["images"]          # list of 16 tensors
    states         = batch["states"]          # (16, max_state_dim)
    actions        = batch["actions"]         # (16, horizon, max_action_dim)
    image_masks    = batch["image_masks"]     # (16, max_views)
    state_mask     = batch["state_mask"]      # (16, max_state_dim)
    action_mask    = batch["action_mask"]     # (16, horizon, max_action_dim)
    embodiment_ids = batch["embodiment_ids"]  # (16,)

    batch_size = len(prompts)

    # ── print full batch summary to terminal ────────────────────────────────
    print(f"\n{'='*70}")
    print(f"MINIBATCH {batch_idx}  —  {batch_size} samples")
    print(f"{'='*70}")
    print(f"states         shape : {tuple(states.shape)}")
    print(f"actions        shape : {tuple(actions.shape)}")
    print(f"image_masks    shape : {tuple(image_masks.shape)}")
    print(f"state_mask     shape : {tuple(state_mask.shape)}")
    print(f"embodiment_ids shape : {tuple(embodiment_ids.shape)}")
    print(f"images               : list of {batch_size} tensors, "
          f"each {tuple(images_batch[0].shape)}")

    print(f"\n{'─'*70}")
    print(f"{'idx':>4}  {'episode/frame':<35}  {'prompt':<35}  {'state (real dims)'}")
    print(f"{'─'*70}")

    for i in range(batch_size):
        # get the pkl path so we know exactly which episode and frame this is
        # NOTE: the dataloader does not expose the index directly,
        # so we show what we can from the batch itself
        n_real_views   = int(image_masks[i].sum().item())
        n_real_state   = int(state_mask[i].sum().item())
        n_real_action  = int(action_mask[i, 0].sum().item())
        state_vals     = states[i].float()[:n_real_state].tolist()
        emb_id         = embodiment_ids[i].item()
        prompt_short   = prompts[i][:40]

        print(f"  {i:>2}  emb_id={emb_id}  views={n_real_views}  "
              f"state_dims={n_real_state}  action_dims={n_real_action}")
        print(f"      prompt : '{prompt_short}'")
        print(f"      state  : {[round(x,4) for x in state_vals]}")
        print(f"      action[0]: "
              f"{[round(x,4) for x in actions[i,0,:n_real_action].float().tolist()]}")
        print()

    # ── one big figure: 16 rows × N_views columns ───────────────────────────
    # each row = one sample in the minibatch
    max_views   = images_batch[0].shape[0]
    n_real_cols = max(int(image_masks[i].sum().item()) for i in range(batch_size))
    n_real_cols = max(n_real_cols, 1)

    fig, axes = plt.subplots(
        batch_size, n_real_cols,
        figsize=(5 * n_real_cols, 3 * batch_size),
        squeeze=False
    )

    fig.suptitle(
        f"Minibatch {batch_idx}  —  {batch_size} samples  "
        f"(each row = one training sample)\n"
        f"states: {tuple(states.shape)}   "
        f"actions: {tuple(actions.shape)}   "
        f"images: list of {batch_size}×{tuple(images_batch[0].shape)}",
        fontsize=10, y=1.001
    )

    for i in range(batch_size):
        images     = images_batch[i]        # (max_views, 3, H, W)
        image_mask = image_masks[i]         # (max_views,)
        n_views    = int(image_mask.sum().item())
        n_state    = int(state_mask[i].sum().item())
        n_act      = int(action_mask[i, 0].sum().item())
        state_vals = states[i].float()[:n_state].tolist()
        act_vals   = actions[i, 0, :n_act].float().tolist()

        for v in range(n_real_cols):
            ax = axes[i][v]
            if v < n_views:
                img_np = tensor_to_displayable(images[v])
                ax.imshow(img_np)
                ax.set_title(
                    f"sample {i}  view {v}\n"
                    f"prompt: '{prompts[i][:35]}'\n"
                    f"state: {[round(x,3) for x in state_vals]}\n"
                    f"action[0]: {[round(x,3) for x in act_vals]}",
                    fontsize=6
                )
            else:
                ax.set_visible(False)
            ax.axis("off")

    plt.tight_layout()
    save_path = os.path.join(save_dir, f"minibatch_{batch_idx:03d}.png")
    plt.savefig(save_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_config_path", type=str, required=True)
    parser.add_argument("--batch_size",          type=int, default=16)
    parser.add_argument("--n_batches",           type=int, default=1,
                        help="how many minibatches to visualize")
    parser.add_argument("--horizon",             type=int, default=50)
    parser.add_argument("--save_dir",            type=str, default="./viz_dataloader")
    parser.add_argument("--num_workers",         type=int, default=4)
    args = parser.parse_args()

    # load dataset
    with open(args.dataset_config_path, "r") as f:
        dataset_config = yaml.safe_load(f)

    print("Loading dataset...")
    dataset = LeRobotDataset(
        config=dataset_config,
        image_size=448,
        action_horizon=args.horizon,
        use_augmentation=False
    )
    print(f"Dataset loaded: {len(dataset)} total samples")
    print(f"Will visualize {args.n_batches} minibatch(es) of size {args.batch_size}")

    # build dataloader exactly like training does
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,               # same as training
        num_workers=args.num_workers,
        pin_memory=False,
        drop_last=True,
        collate_fn=custom_collate_fn
    )

    # grab n_batches minibatches and visualize each
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= args.n_batches:
            break
        visualize_minibatch(batch, dataset, args.save_dir, batch_idx)

    print(f"\nAll done. Images saved to {args.save_dir}/")