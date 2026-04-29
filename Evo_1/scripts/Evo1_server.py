import sys
import os
import asyncio
import websockets
import numpy as np
import cv2
import json
import torch
from PIL import Image
from torchvision import transforms
import time

# =========================
# GPU MEMORY LOGGER
# =========================
gpu_log_file = "gpu_memory_log.txt"
inference_count = 0

def log_gpu(tag: str):
    allocated = torch.cuda.memory_allocated() / 1e9
    reserved = torch.cuda.memory_reserved() / 1e9

    msg = f"{time.strftime('%Y-%m-%d %H:%M:%S')} | {tag} | allocated={allocated:.3f}GB | reserved={reserved:.3f}GB\n"

    print(msg.strip())
    with open(gpu_log_file, "a") as f:
        f.write(msg)


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from scripts.Evo1 import EVO1


# =========================
# NORMALIZER
# =========================
class Normalizer:
    def __init__(self, stats_or_path):
        if isinstance(stats_or_path, str):
            with open(stats_or_path, "r") as f:
                stats = json.load(f)
        else:
            stats = stats_or_path

        def pad_to_24(x):
            x = torch.tensor(x, dtype=torch.float32)
            if x.shape[0] < 24:
                pad = torch.zeros(24 - x.shape[0], dtype=torch.float32)
                x = torch.cat([x, pad], dim=0)
            elif x.shape[0] > 24:
                raise ValueError(f"Input length {x.shape[0]} exceeds expected 24")
            return x

        robot_key = list(stats.keys())[0]
        robot_stats = stats[robot_key]

        self.state_min = pad_to_24(robot_stats["observation.state"]["min"])
        self.state_max = pad_to_24(robot_stats["observation.state"]["max"])
        self.action_min = pad_to_24(robot_stats["action"]["min"])
        self.action_max = pad_to_24(robot_stats["action"]["max"])

    def normalize_state(self, state):
        state_min = self.state_min.to(state.device, dtype=state.dtype)
        state_max = self.state_max.to(state.device, dtype=state.dtype)
        return torch.clamp(
            2 * (state - state_min) / (state_max - state_min + 1e-8) - 1,
            -1.0,
            1.0
        )

    def denormalize_action(self, action):
        action_min = self.action_min.to(action.device, dtype=action.dtype)
        action_max = self.action_max.to(action.device, dtype=action.dtype)

        if action.ndim == 1:
            action = action.view(1, -1)

        return (action + 1.0) / 2.0 * (action_max - action_min + 1e-8) + action_min


# =========================
# MODEL LOADING
# =========================
def load_model_and_normalizer(ckpt_dir):

    config = json.load(open(os.path.join(ckpt_dir, "config.json")))
    stats = json.load(open(os.path.join(ckpt_dir, "norm_stats.json")))

    config["finetune_vlm"] = False
    config["finetune_action_head"] = False
    config["num_inference_timesteps"] = 32

    print("[INFO] Building EVO1 model...")
    model = EVO1(config)

    ckpt_path = os.path.join(ckpt_dir, "checkpoint.pt")
    print(f"[INFO] Loading checkpoint: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")

    print("[INFO] Checkpoint keys:", checkpoint.keys())

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        raise ValueError("No valid model weights found in checkpoint")

    model.load_state_dict(state_dict, strict=True)
    model = model.to("cuda").eval()

    log_gpu("MODEL LOADED")

    normalizer = Normalizer(stats)

    print("[INFO] Model + Normalizer ready.")

    return model, normalizer


# =========================
# IMAGE DECODER
# =========================
def decode_image_from_list(img_list):
    img_array = np.array(img_list, dtype=np.uint8)
    img = cv2.resize(img_array, (448, 448))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(img)
    return transforms.ToTensor()(pil).to("cuda")


# =========================
# INFERENCE
# =========================
def infer_from_json_dict(data, model, normalizer):
    global inference_count
    inference_count += 1

    device = "cuda"

    images = [decode_image_from_list(img) for img in data["image"]]
    assert len(images) == 3

    state = torch.tensor(data["state"], dtype=torch.float32, device=device)

    if state.ndim == 1:
        state = state.unsqueeze(0)

    if state.shape[1] < 24:
        state = torch.cat(
            [state, torch.zeros((1, 24 - state.shape[1]), device=device)],
            dim=1
        )

    norm_state = normalizer.normalize_state(state).to(dtype=torch.float32)

    prompt = data["prompt"]
    image_mask = torch.tensor(data["image_mask"], dtype=torch.int32, device=device)
    action_mask = torch.tensor([data["action_mask"]], dtype=torch.int32, device=device)

    print(f"[INFO] Inference {inference_count}")

    log_gpu(f"BEFORE inference {inference_count}")

    with torch.no_grad() and torch.cuda.amp.autocast(dtype=torch.bfloat16):

        action = model.run_inference(
            images=images,
            image_mask=image_mask,
            prompt=prompt,
            state_input=norm_state,
            action_mask=action_mask
        )

    action = action.reshape(1, -1, 24)
    action = normalizer.denormalize_action(action[0])

    log_gpu(f"AFTER inference {inference_count}")

    # milestones
    if inference_count in [1, 2, 10, 20, 30, 50, 100]:
        log_gpu(f"MILESTONE {inference_count}")

    return action.cpu().numpy().tolist()


# =========================
# SERVER HANDLER
# =========================
async def handle_request(websocket, model, normalizer):
    print("Client connected")
    try:
        async for message in websocket:
            json_data = json.loads(message)
            actions = infer_from_json_dict(json_data, model, normalizer)
            await websocket.send(json.dumps(actions))
            print("Sent action chunk")

    except websockets.exceptions.ConnectionClosed:
        print("Client disconnected.")


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    ckpt_dir = "/home/qxc857/Evo-1/Evo_1/checkpoints/stage2/step_final/"
    port = 9000

    print("Loading EVO_1 model...")
    model, normalizer = load_model_and_normalizer(ckpt_dir)

    async def main():
        print(f"EVO_1 server running at ws://0.0.0.0:{port}")

        async with websockets.serve(
            lambda ws: handle_request(ws, model, normalizer),
            "0.0.0.0",
            port,
            max_size=100_000_000,
        ):
            await asyncio.Future()

    asyncio.run(main())