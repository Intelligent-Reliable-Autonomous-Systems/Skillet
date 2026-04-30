from skillet.policy import OpenVlaPolicy
from PIL import Image
import torch
import numpy as np
import time

# Open image
img = Image.open("data/llm_debug_images/atp6/1776387497806_replan_pick_block_attempt1.jpg").convert("RGB")

# Option 1: via numpy (most common)
img_tens = torch.from_numpy(np.array(img)).permute(2, 0, 1)  # (H, W, 3) uint8
print(img_tens.shape)
policy = OpenVlaPolicy(use_server=False)

st = time.perf_counter()
action = policy.get_action({"rgb": img_tens})
print(f"[WARN] full loop overran by {(st - time.perf_counter()) * 1000:.1f}ms")
print(action)
