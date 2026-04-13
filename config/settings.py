from dataclasses import dataclass
from pathlib import Path
import torch

@dataclass
class Config:
    # Paths
    parent_dir = str(Path(__file__).resolve().parent.parent)
    model_path = f"{parent_dir}/models/base"
    base_model = "Qwen/Qwen2.5-0.5B-Instruct"

    # Hardware and output
    debug = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cpu" else torch.float16
    volatility_window: int = 10
    volatility_update_interval: int = 10
    max_new_tokens: int = 150

    # Hyperparameters
    sink_tokens: int = 4
    recent_tokens: int = 8
    loss_scale: int = 0.35
    base_threshold: int = 0.9