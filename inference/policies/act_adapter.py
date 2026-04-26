"""
ACT adapter. LeRobot 0.5.2 import paths (no .common namespace).
"""
from __future__ import annotations
from typing import Any, Optional

import numpy as np
import torch

from inference.policies.base import register


@register("act")
class ACTAdapter:
    name = "act"
    requires_language = False

    def __init__(self, *, repo: str, revision: str,
                 hf_token: Optional[str] = None, **_):
        from huggingface_hub import HfApi
        from lerobot.policies.act.modeling_act import ACTPolicy   # 0.5.2 path

        api = HfApi(token=hf_token)
        info = api.model_info(repo, revision=revision)

        print(f"[act] Loading {repo}@{revision} (sha={info.sha[:7]})")
        self._policy = ACTPolicy.from_pretrained(repo, revision=revision)
        self._policy.eval()

        self.revision = revision
        self.commit_sha = info.sha

        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")
        self._policy.to(self._device)

        self.chunk_len = getattr(self._policy.config, "chunk_size", 50)
        self.action_dim = getattr(self._policy.config, "action_dim", 6)
        print(f"[act] device={self._device} chunk_len={self.chunk_len} "
              f"action_dim={self.action_dim}")

    def predict_chunk(self, joint_state, images, task, state=None):
        batch = {
            "observation.state": torch.tensor(
                joint_state, dtype=torch.float32, device=self._device
            ).unsqueeze(0),
        }
        for cam_name, rgb in images.items():
            t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            batch[f"observation.images.{cam_name}"] = t.unsqueeze(0).to(self._device)

        with torch.inference_mode():
            if hasattr(self._policy, "predict_action_chunk"):
                chunk = self._policy.predict_action_chunk(batch)
            else:
                chunk = self._policy(batch)
                if isinstance(chunk, dict):
                    chunk = chunk.get("action") or chunk.get("actions")

        if chunk.dim() == 3:
            chunk = chunk[0]
        return chunk.detach().cpu().numpy().astype(np.float32), None

    def reset(self) -> None:
        if hasattr(self._policy, "reset"):
            self._policy.reset()
