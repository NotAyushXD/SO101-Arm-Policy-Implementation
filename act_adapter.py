"""
ACT adapter. Loads an ACT checkpoint from HF Hub and exposes the standard
PolicyAdapter interface.

ACT is stateless from the server's perspective: every call computes a fresh
50-step action chunk from the current observation. (Internally LeRobot's
ACTPolicy maintains a chunk buffer used by select_action(), but we bypass
that by calling the model to get the full chunk directly.)
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch

from server.policies.base import register


@register("act")
class ACTAdapter:
    name = "act"
    requires_language = False     # ACT ignores task strings

    def __init__(self, *, repo: str, revision: str,
                 hf_token: Optional[str] = None, **_):
        # Lazy import so the protocol module loads even when LeRobot isn't
        # installed (useful for the random/echo adapters during dev).
        from huggingface_hub import HfApi
        from lerobot.common.policies.act.modeling_act import ACTPolicy

        api = HfApi(token=hf_token)
        info = api.model_info(repo, revision=revision)

        print(f"[act] Loading {repo}@{revision} (sha={info.sha[:7]})")
        self._policy = ACTPolicy.from_pretrained(repo, revision=revision)
        self._policy.eval()

        self.revision = revision
        self.commit_sha = info.sha

        # Pick the best available device.
        if torch.cuda.is_available():
            self._device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self._device = torch.device("mps")
        else:
            self._device = torch.device("cpu")
        self._policy.to(self._device)

        # Introspect chunk_len and action_dim from the loaded model when
        # possible; fall back to ACT defaults.
        self.chunk_len = getattr(self._policy.config, "chunk_size", 50)
        self.action_dim = getattr(self._policy.config, "action_dim", 6)
        print(f"[act] device={self._device} chunk_len={self.chunk_len} "
              f"action_dim={self.action_dim}")

    def predict_chunk(self, joint_state: np.ndarray,
                      images: dict[str, np.ndarray],
                      task: Optional[str],
                      state: Optional[Any] = None,
                      ) -> tuple[np.ndarray, Optional[Any]]:
        # Build the LeRobot observation dict.
        obs_batch = {
            "observation.state": torch.tensor(
                joint_state, dtype=torch.float32, device=self._device
            ).unsqueeze(0),
        }
        for cam_name, rgb in images.items():
            t = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
            obs_batch[f"observation.images.{cam_name}"] = t.unsqueeze(0).to(self._device)

        with torch.inference_mode():
            chunk = self._call_policy(obs_batch)

        if chunk.dim() == 3:
            chunk = chunk[0]   # drop batch dim
        return chunk.detach().cpu().numpy().astype(np.float32), None  # stateless

    def _call_policy(self, batch: dict) -> torch.Tensor:
        """
        Wrapper around ACT's chunk prediction. LeRobot's API has shifted; we
        try the common shapes in order.
        """
        if hasattr(self._policy, "predict_action_chunk"):
            return self._policy.predict_action_chunk(batch)
        # Fallback: forward pass. The exact key/output shape varies by version.
        out = self._policy(batch)
        if isinstance(out, dict) and "action" in out:
            return out["action"]
        if torch.is_tensor(out):
            return out
        raise RuntimeError(
            "Could not figure out how to extract action chunk from this "
            "ACTPolicy version. Inspect lerobot.common.policies.act and "
            "update server/policies/act_adapter.py:_call_policy."
        )

    def reset(self) -> None:
        # ACT keeps an internal chunk buffer in select_action mode. Reset it
        # so we don't leak state across episodes.
        if hasattr(self._policy, "reset"):
            self._policy.reset()
