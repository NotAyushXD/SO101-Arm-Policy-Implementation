"""
Random-action adapter. Emits Gaussian noise around the current joint state.

Use cases:
  - Debug the network/UI/protocol without loading any real model.
  - Confirm the chunk buffer + smoothing + exec mode logic on the M2 before
    your first real inference call.
  - Sanity-check the observation logging path.

Behavior: each predicted action is the current joint state + a small random
delta. This means in shadow mode the "predicted trajectory" overlay in the UI
will look like brownian motion, which is recognizable and unmistakable —
exactly what you want for "is the pipeline alive."

POLICY_TYPE=random doesn't need a real HF repo, but the protocol requires a
revision string. Pass anything; it just gets echoed back in telemetry.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from server.policies.base import register


@register("random")
class RandomAdapter:
    name = "random"
    requires_language = False

    def __init__(self, *, repo: str = "random", revision: str = "v0",
                 hf_token: Optional[str] = None,
                 chunk_len: int = 50, action_dim: int = 6,
                 noise_std: float = 0.02, **_):
        self.revision = revision
        self.commit_sha = None
        self.chunk_len = chunk_len
        self.action_dim = action_dim
        self._noise_std = noise_std
        self._rng = np.random.default_rng(0)
        print(f"[random] noise_std={noise_std} chunk_len={chunk_len}")

    def predict_chunk(self, joint_state: np.ndarray,
                      images: dict[str, np.ndarray],
                      task: Optional[str],
                      state: Optional[Any] = None,
                      ) -> tuple[np.ndarray, Optional[Any]]:
        # Random walk around current joint state. Each step's target drifts a
        # bit further from the start, which is more interesting than IID noise
        # for visualizing a trajectory.
        chunk = np.empty((self.chunk_len, self.action_dim), dtype=np.float32)
        cur = joint_state.astype(np.float32).copy()
        for i in range(self.chunk_len):
            cur = cur + self._rng.normal(0, self._noise_std, size=self.action_dim).astype(np.float32)
            chunk[i] = cur
        return chunk, None

    def reset(self) -> None:
        # Reseed so each episode looks slightly different but reproducibly.
        self._rng = np.random.default_rng(0)
