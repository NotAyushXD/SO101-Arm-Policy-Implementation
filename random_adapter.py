"""Random walk adapter. No GPU needed. Debug pipeline without a real model."""
from __future__ import annotations
from typing import Any, Optional

import numpy as np
from inference.policies.base import register


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

    def predict_chunk(self, joint_state, images, task, state=None):
        chunk = np.empty((self.chunk_len, self.action_dim), dtype=np.float32)
        cur = joint_state.astype(np.float32).copy()
        for i in range(self.chunk_len):
            cur = cur + self._rng.normal(0, self._noise_std,
                                         size=self.action_dim).astype(np.float32)
            chunk[i] = cur
        return chunk, None

    def reset(self) -> None:
        self._rng = np.random.default_rng(0)
