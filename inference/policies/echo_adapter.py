"""Echo adapter. Returns current joint state — safe live-mode comms test."""
from __future__ import annotations
from typing import Any, Optional

import numpy as np
from inference.policies.base import register


@register("echo")
class EchoAdapter:
    name = "echo"
    requires_language = False

    def __init__(self, *, repo: str = "echo", revision: str = "v0",
                 hf_token: Optional[str] = None,
                 chunk_len: int = 50, action_dim: int = 6, **_):
        self.revision = revision
        self.commit_sha = None
        self.chunk_len = chunk_len
        self.action_dim = action_dim

    def predict_chunk(self, joint_state, images, task, state=None):
        chunk = np.broadcast_to(
            joint_state.astype(np.float32),
            (self.chunk_len, self.action_dim),
        ).copy()
        return chunk, None

    def reset(self) -> None:
        pass
