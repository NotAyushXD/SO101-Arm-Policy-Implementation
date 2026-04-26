"""π₀ adapter — STUB. Implement in week 2."""
from __future__ import annotations
from typing import Any, Optional

import numpy as np
from inference.policies.base import register


@register("pi0")
class Pi0Adapter:
    name = "pi0"
    requires_language = True
    chunk_len = 50
    action_dim = 6

    def __init__(self, *, repo: str, revision: str,
                 hf_token: Optional[str] = None, **_):
        self.revision = revision
        self.commit_sha = None
        raise NotImplementedError(
            "π₀ adapter not implemented yet. Use POLICY_TYPE=act, random, "
            "or echo for now. See LeRobot docs for the π₀ load API in 0.5.2."
        )

    def predict_chunk(self, joint_state, images, task, state=None):
        raise NotImplementedError

    def reset(self) -> None:
        pass
