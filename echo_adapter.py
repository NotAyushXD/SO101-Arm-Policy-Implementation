"""
Echo adapter. Returns the current joint state unchanged for every step in
the chunk.

Why this exists: it's the safest possible "active" policy. In live mode
the arm receives commands equal to its current state — i.e., it shouldn't
move. This lets you exercise the full live-mode code path (action sending,
servo communication, e-stop reflexes) without risking unwanted motion.

Recommended usage during first session:

  1. POLICY_TYPE=random in shadow mode      → pipeline works
  2. POLICY_TYPE=echo   in live mode        → arm doesn't move; comms verified
  3. POLICY_TYPE=act    in shadow mode      → real model, no risk
  4. POLICY_TYPE=act    in live_slow mode   → real model, scaled velocity
  5. POLICY_TYPE=act    in live mode        → full eval

That's a 5-step de-risking ladder. Skip steps at your own risk.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from server.policies.base import register


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
        print(f"[echo] chunk_len={chunk_len} (returns current joint state)")

    def predict_chunk(self, joint_state: np.ndarray,
                      images: dict[str, np.ndarray],
                      task: Optional[str],
                      state: Optional[Any] = None,
                      ) -> tuple[np.ndarray, Optional[Any]]:
        # Same target for every step in the chunk: the current joint state.
        chunk = np.broadcast_to(
            joint_state.astype(np.float32),
            (self.chunk_len, self.action_dim),
        ).copy()
        return chunk, None

    def reset(self) -> None:
        pass
