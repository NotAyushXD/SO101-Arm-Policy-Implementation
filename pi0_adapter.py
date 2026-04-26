"""
π₀ adapter — STUB.

This file exists so the registry has an entry for "pi0". The actual
implementation is week-2 work on the roadmap, after the ACT sanity check
passes.

When you implement it:

  1. The official LeRobot π₀ code is in lerobot.common.policies.pi0
     (assuming you've installed a version that includes it; older v0.2.x
     pins may not have π₀ — you may need to bump the lerobot pin in
     requirements.txt).

  2. π₀ is a flow-matching VLA. It DOES require a language instruction —
     set requires_language = True.

  3. Image preprocessing differs from ACT. π₀ uses its own processor; do not
     just copy the ACT normalization.

  4. Action chunk length is typically 50 (configurable), but verify against
     the loaded checkpoint's config.

  5. Stateful behavior: π₀ as documented is stateless per-call, so leave
     state=None for now. If you decide to cache embeddings across calls in
     the same episode, that's where the opaque `state` param earns its keep.

For experimentation BEFORE you have π₀ working, set POLICY_TYPE=random or
POLICY_TYPE=echo to exercise the full pipeline without any real model.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from server.policies.base import register


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
            "π₀ adapter is not implemented yet. This is week-2 work per the "
            "research roadmap. For now, set POLICY_TYPE=act, random, or echo."
        )

    def predict_chunk(self, joint_state: np.ndarray,
                      images: dict[str, np.ndarray],
                      task: Optional[str],
                      state: Optional[Any] = None,
                      ) -> tuple[np.ndarray, Optional[Any]]:
        raise NotImplementedError

    def reset(self) -> None:
        pass
