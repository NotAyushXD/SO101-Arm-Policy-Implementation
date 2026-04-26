"""
Policy adapter protocol and registry. Server picks the adapter at startup
based on POLICY_TYPE env var.

To add a new policy: drop a new file here, decorate the class with
@register("name"), add `_safe_import("inference.policies.<name>_adapter")`
to __init__.py.
"""
from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class PolicyAdapter(Protocol):
    name: str
    revision: str
    commit_sha: Optional[str]
    chunk_len: int
    requires_language: bool
    action_dim: int

    def predict_chunk(
        self,
        joint_state: np.ndarray,
        images: dict[str, np.ndarray],
        task: Optional[str],
        state: Optional[Any] = None,
    ) -> tuple[np.ndarray, Optional[Any]]: ...

    def reset(self) -> None: ...


ADAPTERS: dict[str, "AdapterFactory"] = {}


class AdapterFactory(Protocol):
    def __call__(self, *, repo: str, revision: str,
                 hf_token: Optional[str] = None, **kwargs) -> PolicyAdapter: ...


def register(name: str):
    def decorator(factory):
        if name in ADAPTERS:
            raise ValueError(f"Adapter '{name}' already registered.")
        ADAPTERS[name] = factory
        return factory
    return decorator


def load_adapter(policy_type: str, *, repo: str, revision: str = "main",
                 hf_token: Optional[str] = None, **kwargs) -> PolicyAdapter:
    if policy_type not in ADAPTERS:
        raise ValueError(
            f"Unknown POLICY_TYPE='{policy_type}'. "
            f"Registered: {sorted(ADAPTERS.keys())}. "
            f"Did you forget to import the adapter module?"
        )
    return ADAPTERS[policy_type](repo=repo, revision=revision, hf_token=hf_token, **kwargs)
