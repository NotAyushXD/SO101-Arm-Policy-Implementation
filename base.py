"""
Policy adapter protocol + registry.

The protocol is deliberately small. Anything that varies unpredictably
between policies (LoRA hooks, tokenizers, KV-cache management, training
loops) lives inside the adapter, not in the protocol.

Adding a new policy:

    1. Create server/policies/<name>_adapter.py.
    2. Subclass PolicyAdapter (or implement the Protocol).
    3. Decorate with @register("<name>").
    4. Set POLICY_TYPE=<name> on the server.

The server treats every adapter identically through this surface. The web
UI, the /infer endpoint, /policy/reload, observation logging, and W&B
telemetry all work without code changes.

Stateful adapters (autoregressive VLAs, KV-cache) use the opaque `state`
return value: the server passes whatever was returned last call back in on
the next call. The protocol doesn't need to know what's inside it.
"""
from __future__ import annotations

from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class PolicyAdapter(Protocol):
    """
    The contract every policy must satisfy. Implement these as instance
    attributes (not methods) so the server can read them without invoking
    anything: name, revision, commit_sha, chunk_len, requires_language,
    action_dim.
    """
    name: str                          # "act", "pi0", "openvla", "random", ...
    revision: str                      # HF revision string this was loaded at
    commit_sha: Optional[str]          # resolved commit SHA, for telemetry
    chunk_len: int                     # action chunk size: 50 for ACT, 1 for OpenVLA
    requires_language: bool            # True if `task` must be non-None
    action_dim: int                    # 6 for SO-101

    def predict_chunk(
        self,
        joint_state: np.ndarray,           # shape [action_dim]
        images: dict[str, np.ndarray],     # name -> HxWx3 uint8 RGB
        task: Optional[str],               # language instruction
        state: Optional[Any] = None,       # opaque per-episode state from last call
    ) -> tuple[np.ndarray, Optional[Any]]:
        """
        Return (action_chunk, new_state).

        action_chunk: shape [chunk_len, action_dim], float32.
        new_state: any opaque object the adapter wants the server to hand
                   back on the next call (KV-cache, action history, etc.).
                   Return None if the adapter is stateless.
        """
        ...

    def reset(self) -> None:
        """Clear any per-episode state. Called at episode boundaries."""
        ...


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Maps name -> factory callable. Factory takes (repo, revision, hf_token, **kwargs)
# and returns a PolicyAdapter instance. Using a callable rather than a class
# directly so adapters can have async-loaded dependencies, custom factory
# logic, etc.
ADAPTERS: dict[str, "AdapterFactory"] = {}


class AdapterFactory(Protocol):
    def __call__(self, *, repo: str, revision: str,
                 hf_token: Optional[str] = None, **kwargs) -> PolicyAdapter: ...


def register(name: str):
    """Decorator: register an adapter class or factory under a name."""
    def decorator(factory):
        if name in ADAPTERS:
            raise ValueError(f"Adapter '{name}' already registered.")
        ADAPTERS[name] = factory
        return factory
    return decorator


def load_adapter(policy_type: str, *, repo: str, revision: str = "main",
                 hf_token: Optional[str] = None, **kwargs) -> PolicyAdapter:
    """Look up an adapter by name and instantiate it."""
    if policy_type not in ADAPTERS:
        available = sorted(ADAPTERS.keys())
        raise ValueError(
            f"Unknown POLICY_TYPE='{policy_type}'. "
            f"Registered: {available}. "
            f"Did you forget to import the adapter module?"
        )
    factory = ADAPTERS[policy_type]
    adapter = factory(repo=repo, revision=revision, hf_token=hf_token, **kwargs)

    # Sanity check: did the adapter actually fill in the required attributes?
    for attr in ("name", "revision", "chunk_len", "requires_language", "action_dim"):
        if not hasattr(adapter, attr):
            raise RuntimeError(
                f"Adapter '{policy_type}' missing required attribute: {attr}"
            )
    return adapter
