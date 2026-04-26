"""
Importing this module registers all known adapters in the ADAPTERS dict.

To add a new adapter: create a new file here, import it below.
"""
from server.policies.base import (
    ADAPTERS, PolicyAdapter, register, load_adapter,
)

# Import order doesn't matter; each module self-registers via @register.
# We use try/except so a single broken adapter doesn't break the server —
# you'll get a clear "not registered" error if you try to use one whose
# import failed.

def _safe_import(modname: str) -> None:
    try:
        __import__(modname)
    except Exception as e:
        print(f"[policies] WARNING: failed to import {modname}: {e}")

_safe_import("server.policies.act_adapter")
_safe_import("server.policies.pi0_adapter")
_safe_import("server.policies.random_adapter")
_safe_import("server.policies.echo_adapter")

__all__ = ["ADAPTERS", "PolicyAdapter", "register", "load_adapter"]
