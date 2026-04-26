"""Auto-imports all adapter modules, gracefully skipping ones that fail."""
from inference.policies.base import (  # noqa: F401
    ADAPTERS, PolicyAdapter, register, load_adapter,
)


def _safe_import(modname: str) -> None:
    try:
        __import__(modname)
    except Exception as e:
        print(f"[policies] WARNING: failed to import {modname}: {e}")


_safe_import("inference.policies.act_adapter")
_safe_import("inference.policies.pi0_adapter")
_safe_import("inference.policies.random_adapter")
_safe_import("inference.policies.echo_adapter")
