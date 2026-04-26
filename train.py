"""
Unified training entrypoint.

Reads a YAML config and a --platform flag, dispatches to lerobot-train with
the right arguments. Does NOT wrap LeRobot's training code — just picks the
right CLI invocation and runs it as a subprocess.

Why this is a thin script (not a wrapper class):
  - LeRobot's training code is the source of truth. Reimplementing it would
    duplicate work and create a second source of architecture knowledge.
  - The complexity here is config loading and platform dispatch, both of
    which are fundamentally just "build a list of CLI args."

Adding a new architecture:
  1. Create configs/<n>.yaml (use act.yaml as template).
  2. Make sure LeRobot 0.5.2 supports --policy.type=<n>. Check with:
       lerobot-train --help | grep policy.type
  3. Run: python training/train.py --config configs/<n>.yaml --platform mac

Usage:
    # Mac smoke test (1k steps for pipeline validation)
    python training/train.py --config configs/act.yaml --platform mac \\
        --override training.steps=1000

    # Kaggle real training (called from the Kaggle notebook)
    python training/train.py --config configs/act.yaml --platform kaggle
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Config loading with simple inheritance
# ---------------------------------------------------------------------------

def load_config(path: Path) -> dict[str, Any]:
    """
    Load a YAML config, resolving `inherits: <other.yaml>` recursively.
    Later configs override earlier ones (deep merge for dicts).
    """
    with open(path) as f:
        cfg = yaml.safe_load(f)

    if "inherits" in cfg:
        parent_path = path.parent / cfg.pop("inherits")
        parent = load_config(parent_path)
        cfg = deep_merge(parent, cfg)

    return cfg


def deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Override wins on leaves."""
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    """
    Apply CLI-style overrides like 'training.steps=1000' to a config.
    Type inference is naive: tries int, float, bool, then falls back to str.
    """
    for ov in overrides:
        if "=" not in ov:
            sys.exit(f"Bad override (need key=value): {ov}")
        key, value = ov.split("=", 1)
        # Try to parse the value sensibly
        if value.lower() in ("true", "false"):
            parsed: Any = value.lower() == "true"
        else:
            try:
                parsed = int(value)
            except ValueError:
                try:
                    parsed = float(value)
                except ValueError:
                    parsed = value
        # Walk the dotted path
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], dict):
                node[p] = {}
            node = node[p]
        node[parts[-1]] = parsed
    return cfg


# ---------------------------------------------------------------------------
# Platform dispatch — the entire Mac vs Kaggle difference
# ---------------------------------------------------------------------------

def platform_args(platform: str) -> list[str]:
    """
    Platform-specific CLI flags. This is the entire surface where Mac and
    Kaggle diverge in the training command.
    """
    if platform == "mac":
        # MPS = Apple Silicon GPU. AMP (mixed precision) is unreliable on MPS
        # for transformer ops as of torch 2.4, so we disable it.
        return [
            "--policy.device=mps",
            "--policy.use_amp=false",
            "--num_workers=2",   # M2 unified memory; few workers is fine
        ]
    if platform == "kaggle":
        return [
            "--policy.device=cuda",
            "--policy.use_amp=true",
            "--num_workers=4",
        ]
    sys.exit(f"Unknown platform: {platform}")


# ---------------------------------------------------------------------------
# Build the lerobot-train command from the merged config
# ---------------------------------------------------------------------------

def build_command(cfg: dict[str, Any], platform: str,
                  output_dir: Path, resume: bool) -> list[str]:
    """Translate the YAML config into the lerobot-train CLI invocation."""
    cmd = ["lerobot-train"]

    # Dataset
    cmd.append(f"--dataset.repo_id={cfg['dataset']['repo_id']}")

    # Policy
    pol = cfg["policy"]
    cmd.append(f"--policy.type={pol['type']}")
    cmd.append(f"--policy.push_to_hub={str(pol.get('push_to_hub', True)).lower()}")
    if "repo_id" in pol:
        cmd.append(f"--policy.repo_id={pol['repo_id']}")
    # Pass through all other policy.* keys generically. This means new
    # architectures can add their own keys to the YAML and they reach LeRobot
    # with no code changes here.
    for k, v in pol.items():
        if k in {"type", "push_to_hub", "repo_id"}:
            continue
        cmd.append(f"--policy.{k}={_yaml_to_cli(v)}")

    # Optimizer
    opt = cfg.get("optimizer", {})
    for k, v in opt.items():
        cmd.append(f"--optimizer.{k}={_yaml_to_cli(v)}")

    # Training schedule
    tr = cfg["training"]
    cmd.append(f"--steps={tr['steps']}")
    cmd.append(f"--batch_size={tr['batch_size']}")
    cmd.append(f"--save_freq={tr['save_freq']}")
    cmd.append(f"--seed={tr.get('seed', 42)}")

    # W&B
    wb = cfg.get("wandb", {})
    cmd.append(f"--wandb.enable={str(wb.get('enable', True)).lower()}")
    if wb.get("project"):
        cmd.append(f"--wandb.project={wb['project']}")
    entity = wb.get("entity") or os.environ.get("WANDB_ENTITY")
    if entity:
        cmd.append(f"--wandb.entity={entity}")

    # Output + job
    cmd.append(f"--output_dir={output_dir}")
    cmd.append(f"--job_name={cfg['name']}-{platform}-{tr.get('seed', 42)}")

    # Platform-specific bits (device, AMP, num_workers)
    cmd.extend(platform_args(platform))

    # Resume
    if resume:
        cmd.append("--resume=true")

    return cmd


def _yaml_to_cli(v: Any) -> str:
    """Convert a YAML value into a CLI-safe string. Bools lowercased."""
    if isinstance(v, bool):
        return str(v).lower()
    return str(v)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Unified SO-101 trainer.")
    p.add_argument("--config", type=Path, required=True,
                   help="Path to YAML config (e.g. configs/act.yaml).")
    p.add_argument("--platform", choices=["mac", "kaggle"], required=True,
                   help="Where this is running. Sets device, AMP, workers.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Where checkpoints go. Defaults to ./runs/<config-name> "
                        "on Mac, /kaggle/working/run on Kaggle.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from the latest checkpoint in output-dir.")
    p.add_argument("--override", action="append", default=[],
                   help="Override config values (e.g. training.steps=1000). "
                        "Repeatable.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the lerobot-train command but don't execute it.")
    args = p.parse_args()

    # Load config and apply overrides
    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)

    # Decide output dir
    if args.output_dir:
        out = args.output_dir
    elif args.platform == "kaggle":
        out = Path("/kaggle/working/run")
    else:
        out = Path("./runs") / cfg["name"]
    out.mkdir(parents=True, exist_ok=True)

    # Build + show the command
    cmd = build_command(cfg, args.platform, out, args.resume)
    print("=" * 70)
    print(f"Config:   {args.config}")
    print(f"Platform: {args.platform}")
    print(f"Output:   {out}")
    print(f"Resume:   {args.resume}")
    print("=" * 70)
    print("Command:")
    print("  " + " \\\n    ".join(cmd))
    print("=" * 70)

    if args.dry_run:
        print("Dry run — exiting without executing.")
        return

    # Run
    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(f"lerobot-train failed with exit code {rc}")
    print("\nTraining complete.")


if __name__ == "__main__":
    main()
