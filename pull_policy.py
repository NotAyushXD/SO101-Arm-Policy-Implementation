"""
Pull a trained ACT policy from the HF Hub to the local Mac.

Separated from run_policy.py so you can pull once and evaluate many times
without re-downloading.

Usage:
    python scripts/pull_policy.py --repo-id user/so101-act-pick-v0
    python scripts/pull_policy.py --repo-id user/so101-act-pick-v0 --revision step-50000
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import snapshot_download


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download a trained policy from HF Hub.")
    p.add_argument("--repo-id", type=str, default=None,
                   help="HF Hub repo id. Defaults to HF_POLICY_REPO in .env.")
    p.add_argument("--revision", type=str, default="main",
                   help="Branch, tag, or commit (e.g. step-50000).")
    p.add_argument("--out-dir", type=Path, default=Path("./checkpoints"),
                   help="Where to put the checkpoint locally.")
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()

    repo_id = args.repo_id or os.environ.get("HF_POLICY_REPO")
    if not repo_id:
        sys.exit("No repo id. Pass --repo-id or set HF_POLICY_REPO in .env.")

    out_dir = args.out_dir / repo_id.split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {repo_id}@{args.revision} → {out_dir}")
    path = snapshot_download(
        repo_id=repo_id,
        revision=args.revision,
        local_dir=str(out_dir),
        # Skip large ancillary files we don't need for inference.
        ignore_patterns=["*.optimizer*", "*.scheduler*", "events.out.tfevents.*"],
    )
    print(f"Done. Policy at: {path}")

    # Sanity: expected LeRobot files present?
    expected = ["config.json", "model.safetensors"]
    missing = [f for f in expected if not (Path(path) / f).exists()]
    if missing:
        print(f"WARNING: expected files missing: {missing}")
        print("The checkpoint may still be usable; LeRobot's file names have "
              "changed across versions. Try running it and see.")


if __name__ == "__main__":
    main()
