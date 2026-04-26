"""
Record teleop demos on the Mac M2 and push to the Hugging Face Hub.

This is a thin wrapper around LeRobot's `record` pipeline. It exists because:
  1. LeRobot's CLI args change between releases; centralising them here means
     one place to fix when versions drift.
  2. We want the Hub push to happen atomically at the end (not per-episode),
     so that a failed session doesn't leave half-datasets on the Hub.
  3. We want a pre-flight check that cameras + leader arm are responsive —
     worst thing is realising at episode 37 that a camera was black.

Usage:
    python scripts/record_demos.py --num-episodes 50 --dataset-name so101-pick-v0
    python scripts/record_demos.py --num-episodes 2  --dataset-name so101-debug --no-push
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Record SO-101 teleop demos.")
    p.add_argument("--num-episodes", type=int, required=True,
                   help="Number of episodes to record in this session.")
    p.add_argument("--dataset-name", type=str, required=True,
                   help="Short name; becomes <HF_USERNAME>/<dataset-name> on the Hub.")
    p.add_argument("--episode-time-s", type=int, default=30,
                   help="Max seconds per episode. Cut yourself off hard so "
                        "you don't end up with 5-minute demos.")
    p.add_argument("--reset-time-s", type=int, default=10,
                   help="Seconds between episodes to reset the scene.")
    p.add_argument("--fps", type=int, default=30,
                   help="Control + camera fps. 30 is the LeRobot default.")
    p.add_argument("--task", type=str,
                   default="Pick up the object and place it in the target zone.",
                   help="Language instruction stored with every frame. Keep it short.")
    p.add_argument("--no-push", action="store_true",
                   help="Skip the final Hub push (useful for debug sessions).")
    p.add_argument("--resume", action="store_true",
                   help="Append to an existing local dataset instead of starting fresh.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Pre-flight
# ---------------------------------------------------------------------------

def preflight(port: str, cam_wrist: int, cam_front: int) -> None:
    """Fail fast before the human starts teleoperating."""
    problems: list[str] = []

    if not Path(port).exists():
        problems.append(f"Serial port {port} does not exist. Check SO101_PORT in .env.")

    try:
        import cv2  # noqa: WPS433 — intentional lazy import
        for name, idx in (("wrist", cam_wrist), ("front", cam_front)):
            cap = cv2.VideoCapture(idx)
            if not cap.isOpened():
                problems.append(f"Camera '{name}' at index {idx} did not open.")
            else:
                ok, _ = cap.read()
                if not ok:
                    problems.append(f"Camera '{name}' opened but returned no frame.")
            cap.release()
    except ImportError:
        problems.append("opencv-python not installed — `pip install opencv-python`.")

    if os.environ.get("HF_TOKEN") is None:
        problems.append("HF_TOKEN not set. Copy .env.example to .env and fill it in.")

    if problems:
        print("Pre-flight failed:", file=sys.stderr)
        for p in problems:
            print(f"  - {p}", file=sys.stderr)
        sys.exit(1)

    print("Pre-flight OK: port, cameras, and HF_TOKEN all look good.")


# ---------------------------------------------------------------------------
# Record
# ---------------------------------------------------------------------------

def build_lerobot_cmd(args: argparse.Namespace, repo_id: str, port: str,
                      cam_wrist: int, cam_front: int, push: bool) -> list[str]:
    """
    Assemble the LeRobot record CLI invocation.

    NOTE: CLI arg names below are for LeRobot v0.2.x. If LeRobot has been
    upgraded and this fails, run `lerobot-record --help` and adjust.
    """
    cmd = [
        "lerobot-record",
        f"--robot.type=so101_follower",
        f"--robot.port={port}",
        f"--teleop.type=so101_leader",
        # Leader port: LeRobot typically auto-discovers a second SO-101 on a
        # different usbmodem. If it can't find it, set --teleop.port=...
        f"--dataset.repo_id={repo_id}",
        f"--dataset.num_episodes={args.num_episodes}",
        f"--dataset.episode_time_s={args.episode_time_s}",
        f"--dataset.reset_time_s={args.reset_time_s}",
        f"--dataset.fps={args.fps}",
        f"--dataset.single_task={args.task!r}",
        # Cameras: two OpenCV indices, named 'wrist' and 'front'
        f"--robot.cameras={{wrist:{{type: opencv, index_or_path: {cam_wrist}, fps: {args.fps}, width: 640, height: 480}}, front:{{type: opencv, index_or_path: {cam_front}, fps: {args.fps}, width: 640, height: 480}}}}",
        f"--dataset.push_to_hub={'true' if push else 'false'}",
    ]
    if args.resume:
        cmd.append("--resume=true")
    return cmd


def main() -> None:
    load_dotenv()
    args = parse_args()

    hf_user = os.environ.get("HF_USERNAME")
    if not hf_user:
        sys.exit("HF_USERNAME not set in .env.")
    repo_id = f"{hf_user}/{args.dataset_name}"

    port = os.environ.get("SO101_PORT", "")
    cam_wrist = int(os.environ.get("CAM_WRIST", 0))
    cam_front = int(os.environ.get("CAM_FRONT", 1))

    preflight(port, cam_wrist, cam_front)

    push = not args.no_push
    cmd = build_lerobot_cmd(args, repo_id, port, cam_wrist, cam_front, push=push)

    print("\nLaunching LeRobot recorder. Use the leader arm to teleop the follower.")
    print("Press the 'next episode' key (usually RIGHT_ARROW) when an episode is done,")
    print("and 'rerecord' (LEFT_ARROW) to redo the last one if something went wrong.\n")
    print("Command:\n  " + " \\\n    ".join(cmd) + "\n")

    rc = subprocess.call(cmd)
    if rc != 0:
        sys.exit(f"lerobot-record exited with code {rc}")

    print(f"\nDone. Dataset at: https://huggingface.co/datasets/{repo_id}")
    print("Recommended: run `datasets-cli scan-cache` to confirm local copy.")


if __name__ == "__main__":
    main()
