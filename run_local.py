"""
Local inference client. Loads a policy on the Mac M2 and runs it against the
SO-101 directly. This is the recommended path for ACT — no network round
trip, ~25 ms inference, sub-millisecond control loop overhead.

For larger models (π₀, OpenVLA) where M2 memory or compute is the bottleneck,
use run_remote.py instead.

Usage:
    python client/run_local.py \\
        --policy-path ./checkpoints/so101-act-pick-bowl-v0 \\
        --num-trials 20

Imports are pinned to LeRobot 0.5.2:
    from lerobot.policies.act.modeling_act import ACTPolicy
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from client.chunk_buffer import ChunkBuffer, ChunkBufferConfig  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run a trained policy locally on the M2.")
    p.add_argument("--policy-path", type=Path, required=True,
                   help="Path to a policy directory downloaded by data/pull_policy.py.")
    p.add_argument("--policy-type", default="act",
                   help="Architecture name (act, pi0, ...). Drives which "
                        "loader to use. Defaults to act.")
    p.add_argument("--num-trials", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--smoothing", choices=["on", "off"], default="on")
    p.add_argument("--chunk-execute-len", type=int, default=30,
                   help="Actions per chunk to execute before requesting a new chunk.")
    p.add_argument("--results-dir", type=Path, default=Path("./eval_results"))
    p.add_argument("--task", default="Pick up the object and drop it in the bowl.",
                   help="Language instruction. Ignored by ACT, used by VLAs.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Trial result + helpers
# ---------------------------------------------------------------------------

@dataclass
class TrialResult:
    trial_idx: int
    success: bool
    steps: int
    wall_clock_s: float
    notes: str


def pick_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    print("WARNING: no GPU. Inference will be slow on CPU.")
    return torch.device("cpu")


def prompt_result(trial_idx: int) -> tuple[bool | None, str]:
    while True:
        ans = input(f"\nTrial {trial_idx}: success? [y/n/r=retry/q=quit]: ").strip().lower()
        if ans == "y":
            return True, input("Notes (enter to skip): ").strip()
        if ans == "n":
            return False, input("What happened? ").strip()
        if ans == "r":
            return None, "RETRY"
        if ans == "q":
            sys.exit("Aborted.")
        print("  Please answer y/n/r/q.")


# ---------------------------------------------------------------------------
# Policy loading — keyed by architecture name
# ---------------------------------------------------------------------------

def load_policy(policy_type: str, path: Path, device: torch.device):
    """
    Load a policy from a local checkpoint directory.

    LeRobot 0.5.2 import paths (no `.common`):
      from lerobot.policies.act.modeling_act import ACTPolicy
      from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    """
    if policy_type == "act":
        from lerobot.policies.act.modeling_act import ACTPolicy
        policy = ACTPolicy.from_pretrained(str(path))
    elif policy_type == "pi0":
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy
        policy = PI0Policy.from_pretrained(str(path))
    else:
        # Generic factory fallback. Lets new architectures work without
        # editing this file as long as LeRobot's factory recognizes them.
        from lerobot.policies.factory import make_policy
        policy = make_policy(pretrained=str(path))

    policy.eval()
    policy.to(device)
    return policy


# ---------------------------------------------------------------------------
# Robot — LeRobot 0.5.2 API
# ---------------------------------------------------------------------------

def open_robot(port: str):
    """
    Open the SO-101 follower for inference. LeRobot 0.5.2 uses a Pydantic
    config object; we build a minimal one here.

    NOTE: if your installed LeRobot has drifted from this exact API, the
    fix is usually to find the SO101FollowerConfig class and adjust kwargs.
    """
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
    cfg = SO101FollowerConfig(port=port, id="follower_arm")
    robot = SO101Follower(cfg)
    robot.connect()
    return robot


# ---------------------------------------------------------------------------
# Per-trial control loop
# ---------------------------------------------------------------------------

def run_trial(policy, robot, chunk_buf: ChunkBuffer, *,
              fps: int, max_steps: int, device: torch.device,
              task: str, policy_type: str) -> tuple[int, float]:
    """Run the policy for up to max_steps. Returns (steps_executed, wall_s)."""
    dt = 1.0 / fps
    chunk_buf.reset()
    if hasattr(policy, "reset"):
        policy.reset()
    t_start = time.monotonic()

    for step in range(max_steps):
        loop_start = time.monotonic()

        if chunk_buf.needs_new_chunk():
            # Request a fresh chunk by running the policy on the current obs.
            chunk = _predict_chunk(policy, robot, device, task, policy_type)
            chunk_buf.install_new_chunk(chunk)

        if chunk_buf.is_stale():
            print("  chunk stale, halting.")
            break

        action = chunk_buf.next_action()
        if action is None:
            print("  no action available, halting.")
            break

        robot.send_action(action.astype(np.float32))

        # Maintain control rate
        elapsed = time.monotonic() - loop_start
        if elapsed < dt:
            time.sleep(dt - elapsed)

    return step + 1, time.monotonic() - t_start


def _predict_chunk(policy, robot, device, task, policy_type) -> np.ndarray:
    """
    One inference call. Returns [chunk_len, action_dim] np.ndarray.

    LeRobot 0.5.2 ACT exposes `predict_action_chunk(batch)` -> Tensor.
    Other policies may have different shapes; we try the common ones.
    """
    obs = robot.get_observation()
    batch = {}
    for k, v in obs.items():
        if torch.is_tensor(v):
            batch[k] = v.unsqueeze(0).to(device)
        elif isinstance(v, np.ndarray):
            t = torch.from_numpy(v)
            if t.dtype == torch.uint8 and t.dim() == 3:  # camera frame
                t = t.permute(2, 0, 1).float() / 255.0
            batch[k] = t.unsqueeze(0).to(device)
        else:
            batch[k] = v

    if policy_type in {"pi0"}:  # VLAs need the language instruction
        batch["task"] = [task]

    with torch.inference_mode():
        if hasattr(policy, "predict_action_chunk"):
            chunk = policy.predict_action_chunk(batch)
        else:
            # Fall back to single-action select_action; build a chunk of 1
            chunk = policy.select_action(batch).unsqueeze(0)

    if chunk.dim() == 3:
        chunk = chunk[0]   # drop batch dim
    return chunk.detach().cpu().numpy().astype(np.float32)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()
    args = parse_args()

    if not args.policy_path.exists():
        sys.exit(f"Policy path {args.policy_path} doesn't exist. "
                 f"Run data/pull_policy.py first.")

    port = os.environ.get("SO101_PORT")
    if not port:
        sys.exit("SO101_PORT not set in .env.")

    device = pick_device()
    print(f"Device: {device}")
    print(f"Loading {args.policy_type} policy from {args.policy_path}...")
    policy = load_policy(args.policy_type, args.policy_path, device)

    print(f"Connecting to SO-101 on {port}...")
    robot = open_robot(port)

    chunk_cfg = ChunkBufferConfig(
        chunk_execute_len=args.chunk_execute_len,
        smoothing_enabled=(args.smoothing == "on"),
    )
    chunk_buf = ChunkBuffer(chunk_cfg)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.results_dir / f"local_{args.policy_path.name}_{stamp}.csv"

    successes = 0
    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["trial", "success", "steps", "wall_s",
                             "smoothing", "chunk_exec_len", "notes"])
            f.flush()
            trial = 0
            while trial < args.num_trials:
                input(f"\n--- Trial {trial+1}/{args.num_trials} ---\n"
                      f"Reset the scene, press ENTER to start.")
                steps, wall = run_trial(
                    policy, robot, chunk_buf,
                    fps=args.fps, max_steps=args.max_steps, device=device,
                    task=args.task, policy_type=args.policy_type,
                )
                ok, notes = prompt_result(trial + 1)
                if ok is None:
                    print("Retrying.")
                    continue
                if ok:
                    successes += 1
                writer.writerow([trial + 1, int(ok), steps, round(wall, 2),
                                 1 if args.smoothing == "on" else 0,
                                 args.chunk_execute_len, notes])
                f.flush()
                trial += 1
    finally:
        robot.disconnect()

    rate = successes / args.num_trials if args.num_trials else 0.0
    print(f"\nResults: {successes}/{args.num_trials} ({rate:.0%})")
    print(f"CSV: {csv_path}")
    if rate >= 0.70:
        print("✅ Meets 70% Week-1 target.")
    else:
        print("❌ Below 70%. Look at: data diversity > training duration > hyperparams.")


if __name__ == "__main__":
    main()
