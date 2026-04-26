"""
Remote inference client. Same control loop as run_local.py, but inference
happens on a remote server (Kaggle/Colab/RunPod) reached via Ngrok.

Use this for VLAs (π₀, OpenVLA) where M2 memory or compute is the bottleneck.
For ACT, run_local.py is faster — no network round trip.

Usage:
    python client/run_remote.py \\
        --server-url https://abc-123.ngrok.io \\
        --num-trials 20

Wire protocol: simple JSON-over-HTTP. The server is in inference/server.py
and exposes POST /infer.
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))
from client.chunk_buffer import ChunkBuffer, ChunkBufferConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Remote inference client.")
    p.add_argument("--server-url", default=None,
                   help="Ngrok URL of the inference server. Defaults to "
                        "SERVER_URL in .env.")
    p.add_argument("--num-trials", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--smoothing", choices=["on", "off"], default="on")
    p.add_argument("--chunk-execute-len", type=int, default=30)
    p.add_argument("--jpeg-quality", type=int, default=85,
                   help="JPEG quality for camera frames sent over the wire.")
    p.add_argument("--results-dir", type=Path, default=Path("./eval_results"))
    p.add_argument("--task", default="Pick up the object and drop it in the bowl.")
    return p.parse_args()


def encode_jpeg(rgb: np.ndarray, quality: int) -> str:
    """RGB uint8 array → base64-encoded JPEG. ~25 Mbps at 30 fps × 2 cams."""
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def open_robot(port: str):
    """LeRobot 0.5.2 SO-101 follower."""
    from lerobot.robots.so101_follower import SO101Follower, SO101FollowerConfig
    cfg = SO101FollowerConfig(port=port, id="follower_arm")
    robot = SO101Follower(cfg)
    robot.connect()
    return robot


def call_server(url: str, joint_state, frames: dict, task: str,
                jpeg_q: int, timeout_s: float = 5.0) -> tuple[np.ndarray, float, dict]:
    """One inference round trip. Returns (chunk, rtt_ms, response_dict)."""
    cams = []
    for name, rgb in frames.items():
        cams.append({
            "name": name,
            "encoding": "jpeg",
            "width": rgb.shape[1],
            "height": rgb.shape[0],
            "data_b64": encode_jpeg(rgb, jpeg_q),
        })

    send_ts = time.monotonic() * 1000
    payload = {
        "joint_state": list(joint_state),
        "cameras": cams,
        "task": task,
        "client_send_ts_ms": send_ts,
        "request_id": str(uuid.uuid4())[:8],
    }
    r = requests.post(f"{url}/infer", json=payload, timeout=timeout_s)
    r.raise_for_status()
    rtt_ms = time.monotonic() * 1000 - send_ts
    body = r.json()
    chunk = np.asarray(body["action_chunk"], dtype=np.float32)
    return chunk, rtt_ms, body


def run_trial(server_url, robot, chunk_buf, *, fps, max_steps, task,
              jpeg_q) -> tuple[int, float, list[float]]:
    """Returns (steps, wall_s, list_of_rtt_ms)."""
    dt = 1.0 / fps
    chunk_buf.reset()
    rtts: list[float] = []
    t_start = time.monotonic()

    for step in range(max_steps):
        loop_start = time.monotonic()

        if chunk_buf.needs_new_chunk():
            obs = robot.get_observation()
            joint = obs.get("observation.state")
            if hasattr(joint, "tolist"):
                joint = joint.tolist()
            # Pull cameras out of the obs dict
            frames = {}
            for k, v in obs.items():
                if k.startswith("observation.images.") and isinstance(v, np.ndarray):
                    name = k.replace("observation.images.", "")
                    frames[name] = v if v.dtype == np.uint8 else (v * 255).astype(np.uint8)
            try:
                chunk, rtt_ms, _ = call_server(server_url, joint, frames, task, jpeg_q)
            except Exception as e:
                print(f"  inference call failed: {e}")
                break
            rtts.append(rtt_ms)
            chunk_buf.install_new_chunk(chunk)

        if chunk_buf.is_stale():
            print("  chunk stale, halting.")
            break

        action = chunk_buf.next_action()
        if action is None:
            break

        robot.send_action(action.astype(np.float32))

        elapsed = time.monotonic() - loop_start
        if elapsed < dt:
            time.sleep(dt - elapsed)

    return step + 1, time.monotonic() - t_start, rtts


def prompt_result(trial: int) -> tuple[bool | None, str]:
    while True:
        ans = input(f"\nTrial {trial}: success? [y/n/r/q]: ").strip().lower()
        if ans == "y":
            return True, input("Notes: ").strip()
        if ans == "n":
            return False, input("What happened? ").strip()
        if ans == "r":
            return None, ""
        if ans == "q":
            sys.exit("Aborted.")


def main() -> None:
    load_dotenv()
    args = parse_args()

    server_url = (args.server_url or os.environ.get("SERVER_URL", "")).rstrip("/")
    if not server_url:
        sys.exit("SERVER_URL not set. Pass --server-url or set in .env.")

    port = os.environ.get("SO101_PORT")
    if not port:
        sys.exit("SO101_PORT not set in .env.")

    print(f"Server: {server_url}")
    # Health check before opening the robot
    r = requests.get(f"{server_url}/healthz", timeout=10)
    r.raise_for_status()
    health = r.json()
    print(f"  Policy on server: {health.get('policy_repo')}@{health.get('policy_revision')}")

    print(f"Connecting to SO-101 on {port}...")
    robot = open_robot(port)

    chunk_buf = ChunkBuffer(ChunkBufferConfig(
        chunk_execute_len=args.chunk_execute_len,
        smoothing_enabled=(args.smoothing == "on"),
    ))

    args.results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.results_dir / f"remote_{stamp}.csv"

    successes = 0
    try:
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["trial", "success", "steps", "wall_s", "rtt_p50_ms", "notes"])
            trial = 0
            while trial < args.num_trials:
                input(f"\n--- Trial {trial+1}/{args.num_trials} --- press ENTER")
                steps, wall, rtts = run_trial(
                    server_url, robot, chunk_buf,
                    fps=args.fps, max_steps=args.max_steps,
                    task=args.task, jpeg_q=args.jpeg_quality,
                )
                rtt_p50 = float(np.median(rtts)) if rtts else 0.0
                print(f"  steps={steps}, wall={wall:.1f}s, rtt_p50={rtt_p50:.0f}ms")
                ok, notes = prompt_result(trial + 1)
                if ok is None:
                    continue
                if ok:
                    successes += 1
                w.writerow([trial + 1, int(ok), steps, round(wall, 2),
                            round(rtt_p50, 1), notes])
                f.flush()
                trial += 1
    finally:
        robot.disconnect()

    rate = successes / args.num_trials if args.num_trials else 0.0
    print(f"\nResults: {successes}/{args.num_trials} ({rate:.0%})")
    print(f"CSV: {csv_path}")


if __name__ == "__main__":
    main()
