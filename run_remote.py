"""
SO-101 remote inference client. Runs on the Mac M2.

Connects to the inference server (Colab/Kaggle/RunPod via Ngrok), captures
camera frames + joint state, sends them, receives action chunks, and either
logs them (shadow/visualize) or commands the robot (live_slow/live).

Every flag from the README is a CLI argument; defaults come from .env.

Usage:
    python client/run_remote.py --exec-mode shadow      --num-trials 1
    python client/run_remote.py --exec-mode visualize   --num-trials 1
    python client/run_remote.py --exec-mode live_slow   --num-trials 5
    python client/run_remote.py --exec-mode live        --num-trials 20

Recommended progression: shadow → visualize → live_slow → live, with at
least one episode of each before moving to the next.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import requests
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.protocol import (  # noqa: E402
    InferenceRequest, InferenceResponse, CameraFrame,
)
from shared.encoding import encode_frame, estimate_bandwidth_mbps  # noqa: E402
from client.chunk_buffer import ChunkBuffer, ChunkBufferConfig  # noqa: E402
from client.exec_modes import ExecModeRunner, ExecModeConfig  # noqa: E402


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SO-101 remote inference client.")
    p.add_argument("--server-url", type=str, default=None,
                   help="Override SERVER_URL from .env. Include scheme (https://).")
    p.add_argument("--exec-mode", choices=["shadow", "visualize", "live_slow", "live"],
                   default=None, help="Override EXEC_MODE from .env.")
    p.add_argument("--num-trials", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--fps", type=int, default=None,
                   help="Control loop fps. Default from .env (MAX_FPS).")
    p.add_argument("--jpeg", choices=["on", "off"], default=None,
                   help="JPEG encoding on/off. Default from .env (JPEG_ENCODE).")
    p.add_argument("--smoothing", choices=["on", "off"], default=None,
                   help="Temporal ensembling on/off. Default from .env (CHUNK_SMOOTHING).")
    p.add_argument("--results-dir", type=Path, default=Path("./remote_eval_results"))
    p.add_argument("--task", type=str,
                   default="Pick up the object and place it in the target zone.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Server client wrapper
# ---------------------------------------------------------------------------

class InferenceClient:
    """Thin HTTP wrapper around the FastAPI server."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        # Reasonable timeout: inference itself is <50ms but RTT may be 300ms.
        self.timeout_s = 5.0

    def healthz(self) -> dict:
        r = self.session.get(f"{self.base_url}/healthz", timeout=self.timeout_s)
        r.raise_for_status()
        return r.json()

    def warmup(self) -> None:
        r = self.session.post(f"{self.base_url}/warmup", timeout=30)
        r.raise_for_status()

    def infer(self, req: InferenceRequest) -> InferenceResponse:
        r = self.session.post(
            f"{self.base_url}/infer",
            json=req.model_dump(),
            timeout=self.timeout_s,
        )
        r.raise_for_status()
        return InferenceResponse(**r.json())

    def report_rtt(self, rtt_ms: float) -> None:
        # Fire-and-forget; don't block the control loop on telemetry.
        try:
            self.session.post(f"{self.base_url}/telemetry/rtt",
                              json={"rtt_ms": rtt_ms}, timeout=2)
        except Exception:
            pass

    def report_success(self, rate: float) -> None:
        try:
            self.session.post(f"{self.base_url}/telemetry/success",
                              json={"success_rate": rate}, timeout=2)
        except Exception:
            pass

    def flush_obs(self, session_name: str) -> dict:
        r = self.session.post(
            f"{self.base_url}/obs/flush",
            json={"session_name": session_name, "push_to_hub": True},
            timeout=300,  # uploads can be slow
        )
        r.raise_for_status()
        return r.json()


# ---------------------------------------------------------------------------
# Robot interface (lazy import)
# ---------------------------------------------------------------------------

def open_robot(port: str, cam_wrist: int, cam_front: int):
    """
    Open the SO-101 plus its two cameras. Returns a (robot, capture_fn) pair
    where capture_fn() → dict[str, np.ndarray] of HxWx3 uint8 RGB frames.
    """
    from lerobot.common.robot_devices.robots.factory import make_robot
    import cv2

    robot = make_robot(robot_type="so101_follower", port=port)
    robot.connect()

    cam_w = cv2.VideoCapture(cam_wrist)
    cam_f = cv2.VideoCapture(cam_front)
    for cap, name in ((cam_w, "wrist"), (cam_f, "front")):
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {name}.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    def capture() -> dict[str, np.ndarray]:
        frames = {}
        for cap, name in ((cam_w, "wrist"), (cam_f, "front")):
            ok, bgr = cap.read()
            if not ok:
                raise RuntimeError(f"Camera {name} returned no frame.")
            frames[name] = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return frames

    def close():
        cam_w.release()
        cam_f.release()
        robot.disconnect()

    return robot, capture, close


# ---------------------------------------------------------------------------
# Main control loop
# ---------------------------------------------------------------------------

def run_trial(client: InferenceClient, robot, capture, exec_runner: ExecModeRunner,
              chunk_buf: ChunkBuffer, *, fps: int, max_steps: int,
              use_jpeg: bool, jpeg_quality: int, task: str,
              trial_idx: int) -> tuple[int, float]:
    """Execute one trial. Returns (steps_executed, wall_clock_s)."""
    dt = 1.0 / fps
    chunk_buf.reset()
    t_start = time.monotonic()

    for step in range(max_steps):
        loop_start = time.monotonic()

        # If we need a new chunk, grab one synchronously. (Async prefetch is
        # a future improvement; not needed for week 1.)
        if chunk_buf.needs_new_chunk():
            try:
                _request_new_chunk(client, robot, capture, chunk_buf,
                                   use_jpeg=use_jpeg, jpeg_quality=jpeg_quality,
                                   task=task)
            except Exception as e:
                print(f"  [trial {trial_idx}] inference call failed: {e}")
                exec_runner.halt(robot)
                break

        # If the chunk is somehow stale (network blip during execution), halt.
        if chunk_buf.is_stale():
            print(f"  [trial {trial_idx}] chunk stale, halting arm.")
            exec_runner.halt(robot)
            break

        # Get next action from buffer (with smoothing if enabled).
        action = chunk_buf.next_action()
        if action is None:
            print(f"  [trial {trial_idx}] no action available, halting.")
            exec_runner.halt(robot)
            break

        # Read state and step the robot under the current exec mode.
        state = np.array(robot.get_observation()["observation.state"],
                         dtype=np.float32)
        exec_runner.step(robot, action, state)

        # Pace the loop.
        elapsed = time.monotonic() - loop_start
        if elapsed < dt:
            time.sleep(dt - elapsed)

    return step + 1, time.monotonic() - t_start


def _request_new_chunk(client: InferenceClient, robot, capture,
                       chunk_buf: ChunkBuffer, *, use_jpeg: bool,
                       jpeg_quality: int, task: str) -> None:
    """Capture obs, call /infer, install chunk into the buffer."""
    frames = capture()
    state = robot.get_observation()["observation.state"]
    if hasattr(state, "tolist"):
        state = state.tolist()

    cam_msgs = [
        encode_frame(name, rgb, use_jpeg=use_jpeg, jpeg_quality=jpeg_quality)
        for name, rgb in frames.items()
    ]

    send_ts_ms = time.monotonic() * 1000
    req = InferenceRequest(
        joint_state=list(state),
        cameras=cam_msgs,
        task=task,
        client_send_ts_ms=send_ts_ms,
        request_id=str(uuid.uuid4())[:8],
    )

    resp = client.infer(req)
    rtt_ms = time.monotonic() * 1000 - send_ts_ms
    client.report_rtt(rtt_ms)

    chunk_buf.install_new_chunk(resp.action_chunk)


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main() -> None:
    load_dotenv()
    args = parse_args()

    # --- resolve config from CLI > env > defaults ---
    server_url = args.server_url or os.environ.get("SERVER_URL")
    if not server_url:
        sys.exit("SERVER_URL not set (pass --server-url or put it in .env).")

    exec_mode = args.exec_mode or os.environ.get("EXEC_MODE", "shadow")
    fps       = args.fps       or int(os.environ.get("MAX_FPS", 30))

    use_jpeg_str = args.jpeg or os.environ.get("JPEG_ENCODE", "on")
    use_jpeg = (use_jpeg_str == "on")
    jpeg_quality = int(os.environ.get("JPEG_QUALITY", 85))

    smoothing_str = args.smoothing or os.environ.get("CHUNK_SMOOTHING", "on")
    smoothing = (smoothing_str == "on")

    chunk_execute_len = int(os.environ.get("CHUNK_EXECUTE_LEN", 30))
    stale_timeout_ms  = int(os.environ.get("STALE_TIMEOUT_MS", 500))
    live_slow_factor  = float(os.environ.get("LIVE_SLOW_FACTOR", 0.3))

    # --- bandwidth pre-flight + raw-mode fps throttle ---
    bw_mbps = estimate_bandwidth_mbps(640, 480, fps, n_cameras=2,
                                       use_jpeg=use_jpeg,
                                       jpeg_quality=jpeg_quality)
    print(f"\nEstimated upload bandwidth: {bw_mbps:.1f} Mbps "
          f"({'JPEG q=' + str(jpeg_quality) if use_jpeg else 'RAW'} @ {fps} fps × 2 cams)")
    if not use_jpeg and fps > 5:
        print("WARNING: raw mode at >5 fps will likely exceed Ngrok bandwidth. "
              "Throttling to 5 fps. Pass --fps explicitly to override.")
        fps = 5

    # --- connect ---
    print(f"Connecting to server at {server_url}...")
    client = InferenceClient(server_url)

    health = client.healthz()
    print(f"  Policy: {health['policy_repo']}@{health['policy_revision']}")
    print(f"  GPU: {health.get('gpu_name', 'cpu')}")
    print(f"  Obs logging: {'ON' if health['obs_logging_enabled'] else 'off'}")

    print("Warming up server (compiles CUDA kernels)...")
    client.warmup()

    # --- robot ---
    port = os.environ.get("SO101_PORT")
    cam_wrist = int(os.environ.get("CAM_WRIST", 0))
    cam_front = int(os.environ.get("CAM_FRONT", 1))
    if not port:
        sys.exit("SO101_PORT not set in .env.")

    print(f"Connecting to SO-101 on {port}...")
    robot, capture, close_robot = open_robot(port, cam_wrist, cam_front)

    # --- exec mode + chunk buffer ---
    exec_cfg = ExecModeConfig(mode=exec_mode, live_slow_factor=live_slow_factor)
    exec_runner = ExecModeRunner(exec_cfg)
    print(f"\nExecution mode: {exec_mode.upper()}"
          f"{' (factor=' + str(live_slow_factor) + ')' if exec_mode == 'live_slow' else ''}")
    if exec_runner.will_move:
        print("⚠️  Arm will move. Keep your hand near the e-stop.")

    chunk_cfg = ChunkBufferConfig(
        chunk_execute_len=chunk_execute_len,
        smoothing_enabled=smoothing,
        stale_timeout_ms=stale_timeout_ms,
    )
    chunk_buf = ChunkBuffer(chunk_cfg)
    print(f"Chunking: execute {chunk_execute_len}/chunk, "
          f"smoothing {'ON' if smoothing else 'off'}, "
          f"stale timeout {stale_timeout_ms} ms")

    # --- trial loop ---
    args.results_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = args.results_dir / f"remote_{exec_mode}_{stamp}.csv"

    successes = 0
    try:
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["trial_idx", "exec_mode", "success", "steps",
                             "wall_clock_s", "smoothing", "jpeg", "fps", "notes"])
            for i in range(args.num_trials):
                input(f"\n--- Trial {i+1}/{args.num_trials} ---\n"
                      f"Reset the scene, then press ENTER to start.")
                steps, wall = run_trial(
                    client, robot, capture, exec_runner, chunk_buf,
                    fps=fps, max_steps=args.max_steps,
                    use_jpeg=use_jpeg, jpeg_quality=jpeg_quality,
                    task=args.task, trial_idx=i+1,
                )

                if exec_runner.will_move:
                    ans = input(f"Trial {i+1}: success? [y/n]: ").strip().lower()
                    success = (ans == "y")
                    if success:
                        successes += 1
                else:
                    print(f"Trial {i+1} ran in {exec_mode} mode "
                          f"(arm did not move; success not labelled).")
                    success = False

                writer.writerow([i+1, exec_mode, int(success), steps,
                                 round(wall, 2),
                                 1 if smoothing else 0,
                                 1 if use_jpeg else 0,
                                 fps, ""])
                f.flush()

        # Summary + push to server.
        if exec_runner.will_move and args.num_trials > 0:
            rate = successes / args.num_trials
            print(f"\nSuccess: {successes}/{args.num_trials} ({rate:.0%})")
            client.report_success(rate)
        print(f"CSV: {csv_path}")

    finally:
        close_robot()


if __name__ == "__main__":
    main()
