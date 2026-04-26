"""
Quick RTT benchmark — run this from the M2 BEFORE connecting the robot.

Sends fake observations to the inference server for ~30 seconds and prints
the latency distribution. If median RTT is over 500 ms you'll know the
remote-inference path won't work for live mode and can adjust expectations
or switch regions before you waste a robot session.

Usage:
    python tools/benchmark_rtt.py --server-url https://abc-123.ngrok.io
    python tools/benchmark_rtt.py --duration 60 --jpeg off    # raw mode test
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
import uuid
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))
from shared.protocol import InferenceRequest  # noqa: E402
from shared.encoding import encode_frame, estimate_bandwidth_mbps  # noqa: E402
from client.run_remote import InferenceClient  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--server-url", type=str, default=None)
    p.add_argument("--duration", type=int, default=30)
    p.add_argument("--fps", type=int, default=10,
                   help="Request rate. Lower than control-loop fps because "
                        "we're benchmarking the network, not running control.")
    p.add_argument("--jpeg", choices=["on", "off"], default="on")
    p.add_argument("--jpeg-quality", type=int, default=85)
    return p.parse_args()


def main() -> None:
    load_dotenv()
    args = parse_args()
    server_url = args.server_url or os.environ.get("SERVER_URL")
    if not server_url:
        sys.exit("SERVER_URL not set.")

    use_jpeg = (args.jpeg == "on")
    bw = estimate_bandwidth_mbps(640, 480, args.fps, n_cameras=2,
                                  use_jpeg=use_jpeg, jpeg_quality=args.jpeg_quality)
    print(f"Estimated bandwidth: {bw:.1f} Mbps")

    client = InferenceClient(server_url)
    health = client.healthz()
    print(f"Server: {health['policy_repo']}@{health['policy_revision']}")
    print(f"GPU: {health.get('gpu_name', 'cpu')}")
    print("Warming up...")
    client.warmup()

    rng = np.random.default_rng(0)
    fake_state = [0.0] * 6

    rtts = []
    inferences = []
    decodes = []
    failures = 0

    print(f"\nBenchmarking for {args.duration}s at {args.fps} req/s "
          f"({'JPEG q=' + str(args.jpeg_quality) if use_jpeg else 'RAW'})...")
    deadline = time.monotonic() + args.duration
    dt = 1.0 / args.fps

    while time.monotonic() < deadline:
        loop_start = time.monotonic()

        # New random frames each call so JPEG isn't artificially compressible.
        frames = {
            "wrist": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
            "front": rng.integers(0, 255, (480, 640, 3), dtype=np.uint8),
        }
        cams = [encode_frame(n, f, use_jpeg=use_jpeg, jpeg_quality=args.jpeg_quality)
                for n, f in frames.items()]
        send_ts = time.monotonic() * 1000
        req = InferenceRequest(
            joint_state=fake_state, cameras=cams,
            client_send_ts_ms=send_ts, request_id=str(uuid.uuid4())[:8],
        )

        try:
            resp = client.infer(req)
            rtt = time.monotonic() * 1000 - send_ts
            rtts.append(rtt)
            inferences.append(resp.inference_ms)
            decodes.append(resp.decode_ms)
        except Exception as e:
            failures += 1
            print(f"  [fail] {e}")

        elapsed = time.monotonic() - loop_start
        if elapsed < dt:
            time.sleep(dt - elapsed)

    print("\n=== Results ===")
    print(f"  Requests: {len(rtts)} ({failures} failed)")
    if not rtts:
        sys.exit("No successful requests; check server connectivity.")

    def quantile(xs, q):
        return statistics.quantiles(xs, n=100)[q-1] if len(xs) >= 100 else statistics.median(xs)

    print(f"  Total RTT (client wall-clock):")
    print(f"    median: {statistics.median(rtts):6.1f} ms")
    print(f"    p95:    {quantile(rtts, 95):6.1f} ms")
    print(f"    p99:    {quantile(rtts, 99):6.1f} ms")
    print(f"    max:    {max(rtts):6.1f} ms")
    print(f"  Server inference: median {statistics.median(inferences):.1f} ms")
    print(f"  Server decode:    median {statistics.median(decodes):.1f} ms")
    print(f"  Network share:    {statistics.median(rtts) - statistics.median(inferences) - statistics.median(decodes):.1f} ms")

    rtt_med = statistics.median(rtts)
    if rtt_med < 100:
        print("\n✅ Excellent. Live control should feel native.")
    elif rtt_med < 300:
        print("\n✅ Workable. Action chunking will hide most of the latency.")
    elif rtt_med < 600:
        print("\n⚠️  Marginal. Live control will feel laggy; consider a closer region.")
    else:
        print("\n❌ Too high. Use a closer GPU region (Singapore/Mumbai for Chennai) "
              "or switch to local M2 inference.")


if __name__ == "__main__":
    main()
