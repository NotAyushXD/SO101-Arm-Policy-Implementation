"""
In-memory observation buffer that pushes to a HF dataset at session end.

Per the design choice: simple buffer-and-flush. NOT a streaming logger; the
risk is that a 60-min session at 30 fps with two 640x480 RGB cameras buffers
~5 GB in RAM. That fits comfortably on a Colab T4 (12 GB system RAM) for
typical week-1 sessions but won't scale forever.

Memory budget:
  30 fps * 60 min * (2 * 640*480*3 bytes + 6*8 + 6*8)
  ≈ 5.0 GB for two RGB streams + state + action

Mitigation: we store JPEG-encoded frames in the buffer (whatever the wire
encoding was), so it's actually closer to ~250 MB for a 60-min session.
"""
from __future__ import annotations

import io
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class LoggedObservation:
    """A single (obs, action_chunk) pair as the server saw it."""
    timestamp_ms: float          # server-local monotonic ms
    joint_state: list[float]
    # Camera frames are stored JPEG-encoded as bytes for memory efficiency.
    # When we flush to HF, these become PNG/JPEG files in the dataset.
    cam_frames_jpeg: dict[str, bytes]
    action_chunk: list[list[float]]
    task: Optional[str] = None
    request_id: Optional[str] = None


@dataclass
class ObservationLogger:
    """Buffer observations in memory; flush to HF at session end."""
    enabled: bool = False
    repo_id: Optional[str] = None
    hf_token: Optional[str] = None
    buffer: list[LoggedObservation] = field(default_factory=list)
    session_start_ms: float = field(default_factory=lambda: time.monotonic() * 1000)

    def log(self, obs: LoggedObservation) -> None:
        if not self.enabled:
            return
        self.buffer.append(obs)

    def __len__(self) -> int:
        return len(self.buffer)

    def estimated_size_mb(self) -> float:
        if not self.buffer:
            return 0.0
        # Rough: average frame size × n frames × n cameras + scalar overhead.
        sample = self.buffer[0]
        bytes_per_obs = sum(len(jpg) for jpg in sample.cam_frames_jpeg.values())
        return (bytes_per_obs * len(self.buffer)) / 1_000_000

    def flush_to_hub(self, session_name: str, push: bool = True) -> tuple[int, Optional[str]]:
        """
        Write buffer to a local Parquet + frame folder, optionally push to HF.

        Returns (n_observations, hub_url_if_pushed).
        """
        if not self.buffer:
            return 0, None

        # Lazy import — only needed at flush time.
        import pandas as pd
        from huggingface_hub import HfApi, create_repo

        out_dir = Path(f"/tmp/obs_dump_{session_name}")
        frames_dir = out_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for i, obs in enumerate(self.buffer):
            # Save each camera frame as a separate file; reference by path.
            frame_paths = {}
            for cam_name, jpeg_bytes in obs.cam_frames_jpeg.items():
                fname = f"{i:06d}_{cam_name}.jpg"
                (frames_dir / fname).write_bytes(jpeg_bytes)
                frame_paths[cam_name] = f"frames/{fname}"

            rows.append({
                "idx": i,
                "timestamp_ms": obs.timestamp_ms,
                "joint_state": obs.joint_state,
                "action_chunk": obs.action_chunk,
                "task": obs.task,
                "request_id": obs.request_id,
                **{f"frame_{k}": v for k, v in frame_paths.items()},
            })

        df = pd.DataFrame(rows)
        parquet_path = out_dir / "observations.parquet"
        df.to_parquet(parquet_path, index=False)

        meta = {
            "session_name": session_name,
            "n_observations": len(self.buffer),
            "session_start_ms": self.session_start_ms,
            "session_end_ms": time.monotonic() * 1000,
            "cameras": list(self.buffer[0].cam_frames_jpeg.keys()),
        }
        (out_dir / "metadata.json").write_text(json.dumps(meta, indent=2))

        hub_url: Optional[str] = None
        if push and self.repo_id and self.hf_token:
            api = HfApi(token=self.hf_token)
            create_repo(self.repo_id, repo_type="dataset", exist_ok=True,
                        token=self.hf_token)
            api.upload_folder(
                folder_path=str(out_dir),
                repo_id=self.repo_id,
                repo_type="dataset",
                path_in_repo=session_name,
                commit_message=f"Session {session_name}: {len(self.buffer)} obs",
            )
            hub_url = f"https://huggingface.co/datasets/{self.repo_id}/tree/main/{session_name}"

        n = len(self.buffer)
        self.buffer.clear()
        return n, hub_url
