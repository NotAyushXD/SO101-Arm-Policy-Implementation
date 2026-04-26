"""
Wire protocol for client <-> server.

Defined as Pydantic models so both ends agree on schema, and so FastAPI
auto-generates request validation and OpenAPI docs.

This module is imported by both server and client — keep dependencies minimal
(pydantic only).
"""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Inference request / response
# ---------------------------------------------------------------------------

class CameraFrame(BaseModel):
    """A single camera frame in either JPEG-encoded or raw form."""
    name: str = Field(..., description="Camera name, e.g. 'wrist' or 'front'.")
    encoding: Literal["jpeg", "raw"] = "jpeg"
    width: int
    height: int
    # Base64-encoded bytes. JPEG: the JPEG file. Raw: row-major RGB uint8.
    data_b64: str


class InferenceRequest(BaseModel):
    """Sent from M2 client to GPU server."""
    # Joint state from the robot (length depends on robot type — 6 for SO-101).
    joint_state: list[float]

    # One or more camera frames. Order doesn't matter; server matches by name.
    cameras: list[CameraFrame]

    # Language instruction for VLA-style policies. ACT ignores it; we send it
    # anyway so the server can be swapped without protocol changes later.
    task: Optional[str] = None

    # Client's monotonic timestamp (ms). Server echoes this back so client
    # can compute end-to-end latency without clock-sync issues.
    client_send_ts_ms: float

    # Optional: client-supplied request id for tracing.
    request_id: Optional[str] = None


class InferenceResponse(BaseModel):
    """Sent from server back to client."""
    # Predicted action chunk. Shape: [chunk_len, action_dim].
    # For ACT on SO-101: typically [50, 6].
    action_chunk: list[list[float]]

    # Echo of client_send_ts_ms — lets client compute RTT without clock sync.
    client_send_ts_ms: float
    # Server's monotonic time when it finished inference (ms since server boot).
    server_inference_done_ms: float
    # Pure inference time excluding network and decode (ms).
    inference_ms: float
    # Decode time (JPEG → tensor) in ms. Useful for the JPEG-vs-raw ablation.
    decode_ms: float

    # Which checkpoint produced this. Lets client log "policy version X had
    # success rate Y" and pin revisions for ablations.
    policy_revision: str
    policy_commit_sha: Optional[str] = None

    request_id: Optional[str] = None


# ---------------------------------------------------------------------------
# Health + warmup (called before episode start)
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: Literal["ok", "loading", "error"]
    policy_loaded: bool
    policy_repo: str
    policy_revision: str
    policy_commit_sha: Optional[str] = None
    gpu_name: Optional[str] = None
    server_uptime_s: float
    # Counts of requests served since boot. Lets the UI show throughput.
    requests_served: int
    obs_logging_enabled: bool


# ---------------------------------------------------------------------------
# Observation logging (used by /obs/flush at session end)
# ---------------------------------------------------------------------------

class FlushObsRequest(BaseModel):
    """Triggered by client at end of session to push buffered obs to HF."""
    session_name: str
    push_to_hub: bool = True


class FlushObsResponse(BaseModel):
    n_observations: int
    pushed_to: Optional[str] = None  # HF dataset URL if push_to_hub was True
    elapsed_s: float
