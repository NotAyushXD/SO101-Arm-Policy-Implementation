"""
FastAPI inference server. Runs on Colab / Kaggle / RunPod — same code
everywhere, only the bootstrap differs per platform.

Policy selection: set POLICY_TYPE environment variable. Built-in adapters:
  - act      ACT (LeRobot)
  - pi0      π₀ (stub — not yet implemented)
  - random   random walk; no GPU; for pipeline debugging
  - echo     echoes current joint state; safe for live-mode comms testing

To add a new policy: drop a file in server/policies/, decorate the class
with @register("name"), and import it in server/policies/__init__.py.
The /infer endpoint, web UI, telemetry, and observation logging all work
without changes.

Endpoints:
  GET  /                     Web UI
  GET  /healthz              Liveness + policy info
  POST /infer                Main inference endpoint
  POST /warmup               Trigger CUDA kernel compile
  GET  /telemetry            Latency stats for UI
  GET  /last-frame/{name}    Most recent frame for UI
  POST /control/mode         Set advisory exec mode (UI -> server)
  GET  /control/state        Current advisory state (client polls)
  POST /policy/reload        Reload at a different revision
  POST /policy/swap          Swap to a different POLICY_TYPE entirely
  POST /telemetry/rtt        Client posts measured RTT
  POST /telemetry/success    Client posts success rate
  POST /obs/flush            Push observation buffer to HF dataset
"""
from __future__ import annotations

import io
import os
import sys
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))

from shared.protocol import (  # noqa: E402
    InferenceRequest, InferenceResponse, HealthResponse,
    FlushObsRequest, FlushObsResponse,
)
from shared.encoding import decode_frame  # noqa: E402
from server.obs_logger import ObservationLogger, LoggedObservation  # noqa: E402
from server.ui import INDEX_HTML  # noqa: E402

# Importing the policies package triggers @register decorators and populates
# the ADAPTERS dict.
from server.policies import ADAPTERS, load_adapter, PolicyAdapter  # noqa: E402


# ---------------------------------------------------------------------------
# Config from env
# ---------------------------------------------------------------------------

POLICY_TYPE     = os.environ.get("POLICY_TYPE", "act")
POLICY_REPO     = os.environ.get("POLICY_REPO", "")  # not required for random/echo
POLICY_REVISION = os.environ.get("POLICY_REVISION", "main")
HF_TOKEN        = os.environ.get("HF_TOKEN")
OBS_LOGGING     = os.environ.get("OBS_LOGGING", "off") == "on"
OBS_LOG_REPO    = os.environ.get("OBS_LOG_REPO")
WANDB_ENABLE    = os.environ.get("WANDB_ENABLE", "on") == "on"
WANDB_PROJECT   = os.environ.get("WANDB_PROJECT", "so101-remote-inf")

_NEEDS_REPO = {"act", "pi0", "openvla"}
if POLICY_TYPE in _NEEDS_REPO and not POLICY_REPO:
    sys.exit(f"POLICY_REPO required for POLICY_TYPE={POLICY_TYPE}.")

print(f"[config] POLICY_TYPE={POLICY_TYPE} POLICY_REPO={POLICY_REPO or '(none)'} "
      f"POLICY_REVISION={POLICY_REVISION}")
print(f"[config] Available adapters: {sorted(ADAPTERS.keys())}")


# ---------------------------------------------------------------------------
# Server state
# ---------------------------------------------------------------------------

class ServerState:
    def __init__(self):
        self.adapter: Optional[PolicyAdapter] = None
        self.policy_type: str = POLICY_TYPE
        self.policy_repo: str = POLICY_REPO
        self.boot_ts: float = time.monotonic()
        self.requests_served: int = 0

        # Per-episode opaque state for stateful adapters (autoregressive VLAs).
        # The server hands whatever the adapter returned last call back in next
        # call. Reset to None on /warmup and on policy swap/reload.
        self.adapter_state: Optional[Any] = None

        self.recent_inference_ms: deque[float] = deque(maxlen=100)
        self.recent_decode_ms:    deque[float] = deque(maxlen=100)
        self.recent_client_rtt:   deque[float] = deque(maxlen=100)
        self.last_frames: dict[str, bytes] = {}
        self.advisory_mode: str = "shadow"
        self.last_success_rate: Optional[float] = None

        self.obs_logger = ObservationLogger(
            enabled=OBS_LOGGING, repo_id=OBS_LOG_REPO, hf_token=HF_TOKEN,
        )
        self.wandb_run = None


STATE = ServerState()


# ---------------------------------------------------------------------------
# Adapter loading
# ---------------------------------------------------------------------------

def _instantiate_adapter(policy_type: str, repo: str, revision: str) -> PolicyAdapter:
    print(f"[adapter] Loading type={policy_type} repo={repo or '(none)'}@{revision}")
    return load_adapter(
        policy_type,
        repo=repo or policy_type,   # debug adapters use the type name as repo
        revision=revision,
        hf_token=HF_TOKEN,
    )


# ---------------------------------------------------------------------------
# FastAPI lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    STATE.adapter = _instantiate_adapter(STATE.policy_type, STATE.policy_repo,
                                         POLICY_REVISION)

    if WANDB_ENABLE:
        try:
            import wandb
            STATE.wandb_run = wandb.init(
                project=WANDB_PROJECT,
                name=f"server-{STATE.policy_type}-{int(time.time())}",
                config={
                    "policy_type": STATE.policy_type,
                    "policy_repo": STATE.policy_repo,
                    "policy_revision": STATE.adapter.revision,
                    "policy_commit_sha": STATE.adapter.commit_sha,
                    "chunk_len": STATE.adapter.chunk_len,
                    "action_dim": STATE.adapter.action_dim,
                },
                reinit=True,
            )
            print(f"[wandb] {STATE.wandb_run.url}")
        except Exception as e:
            print(f"[wandb] disabled: {e}")
            STATE.wandb_run = None

    yield
    if STATE.wandb_run is not None:
        STATE.wandb_run.finish()


app = FastAPI(lifespan=lifespan, title="SO-101 Remote Inference")


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def index() -> HTMLResponse:
    return HTMLResponse(INDEX_HTML)


@app.get("/last-frame/{name}")
def last_frame(name: str) -> Response:
    jpg = STATE.last_frames.get(name)
    if jpg is None:
        return Response(content=b"", media_type="image/jpeg", status_code=204)
    return Response(content=jpg, media_type="image/jpeg")


# ---------------------------------------------------------------------------
# Health + warmup
# ---------------------------------------------------------------------------

@app.get("/healthz", response_model=HealthResponse)
def healthz() -> HealthResponse:
    a = STATE.adapter
    return HealthResponse(
        status="ok" if a is not None else "loading",
        policy_loaded=a is not None,
        policy_repo=STATE.policy_repo or STATE.policy_type,
        policy_revision=a.revision if a else POLICY_REVISION,
        policy_commit_sha=a.commit_sha if a else None,
        gpu_name=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        server_uptime_s=time.monotonic() - STATE.boot_ts,
        requests_served=STATE.requests_served,
        obs_logging_enabled=STATE.obs_logger.enabled,
    )


@app.post("/warmup")
def warmup() -> dict:
    if STATE.adapter is None:
        raise HTTPException(503, "Adapter not loaded.")
    STATE.adapter.reset()
    STATE.adapter_state = None
    print(f"[warmup] {STATE.policy_type} adapter reset.")
    return {"status": "ok", "policy_type": STATE.policy_type}


# ---------------------------------------------------------------------------
# Inference (the hot path — adapter-agnostic)
# ---------------------------------------------------------------------------

@app.post("/infer", response_model=InferenceResponse)
def infer(req: InferenceRequest) -> InferenceResponse:
    if STATE.adapter is None:
        raise HTTPException(503, "Adapter not loaded.")

    request_id = req.request_id or str(uuid.uuid4())[:8]

    # --- decode frames ---
    t_dec_start = time.monotonic()
    images: dict[str, np.ndarray] = {}
    cam_jpegs: dict[str, bytes] = {}
    for frame in req.cameras:
        rgb = decode_frame(frame)
        images[frame.name] = rgb
        if frame.encoding == "jpeg":
            import base64
            cam_jpegs[frame.name] = base64.b64decode(frame.data_b64)
        else:
            from PIL import Image
            buf = io.BytesIO()
            Image.fromarray(rgb).save(buf, format="JPEG", quality=80)
            cam_jpegs[frame.name] = buf.getvalue()
    decode_ms = (time.monotonic() - t_dec_start) * 1000
    STATE.last_frames.update(cam_jpegs)

    # --- check language requirement ---
    if STATE.adapter.requires_language and not req.task:
        raise HTTPException(400, f"{STATE.adapter.name} requires a 'task' string.")

    # --- inference (adapter does the heavy lifting) ---
    joint_state = np.asarray(req.joint_state, dtype=np.float32)
    t_inf_start = time.monotonic()
    chunk_np, new_state = STATE.adapter.predict_chunk(
        joint_state=joint_state,
        images=images,
        task=req.task,
        state=STATE.adapter_state,
    )
    inference_ms = (time.monotonic() - t_inf_start) * 1000
    STATE.adapter_state = new_state

    # --- telemetry ---
    STATE.requests_served += 1
    STATE.recent_inference_ms.append(inference_ms)
    STATE.recent_decode_ms.append(decode_ms)
    if STATE.wandb_run is not None:
        STATE.wandb_run.log({
            "server/inference_ms": inference_ms,
            "server/decode_ms": decode_ms,
            "server/policy_type": STATE.policy_type,
        }, step=STATE.requests_served)

    # --- obs logging ---
    if STATE.obs_logger.enabled:
        STATE.obs_logger.log(LoggedObservation(
            timestamp_ms=time.monotonic() * 1000,
            joint_state=req.joint_state,
            cam_frames_jpeg=cam_jpegs,
            action_chunk=chunk_np.tolist(),
            task=req.task,
            request_id=request_id,
        ))

    return InferenceResponse(
        action_chunk=chunk_np.tolist(),
        client_send_ts_ms=req.client_send_ts_ms,
        server_inference_done_ms=time.monotonic() * 1000,
        inference_ms=inference_ms,
        decode_ms=decode_ms,
        policy_revision=STATE.adapter.revision,
        policy_commit_sha=STATE.adapter.commit_sha,
        request_id=request_id,
    )


# ---------------------------------------------------------------------------
# Telemetry + control
# ---------------------------------------------------------------------------

@app.get("/telemetry")
def telemetry() -> dict:
    def median(d):
        return float(np.median(list(d))) if d else None
    return {
        "inference_ms_p50": median(STATE.recent_inference_ms),
        "decode_ms_p50":    median(STATE.recent_decode_ms),
        "client_rtt_ms_p50": median(STATE.recent_client_rtt),
        "recent_rtt_ms": list(STATE.recent_client_rtt),
        "advisory_mode": STATE.advisory_mode,
        "last_success_rate": STATE.last_success_rate,
        "n_buffered_obs": len(STATE.obs_logger),
        "obs_buffer_mb": round(STATE.obs_logger.estimated_size_mb(), 1),
        "policy_type": STATE.policy_type,
        "available_policy_types": sorted(ADAPTERS.keys()),
    }


class ClientRttReport(BaseModel):
    rtt_ms: float

@app.post("/telemetry/rtt")
def report_rtt(report: ClientRttReport) -> dict:
    STATE.recent_client_rtt.append(report.rtt_ms)
    if STATE.wandb_run is not None:
        STATE.wandb_run.log({"client/rtt_ms": report.rtt_ms},
                            step=STATE.requests_served)
    return {"ok": True}


class ClientSuccessReport(BaseModel):
    success_rate: float

@app.post("/telemetry/success")
def report_success(report: ClientSuccessReport) -> dict:
    STATE.last_success_rate = report.success_rate
    if STATE.wandb_run is not None:
        STATE.wandb_run.log({"client/success_rate": report.success_rate})
    return {"ok": True}


class ModeRequest(BaseModel):
    mode: str

@app.post("/control/mode")
def set_mode(req: ModeRequest) -> dict:
    if req.mode not in {"shadow", "visualize", "live_slow", "live"}:
        raise HTTPException(400, f"Unknown mode: {req.mode}")
    STATE.advisory_mode = req.mode
    print(f"[control] advisory mode = {req.mode}")
    return {"ok": True, "mode": STATE.advisory_mode}


@app.get("/control/state")
def get_state() -> dict:
    return {"advisory_mode": STATE.advisory_mode}


# ---------------------------------------------------------------------------
# Policy management
# ---------------------------------------------------------------------------

class ReloadRequest(BaseModel):
    revision: str

@app.post("/policy/reload")
def reload_policy(req: ReloadRequest) -> dict:
    """Reload the same POLICY_TYPE at a different revision."""
    try:
        new_adapter = _instantiate_adapter(STATE.policy_type, STATE.policy_repo,
                                            req.revision)
    except Exception as e:
        raise HTTPException(500, f"Reload failed: {e}")

    old = STATE.adapter
    STATE.adapter = new_adapter
    STATE.adapter_state = None
    del old
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"ok": True, "revision": new_adapter.revision,
            "commit_sha": new_adapter.commit_sha}


class SwapRequest(BaseModel):
    policy_type: str
    repo: Optional[str] = None
    revision: Optional[str] = "main"

@app.post("/policy/swap")
def swap_policy(req: SwapRequest) -> dict:
    """Swap to a different POLICY_TYPE entirely. Lets you flip between
    e.g. random / echo / act in a single debug session without restarting."""
    if req.policy_type not in ADAPTERS:
        raise HTTPException(400, f"Unknown policy_type. Available: {sorted(ADAPTERS.keys())}")
    repo = req.repo or req.policy_type
    revision = req.revision or "main"
    try:
        new_adapter = _instantiate_adapter(req.policy_type, repo, revision)
    except Exception as e:
        raise HTTPException(500, f"Swap failed: {e}")

    old = STATE.adapter
    STATE.adapter = new_adapter
    STATE.policy_type = req.policy_type
    STATE.policy_repo = repo
    STATE.adapter_state = None
    del old
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"ok": True, "policy_type": req.policy_type,
            "revision": new_adapter.revision}


# ---------------------------------------------------------------------------
# Obs flush
# ---------------------------------------------------------------------------

@app.post("/obs/flush", response_model=FlushObsResponse)
def flush_obs(req: FlushObsRequest) -> FlushObsResponse:
    if not STATE.obs_logger.enabled:
        raise HTTPException(400, "Observation logging is disabled.")
    t0 = time.monotonic()
    n, hub_url = STATE.obs_logger.flush_to_hub(req.session_name, push=req.push_to_hub)
    return FlushObsResponse(
        n_observations=n, pushed_to=hub_url,
        elapsed_s=round(time.monotonic() - t0, 2),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.server:app", host="0.0.0.0", port=8000, reload=False)
