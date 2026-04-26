"""
FastAPI inference server. Runs on Colab/Kaggle/RunPod — same code everywhere,
only the bootstrap differs.

Adapter selection: POLICY_TYPE env var. Built-in: act, pi0, random, echo.

Env vars:
  POLICY_TYPE       — required, one of registered adapter names
  POLICY_REPO       — required for act/pi0; ignored for random/echo
  POLICY_REVISION   — default "main"
  HF_TOKEN          — required (write scope unneeded; read is fine)

Endpoints:
  GET  /healthz    — liveness + which policy is loaded
  POST /infer      — main inference call (body: joint_state, cameras, task)
  POST /warmup     — reset adapter state, trigger CUDA kernel compile
  POST /policy/swap — switch adapter at runtime (body: policy_type, repo)
"""
from __future__ import annotations

import base64
import io
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent.parent))
from inference.policies import ADAPTERS, PolicyAdapter, load_adapter  # noqa: E402


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

POLICY_TYPE     = os.environ.get("POLICY_TYPE", "random")
POLICY_REPO     = os.environ.get("POLICY_REPO", "")
POLICY_REVISION = os.environ.get("POLICY_REVISION", "main")
HF_TOKEN        = os.environ.get("HF_TOKEN")

_NEEDS_REPO = {"act", "pi0", "openvla"}
if POLICY_TYPE in _NEEDS_REPO and not POLICY_REPO:
    sys.exit(f"POLICY_REPO required for POLICY_TYPE={POLICY_TYPE}.")

print(f"[config] POLICY_TYPE={POLICY_TYPE} POLICY_REPO={POLICY_REPO or '(none)'}")
print(f"[config] Available adapters: {sorted(ADAPTERS.keys())}")


# ---------------------------------------------------------------------------
# Wire types
# ---------------------------------------------------------------------------

class CameraFrame(BaseModel):
    name: str
    encoding: str = "jpeg"
    width: int
    height: int
    data_b64: str


class InferenceRequest(BaseModel):
    joint_state: list[float]
    cameras: list[CameraFrame]
    task: Optional[str] = None
    client_send_ts_ms: float
    request_id: Optional[str] = None


class InferenceResponse(BaseModel):
    action_chunk: list[list[float]]
    client_send_ts_ms: float
    inference_ms: float
    decode_ms: float
    policy_revision: str
    policy_commit_sha: Optional[str] = None
    request_id: Optional[str] = None


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class State:
    adapter: Optional[PolicyAdapter] = None
    policy_type: str = POLICY_TYPE
    policy_repo: str = POLICY_REPO
    adapter_state: Any = None
    requests_served: int = 0


STATE = State()


def decode_camera_frame(frame: CameraFrame) -> np.ndarray:
    raw = base64.b64decode(frame.data_b64)
    if frame.encoding == "jpeg":
        return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"), dtype=np.uint8)
    if frame.encoding == "raw":
        return np.frombuffer(raw, dtype=np.uint8).reshape(
            frame.height, frame.width, 3
        ).copy()
    raise ValueError(f"Unknown encoding: {frame.encoding}")


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[adapter] Loading {STATE.policy_type}...")
    STATE.adapter = load_adapter(
        STATE.policy_type,
        repo=STATE.policy_repo or STATE.policy_type,
        revision=POLICY_REVISION,
        hf_token=HF_TOKEN,
    )
    print(f"[adapter] Ready: {STATE.adapter.name}@{STATE.adapter.revision}")
    yield


app = FastAPI(lifespan=lifespan, title="SO-101 Inference")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/healthz")
def healthz() -> dict:
    a = STATE.adapter
    return {
        "status": "ok" if a else "loading",
        "policy_type": STATE.policy_type,
        "policy_repo": STATE.policy_repo or STATE.policy_type,
        "policy_revision": a.revision if a else POLICY_REVISION,
        "policy_commit_sha": a.commit_sha if a else None,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "requests_served": STATE.requests_served,
        "available_adapters": sorted(ADAPTERS.keys()),
    }


@app.post("/warmup")
def warmup() -> dict:
    if STATE.adapter is None:
        raise HTTPException(503, "Adapter not loaded.")
    STATE.adapter.reset()
    STATE.adapter_state = None
    return {"status": "ok"}


@app.post("/infer", response_model=InferenceResponse)
def infer(req: InferenceRequest) -> InferenceResponse:
    if STATE.adapter is None:
        raise HTTPException(503, "Adapter not loaded.")

    request_id = req.request_id or str(uuid.uuid4())[:8]

    t_dec = time.monotonic()
    images = {f.name: decode_camera_frame(f) for f in req.cameras}
    decode_ms = (time.monotonic() - t_dec) * 1000

    if STATE.adapter.requires_language and not req.task:
        raise HTTPException(400, f"{STATE.adapter.name} requires a 'task' string.")

    joint_state = np.asarray(req.joint_state, dtype=np.float32)
    t_inf = time.monotonic()
    chunk, new_state = STATE.adapter.predict_chunk(
        joint_state=joint_state, images=images, task=req.task,
        state=STATE.adapter_state,
    )
    inference_ms = (time.monotonic() - t_inf) * 1000
    STATE.adapter_state = new_state
    STATE.requests_served += 1

    return InferenceResponse(
        action_chunk=chunk.tolist(),
        client_send_ts_ms=req.client_send_ts_ms,
        inference_ms=inference_ms,
        decode_ms=decode_ms,
        policy_revision=STATE.adapter.revision,
        policy_commit_sha=STATE.adapter.commit_sha,
        request_id=request_id,
    )


class SwapRequest(BaseModel):
    policy_type: str
    repo: Optional[str] = None
    revision: Optional[str] = "main"


@app.post("/policy/swap")
def swap(req: SwapRequest) -> dict:
    if req.policy_type not in ADAPTERS:
        raise HTTPException(400, f"Unknown policy_type. Available: {sorted(ADAPTERS.keys())}")
    repo = req.repo or req.policy_type
    new_adapter = load_adapter(
        req.policy_type, repo=repo, revision=req.revision or "main",
        hf_token=HF_TOKEN,
    )
    old = STATE.adapter
    STATE.adapter = new_adapter
    STATE.policy_type = req.policy_type
    STATE.policy_repo = repo
    STATE.adapter_state = None
    del old
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"ok": True, "policy_type": req.policy_type, "revision": new_adapter.revision}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("inference.server:app", host="0.0.0.0", port=8000)
