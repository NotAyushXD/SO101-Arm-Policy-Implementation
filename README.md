# SO-101 Remote Inference

Run ACT inference on a remote GPU (Colab / Kaggle / RunPod), with the SO-101
arm on your Mac M2. Designed as a learning exercise: you can A/B test every
component independently via flags.

## Architecture

```
┌──────────────────────┐    Ngrok HTTPS     ┌────────────────────────┐
│  Mac M2 (client)     │ ─── POST /infer ──▶│  GPU host (server)     │
│                      │                     │                         │
│  - SO-101 arm        │ ◀── action chunk ──│  - FastAPI              │
│  - cameras           │                     │  - ACT policy on GPU   │
│  - control loop      │                     │  - W&B telemetry       │
│  - exec modes        │                     │  - Web UI              │
│  - chunk smoothing   │                     │  - obs logging buffer  │
└──────────────────────┘                     └────────────────────────┘
       Chennai                                Colab / Kaggle / RunPod
```

## Flags you'll be testing

Every "I want to A/B test this" thing is a flag, set in `.env` on the client
and/or via env vars on the server.

### Client-side (Mac M2)

| Flag                 | Values            | Default     | What it does |
|----------------------|-------------------|-------------|--------------|
| `EXEC_MODE`          | `shadow` / `visualize` / `live_slow` / `live` | `shadow` | Progression of safety. Start in shadow, end in live. |
| `LIVE_SLOW_FACTOR`   | `0.0..1.0`        | `0.3`       | Velocity scaling when EXEC_MODE=live_slow. |
| `CHUNK_SMOOTHING`    | `on` / `off`      | `on`        | Temporal ensemble across chunk boundaries. Toggle to see the difference. |
| `CHUNK_EXECUTE_LEN`  | int               | `30`        | How many actions per chunk to execute before requesting a new one. |
| `JPEG_ENCODE`        | `on` / `off`      | `on`        | If off, sends raw RGB. Auto-throttles fps to 5 with a warning. |
| `JPEG_QUALITY`       | `1..100`          | `85`        | When JPEG_ENCODE=on. |
| `MAX_FPS`            | int               | `30`        | Control loop rate cap. |
| `STALE_TIMEOUT_MS`   | int               | `500`       | Halt arm if next chunk doesn't arrive within this. |

### Server-side (Colab / Kaggle / RunPod)

| Flag                 | Values            | Default     | What it does |
|----------------------|-------------------|-------------|--------------|
| `POLICY_REPO`        | HF repo id        | required    | Where to pull the checkpoint from. |
| `POLICY_REVISION`    | branch/tag/SHA    | `main`      | Pin a specific checkpoint for ablation runs. |
| `OBS_LOGGING`        | `on` / `off`      | `off`       | Buffer observations + actions in RAM, push to HF at session end. |
| `OBS_LOG_REPO`       | HF dataset repo id| optional    | Required if OBS_LOGGING=on. |
| `WANDB_ENABLE`       | `on` / `off`      | `on`        | Push latency telemetry to W&B. |

## Execution modes (the four-way toggle)

Built so you can de-risk progressively. Run one or two real episodes per mode
before stepping up to the next.

| Mode          | Arm moves? | Predictions logged? | UI overlay? | When to use |
|---------------|------------|----------------------|-------------|-------------|
| `shadow`      | No         | Yes                  | No          | First connection test. Validates network + protocol only. |
| `visualize`   | No         | Yes                  | Yes (live trajectory) | Eyeball if predictions point at the object. **Use this before any live mode.** |
| `live_slow`   | Yes (slow) | Yes                  | Yes         | First time the arm actually moves. Hand on e-stop. |
| `live`        | Yes        | Yes                  | Yes         | Real evaluation. Only after live_slow looks correct. |

**Important caveat about shadow mode:** because the arm doesn't move,
observations from step N+1 are essentially the same as step N. The policy will
drift out of distribution after ~5 steps and the predictions stop being
informative. Shadow mode tells you about *plumbing*, not *policy quality*.
For policy quality, use `visualize` mode — it shows the predicted trajectory
over the live camera feed so you can sanity-check what the model wants to do.

## Quick start (Colab — recommended for first session)

1. Train ACT on Kaggle per the previous README, get a HF policy repo.
2. Open `bootstraps/colab_bootstrap.ipynb` in Colab. Set runtime to T4 GPU.
3. Add HF_TOKEN, WANDB_API_KEY, NGROK_AUTHTOKEN as Colab Secrets.
4. Run all cells. Last cell prints the public Ngrok URL.
5. On Mac: copy `.env.example` → `.env`, paste the Ngrok URL into `SERVER_URL`.
6. `python client/run_remote.py --exec-mode shadow --num-trials 1`
7. Watch the W&B run for latency stats. Confirm RTT < 500ms.
8. Step up to `--exec-mode visualize`, then `live_slow`, then `live`.

For Kaggle: use `bootstraps/kaggle_bootstrap.py` instead of the Colab notebook.
For RunPod: use `bootstraps/runpod_bootstrap.sh`.

The Python server code in `server/` is **identical across all three**.

## File layout

```
so101_remote/
├── README.md                    (you are here)
├── shared/
│   ├── protocol.py              (request/response schemas — used by both sides)
│   └── encoding.py              (JPEG / raw image encode-decode)
├── server/
│   ├── server.py                (FastAPI app + inference)
│   ├── obs_logger.py            (in-memory buffer → HF dataset push)
│   ├── ui.py                    (web UI HTML + websocket for live updates)
│   ├── policies/                (swappable policy adapters — see below)
│   │   ├── base.py              (PolicyAdapter protocol + registry)
│   │   ├── act_adapter.py       (ACT — current target)
│   │   ├── pi0_adapter.py       (π₀ — week-2 stub)
│   │   ├── random_adapter.py    (random walk for pipeline debugging)
│   │   └── echo_adapter.py      (echoes joint state — safe for live testing)
│   └── requirements.txt
├── client/
│   ├── run_remote.py            (main control loop on M2)
│   ├── chunk_buffer.py          (chunk caching + temporal ensembling)
│   ├── exec_modes.py            (shadow / visualize / live_slow / live)
│   └── requirements.txt
├── bootstraps/
│   ├── colab_bootstrap.py       (paste-and-run cells)
│   ├── kaggle_bootstrap.py      (notebook source for Kaggle)
│   └── runpod_bootstrap.sh      (SSH-and-run for RunPod)
├── .env.example
└── tools/
    └── benchmark_rtt.py         (run before connecting the robot)
```

## Swappable policies

The server doesn't hard-code ACT. It loads an adapter selected by the
`POLICY_TYPE` env var. Built-in adapters:

| `POLICY_TYPE` | Needs `POLICY_REPO`? | What it does | When to use |
|--------------|----------------------|--------------|-------------|
| `act`        | Yes                  | LeRobot ACT checkpoint | Week-1 target |
| `pi0`        | Yes                  | π₀ VLA (stub — not yet implemented) | Week-2 evaluation |
| `random`     | No                   | Random walk around current joint state | Pipeline debugging without a model |
| `echo`       | No                   | Returns current joint state unchanged | Safe live-mode comms test |

Switch policies in three ways:

1. **At server start:** set `POLICY_TYPE` env var.
2. **At runtime via the web UI:** "Swap policy type" dropdown.
3. **At runtime via API:** `POST /policy/swap {"policy_type": "act", "repo": "user/...", "revision": "main"}`.

Switch revisions of the same policy:

- **Web UI:** "Reload policy with revision" input.
- **API:** `POST /policy/reload {"revision": "step-50000"}`.

Both paths free GPU memory from the old policy before loading the new one.

### De-risking ladder for first session

Recommended sequence for the first time you connect the M2 to the server:

```
1. POLICY_TYPE=random   → exec_mode=shadow      → confirms pipeline alive
2. POLICY_TYPE=echo     → exec_mode=live        → arm holds position; comms verified
3. POLICY_TYPE=act      → exec_mode=shadow      → real model, no risk
4. POLICY_TYPE=act      → exec_mode=live_slow   → real model, scaled velocity
5. POLICY_TYPE=act      → exec_mode=live        → full eval
```

You can do steps 1, 2, and 3 by swapping via the UI without restarting the
server. Step 4 onwards is just exec-mode flag changes on the M2 client.

### Adding a new policy

To add OpenVLA, residual, or a custom backbone:

1. Create `server/policies/<name>_adapter.py`.
2. Implement the `PolicyAdapter` protocol — five attributes (`name`,
   `revision`, `commit_sha`, `chunk_len`, `requires_language`, `action_dim`)
   and two methods (`predict_chunk`, `reset`). See `act_adapter.py` for a
   reference implementation.
3. Decorate the class with `@register("<name>")`.
4. Add `_safe_import("server.policies.<name>_adapter")` to
   `server/policies/__init__.py`.
5. Set `POLICY_TYPE=<name>` and you're done. No changes to `server.py`,
   `client/run_remote.py`, the web UI, or the wire protocol.

The protocol passes an opaque `state` parameter to `predict_chunk` and
expects you to return updated state. Stateless adapters like ACT just
return `None`. Stateful ones (KV-caching VLAs, residual policies that
need history) can stash whatever they want there.


