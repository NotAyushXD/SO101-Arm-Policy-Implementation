# SO-101 — Unified Training and Inference

End-to-end project for the SO-101 RL roadmap. Train any supported policy
architecture on either Mac M2 or Kaggle, then run inference either locally on
the M2 or remotely via a hosted server. All artifacts flow through the
Hugging Face Hub.

## Project shape

```
so101/
├── configs/              # YAML configs — one per architecture
│   ├── _base.yaml        # shared defaults
│   ├── act.yaml          # ACT (week 1)
│   └── pi0.yaml          # π₀ stub (week 2)
├── training/
│   ├── train.py          # Unified training entrypoint (Mac OR Kaggle)
│   └── kaggle_notebook.py # Kaggle notebook source
├── inference/
│   ├── server.py         # FastAPI server (for remote inference mode)
│   └── policies/         # Policy adapter registry
├── client/
│   ├── run_local.py      # Mac M2 — policy runs on M2 (use for ACT)
│   ├── run_remote.py     # Mac M2 — policy runs on remote server (use for VLAs)
│   └── chunk_buffer.py   # Shared between both clients
├── data/
│   ├── record_demos.py
│   ├── pull_policy.py
│   └── check_dataset.py
└── requirements/
    ├── local.txt
    └── kaggle.txt
```

## What runs where

| Step | Runs on | Why |
|---|---|---|
| Calibrate, record demos | Mac M2 | Robot is plugged in here |
| Train ACT | Kaggle (T4) | ~60 min, vs 6+ hours on M2 |
| Train ACT (smoke test) | Mac M2 | Verify pipeline before committing to Kaggle |
| Train π₀, OpenVLA | Kaggle (or RunPod) | Too big for M2 |
| Inference for ACT | Mac M2 (local) | M2 is fast enough; zero network latency |
| Inference for π₀, OpenVLA | Remote server | Too big for M2 |
| Robot control loop | Mac M2 | Servo communication is real-time |

## Adding a new architecture

The whole project was designed for this. To add (say) OpenVLA in week 6:

1. Create `configs/openvla.yaml` with hyperparameters.
2. Add `inference/policies/openvla_adapter.py` (registry pattern — see existing adapters).
3. Train: `python training/train.py --config configs/openvla.yaml --platform kaggle`
4. Done. No edits to `train.py`, the client, the server, or any other config.

If LeRobot 0.5.2 doesn't natively support the architecture, you'd add it as
a custom policy. That's a bigger lift and out of scope for this README.

## Quick start

### One-time setup (Mac M2)

```bash
cd so101
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements/local.txt
cp .env.example .env
# Fill in HF_TOKEN, WANDB_API_KEY, SO101_PORT, etc.
```

### Train (Mac smoke test, optional)

```bash
# Quick 1000-step sanity check on the M2 — verify the pipeline before Kaggle
python training/train.py \
    --config configs/act.yaml \
    --platform mac \
    --override steps=1000
```

### Train (Kaggle, the real run)

Open Kaggle → New Notebook → paste cells from `training/kaggle_notebook.py`.
Set Kaggle secrets (HF_TOKEN, WANDB_API_KEY) and Run All. ~60 minutes for ACT.

### Inference, local (recommended for ACT)

```bash
python data/pull_policy.py --repo-id NotAyushXD/so101-act-pick-bowl-v0
python client/run_local.py \
    --policy-path ./checkpoints/so101-act-pick-bowl-v0 \
    --num-trials 20
```

### Inference, remote (for VLAs in week 6+)

Start the server on Kaggle/Colab/RunPod via `inference/server.py` plus the
appropriate bootstrap. Then on Mac:

```bash
python client/run_remote.py \
    --server-url https://abc-123.ngrok.io \
    --num-trials 20
```

## Pinned versions

LeRobot 0.5.2 across both platforms. `requirements/local.txt` and
`requirements/kaggle.txt` enforce this. If LeRobot version drifts and the
training/inference flags break, this is the first thing to check.

## Honest limitations

- Mac training uses MPS, which is slow. Use only for smoke tests.
- Kaggle sessions auto-kill after 9-12 hours. Use `--resume=true` to recover.
- Remote inference adds 200-400ms RTT from Chennai. Not noticeable for VLAs
  (because their inference time is also ~100ms), very noticeable for ACT.
- The π₀ adapter is a stub. Implement it before week 2.
