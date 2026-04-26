#!/usr/bin/env bash
# RunPod bootstrap. SSH into your pod, then:
#   curl -sL https://raw.githubusercontent.com/YOUR_USERNAME/so101_remote/main/bootstraps/runpod_bootstrap.sh | bash
# Or copy this file over and `bash runpod_bootstrap.sh`.
#
# Set these environment variables BEFORE running (e.g. via RunPod's pod env vars):
#   POLICY_TYPE       — act | pi0 | random | echo (default: act)
#   POLICY_REPO       — required for act/pi0/openvla; ignored for random/echo
#   POLICY_REVISION   — default "main"
#   HF_TOKEN          — required
#   WANDB_API_KEY     — optional but recommended
#   NGROK_AUTHTOKEN   — optional; if unset, server is local-only on port 8000
#   OBS_LOGGING       — "on" / "off" (default off)
#   OBS_LOG_REPO      — required if OBS_LOGGING=on

set -euo pipefail

if [ -z "${POLICY_REPO:-}" ] || [ -z "${HF_TOKEN:-}" ]; then
    # Allow no POLICY_REPO if using random/echo
    if [ "${POLICY_TYPE:-act}" != "random" ] && [ "${POLICY_TYPE:-act}" != "echo" ]; then
        echo "POLICY_REPO and HF_TOKEN must be set (unless POLICY_TYPE=random|echo)." >&2
        exit 1
    fi
fi

REPO_DIR="${REPO_DIR:-/workspace/so101_remote}"

echo "==> Cloning / updating repo at $REPO_DIR"
if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/YOUR_USERNAME/so101_remote.git "$REPO_DIR"
else
    (cd "$REPO_DIR" && git pull)
fi
cd "$REPO_DIR"

echo "==> Installing requirements"
pip install -q --upgrade pip
pip install -q -r server/requirements.txt

echo "==> Sanity check"
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.cuda.is_available(), '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"

# Open Ngrok in the background if a token is present.
if [ -n "${NGROK_AUTHTOKEN:-}" ]; then
    echo "==> Opening Ngrok tunnel"
    pip install -q pyngrok
    python -c "
from pyngrok import ngrok, conf
import os, time
conf.get_default().auth_token = os.environ['NGROK_AUTHTOKEN']
for t in ngrok.get_tunnels():
    ngrok.disconnect(t.public_url)
tunnel = ngrok.connect(8000, 'http')
print('=' * 60)
print('Public URL:', tunnel.public_url)
print('=' * 60)
# Block to keep the tunnel alive while uvicorn runs in the foreground.
" &
    sleep 5
fi

echo "==> Starting server on :8000"
exec python -m server.server
