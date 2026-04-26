"""
Kaggle bootstrap. Open a new Kaggle notebook, paste each cell.

Setup before running:
  1. Settings → Accelerator: GPU T4 x2 (or P100)
  2. Settings → Internet: ON
  3. Settings → Persistence: Variables and Files
  4. Add-ons → Secrets:
       - HF_TOKEN
       - WANDB_API_KEY
       - NGROK_AUTHTOKEN
  5. Edit POLICY_REPO in cell 2.

Differences from Colab:
  - Secrets accessed via UserSecretsClient instead of userdata.
  - Kaggle's working dir is /kaggle/working, not /content.
  - Sessions are more aggressively killed; use Modal/Colab/RunPod for long runs.
"""

# %% --- CELL 1: clone repo + install ---
!git clone https://github.com/YOUR_USERNAME/so101_remote.git /kaggle/working/so101_remote || (cd /kaggle/working/so101_remote && git pull)
%cd /kaggle/working/so101_remote
!pip install -q -r server/requirements.txt
!python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"


# %% --- CELL 2: config ---
import os
from kaggle_secrets import UserSecretsClient
secrets = UserSecretsClient()

# EDIT THIS ──────────────────────────────────────────────
os.environ["POLICY_TYPE"]     = "act"              # act | pi0 | random | echo
os.environ["POLICY_REPO"]     = "your-hf-username/so101-act-pick-v0"
os.environ["POLICY_REVISION"] = "main"
os.environ["OBS_LOGGING"]     = "off"
os.environ["OBS_LOG_REPO"]    = "your-hf-username/so101-remote-obs"
os.environ["WANDB_PROJECT"]   = "so101-remote-inf"
# ────────────────────────────────────────────────────────

os.environ["HF_TOKEN"]        = secrets.get_secret("HF_TOKEN")
os.environ["WANDB_API_KEY"]   = secrets.get_secret("WANDB_API_KEY")
os.environ["NGROK_AUTHTOKEN"] = secrets.get_secret("NGROK_AUTHTOKEN")
print("Config loaded.")


# %% --- CELL 3: open Ngrok tunnel ---
from pyngrok import ngrok, conf
conf.get_default().auth_token = os.environ["NGROK_AUTHTOKEN"]
for t in ngrok.get_tunnels():
    ngrok.disconnect(t.public_url)
tunnel = ngrok.connect(8000, "http")
print("=" * 60)
print(f"Public URL: {tunnel.public_url}")
print("=" * 60)


# %% --- CELL 4: start server ---
# NOTE: Kaggle will eventually kill the kernel for "inactivity" even with
# the server running. Expect to restart this notebook every few hours.
import uvicorn
uvicorn.run("server.server:app", host="0.0.0.0", port=8000, log_level="info")
