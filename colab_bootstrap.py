"""
Colab bootstrap. Open a new Colab notebook with T4 GPU, paste each cell.

Or: convert this file to .ipynb with `jupytext --to ipynb colab_bootstrap.py`
and upload directly.

Setup before running:
  1. Runtime → Change runtime type → T4 GPU
  2. Add Colab Secrets (left sidebar, key icon):
       - HF_TOKEN
       - WANDB_API_KEY
       - NGROK_AUTHTOKEN  (free at ngrok.com)
  3. Edit the POLICY_REPO line in cell 2 to your HF policy repo.
"""

# %% --- CELL 1: clone repo + install ---
!git clone https://github.com/YOUR_USERNAME/so101_remote.git /content/so101_remote || (cd /content/so101_remote && git pull)
%cd /content/so101_remote
!pip install -q -r server/requirements.txt
!python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"


# %% --- CELL 2: config ---
import os
from google.colab import userdata

# EDIT THIS ──────────────────────────────────────────────
os.environ["POLICY_TYPE"]     = "act"              # act | pi0 | random | echo
os.environ["POLICY_REPO"]     = "your-hf-username/so101-act-pick-v0"
os.environ["POLICY_REVISION"] = "main"             # or "step-50000"
os.environ["OBS_LOGGING"]     = "off"              # "on" to enable
os.environ["OBS_LOG_REPO"]    = "your-hf-username/so101-remote-obs"
os.environ["WANDB_PROJECT"]   = "so101-remote-inf"
# ────────────────────────────────────────────────────────

os.environ["HF_TOKEN"]         = userdata.get("HF_TOKEN")
os.environ["WANDB_API_KEY"]    = userdata.get("WANDB_API_KEY")
os.environ["NGROK_AUTHTOKEN"]  = userdata.get("NGROK_AUTHTOKEN")
print("Config loaded.")


# %% --- CELL 3: open Ngrok tunnel ---
# This must run BEFORE the server starts, because Colab notebooks block on
# uvicorn.run(). We open the tunnel first, print the URL, then start the
# server in a background thread.
from pyngrok import ngrok, conf
conf.get_default().auth_token = os.environ["NGROK_AUTHTOKEN"]

# Kill any leftover tunnels from a previous run.
for t in ngrok.get_tunnels():
    ngrok.disconnect(t.public_url)

tunnel = ngrok.connect(8000, "http")
print("=" * 60)
print(f"Public URL: {tunnel.public_url}")
print("=" * 60)
print("Copy this into SERVER_URL in your Mac's .env")


# %% --- CELL 4: start server ---
# Runs uvicorn in this notebook process. The cell will not return — that's
# the point; killing the cell stops the server. Use Runtime → Interrupt
# execution to stop cleanly.
import uvicorn
uvicorn.run("server.server:app", host="0.0.0.0", port=8000, log_level="info")
