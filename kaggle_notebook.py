"""
Kaggle notebook source. Each `# %% --- CELL N ---` block goes into its own
Kaggle notebook cell.

This notebook is deliberately small — almost all logic is in training/train.py
and the YAML configs. The notebook just:
  1. Installs deps
  2. Sets up secrets
  3. Clones the project (so train.py and configs/ are available)
  4. Runs train.py
  5. Verifies the push to HF Hub

Before running:
  1. Settings → Accelerator: GPU T4 (single, not x2)
  2. Settings → Internet: ON
  3. Settings → Persistence: Variables and Files
  4. Add-ons → Secrets: HF_TOKEN (write scope), WANDB_API_KEY
  5. Push your local so101/ directory to a public GitHub repo, then edit
     CELL 3 below to clone from your repo.
"""

# %% --- CELL 1: install ---
# Lerobot 0.5.2 to match local. wandb is needed for telemetry.
!pip install -q lerobot==0.5.2 wandb pyyaml
!python -c "import lerobot, torch; print('lerobot', lerobot.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available())"


# %% --- CELL 2: secrets ---
import os
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
os.environ["HF_TOKEN"]      = secrets.get_secret("HF_TOKEN")
os.environ["WANDB_API_KEY"] = secrets.get_secret("WANDB_API_KEY")
# Optional: if you want to override the W&B entity from the YAML
# os.environ["WANDB_ENTITY"] = "your-entity"

# Log into HF and W&B so the subprocess inherits credentials.
!huggingface-cli login --token $HF_TOKEN --add-to-git-credential
!wandb login --relogin $WANDB_API_KEY


# %% --- CELL 3: clone the project ---
# Replace YOUR_USERNAME/so101 with the repo where you've pushed this project.
# This puts training/, configs/ etc. at /kaggle/working/so101.
!rm -rf /kaggle/working/so101
!git clone https://github.com/YOUR_USERNAME/so101.git /kaggle/working/so101
%cd /kaggle/working/so101


# %% --- CELL 4: train ---
# Edit --config to pick the architecture. configs/act.yaml is the week-1 default.
#
# To resume a killed session, add --resume. The script picks up from the last
# checkpoint in /kaggle/working/run.
#
# To override hyperparameters without editing the YAML, use --override:
#   --override training.steps=20000  (e.g. for a quick test)
!python training/train.py \
    --config configs/act.yaml \
    --platform kaggle


# %% --- CELL 5: verify push ---
# Confirm the policy actually landed on HF Hub.
import os, yaml
from huggingface_hub import HfApi

with open("/kaggle/working/so101/configs/act.yaml") as f:
    cfg = yaml.safe_load(f)

repo_id = cfg["policy"]["repo_id"]
api = HfApi(token=os.environ["HF_TOKEN"])
info = api.model_info(repo_id)
print(f"Policy on Hub: https://huggingface.co/{info.id}")
print("Files:")
for f in info.siblings[:20]:
    print(f"  {f.rfilename}")


# %% --- CELL 6 (optional): inspect the W&B run ---
# Pulls the most recent W&B run summary. If final loss is still trending
# down sharply, train for more steps next session.
import wandb
api_wandb = wandb.Api()
entity = os.environ.get("WANDB_ENTITY") or wandb.api.default_entity
runs = api_wandb.runs(f"{entity}/so101", order="-created_at", per_page=1)
if runs:
    r = runs[0]
    print(f"Run: {r.name}")
    print(f"  Final train loss: {r.summary.get('train/loss', 'n/a')}")
    print(f"  URL: {r.url}")
