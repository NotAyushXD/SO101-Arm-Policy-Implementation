"""
Quick sanity check on a LeRobot dataset before training.

Reports:
  - Episode count and total frames
  - Per-episode duration distribution
  - How many frames at the start of each episode are "stationary"
    (action barely changes from initial position)
  - Action range per joint (catches calibration issues)
  - Whether all expected camera streams are present

Run this on Kaggle BEFORE training, or locally if you have the dataset cached.

Usage:
    python scripts/check_dataset.py --repo-id NotAyushXD/so101-arm-pickupbowl-50episodes
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np
from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--repo-id", type=str, required=True,
                   help="HF dataset repo id, e.g. user/dataset-name")
    p.add_argument("--motion-threshold-deg", type=float, default=1.0,
                   help="Joint angle change (degrees) considered 'motion'.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"Loading {args.repo_id}...")
    ds = load_dataset(args.repo_id, split="train")
    print(f"Loaded {len(ds):,} frames.")
    print()

    # Group frames by episode.
    by_episode: dict[int, list[int]] = defaultdict(list)
    for i, ep_idx in enumerate(ds["episode_index"]):
        by_episode[ep_idx].append(i)

    n_episodes = len(by_episode)
    print(f"Episodes: {n_episodes}")
    print(f"Total frames: {len(ds):,}")
    print(f"Mean frames per episode: {len(ds) / n_episodes:.1f}")
    print(f"Approx duration: {len(ds) / 30 / 60:.1f} minutes at 30 fps")
    print()

    # -- episode lengths --
    ep_lengths = np.array([len(v) for v in by_episode.values()])
    print(f"Episode length: min={ep_lengths.min()}, "
          f"median={int(np.median(ep_lengths))}, max={ep_lengths.max()}")
    print()

    # -- static-frames check --
    print(f"Frames before first significant motion "
          f"(>{args.motion_threshold_deg}° change in any joint):")
    motion_starts = []
    for ep_idx, frame_idxs in by_episode.items():
        actions = np.array([ds[i]["action"] for i in frame_idxs])
        deltas = np.linalg.norm(actions - actions[0], axis=1)
        mask = deltas > args.motion_threshold_deg
        first_motion = np.argmax(mask) if mask.any() else len(actions)
        motion_starts.append(first_motion)
    motion_starts = np.array(motion_starts)
    print(f"  median: {int(np.median(motion_starts))} frames "
          f"({np.median(motion_starts)/30:.2f}s)")
    print(f"  mean:   {motion_starts.mean():.1f} frames "
          f"({motion_starts.mean()/30:.2f}s)")
    print(f"  max:    {motion_starts.max()} frames "
          f"({motion_starts.max()/30:.2f}s)")
    if np.median(motion_starts) > 60:
        print("  ⚠️  >2s of static frames per episode is high. Consider "
              "re-recording with more decisive starts, or trimming.")
    elif np.median(motion_starts) > 30:
        print("  ℹ️  ~1-2s of static frames per episode. Workable; "
              "training will be slightly slower than ideal.")
    else:
        print("  ✅ Static-frame budget looks healthy.")
    print()

    # -- action range per joint --
    print("Action range per joint (degrees):")
    actions_all = np.array(ds["action"])
    joint_names = ["shoulder_pan", "shoulder_lift", "elbow_flex",
                   "wrist_flex", "wrist_roll", "gripper"]
    for i, name in enumerate(joint_names):
        col = actions_all[:, i]
        print(f"  {name:14s}: min={col.min():7.2f}, max={col.max():7.2f}, "
              f"range={col.max()-col.min():6.2f}")
    print()

    # -- camera presence --
    expected_cams = ["observation.images.wrist", "observation.images.front"]
    features = ds.features
    print("Camera streams:")
    for cam in expected_cams:
        present = cam in features
        print(f"  {cam}: {'✅ present' if present else '❌ MISSING'}")

    print()
    print("Done. If everything above looks reasonable, you're ready to train.")


if __name__ == "__main__":
    main()
