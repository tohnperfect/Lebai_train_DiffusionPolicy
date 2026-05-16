"""Open the locally-converted LeRobot dataset and print summary stats.

Run from a fresh shell after `convert_local.py` finishes:

    python verify_local.py

This is intentionally a separate process from the converter — lerobot 0.5.1
has an async meta-parquet writer whose flush can lag the end of conversion,
producing a misleading 'Parquet magic bytes not found' error if you try to
re-open the dataset in the same Python process. See PRD_diffusion_policy.md
§10 (pitfall #2) for the full story. Do NOT re-add an in-process verify to
convert_local.py.
"""

import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
RESULT_ROOT = REPO_ROOT / "result"

# Must match convert_local.py — keep these in sync if you change the action mode.
ACTION_MODE = "relative"
ACTION_DELTA_SCALE = 100.0
if ACTION_MODE == "relative":
    REPO_ID = f"local/lebai_duck_pick_delta_x{int(ACTION_DELTA_SCALE)}"
else:
    REPO_ID = "local/lebai_duck_pick"

os.environ["HF_LEROBOT_HOME"] = str(RESULT_ROOT)

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


def main():
    ds = LeRobotDataset(REPO_ID)
    print(f"Loaded: {RESULT_ROOT / REPO_ID}")
    print(f"  Episodes: {ds.num_episodes}")
    print(f"  Frames:   {ds.num_frames}")
    print(f"  FPS:      {ds.fps}")
    print(f"  Features: {list(ds.features.keys())}")
    print()

    sample = ds[0]
    state = sample["observation.state"].numpy()
    action = sample["action"].numpy()
    img = sample["observation.images.base"]
    print("First frame:")
    print(f"  task:   {sample['task']!r}")
    print(f"  state:  shape={state.shape}  values={state.round(3).tolist()}")
    print(f"  action: shape={action.shape}  values={action.round(3).tolist()}")
    print(f"  image:  shape={tuple(img.shape)}  dtype={img.dtype}")

    # Spot-check action magnitudes (PRD §16): in relative mode with x100 scale
    # the per-tick joint deltas should land in ~[-2, 2] with typical |a| ~ 0.5-1.0.
    # Smaller than that (e.g. ~0.005) means the scale was not applied.
    if ds.num_frames > 600:
        idx = 600
        a = ds[idx]["action"].numpy()
        max_abs = float(np.abs(a[:6]).max())
        print()
        print(f"Spot-check ds[{idx}]['action'][:6]:")
        print(f"  values  = {a[:6].round(4).tolist()}")
        print(f"  max|.|  = {max_abs:.4f}")
        if ACTION_MODE == "relative":
            if max_abs < 0.05:
                print("  WARNING: magnitudes look too small for a scaled delta — "
                      f"is ACTION_DELTA_SCALE={ACTION_DELTA_SCALE} actually applied?")
            elif max_abs > 5.0:
                print(f"  WARNING: magnitudes look too large — sanity-check ACTION_DELTA_SCALE.")
            else:
                print("  Scaling looks reasonable.")

    print()
    print("Dataset OK.")


if __name__ == "__main__":
    main()
