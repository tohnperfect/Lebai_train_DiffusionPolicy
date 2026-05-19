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
GRIPPER_SCALE = 0.01
INCLUDE_GRIPPER = True
if ACTION_MODE == "relative":
    if not INCLUDE_GRIPPER or GRIPPER_SCALE == 1.0:
        REPO_ID = f"local/lebai_duck_pick_delta_x{int(ACTION_DELTA_SCALE)}"
    else:
        REPO_ID = (f"local/lebai_duck_pick_delta_x{int(ACTION_DELTA_SCALE)}"
                   f"_g{int(round(1.0 / GRIPPER_SCALE))}")
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

    # Spot-check action magnitudes (PRD §16). In relative mode with x100 joint
    # scale, per-tick joint deltas land in ~[-2, 2] with typical |a| ~ 0.5-1.0.
    # With GRIPPER_SCALE=0.01, the gripper (dim 6) sits in [0, 1] (mean ~0.8).
    if ds.num_frames > 600:
        idx = 600
        a = ds[idx]["action"].numpy()
        joints = a[:6]
        max_abs_joint = float(np.abs(joints).max())
        print()
        print(f"Spot-check ds[{idx}]['action']:")
        print(f"  joints = {joints.round(4).tolist()}    |max|={max_abs_joint:.4f}")
        if ACTION_MODE == "relative":
            if max_abs_joint < 0.05:
                print("  WARNING: joint magnitudes look too small for a scaled delta — "
                      f"is ACTION_DELTA_SCALE={ACTION_DELTA_SCALE} actually applied?")
            elif max_abs_joint > 5.0:
                print("  WARNING: joint magnitudes look too large — sanity-check ACTION_DELTA_SCALE.")
            else:
                print("  Joint scaling looks reasonable.")

        if a.shape[0] >= 7:
            grip = float(a[6])
            print(f"  gripper = {grip:.4f}")
            if ACTION_MODE == "relative":
                expected_max = 100.0 * GRIPPER_SCALE * 1.05
                if abs(grip) > expected_max:
                    print(f"  WARNING: gripper {grip:.3f} exceeds expected scaled range "
                          f"[-{expected_max:.2f}, {expected_max:.2f}] — is GRIPPER_SCALE={GRIPPER_SCALE} actually applied?")
                elif 0.0 <= grip <= 1.05:
                    print(f"  Gripper scaling looks reasonable (scaled to ~[0, 1]).")
                else:
                    print(f"  WARNING: gripper magnitude looks off — check GRIPPER_SCALE={GRIPPER_SCALE}.")

    print()
    print("Dataset OK.")


if __name__ == "__main__":
    main()
