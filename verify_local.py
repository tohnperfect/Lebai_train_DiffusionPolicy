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

REPO_ROOT = Path(__file__).resolve().parent
RESULT_ROOT = REPO_ROOT / "result"
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
    print()
    print("Dataset OK.")


if __name__ == "__main__":
    main()
