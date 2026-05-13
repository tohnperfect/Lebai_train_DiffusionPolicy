"""Convert raw save_state_img.py logs into a LeRobot dataset, locally.

Reads from ./Data/, writes to ./result/local/lebai_duck_pick/.
Run from the repo root:

    python convert_local.py

Then verify in a fresh shell:

    python verify_local.py

(verify must be a separate process — see the trailing note printed at the end
of conversion, and PRD_diffusion_policy.md §10.)
"""

import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
LOG_ROOT = REPO_ROOT / "Data"
RESULT_ROOT = REPO_ROOT / "result"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)

# Must be set before importing lerobot so the dataset lands in ./result/
os.environ["HF_LEROBOT_HOME"] = str(RESULT_ROOT)

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset


# Configuration --------------------------------------------------------------

RUN_NUMBER = None                       # int to pick one log<NNNN>, None to merge all
REPO_ID = "local/lebai_duck_pick"       # same name as the ACT dataset — only the policy differs
INCLUDE_WRIST = True                    # auto-disabled if no wrist data is present
INCLUDE_GRIPPER = True
FPS = 10
IMG_H, IMG_W = 480, 640                 # camera service resolution


# Helpers --------------------------------------------------------------------

def build_state(row):
    s = [float(row[f"jp{i}"]) for i in range(6)]
    if INCLUDE_GRIPPER:
        amp = row.get("claw_amplitude", 0.0)
        s.append(float(amp) if pd.notna(amp) else 0.0)
    return np.array(s, dtype=np.float32)


def build_action(row, next_row):
    tgt = [row.get(f"tgt_jp{i}") for i in range(6)]
    if any(pd.isna(v) for v in tgt):
        tgt = [float(next_row[f"jp{i}"]) for i in range(6)]
    else:
        tgt = [float(v) for v in tgt]
    if INCLUDE_GRIPPER:
        # next-frame gripper trick — see PRD §5
        amp = next_row.get("claw_amplitude", row.get("claw_amplitude", 0.0))
        tgt.append(float(amp) if pd.notna(amp) else 0.0)
    return np.array(tgt, dtype=np.float32)


def load_rgb(run_dir, rel_path):
    full = Path(run_dir) / rel_path
    bgr = cv2.imread(str(full), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {full}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def main():
    # 1. Load CSV index(es)
    if RUN_NUMBER is not None:
        csv_paths = [LOG_ROOT / f"log{RUN_NUMBER:04d}.csv"]
    else:
        csv_paths = sorted(LOG_ROOT.glob("log[0-9][0-9][0-9][0-9].csv"))
    if not csv_paths:
        sys.exit(f"No log CSVs found under {LOG_ROOT}")
    print(f"CSVs to convert: {[p.name for p in csv_paths]}")

    dfs = []
    for p in csv_paths:
        d = pd.read_csv(p)
        d["__run_dir__"] = str(LOG_ROOT / p.stem)
        dfs.append(d)
    df = pd.concat(dfs, ignore_index=True)
    print(f"Total rows: {len(df)}")

    # 2. Wrist detection — pandas reads empty CSV cells as NaN, and astype(str)
    # makes those into the literal three-character string "nan". Filter both.
    wrist_strs = df["wrist"].fillna("").astype(str).str.strip()
    all_have_wrist = ((wrist_strs != "") & (wrist_strs.str.lower() != "nan")).all()
    has_wrist_data = INCLUDE_WRIST and all_have_wrist
    print(f"has_wrist_data = {has_wrist_data}")

    # 3. Build feature schema
    state_dim = 7 if INCLUDE_GRIPPER else 6
    action_dim = state_dim
    features = {
        "observation.images.base": {
            "dtype": "image",
            "shape": (IMG_H, IMG_W, 3),
            "names": ["height", "width", "channel"],
        },
        "observation.state": {"dtype": "float32", "shape": (state_dim,), "names": ["state"]},
        "action":            {"dtype": "float32", "shape": (action_dim,), "names": ["action"]},
    }
    if has_wrist_data:
        features["observation.images.wrist"] = {
            "dtype": "image",
            "shape": (IMG_H, IMG_W, 3),
            "names": ["height", "width", "channel"],
        }

    # 4. Wipe any previous version of this dataset and create fresh
    out_path = RESULT_ROOT / REPO_ID
    if out_path.exists():
        print(f"Removing existing dataset at {out_path}")
        shutil.rmtree(out_path)

    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        robot_type="lebai_lm3",
        fps=FPS,
        features=features,
        image_writer_threads=4,
        image_writer_processes=2,
    )
    print(f"Created empty dataset at {out_path}")
    print(f"  state_dim={state_dim}  action_dim={action_dim}  wrist={has_wrist_data}")

    # 5. Conversion loop
    total_frames = 0
    total_episodes = 0
    groups = sorted(df.groupby(["__run_dir__", "episode"]), key=lambda kv: kv[0])

    for (run_dir, ep_idx), ep_df in groups:
        ep_df = ep_df.sort_values("frame").reset_index(drop=True)
        task = str(ep_df["task"].iloc[0])
        src_name = f"{Path(run_dir).name}/ep{ep_idx:03d}"

        if not task or task == "nan":
            print(f"  skip (empty task): {src_name}")
            continue
        if len(ep_df) < 5:
            print(f"  skip (only {len(ep_df)} frames): {src_name}")
            continue

        desc = f"{src_name} [{task[:30]}]"
        for i in tqdm(range(len(ep_df)), desc=desc, leave=False):
            row = ep_df.iloc[i]
            next_row = ep_df.iloc[i + 1] if i + 1 < len(ep_df) else row
            frame = {
                "observation.images.base": load_rgb(row["__run_dir__"], row["color"]),
                "observation.state": build_state(row),
                "action": build_action(row, next_row),
                "task": task,
            }
            if has_wrist_data and isinstance(row["wrist"], str) and row["wrist"]:
                frame["observation.images.wrist"] = load_rgb(row["__run_dir__"], row["wrist"])
            dataset.add_frame(frame)

        dataset.save_episode()
        total_episodes += 1
        total_frames += len(ep_df)
        print(f"  saved {src_name}: {len(ep_df)} frames")

    print(f"\nDone. {total_episodes} episodes, {total_frames} frames.")
    print(f"Dataset written to {out_path}")
    print()
    print("NOTE: Do not call LeRobotDataset(REPO_ID) in the same Python process —")
    print("lerobot 0.5.1's async meta-parquet writer can lag the read by a beat and")
    print("you'll see a misleading 'Parquet magic bytes not found' error. Run:")
    print()
    print("    python verify_local.py")
    print()
    print("from a fresh shell to confirm the dataset is readable.")


if __name__ == "__main__":
    main()
