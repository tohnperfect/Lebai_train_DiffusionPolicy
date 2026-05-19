"""Convert raw save_state_img.py logs into a LeRobot dataset, locally.

Reads from ./Data/, writes to ./result/<REPO_ID>/.
Run from the repo root:

    python convert_local.py

Then verify in a fresh shell:

    python verify_local.py

(verify must be a separate process — see the trailing note printed at the end
of conversion, and PRD §10.)

Action representation
---------------------
By default the action is a **scaled per-tick joint delta** + **scaled gripper amplitude** (PRD §5, §14):

    action[:6] = (next_jp - jp) * ACTION_DELTA_SCALE       # rad, scaled (default 100x)
    action[6]  =  next_claw_amplitude * GRIPPER_SCALE      # 0-100 -> 0-1 (default 0.01x)

The raw joint deltas at 10 Hz are ~0.005-0.01 rad — too small for the network
to fit quickly. Multiplying by 100 brings target magnitudes to ~0.5-1.0.

The raw gripper amplitude is on a completely different scale ([0, 100], typical
mean ~80) than the scaled joint deltas (~±1). DP's per-dim normalization can
miscalibrate when one dim is two orders of magnitude larger than the others —
in practice this presents as the policy outputting near-zero gripper amplitudes
at inference (see PRD §14 pitfall on gripper collapse). Multiplying the gripper
by 0.01 puts it in roughly the same range as the scaled joint deltas.

Both scales are encoded in REPO_ID so different settings can coexist on disk
(and inference refuses to run with a mismatched scale). Inference undoes both
in `resolve_targets`.

Set ACTION_MODE = "absolute" to fall back to commanding `tgt_jp*` directly
(the old convention, kept for compatibility). Absolute mode does NOT apply
GRIPPER_SCALE — the gripper stays in [0, 100].
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
ACTION_MODE = "relative"                # "relative" (scaled delta) or "absolute"
ACTION_DELTA_SCALE = 100.0              # joint-delta multiplier; only used when ACTION_MODE == "relative"
GRIPPER_SCALE = 0.01                    # gripper-amplitude multiplier; ignored in absolute mode
INCLUDE_WRIST = True                    # auto-disabled if no wrist data is present
INCLUDE_GRIPPER = True
FPS = 10
IMG_H, IMG_W = 480, 640                 # camera service resolution

# Image writer tuning (PRD §14 pitfall): 8 GB M1 OOMs with the lerobot defaults.
# threads=2, processes=0 is the sweet spot — threads share memory; processes
# double it. Do NOT set both to 0 — that disables image writing entirely in
# lerobot 0.5.1 (only the first episode's directory gets created, rest silently dropped).
IMG_WRITER_THREADS   = 2
IMG_WRITER_PROCESSES = 0
DRAIN_EVERY          = 200              # periodic queue drain inside the per-frame loop


if ACTION_MODE == "relative":
    # The "g100" suffix is the *divisor* (1/GRIPPER_SCALE) — clearer to read
    # than the literal fractional multiplier. Joints x100, gripper /100.
    if not INCLUDE_GRIPPER or GRIPPER_SCALE == 1.0:
        REPO_ID = f"local/lebai_duck_pick_delta_x{int(ACTION_DELTA_SCALE)}"
    else:
        REPO_ID = (f"local/lebai_duck_pick_delta_x{int(ACTION_DELTA_SCALE)}"
                   f"_g{int(round(1.0 / GRIPPER_SCALE))}")
elif ACTION_MODE == "absolute":
    REPO_ID = "local/lebai_duck_pick"
else:
    sys.exit(f"Unknown ACTION_MODE: {ACTION_MODE!r}. Use 'relative' or 'absolute'.")


# Helpers --------------------------------------------------------------------

def build_state(row):
    s = [float(row[f"jp{i}"]) for i in range(6)]
    if INCLUDE_GRIPPER:
        amp = row.get("claw_amplitude", 0.0)
        s.append(float(amp) if pd.notna(amp) else 0.0)
    return np.array(s, dtype=np.float32)


def build_action(row, next_row):
    """Action = 6 joint commands + optional gripper amplitude.

    Relative mode (default):
        joints[i] = (next_jp[i] - jp[i]) * ACTION_DELTA_SCALE
        gripper   = next_claw_amplitude * GRIPPER_SCALE
    Absolute mode:
        joints[i] = tgt_jp[i], falling back to next_jp[i] if tgt_jp is NaN
                    (some firmwares don't populate tgt_jp in teaching mode).
        gripper   = next_claw_amplitude  (UNSCALED in absolute mode)

    The next-frame trick (PRD §5) compensates for the gripper's slow actuator —
    pull `claw_amplitude` from `next_row`, not the current row.
    """
    if ACTION_MODE == "relative":
        joints = [
            float((next_row[f"jp{i}"] - row[f"jp{i}"]) * ACTION_DELTA_SCALE)
            for i in range(6)
        ]
    else:
        tgt = [row.get(f"tgt_jp{i}") for i in range(6)]
        if any(pd.isna(v) for v in tgt):
            joints = [float(next_row[f"jp{i}"]) for i in range(6)]
        else:
            joints = [float(v) for v in tgt]
    if INCLUDE_GRIPPER:
        amp = next_row.get("claw_amplitude", row.get("claw_amplitude", 0.0))
        amp = float(amp) if pd.notna(amp) else 0.0
        if ACTION_MODE == "relative":
            amp = amp * GRIPPER_SCALE
        joints.append(amp)
    return np.array(joints, dtype=np.float32)


def load_rgb(run_dir, rel_path):
    full = Path(run_dir) / rel_path
    bgr = cv2.imread(str(full), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"Could not read image: {full}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def get_image_writer(dataset):
    """Return the AsyncImageWriter if there is one, else None.

    lerobot 0.5.1 exposes it as either `dataset.image_writer` (direct attribute)
    or `dataset.writer.image_writer` depending on the build. Try both.
    """
    iw = getattr(dataset, "image_writer", None)
    if iw is None:
        writer = getattr(dataset, "writer", None)
        if writer is not None:
            iw = getattr(writer, "image_writer", None)
    return iw


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
        image_writer_threads=IMG_WRITER_THREADS,
        image_writer_processes=IMG_WRITER_PROCESSES,
    )
    iw = get_image_writer(dataset)
    print(f"Created empty dataset at {out_path}")
    if ACTION_MODE == "relative":
        print(f"  action_mode=relative  joint_scale={ACTION_DELTA_SCALE}  "
              f"gripper_scale={GRIPPER_SCALE} (gripper /= {int(round(1.0 / GRIPPER_SCALE))} on output)")
    else:
        print(f"  action_mode=absolute  scales=n/a")
    print(f"  state_dim={state_dim}  action_dim={action_dim}  wrist={has_wrist_data}")
    print(f"  image_writer: threads={IMG_WRITER_THREADS} processes={IMG_WRITER_PROCESSES} "
          f"present={'yes' if iw is not None else 'no'}")

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

            # Periodic drain so the image-writer queue doesn't grow unbounded
            if iw is not None and (i + 1) % DRAIN_EVERY == 0:
                iw.wait_until_done()

        # Final drain — save_episode reads images back to compute episode stats,
        # so they must be on disk before we call it.
        if iw is not None:
            iw.wait_until_done()
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
