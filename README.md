# Lebai_train_DiffusionPolicy

Train a **Diffusion Policy** on teleoperated demos from a Lebai LM3 6-DOF arm + parallel gripper, and run it on the real robot. Sibling repository to `Lebai_train_ACT` — same data, same robot, similar workflow, different policy.

## Recommended pipeline

```
Local convert → rsync to GPU box → local GPU train → scp checkpoint to robot machine → inference
```

The four stages are strict, in this order. Conversion and inference are `.py` scripts; training is `.py` on a GPU box (preferred) or a Colab notebook (fallback). **Don't run inference on Colab** — it has no LAN access to the robot.

### 1. Local conversion

Drop raw `save_state_img.py` collection logs into `./Data/`. Set up a venv and convert:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install 'lerobot==0.5.1' pandas opencv-python tqdm

python convert_local.py
```

Writes a LeRobot dataset to `./result/local/lebai_duck_pick_delta_x100/`. The repo-id suffix `_delta_x100` encodes the **action representation**: scaled per-tick joint deltas with `ACTION_DELTA_SCALE = 100`. This is the default; see [convert_local.py](convert_local.py) to switch to absolute targets if needed. All four stages must agree on the action mode and scale.

`lerobot==0.5.1` is pinned — newer versions move APIs. See [PRD §7](PRD_diffusion_policy.md).

### 2. Local verification

```bash
python verify_local.py
```

**Run this in a fresh shell**, not in the same Python process that ran the converter. lerobot 0.5.1's async meta-parquet writer can lag the read by a beat, producing a misleading `ArrowInvalid: Parquet magic bytes not found in footer` error if you open the dataset too soon.

Should print episode/frame counts and a spot-check of action magnitudes (relative-mode actions should be roughly `±0.5–1.0`, not `±0.005`).

### 3. GPU training

#### 3a. Local GPU box (recommended)

If you have a Linux box with an NVIDIA GPU, `rsync` the dataset over and train locally:

```bash
# from this repo on your laptop
rsync -avh --progress \
    ./result/local/lebai_duck_pick_delta_x100/ \
    user@gpu:Lebai_train_DiffusionPolicy/result/local/lebai_duck_pick_delta_x100/

# on the GPU box
cd Lebai_train_DiffusionPolicy
./setup_gpu_env.sh            # creates .venv/, installs lerobot==0.5.1
source .venv/bin/activate

# SMOKE TEST FIRST — 200 steps in ~1-3 minutes. Mandatory.
python train_local_gpu.py --smoke-test

# Full run (default 100k steps, batch 8 for T4; bump to 16 on A100).
python train_local_gpu.py
```

`rsync` beats tarball + Drive upload because it's resumable, incremental on re-runs, and doesn't need a 14 GB tarball sitting on disk.

Resume from the latest `step_*` checkpoint is automatic; `--no-resume` starts fresh.

#### 3b. Colab notebook (fallback)

For users without a local GPU. See [train_diffusion_colab.ipynb](train_diffusion_colab.ipynb) — same logic as `train_local_gpu.py`, with extra cells for mounting Drive and extracting the dataset tarball.

Tarball the dataset, upload to `MyDrive/Lebai_train_DiffusionPolicy/`, open the notebook in Colab with a GPU runtime, run all cells:

```bash
cd result && tar -czf lebai_duck_pick_delta_x100.tar.gz local/lebai_duck_pick_delta_x100
```

#### Disk hygiene during training

When you open the dataset, HuggingFace `datasets` makes its own copy at `~/.cache/huggingface/datasets/`. For a ~10 GB lerobot dataset that's another ~10 GB of cache. After a couple of runs you can easily accumulate 30+ GB.

- Clear it after a known-good verification: `rm -rf ~/.cache/huggingface/datasets/`.
- On a disk-constrained GPU box, set `HF_DATASETS_CACHE=/scratch/hf_cache` before launching training.
- Budget **~2×** the dataset size in free disk during training.

### 4. Local inference

Copy the trained checkpoint from the GPU box to the robot machine:

```bash
scp -r user@gpu:Lebai_train_DiffusionPolicy/checkpoints/dp_run01/final/ ./checkpoints/dp_run01/
```

Install the inference deps on the robot machine:

```bash
pip install 'lerobot==0.5.1' lebai_sdk requests opencv-python torch
```

**Always dry-run first.** This prints the predicted action without sending it to the robot:

```bash
python run_inference.py --dry-run
```

Verify `abs target` is close to `current state` (joint move `|max|` ≲ 0.05 rad per tick) before allowing motion. Then:

```bash
SAFETY_OK=1 python run_inference.py --duration 30
SAFETY_OK=1 python run_inference.py --duration 30 --verbose
python run_inference.py --checkpoint ./checkpoints/dp_run01/step_080000 --dry-run
```

The default `--action-mode relative --action-delta-scale 100` matches `convert_local.py`'s defaults. **Mismatching the scale produces wildly wrong joint targets** — always confirm the dry-run output is sane before passing `SAFETY_OK=1`.

`SAFETY_OK=1` is a hard gate. The script refuses to call `movej` without it.

#### Safety checklist (non-negotiable, run through before every non-dry session)

1. **Workspace clear.** No one within the arm's sweep range.
2. **E-stop within reach.** Test it.
3. **Pendant velocity factor turned down.** Especially for the first run with a new checkpoint.
4. **Arm starts in a pose similar to a training episode's first frame.** The policy was trained on those starts and will not generalize to wildly different ones.
5. **Dry-run sane.** Joint deltas should be ≲ 0.05 rad per tick, not flailing.
6. **`MAX_DURATION_S` is a hard cap** (default 30 s). The control loop exits regardless of policy behavior.

## State / action conventions (must match across all stages)

- `observation.state`: `(7,) float32` = `[jp0..jp5, claw_amplitude]` (rad + `[0,100]` gripper amplitude).
- `action` in **relative mode** (default): `(7,) float32` = `[(next_jp-jp)*scale, next_claw_amplitude]`. Joint deltas are scaled by `ACTION_DELTA_SCALE = 100`; gripper is the unscaled next-frame amplitude.
- `action` in **absolute mode**: `(7,) float32` = `[tgt_jp0..tgt_jp5, next_claw_amplitude]` (the old convention).
- Image features `(480, 640, 3) uint8` for the base camera (wrist optional).
- Set `INCLUDE_GRIPPER = False` in [convert_local.py](convert_local.py) to drop the 7th dim → both state and action become `(6,)`. All stages must agree.

## Hardware defaults

Hard-coded in [run_inference.py](run_inference.py); override with `--robot-ip` / `--camera-url`.

| Setting              | Value                              |
| -------------------- | ---------------------------------- |
| Robot IP             | `192.168.31.254`                   |
| Camera service URL   | `http://192.168.31.192:8000`       |
| Image resolution     | 640 × 480                          |
| FPS                  | 10                                 |
| Joint accel / vel    | 1.5 rad/s² / 1.0 rad/s             |
| Blend radius         | 0.05 rad                           |
| Gripper threshold    | 5.0 (rate-limits set_claw calls)   |
| Default max duration | 30 s                               |
| Action mode / scale  | relative / 100.0                   |

## Files

| File | Purpose |
| ---- | ------- |
| [convert_local.py](convert_local.py) | Step 1 — local CLI converter (Data/ → LeRobot dataset) |
| [verify_local.py](verify_local.py)   | Step 2 — local CLI verifier (run in a fresh shell) |
| [setup_gpu_env.sh](setup_gpu_env.sh) | Bootstrap `.venv/` on the GPU box |
| [requirements_gpu.txt](requirements_gpu.txt) | Pinned deps for GPU training |
| [train_local_gpu.py](train_local_gpu.py) | Step 3a — local GPU training (preferred) |
| [train_diffusion_colab.ipynb](train_diffusion_colab.ipynb) | Step 3b — Colab fallback |
| [run_inference.py](run_inference.py) | Step 4 — local inference CLI |
| [PRD_diffusion_policy.md](PRD_diffusion_policy.md) | Full spec, hyperparameters, lerobot 0.5.x gotchas, pitfalls |

## See also

[PRD_diffusion_policy.md](PRD_diffusion_policy.md):

- §5 — Action representation (relative vs absolute, scaling)
- §6 — Diffusion Policy hyperparameters tuned for 10 Hz / 6-DOF
- §7 — lerobot 0.5.x API contract (imports, `PolicyFeature` wrappers, HWC vs CHW, async writer race)
- §11 — Inference loop invariants (`policy.reset()`, non-blocking `movej`, gripper rate-limiting)
- §14 — Numbered pitfalls already paid for in the ACT version + DP-specific additions
