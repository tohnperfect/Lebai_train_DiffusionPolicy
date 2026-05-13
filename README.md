# Lebai_train_DiffusionPolicy

Train a **Diffusion Policy** on teleoperated demos from a Lebai LM3 6-DOF arm + parallel gripper, and run it on the real robot. Sibling repository to `Lebai_train_ACT` — same data, same robot, same workflow, different policy.

## Recommended: convert locally, train on Colab, infer on the robot machine

The pipeline is four steps in strict order. Conversion and inference are `.py` scripts (no notebooks); training is a notebook (no scripts). Don't try to run inference on Colab — it has no LAN access to the robot.

### 1. Local conversion

Drop raw `save_state_img.py` collection logs into `./Data/`, then:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install 'lerobot==0.5.1' pandas opencv-python tqdm

python convert_local.py
```

Writes a LeRobot dataset to `./result/local/lebai_duck_pick/`. The bundled 10 logs convert to 10 episodes / 9246 frames in a few minutes on local SSD.

`lerobot==0.5.1` is pinned — newer versions move APIs. See [PRD §7](PRD_diffusion_policy.md#7-lerobot-05x-api--the-contract-that-bit-us-in-the-act-repo) for what breaks.

### 2. Local verification

```bash
python verify_local.py
```

**Run this in a fresh shell**, not in the same Python process that ran the converter. lerobot 0.5.1's async meta-parquet writer can lag the read by a beat, producing a misleading `ArrowInvalid: Parquet magic bytes not found in footer` error if you open the dataset too soon. See [PRD §10](PRD_diffusion_policy.md#10-step-2--verify_localpy).

Should print `Episodes: 10  Frames: 9246  FPS: 10` and the shape of the first sample.

### 3. Colab training

Pack the dataset for Drive (~3.5 GB for the bundled data):

```bash
cd result && tar -czf lebai_duck_pick.tar.gz local/lebai_duck_pick
```

Upload `lebai_duck_pick.tar.gz` to `MyDrive/Lebai_train_DiffusionPolicy/` (drag-drop in Drive UI).

Open [train_diffusion_colab.ipynb](train_diffusion_colab.ipynb) in Colab:

1. Runtime → Change runtime type → **GPU** (T4 minimum, A100 recommended).
2. Run all cells top-to-bottom.
3. **Smoke-test first.** Flip `SMOKE_TEST = True` in the training-setup cell for a 200-step sanity run; confirm loss is decreasing before committing to the full 100k-step run (which takes hours on T4).

Checkpoints are written to `MyDrive/Lebai_train_DiffusionPolicy/checkpoints/dp_run01/` — Colab can disconnect after a few hours, so the notebook resumes from the latest `step_*` on re-run.

Dataset is extracted from the Drive tarball to Colab's local SSD before training — don't read directly from Drive's FUSE layer during training, it's slow and corruption-prone.

### 4. Local inference

Download `checkpoints/dp_run01/final/` from Drive to a machine on the robot's LAN. Install the inference deps:

```bash
pip install 'lerobot==0.5.1' lebai_sdk requests opencv-python torch
```

**Always dry-run first.** This prints the predicted action without sending it to the robot:

```bash
python run_inference.py --dry-run
```

Verify `delta = action - state` is small (≲ 0.05 rad/joint) before allowing motion. Then:

```bash
SAFETY_OK=1 python run_inference.py --duration 30
SAFETY_OK=1 python run_inference.py --duration 30 --verbose   # every-tick logging
python run_inference.py --checkpoint ./checkpoints/dp_run01/step_080000 --dry-run
```

`SAFETY_OK=1` is a hard gate — the script refuses to call `movej` without it.

#### Safety checklist (non-negotiable, run through before every non-dry session)

1. **Workspace clear.** No one within the arm's sweep range.
2. **E-stop within reach.** Test it.
3. **Pendant velocity factor turned down.** Especially for the first run with a new checkpoint.
4. **Arm starts in a pose similar to a training episode's first frame.** The policy was trained on those starts and will not generalize to wildly different ones.
5. **Dry-run delta sanity-checked.** Joint deltas should be ≲ 0.05 rad, not flailing.
6. **`MAX_DURATION_S` is a hard cap** (default 30 s). The control loop exits regardless of policy behavior.

## What the policy expects (must match the converter)

- `observation.state`: `(7,) float32` = `[jp0..jp5, claw_amplitude]` (rad + `[0,100]` gripper amplitude).
- `action`: `(7,) float32` = `[tgt_jp0..tgt_jp5, next_claw_amplitude]`.
- Image features `(480, 640, 3) uint8` for the base camera (wrist optional, see PRD).
- Set `INCLUDE_GRIPPER = False` in [convert_local.py](convert_local.py) to drop the 7th dim — both state and action become `(6,)`. All three stages must agree.

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

## Files

| File | Purpose |
| ---- | ------- |
| [convert_local.py](convert_local.py) | Step 1 — local CLI converter (Data/ → LeRobot dataset) |
| [verify_local.py](verify_local.py)   | Step 2 — local CLI verifier (run in a fresh shell) |
| [train_diffusion_colab.ipynb](train_diffusion_colab.ipynb) | Step 3 — Colab GPU training |
| [run_inference.py](run_inference.py) | Step 4 — local inference CLI |
| [PRD_diffusion_policy.md](PRD_diffusion_policy.md) | Full spec, hyperparameters, lerobot 0.5.x gotchas, pitfalls |

## See also

[PRD_diffusion_policy.md](PRD_diffusion_policy.md) — exhaustive spec including:

- §6 — Diffusion Policy hyperparameters tuned for 10 Hz / 6-DOF (vs. LeRobot's bimanual-ALOHA-50-Hz defaults)
- §7 — lerobot 0.5.x API contract (import paths, `PolicyFeature` wrappers, HWC vs CHW for VISUAL features, async writer race)
- §11 — Inference loop invariants (`policy.reset()`, non-blocking `movej`, gripper rate-limiting)
- §14 — 14 numbered pitfalls already paid for in the ACT version
