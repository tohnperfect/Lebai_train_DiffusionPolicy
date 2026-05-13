# PRD — Lebai_train_DiffusionPolicy

Sibling repository to `Lebai_train_ACT`. Same data, same robot, same workflow — but trains a **Diffusion Policy** instead of ACT.

This document is intentionally specific. It captures concrete decisions, exact APIs, and known pitfalls from the ACT version of this project so the implementer does not rediscover them.

---

## 1. Project goal

A three-step pipeline to train a Diffusion Policy on teleoperated demos from a **Lebai LM3** 6-DOF arm + parallel gripper and run it on the real robot.

**Workflow (non-negotiable order):**

1. **Local conversion** — Python script, reads raw collection logs in `./Data/`, writes a [LeRobot](https://github.com/huggingface/lerobot) dataset under `./result/`. Run on the developer's laptop.
2. **Local verification** — separate Python script, opens the converted dataset in a fresh process, prints sanity stats. Required because of an async-writer race in lerobot ≥0.5 (see §10).
3. **Colab training** — Jupyter notebook. Reads the dataset from a tarball on Drive, trains, writes checkpoints back to Drive. GPU runtime.
4. **Local inference** — Python script. Loads a Drive checkpoint, talks to the live arm over LAN. Must run on a machine on the robot's network.

Conversion and inference are **`.py` scripts only**. No notebooks for those. Notebooks are reserved for training.

---

## 2. Why Diffusion Policy instead of ACT

DP handles **multi-modal demonstrations** (e.g. "grasp the bottle from the left OR the right") that ACT averages into a wishy-washy mean trajectory. Cost: heavier model, iterative denoising at inference (10–100 forward passes per action) means the 10 Hz tick budget is tight — see §9 for the mitigation (action-horizon execution).

If the developer asks "should I use ACT?", point them at the ACT repo. This repo is DP-only.

---

## 3. Repository layout

```
Lebai_train_DiffusionPolicy/
├── CLAUDE.md                                  # Claude Code's project notes
├── README.md                                  # User-facing docs
├── .gitignore                                 # Data/, .venv/, result/, checkpoints/, act_run01/, .DS_Store, .vscode/
├── Data/                                      # gitignored; user drops raw logs here
├── convert_local.py                           # Step 1 — local CLI converter
├── verify_local.py                            # Step 2 — local CLI verifier (separate process)
├── train_diffusion_colab.ipynb                # Step 3 — Colab training notebook
├── run_inference.py                           # Step 4 — local inference CLI
└── recover_lerobot_dataset.py                 # Optional — for older dataset layouts; not strictly needed for lerobot ≥0.5
```

Notebooks must work out of the box on a fresh Colab runtime (Python 3.10+, T4 GPU). Scripts must work on a Python 3.10+ venv on macOS or Linux.

---

## 4. Data format (input, fixed)

The collector — `save_state_img.py`, **not in this repo** — produces:

```
Data/log<NNNN>.csv                        # flat per-frame index, one row per frame, one CSV per session
Data/log<NNNN>/ep<NNN>/color/000000.jpg   # base RGB
Data/log<NNNN>/ep<NNN>/wrist/000000.jpg   # optional wrist RGB
Data/log<NNNN>/ep<NNN>/state/000000.json  # full SDK state per frame
Data/log<NNNN>/ep<NNN>/episode_meta.json  # task, frame count, fps, timestamps
```

CSV columns include: `frame, color, wrist, jp0..jp5, tgt_jp0..tgt_jp5, claw_amplitude, episode, task`.

`task` is a natural-language string (e.g. `"put the duck in the bowl"`). The converter must preserve it.

---

## 5. State and action conventions (LeRobot features)

Same as the ACT repo:

- **`observation.state`** = `(7,) float32` = `[jp0..jp5, claw_amplitude]` — current joints in rad, gripper amplitude in `[0, 100]`.
- **`action`** = `(7,) float32` = `[tgt_jp0..tgt_jp5, next_claw_amplitude]` — commanded joint targets + **next-frame** gripper amplitude.

The **next-frame gripper trick** compensates for the slow gripper actuator: the command issued at time `t` shows up in the actual amplitude around `t+1`, so the converter pulls `claw_amplitude` from the *next* row. If `tgt_jp*` is empty (some firmwares don't populate it in teaching mode), fall back to `next_row.jp*`.

`INCLUDE_GRIPPER = False` in the converter drops the 7th dim → both state and action become `(6,)`. All three stages must agree.

Image features:
- `observation.images.base` = `(480, 640, 3) uint8`
- `observation.images.wrist` = same, **only if present in all frames** (see §10 pitfall).

---

## 6. Diffusion Policy hyperparameters (defaults — tuned for 10 Hz / 6-DOF)

LeRobot's `DiffusionConfig` defaults assume bimanual ALOHA at 50 Hz. Override these:

| Setting                  | Value           | Why                                                            |
| ------------------------ | --------------- | -------------------------------------------------------------- |
| `n_obs_steps`            | 2               | DP standard — two-frame observation history                    |
| `horizon`                | 16              | Predict 16 actions per forward pass (~1.6 s at 10 Hz)          |
| `n_action_steps`         | 8               | Execute the first 8 of those (~0.8 s) before re-querying       |
| `num_train_timesteps`    | 100             | DDPM schedule length                                            |
| `num_inference_steps`    | 10              | Use DDIM at inference — 10 denoising steps fits the 100 ms budget |
| `vision_backbone`        | `resnet18`      | Same as ACT for fair comparison                                |
| `crop_shape`             | `(84, 84)`      | DP default random crop                                          |
| `noise_scheduler_type`   | `"DDPM"` train, `"DDIM"` infer | Standard pattern                                   |
| `prediction_type`        | `"epsilon"`     | DDPM default                                                    |
| `optimizer_lr`           | `1e-4`          | Same as ACT                                                    |
| `BATCH_SIZE`             | 8 (T4) / 16 (A100) | Fits 16 GB / 80 GB respectively                              |
| `NUM_STEPS`              | 100_000         | DP needs more steps than ACT to converge                       |
| `SAVE_EVERY`             | 10_000          | Larger interval than ACT — DP checkpoints are bigger           |
| `FPS`                    | 10              | Matches the collector                                          |

Verify on a smoke test with 200 steps that loss decreases before committing to a 100k-step run.

---

## 7. lerobot 0.5.x API — the contract that bit us in the ACT repo

**Pin lerobot to a known-working version** in install instructions. As of writing, `lerobot==0.5.1` is what we tested. Newer versions move things again.

### 7.1 Import paths (NOT `lerobot.common.*` in ≥0.5)

```python
# ≥0.5 layout
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.configs.types import FeatureType, PolicyFeature
```

Provide a `try/except` for legacy users on <0.5 (`lerobot.common.*`), but treat ≥0.5 as the default.

### 7.2 `HF_LEROBOT_HOME` is no longer exported from `lerobot_dataset`

Set it as an env var **before importing `lerobot`** and read it back yourself:

```python
import os
from pathlib import Path
os.environ["HF_LEROBOT_HOME"] = "/abs/path/to/result"
# ... later, after import ...
HF_LEROBOT_HOME = Path(os.environ["HF_LEROBOT_HOME"])
```

Or import from the new location: `from lerobot.constants import HF_LEROBOT_HOME`. Wrap both in a try/except.

### 7.3 `cfg.input_features` / `cfg.output_features` need `PolicyFeature` wrappers

Dataset features are plain dicts. The policy config wants `PolicyFeature(type=FeatureType.X, shape=...)`:

```python
def make_policy_features(features_dict):
    out = {}
    for name, spec in features_dict.items():
        if name.startswith("observation.images."):
            ft = FeatureType.VISUAL
        elif name == "observation.state":
            ft = FeatureType.STATE
        elif name == "action":
            ft = FeatureType.ACTION
        else:
            continue
        out[name] = PolicyFeature(type=ft, shape=tuple(spec["shape"]))
    return out

all_feats = make_policy_features(dataset.features)
cfg.input_features  = {k: v for k, v in all_feats.items() if k != "action"}
cfg.output_features = {"action": all_feats["action"]}
```

### 7.4 `policy.forward()` returns a `(loss, ...)` tuple, not `{"loss": ...}`

```python
loss, loss_dict = policy.forward(batch)
loss.backward()
```

**DP-specific:** `DiffusionPolicy.forward`'s signature in lerobot 0.5.1 is `-> tuple[Tensor, None]` — it always returns `(loss, None)`. Only ACT populates the second element (with `l1_loss`, `kld_loss`, etc.). Unpack and discard:

```python
loss, _ = policy.forward(batch)
```

### 7.5 Episode boundaries — no more `episode_data_index`

```python
ep0 = ds.meta.episodes[0]
ep0_from = int(ep0["dataset_from_index"])
ep0_to   = int(ep0["dataset_to_index"])
```

### 7.6 Image `PolicyFeature` shape must be CHW for `DiffusionPolicy` (HWC for ACT)

The dataset stores image features in HWC order — e.g. `observation.images.base = {"shape": (480, 640, 3)}` — because that's what `LeRobotDataset.create()` wants in `features`. When loading a frame, the dataset hands you the image already transposed to CHW, so the runtime tensor is `(3, 480, 640)`. But the `PolicyFeature` you pass to the policy config is *not* automatically transposed — and `DiffusionRgbEncoder.__init__` in lerobot 0.5.1 reads `images_shape[0]` as the channel count when building its dummy probe:

```python
# lerobot/policies/diffusion/modeling_diffusion.py
images_shape = next(iter(config.image_features.values())).shape
...
dummy_shape = (1, images_shape[0], *dummy_shape_h_w)
```

Pass it the dataset's `(480, 640, 3)` and ResNet sees 480 channels and explodes:
`RuntimeError: ... expected input[1, 480, 96, 128] to have 3 channels, but got 480 channels instead`.

ACT's encoder does not have this bug because its dummy probe doesn't use `images_shape[0]` as the channel count. The convention is inconsistent inside lerobot itself (sarm, env configs, and several examples store HWC `(480, 640, 3)`; DP requires CHW). The fix: transpose in `make_policy_features` for VISUAL features only.

```python
if name.startswith("observation.images."):
    h, w, c = spec["shape"]
    shape = (c, h, w)        # HWC -> CHW for DP
else:
    shape = tuple(spec["shape"])
```

Keep this transpose limited to VISUAL features — STATE/ACTION shapes are 1-D and unaffected.

### 7.7 Async meta-parquet writer race

This bit us hard. **Do not call `LeRobotDataset(REPO_ID)` in the same Python process that just wrote the dataset.** The meta parquet at `meta/episodes/chunk-000/file-000.parquet` is finalized by a background thread whose flush can lag the end of conversion. Reading too soon gives `ArrowInvalid: Parquet magic bytes not found in footer.` The file is fine after the process exits.

**Mitigation:** verify in a separate process. `verify_local.py` is mandatory, not optional.

---

## 8. Step 1 — `convert_local.py`

Headless CLI. Reads `./Data/`, writes LeRobot dataset under `./result/local/lebai_duck_pick/`. Must:

1. Set `HF_LEROBOT_HOME=./result` **before** importing lerobot.
2. Accept the configuration as module-level constants — `RUN_NUMBER`, `REPO_ID`, `INCLUDE_WRIST`, `INCLUDE_GRIPPER`, `FPS=10`, `IMG_H=480`, `IMG_W=640`. No CLI args.
3. Wipe `out_path` if it exists (idempotent).
4. Detect wrist data **correctly** — pandas reads empty CSV cells as `NaN`, `astype(str)` makes them `"nan"` (length 3). Filter both:
   ```python
   wrist_strs = df["wrist"].fillna("").astype(str).str.strip()
   all_have_wrist = ((wrist_strs != "") & (wrist_strs.str.lower() != "nan")).all()
   has_wrist_data = INCLUDE_WRIST and all_have_wrist
   ```
   `has_wrist_data` must be `True` only if **all** rows have valid wrist paths. Mixed-presence wrist data is not supported.
5. Iterate `df.groupby(["__run_dir__", "episode"])` in sorted order. For each frame:
   - Build `observation.state` from `jp0..jp5` + `claw_amplitude`.
   - Build `action` from `tgt_jp0..tgt_jp5` (fall back to `next_row.jp*` if NaN) + `next_row.claw_amplitude`.
   - Load both image(s) with `cv2.imread` → BGR → `cv2.cvtColor` → RGB.
   - Call `dataset.add_frame({...})`.
6. After each episode, call `dataset.save_episode()`.
7. Print clear progress (one line per episode).
8. **Do NOT verify in-process at the end.** Print a hint to run `verify_local.py`.

Example imports:

```python
import os, shutil, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
LOG_ROOT = REPO_ROOT / "Data"
RESULT_ROOT = REPO_ROOT / "result"
RESULT_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["HF_LEROBOT_HOME"] = str(RESULT_ROOT)

import cv2, numpy as np, pandas as pd
from tqdm import tqdm

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
```

---

## 9. Step 2 — `verify_local.py`

Standalone CLI. Opens the dataset, prints summary stats, exits non-zero on error.

```python
ds = LeRobotDataset("local/lebai_duck_pick")
print(f"Episodes: {ds.num_episodes}")
print(f"Frames:   {ds.num_frames}")
print(f"FPS:      {ds.fps}")
sample = ds[0]
print(f"state:  shape={sample['observation.state'].shape}")
print(f"action: shape={sample['action'].shape}")
print(f"task:   {sample['task']!r}")
```

The script's docstring must explicitly say "run this in a fresh shell after `convert_local.py`" and explain the async-writer race. This is so future maintainers don't re-add an in-process verify.

---

## 10. Step 3 — `train_diffusion_colab.ipynb`

Single Colab notebook. Cells in this order:

1. **Intro markdown.** Pre-conditions: local conversion done, tarball uploaded, GPU runtime.
2. **Install deps.** `!pip install -q lerobot matplotlib`. Pin a version: `lerobot==0.5.1`.
3. **Mount Drive.** Use `force_remount=True` — Colab leaves stale state otherwise.
4. **Paths cell.**
   - `DRIVE_ROOT = Path("/content/drive/MyDrive/Lebai_train_DiffusionPolicy")`.
   - `LEROBOT_CACHE = Path("/content/lerobot_cache")` — **local SSD, not Drive**.
   - `CHECKPOINT_DIR = DRIVE_ROOT / "checkpoints/dp_run01"` — Drive (small, sequential writes, fine for Drive).
   - `DATASET_TAR = DRIVE_ROOT / "lebai_duck_pick.tar.gz"`.
   - On first run: `subprocess.run(["tar", "-xzf", str(DATASET_TAR), "-C", str(LEROBOT_CACHE)], check=True)`.
   - Set `os.environ["HF_LEROBOT_HOME"]` to the cache path.
5. **Imports.** Define `make_policy_features(...)` helper here.
6. **Dataset metadata + sanity-check viz** (image + state/action plots for episode 0). Use `ds.meta.episodes[0]["dataset_from_index"]` to slice.
7. **Configure policy** with the §6 hyperparameters. Build `DiffusionConfig`, wrap features.
8. **Resume cell.** Check `CHECKPOINT_DIR / "step_*"`, load latest if present, set `start_step` accordingly. Fast-forward the LR scheduler.
9. **Train loop.** `range(start_step, NUM_STEPS)`. Unpack `loss, loss_dict = policy.forward(batch)`. Save checkpoints every `SAVE_EVERY`.
10. **Loss curve plot.**
11. **Next-steps markdown.** Download `final/` to the robot machine and point `run_inference.py` at it.

The dataset must be extracted to **local SSD** before training. Reading from Drive's FUSE layer during a 100k-step run is unbearably slow and corruption-prone.

---

## 11. Step 4 — `run_inference.py`

Headless inference CLI. Same shape as the ACT repo's `run_inference.py`, with these DP-specific differences:

- **Iterative denoising.** `policy.select_action(obs)` already handles this internally for both ACT and DP, so the call site is identical — but per-tick latency is **higher** for DP. Measure and report `loop=Xms`.
- **`n_action_steps` chunking.** DP's internal queue executes `n_action_steps` actions before re-querying. So `select_action` returns an action every tick; only every `n_action_steps`-th call does a real forward pass. Don't write your own chunking — trust the policy's internal queue.

CLI shape (match the ACT repo for consistency):

```bash
# Sanity check, no motion. Always run first.
python run_inference.py --dry-run

# Real run. SAFETY_OK=1 is required.
SAFETY_OK=1 python run_inference.py --duration 30

# Verbose every-tick logging:
SAFETY_OK=1 python run_inference.py --duration 30 --verbose

# Other checkpoint:
python run_inference.py --checkpoint ./dp_run01/step_080000 --dry-run
```

### Required script features (do not skip any):

1. **`policy.reset()` before each control-loop run.** Clears the internal action queue. Skipping it makes the first ~`n_action_steps` actions reuse stale history.
2. **`lebai.movej()` *without* `wait_move()`.** The blend radius (`BLEND_RADIUS = 0.05`) is what makes successive non-blocking joint commands stitch together. Re-adding `wait_move()` blocks the 100 ms tick budget; the loop will fall behind.
3. **Gripper rate-limiting** via `GRIPPER_THRESHOLD = 5.0`. The policy outputs continuous amplitudes; without thresholding, the gripper twitches every tick.
4. **`start_sys()` and `init_claw()` must run before any motion command.** The SDK silently no-ops otherwise.
5. **Dry-run prediction always prints before any motion.** Even when not in `--dry-run` mode. Lets the user catch shape/normalization bugs before the arm moves.
6. **`SAFETY_OK=1` env var gate.** Refuse to send `movej` without it. Print the safety checklist on refusal.
7. **3-second countdown** before the control loop starts.
8. **`MAX_DURATION_S` hard cap** (default 30 s). Loop exits no matter what.
9. **`try/finally`** to call `cam.stop_all()` and `lebai.stop_sys()` on any exit path.
10. **Connection defaults** hard-coded: `ROBOT_IP = "192.168.31.254"`, `CAMERA_URL = "http://192.168.31.192:8000"`. Override via `--robot-ip` / `--camera-url`.

### Verbose output format

Three lines per tick when `--verbose`:

```
  t=  1.0s  step=  10  loop=42ms
    state : [ 1.574 -1.554  1.300 -1.318 -1.571  0.002]    gripper= 99.0
    action: [ 1.575 -1.555  1.301 -1.318 -1.571  0.002]    gripper= 99.0 (SENT|rate-limited)
    delta : [ 0.001 -0.001  0.001  0.000  0.000  0.000]
```

Without `--verbose`: one summary line per second.

---

## 12. README must document the local-convert + Colab-train flow

Headline section: **"Recommended: convert locally, train on Colab, infer on the robot machine."**

Include:

1. Local convert: `python convert_local.py` then `python verify_local.py`.
2. Tarball: `cd result && tar -czf lebai_duck_pick.tar.gz local/lebai_duck_pick`. Note the ~3.5 GB size for ~10 episodes / 9k frames.
3. Upload to Drive at `MyDrive/Lebai_train_DiffusionPolicy/lebai_duck_pick.tar.gz`.
4. Open notebook in Colab → set GPU runtime → run all cells.
5. Download `checkpoints/dp_run01/final/` from Drive to the robot machine.
6. Local inference: dry run, then `SAFETY_OK=1 ...`.

Include the **explicit safety checklist** in the inference section. Non-negotiable.

---

## 13. `.gitignore`

```
Data/
.DS_Store
.venv/
result/
checkpoints/
.vscode/
dp_run01/
```

---

## 14. Common pitfalls (DO NOT REPEAT — these are real, we hit each one)

1. **Don't write the LeRobot dataset to Google Drive directly.** Thousands of small parquet shard writes corrupt files when Colab disconnects. Convert locally or to Colab SSD; transport via tarball.
2. **Don't verify a dataset in the same Python process that wrote it.** Async meta-parquet writer can lag. `ArrowInvalid: Parquet magic bytes not found` is misleading — the file is fine seconds later in a fresh process.
3. **Don't trust `astype(str)` on a pandas column with NaN.** `NaN → "nan"` (length 3), which `.str.len() > 0` returns True for. Use `.fillna("")` then check `!= ""` and `.str.lower() != "nan"`.
4. **Don't pass dataset features directly to `cfg.input_features` / `cfg.output_features`.** They need `PolicyFeature` wrappers in lerobot ≥0.5.
5. **Don't expect `episode_data_index` on the dataset.** Use `ds.meta.episodes[i]["dataset_from_index"]`.
6. **Don't unpack `policy.forward(batch)` as a dict.** It returns `(loss, loss_dict)` tuple in lerobot ≥0.5.
7. **Don't run inference on Colab.** No LAN access to robot/camera. Inference is **local only**.
8. **Don't run inference without a dry-run first.** Always print the predicted action and verify `delta = action - state` is small (~`±0.05` rad per joint) before allowing motion.
9. **Don't re-add `wait_move()` to the control loop.** Blocks the tick budget.
10. **Don't run multiple `set_claw()` calls per tick.** Gripper is a slow actuator; rate-limit changes via `GRIPPER_THRESHOLD`.
11. **Don't change `INCLUDE_GRIPPER` / `INCLUDE_WRIST` / `FPS` in the middle of a project.** Those bake into the dataset features; changing them mid-flight breaks resume and produces shape mismatches downstream.
12. **Don't skip `policy.reset()` between inference runs.** Stale temporal queue → first chunk_size actions are wrong.
13. **Don't pip-install `lerobot` into a conda env that has `opencv-python-headless` installed via conda.** Pip can't uninstall conda packages. Either `conda uninstall opencv-python-headless` first, or use a clean venv.
14. **Don't assume a free Colab session can run 100k steps without disconnecting.** Implement resume from the latest `step_*` checkpoint. Test the resume path before committing to the long run.

---

## 15. Hardware / connection defaults (Lebai LM3 only)

| Setting              | Value                              |
| -------------------- | ---------------------------------- |
| Robot IP             | `192.168.31.254`                   |
| Camera service URL   | `http://192.168.31.192:8000`        |
| Image resolution     | `640 × 480`                         |
| FPS                  | 10                                  |
| Joint acc limit      | 1.5 rad/s²                          |
| Joint vel limit      | 1.0 rad/s                           |
| Blend radius         | 0.05 rad                            |
| Gripper force        | 10                                  |
| Gripper threshold    | 5.0                                 |
| Control period       | 0.1 s (10 Hz)                       |
| Default max duration | 30 s                                |

These come from `grasp_to_the_bowl.py` (Lebai's reference scripted-control example, not in this repo).

---

## 16. Acceptance criteria

The new repo is done when:

- [ ] `python convert_local.py` produces `result/local/lebai_duck_pick/` containing data + meta parquets.
- [ ] `python verify_local.py` opens that dataset cleanly and prints 10 episodes / 9246 frames for the bundled data.
- [ ] `tar -czf result/lebai_duck_pick.tar.gz -C result local/lebai_duck_pick` produces a ~3.5 GB tarball.
- [ ] `train_diffusion_colab.ipynb` runs end-to-end on a fresh Colab GPU runtime: extracts the tarball, builds the policy, runs at least 200 smoke-test steps with decreasing loss, saves a `final/` checkpoint to Drive.
- [ ] `python run_inference.py --dry-run` connects to the camera + robot and prints sensible state/action/delta values.
- [ ] `SAFETY_OK=1 python run_inference.py --duration 30` runs the control loop at 10 Hz with `loop=` under 100 ms.
- [ ] README documents the full flow.

---

## 17. Out of scope

- Multi-task or language-conditioned models.
- Training on collected data with a wrist camera (the reference dataset doesn't include wrist; add it later when needed).
- Pushing datasets or checkpoints to the HuggingFace Hub.
- Anything inference-time on Colab.

---

## 18. References from the ACT version of this work

Specific files to mirror the patterns from (if the implementer has access to `Lebai_train_ACT`):

- `convert_local.py` — module-level constants, wrist detection, episode loop, no in-process verify.
- `verify_local.py` — minimal, separate process.
- `act_training_colab.ipynb` — `paths` cell that extracts a tarball from Drive to local SSD, `resume` cell that picks up the latest `step_*` checkpoint.
- `run_inference.py` — CLI shape, dry-run-always-first, `SAFETY_OK` gate, verbose mode, `try/finally` cleanup.
- `CLAUDE.md` — short hands for the project-specific facts the codebase doesn't make obvious.
