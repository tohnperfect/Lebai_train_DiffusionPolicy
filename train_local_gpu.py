"""Train Diffusion Policy on a local GPU box (no Colab needed).

Headless equivalent of `train_diffusion_colab.ipynb`. Run from the repo root
after `setup_gpu_env.sh` has installed deps into .venv/:

    source .venv/bin/activate

    # Smoke test first — 200 steps, batch 4, no checkpoints. Verifies the env
    # and dataset are wired up correctly in ~1-3 minutes. Mandatory before a
    # full run.
    python train_local_gpu.py --smoke-test

    # Full run — defaults match PRD §6.
    python train_local_gpu.py

    # Resume happens automatically from the latest step_* checkpoint under
    # --checkpoint-dir. Pass --no-resume to start fresh.
    python train_local_gpu.py --no-resume

The dataset must already be converted locally (`python convert_local.py`) and
present under HF_LEROBOT_HOME (defaults to ./result/). If your GPU box is
disk-constrained, set HF_DATASETS_CACHE before running:

    HF_DATASETS_CACHE=/scratch/hf_cache python train_local_gpu.py

HuggingFace datasets makes its own ~14 GB copy at the cache path; budget ~2x
the dataset size in free disk.
"""

import argparse
import os
import sys
import time
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULT_ROOT = REPO_ROOT / "result"
os.environ.setdefault("HF_LEROBOT_HOME", str(DEFAULT_RESULT_ROOT))

# Action mode + scales baked into the default REPO_ID; must match convert_local.py.
DEFAULT_ACTION_MODE        = "relative"
DEFAULT_ACTION_DELTA_SCALE = 100.0
DEFAULT_GRIPPER_SCALE      = 0.01
if DEFAULT_ACTION_MODE == "relative":
    if DEFAULT_GRIPPER_SCALE == 1.0:
        DEFAULT_REPO_ID = f"local/lebai_duck_pick_delta_x{int(DEFAULT_ACTION_DELTA_SCALE)}"
    else:
        DEFAULT_REPO_ID = (
            f"local/lebai_duck_pick_delta_x{int(DEFAULT_ACTION_DELTA_SCALE)}"
            f"_g{int(round(1.0 / DEFAULT_GRIPPER_SCALE))}"
        )
else:
    DEFAULT_REPO_ID = "local/lebai_duck_pick"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo-id", default=DEFAULT_REPO_ID,
                   help=f"LeRobotDataset repo id (default: {DEFAULT_REPO_ID})")
    p.add_argument("--checkpoint-dir", type=Path, default=Path("./checkpoints/dp_run01"),
                   help="Where to save step_* and final checkpoints (default: ./checkpoints/dp_run01)")
    p.add_argument("--num-steps", type=int, default=100_000,
                   help="Total training steps (default: 100_000). Ignored if --smoke-test.")
    p.add_argument("--batch-size", type=int, default=8,
                   help="Per-step batch size (default: 8 for T4-class; raise to 16 on A100).")
    p.add_argument("--num-workers", type=int, default=2,
                   help="DataLoader worker count (default: 2)")
    p.add_argument("--lr", type=float, default=1e-4, help="AdamW learning rate (default: 1e-4)")
    p.add_argument("--log-every", type=int, default=200, help="Steps between log lines (default: 200)")
    p.add_argument("--save-every", type=int, default=10_000,
                   help="Steps between checkpoint saves (default: 10_000)")
    p.add_argument("--smoke-test", action="store_true",
                   help="Run a 200-step / batch-4 sanity check with no checkpoint saves.")
    p.add_argument("--no-resume", action="store_true",
                   help="Skip auto-resume from the latest step_* checkpoint.")
    return p.parse_args()


def make_policy_features(features_dict, FeatureType, PolicyFeature):
    """Wrap dataset features into PolicyFeature objects (lerobot 0.5+).

    DP-specific: image schemas in the dataset are HWC but DiffusionRgbEncoder
    reads images_shape[0] as channels — so transpose VISUAL features to CHW.
    See PRD §7.6.
    """
    out = {}
    for name, spec in features_dict.items():
        shape = tuple(spec["shape"])
        if name.startswith("observation.images."):
            ft = FeatureType.VISUAL
            if len(shape) == 3 and shape[-1] in (1, 3):
                shape = (shape[2], shape[0], shape[1])    # HWC -> CHW
        elif name == "observation.state":
            ft = FeatureType.STATE
        elif name == "action":
            ft = FeatureType.ACTION
        else:
            continue
        out[name] = PolicyFeature(type=ft, shape=shape)
    return out


def main():
    args = parse_args()

    # Apply smoke-test overrides early so the printout reflects what we'll run.
    if args.smoke_test:
        args.num_steps = 200
        args.batch_size = 4
        args.log_every = 20
        args.save_every = 10**9     # never saves during smoke test

    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from torch.utils.data import DataLoader

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
        from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig
        from lerobot.configs.types import FeatureType, PolicyFeature
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
        from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
        from lerobot.configs.types import FeatureType, PolicyFeature

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}  "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")
    else:
        print("  WARNING: no CUDA available — training will be very slow.")
    print(f"HF_LEROBOT_HOME = {os.environ.get('HF_LEROBOT_HOME')}")
    print(f"HF_DATASETS_CACHE = {os.environ.get('HF_DATASETS_CACHE', '(unset, default ~/.cache)')}")
    print(f"Dataset:        {args.repo_id}")
    print(f"Checkpoint dir: {args.checkpoint_dir}")
    if args.smoke_test:
        print("=== SMOKE TEST: 200 steps, batch 4, no checkpoint saves. ===")

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 1. Dataset metadata
    dataset_meta = LeRobotDataset(args.repo_id).meta
    fps = dataset_meta.fps
    print(f"  episodes: {dataset_meta.total_episodes}  frames: {dataset_meta.total_frames}  fps: {fps}")

    # 2. Dataset with delta_timestamps (DP needs past obs + future actions)
    N_OBS_STEPS    = 2
    HORIZON        = 16
    N_ACTION_STEPS = 8
    obs_dt    = [(t - (N_OBS_STEPS - 1)) / fps for t in range(N_OBS_STEPS)]
    action_dt = [t / fps for t in range(HORIZON)]
    delta_timestamps = {
        "observation.state":       obs_dt,
        "observation.images.base": obs_dt,
        "action":                  action_dt,
    }
    if "observation.images.wrist" in dataset_meta.features:
        delta_timestamps["observation.images.wrist"] = obs_dt

    dataset = LeRobotDataset(args.repo_id, delta_timestamps=delta_timestamps)
    print(f"  training frames: {len(dataset)}")

    # 3. Configure policy — PRD §6 overrides
    cfg = DiffusionConfig(
        n_obs_steps=N_OBS_STEPS,
        horizon=HORIZON,
        n_action_steps=N_ACTION_STEPS,
        vision_backbone="resnet18",
        resize_shape=(96, 128),
        crop_shape=(84, 84),
        crop_is_random=True,
        noise_scheduler_type="DDPM",
        num_train_timesteps=100,
        num_inference_steps=10,
        prediction_type="epsilon",
        optimizer_lr=args.lr,
    )
    all_feats = make_policy_features(dataset.features, FeatureType, PolicyFeature)
    cfg.input_features  = {k: v for k, v in all_feats.items() if k != "action"}
    cfg.output_features = {"action": all_feats["action"]}

    policy = DiffusionPolicy(cfg, dataset_stats=dataset.meta.stats)
    policy.to(device); policy.train()
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"Policy: {n_params/1e6:.1f}M parameters")

    # 4. Resume from latest checkpoint
    start_step = 0
    if not args.no_resume:
        ckpts = sorted(args.checkpoint_dir.glob("step_*"))
        if ckpts:
            latest = ckpts[-1]
            print(f"Resuming from {latest}")
            policy = DiffusionPolicy.from_pretrained(latest)
            policy.to(device); policy.train()
            try:
                start_step = int(latest.name.split("_")[1])
            except ValueError:
                start_step = 0
            print(f"  resumed at step {start_step}")
        else:
            print("No checkpoint found — training from scratch.")
    else:
        print("--no-resume set — training from scratch.")

    # 5. Optimizer + dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    def cycle(loader):
        while True:
            for batch in loader:
                yield batch
    step_iter = cycle(dataloader)

    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=cfg.optimizer_lr,
        betas=cfg.optimizer_betas,
        eps=cfg.optimizer_eps,
        weight_decay=cfg.optimizer_weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_steps)
    for _ in range(start_step):
        scheduler.step()

    print(f"Will run steps {start_step} → {args.num_steps}  bs={args.batch_size}")

    # 6. Train
    loss_history = []
    t0 = time.time()

    for step in range(start_step, args.num_steps):
        batch = next(step_iter)
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}

        # DP returns (loss, None). See PRD §7.4.
        loss, _ = policy.forward(batch)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())

        if step % args.log_every == 0:
            elapsed = time.time() - t0
            rate = (step - start_step + 1) / max(elapsed, 1e-6)
            print(f"step {step:6d}  loss={loss.item():.4f}  "
                  f"lr={scheduler.get_last_lr()[0]:.2e}  ({rate:.1f} step/s)")

        if (step + 1) % args.save_every == 0:
            ckpt_path = args.checkpoint_dir / f"step_{step+1:06d}"
            policy.save_pretrained(ckpt_path)
            print(f"  -> saved {ckpt_path}")

    # 7. Final save (skipped during smoke test)
    if not args.smoke_test:
        final_ckpt = args.checkpoint_dir / "final"
        policy.save_pretrained(final_ckpt)
        print(f"\nDone. Final: {final_ckpt}")
    else:
        # Smoke-test diagnostic: did loss go down at all?
        first10 = float(np.mean(loss_history[:10])) if len(loss_history) >= 10 else float("nan")
        last10  = float(np.mean(loss_history[-10:])) if len(loss_history) >= 10 else float("nan")
        print(f"\nSmoke test done. mean loss first 10 steps={first10:.4f}  "
              f"last 10 steps={last10:.4f}  (should be lower)")

    # 8. Loss curve
    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(loss_history, alpha=0.4, label="raw")
        window = max(1, len(loss_history) // 100)
        if len(loss_history) >= window:
            smoothed = np.convolve(loss_history, np.ones(window) / window, mode="valid")
            ax.plot(np.arange(window - 1, len(loss_history)), smoothed, label=f"smoothed (w={window})")
        ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.legend(); ax.grid(True, alpha=0.3)
        ax.set_title("Diffusion Policy training loss")
        plot_path = args.checkpoint_dir / ("loss_smoke.png" if args.smoke_test else "loss.png")
        plt.tight_layout(); plt.savefig(plot_path); plt.close(fig)
        print(f"Loss curve: {plot_path}")
    except Exception as e:
        print(f"(skipping loss plot: {e})")


if __name__ == "__main__":
    main()
