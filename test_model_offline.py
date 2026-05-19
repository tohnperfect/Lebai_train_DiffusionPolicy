"""Test a trained Diffusion Policy against the training dataset itself.

Two complementary checks:

  1. Forward-pass eval (--mode forward, default)
     For N random frames, runs policy.select_action(obs) and compares the
     prediction to the ground-truth action. Use this to confirm the policy
     can reproduce frames it was trained on, and — crucially — to detect
     collapse-to-constant (the ACT failure mode we already burned a run on).

  2. Open-loop rollout (--mode rollout)
     For a chosen episode, walks through every frame sequentially the same
     way run_inference.py does. Returns a CSV of (frame, state, gt action,
     predicted action) so you can compare trajectories visually.

Run from a fresh shell (NOT the same process the converter ran in):

    python test_model_offline.py \\
        --checkpoint checkpoints/dp_run01/final \\
        --repo-id local/lebai_duck_pick_delta_x100 \\
        --n-frames 50

    python test_model_offline.py \\
        --checkpoint checkpoints/dp_run01/final \\
        --mode rollout --episode 0 \\
        --out /tmp/rollout_ep0.csv

A healthy model:
  - per-dim L1 well below the std of the corresponding action dimension
  - mean prediction std (across the sample) close to ground-truth std
    (if pred std << gt std, the model collapsed to predicting a constant)
  - rollout CSV shows predicted trajectory visually tracking the ground truth

A collapsed model:
  - per-dim L1 ≈ std of the action (predictions are basically the mean)
  - pred std ≈ 0 (everything mapped to the same constant)
  - rollout trajectory is flat or nearly flat across an episode

DP-specific notes:
  - Inference is slow (10 DDIM steps per select_action, possibly more with
    DDPM). 50 random frames takes a few minutes on a GPU; minutes-to-tens
    on CPU. Lower --n-frames if the wait is unacceptable.
  - DP uses policy.config.horizon (action steps predicted per forward pass),
    not chunk_size like ACT.
  - policy.reset() between random frames is critical — without it the second
    sample inherits the queue from the first.
"""

import argparse
import csv
import os
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--checkpoint", type=Path, required=True,
                   help="Path to a trained DiffusionPolicy checkpoint directory.")
    p.add_argument("--dataset-path", type=Path, default=Path("./result"),
                   help="HF_LEROBOT_HOME root. Default: ./result")
    p.add_argument("--repo-id", default="local/lebai_duck_pick_delta_x100_g100",
                   help="Dataset repo-id under --dataset-path. "
                        "Default: local/lebai_duck_pick_delta_x100_g100 "
                        "(relative mode, joint scale 100, gripper scale 0.01).")
    p.add_argument("--mode", choices=["forward", "rollout"], default="forward",
                   help="forward: random-frame eval (default). rollout: one full episode.")
    p.add_argument("--n-frames", type=int, default=50,
                   help="Frames to sample in --mode forward. Default: 50 "
                        "(DP inference is slow — keep modest).")
    p.add_argument("--episode", type=int, default=0,
                   help="Episode index to roll out in --mode rollout. Default: 0")
    p.add_argument("--seed", type=int, default=42, help="RNG seed for sampling.")
    p.add_argument("--out", type=Path, default=None,
                   help="CSV path for rollout output. Default: ./rollout_ep{N}.csv")
    return p.parse_args()


def expected_input_keys(policy):
    """Best-effort: return the set of observation keys the policy was trained on.

    Used for the diagnostic printout only. lerobot 0.5.x exposes this as
    `policy.config.input_features.keys()` for DP; fall back to other attribute
    layouts if that fails.
    """
    cfg = policy.config
    try:
        return sorted(cfg.input_features.keys())
    except Exception:
        keys = []
        for attr in ("image_features", "state_feature"):
            v = getattr(cfg, attr, None)
            if isinstance(v, dict):
                keys.extend(v.keys())
            elif v is not None:
                # state_feature is typically a single PolicyFeature
                keys.append("observation.state")
        return sorted(set(keys))


def main():
    args = parse_args()
    if not args.checkpoint.exists():
        sys.exit(f"Checkpoint not found: {args.checkpoint}")

    # HF_LEROBOT_HOME MUST be set before importing lerobot — it's read at import.
    os.environ["HF_LEROBOT_HOME"] = str(args.dataset_path.resolve())

    import numpy as np
    import torch
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
    except ImportError:
        from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy

    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"Loading {args.checkpoint}  (device: {device})")
    policy = DiffusionPolicy.from_pretrained(args.checkpoint)
    policy.to(device).eval()
    horizon = policy.config.horizon
    n_action_steps = policy.config.n_action_steps
    print(f"  horizon={horizon}  n_action_steps={n_action_steps}  "
          f"n_obs_steps={policy.config.n_obs_steps}  "
          f"inference_steps={policy.config.num_inference_steps}")
    print(f"  expects={expected_input_keys(policy)}")

    # Build dataset with chunked actions: sample["action"] becomes (horizon, action_dim).
    base = LeRobotDataset(args.repo_id)
    fps = base.meta.fps
    delta_timestamps = {"action": [t / fps for t in range(horizon)]}
    ds = LeRobotDataset(args.repo_id, delta_timestamps=delta_timestamps)
    print(f"\nDataset: {args.repo_id}")
    print(f"  episodes: {base.meta.total_episodes}  frames: {base.meta.total_frames}  fps: {fps}")

    if args.mode == "forward":
        forward_eval(args, ds, policy, device, np, torch)
    else:
        rollout_eval(args, ds, base, policy, device, np, torch)


def forward_eval(args, ds, policy, device, np, torch):
    rng = np.random.default_rng(args.seed)
    n = min(args.n_frames, len(ds))
    indices = rng.choice(len(ds), size=n, replace=False)

    preds, targets = [], []
    print(f"\nRunning forward pass on {n} random frames "
          f"(this is slow for DP — denoising loop runs per call) ...")

    with torch.inference_mode():
        for k, idx in enumerate(indices):
            sample = ds[int(idx)]
            # Critical for DP: reset the internal action queue. Without this,
            # the second random frame's prediction is contaminated by the
            # queue populated from the first frame's forward pass.
            policy.reset()
            obs = {k_: v.unsqueeze(0).to(device) for k_, v in sample.items()
                   if k_.startswith("observation.")}
            pred = policy.select_action(obs)
            preds.append(pred[0].cpu().numpy())
            # First step of the chunk = ground-truth action for this frame.
            targets.append(sample["action"][0].cpu().numpy())
            if (k + 1) % 10 == 0:
                print(f"  {k + 1}/{n}")

    preds = np.stack(preds)        # (n, action_dim)
    targets = np.stack(targets)    # (n, action_dim)

    action_dim = preds.shape[1]
    print("\n=== Per-dimension stats ===")
    print(f"{'dim':>4}  {'gt_mean':>10}  {'gt_std':>10}  {'pred_mean':>10}  "
          f"{'pred_std':>10}  {'L1':>8}  {'L1/gt_std':>10}  {'corr':>6}")
    for d in range(action_dim):
        gt = targets[:, d]
        pr = preds[:, d]
        l1 = np.mean(np.abs(pr - gt))
        gt_std = gt.std()
        ratio = l1 / gt_std if gt_std > 1e-6 else float("inf")
        corr = np.corrcoef(pr, gt)[0, 1] if pr.std() > 1e-6 else 0.0
        label = "(gripper)" if d == 6 else ""
        print(f"{d:>4}  {gt.mean():10.4f}  {gt_std:10.4f}  {pr.mean():10.4f}  "
              f"{pr.std():10.4f}  {l1:8.4f}  {ratio:10.3f}  {corr:6.3f} {label}")

    overall_l1 = float(np.mean(np.abs(preds - targets)))
    print(f"\nOverall mean L1: {overall_l1:.4f}")
    print()
    print("Interpretation:")
    print("  L1 / gt_std  <  ~0.3   ->  model is learning that dimension well")
    print("  L1 / gt_std  >  ~0.8   ->  predictions are near the dataset mean (collapse)")
    print("  pred_std    <<  gt_std ->  model collapsed; predicts a constant")
    print("  corr near 1            ->  predictions track ground truth")
    print("  corr near 0            ->  predictions uncorrelated with ground truth")


def rollout_eval(args, ds_chunked, ds_single, policy, device, np, torch):
    ep_meta = ds_single.meta.episodes[args.episode]
    fr = int(ep_meta["dataset_from_index"])
    to = int(ep_meta["dataset_to_index"])
    print(f"\nRolling out episode {args.episode}: frames [{fr}, {to})  "
          f"length={to - fr}")

    out_path = args.out or Path(f"./rollout_ep{args.episode}.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    f = open(out_path, "w", newline="")
    w = csv.writer(f)

    action_dim = ds_single.features["action"]["shape"][0]
    state_dim = ds_single.features["observation.state"]["shape"][0]
    w.writerow(
        ["frame"]
        + [f"state_{i}" for i in range(state_dim)]
        + [f"gt_action_{i}" for i in range(action_dim)]
        + [f"pred_action_{i}" for i in range(action_dim)]
    )

    # Single reset for the whole episode — temporal carry-over IS the test
    # for rollout mode, just like real inference.
    policy.reset()
    diffs = []
    preds = []
    with torch.inference_mode():
        for i, abs_idx in enumerate(range(fr, to)):
            sample = ds_single[abs_idx]
            obs = {
                k: v.unsqueeze(0).to(device) for k, v in sample.items()
                if k.startswith("observation.")
            }
            pred = policy.select_action(obs)
            pred_np = pred[0].cpu().numpy()
            gt = sample["action"].cpu().numpy()
            state = sample["observation.state"].cpu().numpy()
            w.writerow([i] + list(state) + list(gt) + list(pred_np))
            diffs.append(pred_np - gt)
            preds.append(pred_np)
            if (i + 1) % 100 == 0:
                print(f"  {i + 1}/{to - fr}")
    f.close()

    diffs = np.stack(diffs)
    preds = np.stack(preds)
    print(f"\nRollout saved to {out_path}")
    print(f"Per-dim mean abs error (pred - gt): {np.round(np.mean(np.abs(diffs), axis=0), 4)}")
    print(f"Per-dim pred std:                   {np.round(preds.std(axis=0), 4)}")
    print()
    print("Quick visual: open the CSV in any plotting tool and overlay")
    print("  gt_action_0..5  vs  pred_action_0..5 over the frame axis.")
    print("  If predicted trajectories track the gt trajectories, the model learned.")
    print("  If predicted trajectories are flat (constant), the model collapsed.")


if __name__ == "__main__":
    main()
