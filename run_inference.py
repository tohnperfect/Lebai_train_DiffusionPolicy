"""Run a trained Diffusion Policy on the live Lebai LM3 arm.

Run from a machine on the same LAN as the robot and camera service.

    # Predict one action without moving the robot — always run this first.
    python run_inference.py --dry-run

    # Real run, 30 s control loop at 10 Hz.
    SAFETY_OK=1 python run_inference.py --duration 30

    # Verbose every-tick logging:
    SAFETY_OK=1 python run_inference.py --duration 30 --verbose

    # Try a different checkpoint:
    python run_inference.py --checkpoint ./checkpoints/dp_run01/step_080000 --dry-run

Action mode (must match the converter — PRD §1 of the v2 update):
  - 'relative' (default): the policy outputs scaled joint deltas + scaled gripper.
        target[i]   = state[i] + action[i] / action_delta_scale       (i in 0..5)
        gripper_amp = action[6] / gripper_scale                       (clipped to [0,100])
  - 'absolute': the policy output is the joint target directly; gripper is unscaled.
        target[i]   = action[i]
        gripper_amp = action[6]

A mismatched --action-delta-scale or --gripper-scale produces wildly wrong
targets — verify the dry-run delta + abs targ match the dataset's REPO_ID
conventions before allowing motion.

DP-specific notes vs. ACT (PRD §11):
  - `policy.select_action(obs)` returns one action per call, but only every
    `n_action_steps`-th call does a real forward pass. The policy's internal
    queue handles chunking — do NOT write your own chunking.
  - `policy.reset()` clears the internal action queue.

SAFETY: before each non-dry run, confirm
  1. Workspace is clear, no one within arm sweep range.
  2. E-stop is within reach.
  3. Pendant velocity factor is low.
  4. The arm starts in a pose similar to one of the training episodes' first frames.

The script will not move the arm without the SAFETY_OK environment variable set:

    SAFETY_OK=1 python run_inference.py --duration 30
"""

import argparse
import base64
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests
import torch

# lebai_sdk is imported lazily inside main() — it's only installed on the
# robot machine, but --help and --dry-run --checkpoint /nonexistent should
# still work on a dev machine without it.

try:
    from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
except ImportError:
    from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy


# ---------------------------------------------------------------------------
# Defaults — change here, or override on the command line.

DEFAULT_CHECKPOINT          = "./checkpoints/dp_run01/final"
DEFAULT_ROBOT_IP            = "192.168.31.254"
DEFAULT_CAMERA_URL          = "http://192.168.31.192:8000"
DEFAULT_ACTION_MODE         = "relative"
DEFAULT_ACTION_DELTA_SCALE  = 100.0
DEFAULT_GRIPPER_SCALE       = 0.01     # 0 means "no gripper scaling" (absolute mode)

# Control loop tick + safety
DEFAULT_DURATION_S = 30.0
PERIOD_S           = 0.1                # 10 Hz

# Joint move limits (keep low for first runs)
JOINT_ACC_LIMIT = 1.5    # rad/s^2
JOINT_VEL_LIMIT = 1.0    # rad/s
BLEND_RADIUS    = 0.05   # rad — smooth blending between successive movej calls

# Gripper command rate limiting
GRIPPER_FORCE     = 10
GRIPPER_THRESHOLD = 5.0  # only send set_claw when amplitude changed > this


# ---------------------------------------------------------------------------
# Camera

class CameraClient:
    def __init__(self, url):
        self.url = url.rstrip("/")
        self.s = requests.Session()

    def start_all(self):
        self.s.post(f"{self.url}/cameras/start_all", json={
            "enable_color": True, "enable_depth": False,
            "color_config": {"width": 640, "height": 480, "fps": 30},
        })

    def stop_all(self):
        try:
            self.s.post(f"{self.url}/cameras/stop_all")
        except Exception:
            pass

    def list_cameras(self):
        return self.s.get(f"{self.url}/cameras").json()

    def color_rgb(self, cid):
        d = self.s.get(f"{self.url}/cameras/{cid}/frame/color",
                       params={"format": "jpeg"}).json()
        jpg = base64.b64decode(d["data"])
        bgr = cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# Observation + action plumbing

def build_observation(policy, cam, lebai, base_cid, wrist_cid, expected_keys, device):
    obs = {}

    rgb = cam.color_rgb(base_cid)                  # (H, W, 3) uint8
    img = torch.from_numpy(rgb).float() / 255.0
    img = img.permute(2, 0, 1).unsqueeze(0)        # (1, 3, H, W)
    obs["observation.images.base"] = img.to(device, non_blocking=True)

    if "observation.images.wrist" in expected_keys:
        if wrist_cid is None:
            raise RuntimeError(
                "Policy was trained with a wrist camera but none is connected."
            )
        rgb_w = cam.color_rgb(wrist_cid)
        img_w = torch.from_numpy(rgb_w).float() / 255.0
        img_w = img_w.permute(2, 0, 1).unsqueeze(0)
        obs["observation.images.wrist"] = img_w.to(device, non_blocking=True)

    kin = lebai.get_kin_data()
    jp = kin["actual_joint_pose"]
    if isinstance(jp, dict):
        jp = [jp.get(f"jp{i}", jp.get(f"j{i}", 0.0)) for i in range(6)]
    state_list = list(jp[:6])

    state_shape = policy.config.input_features["observation.state"].shape
    if state_shape[0] == 7:
        try:
            claw = lebai.get_claw()
            amp = float(claw.get("amplitude", 0.0))
        except Exception:
            amp = 0.0
        state_list.append(amp)

    state = torch.tensor(state_list, dtype=torch.float32).unsqueeze(0)
    obs["observation.state"] = state.to(device, non_blocking=True)
    return obs


def resolve_targets(state_np, action_np, action_mode, action_delta_scale, gripper_scale):
    """Convert the policy's raw output into absolute joint targets + gripper amp.

    Relative mode:
        target[i]   = state[i] + action[i] / action_delta_scale     (joints)
        gripper_amp = action[6] / gripper_scale  (then clipped to [0, 100])
    Absolute mode:
        target[i]   = action[i]
        gripper_amp = action[6]  (assumed already in [0, 100])

    Returns (target_joints: list[float] of length 6, gripper_amp: float | None).
    """
    if action_mode == "relative":
        target_joints = [
            float(state_np[i] + action_np[i] / action_delta_scale)
            for i in range(6)
        ]
    else:  # absolute
        target_joints = [float(action_np[i]) for i in range(6)]

    gripper_amp = None
    if len(action_np) >= 7:
        raw = float(action_np[6])
        if action_mode == "relative" and gripper_scale and gripper_scale != 0.0:
            raw = raw / gripper_scale
        gripper_amp = float(np.clip(raw, 0.0, 100.0))
    return target_joints, gripper_amp


_last_gripper_sent = None

def send_targets(target_joints, gripper_amp, lebai):
    """Issue movej and (rate-limited) set_claw. Returns gripper_sent flag."""
    global _last_gripper_sent

    lebai.movej(
        list(target_joints),
        JOINT_ACC_LIMIT, JOINT_VEL_LIMIT,
        0,                   # 't' arg, 0 means "use a/v limits"
        BLEND_RADIUS,
    )

    gripper_sent = False
    if gripper_amp is not None:
        if (_last_gripper_sent is None
                or abs(gripper_amp - _last_gripper_sent) > GRIPPER_THRESHOLD):
            lebai.set_claw(GRIPPER_FORCE, gripper_amp)
            _last_gripper_sent = gripper_amp
            gripper_sent = True
    return gripper_sent


# ---------------------------------------------------------------------------
# Main

def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", type=Path, default=Path(DEFAULT_CHECKPOINT),
                   help=f"Path to the trained Diffusion Policy checkpoint (default: {DEFAULT_CHECKPOINT})")
    p.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP,
                   help=f"Lebai arm IP (default: {DEFAULT_ROBOT_IP})")
    p.add_argument("--camera-url", default=DEFAULT_CAMERA_URL,
                   help=f"Camera service URL (default: {DEFAULT_CAMERA_URL})")
    p.add_argument("--duration", type=float, default=DEFAULT_DURATION_S,
                   help=f"Hard time limit on the control loop in seconds (default: {DEFAULT_DURATION_S})")
    p.add_argument("--action-mode", choices=["relative", "absolute"], default=DEFAULT_ACTION_MODE,
                   help=f"How to interpret the policy output (default: {DEFAULT_ACTION_MODE}). "
                        f"MUST match the converter's ACTION_MODE — mismatch produces wildly wrong targets.")
    p.add_argument("--action-delta-scale", type=float, default=DEFAULT_ACTION_DELTA_SCALE,
                   help=f"Joint-delta multiplier used by the converter (default: {DEFAULT_ACTION_DELTA_SCALE}). "
                        f"Inference divides action[:6] by this to recover real deltas. "
                        f"Ignored in absolute mode. Must match the converter's ACTION_DELTA_SCALE.")
    p.add_argument("--gripper-scale", type=float, default=DEFAULT_GRIPPER_SCALE,
                   help=f"Gripper-amplitude multiplier used by the converter (default: {DEFAULT_GRIPPER_SCALE}). "
                        f"Inference divides action[6] by this to recover the 0-100 amplitude. "
                        f"Ignored in absolute mode. Must match the converter's GRIPPER_SCALE.")
    p.add_argument("--dry-run", action="store_true",
                   help="Predict one action and print it, but do NOT send to the robot.")
    p.add_argument("-v", "--verbose", action="store_true",
                   help="Print state, predicted delta, and resolved absolute target every tick.")
    return p.parse_args()


def print_dry_run(cur_state, action_np, target_joints, gripper_amp, action_mode, action_delta_scale, gripper_scale):
    print("\n=== Dry-run prediction ===")
    if action_mode == "relative":
        print(f"  action_mode=relative  joint_scale={action_delta_scale}  gripper_scale={gripper_scale}")
    else:
        print(f"  action_mode=absolute  scales=n/a")
    print(f"  current state : {np.round(cur_state[:6], 3)}    "
          f"gripper={cur_state[6]:5.1f}" if len(cur_state) >= 7
          else f"  current state : {np.round(cur_state[:6], 3)}")
    print(f"  policy delta  : {np.round(action_np[:6], 4)}    "
          f"gripper={action_np[6]:5.1f}" if len(action_np) >= 7
          else f"  policy delta  : {np.round(action_np[:6], 4)}")
    print(f"  abs target    : {np.round(target_joints, 3)}    "
          f"gripper={gripper_amp:5.1f}" if gripper_amp is not None
          else f"  abs target    : {np.round(target_joints, 3)}")
    abs_delta_rad = np.array(target_joints) - cur_state[:6]
    print(f"  joint move    : {np.round(abs_delta_rad, 4)}   "
          f"(|max|={np.max(np.abs(abs_delta_rad)):.4f} rad)")


def main():
    args = parse_args()

    if not args.checkpoint.exists():
        sys.exit(f"Checkpoint not found: {args.checkpoint}")

    # 1. Load policy
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading {args.checkpoint}  (device: {device})")
    if args.action_mode == "relative":
        print(f"  action_mode=relative  joint_scale={args.action_delta_scale}  "
              f"gripper_scale={args.gripper_scale}")
    else:
        print(f"  action_mode=absolute  scales=n/a")
    policy = DiffusionPolicy.from_pretrained(args.checkpoint)
    policy.to(device).eval()
    print(f"  horizon={policy.config.horizon}  "
          f"n_action_steps={policy.config.n_action_steps}  "
          f"n_obs_steps={policy.config.n_obs_steps}  "
          f"inference_steps={policy.config.num_inference_steps}")
    expected_keys = set(policy.config.input_features.keys())
    print(f"  expects: {sorted(expected_keys)}")

    # 2. Connect camera + robot
    print(f"\nConnecting camera at {args.camera_url} ...")
    cam = CameraClient(args.camera_url)
    cam.start_all()
    cams = cam.list_cameras()
    base_cid  = cams[0]["serial_number"]
    wrist_cid = cams[1]["serial_number"] if len(cams) > 1 else None
    print(f"  base={base_cid}  wrist={wrist_cid}")

    print(f"Connecting robot at {args.robot_ip} ...")
    import lebai_sdk
    lebai_sdk.init()
    lebai = lebai_sdk.connect(args.robot_ip, False)
    lebai.start_sys()
    lebai.init_claw()
    print(f"  connected={lebai.is_connected()}")

    try:
        # 3. Dry run — predict one action, print, don't send.
        policy.reset()    # CRITICAL: clears the internal action queue
        obs = build_observation(policy, cam, lebai, base_cid, wrist_cid,
                                expected_keys, device)
        with torch.inference_mode():
            action = policy.select_action(obs)
        action_np = action[0].cpu().numpy()
        cur_state = obs["observation.state"][0].cpu().numpy()
        target_joints, gripper_amp = resolve_targets(
            cur_state, action_np, args.action_mode, args.action_delta_scale, args.gripper_scale
        )
        print_dry_run(cur_state, action_np, target_joints, gripper_amp,
                      args.action_mode, args.action_delta_scale, args.gripper_scale)

        if args.dry_run:
            print("\n--dry-run set, exiting without moving the robot.")
            return

        # 4. Safety gate
        if os.environ.get("SAFETY_OK") != "1":
            print()
            print("=" * 70)
            print("Refusing to move the robot.")
            print()
            print("Confirm the safety checklist, then re-run with SAFETY_OK=1:")
            print()
            print("  1. Workspace clear, no one within sweep range.")
            print("  2. E-stop within reach.")
            print("  3. Pendant velocity factor turned down.")
            print("  4. Arm in a pose similar to a training episode's first frame.")
            print()
            print(f"  SAFETY_OK=1 python {sys.argv[0]} --duration {args.duration}")
            print("=" * 70)
            return

        # 5. Control loop
        print(f"\n=== Control loop ({args.duration:.1f}s, {1/PERIOD_S:.0f} Hz) ===")
        print("Stop early with Ctrl-C.")
        print("Starting in 3 s ...")
        time.sleep(3)
        print("RUNNING.")

        policy.reset()
        global _last_gripper_sent
        _last_gripper_sent = None

        t_start = time.time()
        next_tick = time.time()
        step = 0

        while True:
            loop_t = time.time()
            if loop_t - t_start > args.duration:
                print(f"Reached duration limit ({args.duration:.1f}s) — stopping.")
                break

            obs = build_observation(policy, cam, lebai, base_cid, wrist_cid,
                                    expected_keys, device)
            with torch.inference_mode():
                action = policy.select_action(obs)
            action_np = action[0].cpu().numpy()
            state_np = obs["observation.state"][0].cpu().numpy()

            target_joints, gripper_amp = resolve_targets(
                state_np, action_np, args.action_mode, args.action_delta_scale, args.gripper_scale
            )
            gripper_sent = send_targets(target_joints, gripper_amp, lebai)

            step += 1
            log_now = args.verbose or (step % 10 == 0)
            if log_now:
                loop_ms = (time.time() - loop_t) * 1000
                if args.verbose:
                    print(f"  t={loop_t - t_start:5.1f}s  step={step:4d}  loop={loop_ms:.0f}ms")
                    s_tail = f"    gripper={state_np[6]:5.1f}" if len(state_np) >= 7 else ""
                    print(f"    state    : {np.round(state_np[:6], 3)}{s_tail}")
                    print(f"    delta    : {np.round(action_np[:6], 4)}")
                    g_tail = ""
                    if gripper_amp is not None:
                        g_tail = f"    gripper={gripper_amp:5.1f} {'(SENT)' if gripper_sent else '(rate-limited)'}"
                    print(f"    abs targ : {np.round(target_joints, 3)}{g_tail}")
                else:
                    msg = (f"  t={loop_t - t_start:5.1f}s  step={step:4d}  "
                           f"loop={loop_ms:.0f}ms  abs={np.round(target_joints, 2)}")
                    if gripper_amp is not None:
                        msg += f"  g={gripper_amp:.0f}"
                    print(msg)

            next_tick += PERIOD_S
            sleep_for = next_tick - time.time()
            if sleep_for > 0:
                time.sleep(sleep_for)
            else:
                next_tick = time.time()    # don't accumulate negative sleep

        print(f"Stopped after {step} steps ({time.time() - t_start:.1f}s).")

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        print("Cleaning up ...")
        cam.stop_all()
        try:
            lebai.stop_sys()
        except Exception:
            pass
        print("Camera and robot stopped.")


if __name__ == "__main__":
    main()
