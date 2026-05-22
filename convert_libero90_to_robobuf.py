"""
Convert LIBERO-90 HDF5 demonstrations to the robobuf format expected by dit-policy.

The output buffer matches the setup used in the existing DSRL checkpoint
(FINAL_bc_ac10_resnet18_onlyimcond_iter150k_seed20346):
  - state  : robot_states (9-dim) = gripper(2) + ee_pos(3) + ee_ori_quat(4)
  - cam0   : agentview_rgb   → BGR, resized 256×256, JPEG-encoded
  - cam1   : eye_in_hand_rgb → BGR, resized 256×256, JPEG-encoded
  - action : 7-DOF OSC_POSE (already in [-1, 1] → identity norm by default)

Usage:
    python convert_libero90_to_robobuf.py \
        --input_dir  /home/raymond112514/LIBERO/scripts/datasets/libero_90 \
        --output_dir /home/raymond112514/DSRL/data_buffers \
        --num_workers 16

    # skip eye-in-hand camera (1-cam setup)
    python convert_libero90_to_robobuf.py ... --no_second_cam

    # apply min-max normalization instead of identity
    python convert_libero90_to_robobuf.py ... --normalize
"""

import argparse
import glob
import json
import os
import pickle as pkl
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import cv2
import h5py
import numpy as np
from tqdm import tqdm

IMAGE_SIZE = (256, 256)


# ── image helpers ──────────────────────────────────────────────────────────────

def _encode(rgb_img: np.ndarray, size: tuple = IMAGE_SIZE) -> np.ndarray:
    """Resize an RGB uint8 image and return a JPEG-encoded byte array."""
    bgr = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)
    _, encoded = cv2.imencode(".jpg", bgr)
    return encoded


# ── normalization helpers ──────────────────────────────────────────────────────

def _identity_norm(ac_dim: int):
    """No-op normalization — actions already in [-1, 1]."""
    return dict(loc=[0.0] * ac_dim, scale=[1.0] * ac_dim)


def _minmax_norm(all_acs: list):
    """Normalize each action dimension to [-1, 1] using per-dim min/max."""
    arr = np.array(all_acs)          # (N, ac_dim)
    lo, hi = arr.min(0), arr.max(0)
    mid = (hi + lo) / 2.0
    scale = (hi - lo) / 2.0
    scale[scale < 1e-8] = 1.0       # avoid div-by-zero for constant dims
    return dict(loc=mid.tolist(), scale=scale.tolist())


def _value_summary(val) -> str:
    """One-line description of a scalar or array value."""
    if isinstance(val, np.ndarray):
        return f"ndarray dtype={val.dtype}, shape={val.shape}"
    if isinstance(val, (bytes, bytearray)):
        return f"bytes len={len(val)}"
    if isinstance(val, str):
        s = val if len(val) <= 60 else val[:57] + "..."
        return f"str '{s}'"
    return f"{type(val).__name__} {val!r}"


def print_robobuf_structure(
    all_trajs: list,
    *,
    ac_dict: Optional[dict] = None,
    buf_path: Optional[str] = None,
    norm_path: Optional[str] = None,
    max_example_steps: int = 2,
) -> None:
    """
    Print a rough schema of the robobuf pickle (list of trajectories).

    Each trajectory is a list of (obs_dict, action, reward) transitions.
    """
    if not all_trajs:
        print("Buffer is empty.")
        return

    traj_lens = [len(t) for t in all_trajs]
    sample_obs, sample_ac, sample_r = all_trajs[0][0]
    cam_keys = sorted(k for k in sample_obs if k.startswith("enc_cam_"))

    print(f"\n{'='*60}")
    print("Robobuf structure (buf.pkl)")
    print(f"{'='*60}")
    print("Top level:")
    print(f"  list[trajectory]  len={len(all_trajs)}")
    print("  trajectory:")
    print("    list[(obs_dict, action, reward)]")
    print("  obs_dict keys (per transition):")
    for key in sorted(sample_obs.keys()):
        print(f"    {key:12s} → {_value_summary(sample_obs[key])}")
    print("  action:")
    print(f"    {_value_summary(sample_ac)}")
    print(f"  reward: float (example {sample_r})")

    print("\nDataset stats:")
    print(f"  total steps     : {sum(traj_lens)}")
    print(f"  steps / traj    : min={min(traj_lens)}, max={max(traj_lens)}, "
          f"mean={np.mean(traj_lens):.1f}")
    print(f"  cameras         : {cam_keys if cam_keys else '(none)'}")

    task_names = {t[0][0].get("task_name") for t in all_trajs if t}
    print(f"  unique tasks    : {len(task_names)}")
    if len(task_names) <= 5:
        for name in sorted(task_names):
            print(f"    - {name}")
    else:
        for name in sorted(task_names)[:3]:
            print(f"    - {name}")
        print(f"    ... ({len(task_names) - 3} more)")

    if buf_path:
        print(f"\nSaved files:")
        print(f"  buffer : {buf_path}")
    if norm_path and ac_dict is not None:
        print(f"  norm   : {norm_path}")
        print(f"  ac_norm: loc={ac_dict['loc'][:3]}... scale={ac_dict['scale'][:3]}... "
              f"(dim={len(ac_dict['loc'])})")

    print("\nExample trajectory[0] (first transitions):")
    for i, (obs, action, reward) in enumerate(all_trajs[0][:max_example_steps]):
        print(f"  [{i}] task_name={obs.get('task_name')!r} demo_key={obs.get('demo_key')!r} "
              f"traj_t={obs.get('traj_t')} reward={reward}")
        print(f"       state={np.array2string(obs['state'], precision=3, suppress_small=True)}")
        print(f"       action={np.array2string(action, precision=3, suppress_small=True)}")
        for ck in cam_keys:
            enc = obs[ck]
            print(f"       {ck}: JPEG bytes, len={len(enc)}")
    if len(all_trajs[0]) > max_example_steps:
        print(f"  ... ({len(all_trajs[0]) - max_example_steps} more steps in traj 0)")
    print(f"{'='*60}\n")


# ── per-demo worker (runs in subprocess) ──────────────────────────────────────

def _process_demo(args: tuple):
    """
    Worker: convert one demo from an HDF5 file into a robobuf trajectory.

    Returns:
        traj      : list of (obs_dict, action, reward)
        demo_acs  : list of raw action arrays (for computing norm stats)
    """
    hdf5_path, demo_key, use_second_cam, task_name = args

    with h5py.File(hdf5_path, "r") as f:
        demo = f["data"][demo_key]
        T = demo["actions"].shape[0]

        # load everything into memory (avoids repeated HDF5 reads in the loop)
        robot_states = demo["robot_states"][:]        # (T, 9)
        actions      = demo["actions"][:]             # (T, 7)
        rewards      = demo["rewards"][:]             # (T,)
        cam0_imgs    = demo["obs/agentview_rgb"][:]   # (T, 128, 128, 3)
        if use_second_cam:
            cam1_imgs = demo["obs/eye_in_hand_rgb"][:] # (T, 128, 128, 3)

    traj, demo_acs = [], []
    for t in range(T):
        obs = {
            "state": robot_states[t].astype(np.float32),
            "task_name": task_name,
            "demo_key": demo_key,
            "traj_t": t,
            "enc_cam_0": _encode(cam0_imgs[t]),
        }
        if use_second_cam:
            obs["enc_cam_1"] = _encode(cam1_imgs[t])

        action = actions[t].astype(np.float32)
        demo_acs.append(action.copy())
        traj.append((obs, action, float(rewards[t])))

    return traj, demo_acs


# ── main conversion ────────────────────────────────────────────────────────────

def convert(input_dir: str,
            output_dir: str,
            use_second_cam: bool = True,
            normalize: bool = False,
            num_workers: int = 8):

    os.makedirs(output_dir, exist_ok=True)

    hdf5_files = sorted(glob.glob(os.path.join(input_dir, "*.hdf5")))
    if not hdf5_files:
        raise FileNotFoundError(f"No .hdf5 files found in {input_dir}")
    print(f"Found {len(hdf5_files)} HDF5 files in {input_dir}")

    # collect (hdf5_path, demo_key, use_second_cam) for every demo across all tasks
    work_items = []
    for hdf5_path in hdf5_files:
        task_name = os.path.basename(hdf5_path)
        with h5py.File(hdf5_path, "r") as f:
            for key in f["data"].keys():
                work_items.append((hdf5_path, key, use_second_cam, task_name))

    print(f"Total demos to convert: {len(work_items)}")
    print(f"Second camera (eye_in_hand): {'yes' if use_second_cam else 'no'}")
    print(f"Parallel workers: {num_workers}")

    all_trajs, all_acs = [], []

    with ProcessPoolExecutor(max_workers=num_workers) as pool:
        futures = {pool.submit(_process_demo, item): item for item in work_items}
        for fut in tqdm(as_completed(futures), total=len(work_items),
                        desc="Converting demos"):
            hdf5_path, demo_key, _, _ = futures[fut]
            try:
                traj, demo_acs = fut.result()
                all_trajs.append(traj)
                all_acs.extend(demo_acs)
            except Exception as exc:
                print(f"  ERROR in {os.path.basename(hdf5_path)} / {demo_key}: {exc}")
                raise

    # ── action normalization ────────────────────────────────────────────────
    ac_dim = all_acs[0].shape[0]
    if normalize:
        print(f"\nApplying min-max normalization over {len(all_acs)} actions...")
        ac_dict = _minmax_norm(all_acs)
        loc   = np.array(ac_dict["loc"],   dtype=np.float32)
        scale = np.array(ac_dict["scale"], dtype=np.float32)
        for traj in tqdm(all_trajs, desc="Normalizing"):
            for obs, action, _ in traj:
                action -= loc
                action /= scale
    else:
        print("\nUsing identity normalization (actions already in [-1, 1]).")
        ac_dict = _identity_norm(ac_dim)

    # ── save ac_norm.json ────────────────────────────────────────────────────
    norm_path = os.path.join(output_dir, "ac_norm.json")
    with open(norm_path, "w") as f:
        json.dump(ac_dict, f, indent=2)
    print(f"Saved action norm stats → {norm_path}")

    # ── save buf.pkl ─────────────────────────────────────────────────────────
    buf_path = os.path.join(output_dir, "buf.pkl")
    print(f"\nSaving {len(all_trajs)} trajectories → {buf_path} ...")
    with open(buf_path, "wb") as f:
        pkl.dump(all_trajs, f)

    print_robobuf_structure(
        all_trajs,
        ac_dict=ac_dict,
        buf_path=buf_path,
        norm_path=norm_path,
    )


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Convert LIBERO-90 HDF5 demos to robobuf (buf.pkl) format."
    )
    p.add_argument(
        "--input_dir",
        default="/home/raymond112514/LIBERO/scripts/datasets/libero_90",
        help="Directory containing the 90 LIBERO .hdf5 files",
    )
    p.add_argument(
        "--output_dir",
        default="/home/raymond112514/DSRL/data_buffers",
        help="Output directory for buf.pkl and ac_norm.json",
    )
    p.add_argument(
        "--no_second_cam",
        action="store_true",
        help="Only include agentview camera (cam0); skip eye_in_hand (cam1)",
    )
    p.add_argument(
        "--normalize",
        action="store_true",
        help="Apply per-dim min-max normalization. "
             "Default: identity (actions already in [-1, 1])",
    )
    p.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of parallel worker processes (default: 8)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    convert(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        use_second_cam=not args.no_second_cam,
        normalize=args.normalize,
        num_workers=args.num_workers,
    )
