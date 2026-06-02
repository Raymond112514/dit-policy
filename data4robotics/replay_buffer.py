# Copyright (c) Sudeep Dasari, 2023

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import pickle as pkl
import random
import shutil

import numpy as np
import torch
import tqdm
from robobuf import ReplayBuffer as RB
from torch.utils.data import Dataset, IterableDataset

from data4robotics.task_conditioning import TaskConditioningEncoder

# cache loading from the buffer list to half memory overhead
buf_cache = dict()
BUF_SHUFFLE_RNG = 3904767649


# helper functions
_img_to_tensor = (
    lambda x: torch.from_numpy(x.copy()).permute((0, 3, 1, 2)).float() / 255
)
_to_tensor = lambda x: torch.from_numpy(x).float()


def _cached_load(path):
    global buf_cache

    if path in buf_cache:
        return buf_cache[path]

    with open(path, "rb") as f:
        buf = RB.load_traj_list(pkl.load(f))
    buf_cache[path] = buf
    return buf


def _get_imgs(t, cam_idx, past_frames):
    imgs = []
    while len(imgs) < past_frames + 1:
        imgs.append(t.obs.image(cam_idx)[None])

        if t.prev is not None:
            t = t.prev
    return np.concatenate(imgs, axis=0)


class IterableWrapper(IterableDataset):
    def __init__(self, wrapped_dataset, max_count=float("inf")):
        self.wrapped = wrapped_dataset
        self.ctr, self.max_count = 0, max_count

    def __iter__(self):
        self.ctr = 0
        return self

    def __next__(self):
        if self.ctr > self.max_count:
            raise StopIteration

        self.ctr += 1
        idx = int(np.random.choice(len(self.wrapped)))
        return self.wrapped[idx]


class RobobufReplayBuffer(Dataset):
    def __init__(
        self,
        buffer_path,
        transform=None,
        n_test_trans=500,
        mode="train",
        ac_chunk=1,
        cam_indexes=[0],
        goal_indexes=[],
        goal_geom_prob=0.01,
        past_frames=0,
        ac_dim=7,
        task_conditioning=False,
        task_conditioning_mode=None,
        language_embedding_path=None,
        vae_checkpoint="",
        max_transitions=-1,
    ):
        assert mode in ("train", "test"), "Mode must be train/test"
        buf = _cached_load(buffer_path)
        assert len(buf) > n_test_trans, "Not enough transitions!"

        norm_file = os.path.join(os.path.dirname(buffer_path), "ac_norm.json")
        if os.path.exists(norm_file):
            shutil.copyfile(norm_file, "./ac_norm.json")

        # shuffle the list with the fixed seed
        rng = random.Random(BUF_SHUFFLE_RNG)

        # get and shuffle list of buf indices
        index_list = list(range(len(buf)))
        rng.shuffle(index_list)

        # split data according to mode
        index_list = (
            index_list[n_test_trans:] if mode == "train" else index_list[:n_test_trans]
        )
        if max_transitions > 0:
            index_list = index_list[:max_transitions]

        self.transform = transform
        self.s_a_mask = []

        self.cam_indexes = cam_indexes = list(cam_indexes)
        self.past_frames = past_frames
        print(f"Building {mode} buffer with cam_indexes={cam_indexes}")

        self.goal_geom_prob = goal_geom_prob
        self.goal_indexes = set(goal_indexes)
        assert all([g in self.cam_indexes for g in self.goal_indexes])

        self.task_conditioning = task_conditioning
        self._task_encoder = None
        if self.task_conditioning:
            self._task_encoder = TaskConditioningEncoder(
                mode=task_conditioning_mode,
                language_embedding_path=language_embedding_path,
            )
            print(
                f"Task conditioning enabled: mode={task_conditioning_mode}, "
                f"dim={self._task_encoder.dim}"
            )

        for idx in tqdm.tqdm(index_list):
            t = buf[idx]

            loop_t, chunked_actions, loss_mask = t, [], []
            for _ in range(ac_chunk):
                chunked_actions.append(loop_t.action[None])
                loss_mask.append(1.0)

                if loop_t.next is None:
                    break
                loop_t = loop_t.next

            if len(chunked_actions) < ac_chunk:
                for _ in range(ac_chunk - len(chunked_actions)):
                    chunked_actions.append(chunked_actions[-1])
                    loss_mask.append(0.0)

            a_t = np.concatenate(chunked_actions, 0).astype(np.float32)
            if ac_dim != a_t.shape[-1]:
                raise ValueError(
                    f"Action dimension mismatch: expected ac_dim={ac_dim}, "
                    f"but found action shape={a_t.shape} (last dim={a_t.shape[-1]}). "
                    f"Please check your config or buffer data."
                )

            loss_mask = np.array(loss_mask, dtype=np.float32)
            obs_meta = t.obs.obs if self.task_conditioning else None
            self.s_a_mask.append((t, a_t, loss_mask, loop_t, obs_meta))

        self.use_vae = False
        self.z_dim = None
        if vae_checkpoint:
            self._encode_with_vae(vae_checkpoint)

    def _encode_with_vae(self, vae_checkpoint: str, encode_batch_size: int = 4096) -> None:
        import torch
        from data4robotics.vae import load_action_vae, vae_encode_device
        from data4robotics.vae.preprocess import get_vae_image_transform

        device = vae_encode_device()
        print(f"Loading VAE from {vae_checkpoint} on {device} ...")
        ckpt_args = torch.load(
            vae_checkpoint, map_location=device, weights_only=False
        ).get("args", {})
        vae = load_action_vae(vae_checkpoint, device=device)
        self.z_dim = vae.z_dim
        self.use_vae = True

        # Build image preproc if the VAE encoder is image-conditioned
        img_preproc = None
        vae_cam_index = None
        if vae.encoder_use_image:
            img_size = ckpt_args.get("image_size", 128)
            vae_cam_index = ckpt_args.get("cam_index", 0)
            img_preproc = get_vae_image_transform(img_size)
            print(f"  VAE encoder is image-conditioned (cam={vae_cam_index}, size={img_size})")

        print(f"Pre-encoding {len(self.s_a_mask)} transitions into z_dim={self.z_dim} latents ...")
        new_s_a_mask = []
        n = len(self.s_a_mask)

        for start in tqdm.tqdm(range(0, n, encode_batch_size), desc="VAE encoding"):
            entries = self.s_a_mask[start: start + encode_batch_size]

            actions = torch.stack([_to_tensor(e[1]) for e in entries])  # (B, ac_chunk, ac_dim)

            proprio = None
            if vae.encoder_use_proprio:
                proprio = torch.stack([_to_tensor(e[0].obs.state) for e in entries])

            images = None
            if vae.encoder_use_image:
                # Stack as numpy first, single tensor conversion + batch transform
                imgs_np = np.stack([e[0].obs.image(vae_cam_index) for e in entries])  # (B, H, W, C)
                imgs_raw = torch.from_numpy(imgs_np.copy()).permute(0, 3, 1, 2).float() / 255  # (B, C, H, W)
                images = img_preproc(imgs_raw)  # (B, C, H, W)

            actions = actions.to(device)
            if proprio is not None:
                proprio = proprio.to(device)
            if images is not None:
                images = images.to(device)

            with torch.no_grad():
                mu, _ = vae.encode(actions, proprio=proprio, images=images)  # (B, z_dim)

            ones_mask = np.ones(self.z_dim, dtype=np.float32)
            for i, (t, _a_t, _mask, loop_t, obs_meta) in enumerate(entries):
                z = mu[i].cpu().numpy()  # (z_dim,)
                new_s_a_mask.append((t, z, ones_mask, loop_t, obs_meta))

        self.s_a_mask = new_s_a_mask

    def __len__(self):
        return len(self.s_a_mask)

    def __getitem__(self, idx):
        step, a_t, loss_mask, goal, obs_meta = self.s_a_mask[idx]  # obs_meta may be None

        if self.goal_indexes:
            while np.random.uniform() > self.goal_geom_prob and goal.next is not None:
                goal = goal.next

        i_t, o_t = dict(), step.obs.state
        for idx, cam_idx in enumerate(self.cam_indexes):
            i_c = _get_imgs(step, cam_idx, self.past_frames)
            if self.goal_indexes:
                g_c = (
                    _get_imgs(goal, cam_idx, 0)
                    if cam_idx in self.goal_indexes
                    else np.zeros_like(i_c[:1])
                )
                i_c = np.concatenate((g_c, i_c), axis=0)

            i_c = _img_to_tensor(i_c)
            if self.transform is not None:
                i_c = self.transform(i_c)

            i_t[f"cam{idx}"] = i_c

        o_t, a_t = _to_tensor(o_t), _to_tensor(a_t)
        if self.use_vae:
            # a_t is (z_dim,), loss_mask is (z_dim,) all-ones — no chunk expansion needed
            loss_mask = _to_tensor(loss_mask)
        else:
            loss_mask = _to_tensor(loss_mask)[:, None].repeat((1, a_t.shape[-1]))
            assert (
                loss_mask.shape[0] == a_t.shape[0]
            ), "a_t and mask shape must be ac_chunk!"

        if self.task_conditioning:
            task_vec = self._task_encoder.encode_transition_obs(obs_meta)
            task_t = torch.from_numpy(task_vec)
            return (i_t, o_t), a_t, loss_mask, task_t
        return (i_t, o_t), a_t, loss_mask

