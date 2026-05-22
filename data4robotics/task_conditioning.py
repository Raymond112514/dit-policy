# Task conditioning encodings for LIBERO-style multi-task training.

import os
import pickle as pkl
from typing import Optional, Union

import numpy as np

from data4robotics.libero_tasks import NUM_TEXT_TASKS, SCENE_NAMES, TASK_IDX, TASK_NAMES

# Replaces legacy scene_task integer flags.
TASK_CONDITIONING_MODES = ("scene", "text", "language")
TASK_CONDITIONING_DIMS = {
    "scene": 23,
    "text": 90,
    "language": 768,
}


def task_conditioning_dim(mode: Optional[str]) -> int:
    if mode is None:
        raise ValueError("task_conditioning_mode must be set when task conditioning is enabled")
    mode = mode.lower()
    if mode not in TASK_CONDITIONING_DIMS:
        raise ValueError(
            f"Unknown task_conditioning_mode={mode!r}. "
            f"Expected one of {TASK_CONDITIONING_MODES}"
        )
    return TASK_CONDITIONING_DIMS[mode]


def resolve_task_id(task_name: str) -> int:
    """Map robobuf task_name (HDF5 filename) to a task index."""
    if task_name in TASK_IDX:
        return TASK_IDX[task_name]
    if task_name.endswith(".hdf5"):
        alt = task_name.replace(".hdf5", "_demo.hdf5")
        if alt in TASK_IDX:
            return TASK_IDX[alt]
    raise KeyError(
        f"Unknown task_name={task_name!r}. Add it to libero_tasks.TASK_IDX or "
        "re-convert the buffer with task_name metadata."
    )


def scene_one_hot(task_name: str) -> np.ndarray:
    """23-d scene indicator (LIBERO scene prefix of the task)."""
    task_id = resolve_task_id(task_name)

    vec = np.zeros(len(SCENE_NAMES), dtype=np.float32)
    name = TASK_NAMES[task_id] if task_id < len(TASK_NAMES) else task_name
    for j, scene in enumerate(SCENE_NAMES):
        if scene in name or name.startswith(scene):
            vec[j] = 1.0
            return vec
    # LIBERO object/spatial buckets used in DSRL (indices 21/22 for 23-d vecs)
    if "LIBERO_OBJECT" in name or name.startswith("pick_up_the_"):
        if len(vec) >= 22:
            vec[21] = 1.0
        return vec
    if "LIBERO_SPATIAL" in name:
        if len(vec) >= 23:
            vec[22] = 1.0
        return vec
    raise ValueError(f"Could not infer scene for task_name={task_name!r}")


def text_one_hot(task_name: str) -> np.ndarray:
    """90-d LIBERO-90 task ID one-hot."""
    task_id = resolve_task_id(task_name)
    if task_id >= NUM_TEXT_TASKS:
        raise ValueError(
            f"task_id={task_id} exceeds NUM_TEXT_TASKS={NUM_TEXT_TASKS} for mode 'text'. "
            f"Use task_conditioning_mode='language' for LIBERO-OBJECT/SPATIAL tasks."
        )
    vec = np.zeros(NUM_TEXT_TASKS, dtype=np.float32)
    vec[task_id] = 1.0
    return vec


class LanguageEmbeddingStore:
    """Loads frozen language embeddings (e.g. BERT) keyed by instruction bytes."""

    def __init__(self, embedding_path: str):
        if not embedding_path or not os.path.exists(embedding_path):
            raise FileNotFoundError(
                f"language_embedding_path not found: {embedding_path}"
            )
        with open(embedding_path, "rb") as f:
            raw = pkl.load(f)
        self._embeddings = {k: np.asarray(v, dtype=np.float32) for k, v in raw.items()}
        dim = next(iter(self._embeddings.values())).shape[-1]
        if dim != TASK_CONDITIONING_DIMS["language"]:
            raise ValueError(
                f"Expected language dim {TASK_CONDITIONING_DIMS['language']}, got {dim}"
            )

    def lookup(self, language_key: Union[str, bytes]) -> np.ndarray:
        if isinstance(language_key, str):
            key = language_key.encode("utf-8")
        else:
            key = language_key
        if key not in self._embeddings:
            raise KeyError(
                f"Language key not found in embedding table. "
                f"Store 'lang' on robobuf obs or extend the pickle."
            )
        return self._embeddings[key]


class TaskConditioningEncoder:
    def __init__(
        self,
        mode: str,
        language_embedding_path: Optional[str] = None,
    ):
        self.mode = mode.lower()
        self.dim = task_conditioning_dim(self.mode)
        self._language_store = None
        if self.mode == "language":
            self._language_store = LanguageEmbeddingStore(language_embedding_path)

    def encode(
        self,
        task_name: Optional[str] = None,
        lang: Optional[Union[str, bytes]] = None,
    ) -> np.ndarray:
        if self.mode == "scene":
            return scene_one_hot(task_name)
        if self.mode == "text":
            return text_one_hot(task_name)
        if self.mode == "language":
            if lang is not None:
                return self._language_store.lookup(lang)
            if task_name is not None and task_name.encode("utf-8") in getattr(
                self._language_store, "_embeddings", {}
            ):
                return self._language_store.lookup(task_name)
            raise ValueError(
                "language task conditioning requires obs['lang'] (instruction string) "
                "or a language pickle keyed by task_name."
            )
        raise RuntimeError(f"Unhandled mode {self.mode}")

    def encode_transition_obs(self, obs_dict: dict) -> np.ndarray:
        task_name = obs_dict.get("task_name")
        lang = obs_dict.get("lang") or obs_dict.get("language")
        return self.encode(task_name=task_name, lang=lang)
