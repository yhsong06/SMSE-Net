"""Dataset loaders for pre-generated canonical SMSE-Net spectrogram segments."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .audio import normalize_db


class TrainingSegmentDataset(Dataset):
    """TM/noisy-AM inputs and clean-AM targets from canonical ``.npy`` files."""

    def __init__(self, data_dir: str | Path, split: str, mmap: bool = True) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        suffix = "" if split == "train" else "_val"
        root = Path(data_dir)
        mode = "r" if mmap else None
        self.tm = np.load(root / f"acc_seg_data_spectrogram_v19{suffix}.npy", mmap_mode=mode)
        self.clean = np.load(root / f"mic_seg_data_spectrogram_v19{suffix}.npy", mmap_mode=mode)
        self.noisy = np.load(root / f"Noisy_seg_data_spectrogram_v19{suffix}.npy", mmap_mode=mode)
        if not (self.tm.shape == self.clean.shape == self.noisy.shape):
            raise ValueError("TM, clean AM, and noisy AM arrays must have identical shapes.")
        if self.tm.shape[1:] != (256, 4):
            raise ValueError(f"Expected [N, 256, 4] arrays, received {self.tm.shape}")

    def __len__(self) -> int:
        return int(self.tm.shape[0])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        tm = normalize_db(np.asarray(self.tm[index], dtype=np.float32))
        noisy = normalize_db(np.asarray(self.noisy[index], dtype=np.float32))
        clean = normalize_db(np.asarray(self.clean[index], dtype=np.float32))
        inputs = np.stack([tm, noisy], axis=0)
        return {
            "input": torch.from_numpy(inputs),
            "clean": torch.from_numpy(clean[None, ...]),
        }


class TestSegmentDataset(Dataset):
    """Canonical test segments with utterance names and noisy-AM phases."""

    def __init__(self, data_dir: str | Path, mmap: bool = True) -> None:
        root = Path(data_dir)
        mode = "r" if mmap else None
        self.names = np.load(root / "sound_file_name_v19_test.npy")
        self.tm = np.load(root / "acc_seg_data_spectrogram_v19_test.npy", mmap_mode=mode)
        self.clean = np.load(root / "mic_seg_data_spectrogram_v19_test.npy", mmap_mode=mode)
        self.noisy = np.load(root / "Noisy_seg_data_spectrogram_v19_test.npy", mmap_mode=mode)
        self.noisy_phase = np.load(root / "Noisy_seg_data_phase_v19_test.npy", mmap_mode=mode)
        count = len(self.names)
        if not all(len(array) == count for array in [self.tm, self.clean, self.noisy, self.noisy_phase]):
            raise ValueError("Test arrays and sound names must contain the same number of segments.")

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> dict[str, object]:
        tm = normalize_db(np.asarray(self.tm[index], dtype=np.float32))
        noisy = normalize_db(np.asarray(self.noisy[index], dtype=np.float32))
        clean = normalize_db(np.asarray(self.clean[index], dtype=np.float32))
        return {
            "input": torch.from_numpy(np.stack([tm, noisy], axis=0)),
            "clean": torch.from_numpy(clean[None, ...]),
            "noisy_phase": torch.from_numpy(np.asarray(self.noisy_phase[index], dtype=np.float32).copy()),
            "sound_name": str(self.names[index]),
        }
