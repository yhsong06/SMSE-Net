#!/usr/bin/env python3
"""Run waveform-level inference with the canonical full SMSE-Net."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from smse_net.audio import (
    load_waveform,
    preprocess_waveform_pair,
    reconstruct_waveform,
    save_enhanced_waveform,
)
from smse_net.model import build_model, load_checkpoint


@torch.no_grad()
def enhance(
    am_path: Path,
    tm_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, int, int]:
    noisy_am = load_waveform(am_path)
    tm = load_waveform(tm_path)
    tm_segments, am_segments, phase_segments, aligned_length = preprocess_waveform_pair(
        noisy_am, tm
    )
    if len(tm_segments) == 0:
        raise ValueError("No complete four-frame input could be produced.")
    model = build_model().to(device)
    load_checkpoint(model, checkpoint_path, device)
    model.eval()
    outputs: list[np.ndarray] = []
    for start in range(0, len(tm_segments), batch_size):
        stop = start + batch_size
        inputs = np.stack([tm_segments[start:stop], am_segments[start:stop]], axis=1)
        prediction = model(torch.from_numpy(inputs).to(device))
        outputs.append(prediction.cpu().numpy())
    predicted_segments = np.concatenate(outputs, axis=0)
    enhanced = reconstruct_waveform(predicted_segments, phase_segments)
    return enhanced, aligned_length, len(tm_segments)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--am", type=Path, required=True, help="Noisy acoustic-microphone WAV")
    parser.add_argument("--tm", type=Path, required=True, help="Throat-microphone WAV")
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_model.pth"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    enhanced, aligned_length, segment_count = enhance(
        args.am, args.tm, args.checkpoint, torch.device(args.device), args.batch_size
    )
    save_enhanced_waveform(args.output, enhanced)
    print(f"Aligned input samples: {aligned_length}")
    print(f"Four-frame segments: {segment_count}")
    print(f"Enhanced samples: {len(enhanced)}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
