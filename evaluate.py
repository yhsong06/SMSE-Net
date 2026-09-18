#!/usr/bin/env python3
"""Evaluate the canonical full SMSE-Net with PESQ-NB and STOI."""

from __future__ import annotations

import argparse
import json
import re
from collections import OrderedDict
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import DataLoader

from smse_net.audio import SAMPLE_RATE, reconstruct_waveform
from smse_net.dataset import TestSegmentDataset
from smse_net.metrics import pesq_nb, stoi_score
from smse_net.model import build_model, load_checkpoint


SNR_LEVELS = [-20, -15, -10, -5, 0, 5, 10, 15, 20]


def clean_waveform_path(waveform_root: Path, sound_name: str) -> Path:
    speaker = sound_name.split("_", maxsplit=1)[0]
    return waveform_root / speaker / f"{sound_name}_mic.wav"


def read_snr_map(waveform_root: Path) -> dict[str, int]:
    """Read the historical per-speaker CP949 SNR logs when they are available."""
    mapping: dict[str, int] = {}
    name_pattern = re.compile(r"(p\d+_u\d+)")
    snr_pattern = re.compile(r"SNR[^-+\d]*([-+]?\d+(?:\.\d+)?)", re.IGNORECASE)
    for log_path in sorted(waveform_root.glob("p*/noise_snr_usage_log.txt")):
        for line in log_path.read_text(encoding="cp949", errors="replace").splitlines():
            name_match = name_pattern.search(line)
            snr_match = snr_pattern.search(line)
            if name_match and snr_match:
                mapping[name_match.group(1)] = int(round(float(snr_match.group(1))))
    return mapping


@torch.no_grad()
def collect_predictions(
    data_dir: Path,
    checkpoint: Path,
    device: torch.device,
    batch_size: int,
) -> OrderedDict[str, dict[str, list[np.ndarray]]]:
    dataset = TestSegmentDataset(data_dir)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model = build_model().to(device)
    load_checkpoint(model, checkpoint, device)
    model.eval()
    utterances: OrderedDict[str, dict[str, list[np.ndarray]]] = OrderedDict()
    for batch in loader:
        predictions = model(batch["input"].to(device)).cpu().numpy()
        phases = batch["noisy_phase"].numpy()
        for index, sound_name in enumerate(batch["sound_name"]):
            entry = utterances.setdefault(sound_name, {"prediction": [], "phase": []})
            entry["prediction"].append(predictions[index])
            entry["phase"].append(phases[index])
    return utterances


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--waveform-root", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/best_model.pth"))
    parser.add_argument("--output", type=Path, default=Path("evaluation_results.json"))
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")

    utterances = collect_predictions(
        args.data_dir, args.checkpoint, torch.device(args.device), args.batch_size
    )
    snr_map = read_snr_map(args.waveform_root)
    rows: list[dict[str, object]] = []
    for index, (sound_name, segments) in enumerate(utterances.items(), start=1):
        enhanced = reconstruct_waveform(
            np.asarray(segments["prediction"], dtype=np.float32),
            np.asarray(segments["phase"], dtype=np.float32),
        )
        reference_path = clean_waveform_path(args.waveform_root, sound_name)
        reference_16k, _ = librosa.load(reference_path, sr=16_000, mono=True)
        reference = librosa.resample(reference_16k, orig_sr=16_000, target_sr=SAMPLE_RATE)
        row = {
            "sound_name": sound_name,
            "snr_db": snr_map.get(sound_name),
            "pesq_nb": pesq_nb(reference, enhanced, SAMPLE_RATE),
            "stoi": stoi_score(reference, enhanced, SAMPLE_RATE),
        }
        rows.append(row)
        print(
            f"[{index}/{len(utterances)}] {sound_name}: "
            f"PESQ-NB={row['pesq_nb']:.6f}, STOI={row['stoi']:.6f}"
        )

    if not rows:
        raise RuntimeError("No test utterances were reconstructed.")
    total = {
        "count": len(rows),
        "pesq_nb": float(np.mean([float(row["pesq_nb"]) for row in rows])),
        "stoi": float(np.mean([float(row["stoi"]) for row in rows])),
    }
    by_snr: dict[str, dict[str, float | int]] = {}
    for snr in SNR_LEVELS:
        selected = [row for row in rows if row["snr_db"] == snr]
        if selected:
            by_snr[str(snr)] = {
                "count": len(selected),
                "pesq_nb": float(np.mean([float(row["pesq_nb"]) for row in selected])),
                "stoi": float(np.mean([float(row["stoi"]) for row in selected])),
            }
    result = {
        "metric_configuration": {
            "sample_rate": SAMPLE_RATE,
            "pesq_mode": "nb",
            "stoi_extended": False,
            "reported_test_snr_levels_db": SNR_LEVELS,
        },
        "total": total,
        "by_snr": by_snr,
        "utterances": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(total, indent=2))
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    main()
