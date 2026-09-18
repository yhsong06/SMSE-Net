"""Canonical SMSE-Net audio preprocessing and waveform reconstruction."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import librosa
import numpy as np
import soundfile as sf


SAMPLE_RATE = 8_000
N_FFT = 512
WIN_LENGTH = 512
HOP_LENGTH = 128
FRAMES_PER_INPUT = 4
FREQUENCY_BINS = 256
DB_MIN = -100.0
DB_MAX = 60.0
AMPLITUDE_MIN = 1e-5


def load_waveform(path: str | Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    waveform, _ = librosa.load(Path(path), sr=sample_rate, mono=True)
    return np.asarray(waveform, dtype=np.float32)


def stft_magnitude_phase(waveform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spectrum = librosa.stft(
        waveform,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window="hann",
        center=True,
        pad_mode="constant",
    )
    return np.abs(spectrum).astype(np.float32), np.angle(spectrum).astype(np.float32)


def magnitude_to_db(magnitude: np.ndarray) -> np.ndarray:
    db = librosa.amplitude_to_db(
        np.maximum(magnitude, 1e-10),
        ref=1.0,
        amin=AMPLITUDE_MIN,
        top_db=None,
    )
    return np.clip(db, DB_MIN, DB_MAX).astype(np.float32)


def normalize_db(db: np.ndarray) -> np.ndarray:
    clipped = np.clip(db, DB_MIN, DB_MAX)
    return ((clipped - DB_MIN) / (DB_MAX - DB_MIN)).astype(np.float32)


def inverse_normalize_db(normalized: np.ndarray, clip: bool = True) -> np.ndarray:
    values = np.clip(normalized, 0.0, 1.0) if clip else normalized
    return (values * (DB_MAX - DB_MIN) + DB_MIN).astype(np.float32)


def _pad_frames(array: np.ndarray, multiple: int = FRAMES_PER_INPUT) -> np.ndarray:
    remainder = array.shape[1] % multiple
    if remainder == 0:
        return array
    padding = np.zeros(
        (array.shape[0], multiple - remainder),
        dtype=array.dtype,
    )
    return np.hstack([array, padding])


def _split_frames(array: np.ndarray, pad: bool) -> np.ndarray:
    if pad:
        array = _pad_frames(array)
    usable_frames = (array.shape[1] // FRAMES_PER_INPUT) * FRAMES_PER_INPUT
    array = array[:, :usable_frames]
    if usable_frames == 0:
        return np.empty((0, array.shape[0], FRAMES_PER_INPUT), dtype=np.float32)
    return np.stack(
        [
            array[:, start : start + FRAMES_PER_INPUT]
            for start in range(0, usable_frames, FRAMES_PER_INPUT)
        ],
        axis=0,
    ).astype(np.float32)


def preprocess_waveform_pair(
    noisy_am: np.ndarray,
    tm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Return normalized TM/AM segments, noisy-AM phase, and aligned sample length.

    Test/inference behavior matches the canonical evaluator: linear magnitudes and
    phases are zero-padded to a multiple of four frames before dB conversion.
    """
    aligned_length = min(len(noisy_am), len(tm))
    if aligned_length <= N_FFT - 1:
        raise ValueError(f"Waveforms must contain at least {N_FFT} aligned samples.")
    noisy_am = np.asarray(noisy_am[:aligned_length], dtype=np.float32)
    tm = np.asarray(tm[:aligned_length], dtype=np.float32)
    am_magnitude, am_phase = stft_magnitude_phase(noisy_am)
    tm_magnitude, _ = stft_magnitude_phase(tm)

    # Remove the DC bin: 257 -> 256.
    am_magnitude = _pad_frames(am_magnitude[1:, :])
    tm_magnitude = _pad_frames(tm_magnitude[1:, :])
    am_phase = _pad_frames(am_phase[1:, :])

    am_segments = _split_frames(normalize_db(magnitude_to_db(am_magnitude)), pad=False)
    tm_segments = _split_frames(normalize_db(magnitude_to_db(tm_magnitude)), pad=False)
    phase_segments = _split_frames(am_phase, pad=False)
    return tm_segments, am_segments, phase_segments, aligned_length


def reconstruct_waveform(
    predicted_segments: np.ndarray,
    noisy_phase_segments: np.ndarray,
) -> np.ndarray:
    """Assemble segments and apply the canonical noisy-phase iSTFT path.

    The historical evaluator did not pass ``length`` to ``librosa.istft``. This
    function deliberately preserves that behavior instead of inventing trimming.
    """
    predicted_segments = np.asarray(predicted_segments, dtype=np.float32)
    noisy_phase_segments = np.asarray(noisy_phase_segments, dtype=np.float32)
    if predicted_segments.ndim == 4 and predicted_segments.shape[1] == 1:
        predicted_segments = predicted_segments[:, 0]
    if predicted_segments.shape != noisy_phase_segments.shape:
        raise ValueError(
            "Predicted magnitude and noisy phase segment shapes differ: "
            f"{predicted_segments.shape} vs {noisy_phase_segments.shape}"
        )
    predicted_db = inverse_normalize_db(predicted_segments, clip=True)
    combined_db = np.concatenate(list(predicted_db), axis=1)
    combined_phase = np.concatenate(list(noisy_phase_segments), axis=1)
    magnitude = librosa.db_to_amplitude(combined_db, ref=1.0)
    magnitude = np.vstack([np.zeros((1, magnitude.shape[1])), magnitude])
    phase = np.vstack([np.zeros((1, combined_phase.shape[1])), combined_phase])
    spectrum = magnitude * np.exp(1j * phase)
    waveform = librosa.istft(
        spectrum,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        win_length=WIN_LENGTH,
        window="hann",
        center=True,
    )
    return np.asarray(waveform, dtype=np.float32)


def save_enhanced_waveform(path: str | Path, waveform: np.ndarray) -> None:
    """Save with the canonical evaluator's 0.5 peak normalization."""
    waveform = np.asarray(waveform, dtype=np.float32)
    scale = 0.5 / max(float(np.max(np.abs(waveform))), 1e-8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sf.write(Path(path), waveform * scale, SAMPLE_RATE)


def _paired_paths(dataset_root: Path, speakers: Iterable[str]):
    for speaker in speakers:
        speaker_dir = dataset_root / speaker
        if not speaker_dir.is_dir():
            raise FileNotFoundError(f"Missing speaker directory: {speaker_dir}")
        for tm_path in sorted(speaker_dir.glob("*_acc.wav")):
            utterance = tm_path.name[: -len("_acc.wav")]
            clean_path = speaker_dir / f"{utterance}_mic.wav"
            noisy_path = speaker_dir / f"{utterance}_noisy_mic.wav"
            if not clean_path.is_file() or not noisy_path.is_file():
                raise FileNotFoundError(f"Incomplete waveform triplet for {utterance}")
            yield utterance, tm_path, clean_path, noisy_path


def prepare_segmented_split(
    dataset_root: str | Path,
    output_dir: str | Path,
    speakers: Iterable[str],
    split: str,
) -> None:
    """Create canonical segmented arrays from pre-generated waveform triplets.

    This function does not mix DNS noise. ``*_noisy_mic.wav`` must already exist.
    Train/validation discard incomplete trailing frames; test pads them.
    """
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    magnitudes: dict[str, list[np.ndarray]] = {"tm": [], "clean": [], "noisy": []}
    phases: dict[str, list[np.ndarray]] = {"tm": [], "clean": [], "noisy": []}
    names: list[str] = []
    pad = split == "test"

    for utterance, tm_path, clean_path, noisy_path in _paired_paths(Path(dataset_root), speakers):
        tm = load_waveform(tm_path)
        clean = load_waveform(clean_path)
        noisy = load_waveform(noisy_path)
        length = min(len(tm), len(clean), len(noisy))
        signals = {"tm": tm[:length], "clean": clean[:length], "noisy": noisy[:length]}
        utterance_segments: dict[str, np.ndarray] = {}
        utterance_phases: dict[str, np.ndarray] = {}
        for key, signal in signals.items():
            magnitude, phase = stft_magnitude_phase(signal)
            magnitude, phase = magnitude[1:, :], phase[1:, :]
            if pad:
                magnitude, phase = _pad_frames(magnitude), _pad_frames(phase)
            utterance_segments[key] = _split_frames(magnitude_to_db(magnitude), pad=False)
            utterance_phases[key] = _split_frames(phase, pad=False)
        count = min(value.shape[0] for value in utterance_segments.values())
        for key in magnitudes:
            magnitudes[key].extend(utterance_segments[key][:count])
            if split == "test":
                phases[key].extend(utterance_phases[key][:count])
        names.extend([utterance] * count)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    suffix = "" if split == "train" else "_val" if split == "validation" else "_test"
    np.save(output / f"acc_seg_data_spectrogram_v19{suffix}.npy", np.asarray(magnitudes["tm"], dtype=np.float32))
    np.save(output / f"mic_seg_data_spectrogram_v19{suffix}.npy", np.asarray(magnitudes["clean"], dtype=np.float32))
    np.save(output / f"Noisy_seg_data_spectrogram_v19{suffix}.npy", np.asarray(magnitudes["noisy"], dtype=np.float32))
    np.save(output / f"sound_file_name_v19{suffix}.npy", np.asarray(names))
    if split == "test":
        np.save(output / "acc_seg_data_phase_v19_test.npy", np.asarray(phases["tm"], dtype=np.float32))
        np.save(output / "mic_seg_data_phase_v19_test.npy", np.asarray(phases["clean"], dtype=np.float32))
        np.save(output / "Noisy_seg_data_phase_v19_test.npy", np.asarray(phases["noisy"], dtype=np.float32))


def _main() -> None:
    parser = argparse.ArgumentParser(description="Prepare canonical SMSE-Net segmented features.")
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--split", required=True, choices=["train", "validation", "test"])
    parser.add_argument("--split-file", default=Path("splits/taps_speaker_split.json"), type=Path)
    args = parser.parse_args()
    split_data = json.loads(args.split_file.read_text(encoding="utf-8"))
    prepare_segmented_split(args.dataset_root, args.output_dir, split_data[args.split], args.split)


if __name__ == "__main__":
    _main()
