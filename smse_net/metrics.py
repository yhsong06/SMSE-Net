"""Paper metrics used by the public full-model evaluation."""

from __future__ import annotations

import numpy as np


def align_waveforms(reference: np.ndarray, estimate: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    length = min(len(reference), len(estimate))
    if length == 0:
        raise ValueError("Cannot evaluate empty waveforms.")
    return (
        np.asarray(reference[:length], dtype=np.float32),
        np.asarray(estimate[:length], dtype=np.float32),
    )


def pesq_nb(reference: np.ndarray, estimate: np.ndarray, sample_rate: int = 8_000) -> float:
    from pesq import pesq

    if sample_rate != 8_000:
        raise ValueError("PESQ-NB must be evaluated at 8000 Hz.")
    reference, estimate = align_waveforms(reference, estimate)
    return float(pesq(sample_rate, reference, estimate, "nb"))


def stoi_score(reference: np.ndarray, estimate: np.ndarray, sample_rate: int = 8_000) -> float:
    from pystoi import stoi

    reference, estimate = align_waveforms(reference, estimate)
    return float(stoi(reference, estimate, sample_rate, extended=False))
