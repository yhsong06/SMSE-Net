# Reproducibility record

## Canonical identity

| Item | Verified value |
|---|---|
| Internal model name | `v21_accpower_v8_sa_h24_a8` |
| Public class | `smse_net.model.SMSENet` |
| Checkpoint | `checkpoints/best_model.pth` |
| Checkpoint SHA-256 | `4578f3862e0273ede3ad36153431a6afff571a727893f96c4bfd41ef1b89de17` |
| Strict state-dict load | supported with `strict=True` |
| Trainable parameters | 41,950 |
| Best epoch | 93 |
| Validation MSE | `0.0022235455162370266` |
| PESQ-NB | `2.858056921958923` |
| STOI | `0.8959101306862987` |

The public class was extracted from the active full-model path only. It preserves every checkpoint key, including four compatibility scalars that are present in the checkpoint but inactive in the canonical forward computation.

## Preprocessing provenance

The canonical segmented arrays contain positive dB maxima and ranges exceeding 80 dB; this rules out the obsolete `ref=np.max`/default-`top_db` representation. Their creation date, numeric ranges, active fixed-reference preprocessing source, checkpoint metadata, and training logs consistently identify:

```text
librosa.amplitude_to_db(ref=1.0, top_db=None)
clip to [-100, 60] dB
normalize with (D + 100) / 160
```

An older project-root preprocessing script still used `ref=np.max`; it is intentionally absent from this repository.

## Metric aggregation

The historical evaluator reconstructed each utterance by concatenating its predicted four-frame blocks and noisy-AM phase blocks. It clipped model output to `[0,1]`, inverted normalization, restored a zero DC row, and applied iSTFT without a `length` argument. Clean reference and enhanced signals were cropped to their common length. PESQ-NB used 8 kHz `nb` mode; STOI used `extended=False` at 8 kHz. Reported totals are arithmetic means over 1,000 successful utterances, not equal-weighted means over SNR levels.

## Known limitations

1. The exact historical DNS mixing implementation and its training/validation noise-to-utterance manifest are absent. Public preparation begins with pre-generated `*_noisy_mic.wav` files.
2. Raw TAPS/DNS data, participant data, and SNR logs are not distributed. Users must secure data access and confirm its licensing/consent constraints.
3. The original environment manifest is incomplete. `requirements.txt` constrains the observed core software versions, but CUDA/cuDNN and platform details are unavailable.
4. Test preprocessing padded frames but did not preserve original sample length. The public waveform path deliberately retains the historical no-`length` iSTFT behavior.
5. Exact reported metrics require the canonical test waveforms and speaker split. New noise mixtures are compatible experiments, not exact historical reproduction.
6. The placeholder `LICENSE` does not grant redistribution rights. A real code license and separate checkpoint/data-rights review are required before publication.
7. Complete citation metadata was not available in the source archive and must be added before release.

## Excluded work

No structured-pruning implementation, channel mask, pruning sweep, compact-model conversion, pruned checkpoint, MCU firmware/export, baseline, ablation, plotting utility, or MAC-count implementation is included. These exclusions are intentional and do not affect the full-model software path documented here.
