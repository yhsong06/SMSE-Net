# Data preparation

## What this repository supports

The provided feature-preparation code converts already aligned waveform triplets into the exact segmented arrays consumed by training and evaluation. Each utterance must have:

- `*_acc.wav`: throat microphone (TM)
- `*_mic.wav`: clean acoustic microphone (AM) target/reference
- `*_noisy_mic.wav`: noisy acoustic microphone input

The three signals are loaded as mono at 8 kHz and cropped to their common minimum length. Pairing is validated by full utterance identifier. Raw recordings and participant data are not distributed.

## What is not recoverable from the source archive

The manuscript describes DNS noise use, but the audited project does not contain the exact historical procedure that selected DNS files, sampled training/validation SNR, scaled noise, avoided clipping, or mapped noise to utterances. The recoverable preprocessing script starts from an already-created `*_noisy_mic.wav`. This public code therefore does not generate noisy AM waveforms and does not claim to reproduce that missing mixing stage.

To reproduce the reported scores exactly, researchers need the original aligned `*_acc.wav`, `*_mic.wav`, and `*_noisy_mic.wav` files (and the corresponding SNR logs for per-SNR summaries).

## Speaker split

The canonical sidecar arrays establish a fixed 40/10/10 split, stored in `splits/taps_speaker_split.json`:

- train: 40 speakers, 4,000 utterances, 570,319 four-frame segments
- validation: 10 speakers, 1,000 utterances, 140,480 segments
- test: 10 speakers, 1,000 utterances, 145,818 segments

Each speaker contributes 100 utterances. Training never reads validation/test features for optimization, and checkpoint selection uses validation only.

## Exact feature representation

For each waveform:

```python
D = librosa.stft(
    waveform,
    n_fft=512,
    hop_length=128,
    win_length=512,
    window="hann",
    center=True,
    pad_mode="constant",
)
magnitude = np.abs(D)[1:, :]  # discard DC, 257 -> 256
phase = np.angle(D)[1:, :]
log_magnitude = librosa.amplitude_to_db(
    np.maximum(magnitude, 1e-10),
    ref=1.0,
    amin=1e-5,
    top_db=None,
)
log_magnitude = np.clip(log_magnitude, -100.0, 60.0)
```

`amin=1e-5` makes the version-dependent librosa default explicit. Training and validation discard trailing frames that do not form a complete four-frame block. Test and standalone inference zero-pad linear magnitude and phase to a multiple of four frames before dB conversion, matching the historical evaluator.

The stored arrays are dB values. `TrainingSegmentDataset` applies:

```text
Y = (clip(D, -100, 60) + 100) / 160
```

Array filenames are intentionally retained for compatibility with the canonical run:

```text
acc_seg_data_spectrogram_v19.npy
mic_seg_data_spectrogram_v19.npy
Noisy_seg_data_spectrogram_v19.npy
acc_seg_data_spectrogram_v19_val.npy
mic_seg_data_spectrogram_v19_val.npy
Noisy_seg_data_spectrogram_v19_val.npy
sound_file_name_v19_test.npy
acc_seg_data_spectrogram_v19_test.npy
mic_seg_data_spectrogram_v19_test.npy
Noisy_seg_data_spectrogram_v19_test.npy
Noisy_seg_data_phase_v19_test.npy
```

The `v19` filename token is a historical feature-artifact name, not a model version.

## SNR records

The paper reports test SNRs `-20, -15, -10, -5, 0, 5, 10, 15, 20` dB. The evaluator optionally recovers each utterance's SNR from `pXX/noise_snr_usage_log.txt` encoded as CP949. The canonical distribution was 120 utterances at -20 dB and 110 at each other level, totaling 1,000. Overall PESQ-NB/STOI aggregation does not depend on the log.
