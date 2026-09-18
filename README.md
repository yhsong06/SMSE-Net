# SMSE-Net

Reference code for **“Real-time wearable speech enhancement via noise-aware fusion of acoustic and throat microphones in extreme noise.”**

This repository includes the full SMSE-Net model implementation, preprocessing and postprocessing, training, evaluation, waveform-level inference, and the canonical pretrained checkpoint. It contains the canonical full, unpruned SNR-aware Multimodal Speech Enhancement Network (SMSE-Net), and training uses pre-generated acoustic/throat microphone triplets.

## Architecture

SMSE-Net consumes four 8 kHz STFT frames from a throat microphone (TM) and a noisy acoustic microphone (AM). Separate encoders use channels `(6, 12, 24)` and `(12, 24, 48)`, and the TM bottleneck is aligned from 24 to 48 channels. Four single-head, dimension-8 attention streams—AM self-attention, TM self-attention, TM-guided AM cross-attention, and AM-guided TM cross-attention—are reliability-weighted and fused from 192 to 48 channels. Frequency and temporal GRUs both have hidden size 24. See [docs/MODEL.md](docs/MODEL.md).

The canonical model has exactly **41,950 trainable parameters**.

## Installation

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data layout

Raw TAPS audio and DNS noise are not distributed. Feature preparation expects each selected speaker directory to contain pre-generated, aligned triplets:

```text
path/to/taps/
  p01/
    p01_u00_acc.wav
    p01_u00_mic.wav
    p01_u00_noisy_mic.wav
    ...
```

`*_acc.wav` is TM, `*_mic.wav` is clean AM, and `*_noisy_mic.wav` is noisy AM. The exact historical DNS mixing script was not recovered, and training expects pre-generated noisy AM recordings. Prepare each split with:

```bash
python -m smse_net.audio --dataset-root path/to/taps --output-dir data/segmented --split train
python -m smse_net.audio --dataset-root path/to/taps --output-dir data/segmented --split validation
python -m smse_net.audio --dataset-root path/to/taps --output-dir data/segmented --split test
```

The recovered 40/10/10 speaker assignment is in [splits/taps_speaker_split.json](splits/taps_speaker_split.json). See [docs/DATA_PREPARATION.md](docs/DATA_PREPARATION.md) before preparing data.

## Canonical preprocessing

- 8 kHz mono audio; Hann STFT with `n_fft=512`, `win_length=512`, and `hop_length=128`
- remove DC: 257 bins become 256
- four non-overlapping frames per network input
- `librosa.amplitude_to_db(ref=1.0, top_db=None)`, explicitly bounded to `[-100, 60]` dB
- normalize as `(D + 100) / 160`
- invert as `160Y - 100` after clipping predictions to `[0, 1]`
- restore a zero DC row, combine predicted magnitude with noisy-AM phase, then apply Hann iSTFT

The four-frame context spans 112 ms; adjacent network inputs advance by 64 ms.

## Training

```bash
python train.py --data-dir data/segmented --output-dir runs/smse_net
```

The canonical run uses AdamW, learning rate `0.0004`, weight decay `0.01`, batch size 64, 100 epochs, seed 42, 5% linear warmup, cosine decay to `0.00004`, and normalized clean-AM log-magnitude MSE. Only validation MSE selects `best_model.pth`; the test set is never used for selection. See [docs/TRAINING.md](docs/TRAINING.md).

## Evaluation

```bash
python evaluate.py \
  --data-dir data/segmented \
  --waveform-root path/to/taps \
  --checkpoint checkpoints/best_model.pth \
  --output evaluation_results.json
```

Evaluation reports utterance-level PESQ-NB at 8 kHz and STOI with `extended=False`, followed by their arithmetic means. The paper's test set covers SNRs from -20 to 20 dB in 5-dB increments. Per-SNR summaries require the historical `noise_snr_usage_log.txt` files; overall metrics do not.

## Waveform inference

```bash
python inference.py \
  --am path/to/noisy_am.wav \
  --tm path/to/tm.wav \
  --checkpoint checkpoints/best_model.pth \
  --output enhanced.wav
```

Both input files are loaded as mono 8 kHz audio and aligned to their shorter length. The historical reconstruction does not pass the original waveform length to `librosa.istft`; the enhanced file therefore follows the canonical STFT-frame assembly length.

## Pretrained checkpoint and expected result

`checkpoints/best_model.pth` is the unmodified canonical checkpoint for `v21_accpower_v8_sa_h24_a8`:

- SHA-256: `4578f3862e0273ede3ad36153431a6afff571a727893f96c4bfd41ef1b89de17`
- best epoch: 93
- validation MSE: `0.0022235455162370266`
- parameters: 41,950
- PESQ-NB: `2.858056921958923`
- STOI: `0.8959101306862987`

Exact metric reproduction requires the original noisy waveforms, clean references, and dependency behavior. See [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Repository structure

```text
configs/smse_net.yaml          canonical configuration
smse_net/model.py              full SMSE-Net only
smse_net/audio.py              STFT, feature preparation, reconstruction
smse_net/dataset.py            segmented-array datasets
smse_net/metrics.py            PESQ-NB and STOI
train.py                       canonical training
evaluate.py                    canonical full-model evaluation
inference.py                   paired-waveform inference
splits/taps_speaker_split.json recovered speaker split
checkpoints/best_model.pth     canonical full checkpoint
docs/                          technical and reproducibility notes
```

## Scope and exclusions

This release intentionally excludes structured pruning implementations, compact-model conversion, pruned checkpoints, MCU firmware/export, baseline model implementations, ablation implementations, obsolete LAU-Net variants, raw TAPS or DNS datasets, participant data, plots, and experiment logs. MAC-count code is also excluded because it is not needed by the public training/evaluation/inference functions.

## Citation

If you find this work useful, please cite:

**Real-time wearable speech enhancement via noise-aware fusion of acoustic
and throat microphones in extreme noise**

Yonghun Song, Yeeun Kim, Yunsik Kim, and Yoonyoung Chung.

Citation information will be updated upon publication.

## License

The source code in this repository is released under the [MIT License](LICENSE).
