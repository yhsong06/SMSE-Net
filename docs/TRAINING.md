# Canonical training procedure

## Configuration

| Item | Canonical value |
|---|---:|
| Model | `v21_accpower_v8_sa_h24_a8` |
| Optimizer | AdamW |
| Base learning rate | `0.0004` |
| Weight decay | `0.01` |
| Batch size | 64 |
| Epochs | 100 |
| Seed | 42 |
| Workers | 0 |
| Loss | clean-AM normalized log-magnitude MSE |
| Checkpoint criterion | minimum validation MSE |
| Gradient clipping | none |

Input batches are `[B,2,256,4]` in `[TM, noisy AM]` order. Targets are `[B,1,256,4]` normalized clean-AM log magnitudes. The full model is optimized end to end; there is no auxiliary or pruning loss.

## Learning-rate schedule

For the recovered canonical data, `ceil(570319 / 64) = 8912` optimizer steps occur per epoch and 891,200 over 100 epochs. The first `int(891200 * 0.05) = 44,560` steps use linear warmup:

```text
lr(step) = 0.0004 * (step + 1) / 44560
```

The remaining steps follow cosine decay from `0.0004` toward `0.00004`. The public implementation computes this per optimizer step in `learning_rate_at_step`.

## Reproducibility controls

Seed 42 is applied to Python, NumPy, PyTorch CPU, and every CUDA device. cuDNN deterministic mode is enabled and benchmarking is disabled. This reduces nondeterminism but does not guarantee bit-identical results across different PyTorch, CUDA, cuDNN, GPU, or PESQ builds.

Training shuffles only the training loader. Validation is not shuffled. The historical validation set has 140,480 segments and is exactly divisible by 64, so its arithmetic mean over batch MSE equals a segment-weighted mean.

## Checkpointing

After every epoch, `train.py` evaluates the validation split. It writes `best_model.pth` only when the validation MSE is strictly lower than the previous best. The test split is never constructed by `train.py` and cannot influence selection.

The canonical checkpoint reached its minimum validation MSE at epoch 93:

```text
best epoch = 93
validation MSE = 0.0022235455162370266
```

## Command

```bash
python train.py \
  --config configs/smse_net.yaml \
  --data-dir data/segmented \
  --output-dir runs/smse_net
```

The command expects dB-domain arrays produced by `python -m smse_net.audio`; it must not be pointed at older `ref=np.max` arrays.
