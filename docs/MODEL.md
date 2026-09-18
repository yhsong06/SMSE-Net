# SMSE-Net architecture

## Input and output

The network input is `[B, 2, 256, 4]`, ordered as `[TM, noisy AM]`. Both modalities are normalized absolute-reference log-magnitude spectrograms. The output is `[B, 1, 256, 4]`, interpreted as normalized clean-AM log magnitude. The network does not predict phase.

## Encoders

The noisy-AM encoder uses channel widths 12, 24, and 48. The TM encoder uses 6, 12, and 24. Each stage downsamples only frequency by two. The first stage uses standard convolutions; later stages use depthwise-separable frequency convolutions. A `1x1` two-dimensional convolution aligns the final 24-channel TM representation to 48 channels. The bottleneck grid is `32 x 4`.

## Attention streams

All four streams use single-head scaled dot-product attention with projection dimension `d=8`:

1. AM self-attention: `Q=K=V=AM`
2. TM self-attention: `Q=K=V=TM`
3. TM-guided AM cross-attention: `Q=AM`, `K=V=TM`
4. AM-guided TM cross-attention: `Q=TM`, `K=V=AM`

Each `32 x 4` feature map is flattened into 128 tokens. Queries, keys, and values are projected with `1x1` convolutions. Attention uses `softmax(QK^T / sqrt(8))`; the projected result is added to its query through a residual connection and batch-normalized.

## SNR-aware reliability and gains

The canonical model estimates a global reliability score from the normalized input features. Let `x_TM` and `x_AM` be the normalized inputs. The code first constructs a learned-energy proxy:

```text
P(x) = exp(4 * clip(x, 0, 1))
```

It averages TM proxy power over the first 32 frequency bins and all four frames, and AM proxy power over all 256 bins and four frames. With learned positive slope `a`, bias `b`, center `c`, positive scale `s`, and affine sigmoid parameters `alpha`, `beta`:

```text
L_TM = log(mean(P(x_TM)[0:32, :]))
L_AM = log(mean(P(x_AM)))
L_pred = a * L_TM + b
z = (L_AM - L_pred - c) / s
rho = sigmoid(alpha * z + beta)
```

The scalar `rho` is repeated over four fixed bands and interpolated to the `32 x 4` bottleneck. It is sharpened and centered:

```text
rho_s = sigmoid(tau * (rho - 0.5))
r = 2 * rho_s - 1
```

The four gains are:

```text
g_AM       = exp(-s_AM * r)
g_TM       = exp(+s_TM * r)
g_TM_to_AM = exp(+s_TM_to_AM * r)
g_AM_to_TM = exp(+s_AM_to_TM * r)
```

Each gain is divided by the mean of all four gains. They modulate, respectively, AM self-attention, TM self-attention, TM-guided AM, and AM-guided TM. This is a learned reliability proxy over normalized features; it is not a direct physical SNR estimator.

The checkpoint retains four additional scalar parameters that are not consumed by the selected canonical forward path. They remain in `SMSENet` solely to preserve strict checkpoint compatibility and the verified parameter count.

## Fusion, recurrent blocks, and decoder

The four 48-channel streams are concatenated to 192 channels and projected to 48 by a `1x1` convolution, batch normalization, and ReLU. A frequency-axis GRU and then a temporal-axis GRU each use hidden size 24 and project back to 48 channels. Three decoder stages upsample frequency and use skip connections from the noisy-AM encoder. The final convolution emits one channel without an output activation. Clipping to `[0,1]` occurs only during waveform reconstruction/evaluation.

The full model contains exactly 41,950 trainable parameters. There are no pruning, baseline, or ablation branches in this implementation.
