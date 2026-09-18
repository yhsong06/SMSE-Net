"""Canonical full, unpruned SMSE-Net.

Input channel order is [TM, noisy AM] and the expected shape is [B, 2, 256, 4].
The implementation intentionally contains no alternate models or pruning paths.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F


CANONICAL_MODEL_NAME = "v21_accpower_v8_sa_h24_a8"
CANONICAL_PARAMETER_COUNT = 41_950


def _inverse_softplus(value: float) -> float:
    return math.log(math.exp(float(value)) - 1.0)


def _pad_or_crop_1d(x: torch.Tensor, target_frequency: int) -> torch.Tensor:
    current = x.shape[-1]
    if current == target_frequency:
        return x
    if current < target_frequency:
        difference = target_frequency - current
        return F.pad(x, [difference // 2, difference - difference // 2])
    difference = current - target_frequency
    left = difference // 2
    return x[..., left : left + target_frequency]


class _StandardConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, frequency, frames = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(batch * frames, channels, frequency)
        x = self.block(x)
        out_channels, out_frequency = x.shape[1], x.shape[2]
        return x.view(batch, frames, out_channels, out_frequency).permute(0, 2, 3, 1).contiguous()


class _DepthwiseSeparableConv1d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(
                out_channels,
                out_channels,
                kernel_size=5,
                stride=2,
                padding=2,
                groups=out_channels,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, frequency, frames = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(batch * frames, channels, frequency)
        x = self.block(x)
        out_channels, out_frequency = x.shape[1], x.shape[2]
        return x.view(batch, frames, out_channels, out_frequency).permute(0, 2, 3, 1).contiguous()


class _DecoderBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(out_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        batch, channels, frequency, frames = x.shape
        skip_channels, skip_frequency = skip.shape[1], skip.shape[2]
        x = x.permute(0, 3, 1, 2).contiguous().view(batch * frames, channels, frequency)
        skip = skip.permute(0, 3, 1, 2).contiguous().view(
            batch * frames, skip_channels, skip_frequency
        )
        target_frequency = max(x.shape[-1], skip.shape[-1])
        x = _pad_or_crop_1d(x, target_frequency)
        skip = _pad_or_crop_1d(skip, target_frequency)
        x = self.block(torch.cat([x, skip], dim=1))
        out_channels, out_frequency = x.shape[1], x.shape[2]
        return x.view(batch, frames, out_channels, out_frequency).permute(0, 2, 3, 1).contiguous()


class _OutputDecoderBlock(nn.Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, kernel_size=1),
            nn.BatchNorm1d(in_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(in_channels, 1, kernel_size=5, padding=2),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        batch, channels, frequency, frames = x.shape
        skip_channels, skip_frequency = skip.shape[1], skip.shape[2]
        x = x.permute(0, 3, 1, 2).contiguous().view(batch * frames, channels, frequency)
        skip = skip.permute(0, 3, 1, 2).contiguous().view(
            batch * frames, skip_channels, skip_frequency
        )
        target_frequency = max(x.shape[-1], skip.shape[-1])
        x = _pad_or_crop_1d(x, target_frequency)
        skip = _pad_or_crop_1d(skip, target_frequency)
        x = self.block(torch.cat([x, skip], dim=1))
        out_channels, out_frequency = x.shape[1], x.shape[2]
        return x.view(batch, frames, out_channels, out_frequency).permute(0, 2, 3, 1).contiguous()


class _FrequencyGRU(nn.Module):
    def __init__(self, channels: int, hidden_size: int) -> None:
        super().__init__()
        self.gru = nn.GRU(channels, hidden_size, num_layers=1, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(hidden_size, channels), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, frequency, frames = x.shape
        x = x.permute(0, 3, 2, 1).contiguous().view(batch * frames, frequency, channels)
        x, _ = self.gru(x)
        x = self.proj(x)
        return x.view(batch, frames, frequency, channels).permute(0, 3, 2, 1).contiguous()


class _TemporalGRU(nn.Module):
    def __init__(self, channels: int, hidden_size: int) -> None:
        super().__init__()
        self.gru = nn.GRU(channels, hidden_size, num_layers=1, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(hidden_size, channels), nn.ReLU(inplace=True))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, frequency, frames = x.shape
        x = x.permute(0, 2, 3, 1).contiguous().view(batch * frequency, frames, channels)
        x, _ = self.gru(x)
        x = self.proj(x)
        return x.view(batch, frequency, frames, channels).permute(0, 3, 1, 2).contiguous()


class _ScaledDotProductAttention(nn.Module):
    def __init__(self, channels: int, projection_dim: int = 8) -> None:
        super().__init__()
        self.attn_dim = projection_dim
        self.num_heads = 1
        self.head_dim = projection_dim
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Conv2d(channels, projection_dim, kernel_size=1, bias=True)
        self.k_proj = nn.Conv2d(channels, projection_dim, kernel_size=1, bias=True)
        self.v_proj = nn.Conv2d(channels, projection_dim, kernel_size=1, bias=True)
        self.out_proj = nn.Conv2d(projection_dim, channels, kernel_size=1, bias=True)
        self.norm = nn.BatchNorm2d(channels)

    def forward(self, query: torch.Tensor, key_value: torch.Tensor) -> torch.Tensor:
        batch, _, frequency, frames = query.shape
        q = self.q_proj(query).flatten(2).transpose(1, 2).contiguous()
        k = self.k_proj(key_value).flatten(2).transpose(1, 2).contiguous()
        v = self.v_proj(key_value).flatten(2).transpose(1, 2).contiguous()
        tokens = frequency * frames
        q = q.view(batch, tokens, 1, self.head_dim).transpose(1, 2)
        k = k.view(batch, tokens, 1, self.head_dim).transpose(1, 2)
        v = v.view(batch, tokens, 1, self.head_dim).transpose(1, 2)
        attention = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attention = torch.softmax(attention, dim=-1)
        output = torch.matmul(attention, v)
        output = output.transpose(1, 2).contiguous().view(batch, tokens, self.attn_dim)
        output = output.transpose(1, 2).contiguous().view(
            batch, self.attn_dim, frequency, frames
        )
        return self.norm(query + self.out_proj(output))


class _GlobalReliabilityPrior(nn.Module):
    """Global SNR/reliability proxy retained exactly from the canonical model."""

    def __init__(self, n_frequency: int = 256, num_bands: int = 4) -> None:
        super().__init__()
        self.n_freq = n_frequency
        self.num_bands = num_bands
        self.energy_temperature = 4.0
        self.acc_ref_edge = 0.125
        self.eps = 1e-6
        self.acc_ref_end = max(1, min(int(round(self.acc_ref_edge * self.n_freq)), self.n_freq))
        self.band_edges_fixed = self._make_band_edges(self.n_freq)

        # These band-wise parameters remain in the canonical checkpoint even though
        # the selected global forward path does not consume them.
        self.raw_acc_to_mic_slope = nn.Parameter(torch.zeros(1, num_bands, 1, 1))
        self.acc_to_mic_bias = nn.Parameter(torch.zeros(1, num_bands, 1, 1))
        self.log_excess_center = nn.Parameter(torch.zeros(1, num_bands, 1, 1))
        self.raw_log_excess_scale = nn.Parameter(
            torch.tensor(_inverse_softplus(1.0)).view(1, 1, 1, 1).repeat(1, num_bands, 1, 1)
        )
        self.rho_alpha = nn.Parameter(torch.ones(1, num_bands, 1, 1) * 1.5)
        self.rho_beta = nn.Parameter(torch.zeros(1, num_bands, 1, 1))

        self.raw_global_slope = nn.Parameter(torch.zeros(1, 1, 1, 1))
        self.global_bias = nn.Parameter(torch.zeros(1, 1, 1, 1))
        self.global_center = nn.Parameter(torch.zeros(1, 1, 1, 1))
        self.raw_global_scale = nn.Parameter(
            torch.tensor(_inverse_softplus(1.0)).view(1, 1, 1, 1)
        )
        self.global_alpha = nn.Parameter(torch.ones(1, 1, 1, 1) * 1.5)
        self.global_beta = nn.Parameter(torch.zeros(1, 1, 1, 1))

    def _make_band_edges(self, frequency_bins: int) -> list[tuple[int, int]]:
        normalized = [0.0, 0.125, 0.25, 0.5, 1.0]
        indices = [int(round(edge * frequency_bins)) for edge in normalized]
        indices[0], indices[-1] = 0, frequency_bins
        return [
            (
                max(0, min(indices[index], frequency_bins - 1)),
                max(
                    max(0, min(indices[index], frequency_bins - 1)) + 1,
                    min(indices[index + 1], frequency_bins),
                ),
            )
            for index in range(self.num_bands)
        ]

    def _power_proxy(self, x: torch.Tensor) -> torch.Tensor:
        return torch.exp(self.energy_temperature * torch.clamp(x, min=0.0, max=1.0))

    def _expand(self, values: torch.Tensor, frames: int, dtype: torch.dtype) -> torch.Tensor:
        batch = values.shape[0]
        full = torch.zeros(
            batch,
            1,
            self.n_freq,
            frames,
            device=values.device,
            dtype=dtype,
        )
        for band, (start, end) in enumerate(self.band_edges_fixed):
            full[:, :, start:end, :] = values[:, band : band + 1, :, :]
        return full

    def forward(
        self,
        tm: torch.Tensor,
        noisy_am: torch.Tensor,
        target_size: tuple[int, int],
    ) -> torch.Tensor:
        batch = noisy_am.shape[0]
        frames = noisy_am.shape[3]
        tm_power = self._power_proxy(tm)
        am_power = self._power_proxy(noisy_am)
        tm_reference = tm_power[:, :, : self.acc_ref_end, :].mean(
            dim=(2, 3), keepdim=False
        ).clamp_min(self.eps)
        am_energy = am_power.mean(dim=(2, 3), keepdim=False).clamp_min(self.eps)
        tm_log = torch.log(tm_reference).view(batch, 1, 1, 1)
        am_log = torch.log(am_energy).view(batch, 1, 1, 1)
        slope = F.softplus(self.raw_global_slope) + 1e-4
        predicted_speech_log = slope * tm_log + self.global_bias
        log_excess = am_log - predicted_speech_log
        scale = F.softplus(self.raw_global_scale) + self.eps
        standardized = (log_excess - self.global_center) / scale
        rho = torch.sigmoid(self.global_alpha * standardized + self.global_beta)
        rho_bands = rho.repeat(1, self.num_bands, 1, 1)
        rho_full = self._expand(rho_bands, frames, noisy_am.dtype)
        return F.interpolate(rho_full, size=target_size, mode="bilinear", align_corners=False)


class SMSENet(nn.Module):
    """SNR-aware Multimodal Speech Enhancement Network (full model)."""

    def __init__(self) -> None:
        super().__init__()
        am_channels = (12, 24, 48)
        tm_channels = (6, 12, 24)
        hidden_size = 24

        # Four scalar parameters are retained solely because they are present in
        # the published canonical checkpoint and counted in the verified 41,950.
        self.raw_mic_reliable_gain = nn.Parameter(torch.tensor(_inverse_softplus(0.10)))
        self.raw_acc_noise_gain = nn.Parameter(torch.tensor(_inverse_softplus(0.15)))
        self.raw_mic_from_acc_noise_gain = nn.Parameter(torch.tensor(_inverse_softplus(0.25)))
        self.raw_acc_from_mic_noise_gain = nn.Parameter(torch.tensor(_inverse_softplus(0.10)))

        self.raw_rho_tau = nn.Parameter(torch.tensor(_inverse_softplus(2.5)))
        self.raw_s_mic = nn.Parameter(torch.tensor(_inverse_softplus(0.60)))
        self.raw_s_acc = nn.Parameter(torch.tensor(_inverse_softplus(0.45)))
        self.raw_s_mfa = nn.Parameter(torch.tensor(_inverse_softplus(0.75)))
        self.raw_s_afm = nn.Parameter(torch.tensor(_inverse_softplus(0.25)))

        self.mic_down1 = _StandardConv1d(1, am_channels[0])
        self.mic_down2 = _DepthwiseSeparableConv1d(am_channels[0], am_channels[1])
        self.mic_down3 = _DepthwiseSeparableConv1d(am_channels[1], am_channels[2])
        self.acc_down1 = _StandardConv1d(1, tm_channels[0])
        self.acc_down2 = _DepthwiseSeparableConv1d(tm_channels[0], tm_channels[1])
        self.acc_down3 = _DepthwiseSeparableConv1d(tm_channels[1], tm_channels[2])
        self.acc_align3 = nn.Sequential(
            nn.Conv2d(tm_channels[2], am_channels[2], kernel_size=1, bias=False),
            nn.BatchNorm2d(am_channels[2]),
            nn.ReLU(inplace=True),
        )

        self.mic_from_acc_attn = _ScaledDotProductAttention(48, projection_dim=8)
        self.acc_from_mic_attn = _ScaledDotProductAttention(48, projection_dim=8)
        self.support_prior = _GlobalReliabilityPrior(n_frequency=256, num_bands=4)
        self.fuse = nn.Sequential(
            nn.Conv2d(192, 48, kernel_size=1, bias=False),
            nn.BatchNorm2d(48),
            nn.ReLU(inplace=True),
        )
        self.fgru = _FrequencyGRU(48, hidden_size)
        self.tgru = _TemporalGRU(48, hidden_size)
        self.up1 = _DecoderBlock(96, 24)
        self.up2 = _DecoderBlock(48, 12)
        self.up3 = _OutputDecoderBlock(24)
        self.mic_self_attn = _ScaledDotProductAttention(48, projection_dim=8)
        self.acc_self_attn = _ScaledDotProductAttention(48, projection_dim=8)

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        tm = x[:, 0:1, :, :]
        noisy_am = x[:, 1:2, :, :]
        am1 = self.mic_down1(noisy_am)
        am2 = self.mic_down2(am1)
        am3 = self.mic_down3(am2)
        tm1 = self.acc_down1(tm)
        tm2 = self.acc_down2(tm1)
        tm3 = self.acc_down3(tm2)
        tm_aligned = self.acc_align3(tm3)
        return noisy_am, tm, am1, am2, am3, tm_aligned

    def _gain_parameters(self) -> tuple[torch.Tensor, ...]:
        return (
            F.softplus(self.raw_rho_tau) + 1e-4,
            F.softplus(self.raw_s_mic) + 1e-4,
            F.softplus(self.raw_s_acc) + 1e-4,
            F.softplus(self.raw_s_mfa) + 1e-4,
            F.softplus(self.raw_s_afm) + 1e-4,
        )

    def forward_clean(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1:] != (2, 256, 4):
            raise ValueError(f"Expected input [B, 2, 256, 4], received {tuple(x.shape)}")
        noisy_am, tm, am1, am2, am3, tm_aligned = self._encode(x)
        am_self = self.mic_self_attn(am3, am3)
        tm_self = self.acc_self_attn(tm_aligned, tm_aligned)
        am_from_tm = self.mic_from_acc_attn(am3, tm_aligned)
        tm_from_am = self.acc_from_mic_attn(tm_aligned, am3)

        rho = self.support_prior(tm, noisy_am, target_size=(am_self.shape[2], am_self.shape[3]))
        tau, s_am, s_tm, s_am_from_tm, s_tm_from_am = self._gain_parameters()
        sharpened = torch.sigmoid(tau * (rho - 0.5))
        centered = 2.0 * sharpened - 1.0
        gains = [
            torch.exp(-s_am * centered),
            torch.exp(s_tm * centered),
            torch.exp(s_am_from_tm * centered),
            torch.exp(s_tm_from_am * centered),
        ]
        mean_gain = sum(gains) / 4.0
        gains = [gain / mean_gain for gain in gains]
        fused = self.fuse(
            torch.cat(
                [
                    gains[0] * am_self,
                    gains[1] * tm_self,
                    gains[2] * am_from_tm,
                    gains[3] * tm_from_am,
                ],
                dim=1,
            )
        )
        fused = self.fgru(fused)
        fused = self.tgru(fused)
        decoded = self.up1(fused, am3)
        decoded = self.up2(decoded, am2)
        return self.up3(decoded, am1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward_clean(x)


def build_model() -> SMSENet:
    model = SMSENet()
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != CANONICAL_PARAMETER_COUNT:
        raise RuntimeError(
            f"Unexpected parameter count: {parameter_count}; expected {CANONICAL_PARAMETER_COUNT}"
        )
    return model


def _torch_load(path: str | Path, map_location: str | torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: str | torch.device = "cpu",
) -> Mapping[str, Any]:
    checkpoint = _torch_load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, Mapping) or "model_state_dict" not in checkpoint:
        raise ValueError("Expected a training checkpoint containing 'model_state_dict'.")
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    return checkpoint
