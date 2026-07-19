"""PyTorch reimplementation of STA-Net with task-configurable prediction heads.

The backbone preserves the released STA-Net data flow: two fNIRS-guided spatial
alignment (FGSA) blocks, EEG-guided temporal alignment (EGTA), an EEG auxiliary
branch, and learned fusion/fNIRS decision weighting.  TensorFlow is deliberately
not imported.  Classification and continuous sequence regression are explicit
project variants rather than claims about the untouched upstream runner.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


TaskType = Literal["classification", "regression"]


def _pearson_abs_mean(x: Tensor, y: Tensor, eps: float = 1e-6) -> Tensor:
    """Mean absolute Pearson correlation over the feature axis."""

    if x.shape != y.shape:
        raise ValueError(f"Pearson inputs must share shape, got {tuple(x.shape)} and {tuple(y.shape)}")
    x_centered = x - x.mean(dim=1, keepdim=True)
    y_centered = y - y.mean(dim=1, keepdim=True)
    numerator = (x_centered * y_centered).mean(dim=1)
    denominator = x_centered.std(dim=1, unbiased=False) * y_centered.std(dim=1, unbiased=False)
    return (numerator / denominator.clamp_min(eps)).abs().mean()


class SamePadConv3d(nn.Conv3d):
    """Keras-style SAME padding for strided 3-D convolutions."""

    def _padding_for_axis(self, size: int, kernel: int, stride: int, dilation: int) -> tuple[int, int]:
        output = math.ceil(size / stride)
        effective_kernel = dilation * (kernel - 1) + 1
        total = max((output - 1) * stride + effective_kernel - size, 0)
        return total // 2, total - total // 2

    def forward(self, inputs: Tensor) -> Tensor:
        pads = [
            self._padding_for_axis(size, kernel, stride, dilation)
            for size, kernel, stride, dilation in zip(
                inputs.shape[-3:], self.kernel_size, self.stride, self.dilation, strict=True
            )
        ]
        inputs = F.pad(
            inputs,
            (pads[2][0], pads[2][1], pads[1][0], pads[1][1], pads[0][0], pads[0][1]),
        )
        return F.conv3d(
            inputs,
            self.weight,
            self.bias,
            self.stride,
            padding=0,
            dilation=self.dilation,
            groups=self.groups,
        )


class ConvNormELU(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel: tuple[int, int, int], stride: tuple[int, int, int]):
        super().__init__()
        self.conv = SamePadConv3d(in_channels, out_channels, kernel_size=kernel, stride=stride)
        self.norm = nn.BatchNorm3d(out_channels)
        self.activation = nn.ELU()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.activation(self.norm(self.conv(inputs)))


class FgsaLayer(nn.Module):
    """fNIRS-guided spatial alignment with the source correlation regularizer."""

    def __init__(self, channels: int, temporal_kernel: int):
        super().__init__()
        self.channel_pooling = SamePadConv3d(
            channels,
            1,
            kernel_size=(3, 3, temporal_kernel),
            stride=(1, 1, 1),
        )
        self.residual_parameter = nn.Parameter(torch.zeros(()))

    def forward(self, eeg_fusion: Tensor, eeg: Tensor, fnirs: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        if fnirs.ndim != 6:
            raise ValueError(f"fNIRS features must be [B,L,C,H,W,T], got {tuple(fnirs.shape)}")
        batch, lags, channels, height, width, time = fnirs.shape
        pooled = self.channel_pooling(fnirs.reshape(batch * lags, channels, height, width, time))
        pooled = pooled.reshape(batch, lags, 1, pooled.shape[-3], pooled.shape[-2], pooled.shape[-1])
        spatial_attention = torch.sigmoid(pooled.mean(dim=-1, keepdim=True).mean(dim=1))
        guided = eeg_fusion * spatial_attention
        residual_weight = torch.sigmoid(self.residual_parameter)
        aligned = guided + residual_weight * eeg + (1.0 - residual_weight) * eeg_fusion

        eeg_spatial = eeg.mean(dim=(1, 4)).flatten(start_dim=1)
        fnirs_spatial = spatial_attention.flatten(start_dim=1)
        alignment_loss = 1.0 - _pearson_abs_mean(eeg_spatial, fnirs_spatial)
        return aligned, alignment_loss, spatial_attention


class StaConvBlock(nn.Module):
    def __init__(
        self,
        eeg_in_channels: int,
        fnirs_in_channels: int,
        out_channels: int,
        eeg_kernel: tuple[int, int, int],
        eeg_stride: tuple[int, int, int],
        fnirs_kernel: tuple[int, int, int],
        fnirs_stride: tuple[int, int, int],
        temporal_kernel: int,
    ):
        super().__init__()
        self.eeg = ConvNormELU(eeg_in_channels, out_channels, eeg_kernel, eeg_stride)
        self.fnirs = ConvNormELU(fnirs_in_channels, out_channels, fnirs_kernel, fnirs_stride)
        self.eeg_fusion = ConvNormELU(eeg_in_channels, out_channels, eeg_kernel, eeg_stride)
        self.fgsa = FgsaLayer(out_channels, temporal_kernel)

    def _fnirs_forward(self, fnirs: Tensor) -> Tensor:
        batch, lags, channels, height, width, time = fnirs.shape
        features = self.fnirs(fnirs.reshape(batch * lags, channels, height, width, time))
        return features.reshape(batch, lags, *features.shape[1:])

    def forward(self, eeg_fusion: Tensor, eeg: Tensor, fnirs: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        eeg_features = self.eeg(eeg)
        fnirs_features = self._fnirs_forward(fnirs)
        fusion_features = self.eeg_fusion(eeg_fusion)
        aligned, alignment_loss, spatial_attention = self.fgsa(
            fusion_features, eeg_features, fnirs_features
        )
        return aligned, eeg_features, fnirs_features, alignment_loss, spatial_attention


class KerasStyleMultiHeadAttention(nn.Module):
    """Multi-head attention matching Keras' independent per-head ``key_dim``."""

    def __init__(self, input_dim: int, num_heads: int, key_dim: int, dropout: float):
        super().__init__()
        self.num_heads = int(num_heads)
        self.key_dim = int(key_dim)
        projection_dim = self.num_heads * self.key_dim
        self.query = nn.Linear(input_dim, projection_dim)
        self.key = nn.Linear(input_dim, projection_dim)
        self.value = nn.Linear(input_dim, projection_dim)
        self.output = nn.Linear(projection_dim, input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query: Tensor, key_value: Tensor) -> tuple[Tensor, Tensor]:
        batch, query_length, _ = query.shape
        key_length = key_value.shape[1]
        q = self.query(query).reshape(batch, query_length, self.num_heads, self.key_dim).transpose(1, 2)
        k = self.key(key_value).reshape(batch, key_length, self.num_heads, self.key_dim).transpose(1, 2)
        v = self.value(key_value).reshape(batch, key_length, self.num_heads, self.key_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.key_dim)
        attention = torch.softmax(scores, dim=-1)
        context = torch.matmul(self.dropout(attention), v)
        context = context.transpose(1, 2).reshape(batch, query_length, self.num_heads * self.key_dim)
        return self.output(context), attention


class EgtaLayer(nn.Module):
    """EEG-guided temporal alignment over lagged fNIRS features."""

    def __init__(
        self,
        flat_dim: int,
        embedding_dim: int,
        num_heads: int,
        key_dim: int,
        dropout: float,
        max_lags: int,
    ):
        super().__init__()
        self.fusion_projection = nn.Linear(flat_dim, embedding_dim)
        self.query_projection = nn.Linear(flat_dim, embedding_dim)
        self.key_projection = nn.Linear(flat_dim, embedding_dim)
        self.position = nn.Parameter(torch.empty(1, max_lags, embedding_dim))
        nn.init.kaiming_uniform_(self.position, a=math.sqrt(5))
        self.attention = KerasStyleMultiHeadAttention(embedding_dim, num_heads, key_dim, dropout)

    def forward(self, eeg_fusion: Tensor, fnirs: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        eeg_flat = eeg_fusion.flatten(start_dim=1)
        fnirs_flat = fnirs.flatten(start_dim=2)
        lags = fnirs_flat.shape[1]
        if lags > self.position.shape[1]:
            raise ValueError(f"Received {lags} fNIRS lags but model max_lags={self.position.shape[1]}")
        fusion_feature = self.fusion_projection(eeg_flat)
        query = self.query_projection(eeg_flat).unsqueeze(1)
        key_value = self.key_projection(fnirs_flat) + self.position[:, :lags]
        weighted, attention = self.attention(query, key_value)
        weighted = weighted.mean(dim=1)
        alignment_loss = 1.0 - _pearson_abs_mean(query.mean(dim=1), weighted)
        lag_attention = attention.mean(dim=(1, 2))
        return fusion_feature, weighted, lag_attention, alignment_loss


@dataclass(frozen=True)
class STANetConfig:
    task_type: TaskType
    output_dim: int
    sequence_length: int = 1
    dropout: float = 0.5
    embedding_dim: int = 256
    attention_heads: int = 10
    attention_key_dim: int = 256
    max_lags: int = 18

    def __post_init__(self) -> None:
        if self.task_type not in {"classification", "regression"}:
            raise ValueError(f"Unsupported STA-Net task type: {self.task_type}")
        if self.output_dim <= 0 or self.sequence_length <= 0:
            raise ValueError("output_dim and sequence_length must be positive")
        if self.task_type == "classification" and self.sequence_length != 1:
            raise ValueError("classification variants require sequence_length=1")


class STANet(nn.Module):
    """STA-Net backbone with binary, multiclass, or sequence-regression heads."""

    def __init__(self, config: STANetConfig):
        super().__init__()
        self.config = config
        self.block1 = StaConvBlock(
            1,
            2,
            16,
            eeg_kernel=(2, 2, 13),
            eeg_stride=(2, 2, 6),
            fnirs_kernel=(2, 2, 5),
            fnirs_stride=(2, 2, 2),
            temporal_kernel=5,
        )
        self.block2 = StaConvBlock(
            16,
            16,
            32,
            eeg_kernel=(2, 2, 5),
            eeg_stride=(2, 2, 2),
            fnirs_kernel=(2, 2, 3),
            fnirs_stride=(2, 2, 2),
            temporal_kernel=3,
        )
        self.dropout = nn.Dropout(config.dropout)
        flat_dim = 32 * 4 * 4
        self.egta = EgtaLayer(
            flat_dim=flat_dim,
            embedding_dim=config.embedding_dim,
            num_heads=config.attention_heads,
            key_dim=config.attention_key_dim,
            dropout=config.dropout,
            max_lags=config.max_lags,
        )
        self.fusion_hidden = nn.Sequential(nn.ELU(), nn.Linear(config.embedding_dim, config.embedding_dim), nn.ELU())
        self.fnirs_hidden = nn.Sequential(nn.ELU(), nn.Linear(config.embedding_dim, config.embedding_dim), nn.ELU())
        self.eeg_hidden = nn.Sequential(nn.Linear(flat_dim, config.embedding_dim), nn.ELU())
        head_dim = config.output_dim * config.sequence_length
        self.fusion_head = nn.Linear(config.embedding_dim, head_dim)
        self.fnirs_head = nn.Linear(config.embedding_dim, head_dim)
        self.eeg_head = nn.Linear(config.embedding_dim, head_dim)
        self.fusion_gate = nn.Linear(config.embedding_dim, 1)
        self.fnirs_gate = nn.Linear(config.embedding_dim, 1)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv3d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _reshape_prediction(self, values: Tensor) -> Tensor:
        if self.config.task_type == "classification":
            return values
        return values.reshape(values.shape[0], self.config.output_dim, self.config.sequence_length)

    def forward(self, eeg: Tensor, fnirs: Tensor) -> dict[str, Tensor | Mapping[str, Tensor]]:
        if eeg.ndim != 5 or eeg.shape[1] != 1:
            raise ValueError(f"EEG must be [B,1,H,W,T], got {tuple(eeg.shape)}")
        if fnirs.ndim != 6 or fnirs.shape[2] != 2:
            raise ValueError(f"fNIRS must be [B,L,2,H,W,T], got {tuple(fnirs.shape)}")

        fusion1, eeg1, fnirs1, fgsa1_loss, attention1 = self.block1(eeg, eeg, fnirs)
        fusion1, eeg1, fnirs1 = self.dropout(fusion1), self.dropout(eeg1), self.dropout(fnirs1)
        fusion2, eeg2, fnirs2, fgsa2_loss, attention2 = self.block2(fusion1, eeg1, fnirs1)
        fusion2 = self.dropout(fusion2.mean(dim=-1, keepdim=True))
        eeg2 = self.dropout(eeg2.mean(dim=-1, keepdim=True))
        fnirs2 = self.dropout(fnirs2.mean(dim=-1, keepdim=True))

        fusion_feature, fnirs_feature, lag_attention, egta_loss = self.egta(fusion2, fnirs2)
        fusion_feature = self.fusion_hidden(fusion_feature)
        fnirs_feature = self.fnirs_hidden(fnirs_feature)
        eeg_feature = self.eeg_hidden(eeg2.flatten(start_dim=1))

        fusion_raw = self._reshape_prediction(self.fusion_head(fusion_feature))
        fnirs_raw = self._reshape_prediction(self.fnirs_head(fnirs_feature))
        eeg_raw = self._reshape_prediction(self.eeg_head(eeg_feature))
        gates = torch.softmax(
            torch.cat([self.fusion_gate(fusion_feature), self.fnirs_gate(fnirs_feature)], dim=1), dim=1
        )
        if self.config.task_type == "classification":
            fusion_prediction = torch.softmax(fusion_raw, dim=-1)
            fnirs_prediction = torch.softmax(fnirs_raw, dim=-1)
            prediction = gates[:, :1] * fusion_prediction + gates[:, 1:] * fnirs_prediction
            eeg_prediction = torch.softmax(eeg_raw, dim=-1)
        else:
            gate_shape = (gates.shape[0], 1, 1)
            prediction = gates[:, :1].reshape(gate_shape) * fusion_raw + gates[:, 1:].reshape(gate_shape) * fnirs_raw
            fusion_prediction, fnirs_prediction, eeg_prediction = fusion_raw, fnirs_raw, eeg_raw

        return {
            "prediction": prediction,
            "eeg_prediction": eeg_prediction,
            "fusion_prediction": fusion_prediction,
            "fnirs_prediction": fnirs_prediction,
            "fusion_weights": gates,
            "lag_attention": lag_attention,
            "spatial_attention_block1": attention1,
            "spatial_attention_block2": attention2,
            "alignment_losses": {
                "fgsa1": fgsa1_loss,
                "fgsa2": fgsa2_loss,
                "egta": egta_loss,
            },
        }


class STANetObjective(nn.Module):
    """Source-shaped objective for configurable STA-Net task heads."""

    def __init__(
        self,
        task_type: TaskType,
        *,
        main_weight: float = 1.0,
        eeg_aux_weight: float = 1.0,
        alignment_weight: float = 1.0,
        regression_loss: Literal["mse", "smooth_l1"] = "smooth_l1",
        class_weights: Tensor | None = None,
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.task_type = task_type
        self.main_weight = float(main_weight)
        self.eeg_aux_weight = float(eeg_aux_weight)
        self.alignment_weight = float(alignment_weight)
        self.regression_loss = regression_loss
        self.label_smoothing = float(label_smoothing)
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        self.register_buffer("class_weights", class_weights)

    def _masked_regression(self, prediction: Tensor, target: Tensor, mask: Tensor) -> Tensor:
        if prediction.shape != target.shape or target.shape != mask.shape:
            raise ValueError(
                f"Regression prediction/target/mask mismatch: {prediction.shape}, {target.shape}, {mask.shape}"
            )
        elementwise = (
            F.smooth_l1_loss(prediction, target, reduction="none")
            if self.regression_loss == "smooth_l1"
            else F.mse_loss(prediction, target, reduction="none")
        )
        weights = mask.to(dtype=elementwise.dtype)
        return (elementwise * weights).sum() / weights.sum().clamp_min(1.0)

    def forward(
        self,
        outputs: Mapping[str, Tensor | Mapping[str, Tensor]],
        target: Tensor,
        target_valid_mask: Tensor | None = None,
    ) -> dict[str, Tensor]:
        prediction = outputs["prediction"]
        eeg_prediction = outputs["eeg_prediction"]
        if not isinstance(prediction, Tensor) or not isinstance(eeg_prediction, Tensor):
            raise TypeError("STA-Net predictions must be tensors")
        if self.task_type == "classification":
            main = F.cross_entropy(
                prediction.clamp_min(1e-8).log(), target.long(),
                weight=self.class_weights, label_smoothing=self.label_smoothing,
            )
            eeg_aux = F.cross_entropy(
                eeg_prediction.clamp_min(1e-8).log(), target.long(),
                weight=self.class_weights, label_smoothing=self.label_smoothing,
            )
        else:
            if target_valid_mask is None:
                raise ValueError("Regression STA-Net requires target_valid_mask")
            main = self._masked_regression(prediction, target, target_valid_mask)
            eeg_aux = self._masked_regression(eeg_prediction, target, target_valid_mask)
        alignment_values = outputs["alignment_losses"]
        if not isinstance(alignment_values, Mapping):
            raise TypeError("alignment_losses must be a mapping")
        alignment = torch.stack([value for value in alignment_values.values()]).sum()
        total = self.main_weight * main + self.eeg_aux_weight * eeg_aux + self.alignment_weight * alignment
        return {
            "total": total,
            "main": main,
            "eeg_aux": eeg_aux,
            "alignment": alignment,
            **{f"alignment_{name}": value for name, value in alignment_values.items()},
        }
