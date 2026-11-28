"""Isolated mask-correct InfoGCN variant for variable-length pose clips.

This implementation does not alter or replace EXP10RERUN. It retains its model
topology while making temporal normalization and masking explicit.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from experiments.exp10rerun.model_exp10rerun import (
    NUM_JOINTS,
    SelfAttentionGraphConv,
    build_adjacency,
)


def _weights(mask: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return mask[:, None, :, None].to(dtype=x.dtype)


def _zero_invalid(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Selection, not multiplication: invalid NaN/Inf values are discarded."""
    return torch.where(mask[:, None, :, None], x, torch.zeros((), dtype=x.dtype, device=x.device))


class MaskedBatchNorm(nn.Module):
    """BatchNorm over valid temporal positions only for N,C,T or N,C,T,V tensors."""

    def __init__(self, channels: int, dims: int, eps: float = 1e-5, momentum: float | None = 0.1,
                 affine: bool = True, track_running_stats: bool = True):
        super().__init__()
        if dims not in (1, 2):
            raise ValueError("dims must be 1 or 2")
        self.channels, self.dims, self.eps = channels, dims, eps
        self.momentum, self.track_running_stats = momentum, track_running_stats
        if affine:
            self.weight = nn.Parameter(torch.ones(channels))
            self.bias = nn.Parameter(torch.zeros(channels))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        if track_running_stats:
            self.register_buffer("running_mean", torch.zeros(channels))
            self.register_buffer("running_var", torch.ones(channels))
            self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.register_buffer("num_batches_tracked", None)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != self.dims + 2:
            raise ValueError(f"Expected {self.dims + 2}D input, got {x.ndim}D")
        if mask.shape != (x.shape[0], x.shape[2]):
            raise ValueError(f"Expected mask {(x.shape[0], x.shape[2])}, got {tuple(mask.shape)}")
        if x.shape[1] != self.channels:
            raise ValueError("Channel count mismatch")

        valid = mask[:, None, :, None] if self.dims == 2 else mask[:, None, :]
        valid = valid.to(device=x.device, dtype=torch.bool)
        # Sanitize before any arithmetic so masked NaN/Inf cannot enter reductions.
        clean = torch.where(valid, x, torch.zeros((), dtype=x.dtype, device=x.device))
        reduce_dims = (0,) + tuple(range(2, x.ndim))
        view_shape = (1, self.channels) + (1,) * self.dims
        use_batch = self.training or not self.track_running_stats
        count = int(mask.sum().item()) * (x.shape[3] if self.dims == 2 else 1)

        if use_batch and count > 0:
            denom = float(count)
            mean = clean.sum(dim=reduce_dims) / denom
            centered = torch.where(valid, x - mean.view(view_shape), torch.zeros((), dtype=x.dtype, device=x.device))
            var = centered.square().sum(dim=reduce_dims) / denom
            if self.training and self.track_running_stats:
                with torch.no_grad():
                    self.num_batches_tracked.add_(1)
                    factor = (1.0 / float(self.num_batches_tracked)) if self.momentum is None else self.momentum
                    unbiased = var.detach() * (denom / (denom - 1.0)) if denom > 1 else var.detach()
                    self.running_mean.lerp_(mean.detach().to(self.running_mean.dtype), factor)
                    self.running_var.lerp_(unbiased.to(self.running_var.dtype), factor)
        elif self.track_running_stats:
            mean, var = self.running_mean, self.running_var
        else:
            mean = torch.zeros(self.channels, device=x.device, dtype=x.dtype)
            var = torch.ones(self.channels, device=x.device, dtype=x.dtype)

        y = (clean - mean.view(view_shape)) * torch.rsqrt(var.view(view_shape) + self.eps)
        if self.weight is not None:
            y = y * self.weight.view(view_shape) + self.bias.view(view_shape)
        return torch.where(valid, y, torch.zeros((), dtype=y.dtype, device=y.device))


class MaskedTemporalConv(nn.Module):
    def __init__(self, channels: int, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        if channels % 4:
            raise ValueError("channels must be divisible by four")
        self.stride = stride
        b = channels // 4
        self.pre_d1 = nn.Conv2d(channels, b, 1, bias=False)
        self.bn_d1 = MaskedBatchNorm(b, 2)
        self.conv_d1 = nn.Conv2d(b, b, (3, 1), stride=(stride, 1), padding=(1, 0), bias=False)
        self.pre_d2 = nn.Conv2d(channels, b, 1, bias=False)
        self.bn_d2 = MaskedBatchNorm(b, 2)
        self.conv_d2 = nn.Conv2d(b, b, (3, 1), stride=(stride, 1), padding=(2, 0), dilation=(2, 1), bias=False)
        self.pool_conv = nn.Conv2d(channels, b, 1, bias=False)
        self.point = nn.Conv2d(channels, b, 1, stride=(stride, 1), bias=False)
        self.out_bn = MaskedBatchNorm(channels, 2)
        self.dropout = nn.Dropout(dropout)

    @staticmethod
    def _center_mask(mask: torch.Tensor, stride: int, out_t: int) -> torch.Tensor:
        result = mask[:, ::stride]
        if result.shape[1] != out_t:
            raise RuntimeError(f"Mask length {result.shape[1]} does not match convolution output {out_t}")
        return result

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = _zero_invalid(x, mask)
        out_mask = self._center_mask(mask, self.stride, (x.shape[2] + 2 - 3) // self.stride + 1)

        # Pointwise normalization/activation can create values at invalid sites; erase them
        # before any temporal convolution.
        d1 = F.relu(self.bn_d1(self.pre_d1(x), mask), inplace=False)
        d1 = _zero_invalid(d1, mask)
        d1 = self.conv_d1(d1)
        d2 = F.relu(self.bn_d2(self.pre_d2(x), mask), inplace=False)
        d2 = _zero_invalid(d2, mask)
        d2 = self.conv_d2(d2)

        # Masked max pooling treats invalid neighbors as absent, not as zero-valued data.
        padded_x = F.pad(x, (0, 0, 1, 1), value=0.0)
        padded_m = F.pad(mask, (1, 1), value=False)
        windows_x = padded_x.unfold(2, 3, self.stride)
        windows_m = padded_m.unfold(1, 3, self.stride)
        windows_x = windows_x.masked_fill(~windows_m[:, None, :, None, :], -torch.inf)
        pooled = windows_x.max(dim=4).values
        has_valid = windows_m.any(dim=2)
        pooled = torch.where(has_valid[:, None, :, None], pooled, torch.zeros_like(pooled))
        pool = self.pool_conv(pooled)
        point = self.point(x)

        out = torch.cat((d1, d2, pool, point), dim=1)
        out = self.out_bn(out, out_mask)
        out = self.dropout(out)
        return _zero_invalid(out, out_mask), out_mask


class MaskedInfoGCNBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, num_joints: int, stride: int, dropout: float):
        super().__init__()
        self.stride = stride
        self.spatial = SelfAttentionGraphConv(in_channels, out_channels, num_joints=num_joints,
                                              num_heads=3, attn_dim=max(8, out_channels // 4))
        self.spatial_bn = MaskedBatchNorm(out_channels, 2)
        self.temporal = MaskedTemporalConv(out_channels, stride, dropout)
        if in_channels == out_channels and stride == 1:
            self.residual_conv = None
            self.residual_bn = None
        else:
            self.residual_conv = nn.Conv2d(in_channels, out_channels, 1, stride=(stride, 1), bias=False)
            self.residual_bn = MaskedBatchNorm(out_channels, 2)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor, mask: torch.Tensor):
        x = _zero_invalid(x, mask)
        y = self.spatial(x, adjacency, mask=mask)
        y = self.relu(self.spatial_bn(y, mask))
        y = _zero_invalid(y, mask)
        y, out_mask = self.temporal(y, mask)

        residual = x
        if self.residual_conv is not None:
            residual = self.residual_conv(x)
            residual = self.residual_bn(residual, out_mask)
        residual = _zero_invalid(residual, out_mask)
        out = self.relu(y + residual)
        return _zero_invalid(out, out_mask), out_mask


class MaskedInfoGCN(nn.Module):
    """EXP10RERUN-topology classifier with valid-position-only temporal normalization."""

    def __init__(self, in_channels: int = 3, num_joints: int = NUM_JOINTS, num_classes: int = 41,
                 base_channels: int = 64, dropout: float = 0.1, head_dropout: float = 0.3,
                 adjacency: torch.Tensor | None = None, use_ib: bool = False, latent_dim: int = 128):
        super().__init__()
        if base_channels % 4:
            raise ValueError("base_channels must be divisible by four")
        adj = torch.tensor(build_adjacency(), dtype=torch.float32) if adjacency is None else torch.as_tensor(adjacency, dtype=torch.float32)
        if tuple(adj.shape) != (3, num_joints, num_joints):
            raise ValueError(f"Expected adjacency shape (3,{num_joints},{num_joints}), got {tuple(adj.shape)}")
        self.in_channels, self.num_joints, self.num_classes = in_channels, num_joints, num_classes
        self.use_ib = use_ib
        self.register_buffer("A", adj)
        self.data_bn = MaskedBatchNorm(in_channels * num_joints, 1)
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.stem_conv = nn.Conv2d(in_channels, c1, 1, bias=False)
        self.stem_bn = MaskedBatchNorm(c1, 2)
        self.joint_embed = nn.Parameter(torch.zeros(1, c1, 1, num_joints))
        nn.init.trunc_normal_(self.joint_embed, std=0.02)
        specs = [(c1,c1,1),(c1,c1,1),(c1,c2,2),(c2,c2,1),(c2,c3,2)]
        self.blocks = nn.ModuleList([MaskedInfoGCNBlock(a,b,num_joints,s,dropout) for a,b,s in specs])
        self.head_dropout = nn.Dropout(head_dropout)
        if use_ib:
            self.fc_mu = nn.Linear(c3, latent_dim)
            self.fc_logvar = nn.Linear(c3, latent_dim)
            self.classifier = nn.Linear(latent_dim, num_classes)
        else:
            self.fc_mu = self.fc_logvar = None
            self.classifier = nn.Linear(c3, num_classes)

    def forward(self, x: torch.Tensor, frame_mask: torch.Tensor | None = None, return_latent: bool = False):
        if x.ndim == 5 and x.shape[-1] == 1:
            x = x.squeeze(-1)
        if x.ndim != 4 or x.shape[1] != self.in_channels or x.shape[3] != self.num_joints:
            raise ValueError(f"Expected (N,{self.in_channels},T,{self.num_joints}), got {tuple(x.shape)}")
        n, c, t, v = x.shape
        if frame_mask is None:
            mask = torch.ones((n,t), dtype=torch.bool, device=x.device)
        else:
            if frame_mask.shape != (n,t):
                raise ValueError(f"Expected mask {(n,t)}, got {tuple(frame_mask.shape)}")
            mask = frame_mask.to(device=x.device, dtype=torch.bool)
        if not mask.any(dim=1).all():
            raise ValueError("Every sequence must contain at least one valid frame")

        x = _zero_invalid(x, mask)
        x = x.permute(0,3,1,2).contiguous().view(n,v*c,t)
        x = self.data_bn(x, mask)
        x = x.view(n,v,c,t).permute(0,2,3,1).contiguous()
        x = self.stem_conv(x)
        x = F.relu(self.stem_bn(x, mask), inplace=False) + self.joint_embed
        x = _zero_invalid(x, mask)
        for block in self.blocks:
            x, mask = block(x, self.A, mask)
        w = _weights(mask, x)
        denom = (w.sum(dim=2).squeeze(-1) * v).clamp_min(1.0)
        pooled = (x * w).sum(dim=(2,3)) / denom
        pooled = self.head_dropout(pooled)
        if self.use_ib:
            mu = self.fc_mu(pooled)
            logvar = self.fc_logvar(pooled).clamp(min=-10.0, max=2.0)
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(logvar) if self.training else mu
            logits = self.classifier(z)
            return (logits, mu, logvar, z) if return_latent else logits
        logits = self.classifier(pooled)
        return (logits, None, None, pooled) if return_latent else logits


def build_model(**kwargs) -> MaskedInfoGCN:
    return MaskedInfoGCN(**kwargs)
