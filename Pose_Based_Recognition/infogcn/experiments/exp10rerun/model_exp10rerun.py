"""InfoGCN-inspired model architecture for 50-joint, 100-frame, 3-channel Kathakali poses (EXP10RERUN)."""

import importlib.util
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

INFOGCN_ROOT = Path(__file__).resolve().parents[2]
STGCN_ROOT = INFOGCN_ROOT.parent / "stgcn"


def _load_stgcn_graph():
    """Load read-only graph constants/builder from stgcn without writing bytecode there."""
    graph_path = STGCN_ROOT / "shared/graph/graph.py"
    prev_dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        spec = importlib.util.spec_from_file_location("stgcn_shared_graph", graph_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not load graph module from {graph_path}")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.NUM_JOINTS, module.build_adjacency
    finally:
        sys.dont_write_bytecode = prev_dont_write


NUM_JOINTS, build_adjacency = _load_stgcn_graph()


class SelfAttentionGraphConv(nn.Module):
    """Three-head Self-Attention Graph Convolution (SA-GC).

    Combines the 3 anatomical ST-GCN partitions (self, centripetal, centrifugal)
    with clip-level pairwise joint self-attention and a learnable spatial bias,
    implemented in pure PyTorch without external tensor-rearrangement packages.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_joints: int = NUM_JOINTS,
        num_heads: int = 3,
        attn_dim: int = 16,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.num_heads = num_heads
        self.attn_dim = attn_dim

        self.q_proj = nn.Conv1d(in_channels, num_heads * attn_dim, kernel_size=1, bias=False)
        self.k_proj = nn.Conv1d(in_channels, num_heads * attn_dim, kernel_size=1, bias=False)
        self.v_projs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for _ in range(num_heads)
        ])

        self.spatial_bias = nn.Parameter(torch.zeros(num_heads, num_joints, num_joints))
        # Non-zero init ensures gradients flow into Q/K/spatial_bias from epoch 1.
        self.attn_scale = nn.Parameter(torch.full((num_heads,), 0.5))

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply SA-GC to `(N, C_in, T, V)` using structural `adjacency (H, V, V)`."""
        n, _, _, v = x.shape

        if mask is not None:
            weights = mask[:, None, :, None].to(x.dtype)
            denom = weights.sum(dim=2).clamp(min=1.0)
            pooled = (x * weights).sum(dim=2) / denom
        else:
            pooled = x.mean(dim=2)

        q = self.q_proj(pooled).view(n, self.num_heads, self.attn_dim, v).permute(0, 1, 3, 2)
        k = self.k_proj(pooled).view(n, self.num_heads, self.attn_dim, v).permute(0, 1, 3, 2)

        attn_logits = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.attn_dim)
        attn_logits = attn_logits + self.spatial_bias.unsqueeze(0)
        attn = F.softmax(attn_logits, dim=-2)

        alpha = self.attn_scale.view(1, self.num_heads, 1, 1)
        effective_adj = adjacency.unsqueeze(0) + alpha * attn

        out = None
        for h, v_proj in enumerate(self.v_projs):
            projected = v_proj(x)
            agg = torch.einsum("nctv,nvw->nctw", projected, effective_adj[:, h])
            out = agg if out is None else out + agg

        return out


class MultiScaleTemporalConv(nn.Module):
    """Four-branch Multi-Scale Temporal Convolution (MS-TC) for 1D skeleton sequences."""

    def __init__(self, channels: int, stride: int = 1, dropout: float = 0.1):
        super().__init__()
        if channels % 4 != 0:
            raise ValueError(f"MS-TC channels must be divisible by 4, got {channels}")

        branch_ch = channels // 4

        self.branch_d1 = nn.Sequential(
            nn.Conv2d(channels, branch_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                branch_ch,
                branch_ch,
                kernel_size=(3, 1),
                stride=(stride, 1),
                padding=(1, 0),
                dilation=(1, 1),
                bias=False,
            ),
        )

        self.branch_d2 = nn.Sequential(
            nn.Conv2d(channels, branch_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(branch_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                branch_ch,
                branch_ch,
                kernel_size=(3, 1),
                stride=(stride, 1),
                padding=(2, 0),
                dilation=(2, 1),
                bias=False,
            ),
        )

        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.Conv2d(channels, branch_ch, kernel_size=1, bias=False),
        )

        self.branch_1x1 = nn.Conv2d(
            channels,
            branch_ch,
            kernel_size=1,
            stride=(stride, 1),
            bias=False,
        )

        self.bn = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = torch.cat(
            [
                self.branch_d1(x),
                self.branch_d2(x),
                self.branch_pool(x),
                self.branch_1x1(x),
            ],
            dim=1,
        )
        return self.dropout(self.bn(out))


class InfoGCNBlock(nn.Module):
    """Residual SA-GC + MS-TC spatio-temporal block with frame-mask propagation."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_joints: int = NUM_JOINTS,
        num_heads: int = 3,
        stride: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.stride = stride

        self.spatial = SelfAttentionGraphConv(
            in_channels=in_channels,
            out_channels=out_channels,
            num_joints=num_joints,
            num_heads=num_heads,
            attn_dim=max(8, out_channels // 4),
        )
        self.spatial_bn = nn.BatchNorm2d(out_channels)
        self.temporal = MultiScaleTemporalConv(
            channels=out_channels,
            stride=stride,
            dropout=dropout,
        )

        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(
        self,
        x: torch.Tensor,
        adjacency: torch.Tensor,
        mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        input_weights = mask[:, None, :, None].to(x.dtype)
        output_mask = mask[:, :: self.stride]
        output_weights = output_mask[:, None, :, None].to(x.dtype)

        residual = self.residual(x) * output_weights

        y = self.spatial(x, adjacency, mask=mask)
        y = self.relu(self.spatial_bn(y))
        y = y * input_weights

        y = self.temporal(y)
        y = self.relu(y + residual)
        y = y * output_weights

        return y, output_mask


class InfoGCNLoss(nn.Module):
    """Classification loss with optional Information Bottleneck (KL + RBF-MMD) regularization."""

    def __init__(
        self,
        label_smoothing: float = 0.0,
        use_ib: bool = False,
        lambda_kl: float = 1e-4,
        lambda_mmd: float = 1e-3,
    ):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.use_ib = use_ib
        self.lambda_kl = lambda_kl
        self.lambda_mmd = lambda_mmd

    @staticmethod
    def _rbf_mmd(z: torch.Tensor, prior: torch.Tensor) -> torch.Tensor:
        z_sq = (z * z).sum(dim=1, keepdim=True)
        p_sq = (prior * prior).sum(dim=1, keepdim=True)

        dist_zz = torch.clamp(z_sq + z_sq.t() - 2.0 * (z @ z.t()), min=0.0)
        dist_pp = torch.clamp(p_sq + p_sq.t() - 2.0 * (prior @ prior.t()), min=0.0)
        dist_zp = torch.clamp(z_sq + p_sq.t() - 2.0 * (z @ prior.t()), min=0.0)

        sigma = 2.0 * z.shape[1]
        k_zz = torch.exp(-dist_zz / sigma).mean()
        k_pp = torch.exp(-dist_pp / sigma).mean()
        k_zp = torch.exp(-dist_zp / sigma).mean()
        return k_zz + k_pp - 2.0 * k_zp

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        mu: torch.Tensor | None = None,
        logvar: torch.Tensor | None = None,
        z: torch.Tensor | None = None,
        kl_scale: float = 1.0,
    ) -> torch.Tensor:
        loss = self.ce(logits, targets)
        if not self.use_ib or mu is None or logvar is None or z is None:
            return loss

        kl = -0.5 * torch.mean(torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        prior = torch.randn_like(z)
        mmd = self._rbf_mmd(z, prior)
        return loss + (self.lambda_kl * kl_scale) * kl + self.lambda_mmd * mmd


class InfoGCN(nn.Module):
    """InfoGCN-inspired network adapted for 41-class, 100-frame, 50-joint, 3-channel Kathakali data."""

    def __init__(
        self,
        num_classes: int = 41,
        in_channels: int = 3,
        num_joints: int = NUM_JOINTS,
        base_channels: int = 48,
        dropout: float = 0.15,
        head_dropout: float = 0.3,
        use_ib: bool = False,
        latent_dim: int = 128,
        adjacency: torch.Tensor | None = None,
    ):
        super().__init__()
        if base_channels % 4 != 0:
            raise ValueError(f"base_channels must be divisible by 4, got {base_channels}")

        if adjacency is None:
            adj_tensor = torch.tensor(build_adjacency(), dtype=torch.float32)
        else:
            adj_tensor = torch.as_tensor(adjacency, dtype=torch.float32)

        if adj_tensor.shape != (3, num_joints, num_joints):
            raise ValueError(
                f"Expected adjacency shape (3, {num_joints}, {num_joints}), got {tuple(adj_tensor.shape)}"
            )

        self.in_channels = in_channels
        self.num_joints = num_joints
        self.num_classes = num_classes
        self.use_ib = use_ib
        self.register_buffer("A", adj_tensor)

        # Standardizes neck-centered XY and raw SimCC confidence per joint-channel.
        self.data_bn = nn.BatchNorm1d(in_channels * num_joints)

        c1 = base_channels
        c2 = base_channels * 2
        c3 = base_channels * 4

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )
        self.joint_embed = nn.Parameter(torch.zeros(1, c1, 1, num_joints))
        nn.init.trunc_normal_(self.joint_embed, std=0.02)

        self.blocks = nn.ModuleList([
            InfoGCNBlock(c1, c1, num_joints=num_joints, num_heads=3, stride=1, dropout=dropout),
            InfoGCNBlock(c1, c1, num_joints=num_joints, num_heads=3, stride=1, dropout=dropout),
            InfoGCNBlock(c1, c2, num_joints=num_joints, num_heads=3, stride=2, dropout=dropout),
            InfoGCNBlock(c2, c2, num_joints=num_joints, num_heads=3, stride=1, dropout=dropout),
            InfoGCNBlock(c2, c3, num_joints=num_joints, num_heads=3, stride=2, dropout=dropout),
        ])

        self.head_dropout = nn.Dropout(head_dropout)
        if use_ib:
            self.fc_mu = nn.Linear(c3, latent_dim)
            self.fc_logvar = nn.Linear(c3, latent_dim)
            self.classifier = nn.Linear(latent_dim, num_classes)
        else:
            self.fc_mu = None
            self.fc_logvar = None
            self.classifier = nn.Linear(c3, num_classes)

    def forward(
        self,
        x: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
        return_latent: bool = False,
    ):
        if x.ndim == 5 and x.shape[-1] == 1:
            x = x.squeeze(-1)

        if x.ndim != 4 or x.shape[1] != self.in_channels or x.shape[3] != self.num_joints:
            raise ValueError(
                f"Expected input (N, {self.in_channels}, T, {self.num_joints}), got {tuple(x.shape)}"
            )

        n, c, t, v = x.shape
        if frame_mask is None:
            mask = torch.ones((n, t), dtype=torch.bool, device=x.device)
        else:
            if frame_mask.shape != (n, t):
                raise ValueError(f"Expected frame_mask shape ({n}, {t}), got {tuple(frame_mask.shape)}")
            mask = frame_mask.bool()

        # Per-joint-channel normalization matching existing ST-GCN convention.
        x = x.permute(0, 3, 1, 2).contiguous().view(n, v * c, t)
        x = self.data_bn(x)
        x = x.view(n, v, c, t).permute(0, 2, 3, 1).contiguous()

        weights = mask[:, None, :, None].to(x.dtype)
        x = x * weights

        x = self.stem(x) + self.joint_embed
        x = x * weights

        for block in self.blocks:
            x, mask = block(x, self.A, mask)

        out_weights = mask[:, None, :, None].to(x.dtype)
        denom = (out_weights.sum(dim=2).squeeze(-1) * v).clamp(min=1.0)
        pooled = (x * out_weights).sum(dim=(2, 3)) / denom
        pooled = self.head_dropout(pooled)

        if self.use_ib:
            mu = self.fc_mu(pooled)
            logvar = self.fc_logvar(pooled).clamp(min=-10.0, max=2.0)
            if self.training:
                std = torch.exp(0.5 * logvar)
                z = mu + std * torch.randn_like(std)
            else:
                z = mu
            logits = self.classifier(z)
            if return_latent:
                return logits, mu, logvar, z
            return logits

        logits = self.classifier(pooled)
        if return_latent:
            return logits, None, None, pooled
        return logits


def build_model(**kwargs) -> InfoGCN:
    """Factory helper matching repository experiment conventions."""
    return InfoGCN(**kwargs)
