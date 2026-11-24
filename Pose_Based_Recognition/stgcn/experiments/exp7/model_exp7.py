import torch
import torch.nn as nn


class ChannelNorm2d(nn.Module):
    """Normalize channels independently at each frame and joint."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, partitions=3):
        super().__init__()
        self.projections = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for _ in range(partitions)
        ])

    def forward(self, x, adjacency):
        outputs = []
        for k, projection in enumerate(self.projections):
            projected = projection(x)
            outputs.append(torch.einsum(
                "nctv,vw->nctw", projected, adjacency[k]
            ))
        return sum(outputs)


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, adjacency, stride=1):
        super().__init__()

        self.register_buffer("adjacency", adjacency)
        self.stride = stride

        self.spatial = SpatialGraphConv(in_channels, out_channels)
        self.norm1 = ChannelNorm2d(out_channels)

        self.temporal = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(9, 1),
            stride=(stride, 1),
            padding=(4, 0)
        )
        self.norm2 = ChannelNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.residual = (
            nn.Identity()
            if in_channels == out_channels and stride == 1
            else nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=(stride, 1)
            )
        )

    def forward(self, x, mask):
        input_mask = mask.bool()
        output_mask = input_mask[:, ::self.stride]

        input_weights = input_mask[:, None, :, None].to(x.dtype)
        output_weights = output_mask[:, None, :, None].to(x.dtype)

        residual = self.residual(x)
        residual = residual * output_weights

        # Spatial graph convolution
        y = self.spatial(x, self.adjacency)
        y = self.relu(self.norm1(y))
        y = y * input_weights

        # Temporal convolution
        y = self.temporal(y)
        y = y * output_weights
        y = self.norm2(y)

        y = self.relu(y + residual)
        y = y * output_weights

        return y, output_mask


class STGCN(nn.Module):
    def __init__(self, adjacency, num_classes=41, in_channels=2):
        super().__init__()

        adjacency = torch.as_tensor(adjacency, dtype=torch.float32)

        # Preserve canonical XY magnitudes and relative body scale. Applying
        # LayerNorm to each (x,y) pair removes their per-joint coordinate
        # magnitude and makes inputs inconsistent with the saved convention.
        self.input_norm = nn.Identity()

        self.blocks = nn.ModuleList([
            STGCNBlock(in_channels, 64, adjacency),
            STGCNBlock(64, 64, adjacency),
            STGCNBlock(64, 128, adjacency, stride=2),
            STGCNBlock(128, 128, adjacency),
            STGCNBlock(128, 256, adjacency, stride=2),
            STGCNBlock(256, 256, adjacency),
        ])

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x, frame_mask):
        # Input: (N, C, T, V)
        N, C, T, V = x.shape

        mask = frame_mask.bool()
        weights = mask[:, None, :, None].to(x.dtype)

        x = x * weights

        # Canonical XY is already normalized; retain its values as supplied.

        for block in self.blocks:
            x, mask = block(x, mask)

        # Masked global pooling
        weights = mask[:, None, :, None].to(x.dtype)
        pooled = (x * weights).sum(dim=(2, 3))
        denominator = (weights.sum(dim=2) * V).squeeze(-1)
        pooled = pooled / denominator.clamp(min=1)

        return self.classifier(pooled)
