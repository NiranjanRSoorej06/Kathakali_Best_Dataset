import torch
import torch.nn as nn


class ChannelNorm2d(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2).contiguous()


class SingleGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.projection = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, adjacency):
        x = self.projection(x)
        return torch.einsum("nctv,vw->nctw", x, adjacency)


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.stride = stride
        self.spatial = SingleGraphConv(in_channels, out_channels)
        self.norm1 = ChannelNorm2d(out_channels)

        self.temporal = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=(9, 1),
            stride=(stride, 1),
            padding=(4, 0),
        )

        self.norm2 = ChannelNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=1,
                stride=(stride, 1),
            )

    def forward(self, x, adjacency, mask):
        input_weights = mask[:, None, :, None].to(x.dtype)
        output_mask = mask[:, ::self.stride]
        output_weights = output_mask[:, None, :, None].to(x.dtype)

        residual = self.residual(x) * output_weights

        y = self.spatial(x, adjacency)
        y = self.relu(self.norm1(y))
        y = y * input_weights

        y = self.temporal(y)
        y = self.norm2(y)
        y = self.relu(y + residual)
        y = y * output_weights

        return y, output_mask


class STGCN(nn.Module):
    def __init__(self, adjacency, num_classes=41, in_channels=3):
        super().__init__()

        adjacency = torch.as_tensor(adjacency, dtype=torch.float32)
        if adjacency.shape != (50, 50):
            raise ValueError(
                f"Expected adjacency (50, 50), got {adjacency.shape}"
            )

        if in_channels != 3:
            raise ValueError("Experiment 9 requires 3 input channels")

        self.register_buffer("A", adjacency)

        self.blocks = nn.ModuleList([
            STGCNBlock(3, 32),
            STGCNBlock(32, 32),
            STGCNBlock(32, 64, stride=2),
            STGCNBlock(64, 64),
            STGCNBlock(64, 128, stride=2),
        ])

        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes),
        )

    def forward(self, x, frame_mask):
        # Input: (N, 3, 38, 50)
        if x.ndim != 4 or x.shape[1:] != (3, 38, 50):
            raise ValueError(f"Unexpected input shape: {tuple(x.shape)}")

        if frame_mask.shape != (x.shape[0], 38):
            raise ValueError(
                f"Unexpected frame mask shape: {tuple(frame_mask.shape)}"
            )

        mask = frame_mask.bool()
        x = x * mask[:, None, :, None].to(x.dtype)

        for block in self.blocks:
            x, mask = block(x, self.A, mask)

        weights = mask[:, None, :, None].to(x.dtype)
        pooled = (x * weights).sum(dim=(2, 3))
        denominator = (weights.sum(dim=2) * x.shape[3]).squeeze(-1)
        pooled = pooled / denominator.clamp(min=1)

        return self.classifier(pooled)
