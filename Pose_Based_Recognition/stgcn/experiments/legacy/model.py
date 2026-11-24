import torch
import torch.nn as nn
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "shared" / "graph"))
from graph import NUM_JOINTS, build_adjacency


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout=0.1):
        super().__init__()

        self.spatial_convs = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
            for _ in range(3)
        ])

        self.temporal_conv = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels, out_channels,
                kernel_size=(9, 1),
                stride=(stride, 1),
                padding=(4, 0),
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout)
        )

        if in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1,
                    stride=(stride, 1),
                    bias=False
                ),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        residual = self.residual(x)

        spatial_outputs = []

        for k, conv in enumerate(self.spatial_convs):
            projected = conv(x)
            spatial_outputs.append(
                torch.einsum("nctv,vw->nctw", projected, A[k])
            )

        x = sum(spatial_outputs)
        x = self.temporal_conv(x)

        return self.relu(x + residual)


class STGCN(nn.Module):
    def __init__(self, num_classes=41, in_channels=2):
        super().__init__()

        adjacency = torch.tensor(build_adjacency(), dtype=torch.float32)
        self.register_buffer("A", adjacency)

        self.data_bn = nn.BatchNorm1d(in_channels * NUM_JOINTS)

        self.blocks = nn.ModuleList([
            STGCNBlock(in_channels, 32),
            STGCNBlock(32, 32),
            STGCNBlock(32, 64, stride=2),
            STGCNBlock(64, 64),
            STGCNBlock(64, 128, stride=2),
        ])

        self.classifier = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = x.squeeze(-1)
        n, c, t, v = x.shape

        x = x.permute(0, 3, 1, 2).contiguous()
        x = x.view(n, v * c, t)
        x = self.data_bn(x)
        x = x.view(n, v, c, t).permute(0, 2, 3, 1).contiguous()

        for block in self.blocks:
            x = block(x, self.A)

        x = x.mean(dim=(2, 3))
        return self.classifier(x)