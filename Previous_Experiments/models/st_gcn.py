import numpy as np
import torch
import torch.nn as nn


class SpatialGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, num_partitions=3):
        super().__init__()
        self.num_partitions = num_partitions

        self.conv = nn.Conv2d(
            in_channels,
            out_channels * num_partitions,
            kernel_size=1
        )

    def forward(self, x, A):
        # x: (N, C, T, V)
        n, _, t, v = x.shape

        x = self.conv(x)
        x = x.view(
            n, self.num_partitions, -1, t, v
        )

        # Aggregate neighbouring joints
        x = torch.einsum("nkctv,kvw->nctw", x, A)
        return x.contiguous()


class STGCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, A,
                 stride=1, residual=True, dropout=0.0):
        super().__init__()

        self.register_buffer("A", A.clone())

        self.gcn = SpatialGraphConv(in_channels, out_channels)

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=(9, 1),
                stride=(stride, 1),
                padding=(4, 0)
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout)
        )

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels and stride == 1:
            self.residual = nn.Identity()
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels,
                    kernel_size=1, stride=(stride, 1)
                ),
                nn.BatchNorm2d(out_channels)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        res = self.residual(x)
        x = self.gcn(x, self.A)
        x = self.tcn(x)
        return self.relu(x + res)


class STGCN(nn.Module):
    def __init__(self, num_classes=10, in_channels=3,
                 graph_path="graph/stgcn_adjacency.npy",
                 dropout=0.3):
        super().__init__()

        A = torch.tensor(
            np.load(graph_path),
            dtype=torch.float32
        )

        assert A.shape == (3, 50, 50), A.shape

        self.register_buffer("A", A)

        self.data_bn = nn.BatchNorm2d(in_channels)

        self.blocks = nn.ModuleList([
            STGCNBlock(in_channels, 64, A, residual=False,
                       dropout=dropout),
            STGCNBlock(64, 64, A, dropout=dropout),
            STGCNBlock(64, 128, A, stride=2, dropout=dropout),
            STGCNBlock(128, 256, A, stride=2, dropout=dropout)
        ])

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        # Input: (N, C, T, V)
        x = self.data_bn(x)

        for block in self.blocks:
            x = block(x)

        # Global average pooling over time and joints
        x = x.mean(dim=(2, 3))

        return self.classifier(x)
