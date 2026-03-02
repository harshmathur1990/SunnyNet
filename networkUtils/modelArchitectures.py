import torch
import torch.nn as nn
import torch.nn.functional as F


class SunnyNet(nn.Module):
    """
    Generic SunnyNet

    Input  : [B, C, D, H, W]
    Output : [B, Cout, D, 1, 1]

    Works for arbitrary spatial window:
        1x1, 3x3, 5x5, 7x7, 11x11 ...
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        depth,
        window_size,
        hidden_fc=4700,
    ):
        super().__init__()

        self.out_channels = out_channels
        self.depth = depth
        self.window_size = window_size

        # --------------------------------------------------
        # Spatial mixing (adaptive)
        # --------------------------------------------------
        k = window_size
        pad = k // 2

        self.spatial_conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=32,
            kernel_size=(3, k, k),
            padding=(1, pad, pad),
        )

        # collapse spatial dimensions automatically
        self.spatial_pool = nn.AdaptiveAvgPool3d((depth, 1, 1))

        # --------------------------------------------------
        # Vertical RT coupling
        # --------------------------------------------------
        self.conv1 = nn.Conv1d(32, 32, 3)
        self.conv2 = nn.Conv1d(32, 64, 3)
        self.conv3 = nn.Conv1d(64, 128, 3)

        self.max1 = nn.MaxPool1d(2)
        self.max2 = nn.MaxPool1d(2)
        self.max3 = nn.MaxPool1d(2)

        # --------------------------------------------------
        # Lazy Linear removes hardcoded 6144 !!!
        # --------------------------------------------------
        self.fc1 = nn.LazyLinear(hidden_fc)
        self.fc2 = nn.Linear(hidden_fc, out_channels * depth)

        self.dropout = nn.Dropout(0.5)

    # --------------------------------------------------
    def forward(self, x):

        # ----- spatial mixing -----
        x = F.relu(self.spatial_conv(x))

        # H,W → 1,1 automatically
        x = self.spatial_pool(x)

        x = x.squeeze(-1).squeeze(-1)

        # ----- vertical coupling -----
        x = F.relu(self.conv1(x))
        x = self.max1(x)

        x = F.relu(self.conv2(x))
        x = self.max2(x)

        x = F.relu(self.conv3(x))
        x = self.max3(x)

        x = torch.flatten(x, 1)

        # ----- MLP head -----
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.fc2(x)

        return x.view(-1, self.out_channels, self.depth, 1, 1)
