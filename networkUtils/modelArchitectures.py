import torch
import torch.nn as nn
import torch.nn.functional as F


class SunnyNet(nn.Module):
    """
    Generic version of your ORIGINAL SunnyNet.

    Supports:
        1x1, 3x3, 5x5, 7x7, 11x11, ...

    Architecture unchanged.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        depth,
        window
    ):
        super().__init__()

        self.out_channels = out_channels
        self.depth = depth
        self.height = window
        self.width = window

        k = window   # assume square window
        pad = k // 2

        # --------------------------------------------------
        # FIRST LAYER (ONLY DIFFERENCE)
        # --------------------------------------------------
        if k == 1:
            self.first_is_1d = True

            self.conv1 = nn.Conv1d(
                in_channels,
                32,
                kernel_size=3,
                padding=1,
            )
        else:
            self.first_is_1d = False

            self.conv1 = nn.Conv3d(
                in_channels,
                32,
                kernel_size=(k, k, k),
                padding=(pad, 0, 0),
            )

        # --------------------------------------------------
        # REST IDENTICAL
        # --------------------------------------------------
        self.conv2 = nn.Conv1d(32, 32, 3)
        self.conv3 = nn.Conv1d(32, 64, 3)
        self.conv4 = nn.Conv1d(64, 128, 3)

        self.max1 = nn.MaxPool1d(2)
        self.max2 = nn.MaxPool1d(2)
        self.max3 = nn.MaxPool1d(2)

        self.fc1 = nn.Linear(6144, 4700)
        self.fc2 = nn.Linear(4700, out_channels * depth)

        self.dropout = nn.Dropout(0.5)

    # --------------------------------------------------
    def forward(self, x):

        # ---------- first layer ----------
        if self.first_is_1d:
            x = x.squeeze(3).squeeze(3)
            x = F.relu(self.conv1(x))
        else:
            x = F.relu(self.conv1(x))
            x = x.squeeze(3).squeeze(3)

        # ---------- identical pipeline ----------
        x = F.relu(self.conv2(x))
        x = self.max1(x)

        x = F.relu(self.conv3(x))
        x = self.max2(x)

        x = F.relu(self.conv4(x))
        x = self.max3(x)

        x = torch.flatten(x, 1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = self.fc2(x)

        return x.view(-1, self.out_channels, self.depth, 1, 1)


# --------------------------------------------------
# Utilities
# --------------------------------------------------
def _gn(channels, max_groups=8):
    g = min(max_groups, channels)
    while channels % g != 0 and g > 1:
        g -= 1
    return nn.GroupNorm(g, channels)


class Conv3dBlock(nn.Module):
    def __init__(self, cin, cout, k, s, p):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(cin, cout, kernel_size=k, stride=s, padding=p, bias=False),
            _gn(cout),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


class Conv1dBlock(nn.Module):
    def __init__(self, cin, cout, k=5, p=2):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(cin, cout, kernel_size=k, padding=p, bias=False),
            _gn(cout),
            nn.GELU(),
        )

    def forward(self, x):
        return self.block(x)


# --------------------------------------------------
# Main Model
# --------------------------------------------------
class ContextToColumn3D(nn.Module):
    """
    Input:
        [B, Cin, Depth, W, W]

    Output:
        [B, Cout, Depth, 1, 1]
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        depth,
        window,
        base_channels=32,
        use_center_residual=True,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.depth = depth
        self.window = window
        self.use_center_residual = use_center_residual

        # ----------------------------------
        # Stage 1 — feature extraction
        # ----------------------------------
        self.stem = nn.Sequential(
            Conv3dBlock(in_channels, base_channels,
                        k=(3,3,3), s=(1,1,1), p=(1,1,1)),
            Conv3dBlock(base_channels, base_channels*2,
                        k=(3,3,3), s=(1,1,1), p=(1,1,1)),
        )

        ch = base_channels * 2

        # ----------------------------------
        # Stage 2 — horizontal collapse
        # ----------------------------------
        collapse_layers = []
        w = window

        while w > 1:
            collapse_layers.append(
                Conv3dBlock(
                    ch,
                    min(ch*2, 256),
                    k=(3,3,3),
                    s=(1,2,2),   # collapse only x,y
                    p=(1,1,1),
                )
            )
            ch = min(ch*2, 256)
            w = (w + 1) // 2

        self.collapse = nn.Sequential(*collapse_layers)

        # ----------------------------------
        # Stage 3 — column physics network
        # ----------------------------------
        self.column_net = nn.Sequential(
            Conv1dBlock(ch, 256),
            Conv1dBlock(256, 256),
            nn.Conv1d(256, out_channels, kernel_size=1),
        )

        # ----------------------------------
        # Center-column residual branch
        # ----------------------------------
        if use_center_residual:
            self.center_net = nn.Sequential(
                Conv1dBlock(in_channels, 128),
                Conv1dBlock(128, 128),
                nn.Conv1d(128, out_channels, kernel_size=1),
            )

    # --------------------------------------------------
    def forward(self, x):

        if x.shape[2] != self.depth:
            raise ValueError(
                f"Depth mismatch: expected {self.depth}, got {x.shape[2]}"
            )

        if x.shape[-1] != self.window:
            raise ValueError(
                f"Window mismatch: expected {self.window}"
            )

        # ---- main path ----
        h = self.stem(x)
        h = self.collapse(h)

        # [B,C,D,1,1] → [B,C,D]
        h = h.squeeze(-1).squeeze(-1)

        y = self.column_net(h)

        # ---- residual physics shortcut ----
        if self.use_center_residual:
            c = self.window // 2
            center = x[:, :, :, c, c]
            y = y + self.center_net(center)

        return y.unsqueeze(-1).unsqueeze(-1)

