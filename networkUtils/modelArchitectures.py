import torch
import torch.nn as nn
import torch.nn.functional as F


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
