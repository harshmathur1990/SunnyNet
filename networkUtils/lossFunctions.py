import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class WeightedMSE(nn.Module):
    """
    Weighted MSE loss where weights depend only on the target magnitude.
    Designed for NLTE departure coefficients (log b).

    Loss = mean( (1 + |y_true|) * (y_pred - y_true)^2 )
    """
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        w = 1.0 + torch.abs(y_true)
        return (w * (y_pred - y_true) ** 2).mean()


class RelativeMSE(nn.Module):
    def __init__(self, eps=1e-3):
        super().__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        return (((y_pred - y_true) / (torch.abs(y_true) + self.eps))**2).mean()


lines = [(0,1), (0,2), (0, 3), (0, 4), (1,2), (1,3), (1, 4), (2,3), (2, 4), (3, 4)]

chi = [
    1.6339941854018686e-18,
    1.936585907218822e-18,
    2.0424878450955273e-18,
    2.091506177644877e-18,
    2.1802152677122893e-18
]

g = [
    2,
    8,
    18,
    32,
    50,
    1
]


class PhysicsLossAllLines(nn.Module):
    def __init__(
        self,
        lines,
        chi,     # list or array, Joules
        g,       # list or array
        lam_curv=5e-2,
        lam_bar=5e-3,
        eps=1e-2,
        kappa=50.0,
        kB=1.380649e-23,
    ):
        super().__init__()

        self.lines = lines                      # Python list (indices)
        self.lam_curv = lam_curv
        self.lam_bar = lam_bar
        self.eps = eps
        self.kappa = kappa
        self.kB = kB

        # Register atomic constants as buffers (not parameters)
        self.register_buffer("chi", torch.tensor(chi, dtype=torch.float32))
        self.register_buffer("g",   torch.tensor(g,   dtype=torch.float32))

    def forward(self, logb_pred, T):
        """
        logb_pred : (B, Nlevels, Nz)
        T         : (B, Nz)
        """
        L_curv = 0.0
        L_bar  = 0.0

        for (l, u) in self.lines:
            # q(z) = log(b_l / b_u)
            q = logb_pred[:, l, :] - logb_pred[:, u, :]   # (B, Nz)

            # ---- curvature penalty
            d2q = q[:, 2:] - 2.0 * q[:, 1:-1] + q[:, :-2]
            L_curv += (d2q ** 2).mean()

            # ---- LTE ratio from temperature
            dE = self.chi[u] - self.chi[l]   # scalar tensor
            log_nl_over_nu_LTE = (
                dE / (self.kB * T) + torch.log(self.g[l] / self.g[u])
            )

            logR = q + log_nl_over_nu_LTE
            R = torch.exp(logR)

            # ---- barrier near R → 1
            L_bar += (
                F.softplus(self.kappa * (self.eps - (R - 1.0))) / self.kappa
            ).mean()

        nlines = len(self.lines)
        L_curv /= nlines
        L_bar  /= nlines

        return self.lam_curv * L_curv + self.lam_bar * L_bar
