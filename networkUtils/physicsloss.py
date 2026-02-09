import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


c_AHz = 2.99792458e18  # Hz * Å
h  = 6.62607015e-34
c  = 2.99792458e8
kB = 1.380649e-23


lines = [(0,1), (0,2), (0, 3), (0, 4), (1,2), (1,3), (1, 4), (2,3), (2, 4), (3, 4)]


wave = np.array([1215.6701, 1025.7220, 972.53650, 949.74287, 6562.79, 4861.35, 4340.472, 18750, 12820, 40510])


chi = [
    1.6339941854018686e-18,
    1.936585907218822e-18,
    2.0424878450955273e-18,
    2.091506177644877e-18,
    2.1802152677122893e-18
]


def compute_Sv_all_lines_logT_batched(
    T,       # (B, Nz)
    logb,       # (B, Nlevels, Nz)
    chi,        # (Nlevels,)     [J]
    lines,      # list of (l, u)
    nu,         # (Nlines,)      [Hz]
    eps=1e-12
):
    """
    Returns:
        S : (B, Nlines, Nz)
    """

    device = logT.device
    dtype  = logT.dtype

    # line indices
    lines = torch.tensor(lines, device=device)
    l = lines[:, 0]                              # (Nlines,)
    u = lines[:, 1]                              # (Nlines,)

    # log(b_l / b_u)
    logb_ratio = logb[:, l, :] - logb[:, u, :]   # (B, Nlines, Nz)

    # Boltzmann factor Δχ / (kT)
    dchi = (chi[u] - chi[l]).to(dtype)           # (Nlines,)
    boltz = dchi[None, :, None] / (kB * T[:, None, :])
                                                   # (B, Nlines, Nz)

    x = logb_ratio + boltz

    # exp(x) - 1 (stable)
    denom = torch.expm1(x)
    denom = torch.where(
        denom.abs() < 1e-6,
        denom.sign() * 1e-6,
        denom
    )

    prefactor = (2 * h * nu**3) / c**2            # (Nlines,)
    S = prefactor[None, :, None] / (denom + eps) # (B, Nlines, Nz)

    return S


def zigzag_regularizer_multiscale(
    f,                  # (..., Nz)
    min_stride=2,
    max_frac=0.25,
    delta=1e-2
):
    """
    Fully vectorized multi-scale curvature regularizer.

    Penalizes zig-zag patterns up to max_frac * Nz.
    Operates on the last dimension (z).
    """

    Nz = f.shape[-1]
    max_stride = max(min_stride, int(max_frac * Nz))

    # build logarithmic stride list: 2, 4, 8, ...
    strides = []
    s = min_stride
    while s <= max_stride and 2 * s < Nz:
        strides.append(s)
        s *= 2

    if len(strides) == 0:
        return torch.zeros((), device=f.device, dtype=f.dtype)

    total = 0.0
    wsum  = 0.0

    for s in strides:
        # centered second difference with stride s
        mid   = f[..., s:-s]
        left  = f[..., :-2*s]
        right = f[..., 2*s:]

        d2 = right - 2.0 * mid + left

        abs_d2 = d2.abs()
        loss_s = torch.where(
            abs_d2 < delta,
            0.5 * d2**2,
            delta * (abs_d2 - 0.5 * delta)
        ).mean()

        w = 1.0 / (s * s)
        total += w * loss_s
        wsum  += w

    return total / (wsum + 1e-12)


def zigzag_regularizer_Sv_batched(
    S,          # (B, Nlines, Nz)
    lam=1e-3,
    min_stride=2,
    max_frac=0.25,
    delta=1e-2
):
    """
    Global zig-zag regularizer for batched source functions.
    """

    core = zigzag_regularizer_multiscale(
        S,
        min_stride=min_stride,
        max_frac=max_frac,
        delta=delta
    )

    return lam * core


class NLTECompositeLoss(nn.Module):
    """
    Total loss = WeightedMSE(log b) + lambda * zig-zag regularization on S_nu
    """

    def __init__(
        self,
        chi,
        lines,
        wave_angstrom,
        data_loss_func,
        lam=1e-3,
        min_stride=2,
        max_frac=0.25,
        delta=1e-2,
        return_components=True
    ):
        super().__init__()

        self.data_loss = data_loss_func

        self.register_buffer("chi", torch.tensor(chi))
        self.register_buffer(
            "nu",
            torch.tensor(c_AHz / np.array(wave_angstrom))
        )

        self.lines = lines
        self.lam = lam
        self.min_stride = min_stride
        self.max_frac = max_frac
        self.delta = delta
        self.return_components=return_components

    def forward(
        self,
        T,
        logb_pred,
        logb_true
    ):
        # --- data loss ---
        L_data = self.data_loss(logb_pred, logb_true)

        # --- compute S_nu ---
        S = compute_Sv_all_lines_logT_batched(
            T=T,
            logb=logb_pred,
            chi=self.chi,
            lines=self.lines,
            nu=self.nu
        )

        # --- regularization ---
        L_reg = zigzag_regularizer_Sv_batched(
            torch.log10(S),
            lam=self.lam,
            min_stride=self.min_stride,
            max_frac=self.max_frac,
            delta=self.delta
        )

        L_total = L_data + L_reg

        if self.return_components:
            return L_total, {
                "data": L_data.detach(),
                "regularization": L_reg.detach(),
                "lambda": self.lam
            }
        else:
            return L_total
