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


def _nan_stats(x, name):
    if not torch.is_tensor(x):
        return
    n_nan = torch.isnan(x).sum().item()
    if n_nan > 0:
        total = x.numel()
        print(
            f"[NaN DEBUG] {name}: "
            f"{n_nan}/{total} NaNs "
            f"(min={x.nanmin().item():.3e}, "
            f"max={x.nanmax().item():.3e})"
        )


def _range_stats(x, name):
    if not torch.is_tensor(x):
        return
    n_nan = torch.isnan(x).sum().item()
    n_inf = torch.isinf(x).sum().item()
    print(
        f"[RANGE] {name}: "
        f"min={x.nanmin().item():.3e}, "
        f"max={x.nanmax().item():.3e}, "
        f"NaN={n_nan}, Inf={n_inf}"
    )


def compute_Sv_all_lines_T_batched(
    T,
    logb,
    chi,
    lines,
    nu,
    eps=1e-12,
    debug=False
):
    device = T.device
    dtype  = T.dtype

    if debug:
        _nan_stats(T, "T (input)")
        _nan_stats(logb, "logb (input)")

    # line indices
    lines = torch.tensor(lines, device=device)
    l = lines[:, 0]
    u = lines[:, 1]

    logb_ratio = logb[:, l, :] - logb[:, u, :]
    if debug:
        _nan_stats(logb_ratio, "logb_ratio")

    dchi = (chi[u] - chi[l]).to(dtype)
    boltz = dchi[None, :, None] / (kB * T[:, None, :])
    if debug:
        _nan_stats(boltz, "boltz")

    x = logb_ratio + boltz
    if debug:
        _nan_stats(x, "x = logb_ratio + boltz")

    denom = torch.expm1(x)
    denom = torch.where(
        denom.abs() < 1e-6,
        denom.sign() * 1e-6,
        denom
    )
    if debug:
        _nan_stats(denom, "denom = expm1(x)")

    prefactor = (2 * h * nu**3) / c**2
    S = prefactor[None, :, None] / (denom + eps)
    if debug:
        _nan_stats(S, "S_nu")

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


def _clamp_physics_inputs(T, logb):
    """
    Gentle, physically motivated clamps to avoid NaNs.
    """

    # Temperature: never allow zero / negative
    T = torch.clamp(T, min=1.0)   # 1 K floor (safe, non-invasive)

    # log(b): avoid extreme values early in training
    logb = torch.clamp(logb, min=-20.0, max=20.0)

    return T, logb


def _validate_physics_inputs(T, logb, stage="input", strict=False):
    """
    Physics-aware sanity checks.

    strict=False  → warn + clamp
    strict=True   → raise RuntimeError
    """

    msgs = []

    # Temperature must be finite and positive
    if not torch.isfinite(T).all():
        msgs.append("T contains NaN or Inf")

    if (T <= 0).any():
        msgs.append(f"T <= 0 detected (min T = {T.min().item():.3e})")

    # logb must be finite (log departure coefficients)
    if not torch.isfinite(logb).all():
        msgs.append("logb contains NaN or Inf")

    if len(msgs) > 0:
        msg = f"[PHYSICS INPUT ERROR @ {stage}] " + " | ".join(msgs)
        if strict:
            raise RuntimeError(msg)
        else:
            print(msg)


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
        return_components=True,
        debug=False
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

        self.debug = debug
        self._debug_triggered = False  # print once

    def forward(
        self,
        T,
        logb_pred,
        logb_true
    ):

        # ------------------ INPUT VALIDATION ------------------ #
        if self.debug and not self._debug_triggered:
            _validate_physics_inputs(T, logb_pred, stage="forward (pre-clamp)")

        # Optional: soft clamps (recommended)
        T, logb_pred = _clamp_physics_inputs(T, logb_pred)

        if self.debug and not self._debug_triggered:
            _validate_physics_inputs(T, logb_pred, stage="forward (post-clamp)")

        # ------------------ DATA LOSS ------------------ #
        L_data = self.data_loss(logb_pred, logb_true)

        if self.debug and not self._debug_triggered:
            _nan_stats(L_data, "L_data")

        # ------------------ PHYSICS ------------------ #
        S = compute_Sv_all_lines_T_batched(
            T=T,
            logb=logb_pred,
            chi=self.chi,
            lines=self.lines,
            nu=self.nu,
            debug=self.debug and not self._debug_triggered
        )

        # S must be positive for log10
        # ---------- PHYSICAL SANITY CHECK ----------
        if self.debug and not self._debug_triggered:
            if (S <= 0).any():
                print("[PHYSICS ERROR] S_nu <= 0 detected")

                _range_stats(T, "T")
                _range_stats(logb_pred, "logb_pred")
                _range_stats(logb_true, "logb_true")
                _range_stats(S, "S_nu")

                # Optional: identify worst offender
                idx = torch.argmin(S)
                print(f"[PHYSICS ERROR] Most negative S_nu value = {S.flatten()[idx].item():.3e}")

                self._debug_triggered = True

        logS = torch.log10(torch.clamp(S, min=1e-30))

        if self.debug and not self._debug_triggered:
            _nan_stats(logS, "log10(S)")

        # ------------------ REGULARIZATION ------------------ #
        L_reg = zigzag_regularizer_Sv_batched(
            logS,
            lam=self.lam,
            min_stride=self.min_stride,
            max_frac=self.max_frac,
            delta=self.delta
        )

        if self.debug and not self._debug_triggered:
            _nan_stats(L_reg, "L_reg")

        # ------------------ TOTAL ------------------ #
        L_total = L_data + L_reg

        if self.debug and not self._debug_triggered:
            if torch.isnan(L_total):
                print("[NaN DEBUG] NaN detected in L_total — disabling debug")
                self._debug_triggered = True

        if self.return_components:
            return L_total, {
                "data": L_data.detach(),
                "regularization": L_reg.detach(),
                "lambda": self.lam
            }
        else:
            return L_total
