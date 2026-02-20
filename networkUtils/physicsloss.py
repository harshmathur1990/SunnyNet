import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


c_AHz = np.float32(2.99792458e18)  # Hz * Å
h  = np.float32(6.62607015e-34)
c  = np.float32(2.99792458e8)
kB = np.float32(1.380649e-23)


lines = np.array(
    [(0,1), (0,2), (0, 3), (0, 4), (1,2), (1,3), (1, 4), (2,3), (2, 4), (3, 4)],
    dtype=np.float32
)


wave = np.array([1215.6701, 1025.7220, 972.53650, 949.74287, 6562.79, 4861.35, 4340.472, 18750, 12820, 40510], dtype=np.float32)


chi = np.array(
    [
        1.6339941854018686e-18,
        1.936585907218822e-18,
        2.0424878450955273e-18,
        2.091506177644877e-18,
        2.1802152677122893e-18
    ],
    dtype=np.float32
)


def _nan_stats(x, name):
    if not torch.is_tensor(x):
        return
    n_nan = torch.isnan(x).sum().item()
    if n_nan > 0:
        finite = torch.isfinite(x)
        if finite.any():
            xmin = x[finite].min().item()
            xmax = x[finite].max().item()
        else:
            xmin, xmax = float("nan"), float("nan")

        print(
            f"[NaN DEBUG] {name}: "
            f"{n_nan}/{x.numel()} NaNs "
            f"(min={xmin:.3e}, max={xmax:.3e})"
        )


def _range_stats(x, name):
    if not torch.is_tensor(x):
        return

    n_nan = torch.isnan(x).sum().item()
    n_inf = torch.isinf(x).sum().item()

    # mask invalid values for min/max
    finite = torch.isfinite(x)
    if finite.any():
        xmin = x[finite].min().item()
        xmax = x[finite].max().item()
    else:
        xmin = float("nan")
        xmax = float("nan")

    print(
        f"[RANGE] {name}: "
        f"min={xmin:.3e}, "
        f"max={xmax:.3e}, "
        f"NaN={n_nan}, Inf={n_inf}"
    )


def _finite_minmax(x):
    finite = torch.isfinite(x)
    if finite.any():
        return x[finite].min().item(), x[finite].max().item()
    return float("nan"), float("nan")

def _check_tensor(x, name, debug):
    """Cheap summary check; prints only if debug=True."""
    if not debug or not torch.is_tensor(x):
        return
    n_nan = torch.isnan(x).sum().item()
    n_inf = torch.isinf(x).sum().item()
    xmin, xmax = _finite_minmax(x)
    print(f"[CHK] {name:18s} shape={tuple(x.shape)}  min={xmin:.3e} max={xmax:.3e}  NaN={n_nan} Inf={n_inf}")


def _report_S_negative(
    *,
    S, T, logb, l, u, dchi, boltz, logb_ratio, x, expm1_x, denom_raw, denom_fixed, prefactor, eps, nu,
    debug=True,
    pick="most_negative"  # or "first"
):
    """
    Prints a single, exact location and the full causal chain.
    Returns True if it printed.
    """
    if not debug:
        return False

    bad = (S <= 0) | (~torch.isfinite(S))
    if not bad.any():
        return False

    # choose an index
    if pick == "first":
        flat = torch.nonzero(bad.flatten(), as_tuple=False)[0, 0]
    else:  # most_negative among bad if possible
        S_bad = S.clone()
        S_bad[~bad] = float("inf")
        flat = torch.argmin(S_bad.flatten())

    B, NL, Nz = S.shape
    flat = flat.item()
    b = flat // (NL * Nz)
    rem = flat % (NL * Nz)
    li = rem // Nz
    z  = rem % Nz

    # gather scalars
    T_bz       = T[b, z].item()
    l_idx      = l[li].item()
    u_idx      = u[li].item()
    nu_li      = nu[li].item()
    pref_li    = prefactor[li].item()
    dchi_li    = dchi[li].item()

    logb_l     = logb[b, l_idx, z].item()
    logb_u     = logb[b, u_idx, z].item()
    lr_bli_z   = logb_ratio[b, li, z].item()
    bol_bli_z  = boltz[b, li, z].item()
    x_bli_z    = x[b, li, z].item()
    ex_bli_z   = expm1_x[b, li, z].item()
    dr_bli_z   = denom_raw[b, li, z].item()
    df_bli_z   = denom_fixed[b, li, z].item()
    S_bli_z    = S[b, li, z].item()

    # classify exactly where sign became negative
    # (this is not "guessing": it’s a strict logical trace)
    reasons = []
    if not np.isfinite(T_bz) or T_bz <= 0:
        reasons.append("T is non-finite or <= 0 (boltz invalid)")
    if not np.isfinite(logb_l) or not np.isfinite(logb_u):
        reasons.append("logb has non-finite values (logb_ratio invalid)")
    if not np.isfinite(x_bli_z):
        reasons.append("x became non-finite (from logb_ratio or boltz)")
    if np.isfinite(x_bli_z) and x_bli_z < 0:
        reasons.append("x < 0 => expm1(x) < 0 => denom negative => S negative")
    if np.isfinite(df_bli_z) and df_bli_z < 0:
        reasons.append("denom after stabilization is negative (sign preserved) => S negative")
    if np.isfinite(pref_li) and pref_li <= 0:
        reasons.append("prefactor <= 0 (should not happen)")
    if np.isfinite(df_bli_z) and np.isfinite(pref_li) and (pref_li / (df_bli_z + eps)) <= 0:
        reasons.append("final division yields <=0 (sign from denom)")

    print("\n========== [S NEGATIVE ROOT-CAUSE REPORT] ==========")
    print(f"Location: batch={b}, line={li}, z={z}")
    print(f"Levels: l={l_idx}, u={u_idx}")
    print(f"T[b,z] = {T_bz:.6e} K")
    print(f"nu[line] = {nu_li:.6e} Hz")
    print(f"prefactor[line] = {pref_li:.6e}")
    print(f"dchi[line] = {dchi_li:.6e} J")
    print("--- logb at location ---")
    print(f"logb_l = {logb_l:.6e}")
    print(f"logb_u = {logb_u:.6e}")
    print("--- intermediates at location ---")
    print(f"logb_ratio = {lr_bli_z:.6e}")
    print(f"boltz      = {bol_bli_z:.6e}")
    print(f"x          = {x_bli_z:.6e}")
    print(f"expm1(x)   = {ex_bli_z:.6e}")
    print("--- denom path ---")
    print(f"denom_raw   = {dr_bli_z:.6e}")
    print(f"denom_fixed = {df_bli_z:.6e}   (after abs<1e-6 stabilization)")
    print(f"eps         = {eps:.3e}")
    print("--- output ---")
    print(f"S[b,line,z] = {S_bli_z:.6e}")
    print("--- classification ---")
    if reasons:
        for r in reasons:
            print(f"* {r}")
    else:
        print("* Could not classify (unexpected); all upstream values look finite")
    print("====================================================\n")

    return True


def compute_Sv_all_lines_T_batched(
    T,
    logb,
    chi,
    lines,
    nu,
    K_prefactor,
    eps=1e-12,
    denom_floor=1e-6,
    return_x=False,
    debug=False,
    debug_report="most_negative"
):
    """
    Returns:
        S : (B, Nlines, Nz)

    If return_x=True:
        returns (S, x)

    In debug mode:
        prints intermediate diagnostics.
    """

    device = T.device
    dtype  = T.dtype

    if debug:
        print(f"[CHK] T dtype = {T.dtype}")

    _check_tensor(T,    "T (input)",    debug)
    _check_tensor(logb, "logb (input)", debug)

    # ensure lines tensor on device
    if not torch.is_tensor(lines):
        lines = torch.tensor(lines, device=device)
    else:
        lines = lines.to(device)

    l = lines[:, 0]
    u = lines[:, 1]

    # log(b_l / b_u)
    logb_ratio = logb[:, l, :] - logb[:, u, :]
    _check_tensor(logb_ratio, "logb_ratio", debug)

    # Δχ/(kT)
    dchi = (chi[u] - chi[l]).to(device=device, dtype=dtype)
    _check_tensor(dchi, "dchi", debug)

    boltz = dchi[None, :, None] / (kB * T[:, None, :])
    _check_tensor(boltz, "boltz", debug)

    # exponent argument (THIS is the physically important variable)
    x = logb_ratio + boltz
    _check_tensor(x, "x", debug)

    # stable exp(x) - 1
    expm1_x = torch.expm1(x)
    _check_tensor(expm1_x, "expm1(x)", debug)

    # smooth floor instead of hard clamp
    denom = torch.sign(expm1_x) * torch.sqrt(expm1_x**2 + denom_floor**2)
    _check_tensor(denom, "denom (smoothed)", debug)

    _check_tensor(K_prefactor, "prefactor", debug)

    S = K_prefactor[None, :, None] / (denom + eps)
    _check_tensor(S, "S", debug)

    # debug deep inspection
    if debug and ((S <= 0).any() or (~torch.isfinite(S)).any()):
        _report_S_negative(
            S=S, T=T, logb=logb,
            l=l, u=u,
            dchi=dchi,
            boltz=boltz,
            logb_ratio=logb_ratio,
            x=x,
            expm1_x=expm1_x,
            denom_raw=expm1_x,
            denom_fixed=denom,
            prefactor=K_prefactor,
            eps=eps,
            nu=nu,
            debug=True,
            pick=debug_report
        )

    if return_x:
        return S, x
    else:
        return S



def zigzag_regularizer_multiscale(
    f,                  # (..., Nz)
    min_stride=1,
    max_frac=0.25,
    delta=1e-2,
    eps=1e-12,
    normalize=False
):
    """
    Multi-scale curvature regularizer with *equidistant* strides.

    Penalizes zig-zag patterns at all scales from 1 to max_frac * Nz.
    Operates on the last dimension (z).
    """

    if normalize:
        f0 = f - f.mean(dim=-1, keepdim=True)
        f0 = f0 / (f0.std(dim=-1, keepdim=True, correction=0) + eps)
    else:
        f0 = f

    Nz = f.shape[-1]
    max_stride = max(min_stride, int(max_frac * Nz))

    # equidistant strides: 1, 2, 3, ..., max_stride
    strides = range(min_stride, max_stride + 1)

    total = 0.0
    wsum  = 0.0

    for s in strides:
        if 2 * s >= Nz:
            break

        mid   = f0[..., s:-s]
        left  = f0[..., :-2*s]
        right = f0[..., 2*s:]

        d2 = right - 2.0 * mid + left

        abs_d2 = d2.abs()
        loss_s = torch.where(
            abs_d2 < delta,
            0.5 * d2**2,
            delta * (abs_d2 - 0.5 * delta)
        ).mean()

        # optional scale compensation (important)
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
        lam=1e-1,
        lam_S=1e-2,
        min_stride=2,
        max_frac=0.25,
        delta=1e-1,
        return_components=True,
        debug=False
    ):
        super().__init__()

        self.data_loss = data_loss_func

        self.register_buffer(
            "chi",
            torch.tensor(chi, dtype=torch.float32)
        )

        self.register_buffer(
            "lines",
            torch.tensor(lines, dtype=torch.long)
        )

        nu = c_AHz / np.array(wave_angstrom)

        self.register_buffer(
            "nu",
            torch.tensor(nu, dtype=torch.float32)
        )

        self.register_buffer(
            "K_prefactor",
            torch.tensor((2.0 * h * nu/ (c**2)) * nu * nu, dtype=torch.float32)
        )

        self.lam = lam
        self.lam_S = lam_S
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
        # T, logb_pred = _clamp_physics_inputs(T, logb_pred)

        if self.debug and not self._debug_triggered:
            _validate_physics_inputs(T, logb_pred, stage="forward (post-clamp)")

        # ------------------ DATA LOSS ------------------ #
        L_data = self.data_loss(logb_pred, logb_true)

        if self.debug and not self._debug_triggered:
            _nan_stats(L_data, "L_data")

        # ------------------ PHYSICS ------------------ #
        S_pred, x_pred = compute_Sv_all_lines_T_batched(
            T=T,
            logb=logb_pred,
            chi=self.chi,
            lines=self.lines,
            nu=self.nu,
            K_prefactor=self.K_prefactor,
            debug=self.debug and not self._debug_triggered,
            debug_report="most_negative",   # or "first"
            return_x=True
        )

        with torch.no_grad():
            S_true, _ = compute_Sv_all_lines_T_batched(
                T=T,
                logb=logb_true,
                chi=self.chi,
                lines=self.lines,
                nu=self.nu,
                K_prefactor=self.K_prefactor,
                debug=False,
                return_x=True
            )

        logS_pred = torch.log10(torch.clamp(S_pred, min=1e-30))
        logS_true = torch.log10(torch.clamp(S_true, min=1e-30))

        data_loss_S = self.data_loss(logS_pred, logS_true)

        L_S = self.lam_S * data_loss_S

        # S must be positive for log10
        # ---------- PHYSICAL SANITY CHECK ----------
        if self.debug and not self._debug_triggered:
            if (S <= 0).any() or (~torch.isfinite(S)).any():
                print("[PHYSICS ERROR] S_nu <= 0 detected")

                _range_stats(T, "T")
                _range_stats(logb_pred, "logb_pred")
                _range_stats(logb_true, "logb_true")
                _range_stats(S, "S_nu")
                _range_stats(x, 'x')

                # Optional: identify worst offender
                idx = torch.argmin(S)
                print(f"[PHYSICS ERROR] Most negative S_nu value = {S.flatten()[idx].item():.3e}")

                self._debug_triggered = True

        # logS = torch.log10(torch.clamp(S, min=1e-30))

        # if self.debug and not self._debug_triggered:
            # _nan_stats(logS, "log10(S)")

        # x_fluct = x_pred - x_pred.mean(dim=-1, keepdim=True)

        # # ------------------ REGULARIZATION ------------------ #
        # L_reg = zigzag_regularizer_Sv_batched(
        #     x_fluct,
        #     lam=self.lam,
        #     min_stride=self.min_stride,
        #     max_frac=self.max_frac,
        #     delta=self.delta
        # )

        # if self.debug and not self._debug_triggered:
        #     _nan_stats(L_reg, "L_reg")

        # ------------------ TOTAL ------------------ #
        L_total = L_data + L_S  # + L_reg

        if self.debug and not self._debug_triggered:
            if torch.isnan(L_total):
                print("[NaN DEBUG] NaN detected in L_total — disabling debug")
                self._debug_triggered = True

        if self.return_components:
            return L_total, {
                "data": L_data.detach(),
                "source": L_S.detach(),
                # "regularization": L_reg.detach(),
                # "lambda_reg": self.lam,
                # "lambda_source": self.lam_S
            }
        else:
            return L_total
