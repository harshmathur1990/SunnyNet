# ============================================================
# SunnyNet Utilities
# Public API CONTRACTS (DO NOT CHANGE SIGNATURES):
#
#   build_training_set
#   build_solving_set
#   sunnynet_train_model
#   sunnynet_predict_populations
#
# ============================================================

import os
import sys
import numpy as np
import h5py
import torch

from torch.utils.data import DataLoader
from scipy.interpolate import interp1d
from scipy.integrate import cumulative_trapezoid as cumtrapz

from networkUtils.atmosphereFunctions import predict_populations
from networkUtils.modelWrapper import Model
from networkUtils.dataSets import PopulationDataset3d
from networkUtils.trainingFunctions import train


# ============================================================
# ---------------- INTERPOLATION ------------------------------
# ============================================================

def interpolate_everything(rho_xyz, z_scale, values_xyzq, new_cmass):
    """
    Vectorized interpolation onto common column-mass grid.
    """

    def interpolate_column(rho, z, values, new_scale):
        cmass = cumtrapz(rho, -z, initial=0)
        f = interp1d(
            cmass,
            values,
            axis=0,
            kind="linear",
            fill_value="extrapolate",
        )
        return f(new_scale)

    vec = np.vectorize(
        interpolate_column,
        signature="(n),(n),(n,o),(m)->(m,o)",
    )

    return vec(rho_xyz, z_scale, values_xyzq, new_cmass)


# ============================================================
# ---------------- PREPROCESSING ------------------------------
# ============================================================

def _prepare_input_features(temp, vx, vy, vz, ne):

    return np.stack(
        [
            np.log10(temp),
            vx / 100,
            vy / 100,
            vz / 100,
            np.log10(ne),
        ],
        axis=-1,
    )


def _compute_departure_coefficients(lte, nlte):
    return np.log10(nlte / lte)


def _make_inputs_ch_first(
    rho, z_scale,
    temp, vx, vy, vz, ne,
    *, ndep, pad
):

    cmass_grid = np.logspace(-6, 2, ndep)
    pad_cfg = ((pad, pad), (pad, pad), (0, 0), (0, 0))

    features = _prepare_input_features(temp, vx, vy, vz, ne)

    features = interpolate_everything(
        rho, z_scale, features, cmass_grid
    )

    features = np.pad(features, pad_cfg, mode="wrap")

    features = np.transpose(features, (3, 2, 0, 1))

    return features, cmass_grid


def _make_targets_ch_first(
    rho, z_scale,
    lte, nlte,
    *, ndep, pad
):

    cmass_grid = np.logspace(-6, 2, ndep)
    pad_cfg = ((pad, pad), (pad, pad), (0, 0), (0, 0))

    dep = _compute_departure_coefficients(lte, nlte)

    dep = interpolate_everything(
        rho, z_scale, dep, cmass_grid
    )

    dep = np.pad(dep, pad_cfg, mode="wrap")
    dep = np.transpose(dep, (3, 2, 0, 1))

    return dep


# ============================================================
# ---------------- WINDOW EXTRACTION --------------------------
# ============================================================

def _extract_prediction_windows(inputs, out, pad):

    _, _, nxp, nyp = inputs.shape

    for i in range(pad, nxp - pad):
        for j in range(pad, nyp - pad):
            out.append(
                inputs[:, :, i-pad:i+pad+1,
                             j-pad:j+pad+1]
            )


def _extract_training_windows(inputs, targets, win_out, tgt_out, pad):

    _, _, nxp, nyp = inputs.shape

    for i in range(pad, nxp - pad):
        for j in range(pad, nyp - pad):

            win_out.append(
                inputs[:, :, i-pad:i+pad+1,
                             j-pad:j+pad+1]
            )

            tgt_out.append(
                targets[:, :, i, j][:, :, None, None]
            )


# ============================================================
# ---------------- TRAIN / TEST SPLIT -------------------------
# ============================================================

def _train_test_split_global(windows, targets, tr_percent):

    n = len(windows)
    idx = np.arange(n)

    ntrain = int(n * tr_percent / 100)

    tr_idx = np.random.choice(idx, ntrain, replace=False)
    val_idx = np.setxor1d(idx, tr_idx)

    w = np.asarray(windows)
    t = np.asarray(targets)

    return w[tr_idx], t[tr_idx], w[val_idx], t[val_idx]


def _save_training_hdf5(path, tr_w, tr_t, te_w, te_t):

    with h5py.File(path, "w") as f:

        d = f.create_dataset("lte training windows", data=tr_w)
        d.attrs["len"] = len(tr_w)

        f.create_dataset("non lte training points", data=tr_t)

        d = f.create_dataset("lte test windows", data=te_w)
        d.attrs["len"] = len(te_w)

        f.create_dataset("non lte test points", data=te_t)


# ============================================================
# ---------------- SOLVING SET -------------------------------
# ============================================================

def _assert_square_domain(temp):
    nx, ny, _ = temp.shape
    if nx != ny:
        raise ValueError(
            f"SunnyNet requires square domain (nx={nx}, ny={ny})"
        )
    return nx


def build_solving_set(
    rho, z_scale,
    temp, vx, vy, vz, ne,
    save_path="example.hdf5",
    ndep=400,
    pad=1,
):

    if os.path.isfile(save_path):
        raise IOError("Output exists.")

    nx = _assert_square_domain(temp)

    inputs, _ = _make_inputs_ch_first(
        rho, z_scale,
        temp, vx, vy, vz, ne,
        ndep=ndep,
        pad=pad,
    )

    windows = []
    _extract_prediction_windows(inputs, windows, pad)

    windows = np.asarray(windows)

    with h5py.File(save_path, "w") as f:
        d = f.create_dataset("lte test windows", data=windows)

        d.attrs["nx"] = nx
        d.attrs["ndep"] = ndep
        d.attrs["pad"] = pad


# ============================================================
# ---------------- TRAINING SET -------------------------------
# ============================================================

def build_training_set(
    temp_list, vx_list, vy_list, vz_list,
    ne_list, lte_list, nlte_list,
    rho_list, z_list,
    save_path="example.hdf5",
    ndep=400,
    pad=1,
    tr_percent=85,
):

    if os.path.isfile(save_path):
        raise IOError("Output exists.")

    windows = []
    targets = []

    for temp, vx, vy, vz, ne, lte, nlte, rho, z in zip(
        temp_list, vx_list, vy_list, vz_list,
        ne_list, lte_list, nlte_list,
        rho_list, z_list,
    ):

        inputs, _ = _make_inputs_ch_first(
            rho, z,
            temp, vx, vy, vz, ne,
            ndep=ndep,
            pad=pad,
        )

        tgt = _make_targets_ch_first(
            rho, z,
            lte, nlte,
            ndep=ndep,
            pad=pad,
        )

        _extract_training_windows(
            inputs, tgt,
            windows, targets,
            pad,
        )

    tr_w, tr_t, te_w, te_t = _train_test_split_global(
        windows, targets, tr_percent
    )

    _save_training_hdf5(save_path, tr_w, tr_t, te_w, te_t)


# ============================================================
# ---------------- PARAM READERS ------------------------------
# ============================================================

def read_train_params(train_file):

    with h5py.File(train_file, "r") as f:

        in_buf = f["lte training windows"].shape
        out_buf = f["non lte training points"].shape

        return (
            in_buf[0],
            f["lte test windows"].shape[0],
            in_buf[1],
            out_buf[1],
            in_buf[2],
            (in_buf[-1] - 1) // 2,
        )


def read_solve_params(solve_file):

    with h5py.File(solve_file, "r") as f:

        d = f["lte test windows"]
        shape = d.shape

        nx = int(d.attrs.get("nx", np.sqrt(shape[0])))

        return (
            nx,
            shape[1],
            shape[2],
            (shape[-1] - 1) // 2,
        )


# ============================================================
# ---------------- MODEL COMPAT -------------------------------
# ============================================================

def check_model_compat(model_type, pad):

    mapping = {
        "SunnyNet_1x1": 0,
        "SunnyNet_3x3": 1,
        "SunnyNet_5x5": 2,
        "SunnyNet_7x7": 3,
    }

    if model_type not in mapping:
        raise ValueError(model_type)

    return mapping[model_type] == pad


# ============================================================
# ---------------- TRAIN MODEL (API) --------------------------
# ============================================================

def sunnynet_train_model(
    train_path,
    save_folder,
    save_file,
    lines,
    wave,
    chi,
    levels,
    atom_names,
    model_type="SunnyNet_3x3",
    loss_function="MSELoss",
    cuda=True,
    multi_gpu=False,
):

    if not os.path.isdir(save_folder):
        os.mkdir(save_folder)

    tr, te, Cin, Cout, ndep, pad = read_train_params(train_path)

    if not check_model_compat(model_type, pad):
        raise ValueError("Model/window mismatch")

    params = dict(
        model=model_type,
        optimizer="Adam",
        loss_fxn=loss_function,
        learn_rate=1e-3,
        in_channels=Cin,
        out_channels=Cout,
        features=ndep,
        cuda={"use_cuda": cuda, "multi_gpu": multi_gpu},
        mode="training",
        lines=lines,
        wave=wave,
        chi=chi,
        levels=levels,
        atom_names=atom_names,
    )

    config = dict(
        data_path=train_path,
        save_folder=save_folder,
        model_save=save_file,
        early_stopping=10,
        num_epochs=50,
        train_size=tr,
        val_size=te,
        batch_size_train=128,
        batch_size_val=128,
        num_workers=64,
    )

    train_loader = DataLoader(
        PopulationDataset3d(train_path, True),
        batch_size=128,
        num_workers=64,
        pin_memory=True,
    )

    val_loader = DataLoader(
        PopulationDataset3d(train_path, False),
        batch_size=128,
        num_workers=64,
        pin_memory=True,
    )

    model = Model(params)

    losses = train(
        config,
        model,
        {"train": train_loader, "val": val_loader},
    )

    with open(f"{save_folder}{save_file[:-3]}_loss.txt", "w") as f:
        for trl, vall in zip(losses["train"], losses["val"]):
            f.write(f"{trl} {vall}\n")


# ============================================================
# ---------------- PREDICT (API) ------------------------------
# ============================================================

def sunnynet_predict_populations(
    model_path,
    train_path,
    test_path,
    save_path,
    lines,
    wave,
    chi,
    levels,
    atom_names,
    cuda=True,
    model_type="SunnyNet_3x3",
    loss_function="MSELoss",
):

    _, _, Cin, Cout, ndep, pad = read_train_params(train_path)
    nx, Cin2, ndep2, pad2 = read_solve_params(test_path)

    assert Cin == Cin2
    assert ndep == ndep2
    assert pad == pad2

    if os.path.isfile(save_path):
        raise IOError("Output exists.")

    pred_config = dict(
        cuda=cuda,
        model=model_type,
        model_path=model_path,
        in_channels=Cin,
        out_channels=Cout,
        features=ndep,
        mode="testing",
        loss_fxn=loss_function,
        output_XY=nx,
        lines=lines,
        wave=wave,
        chi=chi,
        levels=levels,
        atom_names=atom_names,
    )

    pops = predict_populations(
        test_path,
        train_path,
        pred_config,
    )

    pops = 10 ** pops

    with h5py.File(save_path, "w") as f:
        d = f.create_dataset("populations", data=pops)
        d.attrs["cmass_scale"] = np.logspace(-6, 2, 400)