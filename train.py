import os
import numpy as np
import h5py
import SunnyNet

from helita.sim.multi3d import Multi3dAtmos, Multi3dOut
import matplotlib.pyplot as plt
from interp_utils import interpolate_everything
from matplotlib.colors import LogNorm


SIMULATIONS = {
    "en024048_hion": {
        "base_path": "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion",
        "snaps": ["385", "386"],
    },

    # tomorrow you just add:
    # "en012345_newrun": {
    #     "base_path": ".../en012345_newrun",
    #     "snaps": ["420", "421"],
    # },
}

ATOM_CONFIG = {
    "H": {
        "subdir": "H",

        "lines": np.array(
            [(0,1),(0,2),(0,3),(0,4),
             (1,2),(1,3),(1,4),
             (2,3),(2,4),(3,4)],
            dtype=np.float32
        ),

        "wave": np.array(
            [1215.6701,1025.7220,972.53650,949.74287,
             6562.79,4861.35,4340.472,
             18750,12820,40510],
            dtype=np.float32
        ),

        "chi": np.array([
            1.6339941854018686e-18,
            1.936585907218822e-18,
            2.0424878450955273e-18,
            2.091506177644877e-18,
            2.1802152677122893e-18
        ], dtype=np.float32),
    },

    "CA": {
        "subdir": "CA",

        "lines": np.array(
            [(0,3),(0,4),(1,3),(1,4),(2,4)],
            dtype=np.float32
        ),

        "wave": np.array(
            [3968.47,3933.66,8662.14,8498.02,8542.09],
            dtype=np.float32
        ),

        "chi": np.array([
            2.7115478588655445e-19,
            2.7235960502783243e-19,
            5.004162033597224e-19,
            5.048445871090645e-19,
            1.902059054757628e-18
        ], dtype=np.float32),
    }
}

ACTIVE_SIMS  = ["en024048_hion"]
ACTIVE_ATOMS = ["H"]

MULTI3D_TRAINING_DATA = []

for sim in ACTIVE_SIMS:

    sim_info = SIMULATIONS[sim]
    base = sim_info["base_path"]

    for snap in sim_info["snaps"]:

        entry = {
            "MULTI3D_PATHS": [],
            "MULTI3D_ATMOS": f"{base}/{snap}/atm3d"
        }

        for atom in ACTIVE_ATOMS:
            atom_dir = ATOM_CONFIG[atom]["subdir"]

            entry["MULTI3D_PATHS"].append(
                f"{base}/{snap}/{atom_dir}"
            )

        MULTI3D_TRAINING_DATA.append(entry)

atom_names = ACTIVE_ATOMS
lines = [ATOM_CONFIG[a]["lines"] for a in ACTIVE_ATOMS]
wave  = [ATOM_CONFIG[a]["wave"]  for a in ACTIVE_ATOMS]
chi   = [ATOM_CONFIG[a]["chi"]   for a in ACTIVE_ATOMS]


MULTI3D_PRED_DATA = [
    {
        "MULTI3D_ATMOS": "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/385/atm3d",
        "MESH":  "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/385/mesh",
        "NAME": "en024048_hion_385"
    },
    {
        "MULTI3D_ATMOS": "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/386/atm3d",
        "MESH":  "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/386/mesh",
        "NAME": "en024048_hion_386"
    }
]

levels = [
]

NDEP = 400
PAD = 1
WINDOW = 2 * PAD + 1
TAG = f"{WINDOW}x{WINDOW}"

TRAIN_FILE   = f"3D_sim_train_s123_{TAG}.hdf5"
MODEL_DIR  = f"training_{TAG}/"
MODEL_FILE = f"3D_sim_train_s123_{TAG}.pt"
MODEL_TYPE = f"SunnyNet_{TAG}"


def read_mesh(mesh_file):
    """
    Reads mesh file from Bifrost or MULTI3D.
    Equivalent to Julia read_mesh().
    """

    # Read ALL whitespace-separated numbers
    tmp = np.fromfile(mesh_file, sep=" ", dtype=np.float32)

    inc = 0

    nx = int(tmp[inc])
    inc += 1
    x = tmp[inc:inc + nx]
    inc += nx

    ny = int(tmp[inc])
    inc += 1
    y = tmp[inc:inc + ny]
    inc += ny

    nz = int(tmp[inc])
    inc += 1
    z = tmp[inc:inc + nz]

    return nx, ny, nz, x, y, z


def ensure_levels_loaded():
    global levels
    if len(levels) == 0:
        print("Loading levels from training data...")
        load_training_multi3d_data()


# ------------------------------------------------------------
# STEP 1: LOAD MULTI3D
# ------------------------------------------------------------
def load_training_multi3d_data():

    global levels
    levels = []
    
    print("\n=== LOADING MULTI3D DATA ===")

    atmos_list = []
    rho_list = []
    z_list = []
    temp_list = []
    vx_list = []
    vy_list = []
    vz_list = []
    ne_list = []
    lte_blocks = []
    nlte_blocks = []

    for dataset_idx, dataset in enumerate(MULTI3D_TRAINING_DATA):

        print(f"\n=== DATASET BLOCK ===")

        lte_block = None
        nlte_block = None
        atmos = None

        atmos_path = dataset["MULTI3D_ATMOS"]

        for mpath in dataset["MULTI3D_PATHS"]:

            print(f"\n--- Reading MULTI3D output: {mpath}")

            m3d = Multi3dOut(directory=mpath)
            m3d.readall()

            lte = m3d.atom.nstar * 1e6
            nlte = m3d.atom.n * 1e6

            if dataset_idx == 0:
                levels.append(nlte.shape[-1])

            # concatenate only inside block
            if lte_block is None:
                lte_block = lte
                nlte_block = nlte
            else:
                lte_block  = np.concatenate([lte_block,  lte],  axis=-1)
                nlte_block = np.concatenate([nlte_block, nlte], axis=-1)

            # Load atmos once per block
            if atmos is None:
                nx, ny, nz, _ = lte.shape
                atmos = Multi3dAtmos(atmos_path, nx, ny, nz)

                # IMPORTANT: some Multi3dAtmos need readall()
                if hasattr(atmos, "readall"):
                    atmos.readall()

                rho = atmos.rho * 1e3
                temp = atmos.temp
                vx = atmos.vx
                vy = atmos.vy
                vz = atmos.vz
                ne = atmos.ne * 1e6
                z_scale = m3d.geometry.z * 1e-2

        print("Block total levels:", lte_block.shape[-1])

        # store this block
        lte_blocks.append(lte_block)
        nlte_blocks.append(nlte_block)

        atmos_list.append(atmos)
        rho_list.append(rho)
        temp_list.append(temp)
        vx_list.append(vx)
        vy_list.append(vy)
        vz_list.append(vz)
        ne_list.append(ne)
        z_list.append(z_scale)

    return (
        atmos_list,
        rho_list,
        z_list,
        temp_list,
        vx_list,
        vy_list,
        vz_list,
        ne_list,
        lte_blocks,
        nlte_blocks,
    )


def load_pred_data(mesh_file, atmos_file):
    """
    Loads atmosphere for prediction (no MULTI3D output required).

    Parameters
    ----------
    mesh_file : str
        Path to mesh file (Bifrost/Multi3D mesh)
    atmos_file : str
        Path to atmosphere file (atm3d)

    Returns
    -------
    rho, z_scale, temp, vx, vy, vz, ne
    """

    print("\n=== LOADING PREDICTION ATMOSPHERE ===")

    # --- Read mesh ---
    nx, ny, nz, x, y, z = read_mesh(mesh_file)
    print(f"Grid size from mesh: nx={nx}, ny={ny}, nz={nz}")

    # --- Load atmosphere ---
    atmos = Multi3dAtmos(atmos_file, nx, ny, nz)

    # --- Extract physical variables ---
    rho = atmos.rho * 1e3       # g/cm^3 → kg/m^3
    temp = atmos.temp
    vx = atmos.vx
    vy = atmos.vy
    vz = atmos.vz
    ne = atmos.ne * 1e6         # cm^-3 → m^-3

    # z from mesh is usually in cm
    z_scale = z * 1e-2          # cm → m

    print("Atmosphere loaded successfully.")
    print("rho shape:", rho.shape)

    return rho, z_scale, temp, vx, vy, vz, ne


# ------------------------------------------------------------
# STEP 2: BUILD TRAINING SET
# ------------------------------------------------------------
def build_training(temp, vx, vy, vz, ne, lte, nlte, rho, z):

    print("\n=== BUILD TRAINING SET ===")

    SunnyNet.build_training_set(
        temp, vx, vy, vz,
        ne, lte, nlte,
        rho, z,
        save_path=TRAIN_FILE,
        ndep=NDEP,
        pad=PAD,
        tr_percent=85
    )

# ------------------------------------------------------------
# STEP 3: TRAIN MODEL
# ------------------------------------------------------------
def train_model():

    print("\n=== TRAIN MODEL ===")

    SunnyNet.sunnynet_train_model(
        TRAIN_FILE,
        MODEL_DIR,
        MODEL_FILE,
        lines=lines,
        wave=wave,
        chi=chi,
        levels=levels,
        atom_names=atom_names,
        model_type=MODEL_TYPE,
        cuda=True,
        loss_function="PhysicsLoss",

    )

# ------------------------------------------------------------
# STEP 4: BUILD SOLVING SET
# ------------------------------------------------------------
def build_solving():

    print("\n=== BUILD SOLVING SET ===")

    for PRED_ATMOS in MULTI3D_PRED_DATA:

        PREDICT_FILE = f"3D_sim_predict_{PRED_ATMOS['NAME']}_{TAG}.hdf5"

        if not os.path.exists(PREDICT_FILE):

            rho, z_scale, temp, vx, vy, vz, ne = load_pred_data(
                PRED_ATMOS["MESH"],
                PRED_ATMOS["MULTI3D_ATMOS"]
            )

            SunnyNet.build_solving_set(
                rho, z_scale, temp, vx, vy, vz, ne,
                save_path=PREDICT_FILE,
                ndep=NDEP,
                pad=PAD
            )

# ------------------------------------------------------------
# STEP 5: PREDICT POPULATIONS
# ------------------------------------------------------------
def predict():

    print("\n=== RUNNING PREDICTION ===")

    for PRED_ATMOS in MULTI3D_PRED_DATA:

        PREDICT_FILE = f"3D_sim_predict_{PRED_ATMOS['NAME']}_{TAG}.hdf5"

        OUTPUT_FILE  = f"sunnynet_output_3D_sim_s5_{PRED_ATMOS['NAME']}_{TAG}.hdf5"

        if not os.path.exists(OUTPUT_FILE):

            SunnyNet.sunnynet_predict_populations(
                os.path.join(MODEL_DIR, MODEL_FILE),
                TRAIN_FILE,
                PREDICT_FILE,
                OUTPUT_FILE,
                lines=lines,
                wave=wave,
                chi=chi,
                levels=levels,
                atom_names=atom_names,
                model_type=MODEL_TYPE,
                loss_function="PhysicsLoss"
            )

# ------------------------------------------------------------
# STEP 6: DIAGNOSTICS
# ------------------------------------------------------------
def training_departure_density_per_level(
        rho_list, z_list,
        lte_blocks, nlte_blocks):

    EPS = 1e-30
    cmass_grid = np.logspace(-6, 2, NDEP)

    nlevel = lte_blocks[0].shape[-1]

    results = []

    for lev in range(nlevel):

        X_all = []
        Y_all = []

        for i in range(len(lte_blocks)):

            rho = rho_list[i]
            z   = z_list[i]

            beta = (
                nlte_blocks[i][..., lev] + EPS
            ) / (
                lte_blocks[i][..., lev] + EPS
            )

            logbeta = np.log10(
                np.clip(beta, 1e-12, 1e12)
            )

            logbeta = logbeta[..., None]

            logbeta_cmass = interpolate_everything(
                rho, z, logbeta, cmass_grid
            )[...,0]

            nx, ny, nz = logbeta_cmass.shape

            X = np.repeat(np.log10(cmass_grid), nx*ny)
            Y = np.abs(
                logbeta_cmass.reshape(-1, nz).T
            ).flatten()

            X_all.append(X)
            Y_all.append(Y)

        results.append((
            np.concatenate(X_all),
            np.concatenate(Y_all)
        ))

    return results


def prediction_error_density_per_level(
        rho_list, z_list,
        lte_blocks, nlte_blocks):

    EPS = 1e-30
    nlevel = lte_blocks[0].shape[-1]

    results = []

    for lev in range(nlevel):

        X_all = []
        Y_all = []

        for idx, PRED_ATMOS in enumerate(MULTI3D_PRED_DATA):

            name = PRED_ATMOS["NAME"]
            pred_file = f"sunnynet_output_3D_sim_s5_{name}_{TAG}.hdf5"

            with h5py.File(pred_file,"r") as f:
                beta_pred = f["populations"][..., lev]
                cmass = f["populations"].attrs["cmass_scale"]

            rho = rho_list[idx]
            z   = z_list[idx]

            beta_true = (
                nlte_blocks[idx][..., lev]+EPS
            ) / (
                lte_blocks[idx][..., lev]+EPS
            )

            logbeta_true = np.log10(
                np.clip(beta_true,1e-12,1e12)
            )[...,None]

            logbeta_true = interpolate_everything(
                rho, z, logbeta_true, cmass
            )[...,0]

            logbeta_pred = np.log10(
                np.clip(beta_pred,1e-12,1e12)
            )

            err = np.abs(
                logbeta_pred - logbeta_true
            )

            nx, ny, nz = err.shape

            X = np.repeat(np.log10(cmass), nx*ny)
            Y = err.reshape(-1, nz).T.flatten()

            X_all.append(X)
            Y_all.append(Y)

        results.append((
            np.concatenate(X_all),
            np.concatenate(Y_all)
        ))

    return results


def compute_percentile_profiles(x, y, cmass_grid):

    logcm = np.log10(cmass_grid)

    median = np.zeros_like(logcm)
    p16 = np.zeros_like(logcm)
    p84 = np.zeros_like(logcm)
    p2 = np.zeros_like(logcm)
    p98 = np.zeros_like(logcm)

    # small bin around each cmass
    dx = np.abs(logcm[1] - logcm[0]) * 0.5

    for i, xc in enumerate(logcm):

        mask = (x > xc - dx) & (x < xc + dx)

        if np.sum(mask) < 50:
            median[i] = np.nan
            p16[i] = np.nan
            p84[i] = np.nan
            p2[i] = np.nan
            p98[i] = np.nan
            continue

        vals = y[mask]

        median[i] = np.percentile(vals, 50)
        p16[i] = np.percentile(vals, 16)
        p84[i] = np.percentile(vals, 84)
        p2[i] = np.percentile(vals, 2.5)
        p98[i] = np.percentile(vals, 97.5)

    return median, p16, p84, p2, p98


from matplotlib.colors import LogNorm


def plot_density_grid_with_stats(
        level_data,
        cmass_grid,
        ylabel,
        savepath,
        ncols=3):

    nlevel = len(level_data)
    nrows = int(np.ceil(nlevel/ncols))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4*ncols,3.5*nrows),
        sharex=True,
        sharey=True,
        constrained_layout=True
    )

    axes = np.atleast_2d(axes)

    logcm = np.log10(cmass_grid)

    h_last = None

    for lev,(x,y) in enumerate(level_data):

        r = lev//ncols
        c = lev%ncols
        ax = axes[r,c]

        # ---------------- density ----------------
        h = ax.hist2d(
            x,y,
            bins=[220,220],
            cmap="inferno",
            norm=LogNorm()
        )

        h_last = h

        # ---------------- statistics ----------------
        med,p16,p84,p2,p98 = compute_percentile_profiles(
            x,y,cmass_grid
        )

        # 95% envelope
        ax.plot(logcm,p2,'w--',lw=1,alpha=0.8)
        ax.plot(logcm,p98,'w--',lw=1,alpha=0.8)

        # 68% band
        ax.fill_between(
            logcm,p16,p84,
            color="white",
            alpha=0.25
        )

        # median
        ax.plot(logcm,med,'w',lw=2)

        ax.set_title(f"Level {lev}")
        ax.invert_xaxis()

    # remove empty panels
    for k in range(nlevel,nrows*ncols):
        fig.delaxes(axes.flat[k])

    fig.supxlabel(
        r"$\log_{10}(\mathrm{Column\ Mass})$"
    )
    fig.supylabel(ylabel)

    cbar = fig.colorbar(
        h_last[3],
        ax=axes.ravel().tolist(),
        shrink=0.9
    )
    cbar.set_label("Counts")

    fig.savefig(savepath,bbox_inches="tight")
    plt.close(fig)

    print(f"Saved → {savepath}")


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():

    # ---- TRAINING ----
    if not os.path.exists(TRAIN_FILE):
        atmos, rho, z, temp, vx, vy, vz, ne, lte, nlte = load_training_multi3d_data()
        build_training(temp, vx, vy, vz, ne, lte, nlte, rho, z)
        with h5py.File(TRAIN_FILE) as f:
            print(f["lte training windows"].dtype)

    # ---- TRAIN MODEL ----
    ensure_levels_loaded()

    if not os.path.exists(os.path.join(MODEL_DIR, MODEL_FILE)):
        train_model()

    ensure_levels_loaded()

    # ---- SOLVING ----
    build_solving()

    # ---- PREDICT ----
    predict()

    # ---- Load training blocks once (for diagnostics) ----
    (
        atmos_list,
        rho_list,
        z_list,
        temp_list,
        vx_list,
        vy_list,
        vz_list,
        ne_list,
        lte_blocks,
        nlte_blocks,
    ) = load_training_multi3d_data()

    cmass_grid = np.logspace(-6, 2, NDEP)
    
    intrinsic = training_departure_density_per_level(
        rho_list,
        z_list,
        lte_blocks,
        nlte_blocks
    )

    plot_density_grid_with_stats(
        intrinsic,
        cmass_grid,
        r"$|\log_{10}\beta|$",
        "NLTE_intrinsic_levels_stats.pdf"
    )

    ml = prediction_error_density_per_level(
        rho_list,
        z_list,
        lte_blocks,
        nlte_blocks
    )

    plot_density_grid_with_stats(
        ml,
        cmass_grid,
        r"$|\Delta\log_{10}\beta|$",
        "SunnyNet_error_levels_stats.pdf"
    )


    print("\n=== PIPELINE COMPLETE ===")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
