import os
import numpy as np
import h5py
import SunnyNet

from helita.sim.multi3d import Multi3dAtmos, Multi3dOut
import matplotlib.pyplot as plt
from interp_utils import interpolate_everything


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
def training_departure_profile():

    (
        atmos_list,
        rho_list,
        z_list,
        *_,
        lte_blocks,
        nlte_blocks,
    ) = load_training_multi3d_data()

    cmass_grid = np.logspace(-6, 2, NDEP)

    mean_all = []

    for i in range(len(lte_blocks)):

        rho = rho_list[i]
        z   = z_list[i]

        dep = nlte_blocks[i] / (lte_blocks[i] + 1e-30)
        logdep = np.log10(dep)

        logdep_cmass = interpolate_everything(
            rho, z, logdep, cmass_grid
        )

        mean_all.append(
            np.mean(np.abs(logdep_cmass), axis=(0,1,3))
        )

    mean_all = np.mean(mean_all, axis=0)

    return cmass_grid, mean_all

def prediction_error_profile():

    EPS = 1e-30

    (
        _,
        rho_list,
        z_list,
        *_,
        lte_blocks,
        nlte_blocks,
    ) = load_training_multi3d_data()

    err_all = []

    for idx, PRED_ATMOS in enumerate(MULTI3D_PRED_DATA):

        name = PRED_ATMOS["NAME"]

        pred_file = (
            f"sunnynet_output_3D_sim_s5_{name}_{TAG}.hdf5"
        )

        print(f"\nReading prediction: {pred_file}")

        with h5py.File(pred_file, "r") as f:
            pred_pop = f["populations"][:]   # ← NLTE prediction
            cmass = f["populations"].attrs["cmass_scale"]

        # ----------------------------------
        # TRUE departure coefficient
        # ----------------------------------
        lte_true  = lte_blocks[idx]
        nlte_true = nlte_blocks[idx]

        beta_true = (nlte_true + EPS)/(lte_true + EPS)
        beta_true = np.clip(beta_true, 1e-6, 1e6)

        logbeta_true = np.log10(beta_true)

        logbeta_true = interpolate_everything(
            rho_list[idx],
            z_list[idx],
            logbeta_true,
            cmass
        )

        # ----------------------------------
        # PREDICTED departure coefficient
        # ----------------------------------
        beta_pred = (pred_pop + EPS)/(lte_true + EPS)
        beta_pred = np.clip(beta_pred, 1e-6, 1e6)

        logbeta_pred = np.log10(beta_pred)

        # ----------------------------------
        # ERROR
        # ----------------------------------
        abs_err = np.abs(
            logbeta_pred - logbeta_true
        )

        err_profile = np.mean(
            abs_err,
            axis=(0,1,3)
        )

        err_all.append(err_profile)

    return cmass, np.mean(err_all, axis=0)


def plot_overlay_diagnostics(
    cmass,
    nlte_strength,
    ml_error,
    savepath="NLTE_vs_ML_error.pdf"
):

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "pdf.fonttype": 42,
    })

    logcm = np.log10(cmass)

    fig, ax = plt.subplots(
        figsize=(6.5,4.5)
    )

    ax.plot(
        logcm,
        nlte_strength,
        lw=3,
        label="Intrinsic NLTE departure"
    )

    ax.plot(
        logcm,
        ml_error,
        lw=3,
        linestyle="--",
        label="SunnyNet prediction error"
    )

    ax.set_xlabel(
        r"$\log_{10}(\mathrm{Column\ Mass}\,[g\,cm^{-2}])$"
    )

    ax.set_ylabel(
        r"Mean $|\Delta \log_{10}\beta|$"
    )

    ax.grid(alpha=0.3)
    ax.legend()

    ax.invert_xaxis()

    plt.tight_layout()

    fig.savefig(
        savepath,
        bbox_inches="tight"
    )

    print(f"Saved → {savepath}")

    plt.close(fig)


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

    cmass, nlte_strength = training_departure_profile()
    _, ml_error = prediction_error_profile()

    plot_overlay_diagnostics(
        cmass,
        nlte_strength,
        ml_error
    )

    print("\n=== PIPELINE COMPLETE ===")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
