import os
import numpy as np
import h5py
import SunnyNet

from helita.sim.multi3d import Multi3dAtmos, Multi3dOut
from scipy.integrate import cumulative_trapezoid as cumtrapz
from scipy.interpolate import interp1d

# ------------------------------------------------------------
# CONFIG
# ------------------------------------------------------------
MULTI3D_TRAINING_DATA = [
    {
        "MULTI3D_PATHS": [
             "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/385/H",
             "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/385/CA"
        ],
        "MULTI3D_ATMOS": "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/385/atm3d",
    },
    {
        "MULTI3D_PATHS": [
             "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/386/H",
             "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/386/CA"
        ],
        "MULTI3D_ATMOS": "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/386/atm3d",
    }
]

lines = [
    np.array(
        [(0,1), (0,2), (0, 3), (0, 4), (1,2), (1,3), (1, 4), (2,3), (2, 4), (3, 4)],
        dtype=np.float32
    ),
    np.array(
        [(0, 3), (0, 4), (1, 3), (1, 4), (2, 4)],
        dtype=np.float32
    )
]

wave = [
    np.array([1215.6701, 1025.7220, 972.53650, 949.74287, 6562.79, 4861.35, 4340.472, 18750, 12820, 40510],
        dtype=np.float32
    ),
    np.array([3968.47, 3933.66, 8662.14, 8498.02, 8542.09],
        dtype=np.float32
    )
]

chi = [
    np.array(
        [
            1.6339941854018686e-18,
            1.936585907218822e-18,
            2.0424878450955273e-18,
            2.091506177644877e-18,
            2.1802152677122893e-18
        ],
        dtype=np.float32
    ),
    np.array(
        [
            2.7115478588655445e-19,
            2.7235960502783243e-19,
            5.004162033597224e-19,
            5.048445871090645e-19,
            1.902059054757628e-18
        ],
        dtype=np.float32
    )
]

levels = [
]

atom_names = ["H", "CA"]

NDEP = 400
PAD = 1
WINDOW = 2 * PAD + 1
TAG = f"{WINDOW}x{WINDOW}"

TRAIN_FILE   = f"3D_sim_train_s123_{TAG}.hdf5"
PREDICT_FILE = f"3D_sim_predict_{TAG}.hdf5"
OUTPUT_FILE  = f"sunnynet_output_3D_sim_s5_{TAG}.hdf5"

MODEL_DIR  = f"training_{TAG}/"
MODEL_FILE = f"3D_sim_train_s123_{TAG}.pt"
MODEL_TYPE = f"SunnyNet_{TAG}"

# ------------------------------------------------------------
# UTILS
# ------------------------------------------------------------
def interpolate_everything(rho_arr, z_scale, pops_array, new_cmass_scale):
    """
    Vectorized interpolation onto new cmass grid.
    """

    def interp_column(rho_col, z_col, pops_col, new_scale):
        cmass = cumtrapz(rho_col, -z_col, initial=0)
        f = interp1d(
            cmass,
            pops_col,
            axis=0,
            kind="linear",
            fill_value="extrapolate",
        )
        return f(new_scale)

    vec = np.vectorize(
        interp_column,
        signature="(n),(n),(n,o),(m)->(m,o)"
    )
    return vec(rho_arr, z_scale, pops_array, new_cmass_scale)

# ------------------------------------------------------------
# STEP 1: LOAD MULTI3D
# ------------------------------------------------------------
def load_multi3d():

    global levels
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

    for dataset in MULTI3D_TRAINING_DATA:

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


# ------------------------------------------------------------
# STEP 2: BUILD TRAINING SET
# ------------------------------------------------------------
def build_training(temp, vx, vy, vz, ne, lte, nlte, rho, z):

    print("\n=== BUILD TRAINING SET ===")

    SunnyNet.build_training_set(
        [temp], [vx], [vy], [vz],
        [ne], [lte], [nlte],
        [rho], [z],
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
def build_solving(rho, z, temp, vx, vy, vz, ne):

    print("\n=== BUILD SOLVING SET ===")

    SunnyNet.build_solving_set(
        rho, z, temp, vx, vy, vz, ne,
        save_path=PREDICT_FILE,
        ndep=NDEP,
        pad=PAD
    )

# ------------------------------------------------------------
# STEP 5: PREDICT POPULATIONS
# ------------------------------------------------------------
def predict():

    print("\n=== RUNNING PREDICTION ===")

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
# MAIN
# ------------------------------------------------------------
def main():

    atmos, rho, z, temp, vx, vy, vz, ne, lte, nlte = load_multi3d()

    # ---- TRAINING ----
    if not os.path.exists(TRAIN_FILE):
        build_training(temp, vx, vy, vz, ne, lte, nlte, rho, z)

    # ---- TRAIN MODEL ----
    if not os.path.exists(os.path.join(MODEL_DIR, MODEL_FILE)):
        train_model()

    # ---- SOLVING ----
    if not os.path.exists(PREDICT_FILE):
        build_solving(rho, z, temp, vx, vy, vz, ne)

    # ---- PREDICT ----
    if not os.path.exists(OUTPUT_FILE):
        predict()

    print("\n=== PIPELINE COMPLETE ===")

# ------------------------------------------------------------
if __name__ == "__main__":
    main()
