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
MULTI3D_PATH = "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion_h6_252_3d_output"
MULTI3D_ATMOS = "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/atm3d.en024048_hion"

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

    print("\n=== LOADING MULTI3D DATA ===")

    m3d = Multi3dOut(directory=MULTI3D_PATH)
    m3d.readall()

    lte_pops = m3d.atom.nstar * 1e6  # cm^-3 to m^-3
    nlte_pops = m3d.atom.n * 1e6  # cm^-3 to m^-3

    nx, ny, nz, nlevel = m3d.atom.nstar.shape
    atmos = Multi3dAtmos(MULTI3D_ATMOS, nx, ny, nz)

    rho = atmos.rho * 1e3  #  g cm-3 to kg m-3
    z_scale = m3d.geometry.z * 1e-2  #  g cm-3 to kg m-3

    temp = atmos.temp
    vx = atmos.vx   # km s-1
    vy = atmos.vy   # km s-1
    vz = atmos.vz   # km s-1
    ne = atmos.ne * 1e6  #  cm-3 to m-3

    print("Grid:", nx, ny, nz, "levels:", nlevel)

    return m3d, atmos, rho, z_scale, temp, vx, vy, vz, ne, lte_pops, nlte_pops

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
        model_type=MODEL_TYPE,
        cuda=True,
        loss_function="PhysicsLoss"
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
        model_type=MODEL_TYPE,
        loss_function="PhysicsLoss"
    )

# ------------------------------------------------------------
# STEP 6: OPTIONAL DIAGNOSTICS
# ------------------------------------------------------------
def diagnostics(m3d, atmos, rho, z):

    print("\n=== RUNNING DIAGNOSTICS ===")

    f = h5py.File(OUTPUT_FILE, "r")
    print("Output keys:", list(f.keys()))
    print("Pop shape:", f["populations"].shape)

    dep = m3d.atom.nstar / m3d.atom.n
    new_cmass_scale = np.logspace(-6, 2, NDEP)

    departure = interpolate_everything(rho, z, dep, new_cmass_scale)
    temperature = interpolate_everything(
        rho, z,
        atmos.temp[..., np.newaxis],
        new_cmass_scale
    )[..., 0]

    print("Interpolated temp shape:", temperature.shape)
    print("Interpolated dep shape:", departure.shape)

# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():

    m3d, atmos, rho, z, temp, vx, vy, vz, ne, lte, nlte = load_multi3d()

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
