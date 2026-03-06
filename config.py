import numpy as np


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

        "levels": 6
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

        "levels": 6
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
levels = [ATOM_CONFIG[a]["levels"]   for a in ACTIVE_ATOMS]


MULTI3D_PRED_DATA = [
    # {
    #     "MULTI3D_ATMOS": "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/385/atm3d",
    #     "MESH":  "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/385/mesh",
    #     "NAME": "en024048_hion_385"
    # },
    # {
    #     "MULTI3D_ATMOS": "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/386/atm3d",
    #     "MESH":  "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/386/mesh",
    #     "NAME": "en024048_hion_386"
    # },
    {
        "MULTI3D_ATMOS": "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/465/atm3d",
        "MESH":  "/mn/stornext/d9/data/harshm/bifrost_data/en024048_hion/465/mesh",
        "NAME": "en024048_hion_465"
    }
]


NDEP = 400
PAD = 3
WINDOW = 2 * PAD + 1
TAG = f"{WINDOW}x{WINDOW}"


# MODEL_TYPE = "SunnyNet"
MODEL_TYPE = "ContextToColumn3D"
MULTI_GPU = False

IODIR = "IO/"
TRAIN_FILE   = f"3D_sim_train_s123_{TAG}.hdf5"
MODEL_DIR  = f"training_{MODEL_TYPE}_{TAG}/"
MODEL_FILE = f"3D_sim_train_s123_{TAG}.pt"


