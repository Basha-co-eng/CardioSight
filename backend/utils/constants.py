from pathlib import Path

# ============================================================
# FEDERATED LEARNING CONFIGURATION & DATASETS
# ============================================================

FEDERATED_CLIENTS = [
    "chapman_shaoxing",
    "cpsc_2018",
    "georgia",
    "ningbo",
    "ptb-xl",
]

# ============================================================
# ECG PREPROCESSING CONFIGURATION
# ============================================================

TARGET_FS = 500
TARGET_SECONDS = 10
TARGET_SAMPLES = TARGET_FS * TARGET_SECONDS
SAMPLING_RATE = 500
LEAD_INDEX = 1
LEAD_NAME = "II"

# ============================================================
# 12 TARGET CARDIAC PATHOLOGIES
# ============================================================

CLASS_NAMES = [
    "AF",
    "IAVB",
    "LAD",
    "LBBB",
    "NSIVCB",
    "NSR",
    "PAC",
    "QAb",
    "RBBB",
    "SB",
    "STach",
    "TAb"
]

# ============================================================
# SHAP / FIDUCIAL FEATURE COLUMNS
# ============================================================

FEATURE_COLS = [
    "mean_rr_interval",
    "heart_rate_bpm",
    "p_wave_amplitude",
    "qrs_amplitude",
    "t_wave_amplitude",
    "pr_interval",
    "qt_interval",
    "qrs_duration",
    "n_beats_detected",
]

# ============================================================
# MC DROPOUT PARAMETERS
# ============================================================

N_MC_PASSES = 30
MC_DROPOUT_P = 0.3

# ============================================================
# MODEL CHECKPOINTS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent.parent

FEDAVG_MODEL_PATH = BASE_DIR / "models" / "fedavg" / "global_model_round30.pt"
FEDPROX_MODEL_PATH = BASE_DIR / "models" / "fedprox" / "fedprox_global_model_round30.pt"
FEDOPT_MODEL_PATH = BASE_DIR / "models" / "fedopt" / "fedopt_global_model_round30.pt"
FIDUCIAL_RF_PATH = BASE_DIR / "models" / "fiducial_rf_classifier.pkl"
