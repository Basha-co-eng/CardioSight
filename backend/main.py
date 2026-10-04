from pathlib import Path
from scipy import signal
import wfdb
import numpy as np
from scipy.signal import resample_poly, butter, filtfilt, iirnotch
from math import gcd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import os
import shutil
import tempfile
from functools import lru_cache

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse


def read_header_metadata(hea_path: Path) -> dict:
    record_path = hea_path.with_suffix("")
    header = wfdb.rdheader(str(record_path))
    return {
        "record_id": hea_path.stem,
        "sampling_rate": header.fs,
        "num_leads": header.n_sig,
        "num_samples": header.sig_len,
        "lead_names": list(header.sig_name),
        "file_path": str(record_path),
    }

# ============================================================
# ECG PREPROCESSING CONFIGURATION
# ============================================================

TARGET_FS = 500
TARGET_SECONDS = 10
TARGET_SAMPLES = TARGET_FS * TARGET_SECONDS


# ============================================================
# RESAMPLING
# ============================================================

def resample_signal(
    signal: np.ndarray,
    orig_fs: int,
    target_fs: int
) -> np.ndarray:

    if orig_fs == target_fs:
        return signal

    g = gcd(int(orig_fs), int(target_fs))

    up = int(target_fs // g)
    down = int(orig_fs // g)

    return resample_poly(
        signal,
        up,
        down,
        axis=1
    )

# ============================================================
# BANDPASS FILTER
# ============================================================

def bandpass_filter(
    signal: np.ndarray,
    fs: int,
    low: float = 0.5,
    high: float = 40.0,
    order: int = 4
) -> np.ndarray:

    nyq = fs / 2.0

    b, a = butter(
        order,
        [low / nyq, high / nyq],
        btype="band"
    )

    return filtfilt(
        b,
        a,
        signal,
        axis=1
    )


# ============================================================
# NOTCH FILTER
# ============================================================

def notch_filter(
    signal: np.ndarray,
    fs: int,
    freq: float = 50.0,
    quality: float = 30.0
) -> np.ndarray:

    nyq = fs / 2.0

    b, a = iirnotch(
        freq / nyq,
        quality
    )

    return filtfilt(
        b,
        a,
        signal,
        axis=1
    )


# ============================================================
# HOSPITAL-SPECIFIC NOTCH FREQUENCY
# ============================================================

def get_notch_freq(source_hospital: str) -> float:

    return 60.0 if source_hospital == "georgia" else 50.0


# ============================================================
# FIX ECG LENGTH TO 5000 SAMPLES
# ============================================================

def fix_length(
    signal: np.ndarray,
    target_samples: int
) -> np.ndarray:

    n_leads, n_samples = signal.shape

    if n_samples == target_samples:
        return signal

    elif n_samples < target_samples:

        pad_width = target_samples - n_samples

        return np.pad(
            signal,
            ((0, 0), (0, pad_width)),
            mode="constant"
        )

    else:

        return signal[:, :target_samples]


# ============================================================
# PER-LEAD NORMALIZATION
# ============================================================

def normalize_signal(
    signal: np.ndarray
) -> np.ndarray:

    mean = signal.mean(
        axis=1,
        keepdims=True
    )

    std = signal.std(
        axis=1,
        keepdims=True
    )

    std[std == 0] = 1.0

    return (signal - mean) / std


# ============================================================
# COMPLETE SINGLE-RECORD PREPROCESSING
# ============================================================

def process_one_record(
    file_path: str,
    orig_fs: int,
    source_hospital: str
) -> np.ndarray:

    """
    Load one ECG record and convert it to:

        (12, 5000)

    Processing:
        1. Load WFDB signal
        2. Convert to (leads, samples)
        3. Resample to 500 Hz
        4. Bandpass 0.5–40 Hz
        5. Apply 50/60 Hz notch
        6. Fix length to 5000 samples
        7. Normalize each lead
    """

    record = wfdb.rdrecord(file_path)

    signal = record.p_signal.T

    signal = resample_signal(
        signal,
        orig_fs,
        TARGET_FS
    )

    signal = bandpass_filter(
        signal,
        TARGET_FS
    )

    signal = notch_filter(
        signal,
        TARGET_FS,
        freq=get_notch_freq(source_hospital)
    )

    signal = fix_length(
        signal,
        TARGET_SAMPLES
    )

    signal = normalize_signal(
        signal
    )

    return signal.astype(
        np.float32
    )

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.downsample = None

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


class ResNet1D34(nn.Module):

    def __init__(self, in_channels=12, num_classes=12):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(
                kernel_size=3,
                stride=2,
                padding=1
            )
        )

        self.layer1 = self._make_layer(
            64, 64, num_blocks=3, stride=1
        )

        self.layer2 = self._make_layer(
            64, 128, num_blocks=4, stride=2
        )

        self.layer3 = self._make_layer(
            128, 256, num_blocks=6, stride=2
        )

        self.layer4 = self._make_layer(
            256, 512, num_blocks=3, stride=2
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.fc = nn.Linear(
            512,
            num_classes
        )

    def _make_layer(
        self,
        in_channels,
        out_channels,
        num_blocks,
        stride
    ):

        layers = [
            ResidualBlock1D(
                in_channels,
                out_channels,
                stride
            )
        ]

        for _ in range(1, num_blocks):
            layers.append(
                ResidualBlock1D(
                    out_channels,
                    out_channels,
                    stride=1
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)
        x = x.squeeze(-1)

        x = self.fc(x)

        return x

# ============================================================
# CLASS NAMES
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
# MODEL CHECKPOINTS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

# ------------------------------------------------------------
# DEPLOYMENT CONFIGURATION
# ------------------------------------------------------------
# The backend does not depend on a specific computer.
#
# Local development:
#   Windows PowerShell:
#   $env:CARDIOSIGHT_MODEL_DIR="C:\ECG_Project\training"
#
#   Or use the automatic local fallback below.
#
# Production:
#   Set CARDIOSIGHT_MODEL_DIR to the directory containing the
#   model folders before starting the FastAPI server.
#
# Expected model directory:
#
#   models/
#   ├── fedavg/
#   │   └── global_model_round30.pt
#   ├── fedprox_mu0.001/
#   │   └── fedprox_global_model_round30.pt
#   ├── fedopt/
#   │   └── fedopt_global_model_round30.pt
#   └── fiducial_rf_classifier.pkl
#
# The environment variable takes priority over local defaults.

ENV_MODEL_DIR = os.getenv("CARDIOSIGHT_MODEL_DIR")

if ENV_MODEL_DIR:
    MODEL_DIR = Path(ENV_MODEL_DIR).expanduser()
else:
    # Development fallback for the current training machine.
    # This is NOT required on a deployed server.
    LOCAL_TRAINING_DIR = Path(r"C:\ECG_Project\training")

    if LOCAL_TRAINING_DIR.exists():
        MODEL_DIR = LOCAL_TRAINING_DIR
    else:
        MODEL_DIR = BASE_DIR / "models"

FEDAVG_MODEL_PATH = (
    MODEL_DIR / "fedavg" / "global_model_round30.pt"
)

FEDPROX_MODEL_PATH = (
    MODEL_DIR
    / "fedprox_mu0.001"
    / "fedprox_global_model_round30.pt"
)

FEDOPT_MODEL_PATH = (
    MODEL_DIR
    / "fedopt"
    / "fedopt_global_model_round30.pt"
)

FIDUCIAL_RF_PATH = (
    MODEL_DIR / "fiducial_rf_classifier.pkl"
)


# ============================================================
# MODEL LOADERS
# ============================================================

def _load_required_checkpoint(model: nn.Module, checkpoint_path: Path, device) -> nn.Module:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Invalid checkpoint format: {checkpoint_path}")
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model

def load_fedavg_model(device):
    return _load_required_checkpoint(ResNet1D34(in_channels=12, num_classes=12), FEDAVG_MODEL_PATH, device)

def load_fedprox_model(device):
    return _load_required_checkpoint(ResNet1D34(in_channels=12, num_classes=12), FEDPROX_MODEL_PATH, device)

def load_fedopt_model(device):
    return _load_required_checkpoint(ResNet1D34(in_channels=12, num_classes=12), FEDOPT_MODEL_PATH, device)


# ============================================================
# GENERIC PREDICTION
# ============================================================

def predict(model, signal, device):

    signal_tensor = torch.tensor(
        signal,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    with torch.no_grad():

        logits = model(signal_tensor)

        probabilities = torch.sigmoid(logits)

    return (
        logits.cpu().numpy()[0],
        probabilities.cpu().numpy()[0]
    )

# ============================================================
# GRAD-CAM
# ============================================================

class GradCAM1D:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.activations = None
        self.gradients = None

        self.forward_handle = target_layer.register_forward_hook(
            self._save_activation
        )

        self.backward_handle = target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(self, module, inp, out):
        self.activations = out.detach()

    def _save_gradient(self, module, grad_in, grad_out):
        self.gradients = grad_out[0].detach()

    def generate(
        self,
        x: torch.Tensor,
        target_class_idx: int,
        signal_length: int
    ) -> np.ndarray:

        self.model.zero_grad(set_to_none=True)

        output = self.model(x)

        score = output[0, target_class_idx]

        score.backward()

        weights = self.gradients.mean(
            dim=2,
            keepdim=True
        )

        cam = (weights * self.activations).sum(dim=1)

        cam = F.relu(cam)

        cam = cam.unsqueeze(1)

        cam = F.interpolate(
            cam,
            size=signal_length,
            mode="linear",
            align_corners=False
        )

        cam = cam.squeeze().cpu().numpy()

        cam_min = cam.min()
        cam_max = cam.max()

        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def remove_hooks(self):
        self.forward_handle.remove()
        self.backward_handle.remove()

def generate_gradcam(
    model: torch.nn.Module,
    signal: np.ndarray,
    predicted_class_idx: int,
    device
) -> np.ndarray:

    signal_tensor = torch.tensor(
        signal,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    cam_engine = GradCAM1D(
        model,
        target_layer=model.layer4
    )

    try:
        heatmap = cam_engine.generate(
            signal_tensor,
            target_class_idx=predicted_class_idx,
            signal_length=signal.shape[-1]
        )

        return heatmap

    finally:
        cam_engine.remove_hooks()

# ============================================================
# SHAP / FIDUCIAL FEATURE CLASSIFIER
# ============================================================

import neurokit2 as nk
import joblib
import shap


# ------------------------------------------------------------
# FIDUCIAL FEATURE CONFIGURATION
# ------------------------------------------------------------

SAMPLING_RATE = 500
LEAD_INDEX = 1
LEAD_NAME = "II"

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


# ------------------------------------------------------------
# RF CLASSIFIER CHECKPOINT
# ------------------------------------------------------------

FIDUCIAL_RF_PATH = Path(
    r"C:\ECG_Project\training\fiducial_rf_classifier.pkl"
)

if not FIDUCIAL_RF_PATH.exists():
    FIDUCIAL_RF_PATH = BASE_DIR / "models" / "fiducial_rf_classifier.pkl"


# ------------------------------------------------------------
# LOAD FIDUCIAL RF CLASSIFIER
# ------------------------------------------------------------

def load_fiducial_rf():
    if FIDUCIAL_RF_PATH.exists():
        clf = joblib.load(FIDUCIAL_RF_PATH)
        return clf
    return None

# ------------------------------------------------------------
# FIDUCIAL FEATURE EXTRACTION
# ------------------------------------------------------------

def extract_fiducial_features_from_signal(
    signal: np.ndarray,
    sampling_rate: int = SAMPLING_RATE,
    lead_index: int = LEAD_INDEX
) -> dict | None:
    """Extract real fiducial features from Lead II."""
    try:
        if signal.ndim != 2 or lead_index >= signal.shape[0]:
            return None

        lead_signal = np.asarray(signal[lead_index, :], dtype=float)
        if not np.all(np.isfinite(lead_signal)):
            return None

        _, info = nk.ecg_process(lead_signal, sampling_rate=sampling_rate)
        r_peaks = info.get("ECG_R_Peaks", [])
        if r_peaks is None:
            return None

        r_peaks = np.asarray(r_peaks, dtype=float)
        r_peaks = r_peaks[np.isfinite(r_peaks)].astype(int)
        r_peaks = r_peaks[(r_peaks >= 0) & (r_peaks < len(lead_signal))]
        if len(r_peaks) < 2:
            return None

        rr_intervals = np.diff(r_peaks) / float(sampling_rate)
        rr_intervals = rr_intervals[np.isfinite(rr_intervals) & (rr_intervals > 0)]
        if len(rr_intervals) == 0:
            return None

        mean_rr = float(np.mean(rr_intervals))
        hr_bpm = float(60.0 / mean_rr)

        def safe_amp(key):
            arr = np.asarray(info.get(key, []), dtype=float)
            if arr.size == 0:
                return None
            valid = arr[np.isfinite(arr)].astype(int)
            valid = valid[(valid >= 0) & (valid < len(lead_signal))]
            if len(valid) == 0:
                return None
            return float(np.mean(lead_signal[valid]))

        def safe_diff(start_key, end_key):
            starts = np.asarray(info.get(start_key, []), dtype=float)
            ends = np.asarray(info.get(end_key, []), dtype=float)
            n = min(len(starts), len(ends))
            diffs = []
            for i in range(n):
                if np.isfinite(starts[i]) and np.isfinite(ends[i]):
                    diff = (ends[i] - starts[i]) / float(sampling_rate)
                    if diff > 0 and np.isfinite(diff):
                        diffs.append(diff)
            return float(np.mean(diffs)) if diffs else None

        return {
            "mean_rr_interval": mean_rr,
            "heart_rate_bpm": hr_bpm,
            "p_wave_amplitude": safe_amp("ECG_P_Peaks"),
            "qrs_amplitude": safe_amp("ECG_Q_Peaks"),
            "t_wave_amplitude": safe_amp("ECG_T_Peaks"),
            "pr_interval": safe_diff("ECG_P_Onsets", "ECG_R_Onsets"),
            "qt_interval": safe_diff("ECG_Q_Peaks", "ECG_T_Offsets"),
            "qrs_duration": safe_diff("ECG_Q_Peaks", "ECG_S_Peaks"),
            "n_beats_detected": int(len(r_peaks))
        }
    except Exception:
        return None

def generate_shap_explanation(clf, features: dict, target_class_idx: int):
    """Generate genuine TreeSHAP values for the target class."""
    if clf is None or not hasattr(clf, "estimators_"):
        return None
    try:
        if target_class_idx >= len(clf.estimators_):
            return None
        estimator = clf.estimators_[target_class_idx]
        explainer = shap.TreeExplainer(estimator)
        X = pd.DataFrame([[features[col] for col in FEATURE_COLS]], columns=FEATURE_COLS)
        shap_values = explainer.shap_values(X)
        if isinstance(shap_values, list):
            if len(shap_values) < 2:
                return None
            sv = np.asarray(shap_values[1])
        else:
            sv = np.asarray(shap_values)
            if sv.ndim == 3:
                if sv.shape[-1] < 2:
                    return None
                sv = sv[:, :, 1]
        if sv.ndim != 2 or sv.shape[0] < 1:
            return None
        values = sv[0]
        if len(values) != len(FEATURE_COLS) or not np.all(np.isfinite(values)):
            return None
        return {feat: float(value) for feat, value in zip(FEATURE_COLS, values)}
    except Exception:
        return None


# ============================================================
# MC DROPOUT UNCERTAINTY
# ============================================================

N_MC_PASSES = 30
MC_DROPOUT_P = 0.3


class ResNet1D34_MCDropout(ResNet1D34):
    """
    ResNet1D34 with one Dropout layer before the final
    classification layer.

    The original federated checkpoints were trained without
    this Dropout layer. Since Dropout has no learnable
    parameters, the original checkpoint can still be loaded
    using strict=False.
    """

    def __init__(
        self,
        in_channels=12,
        num_classes=12,
        dropout_p=MC_DROPOUT_P
    ):
        super().__init__(
            in_channels=in_channels,
            num_classes=num_classes
        )

        self.mc_dropout = nn.Dropout(
            p=dropout_p
        )

    def forward(self, x):

        x = self.stem(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.global_pool(x)
        x = x.squeeze(-1)

        x = self.mc_dropout(x)

        x = self.fc(x)

        return x

# ============================================================
# ENABLE MC DROPOUT
# ============================================================

def enable_mc_dropout(model):
    """
    Keep the entire model in evaluation mode so that
    BatchNorm uses learned running statistics.

    Only Dropout layers are switched back to training mode
    so that stochastic predictions are generated.
    """

    model.eval()

    for module in model.modules():

        if isinstance(module, nn.Dropout):
            module.train()

# ============================================================
# SINGLE ECG MC DROPOUT UNCERTAINTY
# ============================================================

def compute_mc_dropout_uncertainty(
    model,
    signal,
    device,
    n_passes=N_MC_PASSES
):
    """
    Run multiple stochastic forward passes for one ECG.

    Returns:
        mean_prediction: shape (12,)
        uncertainty: shape (12,)
    """

    enable_mc_dropout(model)

    signal_tensor = torch.tensor(
        signal,
        dtype=torch.float32
    ).unsqueeze(0).to(device)

    predictions = []

    with torch.no_grad():

        for _ in range(n_passes):

            logits = model(signal_tensor)

            probabilities = torch.sigmoid(logits)

            predictions.append(
                probabilities.cpu().numpy()[0]
            )

    predictions = np.stack(
        predictions,
        axis=0
    )

    mean_prediction = predictions.mean(
        axis=0
    )

    uncertainty = predictions.std(
        axis=0
    )

    return mean_prediction, uncertainty



# ============================================================
# FASTAPI BACKEND
# ============================================================

app = FastAPI(
    title="CardioSight PRO Backend",
    description="Backend API for federated ECG analysis",
    version="1.0.0"
)


# ------------------------------------------------------------
# CORS
# ------------------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5500",
        "http://127.0.0.1:5500",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ------------------------------------------------------------
# DEVICE
# ------------------------------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ------------------------------------------------------------
# MODEL CACHE
# ------------------------------------------------------------

@lru_cache(maxsize=3)
def get_model(model_name: str):

    model_name = model_name.lower()

    if model_name == "fedavg":
        return load_fedavg_model(DEVICE)

    elif model_name == "fedprox":
        return load_fedprox_model(DEVICE)

    elif model_name in ["fedadam", "fedopt"]:
        return load_fedopt_model(DEVICE)

    else:
        raise ValueError(
            "Invalid model. Choose FedAvg, FedProx, or FedAdam."
        )


# ------------------------------------------------------------
# MC-DROPOUT MODEL CACHE
# ------------------------------------------------------------

@lru_cache(maxsize=3)
def get_mc_dropout_model(model_name: str):
    model_name = model_name.lower()
    if model_name == "fedavg":
        checkpoint_path = FEDAVG_MODEL_PATH
    elif model_name == "fedprox":
        checkpoint_path = FEDPROX_MODEL_PATH
    elif model_name in ["fedadam", "fedopt"]:
        checkpoint_path = FEDOPT_MODEL_PATH
    else:
        raise ValueError("Invalid model. Choose FedAvg, FedProx, or FedAdam.")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}")
    model = ResNet1D34_MCDropout(in_channels=12, num_classes=12, dropout_p=MC_DROPOUT_P)
    state_dict = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if not isinstance(state_dict, dict):
        raise RuntimeError(f"Invalid checkpoint format: {checkpoint_path}")
    model.load_state_dict(state_dict, strict=False)
    model.to(DEVICE)
    enable_mc_dropout(model)
    return model


# ------------------------------------------------------------
# FIDUCIAL RF CACHE
# ------------------------------------------------------------

@lru_cache(maxsize=1)
def get_fiducial_classifier():

    return load_fiducial_rf()


# ------------------------------------------------------------
# JSON CONVERSION HELPERS
# ------------------------------------------------------------

def numpy_to_list(values):

    return np.asarray(values).tolist()


def clean_float(value):

    value = float(value)

    if not np.isfinite(value):
        return None

    return value


# ------------------------------------------------------------
# HEALTH CHECK
# ------------------------------------------------------------

@app.get("/api/health")
def health_check():

    return {
        "status": "healthy",
        "device": str(DEVICE),
        "cuda_available": torch.cuda.is_available(),
        "models": {
            "fedavg": FEDAVG_MODEL_PATH.exists(),
            "fedprox": FEDPROX_MODEL_PATH.exists(),
            "fedadam": FEDOPT_MODEL_PATH.exists(),
            "fiducial_shap": FIDUCIAL_RF_PATH.exists()
        }
    }

@app.get("/api/config")
def config_check():
    """
    Deployment-safe configuration check.

    This intentionally returns only model availability and device
    information, not absolute filesystem paths.
    """
    return {
        "model_directory_configured": bool(ENV_MODEL_DIR),
        "model_directory_exists": MODEL_DIR.exists(),
        "models": {
            "fedavg": FEDAVG_MODEL_PATH.exists(),
            "fedprox": FEDPROX_MODEL_PATH.exists(),
            "fedadam": FEDOPT_MODEL_PATH.exists(),
            "fiducial_shap": FIDUCIAL_RF_PATH.exists()
        },
        "device": str(DEVICE)
    }


# ============================================================
# MAIN ECG ANALYSIS API
# ============================================================

@app.post("/api/analyze")
async def analyze_ecg(
    hea_file: UploadFile = File(...),
    mat_file: UploadFile = File(...),
    model: str = Form("fedavg"),
    source_hospital: str = Form("default")
):

    # --------------------------------------------------------
    # VALIDATE MODEL
    # --------------------------------------------------------

    model_name = model.lower()

    if model_name not in ["fedavg", "fedprox", "fedadam"]:
        raise HTTPException(
            status_code=400,
            detail=(
                "Invalid model. "
                "Choose 'fedavg', 'fedprox', or 'fedadam'."
            )
        )


    # --------------------------------------------------------
    # VALIDATE HOSPITAL
    # --------------------------------------------------------

    valid_hospitals = {
        "default",
        "chapman_shaoxing",
        "cpsc_2018",
        "georgia",
        "ningbo",
        "ptb-xl"
    }

    source_hospital = source_hospital.lower()

    if source_hospital not in valid_hospitals:

        raise HTTPException(
            status_code=400,
            detail={
                "message": "Invalid source hospital.",
                "allowed_values": sorted(valid_hospitals)
            }
        )


    # --------------------------------------------------------
    # VALIDATE FILE TYPES
    # --------------------------------------------------------

    hea_name = Path(hea_file.filename or "").name
    mat_name = Path(mat_file.filename or "").name

    if not hea_name.lower().endswith(".hea"):

        raise HTTPException(
            status_code=400,
            detail="First file must be a .hea file."
        )

    if not mat_name.lower().endswith(".mat"):

        raise HTTPException(
            status_code=400,
            detail="Second file must be a .mat file."
        )


    # --------------------------------------------------------
    # VALIDATE RECORD NAMES
    # --------------------------------------------------------

    hea_stem = Path(hea_name).stem
    mat_stem = Path(mat_name).stem

    if hea_stem != mat_stem:

        raise HTTPException(
            status_code=400,
            detail=(
                "The .hea and .mat files must belong "
                "to the same WFDB record."
            )
        )


    # --------------------------------------------------------
    # TEMPORARY DIRECTORY
    # --------------------------------------------------------

    temp_dir = tempfile.mkdtemp(
        prefix="cardiosight_"
    )

    try:

        hea_path = Path(temp_dir) / hea_name
        mat_path = Path(temp_dir) / mat_name


        # ----------------------------------------------------
        # SAVE UPLOADED FILES
        # ----------------------------------------------------

        with open(hea_path, "wb") as buffer:

            shutil.copyfileobj(
                hea_file.file,
                buffer
            )


        with open(mat_path, "wb") as buffer:

            shutil.copyfileobj(
                mat_file.file,
                buffer
            )


        # ----------------------------------------------------
        # READ HEADER
        # ----------------------------------------------------

        metadata = read_header_metadata(
            hea_path
        )


        # ----------------------------------------------------
        # VALIDATE ECG
        # ----------------------------------------------------

        if metadata["num_leads"] != 12:

            raise HTTPException(
                status_code=400,
                detail=(
                    f"Expected 12 ECG leads, "
                    f"but received {metadata['num_leads']}."
                )
            )


        if metadata["sampling_rate"] <= 0:

            raise HTTPException(
                status_code=400,
                detail="Invalid ECG sampling rate."
            )


        # ----------------------------------------------------
        # PREPROCESS ECG
        # ----------------------------------------------------

        processed_signal = process_one_record(
            file_path=str(hea_path.with_suffix("")),
            orig_fs=metadata["sampling_rate"],
            source_hospital=source_hospital
        )


        # ----------------------------------------------------
        # FINAL SHAPE VALIDATION
        # ----------------------------------------------------

        if processed_signal.shape != (12, 5000):

            raise HTTPException(
                status_code=500,
                detail=(
                    "Preprocessing produced an unexpected "
                    f"shape: {processed_signal.shape}. "
                    "Expected (12, 5000)."
                )
            )


        # ----------------------------------------------------
        # LOAD SELECTED FEDERATED MODEL
        # ----------------------------------------------------

        selected_model = get_model(
            model_name
        )


        # ----------------------------------------------------
        # STANDARD PREDICTION
        # ----------------------------------------------------

        logits, probabilities = predict(
            selected_model,
            processed_signal,
            DEVICE
        )


        # ----------------------------------------------------
        # SELECT PREDICTED CLASS
        # ----------------------------------------------------

        predicted_class_idx = int(
            np.argmax(probabilities)
        )

        predicted_class = CLASS_NAMES[
            predicted_class_idx
        ]

        predicted_probability = float(
            probabilities[predicted_class_idx]
        )


        # ----------------------------------------------------
        # ALL CLASS PROBABILITIES
        # ----------------------------------------------------

        class_probabilities = {}

        for class_name, probability in zip(
            CLASS_NAMES,
            probabilities
        ):

            class_probabilities[class_name] = (
                clean_float(probability)
            )


        # ====================================================
        # GRAD-CAM
        # ====================================================

        try:

            gradcam_heatmap = generate_gradcam(
                selected_model,
                processed_signal,
                predicted_class_idx,
                DEVICE
            )

            gradcam_result = {
                "available": True,
                "target_class": predicted_class,
                "target_class_index": predicted_class_idx,
                "values": numpy_to_list(
                    gradcam_heatmap
                )
            }

        except Exception as e:

            gradcam_result = {
                "available": False,
                "message": str(e)
            }


        # ====================================================
        # SHAP
        # ====================================================

        try:

            fiducial_features = (
                extract_fiducial_features_from_signal(
                    processed_signal
                )
            )

            if fiducial_features is None:

                shap_result = {
                    "available": False,
                    "message": (
                        "Unable to extract ECG "
                        "fiducial features."
                    )
                }

            else:

                feature_values = {
                    key: clean_float(value)
                    for key, value in
                    fiducial_features.items()
                    if key != "lead_used"
                }


                # --------------------------------------------
                # Check for invalid feature values
                # --------------------------------------------

                invalid_features = [
                    key
                    for key, value in feature_values.items()
                    if value is None
                ]


                if invalid_features:

                    shap_result = {
                        "available": False,
                        "message": (
                            "SHAP could not be generated "
                            "because some ECG features "
                            "could not be extracted.",
                        ),
                        "invalid_features": invalid_features
                    }

                else:

                    fiducial_clf = (
                        get_fiducial_classifier()
                    )

                    shap_values = (
                        generate_shap_explanation(
                            fiducial_clf,
                            fiducial_features,
                            predicted_class_idx
                        )
                    )

                    if shap_values is None:
                        shap_result = {
                            "available": False,
                            "message": (
                                "Genuine TreeSHAP values could not be generated for this ECG."
                            ),
                            "fiducials": feature_values
                        }
                    else:
                        shap_result = {
                            "available": True,
                            "target_class": predicted_class,
                            "features": {
                                key: clean_float(value)
                                for key, value in shap_values.items()
                            },
                            "fiducials": feature_values
                        }

        except Exception as e:

            shap_result = {
                "available": False,
                "message": str(e)
            }


        # ====================================================
        # MC-DROPOUT UNCERTAINTY
        # ====================================================

        try:

            mc_model = get_mc_dropout_model(
                model_name
            )

            mc_mean_prediction, uncertainty = (
                compute_mc_dropout_uncertainty(
                    mc_model,
                    processed_signal,
                    DEVICE
                )
            )


            # --------------------------------------------
            # Per-class MC results
            # --------------------------------------------

            mc_class_results = {}

            for class_name, mean_value, uncertainty_value in zip(
                CLASS_NAMES,
                mc_mean_prediction,
                uncertainty
            ):

                mc_class_results[class_name] = {
                    "mean_probability": clean_float(
                        mean_value
                    ),
                    "uncertainty_std": clean_float(
                        uncertainty_value
                    )
                }


            # --------------------------------------------
            # Predicted-class uncertainty
            # --------------------------------------------

            predicted_class_uncertainty = (
                float(
                    uncertainty[predicted_class_idx]
                )
            )


            uncertainty_result = {
                "available": True,
                "mc_passes": N_MC_PASSES,
                "dropout_probability": MC_DROPOUT_P,
                "predicted_class": predicted_class,
                "predicted_class_uncertainty": (
                    clean_float(
                        predicted_class_uncertainty
                    )
                ),
                "classes": mc_class_results
            }

        except Exception as e:

            uncertainty_result = {
                "available": False,
                "message": str(e)
            }


        # ====================================================
        # FINAL RESPONSE
        # ====================================================

        return {
            "success": True,

            "record": {
                "record_id": metadata["record_id"],
                "sampling_rate": metadata["sampling_rate"],
                "num_leads": metadata["num_leads"],
                "num_samples": metadata["num_samples"],
                "lead_names": metadata["lead_names"],
                "source_hospital": source_hospital
            },

            "model": {
                "selected": (
                    "FedAdam"
                    if model_name == "fedadam"
                    else model_name.upper()
                )
            },

            "ecg": {
                "sampling_rate": TARGET_FS,
                "lead_names": metadata["lead_names"],
                "signals": processed_signal.tolist()
            },

            "prediction": {
                "class": predicted_class,
                "class_index": predicted_class_idx,
                "probability": predicted_probability
            },

            "probabilities": class_probabilities,

            "gradcam": gradcam_result,

            "shap": shap_result,

            "uncertainty": uncertainty_result
        }


    except HTTPException:
        raise


    except FileNotFoundError as e:

        raise HTTPException(
            status_code=503,
            detail=str(e)
        )

    except RuntimeError as e:

        raise HTTPException(
            status_code=500,
            detail=f"ECG analysis failed: {str(e)}"
        )

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"ECG analysis failed: {str(e)}"
        )


    finally:

        shutil.rmtree(
            temp_dir,
            ignore_errors=True
        )


# ------------------------------------------------------------
# SERVE FRONTEND STATIC ASSETS
# ------------------------------------------------------------
FRONTEND_DIR = BASE_DIR / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")

    @app.get("/")
    def serve_frontend_root():
        index_file = FRONTEND_DIR / "index.html"
        if index_file.exists():
            return FileResponse(str(index_file))
        return {"service": "CardioSight PRO Backend Running"}
