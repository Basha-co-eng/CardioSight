from pathlib import Path
from math import gcd
import numpy as np
from scipy.signal import resample_poly, butter, filtfilt, iirnotch
import wfdb
from backend.utils.constants import TARGET_FS, TARGET_SAMPLES


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


def get_notch_freq(source_hospital: str) -> float:
    return 60.0 if source_hospital.lower() == "georgia" else 50.0


def fix_length(
    signal: np.ndarray,
    target_samples: int = TARGET_SAMPLES
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


def process_one_record(
    file_path: str,
    orig_fs: int,
    source_hospital: str
) -> np.ndarray:
    """
    Load one ECG record and convert it to: (12, 5000)
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

    return signal.astype(np.float32)
