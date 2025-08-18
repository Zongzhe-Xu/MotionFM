# capture24_preproc_butter.py

import argparse
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
import numpy as np
from tqdm import tqdm
from scipy.signal import butter, sosfiltfilt

# ----------------------------
# Defaults (overridable by CLI)
# ----------------------------
DEFAULT_ORIG_HZ = 100
DEFAULT_TARGET_HZ = 20
DEFAULT_WINDOW_MIN = 60      # 1-hour windows
DEFAULT_STRIDE_MIN = 15      # 15-minute stride
DEFAULT_KEEP_TIME = False    # drop 'time' in outputs by default

# Butterworth filter defaults
DEFAULT_BUTTER_ORDER = 4
DEFAULT_BUTTER_CUTOFF_HZ = 20.0  # low-pass cutoff at original Hz

# ----------------------------
# Helpers
# ----------------------------
def parse_annotation(annotation_str):
    """Return (activity, MET) parsed from a semicolon-delimited annotation string."""
    if not isinstance(annotation_str, str) or annotation_str.strip() == "":
        return "", np.nan
    parts = annotation_str.split(";")
    activity = parts[0].split(" ", 1)[-1].strip() if parts else ""
    met = np.nan
    for p in parts:
        if "MET" in p:
            try:
                met = float(p.split("MET")[-1].strip())
            except Exception:
                met = np.nan
    return activity, met

def butter_sos(orig_hz: int, cutoff_hz: float, order: int):
    nyq = 0.5 * orig_hz
    wn = cutoff_hz / nyq
    if wn >= 1.0:
        # Degenerate: no need to filter
        return None
    # Use SOS for numerical stability
    return butter(order, wn, btype="low", output="sos")

def butter_lowpass_part(part: pd.DataFrame, sos, cols=("x", "y", "z")) -> pd.DataFrame:
    """Apply zero-phase Butterworth low-pass (sosfiltfilt) per partition."""
    if sos is None or len(part) == 0:
        return part
    for c in cols:
        if c in part.columns:
            vals = pd.to_numeric(part[c], errors="coerce").to_numpy()
            if np.all(np.isnan(vals)):
                continue
            # simple interpolation to handle NaNs
            s = pd.Series(vals).interpolate(limit_direction="both").to_numpy()
            try:
                part[c] = sosfiltfilt(sos, s, axis=0)
            except Exception:
                # if filtfilt fails due to too-short partition, leave as-is
                pass
    return part

# ----------------------------
# Core
# ----------------------------
def preprocess_file(
    input_file: Path,
    output_dir: Path,
    original_frequency_hz: int,
    target_frequency_hz: int,
    window_minutes: int,
    stride_minutes: int,
    keep_time: bool,
    butter_order: int,
    butter_cutoff_hz: float,
):
    """
    For one patient/file:
      - Read ("time","x","y","z","annotation")
      - Butterworth low-pass (order=butter_order, cutoff=butter_cutoff_hz) at ORIGINAL Hz (zero-phase)
      - Downsample to target_frequency_hz via striding
      - Compute ENMO = max(sqrt(x^2+y^2+z^2) - 1, 0)
      - Sliding windows: window_minutes with stride_minutes
      - Filename includes the chunk start time
    """
    # Read minimal columns
    ddf = dd.read_parquet(input_file, columns=["time", "x", "y", "z", "annotation"])

    # Validate downsample ratio
    if original_frequency_hz % target_frequency_hz != 0:
        raise ValueError(
            f"original_frequency_hz ({original_frequency_hz}) must be an integer multiple "
            f"of target_frequency_hz ({target_frequency_hz})."
        )
    downsample_ratio = original_frequency_hz // target_frequency_hz

    # Precompute SOS once
    sos = butter_sos(original_frequency_hz, butter_cutoff_hz, butter_order)

    # 1) Low-pass at original Hz (per partition)
    ddf_filt = ddf.map_partitions(
        butter_lowpass_part,
        sos=sos,
        cols=("x", "y", "z"),
        meta=ddf,
    )

    # 2) Decimate by striding
    ddf_ds = ddf_filt.map_partitions(lambda part: part.iloc[::downsample_ratio])

    # 3) Compute to pandas
    df = ddf_ds.compute()

    # 4) Parse annotations
    parsed = df["annotation"].apply(parse_annotation)
    df["activity"] = parsed.apply(lambda x: x[0])
    df["MET"] = parsed.apply(lambda x: x[1])

    # 5) ENMO (Euclidean Norm Minus One), non-negative
    df["ENMO"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2) - 1.0
    df["ENMO"] = df["ENMO"].clip(lower=0.0)

    # 6) Keep columns & order (retain 'time' for naming before optional drop)
    cols_order = ["time", "x", "y", "z", "ENMO", "activity", "MET"]
    df = df[[c for c in cols_order if c in df.columns]]

    # 7) Windowing
    rows_per_window = int(window_minutes * 60 * target_frequency_hz)
    step_rows = int(stride_minutes * 60 * target_frequency_hz)
    if rows_per_window <= 0 or step_rows <= 0:
        raise ValueError("window_minutes and stride_minutes must be > 0.")
    max_start = max(0, len(df) - rows_per_window)
    starts = range(0, max_start + 1, step_rows)

    original_name = input_file.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    for start in starts:
        end = start + rows_per_window
        if end > len(df):
            break
        chunk_df = df.iloc[start:end]
        if chunk_df.empty:
            continue

        # Use chunk start time for filename
        try:
            chunk_start_time = pd.to_datetime(chunk_df["time"].iloc[0])
        except Exception:
            chunk_start_time = pd.to_datetime("1970-01-01")
        chunk_start_str = chunk_start_time.strftime("%Y%m%dT%H%M%S")

        # Choose columns to save
        save_cols = (["time"] if keep_time else []) + ["x", "y", "z", "ENMO", "activity", "MET"]
        save_df = chunk_df.drop(columns=["annotation"], errors="ignore")[save_cols]

        # Write parquet
        filename = f"capture24_{original_name}_{chunk_start_str}.parquet"
        (output_dir / filename).parent.mkdir(parents=True, exist_ok=True)
        save_df.to_parquet(output_dir / filename, index=False)

def preprocess_all(
    input_dir: str,
    output_dir: str,
    orig_hz: int,
    target_hz: int,
    window_min: int,
    stride_min: int,
    keep_time: bool,
    butter_order: int,
    butter_cutoff_hz: float,
    start_index: int,
):
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    files = sorted(in_dir.glob("*.parquet"))
    print(f"Found {len(files)} files in {in_dir}")

    # Start from a specific index (helps with 30-min job limits)
    files = files[start_index:]
    print(f"Starting from file index {start_index} ({len(files)} files to process)")

    for idx, f in enumerate(tqdm(files, desc="Preprocessing", unit="file")):
        global_idx = start_index + idx
        print(f"[{global_idx}] Processing {f.name}")
        preprocess_file(
            input_file=f,
            output_dir=out_dir,
            original_frequency_hz=orig_hz,
            target_frequency_hz=target_hz,
            window_minutes=window_min,
            stride_minutes=stride_min,
            keep_time=keep_time,
            butter_order=butter_order,
            butter_cutoff_hz=butter_cutoff_hz,
        )

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess Capture24 parquets into sliding windows (hourly, 15-min stride), "
                    "apply 4th-order Butterworth low-pass before downsampling."
    )
    parser.add_argument("--input_dir", required=True, help="Directory with raw Capture24 parquets")
    parser.add_argument("--output_dir", required=True, help="Directory to write processed chunk parquets")
    parser.add_argument("--orig_hz", type=int, default=DEFAULT_ORIG_HZ, help="Original sampling frequency (Hz)")
    parser.add_argument("--target_hz", type=int, default=DEFAULT_TARGET_HZ, help="Target sampling frequency (Hz)")
    parser.add_argument("--window_min", type=int, default=DEFAULT_WINDOW_MIN, help="Window length (minutes)")
    parser.add_argument("--stride_min", type=int, default=DEFAULT_STRIDE_MIN, help="Stride length (minutes)")
    parser.add_argument("--keep_time", action="store_true", help="Include 'time' column in saved files")
    parser.add_argument("--start_index", type=int, default=0, help="Index of file to start at in sorted list")

    # Butterworth params
    parser.add_argument("--butter_order", type=int, default=DEFAULT_BUTTER_ORDER, help="Butterworth filter order")
    parser.add_argument("--butter_cut_hz", type=float, default=DEFAULT_BUTTER_CUTOFF_HZ, help="Butterworth cutoff frequency (Hz)")

    args = parser.parse_args()

    preprocess_all(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        orig_hz=args.orig_hz,
        target_hz=args.target_hz,
        window_min=args.window_min,
        stride_min=args.stride_min,
        keep_time=args.keep_time,
        butter_order=args.butter_order,
        butter_cutoff_hz=args.butter_cut_hz,
        start_index=args.start_index,
    )