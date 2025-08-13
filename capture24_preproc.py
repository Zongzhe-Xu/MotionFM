import os
import dask.dataframe as dd
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
from scipy.signal import savgol_filter  # <-- SG filter

# ----------------------------
# Defaults (overridable by CLI)
# ----------------------------
DEFAULT_ORIG_HZ = 100
DEFAULT_TARGET_HZ = 20
DEFAULT_WINDOW_MIN = 60      # 1-hour windows
DEFAULT_STRIDE_MIN = 15      # 15-minute stride
DEFAULT_KEEP_TIME = False    # drop 'time' in outputs by default
DEFAULT_CHUNK_MIN = 15 

# Savitzky–Golay defaults (window in milliseconds at ORIGINAL Hz)
DEFAULT_SG_WINDOW_MS = 250   # ~0.25 s window at 100 Hz -> ~25 samples
DEFAULT_SG_POLYORDER = 3

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

def _savgol_params(orig_hz: int, window_ms: int, polyorder: int, n_rows: int):
    """
    Convert window_ms to a valid odd window length in samples, adapt to small partitions.
    Ensures window_length >= polyorder+2 (really needs > polyorder and odd).
    """
    # ms -> samples at original Hz
    win = max(3, int(round(window_ms * orig_hz / 1000.0)))
    if win % 2 == 0:
        win += 1  # must be odd

    # cannot exceed partition length; keep odd
    if n_rows <= 2:
        return None, None  # too small to filter
    if win > n_rows:
        win = n_rows - 1 if (n_rows - 1) % 2 == 1 else n_rows - 2
        if win < 3:
            return None, None

    po = min(polyorder, win - 1)
    if po < 1:
        po = 1
    return win, po

def savgol_smooth_part(part: pd.DataFrame, orig_hz: int, window_ms: int, polyorder: int) -> pd.DataFrame:
    """
    Apply SG smoothing to ['x','y','z'] in a pandas partition at original sampling rate.
    Uses 'interp' mode to reduce edge artifacts within the partition.
    """
    n = len(part)
    if n == 0:
        return part

    win, po = _savgol_params(orig_hz, window_ms, polyorder, n)
    if win is None:
        # too few samples to filter; just return
        return part

    for col in ['x', 'y', 'z']:
        if col in part.columns:
            vals = pd.to_numeric(part[col], errors='coerce').to_numpy()
            # Handle all-NaN or mostly-NaN segments gracefully
            if np.all(np.isnan(vals)):
                continue
            # Replace NaNs with nearest valid values for filtering (simple fill)
            # (Optional) you could do more sophisticated imputation
            s = pd.Series(vals).interpolate(limit_direction='both').to_numpy()
            try:
                part[col] = savgol_filter(s, window_length=win, polyorder=po, mode='interp')
            except Exception:
                # Fallback: leave as-is if SG fails
                pass
    return part

def preprocess_file(
    input_file: Path,
    output_dir: Path,
    original_frequency_hz: int,
    target_frequency_hz: int,
    window_minutes: int,
    stride_minutes: int,
    keep_time: bool,
    sg_window_ms: int,
    sg_polyorder: int,
):
    """
    Pipeline for one patient/file:
      - Read ("time","x","y","z","annotation")
      - Apply Savitzky–Golay smoothing at ORIGINAL Hz
      - Downsample to target_frequency_hz via striding (no explicit anti-aliasing beyond SG)
      - Create overlapping windows of `window_minutes` with step `stride_minutes`
      - Name each output with the CHUNK start time
    """
    # Read only necessary columns
    ddf = dd.read_parquet(input_file, columns=["time", "x", "y", "z", "annotation"])

    # Validate downsample ratio
    if original_frequency_hz % target_frequency_hz != 0:
        raise ValueError(
            f"original_frequency_hz ({original_frequency_hz}) must be an integer multiple "
            f"of target_frequency_hz ({target_frequency_hz})."
        )
    downsample_ratio = original_frequency_hz // target_frequency_hz

    # 1) Savitzky–Golay smoothing per partition at original Hz
    ddf_smooth = ddf.map_partitions(
        savgol_smooth_part,
        orig_hz=original_frequency_hz,
        window_ms=sg_window_ms,
        polyorder=sg_polyorder,
        meta=ddf  # meta helps Dask know schema
    )

    # 2) Downsample (decimate) by simple striding AFTER smoothing
    ddf_ds = ddf_smooth.map_partitions(lambda part: part.iloc[::downsample_ratio])

    # 3) Materialize (pandas) AFTER downsampling
    df = ddf_ds.compute()

    # 4) Parse annotations -> activity, MET (keep 'time' for chunk naming)
    parsed = df["annotation"].apply(parse_annotation)
    df["activity"] = parsed.apply(lambda x: x[0])
    df["MET"] = parsed.apply(lambda x: x[1])

    # 5) Ensure ordering and 'time' present before windowing
    cols_order = ["time", "x", "y", "z", "activity", "MET"]
    df = df[[c for c in cols_order if c in df.columns]]

    # 6) Window/stride math (overlapping windows)
    rows_per_window = int(window_minutes * 60 * target_frequency_hz)
    step_rows = int(stride_minutes * 60 * target_frequency_hz)
    if rows_per_window <= 0 or step_rows <= 0:
        raise ValueError("window_minutes and stride_minutes must be > 0.")

    max_start = max(0, len(df) - rows_per_window)
    starts = range(0, max_start + 1, step_rows)

    original_name = input_file.stem  # e.g., "P010"
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, start in enumerate(starts):
        end = start + rows_per_window
        if end > len(df):
            break  # skip incomplete trailing window; change if you want partials

        chunk_df = df.iloc[start:end]
        if chunk_df.empty:
            continue

        # Use the CHUNK's own start time in the filename
        try:
            chunk_start_time = pd.to_datetime(chunk_df["time"].iloc[0])
        except Exception:
            chunk_start_time = pd.to_datetime("1970-01-01")
        chunk_start_str = chunk_start_time.strftime("%Y%m%dT%H%M%S")

        # Choose columns to save
        save_cols = ["time", "x", "y", "z", "activity", "MET"] if keep_time else ["x", "y", "z", "activity", "MET"]
        save_df = chunk_df.drop(columns=["annotation"], errors="ignore")[save_cols]

        # Example filename
        filename = (
            f"capture24_{original_name}_{chunk_start_str}.parquet"
        )
        (output_dir / filename).parent.mkdir(parents=True, exist_ok=True)
        save_df.to_parquet(output_dir / filename, index=False)

def preprocess_all(input_dir: str, output_dir: str, orig_hz: int, target_hz: int, chunk_min: int, start_index: int):
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    files = sorted(in_dir.glob("*.parquet"))
    print(f"Found {len(files)} files in {in_dir}")

    # Only process starting from start_index
    files = files[start_index:]
    print(f"Starting from file index {start_index} ({len(files)} files to process)")

    for f in tqdm(files, desc="Preprocessing"):
        preprocess_file(
            input_file=f,
            output_dir=out_dir,
            original_frequency_hz=orig_hz,
            target_frequency_hz=target_hz,
            chunk_minutes=chunk_min,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Capture24 parquets into chunks.")
    parser.add_argument("--input_dir", required=True, help="Directory with raw Capture24 parquets")
    parser.add_argument("--output_dir", required=True, help="Directory to write processed chunk parquets")
    parser.add_argument("--orig_hz", type=int, default=DEFAULT_ORIG_HZ, help="Original sampling frequency (Hz)")
    parser.add_argument("--target_hz", type=int, default=DEFAULT_TARGET_HZ, help="Target sampling frequency (Hz)")
    parser.add_argument("--chunk_min", type=int, default=DEFAULT_CHUNK_MIN, help="Chunk length (minutes)")
    parser.add_argument("--start_index", type=int, default=0, help="Index of file to start at in sorted list")
    args = parser.parse_args()

    preprocess_all(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        orig_hz=args.orig_hz,
        target_hz=args.target_hz,
        chunk_min=args.chunk_min,
        start_index=args.start_index,
    )