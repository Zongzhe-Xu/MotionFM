import os
import dask.dataframe as dd
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse

# ----------------------------
# Defaults (overridable by CLI)
# ----------------------------
DEFAULT_ORIG_HZ = 100
DEFAULT_TARGET_HZ = 20
DEFAULT_WINDOW_MIN = 60      # 1-hour windows
DEFAULT_STRIDE_MIN = 15      # 15-minute stride
DEFAULT_KEEP_TIME = False    # drop 'time' in outputs by default

def parse_annotation(annotation_str):
    """Return (activity, MET) parsed from a semicolon-delimited annotation string."""
    if not isinstance(annotation_str, str) or annotation_str.strip() == "":
        return "", np.nan

    parts = annotation_str.split(";")
    # activity: take text after first space of first part (adjust if your format differs)
    activity = parts[0].split(" ", 1)[-1].strip() if parts else ""
    met = np.nan
    for p in parts:
        if "MET" in p:
            try:
                met = float(p.split("MET")[-1].strip())
            except Exception:
                met = np.nan
    return activity, met

def preprocess_file(
    input_file: Path,
    output_dir: Path,
    original_frequency_hz: int,
    target_frequency_hz: int,
    window_minutes: int,
    stride_minutes: int,
    keep_time: bool = DEFAULT_KEEP_TIME,
):
    """
    Pipeline for one patient/file:
      - Downsample to target_frequency_hz via striding (no anti-aliasing).
      - Create overlapping windows of `window_minutes` with step `stride_minutes`.
      - Name each output with the CHUNK start time.
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

    # Downsample within each partition by simple decimation
    ddf_ds = ddf.map_partitions(lambda part: part.iloc[::downsample_ratio])

    # Materialize (pandas) AFTER downsampling
    df = ddf_ds.compute()

    # Parse annotations -> activity, MET (keep 'time' for chunk naming)
    parsed = df["annotation"].apply(parse_annotation)
    df["activity"] = parsed.apply(lambda x: x[0])
    df["MET"] = parsed.apply(lambda x: x[1])

    # Ensure ordering and 'time' present before windowing
    cols_order = ["time", "x", "y", "z", "activity", "MET"]
    df = df[[c for c in cols_order if c in df.columns]]

    # Window/stride math
    rows_per_window = int(window_minutes * 60 * target_frequency_hz)
    step_rows = int(stride_minutes * 60 * target_frequency_hz)
    if rows_per_window <= 0 or step_rows <= 0:
        raise ValueError("window_minutes and stride_minutes must be > 0.")

    # Indices for overlapping windows: start=0..(len - window) step stride
    max_start = max(0, len(df) - rows_per_window)
    starts = range(0, max_start + 1, step_rows)

    original_name = input_file.stem  # e.g., "P010"
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, start in enumerate(starts):
        end = start + rows_per_window
        if end > len(df):
            # skip incomplete trailing window; change to `end = len(df)` if you want partials
            break

        chunk_df = df.iloc[start:end]
        if chunk_df.empty:
            continue

        # Use the CHUNK's own start time in the filename
        try:
            chunk_start_time = pd.to_datetime(chunk_df["time"].iloc[0])
        except Exception:
            # If time parsing fails, fall back to Unix epoch
            chunk_start_time = pd.to_datetime("1970-01-01")
        chunk_start_str = chunk_start_time.strftime("%Y%m%dT%H%M%S")

        # Choose columns to save
        if keep_time:
            save_cols = ["time", "x", "y", "z", "activity", "MET"]
        else:
            save_cols = ["x", "y", "z", "activity", "MET"]

        save_df = chunk_df.drop(columns=["annotation"], errors="ignore")[save_cols]

        # Example: capture24_P010_start_20161215T020400_20Hz_win60m_stride15m_idx000.parquet
        filename = (
            f"capture24_{original_name}_start_{chunk_start_str}_"
            f"{target_frequency_hz}Hz_win{window_minutes}m_stride{stride_minutes}m_idx{str(i).zfill(3)}.parquet"
        )
        (output_dir / filename).parent.mkdir(parents=True, exist_ok=True)
        save_df.to_parquet(output_dir / filename, index=False)

def preprocess_all(
    input_dir: str,
    output_dir: str,
    orig_hz: int,
    target_hz: int,
    window_min: int,
    stride_min: int,
    keep_time: bool = DEFAULT_KEEP_TIME,
):
    in_dir = Path(input_dir)
    out_dir = Path(output_dir)
    files = sorted(in_dir.glob("*.parquet"))
    print(f"Found {len(files)} files in {in_dir}")

    for f in tqdm(files, desc="Preprocessing"):
        preprocess_file(
            input_file=f,
            output_dir=out_dir,
            original_frequency_hz=orig_hz,
            target_frequency_hz=target_hz,
            window_minutes=window_min,
            stride_minutes=stride_min,
            keep_time=keep_time,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess Capture24 parquets into sliding windows (hourly with 15-min stride by default), named by CHUNK start time."
    )
    parser.add_argument("--input_dir", required=True, help="Directory with raw Capture24 parquets")
    parser.add_argument("--output_dir", required=True, help="Directory to write processed chunk parquets")
    parser.add_argument("--orig_hz", type=int, default=DEFAULT_ORIG_HZ, help="Original sampling frequency (Hz)")
    parser.add_argument("--target_hz", type=int, default=DEFAULT_TARGET_HZ, help="Target sampling frequency (Hz)")
    parser.add_argument("--window_min", type=int, default=DEFAULT_WINDOW_MIN, help="Window length (minutes), default 60")
    parser.add_argument("--stride_min", type=int, default=DEFAULT_STRIDE_MIN, help="Stride length (minutes), default 15")
    parser.add_argument("--keep_time", action="store_true", help="Include 'time' column in saved files")
    args = parser.parse_args()

    preprocess_all(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        orig_hz=args.orig_hz,
        target_hz=args.target_hz,
        window_min=args.window_min,
        stride_min=args.stride_min,
        keep_time=args.keep_time,
    )