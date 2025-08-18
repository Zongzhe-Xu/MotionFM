import argparse
import json
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
def parse_annotation(annotation_str: str):
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

def activity_from_annotation(annotation_str: str) -> str:
    """
    Extract activity label only from the annotation string.
    Example: 'Activity Walk; MET 3.2' -> 'Walk'
    """
    if not isinstance(annotation_str, str) or annotation_str.strip() == "":
        return "none"
    parts = annotation_str.split(";")
    if not parts:
        return "none"
    first = parts[0].strip()
    if " " in first:
        return first.split(" ", 1)[-1].strip() or "none"
    return first or "none"

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

def _rle_strings(series: pd.Series) -> list:
    """
    Run-length encode a string series.
    Returns list of dicts: [{"label": <str>, "n": <int>}, ...]
    """
    out = []
    if series.empty:
        return out
    vals = series.astype("string").fillna("none").to_numpy()
    prev = vals[0]
    count = 1
    for v in vals[1:]:
        if v == prev:
            count += 1
        else:
            out.append({"label": str(prev), "n": int(count)})
            prev = v
            count = 1
    out.append({"label": str(prev), "n": int(count)})
    return out

def summarize_chunk_labels(
    chunk_df: pd.DataFrame,
    target_hz: int,
    time_col: str,
    act_col: str,
    patient_id: str,
    chunk_filename: str,
):
    """
    Build activity-only labels for a chunk (no METs).
    Includes:
      - top_activity, top_activity_fraction
      - activities_present (JSON list)
      - activity_fractions (JSON dict)
      - activities_rle (JSON list of {label, n})
    """
    n = len(chunk_df)
    if n == 0:
        return None

    # Timing for bookkeeping
    try:
        start_time = pd.to_datetime(chunk_df[time_col].iloc[0])
        end_time   = pd.to_datetime(chunk_df[time_col].iloc[-1])
    except Exception:
        start_time = pd.to_datetime("1970-01-01")
        end_time   = start_time + pd.to_timedelta(n / target_hz, unit="s")

    # Activity distribution
    act_series = chunk_df[act_col].astype("string").fillna("none")
    counts = act_series.value_counts(dropna=False)
    if counts.sum() == 0:
        top_activity = "none"
        top_frac = 0.0
        activities_present = []
        activity_fractions = {}
    else:
        fractions = (counts / counts.sum()).sort_values(ascending=False)
        top_activity = fractions.index[0]
        top_frac = float(fractions.iloc[0])
        activities_present = sorted(list(set(act_series.tolist())))
        activity_fractions = {str(k): float(v) for k, v in fractions.to_dict().items()}

    # Full activity sequence (compact) via RLE
    activities_rle = _rle_strings(act_series)

    return {
        "chunk_filename": chunk_filename,
        "patient_id": patient_id,
        "start_time": pd.Timestamp(start_time),
        "end_time": pd.Timestamp(end_time),
        "duration_s": float(n) / float(target_hz),
        "top_activity": top_activity,
        "top_activity_fraction": top_frac,
        "activities_present": json.dumps(activities_present),
        "activity_fractions": json.dumps(activity_fractions),
        "activities_rle": json.dumps(activities_rle),
    }

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
      - Slice into windows (window_minutes, stride_minutes)
      - Save chunks with just x,y,z (and time if keep_time=True)
      - Write per-patient labels parquet with activity-only metadata (no METs)
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

    # 4) Extract activity label (no METs in output chunks)
    df["activity"] = df["annotation"].apply(activity_from_annotation)

    # 5) Keep columns & order (retain 'time' for naming before optional drop)
    cols_order = ["time", "x", "y", "z", "activity"]
    df = df[[c for c in cols_order if c in df.columns]]

    # 6) Windowing
    rows_per_window = int(window_minutes * 60 * target_frequency_hz)
    step_rows = int(stride_minutes * 60 * target_frequency_hz)
    if rows_per_window <= 0 or step_rows <= 0:
        raise ValueError("window_minutes and stride_minutes must be > 0.")
    max_start = max(0, len(df) - rows_per_window)
    starts = range(0, max_start + 1, step_rows)

    original_name = input_file.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    # Collect labels rows for this patient file
    labels_rows = []

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

        # Columns to save in the chunk (unchanged: x,y,z plus optional time)
        save_cols = (["time"] if keep_time else []) + ["x", "y", "z"]
        save_df = chunk_df.drop(columns=["annotation"], errors="ignore")[save_cols]

        # Write parquet for the chunk
        filename = f"capture24_{original_name}_{chunk_start_str}.parquet"
        (output_dir / filename).parent.mkdir(parents=True, exist_ok=True)
        save_df.to_parquet(output_dir / filename, index=False)

        # Build labels row for this chunk
        labels_row = summarize_chunk_labels(
            chunk_df=chunk_df,
            target_hz=target_frequency_hz,
            time_col="time",
            act_col="activity",
            patient_id=original_name,
            chunk_filename=filename,
        )
        if labels_row is not None:
            labels_rows.append(labels_row)

    # Write per-patient labels parquet
    if labels_rows:
        labels_df = pd.DataFrame(labels_rows)
        labels_path = output_dir / f"{original_name}_labels.parquet"
        labels_df.to_parquet(labels_path, index=False)

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
        description="Butterworth low-pass → downsample → hourly windows with 15-min stride; "
                    "write x/y/z chunks (+optional time) and per-chunk activity labels parquet."
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