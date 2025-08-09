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
DEFAULT_CHUNK_MIN = 15

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
    chunk_minutes: int,
):
    """
    - Downsample to target_frequency_hz via striding (no anti-aliasing).
    - Split into chunk_minutes windows.
    - Name each output with the CHUNK start time.
    """
    # Read only what we need
    ddf = dd.read_parquet(input_file, columns=["time", "x", "y", "z", "annotation"])

    # Compute downsample ratio (assumes original Hz is correct)
    if original_frequency_hz % target_frequency_hz != 0:
        raise ValueError(
            f"original_frequency_hz ({original_frequency_hz}) must be an integer multiple "
            f"of target_frequency_hz ({target_frequency_hz})."
        )
    downsample_ratio = original_frequency_hz // target_frequency_hz

    # Downsample within each partition by striding
    ddf_ds = ddf.map_partitions(lambda part: part.iloc[::downsample_ratio])

    # Materialize (pandas) AFTER downsampling
    df = ddf_ds.compute()

    # Parse annotations -> activity, MET (keep 'time' for chunk naming)
    parsed = df["annotation"].apply(parse_annotation)
    df["activity"] = parsed.apply(lambda x: x[0])
    df["MET"] = parsed.apply(lambda x: x[1])

    # Reorder (keep 'time' until after chunking)
    cols_order = ["time", "x", "y", "z", "activity", "MET"]
    df = df[[c for c in cols_order if c in df.columns]]

    # Chunking math
    rows_per_chunk = int(chunk_minutes * 60 * target_frequency_hz)
    total_chunks = int(np.ceil(len(df) / rows_per_chunk)) if rows_per_chunk > 0 else 0

    original_name = input_file.stem  # e.g., "P010"
    output_dir.mkdir(parents=True, exist_ok=True)

    for i in range(total_chunks):
        start = i * rows_per_chunk
        end = (i + 1) * rows_per_chunk
        chunk_df = df.iloc[start:end]

        if chunk_df.empty:
            continue

        # Use the CHUNK's own start time in the filename
        try:
            chunk_start_time = pd.to_datetime(chunk_df["time"].iloc[0])
        except Exception:
            # If time parsing fails, fall back to index-based timestamp
            chunk_start_time = pd.to_datetime("1970-01-01")
        chunk_start_str = chunk_start_time.strftime("%Y%m%dT%H%M%S")

        # Drop 'time' in saved output (as requested)
        save_df = chunk_df.drop(columns=["time", "annotation"], errors="ignore")[["x", "y", "z", "activity", "MET"]]

        # Example name: capture24_P010_start_20161215T020400_20Hz_chunk_0.parquet
        filename = f"capture24_{original_name}_{chunk_start_str}.parquet"
        (output_dir / filename).parent.mkdir(parents=True, exist_ok=True)
        save_df.to_parquet(output_dir / filename, index=False)

def preprocess_all(input_dir: str, output_dir: str, orig_hz: int, target_hz: int, chunk_min: int):
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
            chunk_minutes=chunk_min,
        )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Capture24 parquets into 15-min 20Hz chunks named by CHUNK start time.")
    parser.add_argument("--input_dir", required=True, help="Directory with raw Capture24 parquets")
    parser.add_argument("--output_dir", required=True, help="Directory to write processed chunk parquets")
    parser.add_argument("--orig_hz", type=int, default=DEFAULT_ORIG_HZ, help="Original sampling frequency (Hz)")
    parser.add_argument("--target_hz", type=int, default=DEFAULT_TARGET_HZ, help="Target sampling frequency (Hz)")
    parser.add_argument("--chunk_min", type=int, default=DEFAULT_CHUNK_MIN, help="Chunk length (minutes)")
    args = parser.parse_args()

    preprocess_all(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        orig_hz=args.orig_hz,
        target_hz=args.target_hz,
        chunk_min=args.chunk_min,
    )