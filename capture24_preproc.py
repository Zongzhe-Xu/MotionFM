import os
import dask.dataframe as dd
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

ORIGINAL_FREQUENCY_HZ = 100
TARGET_FREQUENCY_HZ = 20
CHUNK_DURATION_MINUTES = 15
ROWS_PER_CHUNK = CHUNK_DURATION_MINUTES * 60 * TARGET_FREQUENCY_HZ
DOWNSAMPLE_RATIO = int(ORIGINAL_FREQUENCY_HZ / TARGET_FREQUENCY_HZ)

def parse_annotation(annotation_str):
    if not isinstance(annotation_str, str) or annotation_str.strip() == "":
        return "", np.nan

    parts = annotation_str.split(";")
    activity = parts[0].split(" ", 1)[-1].strip() if parts else ""
    met = None
    for part in parts:
        if "MET" in part:
            try:
                met = float(part.split("MET")[-1].strip())
            except ValueError:
                met = np.nan
    return activity, met

def preprocess_file(input_file: Path, output_dir: Path):
    df = dd.read_parquet(input_file)

    # Get first timestamp for naming
    start_time = pd.to_datetime(df.head(1)['time'].iloc[0])
    start_time_str = start_time.strftime("%Y%m%dT%H%M%S")

    # Downsample
    df = df.map_partitions(lambda part: part.iloc[::DOWNSAMPLE_RATIO])

    # Compute once after downsampling
    df = df.compute()

    # Parse annotation
    parsed = df['annotation'].apply(parse_annotation)
    df['activity'] = parsed.apply(lambda x: x[0])
    df['MET'] = parsed.apply(lambda x: x[1])

    # Drop unnecessary columns
    df = df.drop(columns=['annotation', 'time'], errors='ignore')

    # Reorder columns
    df = df[['x', 'y', 'z', 'activity', 'MET']]

    # Chunk into 15-minute slices (N rows per chunk)
    total_chunks = int(np.ceil(len(df) / ROWS_PER_CHUNK))
    original_name = input_file.stem  # e.g. "P010"

    for i in range(total_chunks):
        chunk_df = df.iloc[i * ROWS_PER_CHUNK : (i + 1) * ROWS_PER_CHUNK]

        filename = f"{original_name}_start_{start_time_str}_{TARGET_FREQUENCY_HZ}Hz_chunk_{i}.parquet"
        output_path = output_dir / filename

        chunk_df.to_parquet(output_path, index=False)

def preprocess_all(input_dir: str, output_dir: str):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.parquet"))
    print(f"Found {len(files)} files in {input_dir}")

    for f in tqdm(files, desc="Preprocessing"):
        preprocess_file(f, output_dir)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess Capture24 parquet files using Dask")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with raw Capture24 parquets")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save processed parquets")

    args = parser.parse_args()

    preprocess_all(
        input_dir=args.input_dir,
        output_dir=args.output_dir
    )