import os
import dask.dataframe as dd
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

def parse_annotation(annotation_str):
    """Split annotation into activity label and MET value."""
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

def preprocess_file(input_file: Path, output_dir: Path, downsampling_ratio: int):
    # Read using Dask
    df = dd.read_parquet(input_file)

    # Downsample
    df = df.iloc[::downsampling_ratio].compute()

    # Parse annotations
    parsed = df['annotation'].apply(parse_annotation)
    df['activity'] = parsed.apply(lambda x: x[0])
    df['MET'] = parsed.apply(lambda x: x[1])

    # Reorder columns
    cols = ['time', 'x', 'y', 'z', 'activity', 'MET']
    df = df[cols]

    # Save to new path
    output_file = output_dir / input_file.name
    df.to_parquet(output_file, index=False)
    return output_file

def preprocess_all(input_dir: str, output_dir: str, downsampling_ratio: int = 2):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(input_dir.glob("*.parquet"))

    print(f"Found {len(files)} files in {input_dir}")
    for f in tqdm(files, desc="Preprocessing"):
        preprocess_file(f, output_dir, downsampling_ratio)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess Capture24 parquet files using Dask")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with raw Capture24 parquets")
    parser.add_argument("--output_dir", type=str, required=True, help="Where to save processed parquets")
    parser.add_argument("--downsample", type=int, default=2, help="Downsampling ratio (e.g., 2 keeps every 2nd row)")

    args = parser.parse_args()

    preprocess_all(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        downsampling_ratio=args.downsample
    )