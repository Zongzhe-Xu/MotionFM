import argparse
from pathlib import Path
import dask.dataframe as dd
import pandas as pd
import numpy as np
from tqdm import tqdm


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


def process_labels_file(input_file: Path, output_dir: Path, orig_hz: int = 100, target_hz: int = 10):
    """
    Extract time and activity from annotation, downsample to target_hz, and write to a per-patient labels file.
    """
    if orig_hz % target_hz != 0:
        raise ValueError("Original Hz must be divisible by target Hz.")

    ratio = orig_hz // target_hz
    ddf = dd.read_parquet(input_file, columns=["time", "annotation"])

    # Downsample by striding
    ddf = ddf.map_partitions(lambda df: df.iloc[::ratio])

    # Extract activity label
    ddf["activity"] = ddf["annotation"].apply(activity_from_annotation, meta=("annotation", "str"))

    # Drop annotation
    ddf = ddf.drop(columns=["annotation"])

    # Save output
    df = ddf.compute()
    output_file = output_dir / f"{input_file.stem}_labels_{target_hz}Hz.parquet"
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_file, index=False)
    print(f"Saved {output_file.name}")


def batch_process_labels(input_dir: Path, output_dir: Path, orig_hz: int, target_hz: int, start_index: int = 0):
    files = sorted(input_dir.glob("*.parquet"))
    files = files[start_index:]
    print(f"Found {len(files)} files to process from index {start_index}.")

    for idx, f in enumerate(tqdm(files, desc="Extracting labels")):
        try:
            process_labels_file(f, output_dir, orig_hz=orig_hz, target_hz=target_hz)
        except Exception as e:
            print(f"Failed to process {f.name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract and downsample activity labels from Capture24 parquets.")
    parser.add_argument("--input_dir", required=True, help="Directory with raw Capture24 parquets")
    parser.add_argument("--output_dir", required=True, help="Directory to save label-only parquets")
    parser.add_argument("--orig_hz", type=int, default=100, help="Original frequency (Hz)")
    parser.add_argument("--target_hz", type=int, default=10, help="Target downsampled frequency (Hz)")
    parser.add_argument("--start_index", type=int, default=0, help="Start index of files to process")

    args = parser.parse_args()

    batch_process_labels(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        orig_hz=args.orig_hz,
        target_hz=args.target_hz,
        start_index=args.start_index
    )