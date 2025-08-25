import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import timedelta


def load_label_dictionary(label_dict_path: Path) -> dict:
    df = pd.read_csv(label_dict_path)
    return dict(zip(df["annotation"], df["label:WillettsSpecific2018"]))


def map_annotation_to_label(annotation_str: str, label_map: dict) -> str:
    if not isinstance(annotation_str, str) or annotation_str.strip() == "":
        return "none"
    base_annotation = annotation_str.split(";")[0].strip()
    return label_map.get(base_annotation, "none")


def process_raw_file(
    input_file: Path,
    label_map: dict,
    output_dir: Path,
    chunk_length_seconds: int = 3600,
    stride_seconds: int = 900,
    sampling_rate: int = 100,
):
    df = pd.read_parquet(input_file, columns=["time", "annotation"])

    # Map annotations to WillettsSpecific2018 labels
    df["activity"] = df["annotation"].apply(lambda x: map_annotation_to_label(x, label_map))
    df["time"] = pd.to_datetime(df["time"])
    df = df.drop(columns=["annotation"])

    # Set time as index
    df = df.set_index("time")
    df = df.sort_index()

    # Chunking into 1-hour windows, 15-min stride
    chunks = []
    start_time = df.index.min()
    end_time = df.index.max()
    delta = timedelta(seconds=chunk_length_seconds)
    stride = timedelta(seconds=stride_seconds)

    unique_activities = sorted(set(label_map.values()) | {"none"})

    while start_time + delta <= end_time:
        chunk = df.loc[start_time : start_time + delta]
        if not chunk.empty:
            activity_counts = chunk["activity"].value_counts()
            top_activity = activity_counts.idxmax()

            row = {
                "start_time": start_time,
                "end_time": start_time + delta,
                "top_activity": top_activity,
            }

            for act in unique_activities:
                row[act] = int(act in activity_counts)

            chunks.append(row)
        start_time += stride

    # Output
    output_df = pd.DataFrame(chunks)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{input_file.stem}_labels.parquet"
    output_df.to_parquet(output_file, index=False)
    print(f"Saved {output_file.name} with {len(output_df)} chunks")


def batch_process(
    input_dir: Path,
    output_dir: Path,
    label_dict_path: Path,
    chunk_length_seconds: int = 3600,
    stride_seconds: int = 900,
    sampling_rate: int = 100,
):
    label_map = load_label_dictionary(label_dict_path)
    files = sorted(input_dir.glob("*.parquet"))
    print(f"Found {len(files)} raw parquet files")

    for file in tqdm(files, desc="Processing files"):
        try:
            process_raw_file(
                input_file=file,
                label_map=label_map,
                output_dir=output_dir,
                chunk_length_seconds=chunk_length_seconds,
                stride_seconds=stride_seconds,
                sampling_rate=sampling_rate,
            )
        except Exception as e:
            print(f"Failed to process {file.name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Chunk and summarize raw annotation labels.")
    parser.add_argument("--input_dir", required=True, help="Directory with raw Capture24 parquet files")
    parser.add_argument("--output_dir", required=True, help="Directory to save chunked label summaries")
    parser.add_argument("--label_dict", required=True, help="CSV file mapping annotation to WillettsSpecific2018 labels")
    parser.add_argument("--chunk_length", type=int, default=3600, help="Chunk length in seconds (default: 3600)")
    parser.add_argument("--stride", type=int, default=900, help="Stride in seconds between chunks (default: 900)")
    parser.add_argument("--sampling_rate", type=int, default=100, help="Sampling rate of raw data (Hz)")

    args = parser.parse_args()

    batch_process(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        label_dict_path=Path(args.label_dict),
        chunk_length_seconds=args.chunk_length,
        stride_seconds=args.stride,
        sampling_rate=args.sampling_rate,
    )