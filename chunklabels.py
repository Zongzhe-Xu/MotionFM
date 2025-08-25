import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import json

CHUNK_SIZE = 60 * 60 * 10   # 1 hour at 10 Hz
STRIDE_SIZE = 15 * 60 * 10  # 15 minutes at 10 Hz


def load_label_mapping(csv_path):
    df = pd.read_csv(csv_path)
    return dict(zip(df['label:annotation'], df['label:WillettsSpecific2018']))


def one_hot_encode_activities(chunk_activities, all_activities):
    flags = {act: 0 for act in all_activities}
    for act in chunk_activities:
        if act in flags:
            flags[act] = 1
    return flags


def process_single_file(file_path, output_dir, label_map, all_activities):
    df = pd.read_parquet(file_path, columns=["time", "annotation"])
    df["activity"] = df["annotation"].map(label_map).fillna("none")

    data = []
    for start in range(0, len(df) - CHUNK_SIZE + 1, STRIDE_SIZE):
        chunk = df.iloc[start:start + CHUNK_SIZE]
        top_act = chunk["activity"].mode().iloc[0] if not chunk.empty else "none"
        unique_acts = chunk["activity"].unique()
        binary_flags = one_hot_encode_activities(unique_acts, all_activities)
        data.append({
            "patient_id": file_path.stem,
            "start_index": start,
            "most_common_activity": top_act,
            **binary_flags
        })

    out_df = pd.DataFrame(data)
    output_file = output_dir / f"{file_path.stem}_labels.parquet"
    out_df.to_parquet(output_file, index=False)
    print(f"Saved: {output_file.name}")


def batch_process(input_dir, output_dir, label_dict_csv, start_index=0):
    label_map = load_label_mapping(label_dict_csv)
    all_activities = sorted(set(label_map.values()) | {"none"})

    input_files = sorted(input_dir.glob("*.parquet"))
    files_to_process = input_files[start_index:]
    print(f"Found {len(files_to_process)} files to process starting from index {start_index}")

    for file in tqdm(files_to_process, desc="Processing files"):
        try:
            process_single_file(file, output_dir, label_map, all_activities)
        except Exception as e:
            print(f"Error processing {file.name}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate per-chunk activity labels from raw parquets.")
    parser.add_argument("--input_dir", required=True, help="Path to raw parquet files")
    parser.add_argument("--output_dir", required=True, help="Path to save chunked label files")
    parser.add_argument("--label_dict", required=True, help="Path to annotation-label-dictionary.csv")
    parser.add_argument("--start_index", type=int, default=0, help="Start index for file processing")

    args = parser.parse_args()
    batch_process(Path(args.input_dir), Path(args.output_dir), Path(args.label_dict), args.start_index)