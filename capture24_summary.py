import os
import dask.dataframe as dd
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

plt.switch_backend("Agg")

def get_sampling_rate_from_filename(name: str) -> int:
    try:
        return int(name.split("_")[-1].replace("Hz.parquet", ""))
    except Exception:
        print(f"Failed to parse Hz from: {name}")
        return 10

def get_start_time_from_filename(name: str) -> datetime:
    try:
        timestamp_str = name.split("_")[2]
        return datetime.strptime(timestamp_str, "%Y%m%dT%H%M%S")
    except Exception:
        print(f"Failed to parse start time from: {name}")
        return None

def process_file_dask(fpath: Path):
    ddf = dd.read_parquet(fpath)[['x', 'y', 'z', 'activity', 'MET']]
    sampling_freq = get_sampling_rate_from_filename(fpath.name)
    ddf = ddf.reset_index(drop=True)
    ddf['row_index'] = dd.from_array(np.arange(0, len(ddf)))
    ddf['relative_second'] = (ddf['row_index'] / sampling_freq).astype(int)
    ddf['relative_minute'] = (ddf['row_index'] / (sampling_freq * 60)).astype(int)
    # ENMO -- Euclidean Norm Minus One
    ddf['norm'] = ((ddf['x']**2 + ddf['y']**2 + ddf['z']**2) - 1).map_partitions(np.sqrt)
    return ddf, sampling_freq

def aggregate_to_minute_bin(ddf):
    return ddf.groupby('relative_minute').agg({
        'x': 'mean',
        'y': 'mean',
        'z': 'mean',
        'norm': 'mean',
        'MET': 'mean',
        'activity': lambda x: x.value_counts().index[0]
    }).reset_index()

def main(input_dir: str):
    input_dir = Path(input_dir)
    files = sorted(input_dir.glob("*.parquet"))
    print(f"Found {len(files)} files.")

    all_patient_dfs_plot = []
    all_patient_dfs_stats = []
    activity_heatmap_matrix = []
    met_all = []

    for f in tqdm(files):
        ddf, sampling_freq = process_file_dask(f)

        # For plotting (1-minute binning)
        ddf_plot = aggregate_to_minute_bin(ddf)
        df_plot = ddf_plot.compute()
        df_plot['patient'] = f.stem
        all_patient_dfs_plot.append(df_plot)

        # For statistics (raw data)
        df_stats = ddf[['x', 'y', 'z', 'norm', 'MET']].compute()
        all_patient_dfs_stats.append(df_stats)
        met_all.append(df_stats['MET'].dropna().values)

        # Activity per minute
        activity_counts = df_plot.groupby('relative_minute')['activity'].value_counts().unstack(fill_value=0)
        activity_heatmap_matrix.append(activity_counts)

    # === PLOTTING DATA ===
    df_all_plot = pd.concat(all_patient_dfs_plot, ignore_index=True)

    def make_heatmap(data_col, title):
        pivot = df_all_plot.pivot(index='patient', columns='relative_minute', values=data_col)
        plt.figure(figsize=(18, 6))
        sns.heatmap(pivot, cmap='viridis', cbar_kws={'label': data_col})
        plt.title(f"24-Hour Aligned Heatmap: {data_col} (1-min bins)")
        plt.xlabel("Minutes since 00:00:00")
        plt.ylabel("Patient")
        plt.tight_layout()
        plt.savefig(f"{data_col}_heatmap.png")
        plt.close()

    for axis in ['x', 'y', 'z', 'norm']:
        make_heatmap(axis, axis)

    # === ACTIVITY LABEL HEATMAP ===
    activity_combined = pd.concat(activity_heatmap_matrix).groupby('relative_minute').sum().fillna(0)
    bin_edges = [0, 30, 60, 90, 120, 150, float('inf')]
    bin_labels = ['0-30', '30-60', '60-90', '90-120', '120-150', '150+']
    total_activity_freq = activity_combined.sum(axis=1)
    freq_bins = pd.cut(total_activity_freq, bins=bin_edges, labels=bin_labels)
    bin_counts = freq_bins.value_counts(sort=False)

    plt.figure(figsize=(10, 2))
    sns.heatmap([bin_counts.values], cmap="rainbow", xticklabels=bin_labels, yticklabels=["Frequency"])
    plt.title("Activity Label Frequency vs Time (1-min bins)")
    plt.tight_layout()
    plt.savefig("activity_label_frequency.png")
    plt.close()

    # === SUMMARY STATISTICS ===
    df_all_stats = pd.concat(all_patient_dfs_stats, ignore_index=True)
    print("\nSummary statistics (raw data):")
    for col in ['x', 'y', 'z', 'norm']:
        arr = df_all_stats[col].dropna()
        print(f"{col.upper()}: mean={arr.mean():.3f}, std={arr.std():.3f}, min={arr.min():.3f}, max={arr.max():.3f}")

    met_vals = np.concatenate(met_all)
    print(f"MET: mean={np.nanmean(met_vals):.3f}, std={np.nanstd(met_vals):.3f}, min={np.nanmin(met_vals):.3f}, max={np.nanmax(met_vals):.3f}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Dask-based summary + 1-min heatmaps for Capture24 processed parquets")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory with processed Capture24 .parquet files")
    args = parser.parse_args()

    main(args.input_dir)