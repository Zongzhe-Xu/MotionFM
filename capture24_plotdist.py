import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from pathlib import Path
import random

N_PATIENTS = 10  # Change this to sample more/less patients

data_dir = Path("/scratch/besp/shared_data/capture24/distributions")
out_dir = Path("./dist_plots_agg")
out_dir.mkdir(exist_ok=True)

def load_xyz(file):
    return pd.read_parquet(file)[["x", "y", "z"]]

# Identify unique patient IDs
files = list(data_dir.glob("*_raw.parquet"))
patient_ids = sorted(set(f.stem.split("_")[0] for f in files))
sampled_ids = random.sample(patient_ids, min(N_PATIENTS, len(patient_ids)))

raw_all, sg_all, fir_all = [], [], []

for pid in sampled_ids:
    raw_all.append(load_xyz(data_dir / f"{pid}_raw.parquet"))
    sg_all.append(load_xyz(data_dir / f"{pid}_sg.parquet"))
    fir_all.append(load_xyz(data_dir / f"{pid}_fir.parquet"))

# Concatenate across patients
raw_df = pd.concat(raw_all, ignore_index=True)
sg_df = pd.concat(sg_all, ignore_index=True)
fir_df = pd.concat(fir_all, ignore_index=True)

def plot_histograms(raw_df, sg_df, fir_df, save_path, bins=100):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    methods = [("Raw", raw_df), ("SG", sg_df), ("FIR", fir_df)]
    colors = ["black", "blue", "red"]

    for ax, axis in zip(axes, ["x", "y", "z"]):
        for (label, df), color in zip(methods, colors):
            ax.hist(df[axis], bins=bins, alpha=0.4, label=label, color=color, density=True)
        ax.set_title(f"{axis.upper()} distribution")
        ax.legend()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_qq(raw_df, other_df, method_name, save_path):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, axis in zip(axes, ["x", "y", "z"]):
        stats.probplot(other_df[axis], dist="norm", plot=ax)
        ax.set_title(f"Q-Q Plot: {method_name} ({axis.upper()})")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def summary_stats(df):
    return {
        "mean": df.mean(),
        "std": df.std(),
        "skew": df.skew(),
        "kurt": df.kurtosis(),
        "min": df.min(),
        "max": df.max(),
        "p5": df.quantile(0.05),
        "p50": df.quantile(0.5),
        "p95": df.quantile(0.95)
    }

# Histograms
plot_histograms(raw_df, sg_df, fir_df, out_dir / "histograms_agg.png")

# Q–Q plots
plot_qq(raw_df, sg_df, "SG", out_dir / "qq_sg_agg.png")
plot_qq(raw_df, fir_df, "FIR", out_dir / "qq_fir_agg.png")

# Summary stats
stats_raw = summary_stats(raw_df)
stats_sg  = summary_stats(sg_df)
stats_fir = summary_stats(fir_df)

pd.DataFrame({
    "Raw": stats_raw,
    "SG": stats_sg,
    "FIR": stats_fir
}).to_csv(out_dir / "summary_stats_agg.csv")