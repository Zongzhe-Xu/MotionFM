#!/usr/bin/env python3
import argparse
from pathlib import Path
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.switch_backend("Agg")

# ---------- helpers ----------
def find_patients(folder: Path, raw_suffix: str, sg_suffix: str, fir_suffix: str):
    """
    Return mapping: patient -> { 'downsampled': Path|None, 'sg': Path|None, 'fir': Path|None }
    based on files named {patient}_{suffix}.parquet
    """
    mapping = {}
    for p in folder.glob("*.parquet"):
        stem = p.stem
        if "_" not in stem:
            continue
        patient, suffix = stem.split("_", 1)
        entry = mapping.setdefault(patient, {"downsampled": None, "sg": None, "fir": None})
        if suffix == raw_suffix:
            entry["downsampled"] = p
        elif suffix == sg_suffix:
            entry["sg"] = p
        elif suffix == fir_suffix:
            entry["fir"] = p
    return mapping

def load_xyz(path: Path):
    """Load x,y,z DataFrame; drop NaN/Inf rows for clean distributions."""
    df = pd.read_parquet(path)
    cols = [c for c in ["x", "y", "z"] if c in df.columns]
    if len(cols) < 3:
        return None
    df = df[["x", "y", "z"]].astype(float)
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    return df

def aggregate_frames(files):
    """Read a list[Path] into one DataFrame (x,y,z); skip missing/failed."""
    dfs = []
    for f in files:
        if f is None:
            continue
        try:
            df = load_xyz(f)
            if df is not None and len(df):
                dfs.append(df)
        except Exception as e:
            print(f"  ! Skipping {f.name}: {e}")
    if len(dfs) == 0:
        return None
    return pd.concat(dfs, ignore_index=True)

def qq_points(a: pd.Series, b: pd.Series, n=2000):
    """Empirical Q–Q points between two series (same quantile grid)."""
    a = a.dropna().to_numpy()
    b = b.dropna().to_numpy()
    if a.size == 0 or b.size == 0:
        return np.array([]), np.array([])
    q = np.linspace(0.01, 0.99, num=min(n, max(50, min(a.size, b.size))))
    return np.quantile(a, q), np.quantile(b, q)

def summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return metrics x/y/z as columns; rows=metrics."""
    stats = {
        "count": df.count(),
        "mean": df.mean(),
        "std": df.std(),
        "min": df.min(),
        "p25": df.quantile(0.25),
        "p50": df.quantile(0.50),
        "p75": df.quantile(0.75),
        "max": df.max(),
    }
    out = pd.DataFrame(stats).T
    return out[["x", "y", "z"]]

# ---------- plotting ----------
def plot_histograms(df_raw, df_sg, df_fir, out_png, bins=120):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
    triplets = [("X", "x"), ("Y", "y"), ("Z", "z")]
    for ax, (label, col) in zip(axes, triplets):
        ax.hist(df_raw[col], bins=bins, density=True, alpha=0.45, label="Downsampled", histtype="stepfilled")
        ax.hist(df_sg[col],  bins=bins, density=True, alpha=0.45, label="SG",            histtype="stepfilled")
        ax.hist(df_fir[col], bins=bins, density=True, alpha=0.45, label="FIR",           histtype="stepfilled")
        ax.set_title(f"{label}-axis")
        ax.set_xlabel("Acceleration")
        if label == "X":
            ax.set_ylabel("Density")
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle("Accelerometry Distributions (Aggregated)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

def plot_qq_grid(df_ref, df_cmp, cmp_name, out_png, n=2000):
    """Three Q–Q plots: cmp vs ref for x/y/z."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), sharex=True, sharey=True)
    for ax, col, label in zip(axes, ["x", "y", "z"], ["X", "Y", "Z"]):
        qa, qb = qq_points(df_ref[col], df_cmp[col], n=n)
        if qa.size and qb.size:
            ax.scatter(qa, qb, s=6, alpha=0.6)
            lims = [min(qa.min(), qb.min()), max(qa.max(), qb.max())]
            ax.plot(lims, lims, "k--", linewidth=1)  # 45° line
        ax.set_title(f"{label}-axis")
        ax.set_xlabel("Downsampled quantiles")
        if label == "X":
            ax.set_ylabel(f"{cmp_name} quantiles")
        ax.grid(True, alpha=0.25)
    fig.suptitle(f"Q–Q: {cmp_name} vs Downsampled (Aggregated)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser(description="Aggregate accelerometry distributions across patients: histograms, Q–Q, summary stats.")
    ap.add_argument("--folder", default="/scratch/besp/shared_data/capture24/distributions", help="Folder with *_downsampled/_sg/_fir.parquet")
    ap.add_argument("--out_dir", default="dist_plots_agg", help="Where to save outputs")
    ap.add_argument("--raw_suffix", default="downsampled", help="Suffix for raw/downsampled files")
    ap.add_argument("--sg_suffix", default="sg", help="Suffix for SG files")
    ap.add_argument("--fir_suffix", default="fir", help="Suffix for FIR files")
    ap.add_argument("--max_patients", type=int, default=10, help="Sample up to this many patients (0 = all)")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for patient sampling")
    args = ap.parse_args()

    folder = Path(args.folder)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mapping = find_patients(folder, args.raw_suffix, args.sg_suffix, args.fir_suffix)
    patients = sorted(mapping.keys())
    if args.max_patients > 0:
        random.seed(args.seed)
        patients = random.sample(patients, min(args.max_patients, len(patients)))

    # Build file lists
    raw_files = [mapping[p]["downsampled"] for p in patients if mapping[p]["downsampled"] is not None]
    sg_files  = [mapping[p]["sg"]          for p in patients if mapping[p]["sg"] is not None]
    fir_files = [mapping[p]["fir"]         for p in patients if mapping[p]["fir"] is not None]

    print(f"Patients found: {len(mapping)}; using {len(patients)}")
    print(f"Have files — downsampled: {len(raw_files)}, sg: {len(sg_files)}, fir: {len(fir_files)}")

    # Aggregate data
    df_raw = aggregate_frames(raw_files)
    df_sg  = aggregate_frames(sg_files)
    df_fir = aggregate_frames(fir_files)

    if df_raw is None or df_sg is None or df_fir is None:
        print("Nothing to aggregate — check suffixes and folder contents.")
        print("Example names detected in folder:")
        for p in sorted(folder.glob("*.parquet"))[:20]:
            print("  ", p.name)
        return

    # Plots
    plot_histograms(df_raw, df_sg, df_fir, out_dir / "histograms_agg.png")
    plot_qq_grid(df_raw, df_sg,  "SG",  out_dir / "qq_sg_agg.png")
    plot_qq_grid(df_raw, df_fir, "FIR", out_dir / "qq_fir_agg.png")

    # Summary stats CSV
    stats_raw = summary_table(df_raw)
    stats_sg  = summary_table(df_sg)
    stats_fir = summary_table(df_fir)
    wide = pd.concat(
        {"Downsampled": stats_raw, "SG": stats_sg, "FIR": stats_fir},
        axis=1
    )
    wide.to_csv(out_dir / "summary_stats_agg.csv")
    print(f"Saved: {out_dir/'histograms_agg.png'}, {out_dir/'qq_sg_agg.png'}, {out_dir/'qq_fir_agg.png'}, {out_dir/'summary_stats_agg.csv'}")

if __name__ == "__main__":
    main()