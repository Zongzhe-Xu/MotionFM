import argparse
from pathlib import Path

import dask.dataframe as dd
import pandas as pd
import numpy as np
from scipy.signal import savgol_filter, firwin, filtfilt

# ----------------------------
# Helpers
# ----------------------------
def sg_filter(part: pd.DataFrame, orig_hz, window_ms, polyorder):
    n = len(part)
    if n == 0:
        return part
    win = max(3, int(round(window_ms * orig_hz / 1000.0)))
    if win % 2 == 0:
        win += 1
    if win >= n:
        return part
    po = min(polyorder, win - 1)
    for col in ("x", "y", "z"):
        if col in part.columns:
            vals = pd.to_numeric(part[col], errors="coerce").to_numpy()
            if np.all(np.isnan(vals)):
                continue
            s = pd.Series(vals).interpolate(limit_direction="both").to_numpy()
            try:
                part[col] = savgol_filter(s, win, po, mode="interp")
            except Exception:
                pass
    return part

def fir_filter(part: pd.DataFrame, taps: np.ndarray):
    if taps is None or len(part) == 0:
        return part
    padlen = 3 * (len(taps) - 1)
    if len(part) <= padlen:
        return part
    for col in ("x", "y", "z"):
        if col in part.columns:
            vals = pd.to_numeric(part[col], errors="coerce").to_numpy()
            if np.all(np.isnan(vals)):
                continue
            s = pd.Series(vals).interpolate(limit_direction="both").to_numpy()
            try:
                part[col] = filtfilt(taps, 1.0, s, method="pad")
            except Exception:
                pass
    return part

def design_fir(orig_hz, cutoff_hz, numtaps):
    nyq = 0.5 * orig_hz
    wc = cutoff_hz / nyq
    return firwin(numtaps, wc, window="hamming", pass_zero="lowpass")

def decimate(part: pd.DataFrame, decim):
    return part.iloc[::decim]

# ----------------------------
# Core processing
# ----------------------------
def process_file(fpath, out_dir, orig_hz, target_hz, sg_window_ms, sg_polyorder, fir_cut_hz, fir_numtaps, keep_time):
    decim = orig_hz // target_hz
    read_cols = ["time", "x", "y", "z"] if keep_time else ["x", "y", "z"]

    ddf = dd.read_parquet(fpath, columns=read_cols)

    # FIR taps precomputed
    taps = design_fir(orig_hz, fir_cut_hz, fir_numtaps)

    # 1. Downsample only
    downsampled = ddf.map_partitions(lambda p: decimate(p, decim))
    save_df(downsampled, fpath, out_dir, "downsampled", keep_time)

    # 2. SG + downsample
    sg_df = ddf.map_partitions(sg_filter, orig_hz=orig_hz, window_ms=sg_window_ms, polyorder=sg_polyorder)
    sg_ds = sg_df.map_partitions(lambda p: decimate(p, decim))
    save_df(sg_ds, fpath, out_dir, "sg", keep_time)

    # 3. FIR + downsample
    fir_df = ddf.map_partitions(lambda p: fir_filter(p, taps))
    fir_ds = fir_df.map_partitions(lambda p: decimate(p, decim))
    save_df(fir_ds, fpath, out_dir, "fir", keep_time)

def save_df(ddf, fpath, out_dir, suffix, keep_time):
    df = ddf.compute()
    cols = (["time"] if keep_time and "time" in df.columns else []) + ["x", "y", "z"]
    df = df[cols]
    patient = fpath.stem
    out_path = out_dir / f"{patient}_{suffix}.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Compare downsample-only vs SG vs FIR filtering.")
    ap.add_argument("--input_dir", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--orig_hz", type=int, default=100)
    ap.add_argument("--target_hz", type=int, default=10)
    ap.add_argument("--sg_window_ms", type=int, default=250)
    ap.add_argument("--sg_polyorder", type=int, default=3)
    ap.add_argument("--fir_cut_hz", type=float, default=4.0)
    ap.add_argument("--fir_numtaps", type=int, default=121)
    ap.add_argument("--keep_time", action="store_true", default=False)
    ap.add_argument("--start_index", type=int, default=0)
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    files = sorted(in_dir.glob("*.parquet"))[args.start_index:]

    for i, f in enumerate(files, start=args.start_index):
        print(f"[{i}] Processing {f.name}")
        process_file(
            f, out_dir,
            args.orig_hz, args.target_hz,
            args.sg_window_ms, args.sg_polyorder,
            args.fir_cut_hz, args.fir_numtaps,
            args.keep_time
        )