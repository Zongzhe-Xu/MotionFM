import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import json
from collections import Counter

LABEL_SUFFIX = "_labels.parquet"

def analyze_labels(label_dir: Path, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    all_labels = []

    # Step 1: Aggregate all label rows
    for file in tqdm(label_dir.glob(f"*{LABEL_SUFFIX}"), desc="Loading label files"):
        df = pd.read_parquet(file)
        all_labels.append(df)

    if not all_labels:
        print("No label files found.")
        return

    full_df = pd.concat(all_labels, ignore_index=True)

    # Step 2: Top 10 Activities by Frequency (chunk-wise)
    top_activities = full_df["top_activity"].value_counts().nlargest(10)
    print("Top 10 Activities (by chunk frequency):")
    print(top_activities)

    # Plot 1: Bar chart of most frequent top activities
    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_activities.index, y=top_activities.values)
    plt.xticks(rotation=45)
    plt.ylabel("Count of Chunks")
    plt.title("Top 10 Most Frequent Activities (as Top Activity)")
    plt.tight_layout()
    plt.savefig(output_dir / "top_activities.png")
    plt.close()

    # Step 3: Aggregate activity_fractions
    all_fractions = Counter()
    for af in full_df["activity_fractions"].dropna():
        if isinstance(af, str):
            af = json.loads(af)
        all_fractions.update(af)

    # Normalize by number of chunks
    total_chunks = len(full_df)
    for k in all_fractions:
        all_fractions[k] /= total_chunks

    # Plot 2: Overall Activity Time Fractions
    most_common_acts = dict(sorted(all_fractions.items(), key=lambda x: x[1], reverse=True)[:10])
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(most_common_acts.keys()), y=list(most_common_acts.values()))
    plt.xticks(rotation=45)
    plt.ylabel("Fraction of Total Time")
    plt.title("Top 10 Activities by Aggregate Time")
    plt.tight_layout()
    plt.savefig(output_dir / "aggregate_activity_fractions.png")
    plt.close()

    # Step 4: Check for missing label data
    missing = full_df.isnull().mean()

    print("Missingness:")
    print(missing)

    # Plot 3: Missingness heatmap
    plt.figure(figsize=(8, 3))
    sns.heatmap(missing.to_frame().T, cmap="Reds", annot=True)
    plt.title("Missingness Heatmap (Fraction Null)")
    plt.tight_layout()
    plt.savefig(output_dir / "missingness_heatmap.png")
    plt.close()

    # Save summary statistics
    summary_stats = {
        "top_activities": top_activities.to_dict(),
        "activity_time_fractions": most_common_acts,
        "missingness": missing.to_dict(),
    }
    with open(output_dir / "summary_stats.json", "w") as f:
        json.dump(summary_stats, f, indent=2)

    print("Analysis complete. Outputs saved to:", output_dir)

# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze label distributions and missingness.")
    parser.add_argument("--label_dir", required=True, help="Directory containing *_labels.parquet files")
    parser.add_argument("--output_dir", required=True, help="Directory to save plots and stats")

    args = parser.parse_args()
    analyze_labels(Path(args.label_dir), Path(args.output_dir))