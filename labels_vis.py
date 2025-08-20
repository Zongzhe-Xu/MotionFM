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
        df["patient_id"] = file.name.replace(LABEL_SUFFIX, "")  # Add patient ID
        all_labels.append(df)

    if not all_labels:
        print("No label files found.")
        return

    full_df = pd.concat(all_labels, ignore_index=True)

        # Step: Activity distribution per patient
    print("Calculating activity distributions per patient...")

    # Count all activity occurrences per patient
    patient_activity_counts = full_df.groupby("patient_id")["activity_fractions"].apply(list)

    # Merge all JSON activity_fractions into a single Counter per patient
    patient_activity_distributions = {}
    for patient, act_lists in patient_activity_counts.items():
        counter = Counter()
        for af in act_lists:
            if isinstance(af, str):
                af = json.loads(af)
            counter.update(af)
        total = sum(counter.values())
        # Normalize to fractions
        patient_activity_distributions[patient] = {k: v / total for k, v in counter.items()}

    # Convert to dataframe for visualization
    activity_df = pd.DataFrame.from_dict(patient_activity_distributions, orient="index").fillna(0)

    # Save to CSV for inspection
    activity_df.to_csv(output_dir / "activity_distribution_by_patient.csv")

    # Plot stacked bar chart of top activities
    top_activities = activity_df.sum().sort_values(ascending=False).head(10).index
    activity_df_top = activity_df[top_activities]

    plt.figure(figsize=(12, 6))
    activity_df_top.sort_index().plot(kind="bar", stacked=True, figsize=(14, 6), width=0.8)
    plt.title("Top Activity Fractions by Patient")
    plt.ylabel("Fraction of Time")
    plt.xlabel("Patient ID")
    plt.xticks([], [])  # Hide xtick labels if too many
    plt.legend(title="Activity", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(output_dir / "activity_distribution_by_patient.png")
    plt.close()

    print("Saved per-patient activity distribution plots.")

    # ---------------------------
    # Top 10 Activities by Chunk
    # ---------------------------
    top_activities_chunk = full_df["top_activity"].value_counts().nlargest(10)
    print("Top 10 Activities (by chunk frequency):")
    print(top_activities_chunk)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=top_activities_chunk.index, y=top_activities_chunk.values)
    plt.xticks(rotation=45)
    plt.ylabel("Count of Chunks")
    plt.title("Top 10 Most Frequent Activities (by Chunk)")
    plt.tight_layout()
    plt.savefig(output_dir / "top_activities_by_chunk.png")
    plt.close()

    # ---------------------------
    # Top 10 Activities by Patient
    # ---------------------------
    patient_top_acts = (
        full_df.groupby("patient_id")["top_activity"]
        .agg(lambda x: x.value_counts().idxmax())
        .value_counts()
        .nlargest(10)
    )
    print("Top 10 Activities (by patient top-activity):")
    print(patient_top_acts)

    plt.figure(figsize=(10, 5))
    sns.barplot(x=patient_top_acts.index, y=patient_top_acts.values)
    plt.xticks(rotation=45)
    plt.ylabel("Number of Patients")
    plt.title("Top 10 Activities (by Patient Top Activity)")
    plt.tight_layout()
    plt.savefig(output_dir / "top_activities_by_patient.png")
    plt.close()

    # ---------------------------
    # Aggregate activity_fractions
    # ---------------------------
    all_fractions = Counter()
    for af in full_df["activity_fractions"].dropna():
        if isinstance(af, str):
            af = json.loads(af)
        all_fractions.update(af)

    total_chunks = len(full_df)
    for k in all_fractions:
        all_fractions[k] /= total_chunks

    most_common_acts = dict(sorted(all_fractions.items(), key=lambda x: x[1], reverse=True)[:10])
    plt.figure(figsize=(10, 5))
    sns.barplot(x=list(most_common_acts.keys()), y=list(most_common_acts.values()))
    plt.xticks(rotation=45)
    plt.ylabel("Fraction of Total Time")
    plt.title("Top 10 Activities by Aggregate Time")
    plt.tight_layout()
    plt.savefig(output_dir / "aggregate_activity_fractions.png")
    plt.close()

    # ---------------------------
    # Missingness heatmap
    # ---------------------------
    missing = full_df.isnull().mean()
    print("Missingness:")
    print(missing)

    plt.figure(figsize=(8, 3))
    sns.heatmap(missing.to_frame().T, cmap="Reds", annot=True)
    plt.title("Missingness Heatmap (Fraction Null)")
    plt.tight_layout()
    plt.savefig(output_dir / "missingness_heatmap.png")
    plt.close()

    # ---------------------------
    # Summary stats
    # ---------------------------
    summary_stats = {
        "top_activities_by_chunk": top_activities_chunk.to_dict(),
        "top_activities_by_patient": patient_top_acts.to_dict(),
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