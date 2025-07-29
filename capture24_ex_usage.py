import torch
from capture24_dataloader import Capture24Dataset  # or your file name
import os

# Path to the directory containing the .parquet file
parquet_dir = "../../scratch/besp/shared_data/capture24/capture24_parquets"   

# Optionally, just use one file for testing
single_file = sorted(os.listdir(parquet_dir))[0]
single_file_path = os.path.join(parquet_dir, single_file)

# Create dataset with only that one file
dataset = Capture24Dataset(data_dir=parquet_dir, context_len=1000, downsampling_ratio=1.0)

# Load the first sample
sample = dataset[0]

print("Keys in sample:", sample.keys())
print("accel_data shape:", sample['accel_data'].shape)
print("timestamps shape:", sample['timestamps'].shape)
print("annotation:", sample['annotation'])

# Example: visualize first 100 points of x-axis
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(sample['accel_data'][0, :100].numpy())
plt.title('First 100 X-axis Accel Points')
plt.xlabel('Time Step')
plt.ylabel('Acceleration (X)')
plt.grid(True)
plt.show()