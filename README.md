# Accelerometer Data Loading System

A scalable data loading and sampling system for accelerometer data stored in a directory structure with patient folders containing multiple CSV files. Each CSV file represents raw 3-axis accelerometer data for a specific hour over continuous 7 days, with columns for timestamps and x, y, z acceleration.

## Features

- **Scalable Data Loading**: Efficiently handles large datasets with thousands of patients
- **Controllable Sampling Weights**: Sample with different weights for patients, hours, and days
- **Cross-File Sequences**: Extract continuous sequences that span multiple files
- **Metadata Preservation**: Maintain timestamp and patient_id information for each batch element
- **PyTorch Integration**: Native PyTorch DataLoader with custom collate functions
- **NumPy Alternative**: NumPy-only version for environments without PyTorch
- **Caching**: Configurable caching for improved performance
- **Multi-processing**: Support for parallel data loading

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. For PyTorch installation (if needed):
```bash
# CPU only
pip install torch

# With CUDA support (adjust version as needed)
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

## Quick Start

```python
from dataloader import AccelDataLoader

# Create data loader with uniform sampling
loader = AccelDataLoader(
    data_dir="./patient_data",
    context_len=1000,  # 1000 samples per sequence
    batch_size=32,
    num_workers=4
)

# Get a batch
batch = loader.get_batch()
print(f"Batch shapes: {batch['accel_data'].shape}")  # (32, 3, 1000)
```

## Data Structure

The system expects data organized as follows:

```
patient_data/
├── 73557/
│   ├── GT3XPLUS-AccelerationCalibrated-2x5x0.NEO1G22051089.2000-01-04-16-30-00-000-P0000.sensor.csv
│   ├── GT3XPLUS-AccelerationCalibrated-2x5x0.NEO1G22051089.2000-01-04-17-00-00-000-P0000.sensor.csv
│   └── ...
├── 73559/
│   ├── GT3XPLUS-AccelerationCalibrated-2x5x0.NEO1G50016094.2000-01-08-12-30-00-000-P0000.sensor.csv
│   └── ...
└── ...
```

Each CSV file should contain:
- `HEADER_TIMESTAMP`: Timestamp column
- `X`, `Y`, `Z`: 3-axis accelerometer data

## Usage Examples

### Uniform Sampling
```python
loader = AccelDataLoader(
    data_dir="./patient_data",
    context_len=1000,
    batch_size=32
)
```

### Weighted Sampling by Time
```python
from dataloader import create_hour_weights, create_day_weights

# Favor morning hours and weekdays
hour_weights = create_hour_weights(
    peak_hours=[6, 7, 8, 9, 10],  # Morning hours
    peak_weight=3.0,
    base_weight=1.0
)

day_weights = create_day_weights(
    weekend_weight=0.5,  # Lower weight for weekends
    weekday_weight=1.0
)

loader = AccelDataLoader(
    data_dir="./patient_data",
    context_len=1000,
    batch_size=32,
    hour_weights=hour_weights,
    day_weights=day_weights
)
```

### Patient-Specific Sampling
```python
patient_weights = {
    "73557": 2.0,  # Higher weight
    "73559": 1.0   # Normal weight
}

loader = AccelDataLoader(
    data_dir="./patient_data",
    context_len=1000,
    batch_size=32,
    patient_weights=patient_weights
)
```

### Dynamic Weight Updates
```python
# Update weights during training
loader.update_weights(
    patient_weights=new_patient_weights,
    hour_weights=new_hour_weights
)
```

### Training Loop Integration
```python
for epoch in range(num_epochs):
    for batch_idx in range(batches_per_epoch):
        batch = loader.get_batch()
        
        accel_data = batch['accel_data']  # (batch_size, 3, context_len)
        timestamps = batch['timestamps']  # (batch_size, context_len)
        patient_ids = batch['patient_ids']  # (batch_size,)
        
        # Your training code here
        loss = model(accel_data)
        optimizer.step()
```

## Data Format

### Batch Output
```python
{
    'accel_data': torch.Tensor,    # Shape: (batch_size, 3, context_len)
    'timestamps': torch.Tensor,    # Shape: (batch_size, context_len)
    'patient_ids': torch.Tensor    # Shape: (batch_size,)
}
```

## Example Scripts

Run the comprehensive example script:
```bash
python example_usage.py
```

This will demonstrate:
- Uniform sampling
- Cross-file continuous sampling
- Weighted sampling by time and day
- Patient-specific sampling
- Dynamic weight updates
- Training loop integration
- Time-of-day weighted sampling

## ToDo

This system is designed to be extensible. Key areas for enhancement:

- Additional data preprocessing options
- More sophisticated sampling strategies
- Integration with other data sources
- Performance optimizations