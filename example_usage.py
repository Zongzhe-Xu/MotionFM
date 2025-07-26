#!/usr/bin/env python3
"""
Example usage of the accelerometer data loading system.

This script demonstrates how to use the AccelDataLoader for large-scale
pretraining with controllable sampling weights.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from dataloader import AccelDataLoader, create_hour_weights, create_day_weights
import logging
import pandas as pd

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def visualize_batch(batch_data: dict, batch_idx: int = 0, save_path: str = None):
    """
    Visualize a single sample from a batch.
    
    Args:
        batch_data: Dictionary containing batch data
        batch_idx: Index of the sample to visualize
        save_path: Optional path to save the plot
    """
    accel_data = batch_data['accel_data'][batch_idx]  # Shape: (3, context_len)
    timestamps = batch_data['timestamps'][batch_idx]  # Shape: (context_len,)
    patient_id = batch_data['patient_ids'][batch_idx].item()
    
    # Debug: Print timestamp info
    print(f"Timestamp type: {type(timestamps)}")
    print(f"Timestamp shape: {timestamps.shape}")
    print(f"First few timestamps: {timestamps[:5]}")
    print(f"Timestamp dtype: {timestamps.dtype}")
    
    # Convert PyTorch tensor to numpy array
    timestamps_np = timestamps.numpy()
    # Convert nanoseconds since epoch to pandas datetime
    timestamps_dt = pd.to_datetime(timestamps_np, unit='ns')
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8))
    axes[0].plot(timestamps_dt, accel_data[0], label='X-axis')
    axes[0].set_title(f'Patient {patient_id} - X-axis Acceleration')
    axes[0].set_ylabel('Acceleration (g)')
    axes[0].legend()
    
    axes[1].plot(timestamps_dt, accel_data[1], label='Y-axis', color='orange')
    axes[1].set_title(f'Patient {patient_id} - Y-axis Acceleration')
    axes[1].set_ylabel('Acceleration (g)')
    axes[1].legend()
    
    axes[2].plot(timestamps_dt, accel_data[2], label='Z-axis', color='green')
    axes[2].set_title(f'Patient {patient_id} - Z-axis Acceleration')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Acceleration (g)')
    axes[2].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved visualization to {save_path}")
    
    plt.show()

def example_uniform_sampling():
    """Example with uniform sampling across all patients and times."""
    logger.info("=== Example 1: Uniform Sampling ===")
    
    # Create data loader with uniform sampling
    loader = AccelDataLoader(
        data_dir="./patient_data",
        context_len=1000,  # 1000 samples per sequence
        batch_size=8,
        num_workers=2,
        cache_size=1000,
        seed=42
    )
    
    # Get a batch
    batch = loader.get_batch()
    logger.info(f"Batch shapes:")
    logger.info(f"  accel_data: {batch['accel_data'].shape}")
    logger.info(f"  timestamps: {batch['timestamps'].shape}")
    logger.info(f"  patient_ids: {batch['patient_ids'].shape}")
    
    # Show unique patient IDs in this batch
    unique_patients = torch.unique(batch['patient_ids']).tolist()
    logger.info(f"Unique patients in batch: {unique_patients}")
    
    # Visualize first sample
    visualize_batch(batch, batch_idx=0, save_path="uniform_sampling_example.png")

def example_cross_file_sampling():
    """Example demonstrating cross-file continuous sampling."""
    logger.info("=== Example 2: Cross-File Continuous Sampling ===")
    
    # Create data loader with longer context to demonstrate cross-file sampling
    loader = AccelDataLoader(
        data_dir="./patient_data",
        context_len=2000,  # Longer sequence to increase chance of crossing file boundaries
        batch_size=4,
        num_workers=2,
        cache_size=1000,
        seed=42
    )
    
    # Get multiple batches to see cross-file sampling in action
    logger.info("Sampling batches with cross-file sequences...")
    for i in range(3):
        batch = loader.get_batch()
        unique_patients = torch.unique(batch['patient_ids']).tolist()
        logger.info(f"Batch {i+1}: Patients {unique_patients}")
        
        # Show timestamp range for first sample to demonstrate continuity
        if i == 0:
            timestamps = batch['timestamps'][0]
            start_time = pd.to_datetime(timestamps[0], unit='ns')
            end_time = pd.to_datetime(timestamps[-1], unit='ns')
            duration = end_time - start_time
            logger.info(f"Sample sequence duration: {duration}")
            logger.info(f"Start time: {start_time}")
            logger.info(f"End time: {end_time}")
    
    # Visualize a sample
    batch = loader.get_batch()
    visualize_batch(batch, batch_idx=0, save_path="cross_file_sampling_example.png")

def example_weighted_sampling():
    """Example with weighted sampling favoring certain hours and days."""
    logger.info("=== Example 3: Weighted Sampling ===")
    
    # Create weights that favor morning hours (6-10) and weekdays
    hour_weights = create_hour_weights(
        peak_hours=[6, 7, 8, 9, 10],  # Morning hours
        peak_weight=3.0,
        base_weight=1.0
    )
    
    day_weights = create_day_weights(
        weekend_weight=0.5,  # Lower weight for weekends
        weekday_weight=1.0
    )
    
    # Create data loader with weighted sampling
    loader = AccelDataLoader(
        data_dir="./patient_data",
        context_len=1000,
        batch_size=8,
        num_workers=2,
        patient_weights=None,  # Uniform patient weights
        hour_weights=hour_weights,
        day_weights=day_weights,
        cache_size=1000,
        seed=42
    )
    
    # Get multiple batches to see the effect of weighting
    logger.info("Sampling 5 batches with morning/weekday bias...")
    for i in range(5):
        batch = loader.get_batch()
        unique_patients = torch.unique(batch['patient_ids']).tolist()
        logger.info(f"Batch {i+1}: Patients {unique_patients}")
    
    # Visualize a sample
    batch = loader.get_batch()
    visualize_batch(batch, batch_idx=0, save_path="weighted_sampling_example.png")

def example_patient_specific_sampling():
    """Example with patient-specific sampling weights."""
    logger.info("=== Example 4: Patient-Specific Sampling ===")
    
    # Give higher weight to patient 73557
    patient_weights = {
        "73557": 2.0,  # Higher weight
        "73559": 1.0   # Normal weight
    }
    
    # Create data loader with patient-specific weights
    loader = AccelDataLoader(
        data_dir="./patient_data",
        context_len=1000,
        batch_size=8,
        num_workers=2,
        patient_weights=patient_weights,
        cache_size=1000,
        seed=42
    )
    
    # Count patient distribution across multiple batches
    patient_counts = {"73557": 0, "73559": 0}
    total_samples = 0
    
    logger.info("Sampling 10 batches with patient bias...")
    for i in range(10):
        batch = loader.get_batch()
        for patient_id in batch['patient_ids']:
            patient_counts[str(patient_id.item())] += 1
            total_samples += 1
    
    logger.info(f"Patient distribution across {total_samples} samples:")
    for patient_id, count in patient_counts.items():
        percentage = (count / total_samples) * 100
        logger.info(f"  Patient {patient_id}: {count} samples ({percentage:.1f}%)")

def example_dynamic_weight_updating():
    """Example showing how to dynamically update sampling weights."""
    logger.info("=== Example 5: Dynamic Weight Updating ===")
    
    # Start with uniform sampling
    loader = AccelDataLoader(
        data_dir="./patient_data",
        context_len=1000,
        batch_size=8,
        num_workers=2,
        cache_size=1000,
        seed=42
    )
    
    # Get initial batch
    batch = loader.get_batch()
    logger.info("Initial uniform sampling:")
    unique_patients = torch.unique(batch['patient_ids']).tolist()
    logger.info(f"Patients in batch: {unique_patients}")
    
    # Update weights to favor patient 73559
    new_patient_weights = {
        "73557": 0.5,  # Lower weight
        "73559": 2.0   # Higher weight
    }
    
    logger.info("Updating weights to favor patient 73559...")
    loader.update_weights(patient_weights=new_patient_weights)
    
    # Get new batch with updated weights
    batch = loader.get_batch()
    logger.info("After weight update:")
    unique_patients = torch.unique(batch['patient_ids']).tolist()
    logger.info(f"Patients in batch: {unique_patients}")

def example_training_loop():
    """Example of how to use the data loader in a training loop."""
    logger.info("=== Example 6: Training Loop Simulation ===")
    
    # Create data loader
    loader = AccelDataLoader(
        data_dir="./patient_data",
        context_len=1000,
        batch_size=16,
        num_workers=2,
        cache_size=5000,
        seed=42
    )
    
    # Simulate training loop
    num_epochs = 3
    batches_per_epoch = 5
    
    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1}/{num_epochs}")
        
        epoch_loss = 0.0
        for batch_idx in range(batches_per_epoch):
            # Get batch
            batch = loader.get_batch()
            
            # Simulate some processing (e.g., model forward pass)
            accel_data = batch['accel_data']  # Shape: (batch_size, 3, context_len)
            timestamps = batch['timestamps']  # Shape: (batch_size, context_len)
            patient_ids = batch['patient_ids']  # Shape: (batch_size,)
            
            # Simulate loss computation
            batch_loss = torch.mean(accel_data ** 2)  # Dummy loss
            epoch_loss += batch_loss.item()
            
            logger.info(f"  Batch {batch_idx + 1}: Loss = {batch_loss.item():.4f}")
        
        avg_epoch_loss = epoch_loss / batches_per_epoch
        logger.info(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")

def example_time_of_day_weighted_sampling():
    """Example demonstrating time-of-day weighted sampling with cross-file sequences."""
    logger.info("=== Example 7: Time-of-Day Weighted Sampling ===")
    
    # Create weights that heavily favor morning hours (6-10 AM)
    hour_weights = create_hour_weights(
        peak_hours=[6, 7, 8, 9, 10],  # Morning hours
        peak_weight=5.0,  # Much higher weight for morning
        base_weight=0.5   # Lower weight for other hours
    )
    
    # Create data loader with time-of-day weighting
    loader = AccelDataLoader(
        data_dir="./patient_data",
        context_len=1500,  # Medium sequence length
        batch_size=6,
        num_workers=2,
        hour_weights=hour_weights,
        cache_size=1000,
        seed=42
    )
    
    # Collect statistics on sampled times
    sampled_hours = []
    
    logger.info("Sampling batches with morning bias...")
    for i in range(5):
        batch = loader.get_batch()
        unique_patients = torch.unique(batch['patient_ids']).tolist()
        logger.info(f"Batch {i+1}: Patients {unique_patients}")
        
        # Analyze the time distribution of sampled sequences
        for j in range(min(3, len(batch['timestamps']))):  # Check first 3 samples
            timestamps = batch['timestamps'][j]
            start_time = pd.to_datetime(timestamps[0], unit='ns')
            sampled_hours.append(start_time.hour)
            logger.info(f"  Sample {j+1} start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')} (hour: {start_time.hour})")
    
    # Show hour distribution statistics
    hour_counts = {}
    for hour in sampled_hours:
        hour_counts[hour] = hour_counts.get(hour, 0) + 1
    
    logger.info(f"\nHour distribution across {len(sampled_hours)} samples:")
    for hour in sorted(hour_counts.keys()):
        count = hour_counts[hour]
        percentage = (count / len(sampled_hours)) * 100
        logger.info(f"  Hour {hour:02d}: {count} samples ({percentage:.1f}%)")
    
    # Visualize a sample
    batch = loader.get_batch()
    visualize_batch(batch, batch_idx=0, save_path="time_of_day_weighted_example.png")

if __name__ == "__main__":
    # Run all examples
    example_uniform_sampling()
    example_cross_file_sampling()
    example_weighted_sampling()
    example_patient_specific_sampling()
    example_dynamic_weight_updating()
    example_training_loop()
    example_time_of_day_weighted_sampling() 