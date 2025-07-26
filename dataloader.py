import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
import logging
from util import (
    discover_patient_data, 
    create_sampling_weights, 
    sample_file_indices, 
    extract_sequence_from_patient_files,
    get_patient_metadata,
    DataFileInfo
)

logger = logging.getLogger(__name__)

class AccelDataset(Dataset):
    """
    Dataset for accelerometer data with controllable sampling weights.
    
    This dataset samples sequences from patient data files with configurable
    weights for patients, hours, and days of the week.
    """
    
    def __init__(self, 
                 data_dir: str = "./patient_data",
                 context_len: int = 1000,
                 downsampling_ratio: float = 1.0,
                 patient_weights: Optional[Dict[str, float]] = None,
                 hour_weights: Optional[Dict[int, float]] = None,
                 day_weights: Optional[Dict[int, float]] = None,
                 cache_size: int = 1000,
                 seed: Optional[int] = None):
        """
        Initialize the dataset.
        
        Args:
            data_dir: Directory containing patient data folders
            context_len: Number of output points after downsampling
            downsampling_ratio: Downsampling ratio (1.0 = no downsampling, 2.0 = half the points, etc.)
            patient_weights: Optional weights for each patient (default: uniform)
            hour_weights: Optional weights for each hour of day (default: uniform)
            day_weights: Optional weights for each day of week (default: uniform)
            cache_size: Number of samples to pre-generate and cache
            seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.context_len = context_len
        self.downsampling_ratio = downsampling_ratio
        self.cache_size = cache_size
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Discover patient data
        logger.info("Discovering patient data files...")
        self.patient_files = discover_patient_data(data_dir)
        
        if not self.patient_files:
            raise ValueError(f"No patient data found in {data_dir}")
        
        # Create sampling weights
        self.sampling_weights = create_sampling_weights(
            self.patient_files, patient_weights, hour_weights, day_weights
        )
        
        # Pre-generate sample indices
        self._regenerate_cache()
        
        logger.info(f"Initialized dataset with {len(self.patient_files)} patients")
    
    def _regenerate_cache(self):
        """Regenerate the cache of sample indices."""
        self.cached_samples = sample_file_indices(
            self.patient_files, 
            self.sampling_weights, 
            self.cache_size
        )
        self.cache_index = 0
    
    def _get_next_sample(self) -> Tuple[str, int]:
        """Get the next sample from cache, regenerating if necessary."""
        if self.cache_index >= len(self.cached_samples):
            self._regenerate_cache()
        
        sample = self.cached_samples[self.cache_index]
        self.cache_index += 1
        return sample
    
    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return len(self.cached_samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Index of the sample
            
        Returns:
            Dictionary containing:
            - 'accel_data': Tensor of shape (3, context_len) for X, Y, Z accelerations
            - 'timestamps': Tensor of shape (context_len,) containing timestamps
            - 'patient_id': Tensor containing patient ID
        """
        # Get patient_id and start_offset
        patient_id, start_offset = self._get_next_sample()
        
        # Get all files for this patient, sorted by timestamp
        patient_files = self.patient_files[patient_id]
        
        # Sample from the continuous timeline starting at the specified offset
        timestamps, accel_data = extract_sequence_from_patient_files(
            patient_files, self.context_len, start_offset=start_offset, downsampling_ratio=self.downsampling_ratio
        )
        
        if len(accel_data) == 0:
            # If extraction failed, return a zero sequence
            accel_data = np.zeros((self.context_len, 3), dtype=np.float32)
            timestamps = np.zeros(self.context_len, dtype=np.datetime64)
        
        # Convert to tensors
        accel_tensor = torch.from_numpy(accel_data).float().transpose(0, 1)  # (3, context_len)
        timestamps_tensor = torch.from_numpy(timestamps.astype(np.int64))  # Convert to int64 for tensor compatibility
        
        # Create patient ID tensor (encode as integer for now)
        patient_id_int = int(patient_id)  # Assuming patient_id is numeric
        patient_id_tensor = torch.tensor(patient_id_int, dtype=torch.long)
        
        return {
            'accel_data': accel_tensor,  # Shape: (3, context_len)
            'timestamps': timestamps_tensor,  # Shape: (context_len,)
            'patient_id': patient_id_tensor,  # Shape: (1,)
            'patient_id_str': patient_id  # Keep string version for reference
        }
    
    def update_weights(self, 
                      patient_weights: Optional[Dict[str, float]] = None,
                      hour_weights: Optional[Dict[int, float]] = None,
                      day_weights: Optional[Dict[int, float]] = None):
        """
        Update sampling weights and regenerate cache.
        
        Args:
            patient_weights: New weights for patients
            hour_weights: New weights for hours
            day_weights: New weights for days
        """
        self.sampling_weights = create_sampling_weights(
            self.patient_files, patient_weights, hour_weights, day_weights
        )
        self._regenerate_cache()
        logger.info("Updated sampling weights and regenerated cache")

class AccelDataLoader:
    """
    High-level data loader for accelerometer data with batch sampling.
    
    This class provides a convenient interface for sampling batches of sequences
    with controllable weights and metadata preservation.
    """
    
    def __init__(self, 
                 data_dir: str = "./patient_data",
                 context_len: int = 1000,
                 downsampling_ratio: float = 1.0,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 patient_weights: Optional[Dict[str, float]] = None,
                 hour_weights: Optional[Dict[int, float]] = None,
                 day_weights: Optional[Dict[int, float]] = None,
                 cache_size: int = 10000,
                 seed: Optional[int] = None):
        """
        Initialize the data loader.
        
        Args:
            data_dir: Directory containing patient data folders
            context_len: Number of output points after downsampling
            downsampling_ratio: Downsampling ratio (1.0 = no downsampling, 2.0 = half the points, etc.)
            batch_size: Number of samples per batch
            num_workers: Number of worker processes for data loading
            patient_weights: Optional weights for each patient
            hour_weights: Optional weights for each hour of day
            day_weights: Optional weights for each day of week
            cache_size: Number of samples to pre-generate and cache
            seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.context_len = context_len
        self.downsampling_ratio = downsampling_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.cache_size = cache_size
        self.seed = seed
        
        # Create dataset
        self.dataset = AccelDataset(
            data_dir=data_dir,
            context_len=context_len,
            downsampling_ratio=downsampling_ratio,
            patient_weights=patient_weights,
            hour_weights=hour_weights,
            day_weights=day_weights,
            cache_size=cache_size,
            seed=seed
        )
        
        # Create PyTorch DataLoader
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=False,  # We handle shuffling in the dataset
            num_workers=num_workers,
            collate_fn=self._collate_fn,
            drop_last=True
        )
        
        logger.info(f"Initialized data loader with batch_size={batch_size}, context_len={context_len}, downsampling_ratio={downsampling_ratio}")
    
    def _collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Custom collate function to handle the batch data with vectorized operations.
        
        Args:
            batch: List of samples from the dataset
            
        Returns:
            Dictionary containing batched tensors:
            - 'accel_data': Tensor of shape (batch_size, 3, context_len)
            - 'timestamps': Tensor of shape (batch_size, context_len)
            - 'patient_ids': Tensor of shape (batch_size,)
        """
        # Vectorized extraction using list comprehensions
        accel_data = [sample['accel_data'] for sample in batch]
        timestamps = [sample['timestamps'] for sample in batch]
        patient_ids = [sample['patient_id'] for sample in batch]
        
        # Vectorized stacking using torch.stack
        accel_batch = torch.stack(accel_data)  # (batch_size, 3, context_len)
        timestamps_batch = torch.stack(timestamps)  # (batch_size, context_len)
        patient_ids_batch = torch.stack(patient_ids)  # (batch_size,)
        
        return {
            'accel_data': accel_batch,
            'timestamps': timestamps_batch,
            'patient_ids': patient_ids_batch
        }
    
    def get_batch(self) -> Dict[str, torch.Tensor]:
        """
        Get a single batch of data.
        
        Returns:
            Dictionary containing batched tensors
        """
        try:
            return next(iter(self.dataloader))
        except StopIteration:
            # If we run out of data, regenerate the dataset
            self.dataset._regenerate_cache()
            return next(iter(self.dataloader))
    
    def update_weights(self, 
                      patient_weights: Optional[Dict[str, float]] = None,
                      hour_weights: Optional[Dict[int, float]] = None,
                      day_weights: Optional[Dict[int, float]] = None):
        """
        Update sampling weights.
        
        Args:
            patient_weights: New weights for patients
            hour_weights: New weights for hours
            day_weights: New weights for days
        """
        self.dataset.update_weights(patient_weights, hour_weights, day_weights)
    
    def get_patient_metadata(self, patient_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get metadata for a list of patient IDs.
        
        Args:
            patient_ids: List of patient IDs
            
        Returns:
            List of metadata dictionaries
        """
        return [get_patient_metadata(pid) for pid in patient_ids]
    
    def __iter__(self):
        """Make the data loader iterable."""
        return iter(self.dataloader)
    
    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        return len(self.dataloader)

# Example usage and utility functions
def create_hour_weights(peak_hours: List[int] = [8, 12, 18], 
                       peak_weight: float = 2.0,
                       base_weight: float = 1.0) -> Dict[int, float]:
    """
    Create hour weights with peaks during specified hours.
    
    Args:
        peak_hours: Hours to give higher weight (0-23)
        peak_weight: Weight for peak hours
        base_weight: Weight for non-peak hours
        
    Returns:
        Dictionary mapping hour to weight
    """
    weights = {hour: base_weight for hour in range(24)}
    for hour in peak_hours:
        weights[hour] = peak_weight
    return weights

def create_day_weights(weekend_weight: float = 1.5,
                      weekday_weight: float = 1.0) -> Dict[int, float]:
    """
    Create day weights with different weights for weekdays vs weekends.
    
    Args:
        weekend_weight: Weight for weekend days (5=Saturday, 6=Sunday)
        weekday_weight: Weight for weekday days (0-4)
        
    Returns:
        Dictionary mapping day of week to weight
    """
    weights = {day: weekday_weight for day in range(5)}  # Monday-Friday
    weights.update({5: weekend_weight, 6: weekend_weight})  # Saturday, Sunday
    return weights
