import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional, Any
import logging
import time
import pandas as pd
from util import (
    discover_patient_data, 
    load_accel_data,
    get_patient_metadata,
    DataFileInfo
)

logger = logging.getLogger(__name__)

class EfficientAccelDataset(Dataset):
    """
    Efficient Dataset for accelerometer data with on-demand file discovery.
    
    This dataset samples patients first, then discovers files on-demand,
    making it much more efficient for large datasets.
    """
    
    def __init__(self, 
                 data_dir: str = "./patient_data",
                 context_len: int = 1000,
                 downsampling_ratio: float = 1.0,
                 patient_weights: Optional[Dict[str, float]] = None,
                 hour_weights: Optional[Dict[int, float]] = None,
                 cache_size: int = 1000,
                 cache_data: bool = False,  # Whether to cache actual data
                 seed: Optional[int] = None):
        """
        Initialize the efficient dataset.
        
        Args:
            data_dir: Directory containing patient data folders
            context_len: Number of output points after downsampling
            downsampling_ratio: Downsampling ratio (1.0 = no downsampling, 2.0 = half the points, etc.)
            patient_weights: Optional weights for each patient (default: uniform)
            hour_weights: Optional weights for each hour of day (default: uniform)
            cache_size: Number of samples to pre-generate and cache
            seed: Random seed for reproducibility
        """
        self.data_dir = data_dir
        self.context_len = context_len
        self.downsampling_ratio = downsampling_ratio
        self.cache_size = cache_size
        self.cache_data = cache_data
        
        # Set random seed
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Get list of all patients (much faster than discovering all files)
        logger.info("Discovering patient directories...")
        start_time = time.time()
        self.patient_ids = self._discover_patients()
        discovery_time = time.time() - start_time
        
        if not self.patient_ids:
            raise ValueError(f"No patient directories found in {data_dir}")
        
        # Set up weights
        self.patient_weights = patient_weights or {pid: 1.0 for pid in self.patient_ids}
        self.hour_weights = hour_weights or {hour: 1.0 for hour in range(24)}
        
        # Normalize weights
        self._normalize_weights()
        
        # Pre-generate sample indices
        start_time = time.time()
        self._regenerate_cache()
        cache_time = time.time() - start_time
        
        # Performance summary
        logger.info(f"=== Efficient Dataset Initialization Performance ===")
        logger.info(f"Total patients: {len(self.patient_ids)}")
        logger.info(f"Patient discovery: {discovery_time:.2f}s")
        logger.info(f"Cache generation: {cache_time:.2f}s")
        logger.info(f"Total initialization: {discovery_time + cache_time:.2f}s")
        logger.info(f"==================================================")
    
    def _discover_patients(self) -> List[str]:
        """Discover all patient directories without reading files."""
        if not os.path.exists(self.data_dir):
            return []
        
        patient_ids = []
        for item in os.listdir(self.data_dir):
            item_path = os.path.join(self.data_dir, item)
            if os.path.isdir(item_path) and not item.startswith("worker_"):
                patient_ids.append(item)
        
        return sorted(patient_ids)
    
    def _normalize_weights(self):
        """Normalize patient and hour weights."""
        # Normalize patient weights
        patient_weight_sum = sum(self.patient_weights.get(pid, 1.0) for pid in self.patient_ids)
        if patient_weight_sum > 0:
            self.patient_weights = {pid: self.patient_weights.get(pid, 1.0) / patient_weight_sum 
                                  for pid in self.patient_ids}
        
        # Normalize hour weights
        hour_weight_sum = sum(self.hour_weights.get(hour, 1.0) for hour in range(24))
        if hour_weight_sum > 0:
            self.hour_weights = {hour: self.hour_weights.get(hour, 1.0) / hour_weight_sum 
                               for hour in range(24)}
    
    def _discover_patient_files(self, patient_id: str) -> List[DataFileInfo]:
        """Discover files for a specific patient on-demand."""
        patient_dir = os.path.join(self.data_dir, patient_id)
        if not os.path.exists(patient_dir):
            return []
        
        # Pattern to match sensor Parquet files
        import re
        pattern = r'^[\w-]+\.[\w-]+\.(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{3})-P\d{4}\.sensor\.parquet$'
        
        files = []
        for filename in os.listdir(patient_dir):
            match = re.match(pattern, filename)
            if match:
                file_path = os.path.join(patient_dir, filename)
                file_size = os.path.getsize(file_path)
                
                # Extract timestamp components
                year, month, day, hour, minute, second, millisecond = match.groups()
                timestamp = f"{year}-{month}-{day} {hour}:{minute}:{second}.{millisecond}"
                
                # Calculate day of week (0=Monday, 6=Sunday)
                from datetime import datetime
                dt = datetime(int(year), int(month), int(day))
                day_of_week = dt.weekday()
                
                file_info = DataFileInfo(
                    patient_id=patient_id,
                    file_path=file_path,
                    timestamp=timestamp,
                    hour=int(hour),
                    day_of_week=day_of_week,
                    file_size=file_size
                )
                files.append(file_info)
        
        # Sort by timestamp for efficient sequential reading
        files.sort(key=lambda x: x.timestamp)
        return files
    
    def _sample_patient_and_hour(self) -> Tuple[str, int]:
        """Sample a patient and hour based on weights."""
        # Sample patient
        patient_ids = list(self.patient_weights.keys())
        patient_weights = [self.patient_weights[pid] for pid in patient_ids]
        patient_id = np.random.choice(patient_ids, p=patient_weights)
        
        # Sample hour
        hours = list(self.hour_weights.keys())
        hour_weights = [self.hour_weights[hour] for hour in hours]
        hour = np.random.choice(hours, p=hour_weights)
        
        return patient_id, hour
    
    def _get_files_for_hour(self, patient_files: List[DataFileInfo], target_hour: int) -> List[DataFileInfo]:
        """Get all files for a specific hour (can be multiple days)."""
        return [f for f in patient_files if f.hour == target_hour]
    
    def _extract_sequence_from_files(self, all_patient_files: List[DataFileInfo], start_file_idx: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Extract a sequence from files starting at a specific file index, reading chronologically."""
        if not all_patient_files or start_file_idx >= len(all_patient_files):
            return np.array([]), np.array([])
        
        # Calculate how many raw points we need
        raw_context_len = int(self.context_len * self.downsampling_ratio)
        
        timestamps = []
        accel_data = []
        remaining = raw_context_len
        file_idx = start_file_idx
        
        while file_idx < len(all_patient_files) and remaining > 0:
            file_info = all_patient_files[file_idx]
            
            # Load data from this file
            ts, acc = load_accel_data(file_info.file_path)
            
            if len(acc) == 0:
                file_idx += 1
                continue
            
            # Take what we need from this file
            take_count = min(len(acc), remaining)
            timestamps.append(ts[:take_count])
            accel_data.append(acc[:take_count])
            
            remaining -= take_count
            file_idx += 1
        
        if not timestamps:
            return np.array([]), np.array([])
        
        # Concatenate all data
        timestamps = np.concatenate(timestamps)
        accel_data = np.concatenate(accel_data)
        
        # Downsample if needed
        if self.downsampling_ratio > 1.0 and len(accel_data) > 1:
            original_indices = np.arange(len(accel_data))
            target_indices = np.linspace(0, len(accel_data) - 1, self.context_len)
            
            # Interpolate timestamps
            timestamp_numeric = timestamps.astype(np.int64)
            interpolated_timestamps_numeric = np.interp(target_indices, original_indices, timestamp_numeric)
            timestamps = pd.to_datetime(interpolated_timestamps_numeric).values
            
            # Interpolate accel data
            accel_data = np.array([
                np.interp(target_indices, original_indices, accel_data[:, i])
                for i in range(3)
            ]).T
        else:
            # Pad if not enough points
            if len(accel_data) < self.context_len:
                pad_length = self.context_len - len(accel_data)
                accel_data = np.pad(accel_data, ((pad_length, 0), (0, 0)), mode='constant', constant_values=0)
                if len(timestamps) > 0:
                    first_timestamp = timestamps[0]
                    timestamps = np.pad(timestamps, (pad_length, 0), mode='constant', constant_values=first_timestamp)
            
            # Truncate if too many points
            if len(accel_data) > self.context_len:
                accel_data = accel_data[-self.context_len:]
                timestamps = timestamps[-self.context_len:]
        
        return timestamps, accel_data
    
    def _regenerate_cache(self):
        """Regenerate the cache of sample indices."""
        self.cached_samples = []
        
        for _ in range(self.cache_size):
            # Sample patient and hour
            patient_id, hour = self._sample_patient_and_hour()
            
            # Discover files for this patient (on-demand)
            patient_files = self._discover_patient_files(patient_id)
            
            if not patient_files:
                continue
            
            # Get files for the target hour
            hour_files = self._get_files_for_hour(patient_files, hour)
            
            if not hour_files:
                continue
            
            # Sample a random file from this hour
            hour_file_idx = np.random.randint(0, len(hour_files))
            sampled_file = hour_files[hour_file_idx]
            
            # Find the index of this file in the full patient file list
            start_file_idx = patient_files.index(sampled_file)
            
            # Try to extract sequence from all patient files starting at this index
            if self.cache_data:
                timestamps, accel_data = self._extract_sequence_from_files(patient_files, start_file_idx)
            
            # If we don't have enough data, skip this sample
            if len(accel_data) < self.context_len:
                continue
            
            # Store the sample info
            sample_info = {
                'patient_id': patient_id,
                'hour': hour,
                'start_file_idx': start_file_idx,
                'all_patient_files': patient_files  # Keep reference to all files for chronological reading
            }
            
            # Cache actual data if enabled
            if self.cache_data:
                sample_info['cached_timestamps'] = timestamps
                sample_info['cached_accel_data'] = accel_data
            
            self.cached_samples.append(sample_info)
        
        self.cache_index = 0
        logger.info(f"Generated {len(self.cached_samples)} valid samples")
    
    def _get_next_sample(self) -> Dict[str, Any]:
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
        # Get sample info
        sample_info = self._get_next_sample()
        
        # Extract sequence from files (use cached data if available)
        start_time = time.time()
        
        if self.cache_data and 'cached_timestamps' in sample_info:
            # Use cached data
            timestamps = sample_info['cached_timestamps']
            accel_data = sample_info['cached_accel_data']
            load_time = time.time() - start_time
            
            # Log performance every 1000 samples
            if idx % 1000 == 0:
                logger.info(f"Sample {idx}: Used cached data from patient {sample_info['patient_id']} "
                           f"hour {sample_info['hour']} in {load_time:.3f}s")
        else:
            # Load data from files
            timestamps, accel_data = self._extract_sequence_from_files(
                sample_info['all_patient_files'], 
                sample_info['start_file_idx']
            )
            load_time = time.time() - start_time
            
            # Log performance every 1000 samples
            if idx % 1000 == 0:
                logger.info(f"Sample {idx}: Loaded from patient {sample_info['patient_id']} "
                           f"hour {sample_info['hour']} in {load_time:.3f}s")
        
        if len(accel_data) == 0:
            # If extraction failed, return a zero sequence
            accel_data = np.zeros((self.context_len, 3), dtype=np.float32)
            timestamps = np.zeros(self.context_len, dtype=np.datetime64)
        
        # Convert to tensors
        accel_tensor = torch.from_numpy(accel_data).float().transpose(0, 1)  # (3, context_len)
        timestamps_tensor = torch.from_numpy(timestamps.astype(np.int64))  # Convert to int64 for tensor compatibility
        
        # Create patient ID tensor (encode as integer for now)
        patient_id_int = int(sample_info['patient_id'])  # Assuming patient_id is numeric
        patient_id_tensor = torch.tensor(patient_id_int, dtype=torch.long)
        
        return {
            'accel_data': accel_tensor,  # Shape: (3, context_len)
            'timestamps': timestamps_tensor,  # Shape: (context_len,)
            'patient_id': patient_id_tensor,  # Shape: (1,)
            'patient_id_str': sample_info['patient_id']  # Keep string version for reference
        }
    
    def update_weights(self, 
                      patient_weights: Optional[Dict[str, float]] = None,
                      hour_weights: Optional[Dict[int, float]] = None):
        """
        Update sampling weights and regenerate cache.
        
        Args:
            patient_weights: New weights for patients
            hour_weights: New weights for hours
        """
        if patient_weights is not None:
            self.patient_weights = patient_weights
        if hour_weights is not None:
            self.hour_weights = hour_weights
        
        self._normalize_weights()
        self._regenerate_cache()
        logger.info("Updated sampling weights and regenerated cache")

class EfficientAccelDataLoader:
    """
    High-level data loader for accelerometer data with efficient sampling.
    
    This class provides a convenient interface for sampling batches of sequences
    with on-demand file discovery and minimal memory usage.
    """
    
    def __init__(self, 
                 data_dir: str = "./patient_data",
                 context_len: int = 1000,
                 downsampling_ratio: float = 1.0,
                 batch_size: int = 32,
                 num_workers: int = 4,
                 patient_weights: Optional[Dict[str, float]] = None,
                 hour_weights: Optional[Dict[int, float]] = None,
                 cache_size: int = 10000,
                 cache_data: bool = False,  # Whether to cache actual data
                 seed: Optional[int] = None):
        """
        Initialize the efficient data loader.
        
        Args:
            data_dir: Directory containing patient data folders
            context_len: Number of output points after downsampling
            downsampling_ratio: Downsampling ratio (1.0 = no downsampling, 2.0 = half the points, etc.)
            batch_size: Number of samples per batch
            num_workers: Number of worker processes for data loading
            patient_weights: Optional weights for each patient
            hour_weights: Optional weights for each hour of day
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
        self.dataset = EfficientAccelDataset(
            data_dir=data_dir,
            context_len=context_len,
            downsampling_ratio=downsampling_ratio,
            patient_weights=patient_weights,
            hour_weights=hour_weights,
            cache_size=cache_size,
            cache_data=cache_data,
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
        
        # Performance summary
        total_patients = len(self.dataset.patient_ids)
        cache_memory_mb = 0
        if cache_data:
            # Estimate memory for cached data: cache_size * context_len * (3 floats + 1 timestamp) * 8 bytes
            cache_memory_mb = cache_size * context_len * 4 * 8 / (1024*1024)
        
        logger.info(f"=== Efficient DataLoader Performance Summary ===")
        logger.info(f"Batch size: {batch_size}")
        logger.info(f"Context length: {context_len}")
        logger.info(f"Downsampling ratio: {downsampling_ratio}")
        logger.info(f"Number of workers: {num_workers}")
        logger.info(f"Cache size: {cache_size}")
        logger.info(f"Cache data: {cache_data}")
        logger.info(f"Total patients: {total_patients}")
        logger.info(f"Estimated memory per batch: {batch_size * context_len * 3 * 4 / (1024*1024):.1f} MB")
        if cache_data:
            logger.info(f"Estimated cache memory: {cache_memory_mb:.1f} MB")
        logger.info(f"===============================================")
    
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
                      hour_weights: Optional[Dict[int, float]] = None):
        """
        Update sampling weights.
        
        Args:
            patient_weights: New weights for patients
            hour_weights: New weights for hours
        """
        self.dataset.update_weights(patient_weights, hour_weights)
    
    def get_patient_metadata(self, patient_ids: List[str]) -> List[Dict[str, Any]]:
        """
        Get metadata for a list of patient IDs.
        
        Args:
            patient_ids: List of patient IDs
            
        Returns:
            List of metadata dictionaries
        """
        return [get_patient_metadata(pid) for pid in patient_ids]
    
    def benchmark_loading_speed(self, num_batches: int = 10) -> Dict[str, float]:
        """
        Benchmark the data loading speed.
        
        Args:
            num_batches: Number of batches to load for benchmarking
            
        Returns:
            Dictionary with performance metrics
        """
        logger.info(f"=== Starting Efficient Loading Speed Benchmark ({num_batches} batches) ===")
        
        total_time = 0
        total_samples = 0
        batch_times = []
        
        start_time = time.time()
        
        for i in range(num_batches):
            batch_start = time.time()
            try:
                batch = self.get_batch()
                batch_time = time.time() - batch_start
                batch_times.append(batch_time)
                
                batch_size = batch['accel_data'].shape[0]
                total_samples += batch_size
                total_time += batch_time
                
                logger.info(f"Batch {i+1}/{num_batches}: {batch_size} samples in {batch_time:.3f}s "
                           f"({batch_size/batch_time:.1f} samples/sec)")
                
            except Exception as e:
                logger.error(f"Error loading batch {i+1}: {e}")
                break
        
        end_time = time.time()
        
        # Calculate statistics
        avg_batch_time = np.mean(batch_times) if batch_times else 0
        std_batch_time = np.std(batch_times) if batch_times else 0
        samples_per_sec = total_samples / total_time if total_time > 0 else 0
        batches_per_sec = len(batch_times) / total_time if total_time > 0 else 0
        
        results = {
            'total_time': total_time,
            'total_samples': total_samples,
            'avg_batch_time': avg_batch_time,
            'std_batch_time': std_batch_time,
            'samples_per_sec': samples_per_sec,
            'batches_per_sec': batches_per_sec,
            'successful_batches': len(batch_times)
        }
        
        logger.info(f"=== Efficient Benchmark Results ===")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Total samples: {total_samples}")
        logger.info(f"Average batch time: {avg_batch_time:.3f}s ± {std_batch_time:.3f}s")
        logger.info(f"Throughput: {samples_per_sec:.1f} samples/sec")
        logger.info(f"Batch rate: {batches_per_sec:.2f} batches/sec")
        logger.info(f"Successful batches: {len(batch_times)}/{num_batches}")
        logger.info(f"===================================")
        
        return results
    
    def __iter__(self):
        """Make the data loader iterable."""
        return iter(self.dataloader)
    
    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        return len(self.dataloader)

# Keep the old classes for backward compatibility
AccelDataset = EfficientAccelDataset
AccelDataLoader = EfficientAccelDataLoader

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
