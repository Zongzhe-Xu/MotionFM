import os
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
import matplotlib.pyplot as plt
import time
from datetime import datetime, timedelta
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ActiGraph GT3X+ specifications
ACTIGRAPH_MAX_G = 6.0  # Maximum dynamic range in g
ACTIGRAPH_MIN_G = -6.0  # Minimum dynamic range in g
ACTIGRAPH_RESOLUTION = 4096  # 12-bit resolution
ACTIGRAPH_ACCURACY = 0.05  # ±0.05g accuracy

@dataclass
class DataFileInfo:
    """Information about a data file"""
    patient_id: str
    file_path: str
    timestamp: str  # Extracted from filename
    hour: int  # Hour of day (0-23)
    day_of_week: int  # 0=Monday, 6=Sunday
    file_size: int  # File size in bytes

def discover_patient_data(data_dir: str = "./patient_data") -> Dict[str, List[DataFileInfo]]:
    """
    Discover all patient data files and organize them by patient_id.
    
    Args:
        data_dir: Directory containing patient data folders
        
    Returns:
        Dictionary mapping patient_id to list of DataFileInfo objects
    """
    patient_files = {}
    
    if not os.path.exists(data_dir):
        logger.error(f"Data directory {data_dir} does not exist")
        return patient_files
    
    # Pattern to match sensor CSV files
    pattern = r'^[\w-]+\.[\w-]+\.(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{3})-P\d{4}\.sensor\.csv$'
    
    for patient_id in os.listdir(data_dir):
        patient_dir = os.path.join(data_dir, patient_id)
        if not os.path.isdir(patient_dir):
            continue
            
        patient_files[patient_id] = []
        
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
                patient_files[patient_id].append(file_info)
                patient_files[patient_id].sort(key=lambda x: x.timestamp)
        
        logger.info(f"Found {len(patient_files[patient_id])} files for patient {patient_id}")
    
    return patient_files

def load_accel_data(file_path: str, start_idx: Optional[int] = None, 
                   end_idx: Optional[int] = None, downsampling_ratio: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load accelerometer data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        start_idx: Starting index for slicing (optional)
        end_idx: Ending index for slicing (optional)
        downsampling_ratio: Downsampling ratio (1.0 = no downsampling, 2.0 = half the points, etc.)
        
    Returns:
        Tuple of (timestamps, accel_data) where:
        - timestamps: numpy array of timestamps
        - accel_data: numpy array of shape (N, 3) for X, Y, Z accelerations
    """
    try:
        df = pd.read_csv(file_path)
        
        # Extract timestamps and accelerometer data
        timestamps = pd.to_datetime(df['HEADER_TIMESTAMP'], format='%Y-%m-%d %H:%M:%S.%f').values
        accel_data = df[['X', 'Y', 'Z']].values.astype(np.float32)
        
        # Apply slicing if specified
        if start_idx is not None or end_idx is not None:
            timestamps = timestamps[start_idx:end_idx]
            accel_data = accel_data[start_idx:end_idx]
        
        # Apply downsampling with interpolation if specified
        if downsampling_ratio > 1.0:
            # Calculate target number of points
            original_length = len(timestamps)
            target_length = int(original_length / downsampling_ratio)
            
            if target_length > 0:
                # Create interpolation indices
                original_indices = np.arange(original_length)
                target_indices = np.linspace(0, original_length - 1, target_length)
                
                # Interpolate timestamps (convert to numeric for interpolation)
                timestamp_numeric = timestamps.astype(np.int64)
                interpolated_timestamps_numeric = np.interp(target_indices, original_indices, timestamp_numeric)
                timestamps = pd.to_datetime(interpolated_timestamps_numeric, unit='ns').values
                
                # Interpolate accelerometer data
                accel_data = np.array([
                    np.interp(target_indices, original_indices, accel_data[:, i])
                    for i in range(3)
                ]).T
        
        return timestamps, accel_data
        
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return np.array([]), np.array([])

def create_sampling_weights(patient_files: Dict[str, List[DataFileInfo]], 
                          patient_weights: Optional[Dict[str, float]] = None,
                          hour_weights: Optional[Dict[int, float]] = None,
                          day_weights: Optional[Dict[int, float]] = None) -> Dict[str, List[float]]:
    """
    Create sampling weights for each file based on patient, hour, and day preferences.
    
    Args:
        patient_files: Dictionary mapping patient_id to list of DataFileInfo
        patient_weights: Optional weights for each patient (default: uniform)
        hour_weights: Optional weights for each hour of day (default: uniform)
        day_weights: Optional weights for each day of week (default: uniform)
        
    Returns:
        Dictionary mapping patient_id to list of weights for each file
    """
    # Default weights (uniform) - vectorized creation
    if patient_weights is None:
        patient_weights = {pid: 1.0 for pid in patient_files.keys()}
    
    if hour_weights is None:
        hour_weights = {hour: 1.0 for hour in range(24)}
    
    if day_weights is None:
        day_weights = {day: 1.0 for day in range(7)}
    
    # Vectorized weight calculation
    sampling_weights = {}
    
    for patient_id, files in patient_files.items():
        patient_weight = patient_weights.get(patient_id, 1.0)
        
        # Vectorized extraction of hours and days
        hours = np.array([file_info.hour for file_info in files])
        days = np.array([file_info.day_of_week for file_info in files])
        
        # Vectorized weight lookup using numpy operations
        hour_weights_array = np.array([hour_weights.get(h, 1.0) for h in hours])
        day_weights_array = np.array([day_weights.get(d, 1.0) for d in days])
        
        # Vectorized combined weight calculation
        combined_weights = patient_weight * hour_weights_array * day_weights_array
        
        sampling_weights[patient_id] = combined_weights.tolist()
    
    return sampling_weights

def normalize_weights(weights: List[float]) -> List[float]:
    """Normalize weights to sum to 1.0 using vectorized operations"""
    weights_array = np.array(weights)
    total = np.sum(weights_array)
    if total == 0:
        return [1.0 / len(weights)] * len(weights)
    return (weights_array / total).tolist()

def sample_file_indices(patient_files: Dict[str, List[DataFileInfo]],
                       sampling_weights: Dict[str, List[float]],
                       num_samples: int,
                       replace: bool = True,
                       clear_cache: bool = True,
                       max_cache_size: int = 10000) -> List[Tuple[str, int]]:
    """
    Sample patient IDs and time windows based on weights.
    
    Args:
        patient_files: Dictionary mapping patient_id to list of DataFileInfo
        sampling_weights: Dictionary mapping patient_id to list of weights
        num_samples: Number of samples to generate
        replace: Whether to sample with replacement
        clear_cache: Whether to clear the file length cache before processing
        max_cache_size: Maximum number of cached file lengths (LRU behavior)
        
    Returns:
        List of (patient_id, start_offset) tuples where start_offset is the global index
        within the patient's continuous timeline
    """
    start_total = time.time()
    
    # Global cache for file lengths to avoid repeated I/O
    if not hasattr(sample_file_indices, '_file_length_cache'):
        sample_file_indices._file_length_cache = {}
        sample_file_indices._cache_access_order = []
    
    # Clear cache if requested
    if clear_cache:
        sample_file_indices._file_length_cache.clear()
        sample_file_indices._cache_access_order.clear()
    
    # Vectorized patient-level weight calculation
    start_weights = time.time()
    patient_weights = {}
    for patient_id, files in patient_files.items():
        weights = sampling_weights.get(patient_id, [1.0] * len(files))
        patient_weights[patient_id] = np.mean(weights)  # Average weight across files
    
    # Convert to numpy arrays for vectorized operations
    patient_ids = list(patient_weights.keys())
    weights_array = np.array([patient_weights[pid] for pid in patient_ids])
    
    # Vectorized weight normalization
    normalized_weights = weights_array / np.sum(weights_array)
    weights_time = time.time() - start_weights
    
    # Sample patient IDs using numpy
    start_sampling = time.time()
    if replace:
        patient_indices = np.random.choice(len(patient_ids), size=num_samples, p=normalized_weights)
    else:
        # For sampling without replacement, we need to handle the case where num_samples > len(patients)
        if num_samples > len(patient_ids):
            logger.warning(f"Requested {num_samples} samples but only {len(patient_ids)} patients available. Using all patients.")
            patient_indices = np.arange(len(patient_ids))
        else:
            patient_indices = np.random.choice(len(patient_ids), size=num_samples, p=normalized_weights, replace=False)
    sampling_time = time.time() - start_sampling
    
    # For each selected patient, sample a start offset within their timeline
    # based on the time-of-day weights
    start_offsets = time.time()
    samples = []
    for patient_idx in patient_indices:
        patient_id = patient_ids[patient_idx]
        files = patient_files[patient_id]
        file_weights = sampling_weights.get(patient_id, [1.0] * len(files))
        
        # Calculate total length and cumulative weights for this patient
        total_length = 0
        cumulative_weights = []
        cumulative_lengths = []
        
        for i, file_info in enumerate(files):
            # Use cached file length if available
            cache_key = file_info.file_path
            if cache_key not in sample_file_indices._file_length_cache:
                try:
                    # Use faster method to get file length - read only header and count lines
                    with open(file_info.file_path, 'r') as f:
                        # Skip header and count remaining lines
                        next(f)  # Skip header
                        line_count = sum(1 for _ in f)
                    
                    # Add to cache with LRU management
                    if len(sample_file_indices._file_length_cache) >= max_cache_size:
                        # Remove least recently used item
                        lru_key = sample_file_indices._cache_access_order.pop(0)
                        del sample_file_indices._file_length_cache[lru_key]
                    
                    sample_file_indices._file_length_cache[cache_key] = line_count
                    sample_file_indices._cache_access_order.append(cache_key)
                except Exception:
                    # Add to cache with LRU management even for failed reads
                    if len(sample_file_indices._file_length_cache) >= max_cache_size:
                        # Remove least recently used item
                        lru_key = sample_file_indices._cache_access_order.pop(0)
                        del sample_file_indices._file_length_cache[lru_key]
                    
                    sample_file_indices._file_length_cache[cache_key] = 0
                    sample_file_indices._cache_access_order.append(cache_key)
            else:
                # Update access order for LRU
                if cache_key in sample_file_indices._cache_access_order:
                    sample_file_indices._cache_access_order.remove(cache_key)
                sample_file_indices._cache_access_order.append(cache_key)
            
            file_length = sample_file_indices._file_length_cache[cache_key]
            cumulative_lengths.append(total_length)
            total_length += file_length
            
            # Weight for this file (based on time of day)
            file_weight = file_weights[i] if i < len(file_weights) else 1.0
            cumulative_weights.append(file_weight)
        
        if total_length == 0:
            # If patient has no data, skip
            continue
        
        # Vectorized normalization of file weights
        cumulative_weights_array = np.array(cumulative_weights)
        if np.sum(cumulative_weights_array) > 0:
            normalized_file_weights = cumulative_weights_array / np.sum(cumulative_weights_array)
        else:
            normalized_file_weights = np.full(len(files), 1.0 / len(files))
        
        # Sample a file based on weights, then sample within that file
        file_idx = np.random.choice(len(files), p=normalized_file_weights)
        file_start = cumulative_lengths[file_idx]
        
        # Get the length of the selected file (already cached)
        file_length = sample_file_indices._file_length_cache[files[file_idx].file_path]
        
        # Sample a start offset within this file
        if file_length > 0:
            start_offset = file_start + np.random.randint(0, max(1, file_length))
        else:
            start_offset = file_start
        
        samples.append((patient_id, start_offset))
    
    offsets_time = time.time() - start_offsets
    total_time = time.time() - start_total
    
    # Log timing details if it's taking a while
    if total_time > 1.0:
        logger.debug(f"sample_file_indices timing: weights={weights_time:.3f}s, sampling={sampling_time:.3f}s, offsets={offsets_time:.3f}s, total={total_time:.3f}s")
    
    return samples

def clear_file_length_cache():
    """Clear the global file length cache used by sample_file_indices."""
    if hasattr(sample_file_indices, '_file_length_cache'):
        sample_file_indices._file_length_cache.clear()
        sample_file_indices._cache_access_order.clear()
        logger.info("File length cache cleared")

def get_cache_stats():
    """Get statistics about the file length cache."""
    if hasattr(sample_file_indices, '_file_length_cache'):
        cache_size = len(sample_file_indices._file_length_cache)
        total_memory = sum(len(str(k)) + 8 for k in sample_file_indices._file_length_cache.keys())  # Rough estimate
        return {
            'cache_size': cache_size,
            'estimated_memory_bytes': total_memory,
            'cache_keys': list(sample_file_indices._file_length_cache.keys())[:10]  # First 10 keys
        }
    return {'cache_size': 0, 'estimated_memory_bytes': 0, 'cache_keys': []}

def extract_sequence_from_file(file_info: DataFileInfo, 
                             context_len: int,
                             start_offset: Optional[int] = None,
                             downsampling_ratio: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a sequence of specified length from a file.
    Args:
        file_info: DataFileInfo object containing file details
        context_len: Number of output points after downsampling
        start_offset: Optional starting offset (if None, random offset is chosen)
        downsampling_ratio: Downsampling ratio (1.0 = no downsampling, 2.0 = half the points, etc.)
    Returns:
        Tuple of (timestamps, accel_data) for the extracted sequence
    """
    # Compute how many raw points to extract
    raw_context_len = int(context_len * downsampling_ratio)
    timestamps, accel_data = load_accel_data(file_info.file_path)

    if len(accel_data) == 0:
        return np.array([]), np.array([])

    # Determine start index
    if start_offset is None:
        max_start = max(0, len(accel_data) - raw_context_len)
        if max_start == 0:
            start_offset = 0
        else:
            start_offset = np.random.randint(0, max_start + 1)

    # Extract raw sequence
    end_offset = min(start_offset + raw_context_len, len(accel_data))
    sequence_timestamps = timestamps[start_offset:end_offset]
    sequence_accel = accel_data[start_offset:end_offset]

    # Downsample with interpolation if needed
    if downsampling_ratio > 1.0 and len(sequence_accel) > 1:
        original_indices = np.arange(len(sequence_accel))
        target_indices = np.linspace(0, len(sequence_accel) - 1, context_len)
        # Interpolate timestamps
        timestamp_numeric = sequence_timestamps.astype(np.int64)
        interpolated_timestamps_numeric = np.interp(target_indices, original_indices, timestamp_numeric)
        sequence_timestamps = pd.to_datetime(interpolated_timestamps_numeric).values
        # Interpolate accel data
        sequence_accel = np.array([
            np.interp(target_indices, original_indices, sequence_accel[:, i])
            for i in range(3)
        ]).T
    else:
        # If not enough points, pad as before
        if len(sequence_accel) < context_len:
            pad_length = context_len - len(sequence_accel)
            sequence_accel = np.pad(sequence_accel, ((pad_length, 0), (0, 0)), mode='constant', constant_values=0)
            if len(sequence_timestamps) > 0:
                first_timestamp = sequence_timestamps[0]
                sequence_timestamps = np.pad(sequence_timestamps, (pad_length, 0), mode='constant', constant_values=first_timestamp)

    # Ensure output is exactly context_len
    if len(sequence_accel) > context_len:
        sequence_accel = sequence_accel[-context_len:]
        sequence_timestamps = sequence_timestamps[-context_len:]

    return sequence_timestamps, sequence_accel

def get_patient_metadata(patient_id: str) -> Dict[str, Any]:
    """
    Get metadata for a patient. This can be extended to load from external sources.
    
    Args:
        patient_id: Patient identifier
        
    Returns:
        Dictionary containing patient metadata
    """
    # Placeholder for patient metadata
    # In a real implementation, this would load from a database or metadata file
    return {
        "patient_id": patient_id,
        "age": None,
        "gender": None,
        "height": None,
        "weight": None,
        # Add other relevant metadata fields
    }

def visualize_from_csv(file_path: str, x_lim: Optional[Tuple[int, int]] = None):
    """
    Visualize accelerometer data from a CSV file.
    
    Args:
        file_path: Path to the CSV file
        start_idx: Starting index for slicing (optional)
        end_idx: Ending index for slicing (optional)
    """
    df = pd.read_csv(file_path)
    print(len(df))
    magnitude = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2)
    fig, axes = plt.subplots(4, 1, figsize=(12, 12))
    axes[0].plot(df['X'])
    axes[0].set_title('X-axis Acceleration')
    axes[0].set_ylabel('Acceleration (g)')
    axes[1].plot(df['Y'])
    axes[1].set_title('Y-axis Acceleration')
    axes[1].set_ylabel('Acceleration (g)')
    axes[2].plot(df['Z'])
    axes[3].plot(magnitude)
    axes[3].set_title('Magnitude')
    axes[3].set_ylabel('Magnitude (g)')
    if x_lim is not None:
        axes[0].set_xlim(x_lim)
        axes[1].set_xlim(x_lim)
        axes[2].set_xlim(x_lim)
        axes[3].set_xlim(x_lim)
    plt.tight_layout()
    plt.show()

def extract_sequence_from_patient_files(patient_files: list, context_len: int, start_offset: Optional[int] = None, downsampling_ratio: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract a continuous sequence of specified length from a patient's files, spanning file boundaries if needed.
    Args:
        patient_files: List of DataFileInfo objects for a patient, sorted by time
        context_len: Number of output points after downsampling
        start_offset: Optional starting offset in the global timeline (if None, random offset is chosen)
        downsampling_ratio: Downsampling ratio (1.0 = no downsampling, 2.0 = half the points, etc.)
    Returns:
        Tuple of (timestamps, accel_data) for the extracted sequence
    """
    # Compute how many raw points to extract
    raw_context_len = int(context_len * downsampling_ratio)

    # Global cache for file lengths to avoid repeated I/O across function calls
    if not hasattr(extract_sequence_from_patient_files, '_global_file_cache'):
        extract_sequence_from_patient_files._global_file_cache = {}

    file_lengths = []
    for file_info in patient_files:
        cache_key = file_info.file_path
        if cache_key not in extract_sequence_from_patient_files._global_file_cache:
            try:
                with open(file_info.file_path, 'r') as f:
                    next(f)
                    line_count = sum(1 for _ in f)
                extract_sequence_from_patient_files._global_file_cache[cache_key] = line_count
            except Exception as e:
                extract_sequence_from_patient_files._global_file_cache[cache_key] = 0
        file_lengths.append(extract_sequence_from_patient_files._global_file_cache[cache_key])
    file_lengths_array = np.array(file_lengths)
    total_length = np.sum(file_lengths_array)

    if total_length == 0:
        return np.array([]), np.array([])

    # Determine start index
    if start_offset is None:
        max_start = max(0, total_length - raw_context_len)
        if max_start == 0:
            start_offset = 0
        else:
            start_offset = np.random.randint(0, max_start + 1)

    # Find which file to start in
    cumulative_lengths = np.cumsum(np.concatenate(([0], file_lengths_array[:-1])))
    file_idx = np.searchsorted(cumulative_lengths, start_offset, side='right') - 1
    offset_in_file = start_offset - cumulative_lengths[file_idx]

    timestamps = []
    accel_data = []
    remaining = raw_context_len

    while file_idx < len(patient_files) and remaining > 0:
        file_info = patient_files[file_idx]
        start_row = offset_in_file
        nrows = min(file_lengths[file_idx] - start_row, remaining)
        try:
            # Optimized pandas reading with better chunking strategy
            if nrows > 10000:  # For very large chunks, use larger chunk size
                chunks = []
                rows_read = 0
                chunk_size = 10000
                
                while rows_read < nrows:
                    current_chunk_size = min(chunk_size, nrows - rows_read)
                    chunk = pd.read_csv(
                        file_info.file_path, 
                        skiprows=range(1, 1 + start_row + rows_read), 
                        nrows=current_chunk_size
                    )
                    chunks.append(chunk)
                    rows_read += current_chunk_size
                
                df = pd.concat(chunks, ignore_index=True)
            elif nrows > 5000:  # For large chunks, use medium chunk size
                chunks = []
                rows_read = 0
                chunk_size = 5000
                
                while rows_read < nrows:
                    current_chunk_size = min(chunk_size, nrows - rows_read)
                    chunk = pd.read_csv(
                        file_info.file_path, 
                        skiprows=range(1, 1 + start_row + rows_read), 
                        nrows=current_chunk_size
                    )
                    chunks.append(chunk)
                    rows_read += current_chunk_size
                
                df = pd.concat(chunks, ignore_index=True)
            elif nrows > 1000:  # For medium chunks, use smaller chunk size
                chunks = []
                rows_read = 0
                chunk_size = 1000
                
                while rows_read < nrows:
                    current_chunk_size = min(chunk_size, nrows - rows_read)
                    chunk = pd.read_csv(
                        file_info.file_path, 
                        skiprows=range(1, 1 + start_row + rows_read), 
                        nrows=current_chunk_size
                    )
                    chunks.append(chunk)
                    rows_read += current_chunk_size
                
                df = pd.concat(chunks, ignore_index=True)
            else:
                # For small chunks, read directly
                df = pd.read_csv(file_info.file_path, skiprows=range(1, 1+start_row), nrows=nrows)
            
            ts = pd.to_datetime(df['HEADER_TIMESTAMP'], format='%Y-%m-%d %H:%M:%S.%f').values
            acc = df[['X', 'Y', 'Z']].values.astype(np.float32)
        except Exception as e:
            ts = np.array([])
            acc = np.zeros((0, 3), dtype=np.float32)
        timestamps.append(ts)
        accel_data.append(acc)
        remaining -= nrows
        file_idx += 1
        offset_in_file = 0

    if timestamps:
        timestamps = np.concatenate(timestamps)
    else:
        timestamps = np.array([])
    if accel_data:
        accel_data = np.concatenate(accel_data)
    else:
        accel_data = np.zeros((0, 3), dtype=np.float32)

    # Downsample with interpolation if needed
    if downsampling_ratio > 1.0 and len(accel_data) > 1:
        if len(accel_data) < raw_context_len:
            pad_length = raw_context_len - len(accel_data)
            accel_data = np.pad(accel_data, ((pad_length, 0), (0, 0)), mode='constant', constant_values=0.57735)
            if len(timestamps) > 0:
                first_timestamp = timestamps[0]
                timestamps = np.pad(timestamps, (pad_length, 0), mode='constant', constant_values=first_timestamp)

        original_indices = np.arange(len(accel_data))
        target_indices = np.linspace(0, len(accel_data) - 1, context_len)
        timestamp_numeric = timestamps.astype(np.int64)
        interpolated_timestamps_numeric = np.interp(target_indices, original_indices, timestamp_numeric)
        timestamps = pd.to_datetime(interpolated_timestamps_numeric).values
        accel_data = np.array([
            np.interp(target_indices, original_indices, accel_data[:, i])
            for i in range(3)
        ]).T
    else:
        # If not enough points, pad as before
        if len(accel_data) < context_len:
            pad_length = context_len - len(accel_data)
            accel_data = np.pad(accel_data, ((pad_length, 0), (0, 0)), mode='constant', constant_values=0.57735)
            if len(timestamps) > 0:
                first_timestamp = timestamps[0]
                timestamps = np.pad(timestamps, (pad_length, 0), mode='constant', constant_values=first_timestamp)

    # Ensure output is exactly context_len
    if len(accel_data) > context_len:
        accel_data = accel_data[-context_len:]
        timestamps = timestamps[-context_len:]

    return timestamps, accel_data

def visualize_entire_days(patient_id: str, data_dir: str = "./patient_data", downsampling_ratio: float = 1.0):
    """
    Visualize the entire day of a patient's data.
    
    Args:
        patient_id: Patient identifier
        data_dir: Directory containing patient data
        downsampling_ratio: Downsampling ratio (1.0 = no downsampling, 2.0 = half the points, etc.)
    """
    patient_files = discover_patient_data(data_dir)[patient_id]
    patient_files.sort(key=lambda x: x.timestamp)
    print(patient_files[0].timestamp)
    print(patient_files[1].timestamp)

    fig, axes = plt.subplots(3,1, figsize=(12, 8))

    initial_time = patient_files[0].timestamp
    #convert the initial time to a datetime object
    initial_time = datetime.strptime(initial_time, "%Y-%m-%d %H:%M:%S.%f")
    current_time = initial_time
    current_index  = 0
    day_data_X = []
    day_data_Y = []
    day_data_Z = []
    day_data_time = []
    while current_index < len(patient_files):
        print(f"processing {current_index} out of {len(patient_files)}")
        while current_time - initial_time < timedelta(days=1) and current_index < len(patient_files):
            day_data = pd.read_csv(patient_files[current_index].file_path)
            # Apply downsampling with interpolation if specified
            if downsampling_ratio > 1.0:
                # Calculate target number of points
                original_length = len(day_data)
                target_length = int(original_length / downsampling_ratio)
                
                if target_length > 0:
                    # Create interpolation indices
                    original_indices = np.arange(original_length)
                    target_indices = np.linspace(0, original_length - 1, target_length)
                    
                    # Interpolate each column
                    interpolated_data = {}
                    for col in day_data.columns:
                        if col == 'HEADER_TIMESTAMP':
                            # For timestamps, convert to numeric, interpolate, then convert back
                            timestamp_numeric = pd.to_datetime(day_data[col]).astype(np.int64)
                            interpolated_timestamps_numeric = np.interp(target_indices, original_indices, timestamp_numeric)
                            interpolated_data[col] = pd.to_datetime(interpolated_timestamps_numeric)
                        else:
                            # For numeric columns, interpolate directly
                            interpolated_data[col] = np.interp(target_indices, original_indices, day_data[col])
                    
                    day_data = pd.DataFrame(interpolated_data)
            day_data_X.append(day_data['X'])
            day_data_Y.append(day_data['Y'])
            day_data_Z.append(day_data['Z'])
            day_data_time.append(day_data['HEADER_TIMESTAMP'])

            current_index += 1
            if current_index == len(patient_files):
                break
            current_time = patient_files[current_index].timestamp
            current_time = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S.%f")
        day_data_X = np.concatenate(day_data_X)
        day_data_Y = np.concatenate(day_data_Y)
        day_data_Z = np.concatenate(day_data_Z)
        day_data_time = np.concatenate(day_data_time)
        day_data_time = pd.to_datetime(day_data_time)
        # print(day_data_time)
        # Extract only the time component (hours, minutes, seconds) for overlay plotting
        # Convert to pandas Series and extract time component
        day_data_time_series = pd.Series(day_data_time)
        time_of_day = day_data_time_series.dt.time.map(lambda x: x.hour * 3600 + x.minute * 60 + x.second + x.microsecond / 1000000)
        

        
        # axes[0].plot(time_of_day, day_data_X, label=f"Day {initial_time.day}")
        # axes[1].plot(time_of_day, day_data_Y, label=f"Day {initial_time.day}")
        # axes[2].plot(time_of_day, day_data_Z, label=f"Day {initial_time.day}")

        axes[0].plot(day_data_X)
        axes[1].plot(day_data_Y)
        axes[2].plot(day_data_Z)
        
        # Add time labels to x-axis
        if len(time_of_day) > 0:
            # Create time labels at regular intervals
            step = max(1, len(time_of_day) // 12)  # Show ~8 time labels
            x_positions = []
            time_labels = []
            
            for i in range(0, len(time_of_day), step):
                x_positions.append(i)
                time_str = f"{int(time_of_day.iloc[i]//3600):02d}:{int((time_of_day.iloc[i]%3600)//60):02d}"
                time_labels.append(time_str)
            
            # Set custom x-axis ticks and labels
            for ax in axes:
                ax.set_xticks(x_positions)
                ax.set_xticklabels(time_labels, rotation=45, ha='right')
        initial_time = current_time
        day_data_X = []
        day_data_Y = []
        day_data_Z = []
        day_data_time = []
    

    
    # Add time range to x-axis label
    if 'time_of_day' in locals() and len(time_of_day) > 0:
        start_time = f"{int(time_of_day.iloc[0]//3600):02d}:{int((time_of_day.iloc[0]%3600)//60):02d}"
        end_time = f"{int(time_of_day.iloc[-1]//3600):02d}:{int((time_of_day.iloc[-1]%3600)//60):02d}"
        x_label = f'Data Points ({start_time} - {end_time})'
    else:
        x_label = 'Data Points'
    
    axes[0].set_title('X-axis Acceleration')
    axes[0].set_ylabel('Acceleration (g)')
    axes[0].set_xlabel(x_label)
    axes[1].set_title('Y-axis Acceleration')
    axes[1].set_ylabel('Acceleration (g)')
    axes[1].set_xlabel(x_label)
    axes[2].set_title('Z-axis Acceleration')
    axes[2].set_ylabel('Acceleration (g)')
    axes[2].set_xlabel(x_label)
    
    plt.tight_layout()
    plt.show()

def visualize_average_entire_days(patient_id: str, data_dir: str = "./patient_data", downsampling_ratio: float = 1.0):
    """
    Visualize the entire day of a patient's data.
    
    Args:
        patient_id: Patient identifier
        data_dir: Directory containing patient data
        downsampling_ratio: Downsampling ratio (1.0 = no downsampling, 2.0 = half the points, etc.)
    """
    patient_files = discover_patient_data(data_dir)[patient_id]
    patient_files.sort(key=lambda x: x.timestamp)

    fig, axes = plt.subplots(1,1, figsize=(12, 8))

    initial_time = patient_files[0].timestamp
    #convert the initial time to a datetime object
    initial_time = datetime.strptime(initial_time, "%Y-%m-%d %H:%M:%S.%f")
    current_time = initial_time
    current_index  = 0
    day_data_X = []
    day_data_Y = []
    day_data_Z = []
    day_data_time = []
    while current_index < len(patient_files):
        while current_time - initial_time < timedelta(days=1) and current_index < len(patient_files):
            day_data = pd.read_csv(patient_files[current_index].file_path)
            # Apply downsampling with interpolation if specified
            if downsampling_ratio > 1.0:
                # Calculate target number of points
                original_length = len(day_data)
                target_length = int(original_length / downsampling_ratio)
                
                if target_length > 0:
                    # Create interpolation indices
                    original_indices = np.arange(original_length)
                    target_indices = np.linspace(0, original_length - 1, target_length)
                    
                    # Interpolate each column
                    interpolated_data = {}
                    for col in day_data.columns:
                        if col == 'HEADER_TIMESTAMP':
                            # For timestamps, convert to numeric, interpolate, then convert back
                            timestamp_numeric = pd.to_datetime(day_data[col]).astype(np.int64)
                            interpolated_timestamps_numeric = np.interp(target_indices, original_indices, timestamp_numeric)
                            interpolated_data[col] = pd.to_datetime(interpolated_timestamps_numeric)
                        else:
                            # For numeric columns, interpolate directly
                            interpolated_data[col] = np.interp(target_indices, original_indices, day_data[col])
                    
                    day_data = pd.DataFrame(interpolated_data)
            day_data_X.append(day_data['X'])
            day_data_Y.append(day_data['Y'])
            day_data_Z.append(day_data['Z'])
            day_data_time.append(day_data['HEADER_TIMESTAMP'])

            current_index += 1
            if current_index == len(patient_files):
                break
            current_time = patient_files[current_index].timestamp
            current_time = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S.%f")
        day_data_X = np.concatenate(day_data_X)
        day_data_Y = np.concatenate(day_data_Y)
        day_data_Z = np.concatenate(day_data_Z)
        day_data_time = np.concatenate(day_data_time)
        day_data_time = pd.to_datetime(day_data_time)
        
        # Extract only the time component (hours, minutes, seconds) for overlay plotting
        # Convert to pandas Series and extract time component
        day_data_time_series = pd.Series(day_data_time)
        time_of_day = day_data_time_series.dt.time.map(lambda x: x.hour * 3600 + x.minute * 60 + x.second + x.microsecond / 1000000)
        
        # Stack the 1D arrays into a 2D array for mean calculation
        all_data = np.column_stack([day_data_X, day_data_Y, day_data_Z])
        data_mean = np.mean(all_data, axis=1)
        axes.plot(data_mean, label=f"Day {initial_time.day}")
        
        # Add time labels to x-axis
        if len(time_of_day) > 0:
            # Create time labels at regular intervals
            step = max(1, len(time_of_day) // 12)  # Show ~12 time labels
            x_positions = []
            time_labels = []
            
            for i in range(0, len(time_of_day), step):
                x_positions.append(i)
                time_str = f"{int(time_of_day.iloc[i]//3600):02d}:{int((time_of_day.iloc[i]%3600)//60):02d}"
                time_labels.append(time_str)
            
            # Set custom x-axis ticks and labels
            axes.set_xticks(x_positions)
            axes.set_xticklabels(time_labels, rotation=45, ha='right')
        
        initial_time = current_time
        day_data_X = []
        day_data_Y = []
        day_data_Z = []
        day_data_time = []
    
    # Add time range to x-axis label
    if 'time_of_day' in locals() and len(time_of_day) > 0:
        start_time = f"{int(time_of_day.iloc[0]//3600):02d}:{int((time_of_day.iloc[0]%3600)//60):02d}"
        end_time = f"{int(time_of_day.iloc[-1]//3600):02d}:{int((time_of_day.iloc[-1]%3600)//60):02d}"
        x_label = f'Data Points ({start_time} - {end_time})'
    else:
        x_label = 'Data Points'
    
    axes.set_xlabel(x_label)
    axes.set_ylabel('Average Acceleration (g)')
    axes.set_title('Average Daily Activity Pattern')
    axes.legend(loc="upper right")
    plt.tight_layout()
    plt.show()
    
def calculate_window_statistics(patient_id: str, window_hours: float = 2.0, data_dir: str = "./patient_data", combine_days: bool = True, downsampling_ratio: float = 1.0):
    """
    Calculate mean and standard deviation of accelerometer data over sliding time windows.
    
    Args:
        patient_id: Patient identifier
        window_hours: Length of time window in hours (default: 2.0)
        data_dir: Directory containing patient data
        combine_days: If True, calculate statistics across all days combined.
                     If False, calculate statistics separately for each day.
        downsampling_ratio: Downsampling ratio (1.0 = no downsampling, 2.0 = half the points, etc.)
        
    Returns:
        pandas.DataFrame with columns:
        - 'window_start': Window start times (HH:MM format)
        - 'window_end': Window end times (HH:MM format)
        - 'mean_x': Mean X-axis accelerations for each window
        - 'mean_y': Mean Y-axis accelerations for each window
        - 'mean_z': Mean Z-axis accelerations for each window
        - 'std_x': Standard deviations for X-axis accelerations
        - 'std_y': Standard deviations for Y-axis accelerations
        - 'std_z': Standard deviations for Z-axis accelerations
        - 'mean_magnitude': Mean acceleration magnitudes for each window
        - 'std_magnitude': Standard deviations for acceleration magnitudes
        - 'day' (only if combine_days=False): Day number for each window
    """
    patient_files = discover_patient_data(data_dir)[patient_id]
    patient_files.sort(key=lambda x: x.timestamp)
    
    # Convert window_hours to seconds
    window_seconds = window_hours * 3600
    
    # Initialize results
    window_starts = []
    window_ends = []
    means_x = []
    means_y = []
    means_z = []
    stds_x = []
    stds_y = []
    stds_z = []
    means_magnitude = []
    stds_magnitude = []
    days = []  # For per-day mode
    
    if combine_days:
        # Combined days mode - collect all data first
        all_data_X = []
        all_data_Y = []
        all_data_Z = []
        all_data_time = []
        
        initial_time = datetime.strptime(patient_files[0].timestamp, "%Y-%m-%d %H:%M:%S.%f")
        current_time = initial_time
        current_index = 0
        
        while current_index < len(patient_files):
            # Collect data for one day
            day_data_X = []
            day_data_Y = []
            day_data_Z = []
            day_data_time = []
            
            while current_time - initial_time < timedelta(days=1) and current_index < len(patient_files):
                day_data = pd.read_csv(patient_files[current_index].file_path)
                # Apply downsampling with interpolation if specified
                if downsampling_ratio > 1.0:
                    # Calculate target number of points
                    original_length = len(day_data)
                    target_length = int(original_length / downsampling_ratio)
                    
                    if target_length > 0:
                        # Create interpolation indices
                        original_indices = np.arange(original_length)
                        target_indices = np.linspace(0, original_length - 1, target_length)
                        
                        # Interpolate each column
                        interpolated_data = {}
                        for col in day_data.columns:
                            if col == 'HEADER_TIMESTAMP':
                                # For timestamps, convert to numeric, interpolate, then convert back
                                timestamp_numeric = pd.to_datetime(day_data[col]).astype(np.int64)
                                interpolated_timestamps_numeric = np.interp(target_indices, original_indices, timestamp_numeric)
                                interpolated_data[col] = pd.to_datetime(interpolated_timestamps_numeric)
                            else:
                                # For numeric columns, interpolate directly
                                interpolated_data[col] = np.interp(target_indices, original_indices, day_data[col])
                        
                        day_data = pd.DataFrame(interpolated_data)
                day_data_X.append(day_data['X'])
                day_data_Y.append(day_data['Y'])
                day_data_Z.append(day_data['Z'])
                day_data_time.append(day_data['HEADER_TIMESTAMP'])
                
                current_index += 1
                if current_index == len(patient_files):
                    break
                current_time = patient_files[current_index].timestamp
                current_time = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S.%f")
            
            if day_data_X:  # Only process if we have data
                # Concatenate day data and add to all data
                day_data_X = np.concatenate(day_data_X)
                day_data_Y = np.concatenate(day_data_Y)
                day_data_Z = np.concatenate(day_data_Z)
                day_data_time = np.concatenate(day_data_time)
                
                all_data_X.append(day_data_X)
                all_data_Y.append(day_data_Y)
                all_data_Z.append(day_data_Z)
                all_data_time.append(day_data_time)
            
            # Reset for next day
            initial_time = current_time
        
        if not all_data_X:  # No data found
            return pd.DataFrame()
        
        # Concatenate all days
        all_data_X = np.concatenate(all_data_X)
        all_data_Y = np.concatenate(all_data_Y)
        all_data_Z = np.concatenate(all_data_Z)
        all_data_time = np.concatenate(all_data_time)
        all_data_time = pd.to_datetime(all_data_time)
        
        # Convert to time of day in seconds
        all_data_time_series = pd.Series(all_data_time)
        time_of_day = all_data_time_series.dt.time.map(
            lambda x: x.hour * 3600 + x.minute * 60 + x.second + x.microsecond / 1000000
        )
        
        # Calculate acceleration magnitude
        magnitude = np.sqrt(all_data_X**2 + all_data_Y**2 + all_data_Z**2)
        
        # Create windows across the full 24-hour period (0:00 to 24:00)
        # This ensures we get consistent windows regardless of data availability
        window_start_seconds = 0  # Start at midnight
        while window_start_seconds < 24 * 3600:  # 24 hours in seconds
            window_end_seconds = window_start_seconds + window_seconds
            
            # Find data points within this window across all days
            mask = (time_of_day >= window_start_seconds) & (time_of_day < window_end_seconds)
            
            if np.sum(mask) > 0:  # Only process windows with data
                # Calculate statistics for this window
                window_x = all_data_X[mask]
                window_y = all_data_Y[mask]
                window_z = all_data_Z[mask]
                window_mag = magnitude[mask]
                
                # Store results
                window_starts.append(f"{int(window_start_seconds//3600):02d}:{int((window_start_seconds%3600)//60):02d}")
                window_ends.append(f"{int(window_end_seconds//3600):02d}:{int((window_end_seconds%3600)//60):02d}")
                
                means_x.append(np.mean(window_x))
                means_y.append(np.mean(window_y))
                means_z.append(np.mean(window_z))
                means_magnitude.append(np.mean(window_mag))
                
                stds_x.append(np.std(window_x))
                stds_y.append(np.std(window_y))
                stds_z.append(np.std(window_z))
                stds_magnitude.append(np.std(window_mag))
            
            # Move to next window
            window_start_seconds = window_end_seconds
    
    else:
        # Per-day mode - process each day separately
        initial_time = datetime.strptime(patient_files[0].timestamp, "%Y-%m-%d %H:%M:%S.%f")
        current_time = initial_time
        current_index = 0
        day_number = 1
        
        while current_index < len(patient_files):
            # Collect data for one day
            day_data_X = []
            day_data_Y = []
            day_data_Z = []
            day_data_time = []
            
            while current_time - initial_time < timedelta(days=1) and current_index < len(patient_files):
                day_data = pd.read_csv(patient_files[current_index].file_path)
                # Apply downsampling with interpolation if specified
                if downsampling_ratio > 1.0:
                    # Calculate target number of points
                    original_length = len(day_data)
                    target_length = int(original_length / downsampling_ratio)
                    
                    if target_length > 0:
                        # Create interpolation indices
                        original_indices = np.arange(original_length)
                        target_indices = np.linspace(0, original_length - 1, target_length)
                        
                        # Interpolate each column
                        interpolated_data = {}
                        for col in day_data.columns:
                            if col == 'HEADER_TIMESTAMP':
                                # For timestamps, convert to numeric, interpolate, then convert back
                                timestamp_numeric = pd.to_datetime(day_data[col]).astype(np.int64)
                                interpolated_timestamps_numeric = np.interp(target_indices, original_indices, timestamp_numeric)
                                interpolated_data[col] = pd.to_datetime(interpolated_timestamps_numeric)
                            else:
                                # For numeric columns, interpolate directly
                                interpolated_data[col] = np.interp(target_indices, original_indices, day_data[col])
                        
                        day_data = pd.DataFrame(interpolated_data)
                day_data_X.append(day_data['X'])
                day_data_Y.append(day_data['Y'])
                day_data_Z.append(day_data['Z'])
                day_data_time.append(day_data['HEADER_TIMESTAMP'])
                
                current_index += 1
                if current_index == len(patient_files):
                    break
                current_time = patient_files[current_index].timestamp
                current_time = datetime.strptime(current_time, "%Y-%m-%d %H:%M:%S.%f")
            
            if day_data_X:  # Only process if we have data
                # Concatenate day data
                day_data_X = np.concatenate(day_data_X)
                day_data_Y = np.concatenate(day_data_Y)
                day_data_Z = np.concatenate(day_data_Z)
                day_data_time = np.concatenate(day_data_time)
                day_data_time = pd.to_datetime(day_data_time)
                
                # Convert to time of day in seconds
                day_data_time_series = pd.Series(day_data_time)
                time_of_day = day_data_time_series.dt.time.map(
                    lambda x: x.hour * 3600 + x.minute * 60 + x.second + x.microsecond / 1000000
                )
                
                # Calculate acceleration magnitude
                magnitude = np.sqrt(day_data_X**2 + day_data_Y**2 + day_data_Z**2)
                
                # Process windows for this day
                # Create windows across the full 24-hour period (0:00 to 24:00)
                # This ensures we get consistent windows regardless of data availability
                window_start_seconds = 0  # Start at midnight
                while window_start_seconds < 24 * 3600:  # 24 hours in seconds
                    window_end_seconds = window_start_seconds + window_seconds
                    
                    # Find data points within this window
                    mask = (time_of_day >= window_start_seconds) & (time_of_day < window_end_seconds)
                    
                    if np.sum(mask) > 0:  # Only process windows with data
                        # Calculate statistics for this window
                        window_x = day_data_X[mask]
                        window_y = day_data_Y[mask]
                        window_z = day_data_Z[mask]
                        window_mag = magnitude[mask]
                        
                        # Store results
                        window_starts.append(f"{int(window_start_seconds//3600):02d}:{int((window_start_seconds%3600)//60):02d}")
                        window_ends.append(f"{int(window_end_seconds//3600):02d}:{int((window_end_seconds%3600)//60):02d}")
                        days.append(day_number)
                        
                        means_x.append(np.mean(window_x))
                        means_y.append(np.mean(window_y))
                        means_z.append(np.mean(window_z))
                        means_magnitude.append(np.mean(window_mag))
                        
                        stds_x.append(np.std(window_x))
                        stds_y.append(np.std(window_y))
                        stds_z.append(np.std(window_z))
                        stds_magnitude.append(np.std(window_mag))
                    
                    # Move to next window
                    window_start_seconds = window_end_seconds
            
            # Reset for next day
            initial_time = current_time
            day_number += 1
    
    # Create DataFrame
    if combine_days:
        df = pd.DataFrame({
            'window_start': window_starts,
            'window_end': window_ends,
            'mean_x': means_x,
            'mean_y': means_y,
            'mean_z': means_z,
            'std_x': stds_x,
            'std_y': stds_y,
            'std_z': stds_z,
            'mean_magnitude': means_magnitude,
            'std_magnitude': stds_magnitude
        })
    else:
        df = pd.DataFrame({
            'window_start': window_starts,
            'window_end': window_ends,
            'day': days,
            'mean_x': means_x,
            'mean_y': means_y,
            'mean_z': means_z,
            'std_x': stds_x,
            'std_y': stds_y,
            'std_z': stds_z,
            'mean_magnitude': means_magnitude,
            'std_magnitude': stds_magnitude
        })
    
    return df

def plot_window_statistics(patient_id: str, window_hours: float = 2.0, data_dir: str = "./patient_data", combine_days: bool = True, downsampling_ratio: float = 1.0):
    """
    Plot the window statistics for a patient.
    
    Args:
        patient_id: Patient identifier
        window_hours: Length of time window in hours (default: 2.0)
        data_dir: Directory containing patient data
        combine_days: If True, plot statistics across all days combined.
                     If False, plot statistics separately for each day.
        downsampling_ratio: Downsampling ratio (1.0 = no downsampling, 2.0 = half the points, etc.)
    """
    df = calculate_window_statistics(patient_id, window_hours, data_dir, combine_days, downsampling_ratio)
    
    if df.empty:
        print("No data found for the specified patient and time windows.")
        return
    
    # Create subplots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Convert window labels to x-axis positions
    x_positions = range(len(df))
    window_labels = [f"{start}-{end}" for start, end in zip(df['window_start'], df['window_end'])]
    
    # Plot means
    axes[0, 0].plot(x_positions, df['mean_x'], 'r-', label='X-axis', marker='o')
    axes[0, 0].plot(x_positions, df['mean_y'], 'g-', label='Y-axis', marker='s')
    axes[0, 0].plot(x_positions, df['mean_z'], 'b-', label='Z-axis', marker='^')
    axes[0, 0].plot(x_positions, df['mean_magnitude'], 'k-', label='Magnitude', marker='d')
    axes[0, 0].set_title(f'Mean Acceleration ({window_hours}-hour windows)')
    axes[0, 0].set_ylabel('Acceleration (g)')
    axes[0, 0].set_xticks(x_positions)
    axes[0, 0].set_xticklabels(window_labels, rotation=45, ha='right')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot standard deviations
    axes[0, 1].plot(x_positions, df['std_x'], 'r-', label='X-axis', marker='o')
    axes[0, 1].plot(x_positions, df['std_y'], 'g-', label='Y-axis', marker='s')
    axes[0, 1].plot(x_positions, df['std_z'], 'b-', label='Z-axis', marker='^')
    axes[0, 1].plot(x_positions, df['std_magnitude'], 'k-', label='Magnitude', marker='d')
    axes[0, 1].set_title(f'Standard Deviation ({window_hours}-hour windows)')
    axes[0, 1].set_ylabel('Standard Deviation (g)')
    axes[0, 1].set_xticks(x_positions)
    axes[0, 1].set_xticklabels(window_labels, rotation=45, ha='right')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot X vs Y means
    axes[1, 0].scatter(df['mean_x'], df['mean_y'], alpha=0.7)
    axes[1, 0].set_xlabel('Mean X-axis Acceleration (g)')
    axes[1, 0].set_ylabel('Mean Y-axis Acceleration (g)')
    axes[1, 0].set_title('Mean X vs Y Acceleration')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot magnitude vs time
    axes[1, 1].plot(x_positions, df['mean_magnitude'], 'k-', marker='o', linewidth=2)
    axes[1, 1].fill_between(x_positions, 
                           df['mean_magnitude'] - df['std_magnitude'],
                           df['mean_magnitude'] + df['std_magnitude'],
                           alpha=0.3, color='gray')
    axes[1, 1].set_xlabel('Time Windows')
    axes[1, 1].set_ylabel('Acceleration Magnitude (g)')
    axes[1, 1].set_title('Acceleration Magnitude with Standard Deviation')
    axes[1, 1].set_xticks(x_positions)
    axes[1, 1].set_xticklabels(window_labels, rotation=45, ha='right')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\nSummary Statistics for {window_hours}-hour windows:")
    print(f"Number of windows: {len(df)}")
    print(f"Time range: {df['window_start'].iloc[0]} to {df['window_end'].iloc[-1]}")
    print(f"Mean magnitude range: {df['mean_magnitude'].min():.3f} - {df['mean_magnitude'].max():.3f} g")
    print(f"Std magnitude range: {df['std_magnitude'].min():.3f} - {df['std_magnitude'].max():.3f} g")
    
    if not combine_days and 'day' in df.columns:
        print(f"Days covered: {df['day'].min()} to {df['day'].max()}")
    
