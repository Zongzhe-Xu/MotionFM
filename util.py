import os
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
import logging
import time
from datetime import datetime

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
    
    # Pattern to match sensor Parquet files (converted from CSV)
    pattern = r'^[\w-]+\.[\w-]+\.(\d{4})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{3})-P\d{4}\.sensor\.parquet$'
    
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
    Load accelerometer data from a Parquet file.
    
    Args:
        file_path: Path to the Parquet file
        start_idx: Starting index for slicing (optional)
        end_idx: Ending index for slicing (optional)
        downsampling_ratio: Downsampling ratio (1.0 = no downsampling, 2.0 = half the points, etc.)
        
    Returns:
        Tuple of (timestamps, accel_data) where:
        - timestamps: numpy array of timestamps
        - accel_data: numpy array of shape (N, 3) for X, Y, Z accelerations
    """
    try:
        # Load from Parquet file (much faster than CSV)
        df = pd.read_parquet(file_path)
        
        # Extract timestamps and accelerometer data
        # Note: Parquet preserves datetime types, so no need for conversion
        timestamps = df['HEADER_TIMESTAMP'].values
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
    
