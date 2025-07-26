import numpy as np
import torch
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation
import random
from typing import Tuple, Optional, Union

def _ensure_batch(data: torch.Tensor) -> torch.Tensor:
    """Ensure data is (batch, 3, context_len)."""
    if data.ndim == 2:
        return data.unsqueeze(0)
    return data

def _restore_shape(data: torch.Tensor, orig_shape: torch.Size) -> torch.Tensor:
    """Restore shape to (3, context_len) if input was not batched."""
    if len(orig_shape) == 2:
        return data[0]
    return data

def time_warping(data: torch.Tensor, warp_factor: float = 0.1, num_control_points: int = 5) -> torch.Tensor:
    """
    Generate a random cubic spline to stretch and warp the time axis.
    Supports both (3, context_len) and (batch, 3, context_len) input.
    """
    orig_shape = data.shape
    data = _ensure_batch(data)
    batch_size, _, context_len = data.shape
    warped_data = torch.zeros_like(data)
    for b in range(batch_size):
        # Generate random control points for the warping function
        x_control = np.linspace(0, context_len - 1, num_control_points)
        y_control = np.linspace(0, context_len - 1, num_control_points)
        # Add bounded random warping that preserves monotonicity
        warp_noise = np.random.normal(0, warp_factor * context_len * 0.1, num_control_points)
        y_control += warp_noise
        y_control = np.sort(y_control)
        y_control[0] = 0
        y_control[-1] = context_len - 1
        spline = CubicSpline(x_control, y_control)
        original_indices = np.arange(context_len)
        warped_indices = spline(original_indices)
        warped_indices = np.clip(warped_indices, 0, context_len - 1)
        for i in range(3):
            warped_data[b, i] = torch.from_numpy(
                np.interp(original_indices, warped_indices, data[b, i].cpu().numpy())
            )
    return _restore_shape(warped_data, orig_shape)

def add_random_gaussian_noise(data: torch.Tensor, mean: float = 0.0, std: float = 0.01) -> torch.Tensor:
    """
    Add random gaussian noise with predefined mean and std to the sequence of data.
    Supports both (3, context_len) and (batch, 3, context_len) input.
    """
    orig_shape = data.shape
    data = _ensure_batch(data)
    noise = torch.normal(mean=mean, std=std, size=data.shape, device=data.device)
    out = data + noise
    return _restore_shape(out, orig_shape)

#todo, could be potentially slow. Can improve by imposing the same rotation matrix to the entire batch.
def apply_3d_rotation(data: torch.Tensor, max_angle_degrees: float = 30.0) -> torch.Tensor:
    """
    Apply a random 3D rotation to the sequence of data.
    Supports both (3, context_len) and (batch, 3, context_len) input.
    """
    orig_shape = data.shape
    data = _ensure_batch(data)
    batch_size, _, context_len = data.shape
    rotated_data = torch.zeros_like(data)
    for b in range(batch_size):
        angles = np.random.uniform(-max_angle_degrees, max_angle_degrees, 3)
        rotation = Rotation.from_euler('xyz', angles, degrees=True)
        rotation_matrix = torch.from_numpy(rotation.as_matrix()).float().to(data.device)  # (3, 3)
        rotated_data[b] = torch.matmul(rotation_matrix, data[b])
    return _restore_shape(rotated_data, orig_shape)

def scale_data(data: torch.Tensor, scale_range: Tuple[float, float] = (0.8, 1.2)) -> torch.Tensor:
    """
    Scale the sequence of data with predefined range of scale factor.
    Supports both (3, context_len) and (batch, 3, context_len) input.
    """
    orig_shape = data.shape
    data = _ensure_batch(data)
    batch_size = data.shape[0]
    scale_factors = torch.rand(batch_size, 3, 1, device=data.device) * (scale_range[1] - scale_range[0]) + scale_range[0]
    out = data * scale_factors
    return _restore_shape(out, orig_shape)

def invert_data(data: torch.Tensor) -> torch.Tensor:
    """
    Invert the amplitude of the sequence of data by multiplying by -1.
    Supports both (3, context_len) and (batch, 3, context_len) input.
    """
    return -data

def invert_time_axis(data: torch.Tensor) -> torch.Tensor:
    """
    Invert the time axis of the sequence of data.
    Supports both (3, context_len) and (batch, 3, context_len) input.
    """
    orig_shape = data.shape
    data = _ensure_batch(data)
    out = torch.flip(data, dims=[2])
    return _restore_shape(out, orig_shape)

#todo, could be potentially slow. Can improve by imposing the same permutation to the entire batch.
def scramble_data(data: torch.Tensor, num_segments: int = 4) -> torch.Tensor:
    """
    Segment the sequence of data into a predefined number of segments, and concatenate them in a random order.
    Supports both (3, context_len) and (batch, 3, context_len) input.
    """
    orig_shape = data.shape
    data = _ensure_batch(data)
    batch_size, _, context_len = data.shape
    segment_size = context_len // num_segments
    scrambled = torch.zeros_like(data)
    for b in range(batch_size):
        indices = list(range(num_segments))
        random.shuffle(indices)
        segments = [data[b, :, i*segment_size:(i+1)*segment_size if i < num_segments-1 else context_len] for i in range(num_segments)]
        scrambled[b] = torch.cat([segments[i] for i in indices], dim=1)
    return _restore_shape(scrambled, orig_shape)

def shuffle_channels(data: torch.Tensor) -> torch.Tensor:
    """
    Shuffle the channels of the sequence of data.
    Supports both (3, context_len) and (batch, 3, context_len) input.
    Uses the same permutation for the entire batch for speed.
    """
    orig_shape = data.shape
    data = _ensure_batch(data)
    # Generate one permutation for the entire batch
    perm = torch.randperm(3)
    shuffled = data[:, perm]
    return _restore_shape(shuffled, orig_shape)

def apply_preprocessing_pipeline(data: torch.Tensor, transformations: list, **kwargs) -> torch.Tensor:
    """
    Apply a sequence of preprocessing transformations to the data.
    Supports both (3, context_len) and (batch, 3, context_len) input.
    """
    orig_shape = data.shape
    data = _ensure_batch(data)
    processed_data = data.clone()
    for transform in transformations:
        if hasattr(transform, '__call__'):
            processed_data = transform(processed_data, **kwargs)
        else:
            raise ValueError(f"Invalid transformation: {transform}")
    return _restore_shape(processed_data, orig_shape)


from scipy.signal import firwin, lfilter
def bandpass_filter(data: torch.Tensor, sample_rate: float = 80.0, low_f: float = 0.5, high_f: float = 10.0, numtaps: int = 513) -> torch.Tensor:
    """
    Apply a bandpass filter to the accelerometer data.
    Supports both (3, context_len) and (batch, 3, context_len) input.
    
    Args:
        data: Input tensor
        sample_rate: Sampling rate in Hz
        low_f: Lower cutoff frequency in Hz
        high_f: Upper cutoff frequency in Hz
        numtaps: Number of filter taps
        
    Returns:
        Filtered data tensor of same shape as input
    """
    orig_shape = data.shape
    data = _ensure_batch(data)
    batch_size, _, context_len = data.shape
    
    # Calculate normalized frequencies
    nyq_rate = sample_rate / 2.0
    low_f_norm = low_f / nyq_rate
    high_f_norm = high_f / nyq_rate
    delay = 0.5 * numtaps
    # Create bandpass FIR filter
    fir_coeff = firwin(numtaps, [low_f_norm, high_f_norm], pass_zero=False)
    
    # Apply filter to each sample in batch
    filtered_data = torch.zeros_like(data)
    for b in range(batch_size):
        for i in range(3):  # For each axis (X, Y, Z)
            filtered_data[b, i] = torch.from_numpy(
                lfilter(fir_coeff, 1.0, data[b, i].cpu().numpy())
            )
    
    return _restore_shape(filtered_data, orig_shape), delay