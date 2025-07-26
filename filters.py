import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.signal import spectrogram, welch, savgol_filter, convolve2d
from scipy import integrate
from typing import Tuple, Optional, Dict, Any


def stft(signal: np.ndarray, 
         fs: float, 
         win_sec: float, 
         step_sec: float, 
         window: str = 'hann', 
         pad: bool = True,
         detrend: bool = True,
         output_mode: str = 'density') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Short-Time Fourier Transform (STFT) of a signal.
    
    Args:
        signal: Input signal array (1D)
        fs: Sampling frequency in Hz
        win_sec: Window size in seconds
        step_sec: Step size in seconds
        window: Window type ('hann', 'hamming', 'blackman', etc.)
        pad: Whether to pad the signal for better resolution at boundaries
        detrend: Whether to detrend each segment
        output_mode: 'density' for power spectral density (V^2/Hz) or 'spectrum' for power (V^2)
        
    Returns:
        freqs: Frequency array
        times: Time array (center of each window)
        Sxx: Spectrogram array (frequencies x times)
    """
    
    # Detrend if requested
    if detrend:
        window_length = int(fs * 15) * 2 + 1  # Must be odd
        if window_length < len(signal):
            signal = signal - savgol_filter(signal, window_length, polyorder=2)
    
    # Pad signal for better resolution at boundaries
    if pad:
        pad_length = int(fs * ((win_sec - step_sec) // 2))
        signal = np.concatenate([
            np.zeros(pad_length, dtype=np.float64),
            signal,
            np.zeros(pad_length, dtype=np.float64)
        ], axis=0)
    
    # Define window and step sizes
    nperseg = int(win_sec * fs)
    step = int(step_sec * fs)
    
    # Compute spectrogram
    freqs, times, Sxx = spectrogram(
        x=signal, 
        fs=fs,
        window=window, 
        nperseg=nperseg,
        noverlap=nperseg - step, 
        detrend='linear' if detrend else False,
        scaling=output_mode
    )
    
    # Adjust times to original signal coordinates
    if pad:
        pad_sec = pad_length / fs
        times = times - pad_sec
    
    # Keep only frequencies up to Nyquist
    max_freq = fs / 2
    keep = freqs <= max_freq
    freqs = freqs[keep]
    Sxx = Sxx[keep, :]
    
    return freqs, times, Sxx


def stft_average(signal: np.ndarray, 
                fs: float, 
                avg_win_num: int,
                stft_win_sec: float, 
                stft_step_sec: float, 
                window: str = 'hann', 
                pad: bool = True,
                detrend: bool = True,
                output_mode: str = 'density') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute STFT with moving average across time windows.
    This is equivalent to Welch's method but may be faster for some applications.
    
    Args:
        signal: Input signal array (1D)
        fs: Sampling frequency in Hz
        avg_win_num: Number of windows to average over
        stft_win_sec: Window size for STFT in seconds
        stft_step_sec: Step size for STFT in seconds
        window: Window type
        pad: Whether to pad the signal
        detrend: Whether to detrend the signal
        output_mode: 'density' or 'spectrum'
        
    Returns:
        freqs: Frequency array
        times: Time array (after averaging)
        Sxx: Spectrogram array (frequencies x times)
    """
    
    # First compute regular STFT
    freqs, times, Sxx = stft(
        signal, fs, stft_win_sec, stft_step_sec, 
        window, pad, detrend, output_mode
    )
    
    # Apply moving average across time windows
    kernel = np.ones((1, avg_win_num)) / avg_win_num
    Sxx_smooth = convolve2d(Sxx, kernel, mode='valid')
    
    # Adjust times to reflect new time resolution after averaging
    times_smooth = np.convolve(times, np.ones(avg_win_num) / avg_win_num, mode='valid')
    
    return freqs, times_smooth, Sxx_smooth


def welch_method(signal: np.ndarray, 
                fs: float, 
                frame_win_sec: float, 
                frame_step_sec: float, 
                welch_win_sec: float, 
                welch_step_sec: float, 
                window: str = 'hann', 
                pad: bool = True,
                detrend: bool = True,
                output_mode: str = 'density') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply Welch's method repeatedly on sliding frames.
    For each frame, Welch's algorithm splits into smaller segments and averages FFT results.
    
    Args:
        signal: Input signal array (1D)
        fs: Sampling frequency in Hz
        frame_win_sec: Window size for each frame in seconds
        frame_step_sec: Step size for each frame in seconds
        welch_win_sec: Window size for Welch's method in seconds
        welch_step_sec: Step size for Welch's method in seconds
        window: Window type
        pad: Whether to pad the signal
        detrend: Whether to detrend the signal
        output_mode: 'density' or 'spectrum'
        
    Returns:
        freqs: Frequency array
        times: Time array (center of each frame)
        Sxx: Power spectral density array (frequencies x times)
    """
    
    # Detrend if requested
    if detrend:
        window_length = int(fs * 15) * 2 + 1
        if window_length < len(signal):
            signal = signal - savgol_filter(signal, window_length, polyorder=2)
    
    # Pad signal for better resolution at boundaries
    if pad:
        pad_length = int(fs * ((frame_win_sec - frame_step_sec) // 2))
        signal = np.concatenate([
            np.zeros(pad_length, dtype=np.float64),
            signal,
            np.zeros(pad_length, dtype=np.float64)
        ], axis=0)
    
    # Define Welch window and step sizes
    nperseg = int(welch_win_sec * fs)
    step = int(welch_step_sec * fs)
    
    # Process each frame
    nperframe = int(frame_win_sec * fs)
    step_frame = int(frame_step_sec * fs)
    nframe = (len(signal) - nperframe) // step_frame + 1
    
    psd_list = []
    for i in range(nframe):
        # Extract segment for this frame
        seg = signal[i*step_frame : i*step_frame + nperframe]
        
        # Apply Welch's method to this segment
        freqs, psd = welch(  # Only need psd, freqs is already computed
            seg, 
            fs=fs, 
            window=window, 
            nperseg=nperseg,
            noverlap=nperseg - step, 
            detrend='linear' if detrend else False,
            scaling=output_mode
        )
        psd_list.append(psd)
    
    # Stack results
    Sxx = np.asarray(psd_list).T  # (F, T)
    times = np.arange(nframe) * frame_step_sec + (frame_win_sec / 2)  # center of frame
    
    # Adjust times to original signal coordinates
    if pad:
        pad_sec = pad_length / fs
        times = times - pad_sec
    
    # Keep only frequencies up to Nyquist
    max_freq = fs / 2
    keep = freqs <= max_freq
    freqs = freqs[keep]
    Sxx = Sxx[keep, :]
    
    return freqs, times, Sxx


def compute_band_power(spectrogram: np.ndarray, 
                      freqs: np.ndarray, 
                      bands: Dict[str, Tuple[float, float]]) -> Dict[str, np.ndarray]:
    """
    Compute power in frequency bands using Simpson's rule integration.
    
    Args:
        spectrogram: 2D spectrogram array (frequencies x times)
        freqs: 1D frequency array
        bands: Dictionary mapping band names to (fmin, fmax) frequency ranges
        
    Returns:
        band_power: Dictionary mapping band names to 1D power arrays (times)
    """
    band_power = {}
    
    for band_name, (fmin, fmax) in bands.items():
        # Get indices for frequencies in this band
        band_mask = (freqs > fmin) & (freqs <= fmax)
        
        if np.any(band_mask):
            # Integrate power over frequencies in band at each timepoint
            power = integrate.simpson(
                y=spectrogram[band_mask], 
                x=freqs[band_mask], 
                axis=0
            )
            band_power[band_name] = power
        else:
            # No frequencies in this band
            band_power[band_name] = np.zeros(spectrogram.shape[1])
    
    return band_power


def create_spectrogram(data: Dict[str, np.ndarray], 
                      fs: float = 100, 
                      NFFT: int = 256, 
                      noverlap: int = 128,
                      cmap: str = 'viridis', 
                      magnitude: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """
    Legacy function for creating spectrograms from accelerometer data.
    Kept for backward compatibility.
    
    Args:
        data: Dictionary with 'X', 'Y', 'Z' accelerometer data
        fs: Sampling frequency
        NFFT: Number of FFT points
        noverlap: Number of overlapping points
        cmap: Colormap for plotting
        magnitude: Whether to use magnitude (sqrt(X^2 + Y^2 + Z^2)) or individual axes
        
    Returns:
        powerSpectrum, frequenciesFound, time, imageAxis
    """
    if magnitude:
        seq = np.sqrt(data['X']**2 + data['Y']**2 + data['Z']**2)
        powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(
            seq, Fs=fs, NFFT=NFFT, noverlap=noverlap, 
            window=np.hamming(NFFT), cmap=cmap
        )
    else:
        # For individual axes, return X-axis by default
        powerSpectrum, frequenciesFound, time, imageAxis = plt.specgram(
            data['X'], Fs=fs, NFFT=NFFT, noverlap=noverlap, 
            window=np.hamming(NFFT), cmap=cmap
        )
    
    return powerSpectrum, frequenciesFound, time, imageAxis


# Example frequency bands for different applications
ACCELEROMETER_BANDS = {
    "Low": (0.1, 1.0),      # Very low frequency movements
    "Medium": (1.0, 5.0),   # Walking, moderate activity
    "High": (5.0, 20.0),    # Running, high activity
    "Very_High": (20.0, 50.0)  # Tremors, fine movements
}

# EEG_BANDS = {
#     "Delta": (0.5, 4.0),
#     "Theta": (4.0, 8.0),
#     "Alpha": (8.0, 13.0),
#     "Beta": (13.0, 30.0),
#     "Gamma": (30.0, 64.0)
# }

# GENERAL_BANDS = {
#     "Low": (0.1, 1.0),
#     "Medium": (1.0, 10.0),
#     "High": (10.0, 50.0)
# }
    


