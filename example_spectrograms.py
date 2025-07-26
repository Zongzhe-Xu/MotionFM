#!/usr/bin/env python3
"""
Example script demonstrating how to use the new STFT and Welch's method functions
with accelerometer data from the NHANES dataset.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from filters import stft, stft_average, welch_method, compute_band_power, ACCELEROMETER_BANDS
from util import discover_patient_data, load_accel_data
import numpy as np
import matplotlib.pyplot as plt

def example_stft_analysis():
    """Example: Analyze accelerometer data using STFT."""
    
    print("=== STFT Analysis Example ===")
    
    # Load accelerometer data
    patient_files = discover_patient_data("./patient_data")
    if not patient_files:
        print("No patient data found.")
        return
    
    patient_id = list(patient_files.keys())[0]
    files = patient_files[patient_id]
    
    # Load first 30 seconds of data
    timestamps, accel_data = load_accel_data(files[0].file_path)
    
    if len(accel_data) == 0:
        print("No data loaded.")
        return
    
    # Compute magnitude
    magnitude = np.sqrt(accel_data[:, 0]**2 + accel_data[:, 1]**2 + accel_data[:, 2]**2)
    
    # STFT parameters
    fs = 80  # Hz
    win_sec = 2.0
    step_sec = 0.5
    
    print(f"Analyzing {len(magnitude)} samples ({len(magnitude)/fs:.1f} seconds)")
    print(f"STFT parameters: window={win_sec}s, step={step_sec}s")
    
    # Compute STFT
    freqs, times, Sxx = stft(
        signal=magnitude,
        fs=fs,
        win_sec=win_sec,
        step_sec=step_sec,
        window='hann',
        detrend=True,
        output_mode='density'
    )
    
    print(f"STFT result: {Sxx.shape[0]} frequencies, {Sxx.shape[1]} time windows")
    
    # Compute band powers
    band_power = compute_band_power(Sxx, freqs, ACCELEROMETER_BANDS)
    
    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Plot 1: Spectrogram
    im = axes[0, 0].pcolormesh(times, freqs, 10 * np.log10(Sxx + 1e-10), 
                              shading='gouraud', cmap='viridis')
    axes[0, 0].set_ylabel('Frequency (Hz)')
    axes[0, 0].set_title('STFT Spectrogram')
    plt.colorbar(im, ax=axes[0, 0], label='Power (dB)')
    
    # Plot 2: Magnitude Spectrum (average across time)
    avg_spectrum = np.mean(Sxx, axis=1)  # Average across time windows
    axes[0, 1].plot(freqs, avg_spectrum, linewidth=2, color='blue')
    axes[0, 1].set_xlabel('Frequency (Hz)')
    axes[0, 1].set_ylabel('Average Power Spectral Density')
    axes[0, 1].set_title('Magnitude Spectrum (Time-Averaged)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Add frequency band markers to magnitude spectrum
    colors = ['red', 'orange', 'green', 'purple']
    for i, (band_name, (fmin, fmax)) in enumerate(ACCELEROMETER_BANDS.items()):
        band_mask = (freqs >= fmin) & (freqs <= fmax)
        if np.any(band_mask):
            axes[0, 1].axvspan(fmin, fmax, alpha=0.2, color=colors[i], label=band_name)
    axes[0, 1].legend()
    
    # Plot 3: Band powers over time
    for band_name, power in band_power.items():
        axes[1, 0].plot(times, power, label=band_name, linewidth=2)
    
    axes[1, 0].set_xlabel('Time (s)')
    axes[1, 0].set_ylabel('Power')
    axes[1, 0].set_title('Band Powers Over Time')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Magnitude Spectrum in dB scale
    axes[1, 1].plot(freqs, 10 * np.log10(avg_spectrum + 1e-10), linewidth=2, color='red')
    axes[1, 1].set_xlabel('Frequency (Hz)')
    axes[1, 1].set_ylabel('Power (dB)')
    axes[1, 1].set_title('Magnitude Spectrum (dB Scale)')
    axes[1, 1].grid(True, alpha=0.3)
    
    # Add frequency band markers to dB spectrum
    for i, (band_name, (fmin, fmax)) in enumerate(ACCELEROMETER_BANDS.items()):
        band_mask = (freqs >= fmin) & (freqs <= fmax)
        if np.any(band_mask):
            axes[1, 1].axvspan(fmin, fmax, alpha=0.2, color=colors[i], label=band_name)
    axes[1, 1].legend()
    
    plt.tight_layout()
    plt.show()
    
    return freqs, times, Sxx, band_power


def example_welch_analysis():
    return welch_analysis(list(discover_patient_data("./patient_data").keys())[0])

def welch_analysis(patient_id):
    """Example: Analyze accelerometer data using Welch's method for daily activity patterns."""
    
    print("\n=== Daily Activity Analysis with Welch's Method ===")
    
    # Load accelerometer data
    patient_files = discover_patient_data("./patient_data")
    if not patient_files:
        print("No patient data found.")
        return
    
    patient_id = patient_id
    files = patient_files[patient_id]
    
    print(f"Analyzing patient {patient_id} with {len(files)} files (hours)")
    
    # Sort files by time extracted from filename
    def extract_sort_key(file_info):
        filename = file_info.file_path.split('/')[-1]
        parts = filename.split('-')
        # The structure is: GT3XPLUS-AccelerationCalibrated-2x5x0.NEO1G50016094.2000-MM-DD-HH-MM-SS-000-P0000.sensor.csv
        if len(parts) >= 10:
            try:
                # Extract date components from the parts
                date_part = parts[2]  # "2x5x0.NEO1G50016094.2000"
                year = int(date_part.split('.')[-1])  # Extract 2000 from the end
                month = int(parts[3])  # MM
                day = int(parts[4])    # DD
                hour = int(parts[5])   # HH
                minute = int(parts[6]) # MM
                second = int(parts[7]) # SS
                return (year, month, day, hour, minute, second)
            except Exception:
                return (0, 0, 0, 0, 0, 0)
        return (0, 0, 0, 0, 0, 0)
    files = sorted(files, key=extract_sort_key)
    
    # Welch's method parameters
    fs = 80  # Hz
    frame_win_sec = 8.0   # 8-second frames
    frame_step_sec = 2.0  # 2-second steps
    welch_win_sec = 4.0   # 4-second Welch windows
    welch_step_sec = 2.0  # 2-second Welch steps
    
    # Store results for each hour
    all_band_powers = []
    all_psd_data = []  # Store PSD data for frequency plots
    all_raw_signals = []  # Store raw magnitude signals
    all_raw_times = []   # Store time arrays for raw signals
    hour_labels = []
    
    # Process each file (each file represents ~1 hour of data)
    for i, file_info in enumerate(files[:24]):  # Limit to 24 hours
        print(f"Processing hour {i+1}/{min(len(files), 24)}: {file_info.file_path}")
        
        # Load data for this hour
        timestamps, accel_data = load_accel_data(file_info.file_path)
        
        if len(accel_data) == 0:
            print(f"  No data in file {i+1}, skipping...")
            continue
        
        # Compute magnitude
        magnitude = np.sqrt(accel_data[:, 0]**2 + accel_data[:, 1]**2 + accel_data[:, 2]**2)
        
        # Skip if too short
        if len(magnitude) < fs * 10:  # At least 10 seconds
            print(f"  File {i+1} too short ({len(magnitude)/fs:.1f}s), skipping...")
            continue
        
        # Compute Welch's method for this hour
        freqs, times, Sxx = welch_method(
            signal=magnitude,
            fs=fs,
            frame_win_sec=frame_win_sec,
            frame_step_sec=frame_step_sec,
            welch_win_sec=welch_win_sec,
            welch_step_sec=welch_step_sec,
            window='hann',
            detrend=True,
            output_mode='density'
        )
        
        # Compute band powers for this hour
        band_power = compute_band_power(Sxx, freqs, ACCELEROMETER_BANDS)
        
        # Store average band power for this hour
        hourly_avg = {}
        for band_name, power in band_power.items():
            hourly_avg[band_name] = np.mean(power)
        
        all_band_powers.append(hourly_avg)
        
        # Store PSD data (average across time windows)
        avg_psd = np.mean(Sxx, axis=1)  # Average across time windows
        all_psd_data.append(avg_psd)
        
        # Store raw signal data
        all_raw_signals.append(magnitude)
        # Create time array for this hour (in seconds from start of day)
        hour_time = np.arange(len(magnitude)) / fs + i * 3600  # Add hour offset
        all_raw_times.append(hour_time)
        
        # Extract hour from filename for better labeling
        filename = file_info.file_path.split('/')[-1]
        try:
            parts = filename.split('-')
            if len(parts) >= 7:
                hour = parts[5]  # HH
                minute = parts[6]  # MM
                hour_str = f"{hour}:{minute}"
            else:
                hour_str = f"Hour {i+1}"
        except:
            hour_str = f"Hour {i+1}"
        hour_labels.append(hour_str)
        
        print(f"  Processed {len(magnitude)} samples ({len(magnitude)/fs:.1f}s)")
        print(f"  Welch result: {Sxx.shape[0]} frequencies, {Sxx.shape[1]} time windows")
    
    if not all_band_powers:
        print("No valid data found.")
        return
    
    # Create comprehensive daily activity visualization
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))
    
    # Plot 1: Daily band power evolution
    hours = np.arange(len(all_band_powers))
    colors = ['blue', 'green', 'orange', 'red']
    
    for i, (band_name, color) in enumerate(zip(ACCELEROMETER_BANDS.keys(), colors)):
        band_values = [hourly_data[band_name] for hourly_data in all_band_powers]
        axes[0, 0].plot(hours, band_values, label=band_name, color=color, 
                       linewidth=2, marker='o', markersize=4)
    
    axes[0, 0].set_xlabel('Hour of Day')
    axes[0, 0].set_ylabel('Average Band Power')
    axes[0, 0].set_title(f'Daily Activity Pattern - Patient {patient_id}')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xticks(hours[::2])
    axes[0, 0].set_xticklabels([hour_labels[i] for i in hours[::2]], rotation=45)
    
    # Plot 2: Heatmap of all band powers
    band_names = list(ACCELEROMETER_BANDS.keys())
    power_matrix = np.array([[hourly_data[band] for band in band_names] 
                           for hourly_data in all_band_powers])
    
    im = axes[0, 1].imshow(power_matrix.T, aspect='auto', cmap='viridis', 
                          interpolation='nearest')
    axes[0, 1].set_xlabel('Hour of Day')
    axes[0, 1].set_ylabel('Frequency Band')
    axes[0, 1].set_title('Daily Activity Heatmap')
    axes[0, 1].set_yticks(range(len(band_names)))
    axes[0, 1].set_yticklabels(band_names)
    axes[0, 1].set_xticks(hours[::2])
    axes[0, 1].set_xticklabels([hour_labels[i] for i in hours[::2]], rotation=45)
    
    plt.colorbar(im, ax=axes[0, 1], label='Average Power')
    
    # Plot 3: PSD vs Frequency for all hours (overlay)
    axes[1, 0].set_xlabel('Frequency (Hz)')
    axes[1, 0].set_ylabel('Power Spectral Density')
    axes[1, 0].set_title('PSD vs Frequency - All Hours (Overlay)')
    
    # Set consistent frequency range with validation
    freq_min = freqs[0]
    freq_max = freqs[-1]
    
    # Validate frequency range to avoid NaN issues
    if np.isfinite(freq_min) and np.isfinite(freq_max) and freq_min > 0 and freq_max > freq_min:
        axes[1, 0].set_xscale('log')
        axes[1, 0].set_yscale('log')
        axes[1, 0].set_xlim(freq_min, freq_max)
    else:
        # Fallback to linear scale if log scale causes issues
        print(f"Warning: Using linear scale due to invalid frequency range: {freq_min} to {freq_max}")
        axes[1, 0].set_xlim(freq_min, freq_max)
    
    axes[1, 0].grid(True, alpha=0.3)
    
    # Use a colormap for different hours
    cmap = plt.cm.viridis
    for i, (psd_data, hour_label) in enumerate(zip(all_psd_data, hour_labels)):
        color = cmap(i / len(all_psd_data))
        axes[1, 0].plot(freqs, psd_data, color=color, alpha=0.7, linewidth=1, 
                       label=hour_label if i % 4 == 0 else "")  # Label every 4th line
    
    axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    
    # Plot 4: PSD vs Frequency heatmap
    psd_matrix = np.array(all_psd_data).T  # Transpose to get frequencies as rows
    
    # Set appropriate frequency range for log scale with validation
    freq_min = freqs[0]  # Start from the lowest frequency
    freq_max = freqs[-1]  # End at the highest frequency
    
    # Validate frequency range to avoid NaN issues
    if np.isfinite(freq_min) and np.isfinite(freq_max) and freq_min > 0 and freq_max > freq_min:
        # Create log-spaced frequency ticks for better visualization
        freq_ticks = np.logspace(np.log10(freq_min), np.log10(freq_max), 10)
        use_log_scale = True
    else:
        # Fallback to linear scale if log scale causes issues
        freq_ticks = np.linspace(freq_min, freq_max, 10)
        use_log_scale = False
    
    im2 = axes[1, 1].pcolormesh(hours, freqs, psd_matrix, shading='gouraud', cmap='plasma')
    axes[1, 1].set_xlabel('Hour of Day')
    axes[1, 1].set_ylabel('Frequency (Hz)')
    axes[1, 1].set_title('PSD vs Frequency Heatmap')
    
    if use_log_scale:
        axes[1, 1].set_yscale('log')
        axes[1, 1].set_yticks(freq_ticks)
        axes[1, 1].set_yticklabels([f'{f:.1f}' for f in freq_ticks])
    else:
        # Use default linear scale with automatic tick placement
        pass
    
    axes[1, 1].set_xticks(hours[::2])
    axes[1, 1].set_xticklabels([hour_labels[i] for i in hours[::2]], rotation=45)
    
    plt.colorbar(im2, ax=axes[1, 1], label='Power Spectral Density')
    
    plt.tight_layout()
    
    # Save the main analysis plot
    import os
    os.makedirs('./plots', exist_ok=True)
    main_plot_path = f'./plots/welch_analysis_patient_{patient_id}.png'
    plt.savefig(main_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved main analysis plot to: {main_plot_path}")
    plt.show()
    
    # Create separate figure for raw signal visualization
    print(f"\nGenerating raw signal visualization for Patient {patient_id}...")
    
    # Concatenate all raw signals and times
    full_raw_signal = np.concatenate(all_raw_signals)
    full_raw_time = np.concatenate(all_raw_times)
    
    # Create figure for raw signal
    fig_raw, ax_raw = plt.subplots(1, 1, figsize=(20, 6))
    
    # Downsample for visualization (plot every 100th point to avoid overcrowding)
    downsample_factor = max(1, len(full_raw_signal) // 10000)  # Aim for ~10k points
    plot_indices = np.arange(0, len(full_raw_signal), downsample_factor)
    
    ax_raw.plot(full_raw_time[plot_indices] / 3600, full_raw_signal[plot_indices], 
               linewidth=0.5, alpha=0.8, color='blue')
    ax_raw.set_xlabel('Hour of Day')
    ax_raw.set_ylabel('Accelerometer Magnitude')
    ax_raw.set_title(f'Raw Accelerometer Data - Patient {patient_id} (Entire Day)')
    ax_raw.grid(True, alpha=0.3)
    
    # Add hour markers
    for i, hour_label in enumerate(hour_labels):
        ax_raw.axvline(x=i, color='red', linestyle='--', alpha=0.5)
    
    # Add hour labels on x-axis
    ax_raw.set_xticks(range(len(hour_labels)))
    ax_raw.set_xticklabels(hour_labels, rotation=45)
    
    plt.tight_layout()
    
    # Save the raw signal plot
    raw_plot_path = f'./plots/raw_signals_patient_{patient_id}.png'
    plt.savefig(raw_plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved raw signal plot to: {raw_plot_path}")
    plt.show()
    
    # Print summary statistics
    print(f"\n=== Daily Activity Summary for Patient {patient_id} ===")
    for band_name in ACCELEROMETER_BANDS.keys():
        band_values = [hourly_data[band_name] for hourly_data in all_band_powers]
        print(f"{band_name} Band:")
        print(f"  Mean: {np.mean(band_values):.4f}")
        print(f"  Max: {np.max(band_values):.4f} (Hour {np.argmax(band_values)+1})")
        print(f"  Min: {np.min(band_values):.4f} (Hour {np.argmin(band_values)+1})")
        print()
    
    return all_band_powers, hour_labels

def example_comparison():
    """Example: Compare different methods on the same data."""
    
    print("\n=== Method Comparison Example ===")
    
    # Create synthetic signal with known components
    fs = 100
    duration = 20
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    
    # Signal with multiple frequency components
    signal = (np.sin(2 * np.pi * 2 * t) +           # 2 Hz - slow movement
              0.5 * np.sin(2 * np.pi * 8 * t) +     # 8 Hz - walking
              0.3 * np.sin(2 * np.pi * 25 * t) +    # 25 Hz - fast movement
              0.1 * np.random.randn(len(t)))        # noise
    
    print(f"Created synthetic signal with 2 Hz, 8 Hz, and 25 Hz components")
    print(f"Signal length: {len(signal)} samples ({duration} seconds)")
    
    # Method 1: STFT
    freqs1, times1, Sxx1 = stft(signal, fs, 3.0, 0.5)
    
    # Method 2: STFT with averaging
    freqs2, times2, Sxx2 = stft_average(signal, fs, 3, 2.0, 0.25)
    
    # Method 3: Welch's method
    freqs3, times3, Sxx3 = welch_method(signal, fs, 8.0, 1.0, 4.0, 2.0)
    
    # Plot comparison
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    methods = [
        ("STFT", freqs1, times1, Sxx1, 'viridis'),
        ("STFT with Averaging", freqs2, times2, Sxx2, 'plasma'),
        ("Welch's Method", freqs3, times3, Sxx3, 'inferno')
    ]
    
    for i, (name, freqs, times, Sxx, cmap) in enumerate(methods):
        im = axes[i].pcolormesh(times, freqs, 10 * np.log10(Sxx + 1e-10), 
                               shading='gouraud', cmap=cmap)
        axes[i].set_ylabel('Frequency (Hz)')
        axes[i].set_title(f'{name} (Shape: {Sxx.shape})')
        plt.colorbar(im, ax=axes[i], label='Power (dB)')
        
        # Mark known frequencies
        for freq in [2, 8, 25]:
            if freq <= freqs[-1]:
                axes[i].axhline(y=freq, color='red', linestyle='--', alpha=0.7)
    
    axes[-1].set_xlabel('Time (s)')
    plt.tight_layout()
    plt.show()
    
    print("Comparison complete!")
    print("Red dashed lines mark the known frequency components (2, 8, 25 Hz)")

if __name__ == "__main__":
    print("Spectrogram Analysis Examples")
    print("=" * 40)
    
    # Run examples
    try:
        example_stft_analysis()
        example_welch_analysis()
        example_comparison()
        print("\n✅ All examples completed successfully!")
    except Exception as e:
        print(f"\n❌ Error running examples: {e}")
        print("Make sure you have patient data available in ./patient_data/") 