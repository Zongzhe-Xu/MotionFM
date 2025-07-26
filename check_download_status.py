#!/usr/bin/env python3
"""
Check NHANES Download Status

This script compares the current downloaded patients with the full list
from the CDC FTP server to identify:
1. Missing patients that haven't been downloaded
2. Potentially corrupted/incomplete downloads
"""

import os
import requests
import re
import sys
import shutil
from pathlib import Path

def fetch_full_patient_list():
    """Fetch the complete list of available patients from CDC FTP server."""
    print("Fetching complete patient list from CDC FTP server...")
    
    try:
        response = requests.get('https://ftp.cdc.gov/pub/pax_h/', timeout=30)
        response.raise_for_status()
        content = response.text
        
        # Extract .tar.bz2 filenames and clean them
        tar_files = re.findall(r'HREF="([^"]*\.tar\.bz2)"', content)
        tar_files = [f.replace('/pub/pax_h/', '') for f in tar_files]
        
        # Extract patient IDs
        patient_ids = [f.replace('.tar.bz2', '') for f in tar_files]
        
        print(f"Found {len(patient_ids)} total patients available")
        return set(patient_ids)
        
    except requests.RequestException as e:
        print(f"Error fetching patient list: {e}")
        return set()

def cleanup_download_artifacts(data_dir="./patient_data"):
    """Clean up artifacts from interrupted downloads."""
    print(f"Cleaning up download artifacts in {data_dir}...")
    
    if not os.path.exists(data_dir):
        return
    
    artifacts_removed = []
    recovered_patients = []
    
    # Check for partially completed downloads in worker directories
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path) and item.startswith("worker_"):
            # Check if there's a patient directory inside that might be complete
            for subitem in os.listdir(item_path):
                subitem_path = os.path.join(item_path, subitem)
                if os.path.isdir(subitem_path) and not subitem.startswith("worker_"):
                    # This might be a completed patient download
                    patient_dir = os.path.join(data_dir, subitem)
                    if not os.path.exists(patient_dir):
                        # Move the patient directory to the main directory
                        try:
                            shutil.move(subitem_path, patient_dir)
                            recovered_patients.append(subitem)
                        except Exception as e:
                            print(f"Warning: Could not move {subitem} from {item}: {e}")
            
            # Now remove the worker directory
            try:
                shutil.rmtree(item_path)
                artifacts_removed.append(f"worker directory: {item}")
            except Exception as e:
                print(f"Warning: Could not remove {item}: {e}")
    
    # Remove temporary files
    temp_files = [
        "file_list.txt",
        "worker_*_files.txt",
        "worker_*_results.txt", 
        "worker_*_completed.txt",
        "temp_missing_files.txt",
        "temp_missing_files.txt.limited"
    ]
    
    for pattern in temp_files:
        for file_path in Path(data_dir).glob(pattern):
            try:
                file_path.unlink()
                artifacts_removed.append(f"temp file: {file_path.name}")
            except Exception as e:
                print(f"Warning: Could not remove {file_path}: {e}")
    
    if recovered_patients:
        print(f"Recovered {len(recovered_patients)} patient directories from worker artifacts:")
        for patient in recovered_patients[:5]:
            print(f"  - {patient}")
        if len(recovered_patients) > 5:
            print(f"  ... and {len(recovered_patients) - 5} more")
    
    if artifacts_removed:
        print(f"Removed {len(artifacts_removed)} artifacts:")
        for artifact in artifacts_removed[:10]:  # Show first 10
            print(f"  - {artifact}")
        if len(artifacts_removed) > 10:
            print(f"  ... and {len(artifacts_removed) - 10} more")
    else:
        print("No artifacts found to clean up")

def get_downloaded_patients(data_dir="./patient_data"):
    """Get list of currently downloaded patients."""
    print(f"Checking downloaded patients in {data_dir}...")
    
    if not os.path.exists(data_dir):
        print(f"Data directory {data_dir} does not exist")
        return set()
    
    downloaded = set()
    for item in os.listdir(data_dir):
        item_path = os.path.join(data_dir, item)
        if os.path.isdir(item_path):
            # Skip worker directories (they should be cleaned up by now)
            if not item.startswith("worker_"):
                downloaded.add(item)
    
    print(f"Found {len(downloaded)} downloaded patient directories")
    return downloaded

def check_patient_completeness(patient_id, data_dir="./patient_data"):
    """Check if a patient directory is complete (has Parquet files)."""
    patient_dir = os.path.join(data_dir, patient_id)
    
    if not os.path.exists(patient_dir):
        return False, "Directory does not exist"
    
    # Check for Parquet files
    parquet_files = list(Path(patient_dir).glob("*.parquet"))
    
    if not parquet_files:
        # Check for CSV files (incomplete conversion)
        csv_files = list(Path(patient_dir).glob("*.csv"))
        if csv_files:
            return False, f"Incomplete conversion: {len(csv_files)} CSV files, 0 Parquet files"
        else:
            return False, "No data files found"
    
    # Check if we have a reasonable number of files (typically 100-200 per patient)
    # A complete patient should have ~168 files (7 days × 24 hours)
    if len(parquet_files) < 50:
        return False, f"Very few files: {len(parquet_files)} Parquet files (expected ~168)"
    elif len(parquet_files) < 100:
        return False, f"Incomplete: {len(parquet_files)} Parquet files (expected ~168)"
    
    return True, f"Complete: {len(parquet_files)} Parquet files"

def analyze_download_status(data_dir="./patient_data"):
    """Analyze the current download status."""
    print("=== NHANES Download Status Analysis ===\n")
    
    # Clean up artifacts from interrupted downloads first
    cleanup_download_artifacts(data_dir)
    print()
    
    # Get full patient list
    full_patient_list = fetch_full_patient_list()
    if not full_patient_list:
        print("Failed to fetch full patient list. Exiting.")
        return
    
    # Get downloaded patients
    downloaded_patients = get_downloaded_patients(data_dir)
    
    # Find missing patients
    missing_patients = full_patient_list - downloaded_patients
    
    # Filter to only NHANES patients for completeness checking
    downloaded_nhanes_patients = downloaded_patients.intersection(full_patient_list)
    
    # Check completeness of downloaded patients
    print(f"\n=== Checking Download Completeness ===")
    incomplete_patients = []
    complete_patients = []
    
    for patient_id in sorted(downloaded_nhanes_patients):
        is_complete, status = check_patient_completeness(patient_id, data_dir)
        if is_complete:
            complete_patients.append(patient_id)
        else:
            incomplete_patients.append((patient_id, status))
    
    # Generate reports
    print(f"\n=== Download Status Summary ===")
    print(f"Total patients available: {len(full_patient_list)}")
    print(f"Downloaded directories: {len(downloaded_patients)}")
    print(f"Downloaded NHANES patients: {len(downloaded_nhanes_patients)}")
    print(f"Complete downloads: {len(complete_patients)}")
    print(f"Incomplete downloads: {len(incomplete_patients)}")
    print(f"Missing patients: {len(missing_patients)}")
    
    # Save missing patients list
    if missing_patients:
        missing_file = "missing_patients.txt"
        with open(missing_file, 'w') as f:
            for patient_id in sorted(missing_patients):
                f.write(f"{patient_id}\n")
        print(f"\nMissing patients saved to: {missing_file}")
        print(f"First 10 missing patients: {sorted(list(missing_patients))[:10]}")
    
    # Save incomplete patients list
    if incomplete_patients:
        incomplete_file = "incomplete_patients.txt"
        with open(incomplete_file, 'w') as f:
            for patient_id, status in incomplete_patients:
                f.write(f"{patient_id}\t{status}\n")
        print(f"\nIncomplete patients saved to: {incomplete_file}")
        print(f"First 5 incomplete patients:")
        for patient_id, status in incomplete_patients[:5]:
            print(f"  {patient_id}: {status}")
    
    # Save complete patients list
    if complete_patients:
        complete_file = "complete_patients.txt"
        with open(complete_file, 'w') as f:
            for patient_id in sorted(complete_patients):
                f.write(f"{patient_id}\n")
        print(f"\nComplete patients saved to: {complete_file}")
    
    # Note: Use the modified download_parallel_simple.sh with missing_patients.txt
    if missing_patients:
        print(f"\nTo download missing patients, use:")
        print(f"  ./download_parallel_simple.sh [workers] [max_patients] missing_patients.txt")
        print(f"  Example: ./download_parallel_simple.sh 4 100 missing_patients.txt")
    
    print(f"\n=== Recommendations ===")
    if incomplete_patients:
        print(f"- {len(incomplete_patients)} patients have incomplete downloads")
        print(f"  Consider re-downloading these patients")
    
    if missing_patients:
        print(f"- {len(missing_patients)} patients are missing entirely")
        print(f"  Use: ./download_parallel_simple.sh [workers] [max_patients] missing_patients.txt")
    
    print(f"- Total complete patients: {len(complete_patients)}")
    print(f"- Success rate: {len(complete_patients)/len(full_patient_list)*100:.1f}%")
    print(f"- Artifacts cleaned up automatically before analysis")

def main():
    """Main function."""
    data_dir = "./patient_data"
    
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    
    analyze_download_status(data_dir)

if __name__ == "__main__":
    main() 