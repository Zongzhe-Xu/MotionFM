#!/usr/bin/env python3
"""
Parquet Converter for NHANES Patient Data

This script converts CSV files to Parquet format for comparison with NPY.
Parquet advantages:
- Columnar storage (faster for analytics)
- Built-in compression
- Schema preservation
- Better for large datasets
"""

import os
import pandas as pd
import time
import sys
import argparse
from typing import Tuple
import numpy as np

def get_file_size_mb(file_path: str) -> float:
    """Get file size in megabytes."""
    return os.path.getsize(file_path) / (1024 * 1024)

def convert_single_csv_to_parquet(csv_path: str, parquet_path: str) -> bool:
    """
    Convert a single CSV file to Parquet format.
    
    Args:
        csv_path: Path to the input CSV file
        parquet_path: Path to the output Parquet file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Read CSV with optimized settings
        df = pd.read_csv(csv_path, 
                        engine='c',  # Use C engine for speed
                        low_memory=False)
        
        # Save as Parquet with compression
        df.to_parquet(parquet_path, 
                     engine='pyarrow',  # Use pyarrow for better performance
                     compression='snappy',  # Fast compression
                     index=False)  # Don't save index
        
        return True
        
    except Exception as e:
        print(f"Error converting {csv_path}: {e}", file=sys.stderr)
        return False

def main():
    """Main function for CSV to Parquet conversion."""
    parser = argparse.ArgumentParser(description='CSV to Parquet converter')
    parser.add_argument('--single-file', nargs=2, metavar=('CSV_PATH', 'PARQUET_PATH'), 
                       help='Convert a single CSV file to Parquet')
    
    args = parser.parse_args()
    
    # Single file conversion mode
    if args.single_file:
        csv_path, parquet_path = args.single_file
        start_time = time.time()
        success = convert_single_csv_to_parquet(csv_path, parquet_path)
        conversion_time = time.time() - start_time
        
        if success:
            print(f"Conversion successful in {conversion_time:.2f}s")
        else:
            print("Conversion failed")
        
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 