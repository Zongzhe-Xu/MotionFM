#!/usr/bin/env python3
"""
Test script to benchmark dataloader performance with Parquet files.
"""

import time
import logging
from dataloader import EfficientAccelDataLoader

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    """Test the dataloader performance."""
    logger.info("=== NHANES DataLoader Performance Test ===")
    
    # Test parameters
    data_dir = "./patient_data"
    context_len = 1000
    batch_size = 32
    num_workers = 4
    cache_size = 10000
    
    try:
        # Initialize data loader
        logger.info("Initializing data loader...")
        start_time = time.time()
        
        dataloader = EfficientAccelDataLoader(
            data_dir=data_dir,
            context_len=context_len,
            downsampling_ratio=1.0,
            batch_size=batch_size,
            num_workers=num_workers,
            cache_size=cache_size,
            seed=42
        )
        
        init_time = time.time() - start_time
        logger.info(f"DataLoader initialization completed in {init_time:.2f}s")
        
        # Run benchmark
        logger.info("Running performance benchmark...")
        benchmark_results = dataloader.benchmark_loading_speed(num_batches=20)
        
        # Summary
        logger.info("=== Performance Test Summary ===")
        logger.info(f"Initialization time: {init_time:.2f}s")
        logger.info(f"Average throughput: {benchmark_results['samples_per_sec']:.1f} samples/sec")
        logger.info(f"Average batch time: {benchmark_results['avg_batch_time']:.3f}s")
        logger.info(f"Memory per batch: {batch_size * context_len * 3 * 4 / (1024*1024):.1f} MB")
        
        # Test a few individual samples
        logger.info("Testing individual sample loading...")
        for i in range(5):
            start_time = time.time()
            sample = dataloader.dataset[i]
            load_time = time.time() - start_time
            logger.info(f"Sample {i}: {load_time:.3f}s, shape: {sample['accel_data'].shape}")
        
        logger.info("=== Performance test completed ===")
        
    except Exception as e:
        logger.error(f"Error during performance test: {e}")
        raise

if __name__ == "__main__":
    main() 