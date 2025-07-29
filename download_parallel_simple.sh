#!/bin/bash

# Simple Parallelized NHANES Download Script
# Usage: ./download_parallel_simple.sh [num_workers] [max_patients] [patient_list_file] [--replace]
# Example: ./download_parallel_simple.sh 4 10  # Use 4 workers, process max 10 patients
# Example: ./download_parallel_simple.sh 4 10 missing_patients.txt  # Download specific patients
# Example: ./download_parallel_simple.sh 4 10 incomplete_patients.txt --replace  # Replace existing patients

# Default values
NUM_WORKERS=${1:-2}  # Default to 2 workers
MAX_PATIENTS=${2:-5}  # Default to 5 patients
PATIENT_LIST_FILE=${3:-""}  # Optional file with list of patient IDs to download
REPLACE_EXISTING=false  # Default to skipping existing patients

# Check for --replace flag
if [[ "$4" == "--replace" ]]; then
    REPLACE_EXISTING=true
fi

TARGET_DIR="./patient_data"

# Check Python availability - prefer anaconda for pandas
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "ERROR: No Python found"
    exit 1
fi

echo "=== Simple Parallel NHANES Download ==="
echo "Workers: $NUM_WORKERS"
echo "Max patients: $MAX_PATIENTS"
if [ -n "$PATIENT_LIST_FILE" ]; then
    echo "Patient list file: $PATIENT_LIST_FILE"
else
    echo "Patient list file: None (will download from full list)"
fi
echo "Replace existing: $REPLACE_EXISTING"
echo "Target directory: $TARGET_DIR"
echo "Using Python: $PYTHON_CMD"
echo ""

# Test Python environment
echo "Testing Python environment..."
$PYTHON_CMD -c "import pandas; print('✓ pandas available')" 2>/dev/null || echo "✗ pandas not available"
$PYTHON_CMD -c "import pyarrow; print('✓ pyarrow available')" 2>/dev/null || echo "✗ pyarrow not available"

# Check if pyarrow is available
if ! $PYTHON_CMD -c "import pyarrow" 2>/dev/null; then
    echo ""
    echo "ERROR: pyarrow is required for Parquet conversion but not available."
    echo "Please install pyarrow using:"
    echo "  pip install pyarrow"
    echo "  or"
    echo "  conda install pyarrow"
    echo ""
    exit 1
fi
echo ""

# Get script directory for absolute paths BEFORE changing directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Create target directory
mkdir -p "$TARGET_DIR"
cd "$TARGET_DIR" || exit 1

# Check if patient list file is provided
if [ -n "$PATIENT_LIST_FILE" ]; then
    # Use provided patient list
    if [ ! -f "$SCRIPT_DIR/$PATIENT_LIST_FILE" ]; then
        echo "ERROR: Patient list file '$PATIENT_LIST_FILE' not found"
        exit 1
    fi
    
    echo "Creating file list from patient list: $PATIENT_LIST_FILE"
    # Convert patient IDs to .tar.bz2 filenames
    # Handle both simple patient IDs and tab-separated files with status
    while read -r line; do
        # Extract just the patient ID (first column, before any tab or space)
        patient_id=$(echo "$line" | cut -f1 | tr -d '[:space:]')
        if [ -n "$patient_id" ]; then
            echo "${patient_id}.tar.bz2" >> file_list.txt
        fi
    done < "$SCRIPT_DIR/$PATIENT_LIST_FILE"
    
    echo "Created file list with $(wc -l < file_list.txt) files from patient list"
else
    # Fetch the list of .tar.bz2 links from server
    echo "Fetching file list from server..."
    wget -q -O - https://ftp.cdc.gov/pub/pax_h/ | \
      grep -o 'HREF="[^"]*\.tar\.bz2"' | \
      sed 's/HREF="\/pub\/pax_h\///g' | \
      sed 's/"//g' > file_list.txt
fi

# Create a work queue by splitting the file list
echo "Creating work queue..."
TOTAL_FILES=$(wc -l < file_list.txt)

echo "Total files to process: $TOTAL_FILES"

for ((i=1; i<=NUM_WORKERS; i++)); do
    # Calculate start and end lines for this worker
    start_line=$(((i-1) * TOTAL_FILES / NUM_WORKERS + 1))
    if [ $i -eq $NUM_WORKERS ]; then
        # Last worker gets remaining files
        end_line=$TOTAL_FILES
    else
        end_line=$((i * TOTAL_FILES / NUM_WORKERS))
    fi
    
    # Extract files for this worker
    sed -n "${start_line},${end_line}p" file_list.txt > "worker_${i}_files.txt"
    worker_file_count=$(wc -l < "worker_${i}_files.txt")
    echo "Worker $i: $worker_file_count files (lines ${start_line}-${end_line})"
done

# Function to convert CSV to Parquet, then delete CSV
convert_csv_to_parquet() {
    local patient_dir="$1"
    local patient_id="$2"
    local total_parquet_size=0
    local converted_count=0
    local csv_files=($(find "$patient_dir" -name "*.csv"))
    
    # Use the script directory we calculated earlier
    local script_dir="$SCRIPT_DIR"
    
    for csv_file in "${csv_files[@]}"; do
        local filename=$(basename "$csv_file")
        local parquet_filename="${filename%.csv}.parquet"
        local parquet_path="$patient_dir/$parquet_filename"
        
        # Convert to Parquet using absolute paths
        echo "    Converting: $csv_file -> $parquet_path"
        conversion_output=$($PYTHON_CMD "$script_dir/parquet_converter.py" --single-file "$csv_file" "$parquet_path" 2>&1)
        conversion_exit_code=$?
        if [ $conversion_exit_code -ne 0 ]; then
            echo "[ERROR] Failed to convert $csv_file (exit code: $conversion_exit_code)"
            echo "Conversion output: $conversion_output"
            return 1
        fi
        
        # Get size
        if [ -f "$parquet_path" ]; then
            local parquet_size=$(stat -c%s "$parquet_path" 2>/dev/null || echo "0")
            local parquet_size_mb=$(echo "scale=2; $parquet_size / 1048576" | bc -l 2>/dev/null || echo "0")
            total_parquet_size=$(echo "$total_parquet_size + $parquet_size_mb" | bc -l 2>/dev/null || echo "$total_parquet_size")
        fi
        
        # Delete CSV
        rm -f "$csv_file"
        ((converted_count++))
    done
    
    echo "$total_parquet_size"
}

# Function to process a single patient
process_patient() {
    local worker_id="$1"
    local file="$2"
    local patient_id="${file%%.tar.bz2}"
    local start_time=$(date +%s)
    
    # Check if patient already exists
    if [ -d "$patient_id" ]; then
        if [ "$REPLACE_EXISTING" = true ]; then
            echo "[Worker $worker_id] Removing existing directory for $patient_id (--replace flag set)"
            rm -rf "$patient_id"
        else
            echo "[Worker $worker_id] Skipping $patient_id (already exists)"
            echo "$patient_id" >> "skipped_patients.txt"
            return 0  # Return success to avoid counting as failed attempt
        fi
    fi
    
    echo "[Worker $worker_id] Processing $file..."
    
    # Create worker-specific directory to avoid conflicts
    local worker_dir="worker_${worker_id}_${patient_id}"
    mkdir -p "$worker_dir"
    cd "$worker_dir" || exit 1
    
    # Download
    wget -q "https://ftp.cdc.gov/pub/pax_h/$file" -O "$file"
    if [ ! -f "$file" ]; then
        echo "[Worker $worker_id] Failed to download $file"
        cd ..
        rm -rf "$worker_dir"
        return 1
    fi
    
    # Extract
    mkdir -p "$patient_id"
    tar -xjf "$file" -C "$patient_id" 2>/dev/null
    if [ $? -ne 0 ]; then
        echo "[Worker $worker_id] Failed to extract $file"
        cd ..
        rm -rf "$worker_dir"
        return 1
    fi
    
    # Convert
    echo "  Starting conversion..."
    CONVERSION_OUTPUT=$(convert_csv_to_parquet "$patient_id" "$patient_id" 2>&1)
    conversion_exit_code=$?
    if [ $conversion_exit_code -ne 0 ]; then
        echo "[Worker $worker_id] Failed to convert $patient_id"
        echo "Conversion output: $CONVERSION_OUTPUT"
        cd ..
        rm -rf "$worker_dir"
        return 1
    fi
    
    # Get final size
    FINAL_SIZE=$(echo "$CONVERSION_OUTPUT" | tail -n1)
    
    # Move to final location
    cd ..
    mv "$worker_dir/$patient_id" "$patient_id"
    rm -rf "$worker_dir"
    
    # Calculate total time
    local end_time=$(date +%s)
    local total_time=$((end_time - start_time))
    
    # Write results to worker-specific files
    echo "$FINAL_SIZE" >> "worker_${worker_id}_results.txt"
    echo "$patient_id" >> "worker_${worker_id}_completed.txt"
    
    echo "[Worker $worker_id] ✓ Patient $patient_id completed in ${total_time}s, size: ${FINAL_SIZE}MB"
}

# Worker function
worker() {
    local worker_id="$1"
    local files_list="worker_${worker_id}_files.txt"
    local results_file="worker_${worker_id}_results.txt"
    local completed_file="worker_${worker_id}_completed.txt"
    
    # Initialize result files
    > "$results_file"
    > "$completed_file"
    
    # Initialize skipped patients file (shared across workers)
    touch "skipped_patients.txt"
    
    local total_files=$(wc -l < "$files_list")
    echo "[Worker $worker_id] Started with $total_files files"
    
    local processed_count=0
    local total_size=0
    local attempted_count=0  # Track total attempts (successful + failed)
    local skipped_count=0
    
    while read -r file; do
        # Check limits - count ALL attempts, not just successful ones
        if [ "$attempted_count" -ge "$MAX_PATIENTS" ]; then
            echo "[Worker $worker_id] Stopping (patient limit reached: $attempted_count attempts)"
            break
        fi
        

        
        # Increment attempt counter BEFORE processing
        ((attempted_count++))
        
        # Process the file
        process_patient "$worker_id" "$file"
        exit_code=$?
        if [ $exit_code -eq 0 ]; then
            # Check if this was a skip (no size recorded)
            if [ -s "$results_file" ]; then
                # This was a successful download
                ((processed_count++))
                # Get the last result
                local last_size=$(tail -n1 "$results_file")
                total_size=$(echo "$total_size + $last_size" | bc -l 2>/dev/null || echo "$total_size")
                
                # Show progress update
                local progress_percent=$((processed_count * 100 / total_files))
                echo "[Worker $worker_id] Progress: $processed_count/$total_files files completed (${progress_percent}%)"
            else
                # This was a skip
                ((skipped_count++))
                local progress_percent=$((attempted_count * 100 / total_files))
                echo "[Worker $worker_id] Progress: $attempted_count/$total_files files attempted (${progress_percent}%)"
            fi
        else
            echo "[Worker $worker_id] Failed to process $file (attempt $attempted_count/$MAX_PATIENTS)"
            
            # Show progress update even for failed attempts
            local progress_percent=$((attempted_count * 100 / total_files))
            echo "[Worker $worker_id] Progress: $attempted_count/$total_files files attempted (${progress_percent}%)"
        fi
    done < "$files_list"
    
    local final_progress_percent=$((processed_count * 100 / total_files))
    echo "[Worker $worker_id] Finished - processed $processed_count/$total_files patients successfully (${final_progress_percent}%), skipped $skipped_count, attempted $attempted_count total, total size: ${total_size}MB"
}

# Start workers
echo "Starting $NUM_WORKERS workers..."
for ((i=1; i<=NUM_WORKERS; i++)); do
    worker "$i" &
    WORKER_PIDS[$i]=$!
done

# Wait for all workers to finish
echo "Waiting for workers to complete..."
for ((i=1; i<=NUM_WORKERS; i++)); do
    wait ${WORKER_PIDS[$i]}
done

# Combine results
echo ""
echo "=== Combining Results ==="
TOTAL_SIZE=0
TOTAL_COUNT=0

for ((i=1; i<=NUM_WORKERS; i++)); do
    if [ -f "worker_${i}_results.txt" ]; then
        worker_size=$(awk '{sum+=$1} END {print sum}' "worker_${i}_results.txt" 2>/dev/null || echo "0")
        worker_count=$(wc -l < "worker_${i}_results.txt" 2>/dev/null || echo "0")
        TOTAL_SIZE=$(echo "$TOTAL_SIZE + $worker_size" | bc -l 2>/dev/null || echo "$TOTAL_SIZE")
        TOTAL_COUNT=$((TOTAL_COUNT + worker_count))
        
        echo "Worker $i: $worker_count patients, ${worker_size}MB"
    fi
done

echo ""
echo "=== Download Complete ==="
echo "Processed $TOTAL_COUNT patient(s)"
echo "Total size: ${TOTAL_SIZE}MB"
echo "Workers used: $NUM_WORKERS"

# Clean up temporary files
rm -f worker_*_files.txt worker_*_results.txt worker_*_completed.txt 