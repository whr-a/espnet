#!/bin/bash
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Launcher script for comprehensive MoE expert parallelism tests

set -e  # Exit on error

# Default values
NUM_GPUS=2
DTYPE="bfloat16"
# DTYPE="float32"
BATCH_SIZE=2
SEQ_LENGTH=64
EP_SIZE=2
VERBOSE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --dtype)
            DTYPE="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --seq_length)
            SEQ_LENGTH="$2"
            shift 2
            ;;
        --ep_size)
            EP_SIZE="$2"
            shift 2
            ;;
        --verbose)
            VERBOSE="--verbose"
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --num_gpus N       Number of GPUs to use (default: 2)"
            echo "  --dtype TYPE       Data type: float32, float16, bfloat16 (default: bfloat16)"
            echo "  --batch_size N     Batch size for testing (default: 2)"
            echo "  --seq_length N     Sequence length for testing (default: 64)"
            echo "  --ep_size N        Expert parallelism size (default: 2)"
            echo "  --verbose          Enable verbose output"
            echo "  --help             Show this help message"
            echo ""
            echo "Examples:"
            echo "  # Basic test with 2 GPUs"
            echo "  $0"
            echo ""
            echo "  # Test with 4 GPUs and float32 precision"
            echo "  $0 --num_gpus 4 --dtype float32 --ep_size 4"
            echo ""
            echo "  # Verbose test with larger batch"
            echo "  $0 --batch_size 4 --verbose"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Validate EP size vs number of GPUs
if [ $EP_SIZE -gt $NUM_GPUS ]; then
    echo "Error: ep_size ($EP_SIZE) cannot be greater than num_gpus ($NUM_GPUS)"
    exit 1
fi

# Print configuration
echo "=========================================="
echo "MoE Expert Parallelism Test Configuration"
echo "=========================================="
echo "Number of GPUs: $NUM_GPUS"
echo "Expert Parallel Size: $EP_SIZE"
echo "Data Type: $DTYPE"
echo "Batch Size: $BATCH_SIZE"
echo "Sequence Length: $SEQ_LENGTH"
echo "Verbose: ${VERBOSE:-No}"
echo "=========================================="
echo ""

# Set environment variables for better performance
export OMP_NUM_THREADS=1
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# Check if DeepSpeed is available
if ! command -v deepspeed &> /dev/null; then
    echo "Error: DeepSpeed is not installed or not in PATH"
    echo "Please install DeepSpeed: pip install deepspeed"
    exit 1
fi

# Check if we have enough GPUs
AVAILABLE_GPUS=$(nvidia-smi -L | wc -l)
if [ $AVAILABLE_GPUS -lt $NUM_GPUS ]; then
    echo "Error: Requested $NUM_GPUS GPUs but only $AVAILABLE_GPUS available"
    exit 1
fi

# Create log directory
LOG_DIR="logs/moe_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p $LOG_DIR

echo "Logs will be saved to: $LOG_DIR"
echo ""

# Run the test
echo "Starting DeepSpeed distributed test..."
echo "Command: deepspeed --num_gpus=$NUM_GPUS test_replace_moe_comprehensive.py \\"
echo "    --ep_size $EP_SIZE --dtype $DTYPE \\"
echo "    --batch_size $BATCH_SIZE --seq_length $SEQ_LENGTH $VERBOSE"
echo ""

# Run with DeepSpeed
deepspeed --num_gpus=$NUM_GPUS test_replace_moe_comprehensive.py \
    --ep_size $EP_SIZE \
    --dtype $DTYPE \
    --batch_size $BATCH_SIZE \
    --seq_length $SEQ_LENGTH \
    $VERBOSE \
    2>&1 | tee $LOG_DIR/test_output.log

# Check exit status
if [ ${PIPESTATUS[0]} -eq 0 ]; then
    echo ""
    echo "✓✓✓ Test completed successfully!"
    echo "Results saved to: $LOG_DIR/test_output.log"
else
    echo ""
    echo "✗✗✗ Test failed!"
    echo "Check logs at: $LOG_DIR/test_output.log"
    exit 1
fi