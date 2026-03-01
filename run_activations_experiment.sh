#!/bin/bash

# Script to run experiments with different activation functions
# Usage: ./run_activations_experiment.sh [OPTIONS]

# Default values
ORIGINAL_ACTIVATION="GELU"
REPLACED_NORMALIZATION="none"
CONFIG_PATH="./configs/config.yaml"
PYTHON_SCRIPT="runner.py"
LOG_DIR="./activation_experiments"
DRY_RUN=false

# List of activation functions to test
ACTIVATIONS=(
    "ReLUSquared"
    "BSiLU"
    "SUGARBSiLU"
    "NoisyReLU"
    "GELU"
    "TopKSparseGELU-10"
    "TopKSparseGELU-25"
    "TopKSparseGELU-50"
    "TopKSparseGELU-75"
    "TopKSparseGELU-90"
)

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --original_activation)
            ORIGINAL_ACTIVATION="$2"
            shift 2
            ;;
        --replaced_normalization)
            REPLACED_NORMALIZATION="$2"
            shift 2
            ;;
        --config)
            CONFIG_PATH="$2"
            shift 2
            ;;
        --python_script)
            PYTHON_SCRIPT="$2"
            shift 2
            ;;
        --log_dir)
            LOG_DIR="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --original_activation ACTIVATION  Original activation to replace (default: GELU)"
            echo "  --replaced_normalization NORM     Normalization to use (default: QuantileBatchNorm2d-50)"
            echo "  --config PATH                     Path to config file (default: ./configs/config.yaml)"
            echo "  --python_script PATH              Path to Python script (default: runner.py)"
            echo "  --log_dir PATH                    Directory for logs (default: ./activation_experiments)"
            echo "  --dry-run                         Show commands without executing"
            echo "  --help                            Show this help message"
            echo ""
            echo "This script will run experiments with the following activations:"
            for act in "${ACTIVATIONS[@]}"; do
                echo "  - $act"
            done
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Create log directory if it doesn't exist
if [ ! -d "$LOG_DIR" ]; then
    echo "Creating log directory: $LOG_DIR"
    if [ "$DRY_RUN" = false ]; then
        mkdir -p "$LOG_DIR"
    fi
fi

# Print configuration
echo "=== Activation Experiment Runner ==="
echo "Original activation: $ORIGINAL_ACTIVATION"
echo "Replaced normalization: $REPLACED_NORMALIZATION"
echo "Config file: $CONFIG_PATH"
echo "Python script: $PYTHON_SCRIPT"
echo "Log directory: $LOG_DIR"
echo "Number of activations to test: ${#ACTIVATIONS[@]}"
echo ""

# Check if config file exists
if [ ! -f "$CONFIG_PATH" ]; then
    echo "Error: Config file not found: $CONFIG_PATH"
    exit 1
fi

# Check if Python script exists
if [ ! -f "$PYTHON_SCRIPT" ]; then
    echo "Error: Python script not found: $PYTHON_SCRIPT"
    exit 1
fi

# Run experiments for each activation
for activation in "${ACTIVATIONS[@]}"; do
    echo "=== Running experiment with activation: $activation ==="
    
    # Create a timestamp for this run
    TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
    RUN_LOG="$LOG_DIR/${activation}_${TIMESTAMP}.log"
    
    # Build the command
    CMD="python $PYTHON_SCRIPT \
        --original_activation \"$ORIGINAL_ACTIVATION\" \
        --replaced_activation \"$activation\" \
        --original_normalization \"BatchNorm2d\" \
        --replaced_normalization \"$REPLACED_NORMALIZATION\""
    
    echo "Command: $CMD"
    echo "Log file: $RUN_LOG"
    
    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would execute: $CMD"
        echo "[DRY RUN] Would log to: $RUN_LOG"
    else
        # Execute the command and log output
        echo "Starting at: $(date)"
        eval $CMD 2>&1 | tee "$RUN_LOG"
        
        # Check exit status
        if [ ${PIPESTATUS[0]} -eq 0 ]; then
            echo "✓ Successfully completed: $activation"
        else
            echo "✗ Failed: $activation (see $RUN_LOG for details)"
        fi
        echo "Completed at: $(date)"
    fi
    
    echo ""
    
    # Optional: Add a small delay between runs
    if [ "$DRY_RUN" = false ]; then
        sleep 2
    fi
done

echo "=== All experiments completed ==="
echo "Logs saved in: $LOG_DIR"
echo "Summary of activations tested:"
for activation in "${ACTIVATIONS[@]}"; do
    echo "  - $activation"
done