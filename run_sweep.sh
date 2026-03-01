#!/bin/bash

# Hardcoded config path
CONFIG_PATH="./configs/debug_config.yaml"



# Activation functions to iterate over
ACTIVATIONS=(
    # "GELU"
    # "ReLUSquared"
    # "BSiLU"
    # "SUGARBSiLU"
    # "NoisyReLU"
    "TopKSparseGELU-10"
    "TopKSparseGELU-25"
    "TopKSparseGELU-50"
    "TopKSparseGELU-75"
    "TopKSparseGELU-90"
)

echo "========================================"
echo "Running activation sweep with config: $CONFIG_PATH"
echo "========================================"

for ACT in "${ACTIVATIONS[@]}"; do
    echo ""
    echo ">>> Running with activation: $ACT"
    echo "===================================="
    
    python runner.py --config "$CONFIG_PATH" --replaced_activation "$ACT" --original_activation "ReLU"
    
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: runner.py failed with exit code $EXIT_CODE for activation $ACT"
        # Uncomment below to stop on first failure
        # exit $EXIT_CODE
    fi
done

echo ""
echo "========================================"
echo "Sweep complete!"
echo "========================================"