#!/bin/bash

# Hardcoded config path
CONFIG_PATH="./configs/config_device.yaml"



# Activation functions to iterate over
NORMS=(
# "QuantileLayerNorm-10"
# "QuantileLayerNorm-25"
# "QuantileLayerNorm-50"
"LayerNorm"
"QuantileLayerNorm-75"
"QuantileLayerNorm-90"

)

echo "========================================"
echo "Running normalizations sweep with config: $CONFIG_PATH"
echo "========================================"

for NORM in "${NORMS[@]}"; do
    echo ""
    echo ">>> Running with normalization: $NORM"
    echo "===================================="
    
    python runner.py --config "$CONFIG_PATH" --replaced_normalization "$NORM" --original_normalization "RMSNorm"
    
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
        echo "ERROR: runner.py failed with exit code $EXIT_CODE for normalization $NORM"
        # Uncomment below to stop on first failure
        # exit $EXIT_CODE
    fi
done

echo ""
echo "========================================"
echo "Sweep complete!"
echo "========================================"