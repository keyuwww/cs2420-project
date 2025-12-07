#!/bin/bash
# Helper script to download LLaVA model manually (avoids rate limits)

CACHE_DIR="/n/netscratch/dam_lab/Lab/sliaw/cs2432_artifacts"
MODEL_NAME="llava-hf/llava-1.5-7b-hf"

echo "Downloading $MODEL_NAME to $CACHE_DIR"
echo "This may take a while (model is ~13GB)..."

# Create cache directory
mkdir -p "$CACHE_DIR"

# Download using huggingface-cli
huggingface-cli download "$MODEL_NAME" \
    --cache-dir "$CACHE_DIR" \
    --local-dir "$CACHE_DIR/$MODEL_NAME" \
    --local-dir-use-symlinks False

echo "✓ Download complete!"
echo "Model cached at: $CACHE_DIR"




