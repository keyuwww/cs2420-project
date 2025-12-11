#!/bin/bash
#SBATCH --job-name=lora_eval
#SBATCH --output=logs/lora_eval_%j.out
#SBATCH --error=logs/lora_eval_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --account=kempner_dam_lab

# Create logs directory if it doesn't exist
mkdir -p logs

# Set up environment
module load python/3.9
source ~/.bashrc

# Navigate to the llava directory
cd /n/home10/sliaw/cs2420-project/llava

# Set Hugging Face token (user should set this)
# Option A (preferred): export HF_TOKEN=your_token_here before sbatch
# Option B: edit the line below to hardcode: HF_TOKEN="your_token_here"
HF_TOKEN=${HF_TOKEN:-""}

# Run evaluation
# Note: Uses best checkpoint/threshold from checkpoint-dir
python3 evaluate_lora.py \
    --image-dir /n/home10/sliaw/cs2420-project \
    --output-dir /n/home10/sliaw/cs2420-project/llava/lora_results \
    --model-name llava-hf/llava-1.5-7b-hf \
    --checkpoint-dir /n/netscratch/dam_lab/Lab/sliaw/cs2432_artifacts/lora_checkpoints \
    --cache-dir /n/netscratch/dam_lab/Lab/sliaw/cs2432_artifacts \
    --train-file /n/home10/sliaw/cs2420-project/llava/lora_data/bench_train.fixed.jsonl \
    --val-file /n/home10/sliaw/cs2420-project/llava/lora_data/bench_val.fixed.jsonl \
    --test-file /n/home10/sliaw/cs2420-project/llava/lora_data/bench_test.fixed.jsonl \
    --eval-split test \
    --batch-size 4 \
    ${HF_TOKEN:+--hf-token $HF_TOKEN}

echo "Evaluation completed!"
