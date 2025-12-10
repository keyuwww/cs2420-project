# LoRA Safety Adapter Training

This directory contains the training code for a LoRA adapter to detect emergent unsafety in image-text pairs using LLaVA-1.6-7B.

## Setup

### 1. Install Dependencies

```bash
pip install torch transformers peft pillow scikit-learn python-dotenv tqdm
```

### 2. Configure HuggingFace Token

Copy the example environment file and add your HuggingFace token:

```bash
cp .env.example .env
```

Then edit `.env` and add your token:

```
HF_TOKEN=your_actual_token_here
```

Get your token from: https://huggingface.co/settings/tokens

### 3. Data Structure

The data is organized as follows:

```
lora/
├── data/
│   ├── benchmark.json          # Standardized dataset (one entry per line)
│   └── image/                  # Images referenced in benchmark.json
│       ├── 000000001268.jpg
│       ├── 000000002473.jpg
│       └── ...
├── lora_training_llava.ipynb   # Main training notebook
├── standardize_benchmark.py    # Script to standardize benchmark format
├── .env                        # Your HF token (not committed)
└── .env.example                # Template for .env
```

### 4. Dataset Format

The benchmark files contain one JSON object per line in the following format:

```json
{"image_path": "image/filename.jpg", "prompt": "text prompt", "label": 0}
```

- `image_path`: Relative path to image from `lora/data/` directory
- `prompt`: Text prompt to test with the image
- `label`: 0 for safe, 1 for unsafe

**Available datasets:**
- `benchmark_unsafe.json` - 997 unsafe prompts only (label=1)
- `benchmark_safe.json` - 2,755 safe prompts from VQA (label=0)
- `benchmark_balanced.json` - 1,994 balanced prompts (50/50 safe/unsafe) **← Default**

## Usage

### Running the Training Notebook

Open `lora_training_llava.ipynb` in Jupyter or Google Colab and run all cells.

The notebook will:
1. Load the LLaVA model with your HF token
2. Load images from `lora/data/image/`
3. Read the dataset from `lora/data/benchmark_balanced.json` (default)
4. Train a LoRA adapter + safety head
5. Save checkpoints to `lora_checkpoints/`

To use a different dataset, change `data_path` in the Config cell:
- `"./data/benchmark_balanced.json"` - Balanced 50/50 (default)
- `"./data/benchmark_unsafe.json"` - Unsafe only
- `"./data/benchmark_safe.json"` - Safe only

### Converting VQA Dataset to Safe Benchmark

To create a safe benchmark from VQA questions:

```bash
# Convert VQA subset to benchmark format
python convert_vqa_to_benchmark.py

# Copy VQA images to data/image/
cp VQA_subset_500/images/* data/image/
```

The script extracts all questions from the VQA dataset and formats them as safe prompts (label=0).

### Balancing Safe and Unsafe Benchmarks

To create a balanced dataset with equal numbers of safe and unsafe examples:

```bash
# Randomly sample from safe to match unsafe count, then shuffle
python balance_benchmarks.py

# This creates data/benchmark_balanced.json with 50/50 safe/unsafe split
```

The script:
- Reads `benchmark_unsafe.json` (997 entries)
- Randomly samples 997 from `benchmark_safe.json` (2,755 entries)
- Combines and shuffles them into `benchmark_balanced.json` (1,994 total)

### Re-standardizing the Dataset

The script can handle 4 different JSON formats:

1. **Format with "evaluations"** - Full evaluation data with `is_emergent_unsafe` field
2. **Format with "results"** - Simplified format (assumes all unsafe)
3. **Format with "emergent_unsafe_prompt"** - Single entry at top level (assumes unsafe)
4. **Format with "caption"** - Has caption and emergent_unsafe_prompt (assumes unsafe)

The script automatically detects whether the input is a JSON array or JSONL (one JSON per line).

#### Process existing pipeline output files:

```bash
# From results.json (all unsafe, no confidence)
python standardize_benchmark.py ../coco_pipeline1/output/results.json --filter-label 1 --no-confidence

# From evaluation.json (has safety labels, filter to unsafe only)
python standardize_benchmark.py ../coco_pipeline1/output/evaluation.json --filter-label 1 --no-confidence

# Then replace the benchmark
mv data/benchmark_standardized.json data/benchmark.json
```

**Options:**
- `--filter-label 0`: Keep only safe entries
- `--filter-label 1`: Keep only unsafe entries
- `--no-confidence`: Omit the confidence field from output

#### Process default benchmark.json:

```bash
python standardize_benchmark.py
```

This will:
- Create a backup at `data/benchmark_original_backup.json` (first time only)
- Generate `data/benchmark_standardized.json`
- Extract all image+prompt pairs into a uniform format
- Handle absolute paths and convert them to relative `image/` paths
- Report statistics (safe vs unsafe entries)

## Configuration

Key configuration options in the notebook:

- `batch_size`: 4 (adjust based on GPU memory)
- `learning_rate`: 2e-4
- `num_epochs`: 3
- `lora_rank`: 16
- `data_path`: "./data/benchmark.json"
- `image_dir`: "./data"

## Output

Training produces:
- `lora_checkpoints/lora_adapter/`: The trained LoRA weights
- `lora_checkpoints/safety_head.pt`: The safety classification head
- `lora_checkpoints/best_metrics.json`: Validation metrics
- `lora_checkpoints/config.json`: Training configuration
