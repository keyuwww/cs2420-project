# Scripts Directory

Organized Python scripts for the LoRA Safety Adapter project.

## Data Preparation (`data_prep/`)

Scripts to download, convert, and prepare benchmark datasets:

1. **`download_vqa.py`** - Download VQA dataset
   ```bash
   python scripts/data_prep/download_vqa.py
   ```

2. **`convert_vqa_to_benchmark.py`** - Convert VQA to safe benchmark format
   ```bash
   python scripts/data_prep/convert_vqa_to_benchmark.py
   ```

3. **`convert_unsafe_to_benchmark.py`** - Convert unsafe data to benchmark format
   ```bash
   python scripts/data_prep/convert_unsafe_to_benchmark.py
   ```

4. **`standardize_benchmark.py`** - Standardize benchmark file formats
   ```bash
   python scripts/data_prep/standardize_benchmark.py
   ```

5. **`balance_benchmarks.py`** - Balance safe/unsafe datasets (50/50 split)
   ```bash
   python scripts/data_prep/balance_benchmarks.py
   ```

6. **`split_train_val_test.py`** - Split dataset into train/val/test sets
   ```bash
   python scripts/data_prep/split_train_val_test.py
   ```

## Inference (`inference/`)

Scripts to run models and generate predictions:

- **`run_benchmark_modal.py`** - Run benchmarks on Modal GPU
  ```bash
  modal run scripts/inference/run_benchmark_modal.py
  ```

## Analysis (`analysis/`)

Scripts to analyze model results:

- **`analyze_baseline.py`** - Analyze baseline model performance
  ```bash
  python scripts/analysis/analyze_baseline.py
  ```

## Typical Workflow

1. Download data: `python scripts/data_prep/download_vqa.py`
2. Convert to benchmark format: `python scripts/data_prep/convert_vqa_to_benchmark.py`
3. Convert unsafe data: `python scripts/data_prep/convert_unsafe_to_benchmark.py`
4. Balance datasets: `python scripts/data_prep/balance_benchmarks.py`
5. Split train/val/test: `python scripts/data_prep/split_train_val_test.py`
6. Run inference: `modal run scripts/inference/run_benchmark_modal.py`
7. Analyze results: `python scripts/analysis/analyze_baseline.py`
