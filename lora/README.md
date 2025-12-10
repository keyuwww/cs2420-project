# Scripts Organization

## Current Structure (8 files in lora/)

### Data Preparation Scripts
- `download_vqa.py` - Download VQA dataset
- `convert_vqa_to_benchmark.py` - Convert VQA to benchmark format (safe data)
- `convert_unsafe_to_benchmark.py` - Convert unsafe data to benchmark format
- `standardize_benchmark.py` - Standardize benchmark format
- `balance_benchmarks.py` - Balance safe/unsafe datasets
- `split_train_val_test.py` - Split into train/val/test sets

### Inference Scripts
- `run_benchmark_modal.py` - Run benchmarks on Modal

### Analysis Scripts
- `analyze_baseline.py` - Analyze baseline model results

## Proposed Organization

```
lora/
├── scripts/
│   ├── data_prep/
│   │   ├── download_vqa.py
│   │   ├── convert_vqa_to_benchmark.py
│   │   ├── convert_unsafe_to_benchmark.py
│   │   ├── standardize_benchmark.py
│   │   ├── balance_benchmarks.py
│   │   └── split_train_val_test.py
│   ├── inference/
│   │   └── run_benchmark_modal.py
│   └── analysis/
│       └── analyze_baseline.py
├── data/
└── README.md
```
