#!/usr/bin/env python3
"""
Split benchmark dataset into train/validation/test sets.
Uses stratified split to maintain label balance across all sets.
"""

import json
import random
from pathlib import Path
from collections import defaultdict

def split_dataset(input_file, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    Split dataset into train/val/test sets with stratified sampling.

    Args:
        input_file: Path to input JSONL file
        output_dir: Directory to save split files
        train_ratio: Proportion for training set (default 0.7)
        val_ratio: Proportion for validation set (default 0.15)
        test_ratio: Proportion for test set (default 0.15)
        seed: Random seed for reproducibility
    """
    # Validate ratios
    if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    print(f"Reading dataset from: {input_file}")

    # Read all entries
    with open(input_file, 'r', encoding='utf-8') as f:
        entries = [json.loads(line.strip()) for line in f if line.strip()]

    # Group entries by label for stratified split
    entries_by_label = defaultdict(list)
    for entry in entries:
        label = entry.get('label', 0)
        entries_by_label[label].append(entry)

    print(f"\nTotal entries: {len(entries)}")
    for label, label_entries in entries_by_label.items():
        print(f"  Label {label}: {len(label_entries)} entries")

    # Shuffle each label group with seed
    random.seed(seed)
    for label_entries in entries_by_label.values():
        random.shuffle(label_entries)

    # Split each label group
    train_data = []
    val_data = []
    test_data = []

    for label, label_entries in entries_by_label.items():
        n = len(label_entries)
        train_end = int(n * train_ratio)
        val_end = train_end + int(n * val_ratio)

        train_data.extend(label_entries[:train_end])
        val_data.extend(label_entries[train_end:val_end])
        test_data.extend(label_entries[val_end:])

        print(f"\nLabel {label} split:")
        print(f"  Train: {train_end}")
        print(f"  Val: {val_end - train_end}")
        print(f"  Test: {n - val_end}")

    # Shuffle combined splits
    random.shuffle(train_data)
    random.shuffle(val_data)
    random.shuffle(test_data)

    # Create output directory if needed
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    # Write splits to files
    train_file = output_dir / "benchmark_train.jsonl"
    val_file = output_dir / "benchmark_val.jsonl"
    test_file = output_dir / "benchmark_test.jsonl"

    print(f"\nWriting splits to {output_dir}/")

    with open(train_file, 'w', encoding='utf-8') as f:
        for entry in train_data:
            f.write(json.dumps(entry) + '\n')

    with open(val_file, 'w', encoding='utf-8') as f:
        for entry in val_data:
            f.write(json.dumps(entry) + '\n')

    with open(test_file, 'w', encoding='utf-8') as f:
        for entry in test_data:
            f.write(json.dumps(entry) + '\n')

    # Print summary
    print(f"\n{'='*60}")
    print(f"Dataset Split Complete!")
    print(f"{'='*60}")
    print(f"\nTrain set: {len(train_data)} entries ({train_ratio*100:.1f}%)")
    print(f"  - Unsafe (label=1): {sum(1 for e in train_data if e['label'] == 1)}")
    print(f"  - Safe (label=0): {sum(1 for e in train_data if e['label'] == 0)}")
    print(f"  - File: {train_file}")

    print(f"\nValidation set: {len(val_data)} entries ({val_ratio*100:.1f}%)")
    print(f"  - Unsafe (label=1): {sum(1 for e in val_data if e['label'] == 1)}")
    print(f"  - Safe (label=0): {sum(1 for e in val_data if e['label'] == 0)}")
    print(f"  - File: {val_file}")

    print(f"\nTest set: {len(test_data)} entries ({test_ratio*100:.1f}%)")
    print(f"  - Unsafe (label=1): {sum(1 for e in test_data if e['label'] == 1)}")
    print(f"  - Safe (label=0): {sum(1 for e in test_data if e['label'] == 0)}")
    print(f"  - File: {test_file}")

    return train_data, val_data, test_data

if __name__ == "__main__":
    import sys

    # Go up to lora/ directory, then into data/
    lora_dir = Path(__file__).parent.parent.parent
    data_dir = lora_dir / "data"

    # Default paths
    input_file = data_dir / "benchmark_balanced.jsonl"
    output_dir = data_dir

    # Default split ratios (70% train, 15% val, 15% test)
    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    # Allow command-line arguments
    if len(sys.argv) > 1:
        input_file = Path(sys.argv[1])
    if len(sys.argv) > 2:
        output_dir = Path(sys.argv[2])
    if len(sys.argv) > 3:
        train_ratio = float(sys.argv[3])
    if len(sys.argv) > 4:
        val_ratio = float(sys.argv[4])
    if len(sys.argv) > 5:
        test_ratio = float(sys.argv[5])

    # Validate input file
    if not input_file.exists():
        print(f"Error: Input file not found at: {input_file}")
        print(f"\nUsage: python {Path(__file__).name} [input.jsonl] [output_dir] [train_ratio] [val_ratio] [test_ratio]")
        print(f"Example: python {Path(__file__).name} data/benchmark_balanced.jsonl data/ 0.7 0.15 0.15")
        sys.exit(1)

    print(f"=== Splitting Dataset (Stratified) ===\n")
    print(f"Split ratios: Train={train_ratio:.1%}, Val={val_ratio:.1%}, Test={test_ratio:.1%}")

    # Perform split
    split_dataset(input_file, output_dir, train_ratio, val_ratio, test_ratio)
