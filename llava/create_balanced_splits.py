#!/usr/bin/env python3
"""
Create balanced train/val/test splits from a JSONL file.

This script ensures that each split maintains a similar distribution of labels,
making the splits balanced across classes.
"""

import json
import argparse
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from collections import Counter


def load_jsonl(file_path: str) -> List[Dict]:
    """Load data from JSONL file."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """Save data to JSONL file."""
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def create_balanced_splits(
    data: List[Dict],
    train_split: float = 0.7,
    val_split: float = 0.15,
    test_split: float = 0.15,
    seed: int = 42,
    label_key: str = "label"
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Create balanced train/val/test splits maintaining label distribution.
    
    Args:
        data: List of data samples (each must have a label field)
        train_split: Proportion for training set
        val_split: Proportion for validation set
        test_split: Proportion for test set
        seed: Random seed for reproducibility
        label_key: Key in the data dict that contains the label
    
    Returns:
        Tuple of (train_data, val_data, test_data)
    """
    assert abs(train_split + val_split + test_split - 1.0) < 1e-6, \
        f"Splits must sum to 1.0, got {train_split + val_split + test_split}"
    
    # Separate data by label
    label_to_data = {}
    for item in data:
        label = item.get(label_key)
        if label is None:
            raise ValueError(f"Missing '{label_key}' field in data item: {item}")
        if label not in label_to_data:
            label_to_data[label] = []
        label_to_data[label].append(item)
    
    # Print label distribution
    print(f"\nOriginal data distribution:")
    for label, items in sorted(label_to_data.items()):
        print(f"  Label {label}: {len(items)} samples ({100*len(items)/len(data):.1f}%)")
    
    # Set random seed
    np.random.seed(seed)
    
    # Shuffle each label group
    for label in label_to_data:
        np.random.shuffle(label_to_data[label])
    
    # Split each label group proportionally
    train_data = []
    val_data = []
    test_data = []
    
    for label, items in label_to_data.items():
        n = len(items)
        train_size = int(n * train_split)
        val_size = int(n * val_split)
        
        train_data.extend(items[:train_size])
        val_data.extend(items[train_size:train_size + val_size])
        test_data.extend(items[train_size + val_size:])
    
    # Shuffle the final splits
    np.random.shuffle(train_data)
    np.random.shuffle(val_data)
    np.random.shuffle(test_data)
    
    # Print split distributions
    print(f"\nSplit distributions:")
    for split_name, split_data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        label_counts = Counter(item[label_key] for item in split_data)
        total = len(split_data)
        print(f"\n{split_name} ({total} samples):")
        for label in sorted(label_counts.keys()):
            count = label_counts[label]
            print(f"  Label {label}: {count} samples ({100*count/total:.1f}%)")
    
    return train_data, val_data, test_data


def main():
    parser = argparse.ArgumentParser(
        description="Create balanced train/val/test splits from JSONL file"
    )
    parser.add_argument(
        "--input-file",
        type=str,
        required=True,
        help="Input JSONL file (e.g., benchmark_balanced.jsonl)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=".",
        help="Output directory for split files (default: current directory)"
    )
    parser.add_argument(
        "--train-split",
        type=float,
        default=0.7,
        help="Training set proportion (default: 0.7)"
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Validation set proportion (default: 0.15)"
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.15,
        help="Test set proportion (default: 0.15)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--label-key",
        type=str,
        default="label",
        help="Key in JSON objects that contains the label (default: 'label')"
    )
    parser.add_argument(
        "--train-output",
        type=str,
        default=None,
        help="Output filename for training split (default: train.jsonl)"
    )
    parser.add_argument(
        "--val-output",
        type=str,
        default=None,
        help="Output filename for validation split (default: val.jsonl)"
    )
    parser.add_argument(
        "--test-output",
        type=str,
        default=None,
        help="Output filename for test split (default: test.jsonl)"
    )
    
    args = parser.parse_args()
    
    # Validate splits
    if not (0 < args.train_split < 1 and 0 < args.val_split < 1 and 0 < args.test_split < 1):
        raise ValueError("All splits must be between 0 and 1")
    
    if abs(args.train_split + args.val_split + args.test_split - 1.0) > 1e-6:
        raise ValueError(
            f"Splits must sum to 1.0, got {args.train_split + args.val_split + args.test_split}"
        )
    
    # Load data
    print(f"Loading data from {args.input_file}...")
    data = load_jsonl(args.input_file)
    print(f"Loaded {len(data)} samples")
    
    # Create balanced splits
    train_data, val_data, test_data = create_balanced_splits(
        data,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        seed=args.seed,
        label_key=args.label_key
    )
    
    # Set output filenames
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = output_dir / (args.train_output or "train.jsonl")
    val_file = output_dir / (args.val_output or "val.jsonl")
    test_file = output_dir / (args.test_output or "test.jsonl")
    
    # Save splits
    print(f"\nSaving splits...")
    save_jsonl(train_data, train_file)
    print(f"  Train: {train_file} ({len(train_data)} samples)")
    
    save_jsonl(val_data, val_file)
    print(f"  Val: {val_file} ({len(val_data)} samples)")
    
    save_jsonl(test_data, test_file)
    print(f"  Test: {test_file} ({len(test_data)} samples)")
    
    print(f"\n✓ Balanced splits created successfully!")


if __name__ == "__main__":
    main()


