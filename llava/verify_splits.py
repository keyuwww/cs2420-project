#!/usr/bin/env python3

import json
import numpy as np
from pathlib import Path
from data_utils import split_data
from collections import Counter

def load_data(data_path: str):
    with open(data_path, 'r') as f:
        data = []
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data

def check_label_distribution(data, split_name="all"):
    labels = [int(sample.get("label", 0)) for sample in data]
    label_counts = Counter(labels)
    total = len(labels)
    
    print(f"\n{split_name.upper()} Split Label Distribution:")
    print(f"  Total samples: {total}")
    print(f"  Safe (label 0): {label_counts[0]:4d} ({label_counts[0]/total*100:.2f}%)")
    print(f"  Unsafe (label 1): {label_counts[1]:4d} ({label_counts[1]/total*100:.2f}%)")
    
    return label_counts, total

def verify_splits_identical(data_path, train_split=0.7, val_split=0.15, test_split=0.15, seed=42, num_tests=3):
    print("="*70)
    print("VERIFYING SPLIT CONSISTENCY (Same Seed = Same Splits)")
    print("="*70)
    
    data = load_data(data_path)
    
    all_splits = []
    for i in range(num_tests):
        train, val, test = split_data(data, train_split, val_split, test_split, seed)
        all_splits.append({
            'train': [sample.get('image_path', '') + sample.get('prompt', '') for sample in train],
            'val': [sample.get('image_path', '') + sample.get('prompt', '') for sample in val],
            'test': [sample.get('image_path', '') + sample.get('prompt', '') for sample in test]
        })
    
    train_identical = all(all_splits[0]['train'] == s['train'] for s in all_splits[1:])
    val_identical = all(all_splits[0]['val'] == s['val'] for s in all_splits[1:])
    test_identical = all(all_splits[0]['test'] == s['test'] for s in all_splits[1:])
    
    if train_identical and val_identical and test_identical:
        print(f"✓ PASS: All {num_tests} runs with seed={seed} produced identical splits")
    else:
        print(f"✗ FAIL: Splits differ across runs (this should not happen!)")
        return False
    
    return True

def check_label_balance(data_path, train_split=0.7, val_split=0.15, test_split=0.15, seed=42):
    print("="*70)
    print("CHECKING LABEL BALANCE ACROSS SPLITS")
    print("="*70)
    
    data = load_data(data_path)
    
    check_label_distribution(data, "Overall")
    
    train_data, val_data, test_data = split_data(data, train_split, val_split, test_split, seed)
    
    train_counts, train_total = check_label_distribution(train_data, "Train")
    val_counts, val_total = check_label_distribution(val_data, "Val")
    test_counts, test_total = check_label_distribution(test_data, "Test")
    
    train_safe_pct = train_counts[0] / train_total * 100 if train_total > 0 else 0
    train_unsafe_pct = train_counts[1] / train_total * 100 if train_total > 0 else 0
    
    val_safe_pct = val_counts[0] / val_total * 100 if val_total > 0 else 0
    val_unsafe_pct = val_counts[1] / val_total * 100 if val_total > 0 else 0
    
    test_safe_pct = test_counts[0] / test_total * 100 if test_total > 0 else 0
    test_unsafe_pct = test_counts[1] / test_total * 100 if test_total > 0 else 0
    
    overall_counts, overall_total = Counter([int(s.get("label", 0)) for s in data]), len(data)
    overall_safe_pct = overall_counts[0] / overall_total * 100 if overall_total > 0 else 0
    overall_unsafe_pct = overall_counts[1] / overall_total * 100 if overall_total > 0 else 0
    
    print(f"\n{'='*70}")
    print("LABEL PROPORTION COMPARISON")
    print(f"{'='*70}")
    print(f"{'Split':<10} {'Safe %':<12} {'Unsafe %':<12} {'Difference from Overall'}")
    print(f"{'-'*70}")
    print(f"{'Overall':<10} {overall_safe_pct:>6.2f}%      {overall_unsafe_pct:>6.2f}%      (baseline)")
    print(f"{'Train':<10} {train_safe_pct:>6.2f}%      {train_unsafe_pct:>6.2f}%      {abs(train_safe_pct - overall_safe_pct):.2f}% diff")
    print(f"{'Val':<10} {val_safe_pct:>6.2f}%      {val_unsafe_pct:>6.2f}%      {abs(val_safe_pct - overall_safe_pct):.2f}% diff")
    print(f"{'Test':<10} {test_safe_pct:>6.2f}%      {test_unsafe_pct:>6.2f}%      {abs(test_safe_pct - overall_safe_pct):.2f}% diff")
    
    threshold = 5.0
    train_balanced = abs(train_safe_pct - overall_safe_pct) < threshold
    val_balanced = abs(val_safe_pct - overall_safe_pct) < threshold
    test_balanced = abs(test_safe_pct - overall_safe_pct) < threshold
    
    print(f"\n{'='*70}")
    if train_balanced and val_balanced and test_balanced:
        print("✓ BALANCED: All splits are within 5% of overall distribution")
        return True
    else:
        print("⚠ IMBALANCED: Some splits differ by more than 5% from overall")
        if not train_balanced:
            print(f"  - Train split differs by {abs(train_safe_pct - overall_safe_pct):.2f}%")
        if not val_balanced:
            print(f"  - Val split differs by {abs(val_safe_pct - overall_safe_pct):.2f}%")
        if not test_balanced:
            print(f"  - Test split differs by {abs(test_safe_pct - overall_safe_pct):.2f}%")
        print("\n  Note: This is usually fine for model training, but you may want")
        print("  to use stratified splitting if you need exact balance.")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Verify data splits")
    parser.add_argument("--data-path", type=str, default="benchmark_balanced.jsonl",
                       help="Path to data file")
    parser.add_argument("--train-split", type=float, default=0.7,
                       help="Training split ratio")
    parser.add_argument("--val-split", type=float, default=0.15,
                       help="Validation split ratio")
    parser.add_argument("--test-split", type=float, default=0.15,
                       help="Test split ratio")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("TEST 1: Split Consistency (Same Seed = Same Splits)")
    print("="*70)
    consistent = verify_splits_identical(
        args.data_path, args.train_split, args.val_split, args.test_split, args.seed
    )
    
    print("\n" + "="*70)
    print("TEST 2: Label Balance Across Splits")
    print("="*70)
    balanced = check_label_balance(
        args.data_path, args.train_split, args.val_split, args.test_split, args.seed
    )
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Split Consistency: {'✓ PASS' if consistent else '✗ FAIL'}")
    print(f"Label Balance: {'✓ BALANCED' if balanced else '⚠ IMBALANCED (but likely OK)'}")
    print("\nNote: Both evaluate_baseline.py and train_lora.py use the same")
    print("      split_data() function from data_utils.py with the same seed,")
    print("      so they will produce identical splits.")

if __name__ == "__main__":
    main()




