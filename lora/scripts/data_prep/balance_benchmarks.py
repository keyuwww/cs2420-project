#!/usr/bin/env python3
"""
Balance safe and unsafe benchmarks by randomly sampling to match counts.
Supports JSONL format (one JSON object per line).
"""

import json
import random
from pathlib import Path

def balance_benchmarks(unsafe_file, safe_file, output_file, seed=42):
    """
    Randomly sample from safe benchmark to match unsafe count,
    then combine them into a balanced output file.

    Reads JSONL files (one JSON object per line) and outputs balanced JSONL.
    """
    # Read unsafe entries (JSONL format)
    print(f"Reading unsafe entries from: {unsafe_file}")
    with open(unsafe_file, 'r', encoding='utf-8') as f:
        unsafe_entries = [json.loads(line.strip()) for line in f if line.strip()]

    # Read safe entries (JSONL format)
    print(f"Reading safe entries from: {safe_file}")
    with open(safe_file, 'r', encoding='utf-8') as f:
        safe_entries = [json.loads(line.strip()) for line in f if line.strip()]

    unsafe_count = len(unsafe_entries)
    safe_count = len(safe_entries)

    print(f"Unsafe entries: {unsafe_count}")
    print(f"Safe entries: {safe_count}")

    # Randomly sample from safe to match unsafe count
    random.seed(seed)
    sampled_safe = random.sample(safe_entries, min(unsafe_count, safe_count))

    print(f"Sampled safe entries: {len(sampled_safe)}")

    # Combine and shuffle
    balanced = unsafe_entries + sampled_safe
    random.shuffle(balanced)

    # Write to output (JSONL format)
    print(f"Writing balanced output to: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in balanced:
            f.write(json.dumps(entry) + '\n')

    print(f"\nBalanced benchmark created!")
    print(f"  Total entries: {len(balanced)}")
    print(f"  Unsafe (label=1): {sum(1 for e in balanced if e['label'] == 1)}")
    print(f"  Safe (label=0): {sum(1 for e in balanced if e['label'] == 0)}")
    print(f"  Output: {output_file}")

if __name__ == "__main__":
    import sys

    # Go up to lora/ directory, then into data/
    lora_dir = Path(__file__).parent.parent.parent
    data_dir = lora_dir / "data"

    # Default JSONL file paths
    unsafe_file = data_dir / "benchmark_unsafe.jsonl"
    safe_file = data_dir / "benchmark_safe.jsonl"
    output_file = data_dir / "benchmark_balanced.jsonl"

    # Allow command-line arguments
    if len(sys.argv) > 1:
        unsafe_file = Path(sys.argv[1])
    if len(sys.argv) > 2:
        safe_file = Path(sys.argv[2])
    if len(sys.argv) > 3:
        output_file = Path(sys.argv[3])

    # Validate input files exist
    if not unsafe_file.exists():
        print(f"Error: Unsafe benchmark not found at: {unsafe_file}")
        print(f"\nUsage: python {Path(__file__).name} [unsafe.jsonl] [safe.jsonl] [output.jsonl]")
        sys.exit(1)

    if not safe_file.exists():
        print(f"Error: Safe benchmark not found at: {safe_file}")
        print(f"\nUsage: python {Path(__file__).name} [unsafe.jsonl] [safe.jsonl] [output.jsonl]")
        sys.exit(1)

    print(f"=== Balancing Benchmarks (JSONL format) ===\n")
    balance_benchmarks(unsafe_file, safe_file, output_file)
