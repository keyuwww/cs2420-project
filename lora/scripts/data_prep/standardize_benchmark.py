#!/usr/bin/env python3
"""
Standardize benchmark.json to have exactly one image + prompt pair per line.
The standardized format will be:
{"image_path": "image/filename.jpg", "prompt": "...", "label": 0 or 1, "confidence": 0-3}

Handles 4 different input formats:
1. Format with "evaluations" key (has is_emergent_unsafe field)
2. Format with "results" key (assumes all unsafe)
3. Format with "emergent_unsafe_prompt" key (single entry, assumes unsafe)
4. Format with "caption" key and "emergent_unsafe_prompt" (single entry, assumes unsafe)
"""

import json
import os
from pathlib import Path

def extract_entries_from_old_format(entry, image_dir):
    """Extract standardized entries from various JSON formats."""
    results = []
    image_path = entry.get("image_path", "")

    # Convert old path format to new format (image/filename.jpg)
    if image_path:
        # Handle absolute paths
        if image_path.startswith("/"):
            filename = os.path.basename(image_path)
        else:
            filename = os.path.basename(image_path)

        new_image_path = f"image/{filename}"

        # Check if image exists
        if not os.path.exists(os.path.join(image_dir, new_image_path)):
            print(f"Warning: Image not found: {new_image_path}")
            return results
    else:
        return results

    # Format 1: has "evaluations" key with full evaluation data
    # Example: {"evaluations": [{"emergent_prompt": "...", "evaluation": {"is_emergent_unsafe": true, "severity": "high"}}]}
    if "evaluations" in entry:
        for eval_item in entry["evaluations"]:
            emergent_prompt = eval_item.get("emergent_prompt", "")
            evaluation = eval_item.get("evaluation", {})

            is_unsafe = evaluation.get("is_emergent_unsafe", False)
            severity_map = {"low": 0, "medium": 1, "high": 2}
            severity = evaluation.get("severity", "low")
            confidence = severity_map.get(severity, 0)

            if emergent_prompt:
                results.append({
                    "image_path": new_image_path,
                    "prompt": emergent_prompt,
                    "label": 1 if is_unsafe else 0,
                    "confidence": confidence
                })

    # Format 2: has "results" key with simplified format
    # Example: {"results": [{"emergent_unsafe_prompt": "...", "meta": {"severity": "high"}}]}
    elif "results" in entry:
        for result_item in entry["results"]:
            prompt = result_item.get("emergent_unsafe_prompt", "")
            meta = result_item.get("meta", {})
            severity = meta.get("severity", "low")

            # Assume all items in "results" format are unsafe (label=1)
            severity_map = {"low": 1, "medium": 2, "high": 3}
            confidence = severity_map.get(severity, 1)

            if prompt:
                results.append({
                    "image_path": new_image_path,
                    "prompt": prompt,
                    "label": 1,  # Assume unsafe
                    "confidence": confidence
                })

    # Format 3: Direct format with "emergent_unsafe_prompt" at top level (single entry)
    # Example: {"image_path": "...", "emergent_unsafe_prompt": "...", "meta": {"severity": "high"}}
    elif "emergent_unsafe_prompt" in entry:
        prompt = entry.get("emergent_unsafe_prompt", "")
        meta = entry.get("meta", {})
        severity = meta.get("severity", "low")

        # Assume unsafe (label=1)
        severity_map = {"low": 1, "medium": 2, "high": 3}
        confidence = severity_map.get(severity, 1)

        if prompt:
            results.append({
                "image_path": new_image_path,
                "prompt": prompt,
                "label": 1,  # Assume unsafe
                "confidence": confidence
            })

    # Format 4: Has "caption" key with "emergent_unsafe_prompt" (single entry)
    # Example: {"image_path": "...", "caption": "...", "emergent_unsafe_prompt": "...", "meta": {"severity": "high"}}
    # This is similar to Format 3, but we already handled it above since we check for "emergent_unsafe_prompt"

    return results

def standardize_benchmark(input_path, output_path, image_dir, filter_label=None, include_confidence=True):
    """Standardize the benchmark.json file.

    Args:
        input_path: Path to input JSON file
        output_path: Path to output JSONL file
        image_dir: Directory containing images
        filter_label: If set (0 or 1), only include entries with this label
        include_confidence: If False, omit the confidence field from output
    """
    print(f"Reading from: {input_path}")
    print(f"Writing to: {output_path}")

    all_entries = []

    with open(input_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()

    # Try to parse as JSON array first
    try:
        data_array = json.loads(content)
        if isinstance(data_array, list):
            print(f"Detected JSON array format with {len(data_array)} items")
            for idx, entry in enumerate(data_array, 1):
                standardized = extract_entries_from_old_format(entry, image_dir)
                all_entries.extend(standardized)

                if idx % 100 == 0:
                    print(f"Processed {idx} items, extracted {len(all_entries)} entries so far...")
        else:
            # Single object
            standardized = extract_entries_from_old_format(data_array, image_dir)
            all_entries.extend(standardized)

    except json.JSONDecodeError:
        # Not a JSON array, try JSONL (one JSON object per line)
        print("Detected JSONL format (one JSON per line)")
        lines = content.split('\n')
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
                standardized = extract_entries_from_old_format(entry, image_dir)
                all_entries.extend(standardized)

                if line_num % 100 == 0:
                    print(f"Processed {line_num} lines, extracted {len(all_entries)} entries so far...")

            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue

    print(f"\nTotal entries extracted: {len(all_entries)}")
    print(f"Unsafe entries (label=1): {sum(1 for e in all_entries if e['label'] == 1)}")
    print(f"Safe entries (label=0): {sum(1 for e in all_entries if e['label'] == 0)}")

    # Filter by label if requested
    if filter_label is not None:
        before_count = len(all_entries)
        all_entries = [e for e in all_entries if e['label'] == filter_label]
        print(f"Filtered to label={filter_label}: {len(all_entries)} entries (removed {before_count - len(all_entries)})")

    # Write standardized format (one JSON object per line)
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in all_entries:
            # Optionally remove confidence field
            if not include_confidence and 'confidence' in entry:
                output_entry = {k: v for k, v in entry.items() if k != 'confidence'}
            else:
                output_entry = entry
            f.write(json.dumps(output_entry) + '\n')

    print(f"\nStandardized benchmark saved to: {output_path}")

if __name__ == "__main__":
    import sys
    import argparse

    parser = argparse.ArgumentParser(description='Standardize benchmark JSON files')
    parser.add_argument('input_file', nargs='?', help='Input JSON file (optional)')
    parser.add_argument('--filter-label', type=int, choices=[0, 1],
                       help='Filter to only include entries with this label (0=safe, 1=unsafe)')
    parser.add_argument('--no-confidence', action='store_true',
                       help='Omit the confidence field from output')

    args = parser.parse_args()

    # Go up to lora/ directory, then into data/
    lora_dir = Path(__file__).parent.parent.parent
    data_dir = lora_dir / "data"
    image_dir = data_dir
    output_file = data_dir / "benchmark_standardized.json"

    # Check if input file is provided as argument
    if args.input_file:
        input_source = Path(args.input_file)
        if not input_source.exists():
            print(f"Error: Input file not found: {input_source}")
            sys.exit(1)
        print(f"Using provided input file: {input_source}")
    else:
        # Default behavior: use existing files
        input_file = data_dir / "benchmark.json"
        backup_file = data_dir / "benchmark_original_backup.json"

        # Check if backup exists, if so use that as input
        if backup_file.exists():
            print(f"Found backup file. Using it as input source...")
            input_source = backup_file
        else:
            # Create backup first
            if input_file.exists():
                print("Creating backup of original file...")
                with open(input_file, 'r') as src, open(backup_file, 'w') as dst:
                    dst.write(src.read())
                print(f"Backup saved to: {backup_file}")
            input_source = input_file

    # Standardize
    standardize_benchmark(
        input_source,
        output_file,
        data_dir,
        filter_label=args.filter_label,
        include_confidence=not args.no_confidence
    )

    print("\nDone! Review the output and replace the original if satisfied.")
    print(f"To replace: mv {output_file} {data_dir / 'benchmark.json'}")
    print(f"\nUsage examples:")
    print(f"  python {Path(__file__).name} input.json")
    print(f"  python {Path(__file__).name} input.json --filter-label 1 --no-confidence")
    print(f"  python {Path(__file__).name} --filter-label 1  # uses existing benchmark")
