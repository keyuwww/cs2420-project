#!/usr/bin/env python3
"""
Filter samples that didn't get clean yes/no answers and create a rerun file.
"""

import json
from pathlib import Path

def filter_failed_samples(results_file, original_file, output_file):
    """
    Extract samples where llava_answer is not 'yes' or 'no'.

    Args:
        results_file: Path to results JSONL with llava_answer field
        original_file: Path to original benchmark file (to get full data)
        output_file: Path to save filtered samples for rerun
    """
    results_path = Path(results_file)
    original_path = Path(original_file)
    output_path = Path(output_file)

    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return

    print(f"Reading results from: {results_path}")

    # Read results and find problematic answers
    failed_samples = []
    good_count = 0

    with open(results_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if not line.strip():
                continue

            try:
                result = json.loads(line)
                answer = result.get('llava_answer', '').lower().strip()

                # Check if answer is clean yes or no
                if answer not in ['yes', 'no']:
                    # This sample needs to be rerun
                    failed_samples.append(result)
                    print(f"Line {line_num}: '{answer}' (needs rerun)")
                else:
                    good_count += 1

            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue

    print(f"\nSummary:")
    print(f"  Good answers (yes/no): {good_count}")
    print(f"  Failed answers: {len(failed_samples)}")

    if len(failed_samples) == 0:
        print("\nAll samples have clean yes/no answers. No rerun needed!")
        return

    # Write failed samples to rerun file (original format without llava fields)
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in failed_samples:
            # Keep only original fields (image_path, prompt, label)
            rerun_sample = {
                'image_path': sample['image_path'],
                'prompt': sample['prompt'],
                'label': sample['label']
            }
            f.write(json.dumps(rerun_sample) + '\n')

    print(f"\nCreated rerun file: {output_path}")
    print(f"  {len(failed_samples)} samples to rerun")

    return failed_samples

if __name__ == "__main__":
    import sys

    script_dir = Path(__file__).parent
    data_dir = script_dir / "data"

    # Default paths
    results_file = data_dir / "results_test_llava15.jsonl"
    original_file = data_dir / "benchmark_test.jsonl"
    output_file = data_dir / "benchmark_test_rerun.jsonl"

    if len(sys.argv) > 1:
        results_file = Path(sys.argv[1])
    if len(sys.argv) > 2:
        original_file = Path(sys.argv[2])
    if len(sys.argv) > 3:
        output_file = Path(sys.argv[3])

    print("=== Filtering Failed Samples ===\n")
    filter_failed_samples(results_file, original_file, output_file)

    print("\nNext step:")
    print(f"Run: modal run llava_batch_modal.py --data-json {output_file} --images-root data --output-json data/results_test_rerun.jsonl")
