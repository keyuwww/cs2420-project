#!/usr/bin/env python3
"""
Convert eval-agreed-unsafe.jsonl to benchmark format.
Extracts emergent_unsafe_prompt as unsafe prompts with label=1.

Output format:
{"image_path": "image/filename.jpg", "prompt": "unsafe prompt text", "label": 1}
"""

import json
import os
from pathlib import Path

def convert_unsafe_to_benchmark(input_path, output_path):
    """
    Convert eval-agreed-unsafe.jsonl to benchmark format.

    Args:
        input_path: Path to eval-agreed-unsafe.jsonl
        output_path: Path to output benchmark_unsafe.jsonl
    """
    print(f"Reading unsafe data from: {input_path}")

    benchmark_entries = []

    with open(input_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                entry = json.loads(line.strip())

                # Extract image path and convert to image/ format
                original_path = entry.get('image_path', '')
                filename = os.path.basename(original_path)
                image_path = f"image/{filename}"

                # Extract unsafe prompt
                prompt = entry.get('emergent_unsafe_prompt', '').strip()

                if prompt:
                    benchmark_entries.append({
                        'image_path': image_path,
                        'prompt': prompt,
                        'label': 1  # Unsafe
                    })
                else:
                    print(f"Warning: Line {line_num} has no emergent_unsafe_prompt")

            except json.JSONDecodeError as e:
                print(f"Error parsing line {line_num}: {e}")
                continue

    # Write to output file (JSONL format)
    with open(output_path, 'w', encoding='utf-8') as f:
        for entry in benchmark_entries:
            f.write(json.dumps(entry) + '\n')

    print(f"\nConversion complete!")
    print(f"  Total unsafe prompts extracted: {len(benchmark_entries)}")
    print(f"  Output saved to: {output_path}")

    return benchmark_entries

if __name__ == "__main__":
    import sys

    # Go up to lora/ directory, then into data/
    lora_dir = Path(__file__).parent.parent.parent
    data_dir = lora_dir / "data"

    # Default paths
    input_file = data_dir / "eval-agreed-unsafe.jsonl"
    output_file = data_dir / "benchmark_unsafe.jsonl"

    if len(sys.argv) > 1:
        input_file = Path(sys.argv[1])

    if len(sys.argv) > 2:
        output_file = Path(sys.argv[2])

    if not input_file.exists():
        print(f"Error: Input file not found at: {input_file}")
        print(f"\nUsage: python {Path(__file__).name} [input.jsonl] [output.jsonl]")
        sys.exit(1)

    # Convert
    convert_unsafe_to_benchmark(input_file, output_file)

    print(f"\nDone! Unsafe benchmark created.")
    print(f"\nNext steps:")
    print(f"1. Copy images from coco_val_sample-SPECIFIC to lora/data/image/:")
    print(f"   cp coco_val_sample-SPECIFIC/*.jpg lora/data/image/")
    print(f"2. Balance with safe benchmark if needed:")
    print(f"   python balance_benchmarks.py")
