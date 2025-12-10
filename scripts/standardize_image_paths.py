#!/usr/bin/env python3
"""
Standardize all image paths to use consistent folder name
"""

import json
import os

# Standard folder to use
STANDARD_FOLDER = "coco_val_sample-SPECIFIC/"

input_file = 'dedup_data_minimal.jsonl'
output_file = 'dedup_data_minimal_standardized.jsonl'

total = 0
changed = 0
path_variations = set()

with open(input_file, 'r', encoding='utf-8') as f_in, \
     open(output_file, 'w', encoding='utf-8') as f_out:

    for line in f_in:
        entry = json.loads(line)
        total += 1

        # Extract just the filename
        original_path = entry['image_path']
        filename = os.path.basename(original_path)
        folder = os.path.dirname(original_path)

        # Track different folder variations
        if folder:
            path_variations.add(folder)

        # Create standardized path
        standardized_path = STANDARD_FOLDER + filename

        # Update if different
        if original_path != standardized_path:
            entry['image_path'] = standardized_path
            changed += 1

        # Write updated entry
        f_out.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"✓ Standardization complete!")
print(f"\nTotal entries: {total}")
print(f"Changed paths: {changed}")
print(f"Unchanged paths: {total - changed}")

print(f"\nOriginal folder variations found ({len(path_variations)}):")
for folder in sorted(path_variations):
    print(f"  - '{folder}'")

print(f"\nStandardized all to: '{STANDARD_FOLDER}'")
print(f"\n✓ Saved to: {output_file}")
