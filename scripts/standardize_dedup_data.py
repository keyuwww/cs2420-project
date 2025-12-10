#!/usr/bin/env python3
"""
Standardize dedup_data.jsonl to have consistent format across all rows
"""

import json

def determine_pipeline(entry):
    """
    Determine pipeline based on caption presence and caption_level.
    Pipeline 1: Has caption_level (brief/comprehensive/detailed)
    Pipeline 2: Has caption but no caption_level
    Pipeline 3: No caption
    """
    # Format 3: has 'captions' and 'results' with caption_level
    if 'captions' in entry and 'results' in entry:
        return "1"

    # Format 1 & 2: has singular 'caption'
    if 'caption' in entry and entry.get('caption'):
        return "2"

    # No caption
    return "3"

def standardize_entry(entry):
    """
    Convert an entry to standardized format.
    If entry has 'results' array, return list of standardized entries.
    Otherwise return single standardized entry.
    """
    # Handle Format 3: captions (plural) with results array
    if 'captions' in entry and 'results' in entry:
        standardized_entries = []
        for result in entry['results']:
            std_entry = {
                "image_path": entry.get('image_path', None),
                "caption_index": None,  # Not provided in this format
                "caption": result.get('caption_text', None),
                "caption_level": result.get('caption_level', None),
                "emergent_unsafe_prompt": result.get('emergent_unsafe_prompt', None),
                "meta": {
                    "tone": result.get('meta', {}).get('tone', None),
                    "severity": result.get('meta', {}).get('severity', None),
                    "relation": result.get('meta', {}).get('relation', None)
                },
                "evaluation": result.get('evaluation', None),
                "pipeline": "1"
            }
            standardized_entries.append(std_entry)
        return standardized_entries

    # Handle Format 1 & 2: singular caption
    else:
        meta = entry.get('meta', {})
        std_entry = {
            "image_path": entry.get('image_path', None),
            "caption_index": entry.get('caption_index', None),
            "caption": entry.get('caption', None),
            "caption_level": None,  # These formats don't have caption_level
            "emergent_unsafe_prompt": entry.get('emergent_unsafe_prompt', None),
            "meta": {
                "tone": meta.get('tone', None),
                "severity": meta.get('severity', None),
                "relation": meta.get('relation', None)
            },
            "evaluation": entry.get('evaluation', None),
            "pipeline": determine_pipeline(entry)
        }
        return [std_entry]

# Process the file
print("Reading dedup_data.jsonl...")
input_file = 'dedup_data.jsonl'
output_file = 'dedup_data_standardized.jsonl'

total_input = 0
total_output = 0
pipeline_counts = {"1": 0, "2": 0, "3": 0}

with open(input_file, 'r', encoding='utf-8') as f_in, \
     open(output_file, 'w', encoding='utf-8') as f_out:

    for line in f_in:
        total_input += 1
        entry = json.loads(line.strip())

        # Standardize and write
        standardized = standardize_entry(entry)
        for std_entry in standardized:
            f_out.write(json.dumps(std_entry, ensure_ascii=False) + '\n')
            total_output += 1
            pipeline_counts[std_entry['pipeline']] += 1

print(f"✓ Standardization complete!")
print(f"\nInput:  {total_input} entries")
print(f"Output: {total_output} entries")
print(f"\nPipeline distribution:")
print(f"  Pipeline 1 (has caption_level): {pipeline_counts['1']} ({pipeline_counts['1']/total_output*100:.1f}%)")
print(f"  Pipeline 2 (has caption, no level): {pipeline_counts['2']} ({pipeline_counts['2']/total_output*100:.1f}%)")
print(f"  Pipeline 3 (no caption): {pipeline_counts['3']} ({pipeline_counts['3']/total_output*100:.1f}%)")
print(f"\n✓ Saved to: {output_file}")
