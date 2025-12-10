#!/usr/bin/env python3
"""
Create a minimal format version that only includes non-null fields
"""

import json

def create_minimal_entry(entry):
    """Remove null fields and flatten structure."""
    minimal = {
        "image_path": entry["image_path"],
        "caption": entry["caption"],
        "emergent_unsafe_prompt": entry["emergent_unsafe_prompt"],
        "pipeline": entry["pipeline"]
    }

    # Add caption_index only if not null
    if entry.get("caption_index") is not None:
        minimal["caption_index"] = entry["caption_index"]

    # Add caption_level only if not null
    if entry.get("caption_level") is not None:
        minimal["caption_level"] = entry["caption_level"]

    # Flatten meta fields (only non-null)
    meta = entry.get("meta", {})
    if meta.get("tone") is not None:
        minimal["tone"] = meta["tone"]
    if meta.get("severity") is not None:
        minimal["severity"] = meta["severity"]
    if meta.get("relation") is not None:
        minimal["relation"] = meta["relation"]

    # Add evaluation only if not null
    if entry.get("evaluation") is not None:
        minimal["evaluation"] = entry["evaluation"]

    return minimal

# Process file
print("Creating minimal format...")
input_file = 'dedup_data_standardized.jsonl'
output_file = 'dedup_data_minimal.jsonl'

total = 0
with open(input_file, 'r', encoding='utf-8') as f_in, \
     open(output_file, 'w', encoding='utf-8') as f_out:

    for line in f_in:
        entry = json.loads(line)
        minimal = create_minimal_entry(entry)
        f_out.write(json.dumps(minimal, ensure_ascii=False) + '\n')
        total += 1

print(f"✓ Created minimal format: {output_file}")
print(f"✓ Processed {total} entries")
print(f"\nChanges:")
print(f"  - Removed null fields")
print(f"  - Flattened meta.* fields to top level")
print(f"  - Kept only fields with actual data")
