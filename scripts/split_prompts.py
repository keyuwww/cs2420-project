#!/usr/bin/env python3
"""
Split nested prompts into individual lines.

Takes a JSONL file with nested "results" arrays and splits each prompt
into its own line for easier processing.

Usage:
    python3 split_prompts.py --input to_run_on_claude_1500.jsonl --output to_run_on_claude_1500_split.jsonl
"""

import json
import argparse
import sys

def split_prompts(input_file, output_file):
    """Split nested prompts into individual JSONL lines."""

    total_entries = 0
    total_prompts = 0

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            if not line.strip():
                continue

            entry = json.loads(line)
            total_entries += 1

            image_path = entry.get("image_path", "")

            # Check if this entry has nested results
            if "results" in entry:
                # Split each result into its own line
                for result in entry.get("results", []):
                    split_entry = {
                        "image_path": image_path,
                        "caption_index": result.get("caption_index", None),
                        "caption_level": result.get("caption_level", "unknown"),
                        "caption_text": result.get("caption_text", ""),
                        "emergent_unsafe_prompt": result.get("emergent_unsafe_prompt", ""),
                        "meta": result.get("meta", {})
                    }

                    outfile.write(json.dumps(split_entry, ensure_ascii=False) + '\n')
                    total_prompts += 1
            else:
                # Already in simple format, just copy it
                outfile.write(line)
                total_prompts += 1

    print(f"✓ Processed {total_entries} entries")
    print(f"✓ Created {total_prompts} individual prompt lines")
    print(f"✓ Output saved to: {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Split nested prompts into individual lines")
    parser.add_argument("--input", required=True, help="Input JSONL file")
    parser.add_argument("--output", required=True, help="Output JSONL file")
    args = parser.parse_args()

    split_prompts(args.input, args.output)

if __name__ == "__main__":
    main()
