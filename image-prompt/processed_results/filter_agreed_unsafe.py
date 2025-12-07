#!/usr/bin/env python3
"""
Filter entries where both Claude and Gemini agree that is_emergent_unsafe is true.
Saves entries without explanation fields but with evaluation_meta containing both evaluations.
"""

import json
import sys
from pathlib import Path


def remove_explanation(evaluation):
    """Remove the explanation field from evaluation dict."""
    eval_copy = evaluation.copy()
    eval_copy.pop('explanation', None)
    return eval_copy


def load_jsonl(filepath):
    """Load JSONL file and return list of dicts."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data, filepath):
    """Save list of dicts to JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def main():
    # File paths
    claude_file = Path(__file__).parent / 'evaluation-claude.jsonl'
    gemini_file = Path(__file__).parent / 'evaluation-updated-gemini.jsonl'
    output_file = Path(__file__).parent / 'evaluation-agreed-unsafe.jsonl'
    
    # Load both files
    print(f"Loading {claude_file}...")
    claude_data = load_jsonl(claude_file)
    print(f"Loaded {len(claude_data)} entries from Claude evaluation")
    
    print(f"Loading {gemini_file}...")
    gemini_data = load_jsonl(gemini_file)
    print(f"Loaded {len(gemini_data)} entries from Gemini evaluation")
    
    # Create a mapping by image_path for easier lookup
    claude_by_path = {item['image_path']: item for item in claude_data}
    gemini_by_path = {item['image_path']: item for item in gemini_data}
    
    # Find entries where both agree that is_emergent_unsafe is true
    agreed_unsafe = []
    
    # Check all image paths that appear in both files
    common_paths = set(claude_by_path.keys()) & set(gemini_by_path.keys())
    print(f"\nFound {len(common_paths)} common image paths")
    
    for image_path in common_paths:
        claude_entry = claude_by_path[image_path]
        gemini_entry = gemini_by_path[image_path]
        
        claude_unsafe = claude_entry.get('evaluation', {}).get('is_emergent_unsafe', False)
        gemini_unsafe = gemini_entry.get('evaluation', {}).get('is_emergent_unsafe', False)
        
        # Check if both agree that it's unsafe
        if claude_unsafe and gemini_unsafe:
            # Create output entry
            output_entry = {
                'image_path': image_path,
                'emergent_prompt': claude_entry['emergent_prompt'],  # Should be same in both
                'meta': claude_entry['meta'],  # Should be same in both
                'evaluation_meta': {
                    'evaluation_claude': remove_explanation(claude_entry['evaluation']),
                    'evaluation_gemini': remove_explanation(gemini_entry['evaluation'])
                }
            }
            agreed_unsafe.append(output_entry)
    
    # Save results
    print(f"\nFound {len(agreed_unsafe)} entries where both LLMs agree is_emergent_unsafe is true")
    save_jsonl(agreed_unsafe, output_file)
    print(f"Saved results to {output_file}")
    
    # Print some statistics
    if agreed_unsafe:
        print(f"\nFirst few entries:")
        for i, entry in enumerate(agreed_unsafe[:3], 1):
            print(f"\n{i}. {Path(entry['image_path']).name}")
            print(f"   Prompt: {entry['emergent_prompt'][:80]}...")
            print(f"   Claude severity: {entry['evaluation_meta']['evaluation_claude'].get('severity', 'N/A')}")
            print(f"   Gemini severity: {entry['evaluation_meta']['evaluation_gemini'].get('severity', 'N/A')}")


if __name__ == '__main__':
    main()

