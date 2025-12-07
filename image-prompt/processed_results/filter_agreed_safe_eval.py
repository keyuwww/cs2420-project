#!/usr/bin/env python3
"""
Filter entries from claude_eval.jsonl and gemini_eval.jsonl where both LLMs agree 
that is_emergent_unsafe is true (i.e., both say it's unsafe).
Matches entries using (emergent_prompt/emergent_unsafe_prompt, image_path) pairs.
"""

import json
import sys
from pathlib import Path


def remove_explanation(evaluation):
    """Remove the explanation field from evaluation dict."""
    eval_copy = evaluation.copy()
    eval_copy.pop('explanation', None)
    return eval_copy


def get_prompt_text(entry):
    """Get prompt text from either emergent_unsafe_prompt or emergent_prompt field."""
    return entry.get('emergent_unsafe_prompt', '') or entry.get('emergent_prompt', '')


def infer_pipeline(entry):
    """Infer pipeline number from entry structure."""
    # If pipeline field exists, use it
    if 'pipeline' in entry:
        pipeline_val = entry.get('pipeline')
        return int(pipeline_val) if pipeline_val is not None else None
    # If caption_index exists, it's likely pipeline 2
    if 'caption_index' in entry:
        return 2
    # If caption_level exists, it's likely pipeline 1
    if 'caption_level' in entry:
        return 1
    return None


def create_match_key(entry):
    """Create a unique key for matching entries between files using (prompt, image_path)."""
    prompt = get_prompt_text(entry)
    image_path = entry.get('image_path', '')
    return (prompt, image_path)


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
    script_dir = Path(__file__).parent
    claude_file = script_dir / 'claude_eval.jsonl'
    gemini_file = script_dir / 'gemini_eval.jsonl'
    output_file = script_dir / 'eval-agreed-unsafe.jsonl'
    
    # Load both files
    print(f"Loading {claude_file}...")
    claude_data = load_jsonl(claude_file)
    print(f"Loaded {len(claude_data)} entries from Claude evaluation")
    
    print(f"Loading {gemini_file}...")
    gemini_data = load_jsonl(gemini_file)
    print(f"Loaded {len(gemini_data)} entries from Gemini evaluation")
    
    # Create a mapping by match key (prompt, image_path)
    claude_by_key = {}
    for entry in claude_data:
        key = create_match_key(entry)
        if key in claude_by_key:
            print(f"Warning: Duplicate key found in Claude data: {key[1][:50]}...")
        claude_by_key[key] = entry
    
    gemini_by_key = {}
    for entry in gemini_data:
        key = create_match_key(entry)
        if key in gemini_by_key:
            print(f"Warning: Duplicate key found in Gemini data: {key[1][:50]}...")
        gemini_by_key[key] = entry
    
    # Find entries where both agree that is_emergent_unsafe is false (safe)
    agreed_safe = []
    matched_keys = set(claude_by_key.keys()) & set(gemini_by_key.keys())
    print(f"\nFound {len(matched_keys)} matching (prompt, image_path) pairs between files")
    
    unmatched_claude = set(claude_by_key.keys()) - set(gemini_by_key.keys())
    unmatched_gemini = set(gemini_by_key.keys()) - set(claude_by_key.keys())
    
    if unmatched_claude:
        print(f"Warning: {len(unmatched_claude)} entries in Claude file have no match in Gemini file")
    if unmatched_gemini:
        print(f"Warning: {len(unmatched_gemini)} entries in Gemini file have no match in Claude file")
    
    for key in matched_keys:
        claude_entry = claude_by_key[key]
        gemini_entry = gemini_by_key[key]
        
        claude_unsafe = claude_entry.get('evaluation', {}).get('is_emergent_unsafe', False)  # Default to False if missing
        gemini_unsafe = gemini_entry.get('evaluation', {}).get('is_emergent_unsafe', False)  # Default to False if missing
        
        # Check if both agree that it's unsafe (is_emergent_unsafe is True)
        if claude_unsafe and gemini_unsafe:
            # Get pipeline value (try direct field first, then infer)
            pipeline_val = claude_entry.get('pipeline')
            if pipeline_val is None:
                # Infer from structure
                pipeline_val = infer_pipeline(claude_entry)
            
            # Create output entry
            output_entry = {
                'image_path': claude_entry.get('image_path', ''),
                'caption': claude_entry.get('caption', ''),
                'emergent_unsafe_prompt': get_prompt_text(claude_entry),
                'pipeline': pipeline_val,
                'caption_index': claude_entry.get('caption_index'),
                'tone': claude_entry.get('tone'),
                'evaluation_meta': {
                    'evaluation_claude': remove_explanation(claude_entry.get('evaluation', {})),
                    'evaluation_gemini': remove_explanation(gemini_entry.get('evaluation', {}))
                }
            }
            agreed_safe.append(output_entry)
    
    # Save results
    print(f"\nFound {len(agreed_safe)} entries where both LLMs agree is_emergent_unsafe is true (unsafe)")
    save_jsonl(agreed_safe, output_file)
    print(f"Saved results to {output_file}")
    
    # Print some statistics
    if agreed_safe:
        print(f"\nFirst few entries:")
        for i, entry in enumerate(agreed_safe[:3], 1):
            print(f"\n{i}. {Path(entry['image_path']).name}")
            print(f"   Caption: {entry['caption'][:60]}...")
            print(f"   Prompt: {entry['emergent_unsafe_prompt'][:80]}...")
            print(f"   Claude severity: {entry['evaluation_meta']['evaluation_claude'].get('severity', 'N/A')}")
            print(f"   Gemini severity: {entry['evaluation_meta']['evaluation_gemini'].get('severity', 'N/A')}")
        
        # Count by severity
        claude_severities = {}
        gemini_severities = {}
        for entry in agreed_safe:
            claude_sev = entry['evaluation_meta']['evaluation_claude'].get('severity', 'unknown')
            gemini_sev = entry['evaluation_meta']['evaluation_gemini'].get('severity', 'unknown')
            claude_severities[claude_sev] = claude_severities.get(claude_sev, 0) + 1
            gemini_severities[gemini_sev] = gemini_severities.get(gemini_sev, 0) + 1
        
        print(f"\nSeverity distribution (Claude): {claude_severities}")
        print(f"Severity distribution (Gemini): {gemini_severities}")


if __name__ == '__main__':
    main()

