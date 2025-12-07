#!/usr/bin/env python3
"""
Check if there's a 1-1 correspondence between (prompt, image_path) pairs
in claude_eval.jsonl and gemini_eval.jsonl
"""

import json
from collections import defaultdict
from pathlib import Path


def get_prompt_text(entry):
    """Get prompt text from either emergent_unsafe_prompt or emergent_prompt field."""
    return entry.get('emergent_unsafe_prompt', '') or entry.get('emergent_prompt', '')


def load_jsonl(filepath):
    """Load JSONL file and return list of dicts."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def main():
    script_dir = Path(__file__).parent
    claude_file = script_dir / 'claude_eval.jsonl'
    gemini_file = script_dir / 'gemini_eval.jsonl'
    
    print("Loading files...")
    claude_data = load_jsonl(claude_file)
    gemini_data = load_jsonl(gemini_file)
    
    print(f"Claude entries: {len(claude_data)}")
    print(f"Gemini entries: {len(gemini_data)}")
    
    # Build sets of (prompt, image_path) pairs
    claude_pairs = set()
    gemini_pairs = set()
    
    claude_prompt_to_images = defaultdict(set)
    claude_image_to_prompts = defaultdict(set)
    gemini_prompt_to_images = defaultdict(set)
    gemini_image_to_prompts = defaultdict(set)
    
    for entry in claude_data:
        prompt = get_prompt_text(entry)
        image_path = entry.get('image_path', '')
        if prompt and image_path:
            pair = (prompt, image_path)
            claude_pairs.add(pair)
            claude_prompt_to_images[prompt].add(image_path)
            claude_image_to_prompts[image_path].add(prompt)
    
    for entry in gemini_data:
        prompt = get_prompt_text(entry)
        image_path = entry.get('image_path', '')
        if prompt and image_path:
            pair = (prompt, image_path)
            gemini_pairs.add(pair)
            gemini_prompt_to_images[prompt].add(image_path)
            gemini_image_to_prompts[image_path].add(prompt)
    
    print(f"\nClaude unique (prompt, image_path) pairs: {len(claude_pairs)}")
    print(f"Gemini unique (prompt, image_path) pairs: {len(gemini_pairs)}")
    
    # Check 1-1 correspondence
    print("\n=== Checking 1-1 Correspondence ===")
    
    # Check if each prompt maps to exactly one image
    claude_prompt_violations = {p: imgs for p, imgs in claude_prompt_to_images.items() if len(imgs) > 1}
    gemini_prompt_violations = {p: imgs for p, imgs in gemini_prompt_to_images.items() if len(imgs) > 1}
    
    print(f"\nClaude: Prompts that map to multiple images: {len(claude_prompt_violations)}")
    if claude_prompt_violations:
        print("  Examples:")
        for prompt, images in list(claude_prompt_violations.items())[:3]:
            print(f"    Prompt: {prompt[:60]}...")
            print(f"    Images: {len(images)}")
    
    print(f"\nGemini: Prompts that map to multiple images: {len(gemini_prompt_violations)}")
    if gemini_prompt_violations:
        print("  Examples:")
        for prompt, images in list(gemini_prompt_violations.items())[:3]:
            print(f"    Prompt: {prompt[:60]}...")
            print(f"    Images: {len(images)}")
    
    # Check if each image maps to exactly one prompt
    claude_image_violations = {img: prompts for img, prompts in claude_image_to_prompts.items() if len(prompts) > 1}
    gemini_image_violations = {img: prompts for img, prompts in gemini_image_to_prompts.items() if len(prompts) > 1}
    
    print(f"\nClaude: Images that map to multiple prompts: {len(claude_image_violations)}")
    if claude_image_violations:
        print("  Examples:")
        for img, prompts in list(claude_image_violations.items())[:3]:
            print(f"    Image: {Path(img).name}")
            print(f"    Prompts: {len(prompts)}")
    
    print(f"\nGemini: Images that map to multiple prompts: {len(gemini_image_violations)}")
    if gemini_image_violations:
        print("  Examples:")
        for img, prompts in list(gemini_image_violations.items())[:3]:
            print(f"    Image: {Path(img).name}")
            print(f"    Prompts: {len(prompts)}")
    
    # Check overlap between files
    common_pairs = claude_pairs & gemini_pairs
    claude_only = claude_pairs - gemini_pairs
    gemini_only = gemini_pairs - claude_pairs
    
    print(f"\n=== Overlap Analysis ===")
    print(f"Common (prompt, image_path) pairs: {len(common_pairs)}")
    print(f"Claude-only pairs: {len(claude_only)}")
    print(f"Gemini-only pairs: {len(gemini_only)}")
    
    # Check if it's 1-1
    is_one_to_one = (
        len(claude_prompt_violations) == 0 and
        len(gemini_prompt_violations) == 0 and
        len(claude_image_violations) == 0 and
        len(gemini_image_violations) == 0 and
        len(claude_only) == 0 and
        len(gemini_only) == 0
    )
    
    print(f"\n=== Conclusion ===")
    if is_one_to_one:
        print("✓ Perfect 1-1 correspondence: Each (prompt, image_path) pair appears exactly once in both files")
    else:
        print("✗ NOT a perfect 1-1 correspondence")
        if claude_prompt_violations or gemini_prompt_violations:
            print("  - Some prompts map to multiple images")
        if claude_image_violations or gemini_image_violations:
            print("  - Some images map to multiple prompts")
        if claude_only or gemini_only:
            print("  - Some pairs exist in only one file")
    
    return is_one_to_one


if __name__ == '__main__':
    main()

