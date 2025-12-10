import json
import os

print("=" * 80)
print("MERGING EVALUATION WITH ORIGINAL DATA")
print("=" * 80)

input_file = 'dedup_data_1300.jsonl'
eval_file = 'dedup_data_1300_gemini_eval.jsonl'
output_file = 'dedup_data_1300_with_gemini_eval.jsonl'

# Load evaluation results into a dictionary keyed by (image_filename, prompt)
evaluations = {}
with open(eval_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            entry = json.loads(line)
            img_path = entry.get('image_path', '')
            img_filename = os.path.basename(img_path)
            prompt = entry.get('emergent_unsafe_prompt', entry.get('emergent_prompt', ''))
            
            # Use filename + prompt as key (matching the deduplication logic)
            key = f"{img_filename}::{prompt}"
            evaluations[key] = entry.get('evaluation', {})

print(f"Loaded {len(evaluations)} evaluations from {eval_file}")

# Merge with input file
merged = []
matched = 0
unmatched = 0

with open(input_file, 'r', encoding='utf-8') as f:
    for line in f:
        if line.strip():
            entry = json.loads(line)
            img_path = entry.get('image_path', '')
            img_filename = os.path.basename(img_path)
            prompt = entry.get('emergent_unsafe_prompt', '')
            
            # Create key to match
            key = f"{img_filename}::{prompt}"
            
            # Add evaluation if found
            if key in evaluations:
                entry['evaluation'] = evaluations[key]
                matched += 1
            else:
                # No evaluation found - add null evaluation
                entry['evaluation'] = {
                    "safe_alone": None,
                    "unsafe_with_context": None,
                    "implied_danger": "No evaluation found",
                    "severity": None,
                    "is_emergent_unsafe": None,
                    "explanation": "No evaluation found for this entry"
                }
                unmatched += 1
            
            merged.append(entry)

# Write merged data
with open(output_file, 'w', encoding='utf-8') as f:
    for entry in merged:
        f.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"Loaded {len(merged)} entries from {input_file}")
print(f"\nMatching results:")
print(f"  ✓ Matched: {matched} ({matched/len(merged)*100:.1f}%)")
print(f"  ✗ Unmatched: {unmatched} ({unmatched/len(merged)*100:.1f}%)")
print(f"\n✓ Saved merged data to: {output_file}")

# Show sample entry
if merged:
    print("\nSample merged entry (first entry):")
    print(json.dumps(merged[0], indent=2, ensure_ascii=False)[:500] + "...")
