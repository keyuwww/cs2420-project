#!/usr/bin/env python3
import json
import re
import os
import argparse
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Define banned words for safety check
BANLIST = re.compile(r"(burn|fire|acid|explode|shock|injure|kill|poison|crash|ignite|melt|break|stab|electrocute)", re.I)

def analyze_prompts(input_file):
    """Analyze the generated prompts from the output file"""
    print(f"Analyzing prompts from: {input_file}")
    
    # Check if file exists
    if not os.path.exists(input_file):
        print(f"Error: File {input_file} not found")
        return
    
    # Read the JSONL file
    entries = []
    try:
        with open(input_file, 'r') as f:
            for line in f:
                if line.strip():
                    entries.append(json.loads(line))
    except Exception as e:
        print(f"Error reading file: {e}")
        return
    
    print(f"Loaded {len(entries)} entries")
    
    # Initialize counters and data structures
    total_prompts = 0
    caption_level_counts = Counter()
    tone_counts = Counter()
    relation_counts = Counter()
    word_counts = []
    banned_word_entries = []
    prompt_lengths = []
    
    # Analyze each entry
    for entry in entries:
        if 'results' not in entry or not entry['results']:
            continue
        
        for result in entry['results']:
            total_prompts += 1
            
            # Count caption levels
            caption_level = result.get('caption_level', 'unknown')
            caption_level_counts[caption_level] += 1
            
            # Count tones and relations
            meta = result.get('meta', {})
            tone = meta.get('tone', 'unknown')
            relation = meta.get('relation', 'unknown')
            tone_counts[tone] += 1
            relation_counts[relation] += 1
            
            # Analyze prompt text
            prompt_text = result.get('emergent_unsafe_prompt', '')
            words = prompt_text.split()
            word_count = len(words)
            word_counts.append(word_count)
            prompt_lengths.append(len(prompt_text))
            
            # Check for banned words
            if BANLIST.search(prompt_text):
                banned_word_entries.append({
                    'image_path': entry.get('image_path', ''),
                    'prompt': prompt_text,
                    'banned_word': BANLIST.search(prompt_text).group(0)
                })
    
    # Calculate statistics
    avg_word_count = sum(word_counts) / len(word_counts) if word_counts else 0
    max_word_count = max(word_counts) if word_counts else 0
    min_word_count = min(word_counts) if word_counts else 0
    
    avg_prompt_length = sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0
    max_prompt_length = max(prompt_lengths) if prompt_lengths else 0
    min_prompt_length = min(prompt_lengths) if prompt_lengths else 0
    
    # Print summary statistics
    print("\n=== SUMMARY STATISTICS ===")
    print(f"Total prompts analyzed: {total_prompts}")
    print(f"Average word count: {avg_word_count:.2f} words (min: {min_word_count}, max: {max_word_count})")
    print(f"Average prompt length: {avg_prompt_length:.2f} characters (min: {min_prompt_length}, max: {max_prompt_length})")
    print(f"Prompts with banned words: {len(banned_word_entries)}")
    
    print("\n=== CAPTION LEVEL DISTRIBUTION ===")
    for level, count in caption_level_counts.most_common():
        percentage = (count / total_prompts) * 100 if total_prompts else 0
        print(f"{level}: {count} ({percentage:.1f}%)")
    
    print("\n=== TONE DISTRIBUTION ===")
    for tone, count in tone_counts.most_common():
        percentage = (count / total_prompts) * 100 if total_prompts else 0
        print(f"{tone}: {count} ({percentage:.1f}%)")
    
    print("\n=== RELATION DISTRIBUTION ===")
    for relation, count in relation_counts.most_common():
        percentage = (count / total_prompts) * 100 if total_prompts else 0
        print(f"{relation}: {count} ({percentage:.1f}%)")
    
    if banned_word_entries:
        print("\n=== PROMPTS WITH BANNED WORDS ===")
        for entry in banned_word_entries:
            print(f"Image: {os.path.basename(entry['image_path'])}")
            print(f"Prompt: {entry['prompt']}")
            print(f"Banned word: {entry['banned_word']}")
            print()
    
    # Create word count histogram
    plt.figure(figsize=(10, 6))
    plt.hist(word_counts, bins=range(0, max(word_counts) + 2), alpha=0.7, color='blue')
    plt.title('Distribution of Prompt Word Counts')
    plt.xlabel('Word Count')
    plt.ylabel('Number of Prompts')
    plt.grid(True, alpha=0.3)
    plt.savefig('word_count_distribution.png')
    print("Word count distribution saved as 'word_count_distribution.png'")
    
    # Create tone distribution pie chart
    plt.figure(figsize=(10, 8))
    labels = [tone for tone, _ in tone_counts.most_common()]
    sizes = [count for _, count in tone_counts.most_common()]
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Distribution of Prompt Tones')
    plt.savefig('tone_distribution.png')
    print("Tone distribution saved as 'tone_distribution.png'")
    
    return {
        'total_prompts': total_prompts,
        'caption_level_counts': caption_level_counts,
        'tone_counts': tone_counts,
        'relation_counts': relation_counts,
        'avg_word_count': avg_word_count,
        'banned_word_entries': banned_word_entries
    }

def main():
    parser = argparse.ArgumentParser(description="Evaluate generated prompts from output file")
    parser.add_argument("--input", type=str, default="../output_250.jsonl", help="Input JSONL file")
    args = parser.parse_args()
    
    analyze_prompts(args.input)

if __name__ == "__main__":
    main()
