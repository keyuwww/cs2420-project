#!/usr/bin/env python3
import json
import os
import argparse
import subprocess
import sys

def main():
    parser = argparse.ArgumentParser(description="Resume processing from checkpoint")
    parser.add_argument("--checkpoint", type=str, default="../checkpoint.json", help="Checkpoint file path")
    args = parser.parse_args()
    
    checkpoint_file = args.checkpoint
    
    # Check if checkpoint file exists
    if not os.path.exists(checkpoint_file):
        print(f"Error: Checkpoint file '{checkpoint_file}' not found.", file=sys.stderr)
        return 1
    
    # Load checkpoint data
    try:
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in checkpoint file '{checkpoint_file}'.", file=sys.stderr)
        return 1
    
    # Display checkpoint information
    print("Resuming from checkpoint:")
    print(f"  Timestamp: {checkpoint.get('timestamp', 'unknown')}")
    print(f"  Progress: {checkpoint.get('processed_count', 0)}/{checkpoint.get('max_count', 250)} images")
    print(f"  Output file: {checkpoint.get('output_file', '../output_250.jsonl')}")
    
    # Run the main script
    cmd = [
        "python", 
        "process_250_images.py",
        "--output", checkpoint.get('output_file', '../output_250.jsonl'),
        "--checkpoint", checkpoint_file
    ]
    
    print("\nRunning command:", " ".join(cmd))
    return subprocess.call(cmd)

if __name__ == "__main__":
    sys.exit(main())
