#!/usr/bin/env python3
"""Quick analysis over the COCO caption → emergent prompt output.

Reads every JSONL produced by `coco_caption_pipeline.py`, groups the
entries by image, and reports how many of the target 550 SPECIFIC images
obtained an emergent prompt (plus some per-caption statistics).
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from pathlib import Path


def load_prompts(pattern: str) -> list[dict]:
    entries = []
    for path in sorted(glob.glob(pattern)):
        with open(path, "r", encoding="utf-8") as fd:
            for line in fd:
                line = line.strip()
                if not line:
                    continue

                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"skipping malformed line in {path}", file=os.sys.stderr)
    return entries


def summarize(entries: list[dict], target_image_count: int) -> dict:
    grouped = defaultdict(list)
    for entry in entries:
        grouped[entry["image_path"]].append(entry)

    total_prompts = len(entries)
    unique_images = len(grouped)
    percent = (unique_images / target_image_count * 100) if target_image_count else 0

    meta_counts = defaultdict(int)
    for entry in entries:
        tone = entry.get("meta", {}).get("tone", "unknown")
        relation = entry.get("meta", {}).get("relation", "unknown")
        meta_counts[f"{tone}|{relation}"] += 1

    return {
        "total_prompts": total_prompts,
        "unique_images": unique_images,
        "share_of_target": percent,
        "target_images": target_image_count,
        "per_image_avg": (total_prompts / unique_images) if unique_images else 0,
        "meta_breakdown": dict(meta_counts),
    }


def main():
    parser = argparse.ArgumentParser(description="Summarize emergent prompt outputs.")
    parser.add_argument(
        "--pattern",
        default="pipeline2_caption_to_prompt/processed_results/data/prompts/coco_caption_prompts*.jsonl",
        help="Glob matching the JSONL files to analyze.",
    )
    parser.add_argument(
        "--target-images",
        type=int,
        default=550,
        help="Expected number of SPECIFIC COCO images to measure against.",
    )

    args = parser.parse_args()
    entries = load_prompts(args.pattern)
    if not entries:
        print("No entries found; check the glob pattern.", file=os.sys.stderr)
        return

    summary = summarize(entries, args.target_images)
    print(f"Total prompts collected: {summary['total_prompts']}")
    print(f"Unique images with prompts: {summary['unique_images']} / {summary['target_images']} "
          f"({summary['share_of_target']:.1f}%)")
    print(f"Average prompts per image: {summary['per_image_avg']:.2f}")
    print("Meta breakdown (tone|relation):")
    for meta, count in sorted(summary["meta_breakdown"].items(), key=lambda item: -item[1]):
        print(f"  {meta}: {count}")


if __name__ == "__main__":
    main()
