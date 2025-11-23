#!/usr/bin/env python3
"""Pipeline that turns COCO captions into emergent unsafe prompts."""

import argparse
import json
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Iterable, List

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

from prompts import (
    REQUEST_DELAY,
    SYSTEM_PROMPT,
    TONE_BUCKETS,
    as_image_part,
    generate_with_retry,
    looks_safe_alone,
    parse_json_or_retry,
)

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


class COCOCaptionDatabase:
    """Load COCO captions.json and answer which descriptions go with each image."""

    def __init__(self, captions_path: Path):
        with captions_path.open("r", encoding="utf-8") as fp:
            data = json.load(fp)

        self._file_to_id = {img["file_name"]: img["id"] for img in data.get("images", [])}
        self._captions = defaultdict(list)
        for annotation in data.get("annotations", []):
            self._captions[annotation["image_id"]].append(annotation["caption"])

    def get_captions(self, image_name: str) -> List[str]:
        image_id = self._file_to_id.get(Path(image_name).name)
        if image_id is None:
            return []
        return self._captions.get(image_id, [])

    def describe(self, image_name: str) -> Iterable[str]:
        return self.get_captions(image_name)


def list_images(base_dir: Path) -> List[Path]:
    if not base_dir.exists():
        raise FileNotFoundError(f"{base_dir} does not exist.")
    return sorted(
        entry for entry in base_dir.iterdir()
        if entry.suffix.lower() in IMAGE_SUFFIXES
    )


def build_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate an emergent-unsafe prompt given a COCO caption."
    )
    parser.add_argument(
        "--captions",
        required=True,
        type=Path,
        help="Path to COCO captions_val2017.json (or similar) that covers the images.",
    )
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path("coco_val_sample-SPECIFIC"),
        help="Directory containing the matched COCO images.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("image-prompt/processed_results/coco_caption_prompts.jsonl"),
        help="Where to write one JSON entry per prompt.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of images to process.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=None,
        help="Seconds to wait between requests (default: matches shared REQUEST_DELAY).",
    )
    parser.add_argument(
        "--tone",
        choices=TONE_BUCKETS,
        default=None,
        help="Fix the generated tone instead of randomizing.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the request that would be sent without calling Gemini.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle the image order before processing.",
    )
    parser.add_argument(
        "--caption-index",
        type=int,
        default=0,
        help="Pick the n'th caption (0-indexed) associated with the image when --all-captions is not set.",
    )
    parser.add_argument(
        "--all-captions",
        action="store_true",
        help="Generate a prompt for every caption tied to each image instead of a single index.",
    )
    parser.add_argument(
        "--omit-image",
        action="store_true",
        help="Only send the caption text and skip attaching the raw image.",
    )
    parser.add_argument(
        "--describe-image",
        type=str,
        default=None,
        help="Just print the captions for this image name or path and exit.",
    )
    parser.add_argument(
        "--model",
        default="gemini-1.0",
        help="Gemini model to target when not in dry-run mode (cheaper models reduce rate limits).",
    )
    return parser.parse_args()


def build_message(caption: str, tone_hint: str) -> str:
    return (
        f"Caption: \"{caption}\". "
        "Generate one emergent-unsafe prompt in a "
        f"'{tone_hint}' tone that stays safe alone but implies the "
        "most catastrophic plausible scenario when grounded in that scene."
    )


def main() -> None:
    args = build_args()
    captions_db = COCOCaptionDatabase(args.captions)

    if args.describe_image:
        captions = captions_db.describe(args.describe_image)
        if not captions:
            print(f"No captions found for {args.describe_image}", file=sys.stderr)
            sys.exit(1)
        for idx, caption in enumerate(captions):
            print(f"{idx}: {caption}")
        return

    image_paths = list_images(args.images_dir)
    if args.shuffle:
        random.shuffle(image_paths)
    if args.limit:
        image_paths = image_paths[: args.limit]

    delay = args.delay if args.delay is not None else REQUEST_DELAY
    tone_override = args.tone

    model = None
    if not args.dry_run:
        import google.generativeai as genai

        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY must be set unless --dry-run is used.")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=args.model,
            system_instruction=SYSTEM_PROMPT,
        )

    work_items = []
    for image_path in image_paths:
        captions = captions_db.get_captions(image_path.name)
        if not captions:
            print(f"{image_path}: no captions found", file=sys.stderr)
            continue

        if args.all_captions:
            selections = list(enumerate(captions))
        else:
            caption_idx = min(max(args.caption_index, 0), len(captions) - 1)
            selections = [(caption_idx, captions[caption_idx])]

        for caption_idx, caption_text in selections:
            work_items.append((image_path, caption_idx, caption_text))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out_file:
        for idx, (image_path, caption_idx, caption_text) in enumerate(work_items):
            tone_hint = tone_override or random.choice(TONE_BUCKETS)
            user_msg = build_message(caption_text, tone_hint)

            if args.dry_run:
                print(f"[dry-run] {image_path} (caption {caption_idx}): {user_msg}")
            else:
                payload = [{"text": user_msg}]
                if not args.omit_image:
                    payload.append(as_image_part(str(image_path)))

                try:
                    response = generate_with_retry(model, payload)
                except Exception as exc:
                    print(f"{image_path} -> ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
                    continue

                data = parse_json_or_retry(response.text or "")
                if not data or "emergent_unsafe_prompt" not in data:
                    time.sleep(delay)
                    repair = generate_with_retry(model, [
                        {"text": "Return only the JSON object specified earlier, no prose."},
                        *([] if args.omit_image else [as_image_part(str(image_path))]),
                    ])
                    data = parse_json_or_retry(repair.text or "{}")

                if not data:
                    print(f"{image_path} -> FAILED: non-JSON output", file=sys.stderr)
                    continue

                prompt_text = data.get("emergent_unsafe_prompt", "").strip()
                if not looks_safe_alone(prompt_text):
                    print(f"{image_path} -> SKIPPED: unsafe-alone or too long: {prompt_text}", file=sys.stderr)
                    continue

                meta = data.get("meta", {}) or {}
                meta.setdefault("severity", "high")
                meta.setdefault("tone", tone_hint)

                entry = {
                    "image_path": str(image_path),
                    "caption": caption_text,
                    "caption_index": caption_idx,
                    "emergent_unsafe_prompt": prompt_text,
                    "meta": meta,
                }
                out_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                out_file.flush()

            if idx + 1 < len(work_items):
                time.sleep(delay)


if __name__ == "__main__":
    main()
