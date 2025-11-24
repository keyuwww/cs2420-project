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
        "--chunk-size",
        type=int,
        default=0,
        help="Break output into multiple files every CHUNK_SIZE saved entries (0 disables chunking).",
    )
    parser.add_argument(
        "--start-offset",
        type=int,
        default=0,
        help="Skip the first N caption entries before processing (useful for resuming).",
    )
    parser.add_argument(
        "--data-version",
        type=str,
        default=None,
        help="Optional data version suffix to embed in the output filename (e.g., --data-version v2).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output if it already exists instead of creating a new file.",
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
        default="models/gemini-flash-lite-latest",
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


def resolve_output_path(path: Path, force: bool) -> Path:
    if force or not path.exists():
        return path

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    base = path.stem
    suffix = path.suffix
    parent = path.parent

    candidate = parent / f"{base}-{timestamp}{suffix}"
    counter = 1
    while candidate.exists():
        candidate = parent / f"{base}-{timestamp}-{counter}{suffix}"
        counter += 1
    return candidate


def chunked_output_path(base: Path, chunk_index: int) -> Path:
    if chunk_index == 0:
        return base

    parent = base.parent
    suffix = base.suffix
    stem = base.stem
    candidate = parent / f"{stem}-chunk{chunk_index}{suffix}"
    counter = 1
    while candidate.exists():
        candidate = parent / f"{stem}-chunk{chunk_index}-{counter}{suffix}"
        counter += 1
    return candidate


def versioned_path(base: Path, version: Optional[str]) -> Path:
    if not version:
        return base
    stem = base.stem
    suffix = base.suffix
    return base.with_name(f"{stem}-{version}{suffix}")

def process_caption(
    image_path: Path,
    caption_idx: int,
    caption_text: str,
    tone_override: str | None,
    args: argparse.Namespace,
    model,
    out_file,
    delay: float,
) -> tuple[str, bool]:
    tone_hint = tone_override or random.choice(TONE_BUCKETS)
    user_msg = build_message(caption_text, tone_hint)

    if args.dry_run:
        print(f"[dry-run] {image_path}: {user_msg}")
        return "dry-run", False

    payload = [{"text": user_msg}]
    if not args.omit_image:
        payload.append(as_image_part(str(image_path)))

    try:
        response = generate_with_retry(model, payload)
    except Exception as exc:
        print(f"{image_path} -> ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return f"ERROR: {type(exc).__name__}", False

    data = parse_json_or_retry(response.text or "")
    if not data or "emergent_unsafe_prompt" not in data:
        time.sleep(delay)
        repair_payload = [
            {"text": "Return only the JSON object specified earlier, no prose."},
            *([] if args.omit_image else [as_image_part(str(image_path))]),
        ]
        repair = generate_with_retry(model, repair_payload)
        data = parse_json_or_retry(repair.text or "{}")

    if not data:
        print(f"{image_path} -> FAILED: non-JSON output", file=sys.stderr)
        return "FAILED: non-JSON output", False

    prompt_text = data.get("emergent_unsafe_prompt", "").strip()
    if not looks_safe_alone(prompt_text):
        print(f"{image_path} -> SKIPPED: unsafe-alone or too long: {prompt_text}", file=sys.stderr)
        return "SKIPPED: unsafe-alone or too long", False

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
    if out_file:
        out_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
        out_file.flush()

    return "prompt saved", True


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

    if args.start_offset:
        if args.start_offset >= len(work_items):
            print("Start offset exceeds total work items; nothing to do.", file=sys.stderr)
            return
        work_items = work_items[args.start_offset:]

    if not work_items:
        print("No work items after applying filters/offset; exiting.", file=sys.stderr)
        return

    base_output = versioned_path(args.output, args.data_version)
    output_path = resolve_output_path(base_output, args.force)
    chunk_index = 0
    entries_in_chunk = 0
    saved_count = 0

    def open_chunk_file(idx: int) -> TextIO:
        chunk_path = chunked_output_path(output_path, idx)
        chunk_path.parent.mkdir(parents=True, exist_ok=True)
        return chunk_path.open("w", encoding="utf-8")

    out_file: Optional[TextIO] = None if args.dry_run else open_chunk_file(chunk_index)
    try:
        total_items = len(work_items)
        for idx, (image_path, caption_idx, caption_text) in enumerate(work_items):
            status, saved = process_caption(
                image_path,
                caption_idx,
                caption_text,
                tone_override,
                args,
                model,
                out_file,
                delay,
            )

            print(
                f"[{idx + 1}/{total_items}] {image_path.name} caption {caption_idx}: {status}",
                file=sys.stderr,
            )

            if saved:
                saved_count += 1
                entries_in_chunk += 1

            if args.chunk_size > 0 and entries_in_chunk >= args.chunk_size and idx + 1 < total_items:
                if out_file:
                    out_file.close()
                chunk_index += 1
                entries_in_chunk = 0
                out_file = open_chunk_file(chunk_index)

            processed = idx + 1
            if processed % 50 == 0 or processed == total_items:
                percent = processed / total_items * 100
                saved_rate = saved_count / processed * 100 if processed else 0
                print(
                    f"[checkpoint] {processed}/{total_items} processed ({percent:.1f}%). "
                    f"Saved {saved_count} prompts ({saved_rate:.1f}% of processed)",
                    file=sys.stderr,
                )

            if idx + 1 < total_items:
                time.sleep(delay)
    finally:
        if out_file:
            out_file.close()


if __name__ == "__main__":
    main()
