# Processing 500 Images with Checkpoint Support

This set of scripts allows you to process the first 500 images from the COCO dataset with checkpoint support to handle rate limiting issues.

## Scripts

1. `process_250_images.py` - Main script to process images and generate captions and emergent unsafe prompts
2. `resume_from_checkpoint.py` - Helper script to resume processing from a checkpoint

## Usage

### Initial Run

To start processing the first 250 images:

```bash
python process_250_images.py --dir ../coco_val_sample-SPECIFIC --output ../output_500.jsonl
```

Additional options:
- `--delay N` - Set delay between API calls in seconds (default: 5)
- `--max-images N` - Set maximum number of images to process (default: 500)
- `--checkpoint-interval N` - Save checkpoint every N images (default: 10)
- `--checkpoint PATH` - Set checkpoint file path (default: ../checkpoint.json)

### Resuming After Rate Limit

If you hit a rate limit or the script stops for any reason, you can resume from the last checkpoint:

```bash
python resume_from_checkpoint.py
```

or

```bash
python resume_from_checkpoint.py --checkpoint ../checkpoint.json
```

## How It Works

1. The script processes images in batches, saving progress to a checkpoint file at regular intervals
2. If processing is interrupted, you can resume from the last checkpoint
3. The script keeps track of which images have been processed and will only process new images
4. Results are appended to the output file in JSONL format (one JSON object per line)

## Output Format

The output file is in JSONL format with each line containing a JSON object with:

```json
{
  "image_path": "path/to/image.jpg",
  "captions": {
    "brief": "...",
    "detailed": "...",
    "comprehensive": "..."
  },
  "results": [
    {
      "caption_level": "brief",
      "caption_text": "...",
      "emergent_unsafe_prompt": "...",
      "meta": {
        "severity": "high",
        "tone": "...",
        "relation": "..."
      }
    },
    // Similar objects for "detailed" and "comprehensive" levels
  ]
}
```

## Checkpoint Format

The checkpoint file is a JSON file with:

```json
{
  "output_file": "../output_250.jsonl",
  "processed_count": 42,
  "max_count": 250,
  "timestamp": "2025-11-26 15:30:45"
}
```
