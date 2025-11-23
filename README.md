# cs2420-project

## Structure
- `coco_val_sample-SPECIFIC` and `coco_val_sample-RANDOM` each ship a handful of COCO `val2017` images with their `instances_val2017.json` metadata.
- `coco_pipeline1/src/save_captions.py` can extract mappings from a full `captions_val2017.json` file when you have it locally.
- `image-prompt/processed_results` now contains the emergent prompt harness (`emerge.py`), shared prompt helpers, and a caption-driven pipeline that ties COCO captions to Gemini output.

## Caption → Unsafe prompt pipeline
1. Download the matching COCO caption file (typically `annotations/captions_val2017.json`) from the official MS COCO release and place it somewhere accessible.
2. Ensure your `GEMINI_API_KEY` is exported in the environment so the Gemini client can authenticate.
3. Run the pipeline:

   ```bash
   GEMINI_API_KEY=… python image-prompt/processed_results/coco_caption_pipeline.py \
     --captions /path/to/captions_val2017.json \
     --images-dir coco_val_sample-SPECIFIC \
     --output image-prompt/processed_results/coco_caption_prompts.jsonl
   ```

   The script walks the image directory in order (or shuffles with `--shuffle`), looks up the caption, and asks the Gemini model specified by `--model` to craft one JSON-safe unsafe prompt. It writes one JSON object per line, including the caption used and the model metadata.

4. Use `--dry-run` to verify the request text without hitting the API, or `--describe-image 000000003156.jpg` to print every caption associated with a COCO filename so you can eyeball what textual grounding is available.

5. Control which caption is routed to Gemini with `--caption-index N` (default is 0) or add `--all-captions` to emit one prompt per caption. Each JSONL entry then records `caption_index` so you can trace back to the source description.

## Shared prompt helpers
- `image-prompt/processed_results/prompts.py` exports the system prompt, tone buckets, safe-text checker, and retry logic so both `emerge.py` and the caption pipeline stay in sync.
- Set `COCO_IMAGE_DIR` if you want `emerge.py` to run against another directory of COCO images instead of the default path baked into the script.
