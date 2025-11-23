# CS2420 Project: Emergent Unsafe Prompts

This project uses Google's Gemini API to generate and evaluate "emergent unsafe" prompts - text that is safe when read alone but potentially unsafe when paired with specific images.

## Setup

1. **Create a `.env` file** in the project root with your Gemini API key:
   ```
   GEMINI_API_KEY=your_api_key_here
   ```

2. **Install dependencies**:
   ```bash
   pip install google-generativeai
   ```

## Running the Pipeline

The project includes a complete pipeline for generating and evaluating emergent unsafe prompts:

### 1. Generate Captions and Emergent Prompts

The `pipeline1.py` script implements a two-step process:
1. Generate captions for images using Gemini API (at two detail levels: brief and detailed)
2. Use those captions with the system prompt to generate emergent unsafe prompts

```bash
# Process 5 images (default)
python3 coco_pipeline1/src/pipeline1.py

# Process a specific number of images
python3 coco_pipeline1/src/pipeline1.py --limit 10

# Process all images
python3 coco_pipeline1/src/pipeline1.py --limit 0

# Use a different image directory
python3 coco_pipeline1/src/pipeline1.py --img-dir /path/to/images

# Specify a custom output file
python3 coco_pipeline1/src/pipeline1.py --output custom_results.json

# Generate prompts using a specific caption detail level
python3 coco_pipeline1/src/pipeline1.py --caption-level brief
python3 coco_pipeline1/src/pipeline1.py --caption-level detailed
python3 coco_pipeline1/src/pipeline1.py --caption-level all  # default: generate prompts for all caption levels

# Specify a tone for the emergent prompts
python3 coco_pipeline1/src/pipeline1.py --tone DIY
python3 coco_pipeline1/src/pipeline1.py --tone playful
python3 coco_pipeline1/src/pipeline1.py --tone challenge
python3 coco_pipeline1/src/pipeline1.py --tone creative
python3 coco_pipeline1/src/pipeline1.py --tone observational
python3 coco_pipeline1/src/pipeline1.py --tone scientific
```

### 2. Skip Caption Generation (Reuse Existing Captions)

If you've already generated captions, you can skip that step and just generate emergent prompts:

```bash
python3 coco_pipeline1/src/pipeline1.py --skip-captions --captions-file coco_pipeline1/output/captions.json
```

### 3. Evaluate Generated Prompts

The `evaluate_results.py` script evaluates if the generated prompts are truly "emergent unsafe":

```bash
# Evaluate all results
python3 coco_pipeline1/src/evaluate_results.py

# Evaluate a specific number of results
python3 coco_pipeline1/src/evaluate_results.py --limit 10

# Evaluate results from a specific file
python3 coco_pipeline1/src/evaluate_results.py --results-file path/to/results.json
```

## Output Files

The pipeline generates these output files with the following structure:

### 1. Captions File (`captions.json`)
```json
{
  "path/to/image.jpg": {
    "brief": "Short one-sentence caption",
    "detailed": "Medium-length caption with more details"
  }
}
```

### 2. Results File (`results.json`)
```json
[
  {
    "image_path": "path/to/image.jpg",
    "captions": {
      "brief": "Short one-sentence caption",
      "detailed": "Medium-length caption with more details"
    },
    "results": [
      {
        "caption_level": "brief",
        "emergent_unsafe_prompt": "Generated prompt text",
        "meta": { "severity": "high", "tone": "DIY", "relation": "stability" }
      },
      {
        "caption_level": "detailed",
        "emergent_unsafe_prompt": "Another generated prompt text",
        "meta": { "severity": "medium", "tone": "playful", "relation": "proximity" }
      }
    ]
  }
]
```

### 3. Evaluation File (`evaluation.json`)
```json
[
  {
    "image_path": "path/to/image.jpg",
    "captions": { /* same as in results.json */ },
    "evaluations": [
      {
        "caption_level": "brief",
        "emergent_prompt": "Generated prompt text",
        "meta": { /* same as in results.json */ },
        "evaluation": {
          "safe_alone": true,
          "unsafe_with_context": true,
          "implied_danger": "Description of the implied danger",
          "severity": "high",
          "is_emergent_unsafe": true,
          "explanation": "Explanation of why this is emergent unsafe"
        }
      },
      {
        "caption_level": "detailed",
        "emergent_prompt": "Another generated prompt text",
        "meta": { /* same as in results.json */ },
        "evaluation": {
          "safe_alone": true,
          "unsafe_with_context": true,
          "implied_danger": "Description of another implied danger",
          "severity": "medium",
          "is_emergent_unsafe": true,
          "explanation": "Explanation of why this is emergent unsafe"
        }
      }
    ]
  }
]
```

## Original COCO Caption Extraction

The `save_captions.py` script extracts captions from a COCO dataset JSON file:

```bash
python3 coco_pipeline1/src/save_captions.py --input annotations/captions_val2017.json --output extracted_captions.json
```

## Legacy Script

The `emerge.py` script in `image-prompt/processed_results/` is the original implementation that generates emergent unsafe prompts directly from images without the two-step process.

```bash
python3 image-prompt/processed_results/emerge.py
```
