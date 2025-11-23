# Emergent Unsafe Prompt Viewer

Interactive web viewer to explore generated image captions and emergent unsafe prompts.

## Features

- 🖼️ View images with their generated captions (brief, detailed, comprehensive)
- 📝 See all 3 emergent unsafe prompts (one per caption level)
- 🎨 Beautiful, responsive UI with gradient design
- 🔍 Dropdown to navigate between images
- 🏷️ Metadata badges showing tone, severity, and relation

## Running Locally (Without Docker)

1. Install dependencies:
```bash
cd viewer
uv pip install -r requirements.txt
```

2. Run the app (assumes your results are in `../output.jsonl` and images in `../coco_val_sample-SPECIFIC`):
```bash
python app.py
```

3. Open your browser to: http://localhost:5000

### Custom Paths

If your files are in different locations:
```bash
RESULTS_FILE=/path/to/output.jsonl IMAGE_DIR=/path/to/images python app.py
```

## Running with Docker

### Option 1: Docker Compose (Recommended)

1. Make sure your results file is at `../output.jsonl` and images at `../coco_val_sample-SPECIFIC`

2. Run:
```bash
cd viewer
docker-compose up --build
```

3. Open: http://localhost:5000

4. Stop with: `Ctrl+C` or `docker-compose down`

### Option 2: Docker Only

```bash
cd viewer
docker build -t prompt-viewer .
docker run -p 5000:5000 \
  -v $(pwd)/../output.jsonl:/data/output.jsonl:ro \
  -v $(pwd)/../coco_val_sample-SPECIFIC:/data/images:ro \
  -e RESULTS_FILE=/data/output.jsonl \
  -e IMAGE_DIR=/data/images \
  prompt-viewer
```

## File Structure

```
viewer/
├── app.py                  # Flask backend
├── templates/
│   └── index.html         # Frontend UI
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker image definition
├── docker-compose.yml    # Docker Compose config
└── README.md            # This file
```

## Expected Data Format

The viewer expects a JSONL file where each line is a JSON object with this structure:

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
        "tone": "DIY",
        "severity": "high",
        "relation": "stability"
      }
    },
    ...
  ]
}
```
