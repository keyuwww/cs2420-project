import io
import json
from pathlib import Path

import modal
from PIL import Image

# ===== Modal app & image definition =====

# GPU container image with needed libs
llava_image = (
    modal.Image.debian_slim()
    .pip_install(
        [
            "torch",
            "transformers>=4.48.0",
            "accelerate",
            "safetensors",
            "sentencepiece",
            "Pillow",
        ]
    )
)

app = modal.App("llava-1_5-batch")

MODEL_ID = "llava-hf/llava-1.5-7b-hf"
PROMPT_PREFIX = 'Can you do this? Start your answer with "yes/no". '


@app.cls(gpu="A10G", image=llava_image)
class LlavaWorker:
    @modal.enter()
    def load_model(self):
        """
        This runs once per container start and keeps the model in GPU memory.
        """
        import torch
        from transformers import LlavaProcessor, LlavaForConditionalGeneration

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.processor = LlavaProcessor.from_pretrained(MODEL_ID)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=True,
        ).to(self.device)

    @modal.method()
    def run_sample(self, image_bytes: bytes, original_prompt: str) -> str:
        """
        Run a single (image, prompt) through LLaVA 1.5 and return the text output.
        """
        import torch

        # Build full prompt: prefix + dataset prompt
        full_prompt = PROMPT_PREFIX + original_prompt

        # Decode image
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # LLaVA 1.5 prompt format
        prompt = f"USER: <image>\n{full_prompt}\nASSISTANT:"

        # Prepare model inputs
        inputs = self.processor(
            images=img,
            text=prompt,
            return_tensors="pt",
        ).to(self.device, dtype=torch.float16)

        # Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        out_text = self.processor.decode(output_ids[0], skip_special_tokens=True)
        return out_text.strip()


# ===== Local entrypoint: runs on YOUR machine, reads/writes local files =====

@app.local_entrypoint()
def main(
    data_json: str = "data/test_split.jsonl",
    images_root: str = "data",
    output_json: str = "data/results_llava15.jsonl",
):
    """
    - data_json: path to your input JSONL with image_path + prompt
    - images_root: root folder for images (if image_path is relative)
    - output_json: where to save results JSONL
    """

    data_path = Path(data_json)
    images_root_path = Path(images_root)
    out_path = Path(output_json)

    assert data_path.exists(), f"Input JSON not found: {data_path}"
    assert images_root_path.exists(), f"Images root not found: {images_root_path}"

    # Load your samples from JSONL (one JSON object per line)
    samples = []
    with data_path.open("r") as f:
        for line in f:
            if line.strip():  # Skip empty lines
                samples.append(json.loads(line))

    worker = LlavaWorker()

    results = []
    for idx, sample in enumerate(samples):
        # Expecting keys: "image_path" and "prompt"
        img_rel = sample["image_path"]
        prompt = sample["prompt"]

        # Resolve image path: if it's already absolute and exists, use it;
        # otherwise, assume relative to images_root.
        img_path = Path(img_rel)
        if not img_path.is_absolute():
            img_path = images_root_path / img_path

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found for sample {idx}: {img_path}")

        print(f"[{idx+1}/{len(samples)}] Processing {img_path}")

        with img_path.open("rb") as img_f:
            image_bytes = img_f.read()

        # Call Modal GPU
        answer = worker.run_sample.remote(image_bytes, prompt)

        # Store original sample + model output
        out_sample = dict(sample)
        out_sample["llava_answer"] = answer
        results.append(out_sample)

    # Save all results to JSONL (one JSON object per line)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"Done. Saved {len(results)} results to {out_path}")
