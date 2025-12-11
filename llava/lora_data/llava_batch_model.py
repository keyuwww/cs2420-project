import io
import json
from pathlib import Path

import modal
from PIL import Image


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
        import torch

        full_prompt = PROMPT_PREFIX + original_prompt

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        prompt = f"USER: <image>\n{full_prompt}\nASSISTANT:"

        inputs = self.processor(
            images=img,
            text=prompt,
            return_tensors="pt",
        ).to(self.device, dtype=torch.float16)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        out_text = self.processor.decode(output_ids[0], skip_special_tokens=True)
        return out_text.strip()



@app.local_entrypoint()
def main(
    data_json: str = "data/test_split.jsonl",
    images_root: str = "data",
    output_json: str = "data/results_llava15.jsonl",
):

    data_path = Path(data_json)
    images_root_path = Path(images_root)
    out_path = Path(output_json)

    assert data_path.exists(), f"Input JSON not found: {data_path}"
    assert images_root_path.exists(), f"Images root not found: {images_root_path}"

    samples = []
    with data_path.open("r") as f:
        for line in f:
            if line.strip():  # Skip empty lines
                samples.append(json.loads(line))

    worker = LlavaWorker()

    results = []
    for idx, sample in enumerate(samples):
        img_rel = sample["image_path"]
        prompt = sample["prompt"]

        img_path = Path(img_rel)
        if not img_path.is_absolute():
            img_path = images_root_path / img_path

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found for sample {idx}: {img_path}")

        print(f"[{idx+1}/{len(samples)}] Processing {img_path}")

        with img_path.open("rb") as img_f:
            image_bytes = img_f.read()

        answer = worker.run_sample.remote(image_bytes, prompt)

        out_sample = dict(sample)
        out_sample["llava_answer"] = answer
        results.append(out_sample)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    print(f"Done. Saved {len(results)} results to {out_path}")
