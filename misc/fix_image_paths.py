import json
from pathlib import Path

root = Path("/n/home10/sliaw/cs2420-project/llava")
images_dir = root / "images"
lora_dir = root / "lora_data"

def fix_file(jsonl_path: Path):
    out_path = jsonl_path.with_suffix(".fixed.jsonl")
    missing = 0
    total = 0
    with jsonl_path.open() as fin, out_path.open("w") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            rel = Path(obj["image_path"]).name  # take basename
            full = images_dir / rel
            obj["image_path"] = str(full)
            total += 1
            if not full.exists():
                missing += 1
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    print(f"{jsonl_path.name}: wrote {out_path.name}, missing={missing}/{total}")

for jsonl in lora_dir.glob("*.jsonl"):
    fix_file(jsonl)