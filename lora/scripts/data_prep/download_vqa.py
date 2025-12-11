import os
import json
import random
import urllib.request
import zipfile

# -------------------------------------------------
# Config
# -------------------------------------------------
OUTPUT_DIR = "VQA_subset_1602"
NUM_SAMPLES = 1602

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "images"), exist_ok=True)

# === OFFICIAL VQA v2 DOWNLOAD URLS (from visualqa.org) ===
# Questions + annotations (v2.0, train split), as ZIPs on cvmlp.s3.amazonaws.com
VQA_TRAIN_Q_ZIP = "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip"
VQA_TRAIN_A_ZIP = "https://cvmlp.s3.amazonaws.com/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip"

# COCO single-image URL template for train2014 images
COCO_IMG_URL = "http://images.cocodataset.org/train2014/COCO_train2014_{:012d}.jpg"
# If your network blocks http, you can try:
# COCO_IMG_URL = "https://images.cocodataset.org/train2014/COCO_train2014_{:012d}.jpg"

# -------------------------------------------------
# Helpers
# -------------------------------------------------
def download(url, out_path):
    if os.path.exists(out_path):
        print(f"[skip] Already exists: {out_path}")
        return
    print(f"[download] {url}")
    try:
        urllib.request.urlretrieve(url, out_path)
        print(f"[ok] Saved to {out_path}")
    except Exception as e:
        print(f"[ERROR] Failed to download {url}: {e}")
        raise

def unzip(zip_path, extract_to):
    print(f"[unzip] {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print("[ok] Unzipped into", extract_to)

# -------------------------------------------------
# 1) Download + unzip train questions/annotations
# -------------------------------------------------
train_q_zip_path = os.path.join(OUTPUT_DIR, "v2_Questions_Train_mscoco.zip")
train_a_zip_path = os.path.join(OUTPUT_DIR, "v2_Annotations_Train_mscoco.zip")

download(VQA_TRAIN_Q_ZIP, train_q_zip_path)
download(VQA_TRAIN_A_ZIP, train_a_zip_path)

unzip(train_q_zip_path, OUTPUT_DIR)
unzip(train_a_zip_path, OUTPUT_DIR)

# Inside those zips, the filenames are:
questions_path = os.path.join(
    OUTPUT_DIR, "v2_OpenEnded_mscoco_train2014_questions.json"
)
answers_path = os.path.join(
    OUTPUT_DIR, "v2_mscoco_train2014_annotations.json"
)

if not os.path.exists(questions_path):
    raise FileNotFoundError(f"Questions JSON not found at {questions_path}")
if not os.path.exists(answers_path):
    raise FileNotFoundError(f"Annotations JSON not found at {answers_path}")

print("[info] Loading questions and annotations JSON…")

with open(questions_path, "r") as f:
    questions = json.load(f)["questions"]

with open(answers_path, "r") as f:
    answers = json.load(f)["annotations"]

print(f"[info] Total questions:  {len(questions)}")
print(f"[info] Total annotations: {len(answers)}")

# -------------------------------------------------
# 2) Group answers by image_id
# -------------------------------------------------
answers_by_img = {}
for a in answers:
    answers_by_img.setdefault(a["image_id"], []).append(a)

all_image_ids = list(answers_by_img.keys())
print(f"[info] Unique image_ids in train annotations: {len(all_image_ids)}")

if NUM_SAMPLES > len(all_image_ids):
    raise ValueError(f"NUM_SAMPLES={NUM_SAMPLES} > total images={len(all_image_ids)}")

subset_ids = random.sample(all_image_ids, NUM_SAMPLES)
print(f"[info] Selected {NUM_SAMPLES} image_ids for subset")

# Pre-index questions by image_id to avoid O(N^2) scanning
questions_by_img = {}
for q in questions:
    questions_by_img.setdefault(q["image_id"], []).append(q)

# -------------------------------------------------
# 3) Build subset & download images
# -------------------------------------------------
subset = []

for i, img_id in enumerate(subset_ids, start=1):
    img_url = COCO_IMG_URL.format(img_id)
    img_save = os.path.join(OUTPUT_DIR, "images", f"{img_id}.jpg")

    # Download the image if we don't already have it
    if not os.path.exists(img_save):
        try:
            urllib.request.urlretrieve(img_url, img_save)
        except Exception as e:
            print(f"[WARN] Could not download image {img_id}: {e}")
            # Skip this sample entirely if the image isn't available
            continue

    qs = questions_by_img.get(img_id, [])
    ans = answers_by_img.get(img_id, [])

    if not qs or not ans:
        # If somehow we have no questions or answers, skip
        print(f"[WARN] No Q/A for image_id {img_id}, skipping")
        continue

    subset.append({
        "image_id": img_id,
        "image_file": f"images/{img_id}.jpg",
        "questions": qs,
        "answers": ans,
    })

    if i % 20 == 0 or i == len(subset_ids):
        print(f"[progress] Processed {i}/{len(subset_ids)} images")

# -------------------------------------------------
# 4) Save subset JSON
# -------------------------------------------------
subset_path = os.path.join(OUTPUT_DIR, "subset.json")
with open(subset_path, "w") as f:
    json.dump(subset, f, indent=2)

print("\n✅ Done!")
print(f"Final subset size (with successfully downloaded images): {len(subset)}")
print(f"Images dir:   {os.path.join(OUTPUT_DIR, 'images')}")
print(f"Subset JSON:  {subset_path}")
