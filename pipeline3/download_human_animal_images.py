#!/usr/bin/env python3
import os
import json
import requests
import argparse
from pycocotools.coco import COCO
import shutil
from tqdm import tqdm
import random

def download_image(url, save_path):
    """Download an image from URL to the specified path"""
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Download COCO images with humans and animals')
    parser.add_argument('--output_dir', type=str, default='coco_val_sample-SPECIFIC',
                        help='Output directory for downloaded images')
    parser.add_argument('--count', type=int, default=500,
                        help='Number of images to download')
    parser.add_argument('--data_dir', type=str, default='annotations',
                        help='Directory containing COCO annotations')
    parser.add_argument('--dataset', type=str, default='val2017',
                        help='Dataset to use (train2017 or val2017)')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize COCO API
    ann_file = os.path.join(args.data_dir, f'instances_{args.dataset}.json')
    if not os.path.exists(ann_file):
        print(f"Annotation file {ann_file} not found. Downloading...")
        os.makedirs(args.data_dir, exist_ok=True)
        url = f"http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
        # This would download and extract the annotations, but for now let's check if we have them
        print(f"Please download annotations from {url} and extract to {args.data_dir}")
        return
    
    coco = COCO(ann_file)
    
    # Category IDs for humans and animals in COCO
    human_cat_id = 1  # person category in COCO
    
    # Animal category IDs in COCO (partial list)
    animal_cat_ids = [
        16, 17, 18, 19, 20,  # bird, cat, dog, horse, sheep
        21, 22, 23, 24, 25,  # cow, elephant, bear, zebra, giraffe
        # Add more animal categories if needed
    ]
    
    # Get image IDs containing humans
    human_img_ids = coco.getImgIds(catIds=[human_cat_id])
    print(f"Found {len(human_img_ids)} images with humans")
    
    # Get image IDs containing animals
    animal_img_ids = coco.getImgIds(catIds=animal_cat_ids)
    print(f"Found {len(animal_img_ids)} images with animals")
    
    # Get image IDs containing both humans and animals
    human_animal_img_ids = list(set(human_img_ids) & set(animal_img_ids))
    print(f"Found {len(human_animal_img_ids)} images with both humans and animals")
    
    # If we don't have enough images with both, add some with just humans and some with just animals
    if len(human_animal_img_ids) < args.count:
        remaining = args.count - len(human_animal_img_ids)
        print(f"Need {remaining} more images to reach target count of {args.count}")
        
        # Add images with just humans
        human_only_ids = list(set(human_img_ids) - set(human_animal_img_ids))
        random.shuffle(human_only_ids)
        human_only_ids = human_only_ids[:remaining//2]
        
        # Add images with just animals
        animal_only_ids = list(set(animal_img_ids) - set(human_animal_img_ids))
        random.shuffle(animal_only_ids)
        animal_only_ids = animal_only_ids[:remaining - len(human_only_ids)]
        
        # Combine all image IDs
        selected_img_ids = human_animal_img_ids + human_only_ids + animal_only_ids
    else:
        # If we have enough images with both humans and animals, randomly select the required number
        random.shuffle(human_animal_img_ids)
        selected_img_ids = human_animal_img_ids[:args.count]
    
    print(f"Selected {len(selected_img_ids)} images for download")
    
    # Get image info for selected IDs
    selected_imgs = coco.loadImgs(selected_img_ids)
    
    # Base URL for COCO images
    base_url = f"http://images.cocodataset.org/{args.dataset}/"
    
    # Download images
    success_count = 0
    for img in tqdm(selected_imgs, desc="Downloading images"):
        file_name = img['file_name']
        img_url = base_url + file_name
        save_path = os.path.join(args.output_dir, file_name)
        
        if download_image(img_url, save_path):
            success_count += 1
    
    print(f"Successfully downloaded {success_count}/{len(selected_imgs)} images to {args.output_dir}")

if __name__ == "__main__":
    main()
