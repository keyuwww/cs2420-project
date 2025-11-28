#!/usr/bin/env python3
import os
import json
import requests
import argparse
from pycocotools.coco import COCO
import shutil
import random
import concurrent.futures
from tqdm import tqdm

def download_image(img_info):
    """Download an image from URL to the specified path"""
    img_id, img_url, save_path = img_info
    try:
        response = requests.get(img_url, stream=True)
        response.raise_for_status()
        with open(save_path, 'wb') as out_file:
            shutil.copyfileobj(response.raw, out_file)
        return True, img_id
    except Exception as e:
        print(f"Error downloading {img_url}: {e}")
        return False, img_id

def main():
    parser = argparse.ArgumentParser(description='Download COCO images')
    parser.add_argument('--output_dir', type=str, default='coco_val_sample-SPECIFIC',
                        help='Output directory for downloaded images')
    parser.add_argument('--count', type=int, default=500,
                        help='Number of images to download')
    parser.add_argument('--data_dir', type=str, default='annotations',
                        help='Directory containing COCO annotations')
    parser.add_argument('--dataset', type=str, default='val2017',
                        help='Dataset to use (train2017 or val2017)')
    parser.add_argument('--workers', type=int, default=10,
                        help='Number of parallel download workers')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Count existing images
    existing_images = [f for f in os.listdir(args.output_dir) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Found {len(existing_images)} existing images in {args.output_dir}")
    
    # Calculate how many more images we need
    remaining_count = max(0, args.count - len(existing_images))
    if remaining_count == 0:
        print(f"Already have {len(existing_images)} images, which meets or exceeds the target count of {args.count}")
        return
    
    print(f"Need to download {remaining_count} more images")
    
    # Initialize COCO API
    ann_file = os.path.join(args.data_dir, f'instances_{args.dataset}.json')
    if not os.path.exists(ann_file):
        print(f"Annotation file {ann_file} not found.")
        return
    
    coco = COCO(ann_file)
    
    # Get all image IDs
    img_ids = coco.getImgIds()
    print(f"Found {len(img_ids)} total images in COCO {args.dataset}")
    
    # Shuffle image IDs to get a random selection
    random.shuffle(img_ids)
    
    # Select the required number of images
    selected_img_ids = img_ids[:min(remaining_count * 2, len(img_ids))]  # Get more than needed in case some downloads fail
    
    # Get image info for selected IDs
    selected_imgs = coco.loadImgs(selected_img_ids)
    
    # Base URL for COCO images
    base_url = f"http://images.cocodataset.org/{args.dataset}/"
    
    # Prepare download tasks
    download_tasks = []
    for img in selected_imgs:
        file_name = img['file_name']
        img_url = base_url + file_name
        save_path = os.path.join(args.output_dir, file_name)
        
        # Skip if file already exists
        if os.path.exists(save_path):
            continue
            
        download_tasks.append((img['id'], img_url, save_path))
    
    # Limit tasks to the number we need
    download_tasks = download_tasks[:remaining_count]
    
    if not download_tasks:
        print("No new images to download")
        return
    
    print(f"Downloading {len(download_tasks)} images with {args.workers} workers")
    
    # Download images in parallel
    success_count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download_image, task) for task in download_tasks]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Downloading"):
            success, _ = future.result()
            if success:
                success_count += 1
    
    print(f"Successfully downloaded {success_count}/{len(download_tasks)} new images")
    print(f"Total images in directory: {len(existing_images) + success_count}")

if __name__ == "__main__":
    main()
