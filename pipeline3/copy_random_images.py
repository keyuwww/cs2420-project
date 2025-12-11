#!/usr/bin/env python3
import os
import shutil
import random
import argparse
from glob import glob

def main():
    parser = argparse.ArgumentParser(description='Copy random images to SPECIFIC folder')
    parser.add_argument('--source_dir', type=str, required=True,
                        help='Source directory containing images')
    parser.add_argument('--output_dir', type=str, default='coco_val_sample-SPECIFIC',
                        help='Output directory for copied images')
    parser.add_argument('--count', type=int, default=500,
                        help='Number of images to copy')
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get list of image files in source directory
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png']:
        image_files.extend(glob(os.path.join(args.source_dir, '**', ext), recursive=True))
    
    print(f"Found {len(image_files)} images in source directory")
    
    # If we don't have enough images, use all of them
    if len(image_files) <= args.count:
        selected_files = image_files
    else:
        # Randomly select the required number of images
        selected_files = random.sample(image_files, args.count)
    
    print(f"Selected {len(selected_files)} images for copying")
    
    # Copy images to output directory
    success_count = 0
    for src_path in selected_files:
        file_name = os.path.basename(src_path)
        dst_path = os.path.join(args.output_dir, file_name)
        
        try:
            shutil.copy2(src_path, dst_path)
            success_count += 1
            print(f"Copied {success_count}/{len(selected_files)}: {file_name}")
        except Exception as e:
            print(f"Error copying {src_path}: {e}")
    
    print(f"Successfully copied {success_count}/{len(selected_files)} images to {args.output_dir}")

if __name__ == "__main__":
    main()
