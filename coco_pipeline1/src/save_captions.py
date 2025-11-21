#!/usr/bin/env python3
import json
import argparse
from collections import defaultdict

def save_captions(captions_file, output_file, limit=None):
    """
    Extract image captions from COCO dataset and save to a JSON file.
    
    Args:
        captions_file: Path to the COCO captions JSON file
        output_file: Path to save the extracted captions
        limit: Maximum number of images to extract captions for (None for all)
    """
    # Load the captions file
    print(f"Loading captions from {captions_file}...")
    with open(captions_file, 'r') as f:
        data = json.load(f)
    
    # Create a mapping from image_id to image filename
    image_id_to_filename = {}
    for image in data['images']:
        image_id_to_filename[image['id']] = image['file_name']
    
    # Group captions by image_id
    image_captions = defaultdict(list)
    for annotation in data['annotations']:
        image_id = annotation['image_id']
        caption = annotation['caption']
        image_captions[image_id].append(caption)
    
    # Create a structured output
    output_data = []
    for i, (image_id, captions) in enumerate(image_captions.items()):
        if limit is not None and i >= limit:
            break
            
        filename = image_id_to_filename.get(image_id, f"Unknown")
        output_data.append({
            "image_id": image_id,
            "file_name": filename,
            "captions": captions
        })
    
    # Save to output file
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nSaved captions for {len(output_data)} images to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract COCO image captions and save to a file")
    parser.add_argument("--captions", default="../annotations/captions_val2017.json", 
                        help="Path to COCO captions JSON file")
    parser.add_argument("--output", default="../extracted_captions.json",
                        help="Path to save the extracted captions")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of images to extract captions for (default: all)")
    args = parser.parse_args()
    
    save_captions(args.captions, args.output, args.limit)
