import os
import json
import numpy as np
import cv2
from tqdm import tqdm
import argparse

def collect_labels(json_folder):
    """
    Collect all unique labels from LabelMe JSON files.

    Args:
        json_folder (str): Path to the folder containing LabelMe JSON files.

    Returns:
        dict: Mapping from label names to unique integer IDs.
    """
    labels = set()
    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
    print("Collecting labels from JSON files...")
    for json_file in tqdm(json_files, desc="Scanning JSON files"):
        json_path = os.path.join(json_folder, json_file)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for shape in data.get('shapes', []):
                    label = shape.get('label')
                    if label:
                        labels.add(label)
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            continue

    if not labels:
        raise ValueError("No labels found in the JSON files.")

    sorted_labels = sorted(labels)
    label_to_id = {label: idx + 1 for idx, label in enumerate(sorted_labels)}  # Start labeling from 1
    print("\nLabel to ID mapping:")
    for label, idx in label_to_id.items():
        print(f"  {label}: {idx}")
    
    return label_to_id

def create_mask(json_file, image_path, label_to_id, target_size=(256, 256)):
    """
    Create a mask image from a LabelMe JSON file.

    Args:
        json_file (str): Path to the LabelMe JSON file.
        image_path (str): Path to the corresponding image file.
        label_to_id (dict): Mapping from label names to integer IDs.
        target_size (tuple): Desired size for the mask image.

    Returns:
        np.ndarray: Mask image with integer labels.
    """
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error reading {json_file}: {e}")
        return None

    # Load the image to get dimensions if target_size is not provided
    if target_size:
        width, height = target_size
    else:
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Failed to load image {image_path}. Skipping mask creation.")
            return None
        height, width = image.shape[:2]

    # Initialize mask
    mask = np.zeros((height, width), dtype=np.uint8)

    # Process each shape
    for shape in data.get('shapes', []):
        label = shape.get('label')
        points = shape.get('points')
        shape_type = shape.get('shape_type', 'polygon')

        if not label or not points:
            continue

        label_id = label_to_id.get(label)
        if not label_id:
            continue  # Skip labels not in the mapping

        # Convert points to integer coordinates
        polygon = np.array(points, dtype=np.int32)
        polygon = polygon.reshape((-1, 1, 2))  # Required shape for cv2.fillPoly

        if shape_type == 'polygon':
            cv2.fillPoly(mask, [polygon], label_id)
        elif shape_type == 'rectangle':
            # Rectangle is defined by two points: top-left and bottom-right
            if len(points) >= 2:
                top_left = tuple(map(int, points[0]))
                bottom_right = tuple(map(int, points[1]))
                cv2.rectangle(mask, top_left, bottom_right, label_id, thickness=-1)
        elif shape_type == 'circle':
            # Circle is defined by center and a point on the circumference
            if len(points) >= 2:
                center = tuple(map(int, points[0]))
                radius = int(np.linalg.norm(np.array(points[1]) - np.array(points[0])))
                cv2.circle(mask, center, radius, label_id, thickness=-1)
        else:
            # Default to polygon if shape_type is unrecognized
            cv2.fillPoly(mask, [polygon], label_id)

    return mask

def save_mask(mask, save_path):
    """
    Save the mask image as a PNG file.

    Args:
        mask (np.ndarray): Mask image with integer labels.
        save_path (str): Path to save the mask PNG file.
    """
    try:
        cv2.imwrite(save_path, mask)
    except Exception as e:
        print(f"Error saving mask {save_path}: {e}")

def main(json_folder, image_folder, mask_folder, target_size=(256, 256)):
    """
    Main function to convert LabelMe JSON annotations to mask PNG images.

    Args:
        json_folder (str): Path to the folder containing LabelMe JSON files.
        image_folder (str): Path to the folder containing corresponding image files.
        mask_folder (str): Path to the folder where mask PNGs will be saved.
        target_size (tuple): Desired size for the mask images.
    """
    if not os.path.exists(mask_folder):
        os.makedirs(mask_folder)
        print(f"Created mask directory at: {mask_folder}")

    # Step 1: Collect all unique labels and assign IDs
    label_to_id = collect_labels(json_folder)

    # Save label mapping
    label_mapping_path = os.path.join(mask_folder, 'label_mapping.json')
    try:
        with open(label_mapping_path, 'w', encoding='utf-8') as f:
            json.dump(label_to_id, f, indent=4)
        print(f"\nSaved label mapping to {label_mapping_path}")
    except Exception as e:
        print(f"Error saving label mapping: {e}")

    # Step 2: Process each JSON file to create masks
    json_files = [f for f in os.listdir(json_folder) if f.endswith('.json')]
    print("\nConverting JSON annotations to PNG masks...")
    for json_file in tqdm(json_files, desc="Processing JSON files"):
        json_path = os.path.join(json_folder, json_file)
        
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {json_file}: {e}")
            continue

        # Get image filename from JSON
        image_path_in_json = data.get('imagePath')
        if not image_path_in_json:
            print(f"Warning: 'imagePath' not found in {json_file}. Skipping.")
            continue

        # Extract the image filename
        image_filename = os.path.basename(image_path_in_json)
        image_path = os.path.join(image_folder, image_filename)

        if not os.path.exists(image_path):
            print(f"Warning: Image file {image_path} does not exist. Skipping {json_file}.")
            continue

        # Create mask
        mask = create_mask(json_path, image_path, label_to_id, target_size)

        if mask is None:
            print(f"Warning: Failed to create mask for {json_file}.")
            continue

        # Save mask with the same base name as the image
        mask_filename = os.path.splitext(image_filename)[0] + '.png'
        mask_save_path = os.path.join(mask_folder, mask_filename)
        save_mask(mask, mask_save_path)

    print("\nMask conversion completed successfully!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert LabelMe JSON annotations to mask PNG images.")
    parser.add_argument("json_folder", help="Path to the folder containing LabelMe JSON files.")
    parser.add_argument("image_folder", help="Path to the folder containing corresponding image files.")
    parser.add_argument("mask_folder", help="Path to the folder where mask PNGs will be saved.")
    parser.add_argument(
        "--target_size",
        nargs=2,
        type=int,
        default=(256, 256),
        help="Desired size for the mask images (width height). Default is 256 256."
    )

    args = parser.parse_args()

    main(args.json_folder, args.image_folder, args.mask_folder, tuple(args.target_size))
