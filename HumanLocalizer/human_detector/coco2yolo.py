import os
import json
import random
from tqdm import tqdm
from collections import defaultdict
from shutil import copyfile  # Better for cross-platform file copying

# Paths
DATA_TYPE = "val"  # "train" or "val"
ANNOT_JSON = f'coco_annotations/annotations/person_keypoints_{DATA_TYPE}2017.json'
INSTANCES_JSON = f'coco_annotations/annotations/instances_{DATA_TYPE}2017.json'  # Fixed path
IMG_DIR = f'coco_images/{DATA_TYPE}2017'
OUT_IMG_DIR = f'yolo_coco_person/images/{DATA_TYPE}'
OUT_LBL_DIR = f'yolo_coco_person/labels/{DATA_TYPE}'

# Configurable
no_person_ratio = 0.2  # Percentage of no-person images
min_keypoints = 5  # Minimum visible keypoints to keep annotation


# Create output dirs
os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LBL_DIR, exist_ok=True)

# Load COCO keypoints annotations
with open(ANNOT_JSON) as f:
    coco_kpts = json.load(f)

# Load COCO instance annotations (for no-person images)
with open(INSTANCES_JSON) as f:  # Using the correct variable name
    coco_instances = json.load(f)

# Build mappings from keypoints data
image_id_to_file = {img['id']: img['file_name'] for img in coco_kpts['images']}
image_id_to_size = {img['id']: (img['width'], img['height']) for img in coco_kpts['images']}

# Group keypoint annotations by image
kpts_per_image = defaultdict(list)
for ann in coco_kpts['annotations']:
    if ann['num_keypoints'] >= min_keypoints:  # Filter low-quality annotations
        kpts_per_image[ann['image_id']].append(ann)

# Get all image IDs from instances (for no-person images)
all_instance_images = {img['id']: img['file_name'] for img in coco_instances['images']}
person_image_ids = set(kpts_per_image.keys())

# Get no-person images (must exist in instances and not in keypoints)
no_person_image_ids = [img_id for img_id in all_instance_images
                       if img_id not in person_image_ids and img_id in image_id_to_size]

# Sample no-person images
num_no_person = int(no_person_ratio * len(person_image_ids))
sampled_no_person = random.sample(no_person_image_ids, min(num_no_person, len(no_person_image_ids)))

print(f"Person images: {len(person_image_ids)}")
print(f"No-person images: {len(sampled_no_person)}")

# Process images with keypoints
for img_id, anns in tqdm(kpts_per_image.items(), desc="Processing pose annotations"):
    fname = image_id_to_file[img_id]
    src_path = os.path.join(IMG_DIR, fname)
    dst_img = os.path.join(OUT_IMG_DIR, fname)
    dst_lbl = os.path.join(OUT_LBL_DIR, fname.replace('.jpg', '.txt'))

    # Copy image (using copyfile instead of os.link for compatibility)
    if not os.path.exists(dst_img):
        copyfile(src_path, dst_img)

    # Get image dimensions
    w, h = image_id_to_size[img_id]

    # Write YOLO pose labels
    with open(dst_lbl, 'w') as f:
        for ann in anns:
            # Bounding box (COCO to YOLO format)
            x, y, bw, bh = ann['bbox']
            x_center = (x + bw / 2) / w
            y_center = (y + bh / 2) / h
            bw_norm = bw / w
            bh_norm = bh / h

            # Keypoints (17 points, 3 values each: x, y, visibility)
            kpts = ann['keypoints']
            yolo_kpts = []
            for i in range(0, 51, 3):  # 17*3=51
                px = kpts[i] / w if kpts[i] > 0 else 0
                py = kpts[i + 1] / h if kpts[i + 1] > 0 else 0
                vis = min(2, kpts[i + 2])  # COCO: 0=missing, 1=occluded, 2=visible
                yolo_kpts.extend([px, py, vis])

            # Write line: class, bbox, keypoints
            line = [0, x_center, y_center, bw_norm, bh_norm] + yolo_kpts
            f.write(' '.join(map(str, line)) + '\n')

# Process no-person images
for img_id in tqdm(sampled_no_person, desc="Processing no-person images"):
    if img_id not in all_instance_images:  # Skip if image info not found
        continue

    fname = all_instance_images[img_id]
    src_path = os.path.join(IMG_DIR, fname)
    dst_img = os.path.join(OUT_IMG_DIR, fname)
    dst_lbl = os.path.join(OUT_LBL_DIR, fname.replace('.jpg', '.txt'))

    if not os.path.exists(dst_img):
        copyfile(src_path, dst_img)

    # Create empty label file
    with open(dst_lbl, 'w') as f:
        pass  # Empty file