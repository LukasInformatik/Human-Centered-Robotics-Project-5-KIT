import os
import json
import random
from tqdm import tqdm
from PIL import Image
from collections import defaultdict
# Paths
ANNOT_JSON = 'coco_annotations/annotations/instances_train2017.json'
# ANNOT_JSON = 'coco_annotations/annotations/instances_val2017.json'
IMG_DIR     = 'coco_images/train2017'
# IMG_DIR     = 'coco_images/val2017'
OUT_IMG_DIR = 'yolo_coco_person/images/train'
# OUT_IMG_DIR = 'yolo_coco_person/images/val'
OUT_LBL_DIR = 'yolo_coco_person/labels/train'
# OUT_LBL_DIR = 'yolo_coco_person/labels/val'

# Configurable
no_person_ratio = 0.2  # e.g., 0.2 = 20% of the final dataset are no-person images

# Create output dirs
os.makedirs(OUT_IMG_DIR, exist_ok=True)
os.makedirs(OUT_LBL_DIR, exist_ok=True)

# Load COCO annotations
with open(ANNOT_JSON) as f:
    coco = json.load(f)

# Build image ID → filename map
images = {img['id']: img['file_name'] for img in coco['images']}

# Only keep annotations of category_id == 1 (person)
person_anns = [ann for ann in coco['annotations'] if ann['category_id'] == 1]

# Group person annotations by image
anns_per_image = defaultdict(list)
for ann in person_anns:
    anns_per_image[ann['image_id']].append(ann)

# Identify all image IDs with and without people
all_image_ids = set(images.keys())
person_image_ids = set(anns_per_image.keys())
no_person_image_ids = list(all_image_ids - person_image_ids)

# How many empty (no-person) images to include
num_person_imgs = len(person_image_ids)
num_no_person = int(no_person_ratio * num_person_imgs)

# Randomly sample non-person images
sampled_no_person_ids = random.sample(no_person_image_ids, min(num_no_person, len(no_person_image_ids)))

print(f"[INFO] Total person images: {num_person_imgs}")
print(f"[INFO] Adding {len(sampled_no_person_ids)} no-person images")

# Process person images
for img_id, anns in tqdm(anns_per_image.items(), desc="Processing person images"):
    fname = images[img_id]
    src_path = os.path.join(IMG_DIR, fname)
    dst_img  = os.path.join(OUT_IMG_DIR, fname)
    dst_lbl  = os.path.join(OUT_LBL_DIR, fname.replace('.jpg', '.txt'))

    # Copy image
    if not os.path.exists(dst_img):
        os.link(src_path, dst_img)

    # Read image size
    w, h = Image.open(src_path).size

    # Write label file
    with open(dst_lbl, 'w') as fout:
        for ann in anns:
            x, y, bw, bh = ann['bbox']
            x_c = x + bw / 2
            y_c = y + bh / 2

            x_c /= w;  bw /= w
            y_c /= h;  bh /= h

            fout.write(f"0 {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")

# Process no-person images
for img_id in tqdm(sampled_no_person_ids, desc="Processing no-person images"):
    fname = images[img_id]
    src_path = os.path.join(IMG_DIR, fname)
    dst_img  = os.path.join(OUT_IMG_DIR, fname)
    dst_lbl  = os.path.join(OUT_LBL_DIR, fname.replace('.jpg', '.txt'))

    # Copy image and create empty label file
    if not os.path.exists(dst_img):
        os.link(src_path, dst_img)
    with open(dst_lbl, 'w') as fout:
        pass  # Empty file = no annotations
