import os
import json
from tqdm import tqdm

# Paths
# ANNOT_JSON = 'coco_annotations/annotations/instances_train2017.json'
ANNOT_JSON = 'coco_annotations/annotations/instances_val2017.json'
# IMG_DIR     = 'coco_images/train2017'
IMG_DIR     = 'coco_images/val2017'
# OUT_IMG_DIR = 'yolo_coco_person/images/train'
OUT_IMG_DIR = 'yolo_coco_person/images/val'
# OUT_LBL_DIR = 'yolo_coco_person/labels/train'
OUT_LBL_DIR = 'yolo_coco_person/labels/val'

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

# Group annotations by image
from collections import defaultdict
anns_per_image = defaultdict(list)
for ann in person_anns:
    anns_per_image[ann['image_id']].append(ann)

# Conversion loop
for img_id, anns in tqdm(anns_per_image.items()):
    fname = images[img_id]
    src_path = os.path.join(IMG_DIR, fname)
    dst_img  = os.path.join(OUT_IMG_DIR, fname)
    dst_lbl  = os.path.join(OUT_LBL_DIR, fname.replace('.jpg', '.txt'))

    # Copy image
    if not os.path.exists(dst_img):
        os.link(src_path, dst_img)

    # Read image size
    from PIL import Image
    w, h = Image.open(src_path).size

    # Write label file
    with open(dst_lbl, 'w') as fout:
        for ann in anns:
            # COCO bbox: [x, y, width, height]
            x, y, bw, bh = ann['bbox']
            x_c = x + bw / 2
            y_c = y + bh / 2

            # Normalize
            x_c /= w;  bw /= w
            y_c /= h;  bh /= h

            # class_id 0 for “person”
            fout.write(f"0 {x_c:.6f} {y_c:.6f} {bw:.6f} {bh:.6f}\n")
