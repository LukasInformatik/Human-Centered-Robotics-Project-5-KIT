### Download COCO dataset 2017

```bash
# 1. Create directories
mkdir -p coco_images/{train2017,val2017}
mkdir -p coco_annotations

# 2. Download images
wget http://images.cocodataset.org/zips/train2017.zip -O coco_images/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip   -O coco_images/val2017.zip
unzip coco_images/train2017.zip -d coco_images/train2017
unzip coco_images/val2017.zip   -d coco_images/val2017

# 3. Download annotations
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip \
     -O coco_annotations/annotations_trainval2017.zip
unzip coco_annotations/annotations_trainval2017.zip -d coco_annotations
```

if wget does not work, try `curl`:

```bash
curl -o coco_images/train2017.zip http://images.cocodataset.org/zips/train2017.zip
curl -o coco_images/val2017.zip http://images.cocodataset.org/zips/val2017.zip
curl -o coco_annotations\annotations_trainval2017.zip http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```