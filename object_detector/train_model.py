from ultralytics import YOLO

def main():
    # Load a small YOLOv8 model
    model = YOLO('runs/detect/yolov8_person_coco10/weights/best.pt')

    # Train on your new coco-person dataset
    model.train(
        data='person_coco.yaml',
        imgsz=640,
        epochs=50,
        batch=-1,
        patience=30,
        name='yolov8_person_coco'
    )

if __name__ == '__main__':
    main()
