import os

import os
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO('runs/detect/yolov8_person_coco10/weights/best.pt')

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
pipeline.start(config)
print("[INFO] RealSense streaming started")

# Global selected ID variable
selected_id = None

# Click handler for person selection
def click_handler(event, x, y, flags, param):
    global selected_id
    if event == cv2.EVENT_LBUTTONDOWN:
        boxes = param['boxes']
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if x1 <= x <= x2 and y1 <= y <= y2:
                selected_id = int(box.id[0])
                print(f"[INFO] Selected track ID: {selected_id}")
                break

# Set up window and callback
cv2.namedWindow('Tracked')
cv2.setMouseCallback('Tracked', click_handler, {'boxes': []})

try:
    while True:
        # Get frame from RealSense
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())

        # Run detection + tracking
        results = model.track(frame, conf=0.25, tracker="bytetrack.yaml", persist=True)

        for r in results:
            boxes = r.boxes
            # Update boxes for callback context
            cv2.setMouseCallback('Tracked', click_handler, {'boxes': boxes})

            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                tid = int(box.id[0]) if box.id is not None else -1
                color = (0, 0, 255) if tid == selected_id else (0, 255, 0)
                label = f"ID {tid}" if tid >= 0 else "Untracked"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Display frame
        cv2.imshow('Tracked', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
