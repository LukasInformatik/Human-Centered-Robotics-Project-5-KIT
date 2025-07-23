import os

import os
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# Load your trained YOLOv8 model
model = YOLO('models/yolov8n_80_epochs.pt')

# Initialize RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
pipeline.start(config)
print("[INFO] RealSense streaming started")

try:
    while True:
        # Get latest color frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert to numpy array
        frame = np.asanyarray(color_frame.get_data())

        # Run YOLOv8 inference
        results = model.predict(frame, stream=False, imgsz=640, conf=0.25)

        # Draw bounding boxes
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = box.conf[0]
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"person {conf:.2f}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        # Display
        cv2.imshow('RealSense YOLOv8 Detection', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
