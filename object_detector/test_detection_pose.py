import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# Load YOLOv8 pose model
model = YOLO('object_detector/models/yolov8n_pose.pt')

# RealSense setup
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

# Tracking configuration
TRACKER_CONFIG = "botsort.yaml"
selected_id = None

def click_handler(event, x, y, flags, param):
    global selected_id
    if event == cv2.EVENT_LBUTTONDOWN:
        results = param['results']
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if x1 <= x <= x2 and y1 <= y <= y2 and box.id is not None:
                selected_id = int(box.id[0])
                print(f"Selected person ID: {selected_id}")
                break

cv2.namedWindow('Person Pose Tracking')
cv2.setMouseCallback('Person Pose Tracking', click_handler, {'results': None})

try:
    while True:
        # Get RealSense frame
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue
        frame = np.asanyarray(color_frame.get_data())

        # Run pose estimation + tracking
        results = model.track(
            frame,
            conf=0.3,  # Lower confidence for better pose detection
            tracker=TRACKER_CONFIG,
            persist=True,
            verbose=False,
        )

        # Update click handler context
        cv2.setMouseCallback('Person Pose Tracking', click_handler, {'results': results})

        # Visualize results
        annotated_frame = results[0].plot(boxes=False)  # Disable auto-box plotting
        
        # Custom drawing for selected person
        for box, kpts in zip(results[0].boxes, results[0].keypoints):
            tid = int(box.id[0]) if box.id is not None else -1
            
            # Highlight selected person
            if tid == selected_id:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"ID {tid}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                
                # Draw all keypoints for selected person
                for x, y in kpts.xy[0]:
                    cv2.circle(annotated_frame, (int(x), int(y)), 5, (0, 255, 0), -1)

        cv2.imshow('Person Pose Tracking', annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()