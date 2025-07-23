import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO('object_detector/models/yolov8n_80_epochs.pt')

# Set up RealSense pipeline
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)

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

# List of tracker configs to test
trackers = [
    ("bytetrack.yaml", "ByteTrack"),
    ("botsort.yaml", "BoT-SORT"),
    ("strongsort.yaml", "StrongSORT")
]

try:
    for tracker_cfg, tracker_name in trackers:
        print(f"\n[INFO] Testing tracker: {tracker_name}")
        selected_id = None

        while True:
            # Get frame from RealSense
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())

            # Run detection + tracking (suppress logging with verbose=False)
            results = model.track(frame, conf=0.25, tracker=tracker_cfg, persist=True, verbose=False)

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
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                break

        # Reset window before next tracker
        cv2.destroyAllWindows()
        cv2.namedWindow('Tracked')
        cv2.setMouseCallback('Tracked', click_handler, {'boxes': []})

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
