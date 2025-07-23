import cv2
import numpy as np
from ultralytics import YOLO

# Initialize YOLO model
model = YOLO('models/yolov8n_80_epochs.pt')
selected_id = None  # Track selected person ID

# Mouse callback for selecting boxes
def select_box(event, x, y, flags, param):
    global selected_id
    if event == cv2.EVENT_LBUTTONDOWN:
        boxes = param['boxes']
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if x1 <= x <= x2 and y1 <= y <= y2 and box.id is not None:
                selected_id = int(box.id[0])
                print(f"Selected ID: {selected_id}")
                break

# Setup video capture
video_path = 'video_realsense.avi' 
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError("Cannot open video file")

# Get video properties for output
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('annotated_output.avi', fourcc, fps, (width, height))

# Create window and set callback
cv2.namedWindow('Person Tracking')
cv2.setMouseCallback('Person Tracking', select_box, {'boxes': []})

try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO tracking
        results = model.track(
            frame, 
            conf=0.25, 
            tracker="bytetrack.yaml", 
            persist=True,
            verbose=False  # Disable console output
        )
        
        # Update callback data
        if results[0].boxes.id is not None:
            cv2.setMouseCallback('Person Tracking', select_box, {'boxes': results[0].boxes})
        
        # Draw bounding boxes
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            tid = int(box.id[0]) if box.id is not None else -1
            
            # Highlight selected person
            if tid == selected_id:
                color = (0, 0, 255)  # Red
                thickness = 3
            else:
                color = (0, 255, 0)  # Green
                thickness = 2
                
            # Draw box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
            cv2.putText(frame, f"ID {tid}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Display and save
        cv2.imshow('Person Tracking', frame)
        out.write(frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()