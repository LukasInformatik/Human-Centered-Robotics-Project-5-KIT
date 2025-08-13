import cv2
import mediapipe as mp
import time

class KeypointTracker:
    def __init__(self, visualize=True):
        self.visualize = visualize
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(static_image_mode=True,
                                    model_complexity=1,
                                    enable_segmentation=False,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)
        self.mp_draw = mp.solutions.drawing_utils
        self.landmark_enum = self.mp_pose.PoseLandmark
        
        # For tracking pose 1 timing per ID
        self.id_timing = {}  # Dictionary to store timing info per ID
        
    def detect_pose(self, rgb, bbox):
        x, y, w, h = bbox
        roi = rgb[y:y+h, x:x+w]
        
        if roi.size == 0:
            return {}  # ROI leer -> gib leeres Dict zurück
            
        results = self.pose.process(roi)
        keypoints = {}
        
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # Extrahiere relevante Keypoints
            for name in ['LEFT_HIP', 'RIGHT_HIP', 'LEFT_SHOULDER', 'RIGHT_SHOULDER']:
                lm = landmarks[self.landmark_enum[name]]
                abs_x = int(x + lm.x * w)
                abs_y = int(y + lm.y * h)
                keypoints[name.lower()] = (abs_x, abs_y)
            
            # Optional visualisieren
            if self.visualize:
                # Kopie der ROI für Visualisierung
                roi_vis = roi.copy()
                self.mp_draw.draw_landmarks(
                    roi_vis,
                    results.pose_landmarks,
                    self.mp_pose.POSE_CONNECTIONS
                )
                cv2.imshow("Pose Visualization", cv2.cvtColor(roi_vis, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)
        
        return keypoints
    
    def check_for_signal_pose(self, rgb, bbox, id):
        """
        Detects specific signal poses for a given ID:
        0: No pose detected
        1: One Hand is above the head for more than a second
        2: Both Hands are above the head
        
        Args:
            rgb: RGB image
            bbox: Bounding box (x, y, w, h)
            id: Unique identifier for tracking timing independently
            
        Returns:
            int: pose_id (0, 1, or 2)
        """
        x, y, w, h = bbox
        roi = rgb[y:y+h, x:x+w]
        
        # Initialize timing info for this ID if not exists
        if id not in self.id_timing:
            self.id_timing[id] = {
                'one_hand_up_start_time': None,
                'pose_1_detected': False
            }
        
        if roi.size == 0:
            # Reset timing for this ID if no valid ROI
            self.id_timing[id]['one_hand_up_start_time'] = None
            self.id_timing[id]['pose_1_detected'] = False
            return 0
            
        results = self.pose.process(roi)
        
        if not results.pose_landmarks:
            # Reset timing for this ID if no pose detected
            self.id_timing[id]['one_hand_up_start_time'] = None
            self.id_timing[id]['pose_1_detected'] = False
            return 0
            
        landmarks = results.pose_landmarks.landmark
        
        # Get hand and head keypoints
        try:
            # Head/nose position
            nose = landmarks[self.landmark_enum.NOSE]
            
            # Hand positions
            left_wrist = landmarks[self.landmark_enum.LEFT_WRIST]
            right_wrist = landmarks[self.landmark_enum.RIGHT_WRIST]
            
            # Convert to pixel coordinates within ROI
            nose_y = nose.y * h
            left_wrist_y = left_wrist.y * h
            right_wrist_y = right_wrist.y * h
            
            # Check if hands are above head (lower y values = higher in image)
            left_hand_up = left_wrist_y < nose_y
            right_hand_up = right_wrist_y < nose_y
            
            # Check for both hands up (pose 2)
            if left_hand_up and right_hand_up:
                # Reset pose 1 timing for this ID since we have pose 2
                self.id_timing[id]['one_hand_up_start_time'] = None
                self.id_timing[id]['pose_1_detected'] = False
                return 2
            
            # Check for one hand up (potential pose 1)
            elif left_hand_up or right_hand_up:
                current_time = time.time()
                
                # Start timing if this is the first detection for this ID
                if self.id_timing[id]['one_hand_up_start_time'] is None:
                    self.id_timing[id]['one_hand_up_start_time'] = current_time
                
                # Check if one hand has been up for more than 1 second for this ID
                elif current_time - self.id_timing[id]['one_hand_up_start_time'] > 1.0:
                    self.id_timing[id]['pose_1_detected'] = True
                    return 1
                
                # Still building up to 1 second
                return 0
            
            # No hands up - reset timing for this ID
            else:
                self.id_timing[id]['one_hand_up_start_time'] = None
                self.id_timing[id]['pose_1_detected'] = False
                return 0
                
        except (KeyError, AttributeError):
            # If we can't get the required keypoints, reset for this ID and return 0
            self.id_timing[id]['one_hand_up_start_time'] = None
            self.id_timing[id]['pose_1_detected'] = False
            return 0