import cv2
import mediapipe as mp

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

    def detect_pose(self, rgb, bbox):
        x, y, w, h = bbox

        roi = rgb[y:y+h, x:x+w]
        if roi.size == 0:
            return {}  # ROI empty

        results = self.pose.process(roi)

        keypoints = {}
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark

            # Extract keypoints
            for name in ['LEFT_HIP', 'RIGHT_HIP', 'LEFT_SHOULDER', 'RIGHT_SHOULDER']:
                lm = landmarks[self.landmark_enum[name]]
                abs_x = int(x + lm.x * w)
                abs_y = int(y + lm.y * h)
                keypoints[name.lower()] = (abs_x, abs_y)

            # optional visualize
        if self.visualize:
            roi_vis = roi.copy()
            self.mp_draw.draw_landmarks(
                roi_vis,
                results.pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS
            )
            cv2.imshow("Pose Visualization", cv2.cvtColor(roi_vis, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        return keypoints
