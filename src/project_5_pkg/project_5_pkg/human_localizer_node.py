import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo, Imu
from cv_bridge import CvBridge
import cv2
from message_filters import Subscriber, ApproximateTimeSynchronizer
import numpy as np
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

from project_5_pkg.scripts.keypoint_tracker import KeypointTracker
from project_5_pkg.scripts.human_localizer import HumanLocalizer
from project_5_pkg.scripts.human_tracker import HumanTracker

class HumanLocalizerPipeline(Node):
    def __init__(self):
        super().__init__('human_localizer_pipeline')
        self.bridge = CvBridge()

        # Init pipeline components
        self.kp_tracker = KeypointTracker(visualize=True)
        self.localizer = HumanLocalizer(self)
        self.human_tracker = HumanTracker()

        # Define QoS profiles
        camera_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Camera topics - using message_filters.Subscriber
        self.sub_color = Subscriber(
            self, 
            Image, 
            '/go2/d435i/color/image_raw',
            qos_profile=camera_qos  # Note: 'qos' instead of 'qos_profile'
        )
        self.sub_depth = Subscriber(
            self, 
            Image, 
            '/go2/d435i/aligned_depth_to_color/image_raw',
            qos_profile=camera_qos
        )
        self.sub_info = Subscriber(
            self, 
            CameraInfo, 
            '/go2/d435i/color/camera_info',
            qos_profile=camera_qos
        )
        
        self.ts = ApproximateTimeSynchronizer(
            [self.sub_color, self.sub_depth, self.sub_info],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.synced_callback)
        self.get_logger().info('HumanLocalizerPipeline initialized')

    def synced_callback(self, img_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        try:
            frame = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')


            # --- Hier yolo tracker auf 'rgb' anwenden ---
                # bounding box (x,y, width, height)
            if True:
                bb, _ = self.human_tracker.track(rgb)
            else:
                x = y = 0
                height, width = frame.shape[:2]
                bb = [x,y,width, height]

            # --- Hier mediapipe pose auf jeder BBox laufen lassen ---
            keypoints = self.kp_tracker.detect_pose(rgb, bb)

            # --- Hier Pixelkoordinaten mit Tiefendaten und (fx,fy,cx,cy) in 3D reprojizieren ---
            x3d, z3d = self.localizer.localize(keypoints, depth, info_msg)
            print(f"Durchschnittliche Position: x={x3d/1000:.2f} m, z={z3d/1000:.2f} m")


        except Exception as e:
            self.get_logger().error(f'Fehler im synchronized callback: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = HumanLocalizerPipeline()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
