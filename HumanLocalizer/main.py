import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import cv2
from message_filters import Subscriber, ApproximateTimeSynchronizer

from keypoint_tracker import KeypointTracker
from human_localizer import HumanLocalizer
from human_tracker import HumanTracker

class HumanLocalizerPipeline(Node):
    def __init__(self):
        super().__init__('human_localizer_pipeline')
        self.bridge = CvBridge()

        #Init pipeline components
        self.kp_tracker = KeypointTracker(visualize=False)
        self.localizer = HumanLocalizer()
        self.human_tracker = HumanTracker()

        # Camera topics
        self.sub_color = Subscriber(self, Image, '/go2/D453i/color/image_raw')
        self.sub_depth = Subscriber(self, Image, '/go2/D453i/depth/image_rect_raw')
        self.sub_info  = Subscriber(self, CameraInfo, '/go2/D453i/color/camera_info')

        # ApproximateTimeSynchronizer
        self.ts = ApproximateTimeSynchronizer(
            [self.sub_color, self.sub_depth, self.sub_info],
            queue_size=10,
            slop=0.05,  # 50ms tolerance
        )
        self.ts.registerCallback(self.synced_callback)
        self.get_logger().info('HumanLocalizerPipeline initialized with synchronized topics')


    def synced_callback(self, img_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        try:
            frame = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='bgr8')
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            depth = self.bridge.imgmsg_to_cv2(depth_msg, desired_encoding='32FC1')


            # --- Hier yolo tracker auf 'rgb' anwenden ---
                # bounding box (x,y, width, height)
            if False:
                bb, _ = self.human_tracker.track(rgb)
            else:
                x = y = 0
                height, width = frame.shape[:2]
                bb = [x,y,width, height]

            # --- Hier mediapipe pose auf jeder BBox laufen lassen ---
            keypoints = self.kp_tracker.detect_pose(rgb, bb)
            print(keypoints)

            # --- Hier Pixelkoordinaten mit Tiefendaten und (fx,fy,cx,cy) in 3D reprojizieren ---
            x, z = self.localizer.localize(keypoints, depth, info_msg)
            print(f"Durchschnittliche Position: x={x/1000:.2f} m, z={z/1000:.2f} m")

        except Exception as e:
            self.get_logger().error(f'Fehler im synchronisierten Callback: {e}')


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
