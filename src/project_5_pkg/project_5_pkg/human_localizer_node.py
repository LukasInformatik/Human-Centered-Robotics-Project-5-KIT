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
        self.localizer = HumanLocalizer()
        self.human_tracker = HumanTracker()

        # Define QoS profiles
        camera_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.RELIABLE,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )
        
        imu_qos = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=5
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
            '/go2/d435i/depth/image_rect_raw',
            qos_profile=camera_qos
        )
        self.sub_info = Subscriber(
            self, 
            CameraInfo, 
            '/go2/d435i/color/camera_info',
            qos_profile=camera_qos
        )
        
        # IMU subscriber - using regular create_subscription
        self.latest_imu_data = None
        self.imu_sub = self.create_subscription(
            Imu,
            '/go2/d435i/accel/sample',
            self.imu_callback,
            qos_profile=imu_qos
        )

        # Synchronizer (without IMU)
        self.ts = ApproximateTimeSynchronizer(
            [self.sub_color, self.sub_depth, self.sub_info],
            queue_size=10,
            slop=0.1
        )
        self.ts.registerCallback(self.synced_callback)
        self.get_logger().info('HumanLocalizerPipeline initialized')

    def synced_callback(self, img_msg: Image, depth_msg: Image, info_msg: CameraInfo):
        if self.latest_imu_data is None:
            self.get_logger().warn("No IMU data received yet!")
            return None, None

        if self.latest_imu_data is None:
            self.get_logger().warn("No IMU data received yet!")
            return
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
            
            # IMU
            acc = (
                self.latest_imu_data.linear_acceleration.x,
                self.latest_imu_data.linear_acceleration.y,
                self.latest_imu_data.linear_acceleration.z
            )
            R, roll, pitch = self.acc2rotmat(acc)

            # --- Hier Pixelkoordinaten mit Tiefendaten und (fx,fy,cx,cy) in 3D reprojizieren ---
            x3d, z3d = self.localizer.localize(keypoints, depth, info_msg, R)
            print(f"Durchschnittliche Position: x={x3d/1000:.2f} m, z={z3d/1000:.2f} m")


        except Exception as e:
            self.get_logger().error(f'Fehler im synchronized callback: {e}')

    def imu_callback(self, msg: Imu):
        """Speichert die neuesten IMU-Daten für die spätere Verwendung."""
        self.latest_imu_data = msg
        self.latest_imu_timestamp = msg.header.stamp
    @staticmethod
    def acc2rotmat(acc):
        ax, ay, az = acc

        roll = np.arctan2(ay, az)
        pitch = np.arctan2(-ax, np.sqrt(ay * ay + az * az))

        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll),  np.cos(roll)]
        ])

        Ry = np.array([
            [ np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        R = Ry @ Rx
        #print(
        #    f"IMU: Roll={np.degrees(roll):.2f}°, Pitch={np.degrees(pitch):.2f}°\nR=\n{R}"
        #    )
        return R, roll, pitch

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
