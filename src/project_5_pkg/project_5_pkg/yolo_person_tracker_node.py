import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory
import os

class PersonTrackerNode(Node):
    def __init__(self):
        super().__init__('person_tracker_node')
        package_share = get_package_share_directory('project_5_pkg')  # use your exact package name
        model_path = os.path.join(package_share, 'models', 'yolov8n_80_epochs.pt')
        self.model = YOLO(model_path)
        self.bridge = CvBridge()
        self.selected_id = None
        self.boxes = []

        self.subscription = self.create_subscription(
            Image,
            '/camera/camera/color/image_raw',
            self.image_callback,
            10
        )

        cv2.namedWindow('Tracked', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Tracked', self.click_handler)

        self.get_logger().info("YOLOv8 Person Tracker Node Initialized")

    def click_handler(self, event, x, y, flags, param=None):
        if event == cv2.EVENT_LBUTTONDOWN:
            for box in self.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.selected_id = int(box.id[0])
                    self.get_logger().info(f"Selected Track ID: {self.selected_id}")
                    break

    def image_callback(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')

        results = self.model.track(frame, conf=0.25, tracker="bytetrack.yaml", persist=True)

        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        for r in results:
            self.boxes = r.boxes

            for box in self.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                tid = int(box.id[0]) if box.id is not None else -1
                color = (0, 0, 255) if tid == self.selected_id else (0, 255, 0)
                label = f"ID {tid}" if tid >= 0 else "Untracked"
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_bgr, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.imshow('Tracked', frame_bgr)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = PersonTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
