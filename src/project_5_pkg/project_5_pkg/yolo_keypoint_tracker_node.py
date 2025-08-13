import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
from ament_index_python.packages import get_package_share_directory
import os
from std_msgs.msg import Int32MultiArray  # For publishing bounding boxes


class PersonTrackerNode(Node):
    def __init__(self):
        super().__init__('person_tracker_node')
        # Load YOLO model
        package_share = get_package_share_directory('project_5_pkg')
        model_path = os.path.join(package_share, 'models', 'yolov8n_pose.pt')
        self.model = YOLO(model_path)
        self.bridge = CvBridge()

        # Tracking variables
        self.selected_id = None
        self.boxes = []

        # ROS subscribers and publishers
        self.subscription = self.create_subscription(
            Image,
            '/go2/d435i/color/image_raw',
            self.image_callback,
            10
        )

        # OpenCV window setup
        cv2.namedWindow('Person Tracking', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Person Tracking', self.click_handler)
        self.get_logger().info("Person Tracker Node Initialized - Click on a person to track")

    def click_handler(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            results = param['results']
            for box in results[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if x1 <= x <= x2 and y1 <= y <= y2 and box.id is not None:
                    self.selected_id = int(box.id[0])
                    print(f"Selected person ID: {self.selected_id}")
                    break

    def image_callback(self, msg):
        # Convert ROS image to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Run YOLOv8 tracking
        results = self.model.track(
            frame,
            conf=0.3,
            tracker="botsort.yaml",
            persist=True,
            verbose=False
        )

        # Process results
        current_selected_box = None
        self.boxes = []

        if results[0].boxes.id is not None:
            for box, kpts in zip(results[0].boxes, results[0].keypoints):
                tid = int(box.id[0]) if box.id is not None else -1
                
                # Highlight selected person
                if tid == self.selected_id:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame_bgr, f"ID {tid}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # Draw all keypoints for selected person
                    for x, y in kpts.xy[0]:
                        cv2.circle(frame_bgr, (int(x), int(y)), 5, (0, 255, 0), -1)
                else:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 255), 2)
                    cv2.putText(frame_bgr, f"ID {tid}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                    # Draw all keypoints for selected person
                    for x, y in kpts.xy[0]:
                        cv2.circle(frame_bgr, (int(x), int(y)), 5, (0, 255, 0), -1)

        # Display instructions
        cv2.putText(frame_bgr, "Click on a person to track", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show the frame
        cv2.imshow('Person Tracking', frame_bgr)
        cv2.waitKey(1)


def main(args=None):
    rclpy.init(args=args)
    node = PersonTrackerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node shutdown by user")
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()