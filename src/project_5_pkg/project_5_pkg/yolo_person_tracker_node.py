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
        model_path = os.path.join(package_share, 'models', 'yolov8n_80_epochs.pt')
        self.model = YOLO(model_path)
        self.bridge = CvBridge()

        # Tracking variables
        self.selected_id = None
        self.boxes = []
        self.last_known_box = None  # Store last known position if person disappears

        # ROS subscribers and publishers
        self.subscription = self.create_subscription(
            Image,
            '/go2/d435i/color/image_raw',
            self.image_callback,
            10
        )
        self.bbox_publisher = self.create_publisher(
            Int32MultiArray,
            'tracked_person_bbox',
            10
        )

        # OpenCV window setup
        cv2.namedWindow('Person Tracking', cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('Person Tracking', self.click_handler)
        self.get_logger().info("Person Tracker Node Initialized - Click on a person to track")

    def click_handler(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            for box in self.boxes:
                if box.id is None:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if x1 <= x <= x2 and y1 <= y <= y2:
                    self.selected_id = int(box.id[0])
                    self.get_logger().info(f"Tracking person with ID: {self.selected_id}")
                    break

    def image_callback(self, msg):
        # Convert ROS image to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='rgb8')
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Run YOLOv8 tracking
        results = self.model.track(
            frame,
            conf=0.25,
            tracker="strongsort.yaml",
            persist=True,
            verbose=False
        )

        # Process results
        current_selected_box = None
        self.boxes = []

        if results[0].boxes.id is not None:
            self.boxes = results[0].boxes
            for box in self.boxes:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                tid = int(box.id[0])

                # Check if this is the selected person
                if tid == self.selected_id:
                    color = (0, 0, 255)  # Red for selected person
                    current_selected_box = (x1, y1, x2, y2)
                    # Publish bounding box coordinates
                    self.publish_bbox(x1, y1, x2, y2)
                else:
                    color = (0, 255, 0)  # Green for other people

                # Draw bounding box and ID
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame_bgr, f"ID {tid}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Handle case where selected person disappears from frame
        if self.selected_id is not None and current_selected_box is None:
            if self.last_known_box:
                x1, y1, x2, y2 = self.last_known_box
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame_bgr, f"ID {self.selected_id} (OFFSCREEN)",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            self.last_known_box = current_selected_box

        # Display instructions
        cv2.putText(frame_bgr, "Click on a person to track", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # Show the frame
        cv2.imshow('Person Tracking', frame_bgr)
        cv2.waitKey(1)

    def publish_bbox(self, x1, y1, x2, y2):
        """Publish bounding box coordinates as [x1, y1, x2, y2]"""
        bbox_msg = Int32MultiArray()
        bbox_msg.data = [x1, y1, x2, y2]
        self.bbox_publisher.publish(bbox_msg)


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