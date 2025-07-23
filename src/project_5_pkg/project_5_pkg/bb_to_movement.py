import rclpy
from rclpy.node import Node
from ament_index_python.packages import get_package_share_directory
from std_msgs.msg import Int32MultiArray  # For publishing bounding boxes
from geometry_msgs.msg import Twist
import time
import math


class Go2MovementPublisher(Node):
    def __init__(self):
        super().__init__('person_tracker_node')
        # Load YOLO model
        package_share = get_package_share_directory('project_5_pkg')


        # ROS subscribers and publishers
        self.subscription = self.create_subscription(
            Int32MultiArray,
            '/bounding_box',
            self.bb_callback,
            10
        )
        
        self.publisher = self.create_publisher(
            Twist,
            '/cmd_vel',
            10
        )

        self.get_logger().info("Started BB-to-Movement Node")

        self.last_command_time = time.time()
        self.timeout_duration = 1.5  # 3 seconds
        self.is_moving = False
        self.safety_stop_sent = False
        
        # Timer for safety timeout check
        self.safety_timer = self.create_timer(0.1, self.safety_timeout_callback)  # 10Hz
        


    def publish_movement(self, linear_x=0.0, linear_y=0.0, angular_z=0.0):
        """
        Publish movement command
        
        Args:
            linear_x: Forward/backward velocity (m/s)
            linear_y: Left/right velocity (m/s) 
            angular_z: Rotation velocity (rad/s)
        """
        msg = Twist()
        msg.linear.x = linear_x
        msg.linear.y = linear_y
        msg.linear.z = 0.0
        
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = angular_z
        
        self.publisher.publish(msg)
        self.get_logger().info(f'Published: linear_x={linear_x}, linear_y={linear_y}, angular_z={angular_z}')

        # Update safety timeout tracking
        self.last_command_time = time.time()
        self.is_moving = (linear_x != 0.0 or linear_y != 0.0 or angular_z != 0.0)
        self.safety_stop_sent = False  # Reset safety stop flag when new command is sent
        
        

    def stop(self):
        """Stop all movement"""
        msg = Twist()
        msg.linear.x = 0.0
        msg.linear.y = 0.0
        msg.linear.z = 0.0
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0
        
        self.publisher.publish(msg)
        self.get_logger().info('STOP command sent')
        
        self.is_moving = False

    def keep_alive(self):
        """
        Continue current movement for the specified time
        """
        self.last_command_time = time.time()
        self.safety_stop_sent = False

    def safety_timeout_callback(self):
        """
        Ensure the robot stops if no commad is received
        """
        current_time = time.time()
        time_since_last_command = current_time - self.last_command_time
        
        if time_since_last_command > self.timeout_duration and self.is_moving and not self.safety_stop_sent:
            self.get_logger().warn(f'No movement command received for {self.timeout_duration} seconds - SAFETY STOP activated!')
            self.stop()
            self.safety_stop_sent = True
            self.is_moving = False

    def get_pos(self, image_x_offset, image_y_offset, depth, image_width=1280, image_height=720, fov_horizontal=91.2, fov_vertical=65.5):
        """
        Calculate the 3D position and angle of a point from an RGBD image.
        Intel® RealSense™ Tiefenkamera D435i 
            1280x720
            30fps
            FoV (HxVxD) 91.2 x 65.5 x 100.6. 
        
        Args:
            image_x_offset (int): X pixel coordinate (0 to image_width-1)
            image_y_offset (int): Y pixel coordinate (0 to image_height-1) 
            depth (float): Depth value at the pixel (in meters or consistent units)
            image_width (int): Width of the image in pixels
            image_height (int, optional): Height of the image in pixels. If None, assumes square image
            fov_horizontal (float): Horizontal field of view in degrees (default: 60°)
            fov_vertical (float, optional): Vertical field of view in degrees. If None, calculated from aspect ratio
        
        Returns:
            dict: Contains:
                - 'x': Horizontal distance from camera center (positive = right)
                - 'y': Vertical distance from camera center (positive = up)  
                - 'z': Depth distance from camera (forward)
                - 'horizontal_angle': Angle in horizontal plane relative to camera forward (degrees)
                - 'vertical_angle': Angle in vertical plane relative to camera forward (degrees)
                - 'distance_3d': Total 3D distance from camera
        """
        
        # Set default image height if not provided
        if image_height is None:
            image_height = image_width
        
        # Calculate vertical FOV if not provided (maintaining aspect ratio)
        if fov_vertical is None:
            aspect_ratio = image_height / image_width
            fov_vertical = fov_horizontal * aspect_ratio
        
        # Convert FOV to radians
        fov_h_rad = math.radians(fov_horizontal)
        fov_v_rad = math.radians(fov_vertical)
        
        # Calculate the center of the image
        center_x = image_width / 2.0
        center_y = image_height / 2.0
        
        # Calculate the offset from center in pixels
        pixel_offset_x = image_x_offset - center_x
        pixel_offset_y = center_y - image_y_offset  # Flip Y to make positive = up
        
        # Calculate angles per pixel
        angle_per_pixel_h = fov_h_rad / image_width
        angle_per_pixel_v = fov_v_rad / image_height
        
        # Calculate the angles from the camera's forward direction
        horizontal_angle_rad = pixel_offset_x * angle_per_pixel_h
        vertical_angle_rad = pixel_offset_y * angle_per_pixel_v
        
        # Convert angles to degrees
        horizontal_angle_deg = math.degrees(horizontal_angle_rad)
        vertical_angle_deg = math.degrees(vertical_angle_rad)
        
        # Calculate 3D position
        # In camera coordinate system: Z is forward, X is right, Y is up
        z = depth * math.cos(horizontal_angle_rad) * math.cos(vertical_angle_rad)
        x = depth * math.sin(horizontal_angle_rad) * math.cos(vertical_angle_rad)
        y = depth * math.sin(vertical_angle_rad)
        
        # Calculate total 3D distance
        distance_2d = math.sqrt(x*x + z*z)
        
        return {
            'x': x,
            'y': y, 
            'z': z,
            'horizontal_angle': horizontal_angle_deg,
            'vertical_angle': vertical_angle_deg,
            'distance_2d': distance_2d
        }


    def bb_callback(self, msg):
        '''
        msg shoul look like [x,y,depth]
        '''
        self.message_count += 1
        self.last_received_time = time.now()
        data = msg.data
        if len(data) == 4:
            x_offset, y_offset, depth = data

        self.get_logger().info(f'Received: x={x_offset}, y={y_offset}, depth={depth}')
        if depth == 0:
            # Human not found
            # rotate to find human
            self.publish_movement(angular_z=y_offset)
            return
        
        pos = self.get_pos(x_offset, y_offset, depth)

        rotation_constant = 1
        movement_constant = 1
        distance_constant = 1

        dist = pos.distance_2d * distance_constant
        rot = rotation_constant if pos.horizontal_angle > 0 else -rotation_constant
        
        # meter is asumed
        if dist > 1:
            # move closer
            self.publish_movement(linear_x=movement_constant, angular_z=rot)
        elif dist < 1 and dist > 0.6:
            # dont move, just rotate
             self.publish_movement(angular_z=rot)
        if dist < 0.6:
            # step back
             self.publish_movement(linear_x=-movement_constant, angular_z=rot)
        


def main(args=None):
    rclpy.init(args=args)
    node = Go2MovementPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Node shutdown by user")
        node.stop()

    finally:
        node.destroy_node()
        rclpy.shutdown()



if __name__ == '__main__':
    main()
