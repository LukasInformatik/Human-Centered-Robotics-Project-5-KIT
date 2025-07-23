#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import numpy as np
import time
from collections import deque

# Replace with the correct import from your Unitree SDK Python binding
from unitree_sdk2py.core.locomotion_client import LocomotionClient

class PIDController:
    def __init__(self, kp, ki, kd, output_limit):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limit = output_limit
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

    def compute(self, error, current_time):
        dt = current_time - self.prev_time if self.prev_time else 0.01
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0

        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        output = np.clip(output, -self.output_limit, self.output_limit)

        self.prev_error = error
        self.prev_time = current_time

        return output

class MovementControlNode(Node):
    def __init__(self):
        super().__init__('movement_control_node')

        # Parameters (tunable)
        self.desired_distance = 1.2  # meters
        self.max_forward_speed = 0.6  # m/s
        self.max_yaw_rate = 1.0  # rad/s
        self.distance_tolerance = 0.15  # meters
        self.x_offset_tolerance = 0.1  # meters
        self.control_rate = 10.0  # Hz

        # Control gains (tunable)
        self.K_vx = 0.6
        self.K_yaw = 1.2

        # Initialize Unitree SDK client
        self.client = LocomotionClient()
        self.client.Init()
        self.client.Stand()  # Ensure the robot stands up

        self.last_position_time = time.time()
        self.position_timeout = 1.0  # seconds

        # Subscriber to person position topic
        self.sub_position = self.create_subscription(
            Point, 
            '/human_relative_position', 
            self.position_callback, 
            10
        )

        # Timer for control loop
        self.control_timer = self.create_timer(1.0 / self.control_rate, self.control_loop)

        # Initialize state
        self.current_target = None

        self.get_logger().info("Movement Control node initialized and robot in standby.")

    def position_callback(self, msg: Point):
        self.current_target = msg
        self.last_position_time = time.time()

    def control_loop(self):
        now = time.time()
        if self.current_target is None or (now - self.last_position_time) > self.position_timeout:
            # Person lost or no updates
            self.get_logger().warn("Target lost or stale data. Stopping robot.")
            self.client.StopMove()
            return

        x_offset = self.current_target.x
        distance = self.current_target.z

        # Distance control
        distance_error = distance - self.desired_distance
        if abs(distance_error) > self.distance_tolerance:
            vx = np.clip(self.K_vx * distance_error, -self.max_forward_speed, self.max_forward_speed)
        else:
            vx = 0.0  # Within tolerance, stop moving forward/backward

        # Yaw control (rotate towards human)
        if abs(x_offset) > self.x_offset_tolerance:
            yaw_rate = np.clip(-self.K_yaw * x_offset, -self.max_yaw_rate, self.max_yaw_rate)
        else:
            yaw_rate = 0.0  # Target centered, no rotation needed

        # vy (lateral speed), not used for now
        vy = 0.0

        # Debugging outputs
        self.get_logger().info(
            f"Commanding velocities -> vx: {vx:.2f} m/s, yaw_rate: {yaw_rate:.2f} rad/s"
        )

        # Send command to robot (continuous_move=True keeps the robot moving smoothly)
        self.client.Move(vx, vy, yaw_rate, continuous_move=False)

def main(args=None):
    rclpy.init(args=args)
    control_node = MovementControlNode()

    try:
        rclpy.spin(control_node)
    except KeyboardInterrupt:
        pass
    finally:
        control_node.client.StopMove()
        control_node.client.LayDown()
        control_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
