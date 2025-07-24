#!/usr/bin/env python3

import numpy as np
if not hasattr(np, 'float'):
    np.float = float

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from ahrs.filters import Madgwick
from tf_transformations import (
    euler_from_quaternion,
    quaternion_from_euler,
    quaternion_inverse,
    quaternion_multiply,
)

# Parent frame for the transform: 'camera' or 'map'
PARENT = 'camera'


class CameraOrientationNode(Node):
    """
    ROS2 node using a Madgwick filter to estimate camera tilt (X-axis only)
    and publish the transform from PARENT to 'camera_orientation'.
    """

    def __init__(self):
        super().__init__('camera_orientation_node')

        # Madgwick filter setup
        self.filter = Madgwick(Dt=1/50.0, beta=0.2)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
        self.prev_time = None
        self.q0 = None  # Initial filter orientation

        # Static offset quaternion for frame alignment
        if PARENT == 'map':
            # Optical camera frame to map: -90° about Z
            self.q_offset = quaternion_from_euler(0.0, 0.0, -np.pi/2, axes='sxyz')
        else:
            # Map to camera parent: +90° about X
            self.q_offset = quaternion_from_euler(np.pi/2, 0.0, 0.0, axes='sxyz')

        # TF broadcaster
        self.br = TransformBroadcaster(self)

        # IMU subscription
        qos = QoSProfile(depth=10)
        qos.reliability = QoSReliabilityPolicy.BEST_EFFORT
        self.create_subscription(
            Imu,
            '/go2/d435i/imu',
            self.imu_callback,
            qos_profile=qos,
        )

        self.get_logger().info('CameraOrientationNode started')

    def imu_callback(self, msg: Imu):
        # Compute elapsed time
        timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        dt = self.filter.Dt if self.prev_time is None else timestamp - self.prev_time
        self.prev_time = timestamp
        self.filter.Dt = dt

        # Read IMU data
        gyro = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.z,
            msg.angular_velocity.y,
        ])
        acc = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
        ])

        # Update filter and set initial orientation
        self.q = self.filter.updateIMU(self.q, gyr=gyro, acc=acc)

        if self.q0 is None:
            self.q0 = self.q.copy()

        # Relative orientation: q_rel = inv(q0) * q
        q = [self.q[1], self.q[2], self.q[3], self.q[0]]
        q0 = [self.q0[1], self.q0[2], self.q0[3], self.q0[0]]
        q_rel = quaternion_multiply(quaternion_inverse(q0), q)

        # Extract roll only, zero pitch and yaw
        roll, _, _ = euler_from_quaternion(q_rel, axes='sxyz')
        q_roll = quaternion_from_euler(roll, 0.0, 0.0, axes='sxyz')

        # Apply static offset
        q_final = quaternion_multiply(self.q_offset, q_roll)

        # Publish transform
        tf_msg = TransformStamped()
        tf_msg.header.stamp = msg.header.stamp
        tf_msg.header.frame_id = PARENT
        tf_msg.child_frame_id = 'camera_orientation'
        tf_msg.transform.translation.x = 0.0
        tf_msg.transform.translation.y = 0.0
        tf_msg.transform.translation.z = 0.5
        tf_msg.transform.rotation.x = q_final[0]
        tf_msg.transform.rotation.y = q_final[1]
        tf_msg.transform.rotation.z = q_final[2]
        tf_msg.transform.rotation.w = q_final[3]
        self.br.sendTransform(tf_msg)


def main(args=None):
    rclpy.init(args=args)
    node = CameraOrientationNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()