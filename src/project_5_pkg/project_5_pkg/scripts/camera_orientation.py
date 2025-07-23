#!/usr/bin/env python3
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
from ahrs.filters import Madgwick
from tf_transformations import quaternion_matrix

class CameraOrientationNode(Node):
    """
    ROS2-Node, die aus IMU-Daten mit einem Madgwick-Filter die Kamerorientierung schätzt
    und als TF-Transform zwischen Welt- und Kamerarahmen published.
    """
    def __init__(self):
        super().__init__('camera_orientation_node')

        beta = 0.1
        sample_period = 1/50.0

        # Madgwick flter init
        self.filter = Madgwick(sampleperiod=sample_period, beta=beta)
        self.q = np.array([1.0, 0.0, 0.0, 0.0])  # [w, x, y, z]
        self.prev_time = None

        # TF-Broadcaster
        self.br = TransformBroadcaster(self)

        # IMU-Daten abonnieren
        self.create_subscription(
            Imu,
            '/go2/d435i/accel/sample',
            self.imu_callback,
            10
        )

        self.get_logger().info('CameraOrientationNode started')

    def imu_callback(self, msg: Imu):
        gyro = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])
        acc = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])

        # Zeitdifferenz berechnen
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        if self.prev_time is None:
            dt = self.filter.sampleperiod
        else:
            dt = t - self.prev_time
        self.prev_time = t
        self.filter.sampleperiod = dt

        # Madgwick-Filter Update
        self.q = self.filter.updateIMU(self.q, gyr=gyro, acc=acc)

        # TF-Transform erstellen und senden
        tf_msg = TransformStamped()
        tf_msg.header.stamp = msg.header.stamp
        tf_msg.header.frame_id = 'realsense_d435i_joint'
        tf_msg.child_frame_id = 'camera'
        tf_msg.transform.translation.x = 0.0
        tf_msg.transform.translation.y = 0.0
        tf_msg.transform.translation.z = 0.0
        tf_msg.transform.rotation.w = float(self.q[0])
        tf_msg.transform.rotation.x = float(self.q[1])
        tf_msg.transform.rotation.y = float(self.q[2])
        tf_msg.transform.rotation.z = float(self.q[3])
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
