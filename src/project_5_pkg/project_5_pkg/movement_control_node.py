import sys
print("Python interpreter:", sys.executable)
import time
from collections import deque

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
import numpy as np

# DDS channel setup
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
# High-level sports-mode client
from unitree_sdk2py.go2.sport.sport_client import SportClient

class MovementControlNode(Node):
    def __init__(self):
        super().__init__('movement_control_node')

        # === 1. Initialize DDS & SDK client ===
        # Choose ROS_DOMAIN_ID=0 and interface
        ChannelFactoryInitialize(1, 'lo')                 # local DDS setup
        # ChannelFactoryInitialize(0, 'enx3c18a0238600')    # setup DDS participant/interface
        self.client = SportClient()                       # high-level Go2 client           
        self.client.Init()                                # create underlying DDS channels  
        self.client.StandUp()                             # bring robot to standing mode    

        # === 2. Controller parameters ===
        self.desired_distance   = 1.2   # m
        self.prediction_horizon = 0.5   # s
        self.control_rate       = 10.0  # Hz
        
        # Deadbands
        self.distance_deadband = 0.1    # meters
        self.yaw_deadband      = 0.05   # radians  

        # PID gains (tune these for your setup)
        self.distance_pid = PIDController(kp=0.8, ki=0.05, kd=0.2, output_limit=0.7)
        self.yaw_pid      = PIDController(kp=1.5, ki=0.05, kd=0.3, output_limit=1.2)

        # EMA filter factor
        self.alpha = 0.5

        # Buffer for predictive control (store last ~0.5 s of data)
        self.position_buffer = deque(maxlen=int(self.control_rate * self.prediction_horizon))

        # Tracking data timeout
        self.last_received_time = time.time()
        self.timeout = 1  # s

        # === 3. ROS2 subscription & timer ===
        self.create_subscription(Point, '/human_relative_position',
                                 self.position_callback, 10)
        self.create_timer(1.0/self.control_rate, self.control_loop)

        self.get_logger().info("PersonFollowerPIDNode ready. Entering control loop.")

    def position_callback(self, msg: Point):
        now = time.time()
        self.last_received_time = now

        # Exponential Moving Average filter
        if not self.position_buffer:
            fx, fz = msg.x, msg.z
        else:
            fx = self.alpha * msg.x + (1 - self.alpha) * self.position_buffer[-1][1]
            fz = self.alpha * msg.z + (1 - self.alpha) * self.position_buffer[-1][2]

        # Store (timestamp, filtered_x, filtered_z)
        self.position_buffer.append((now, fx, fz))

    def predict(self):
        if len(self.position_buffer) < 2:
            return self.position_buffer[-1][1:] if self.position_buffer else (0.0, 0.0)
        t0, x0, z0 = self.position_buffer[0]
        t1, x1, z1 = self.position_buffer[-1]
        dt = t1 - t0
        if dt <= 0:
            return x1, z1
        vx = (x1 - x0) / dt
        vz = (z1 - z0) / dt
        return x1 + vx*self.prediction_horizon, z1 + vz*self.prediction_horizon

    def control_loop(self):
        now = time.time()
        # Timeout check
        if now - self.last_received_time > self.timeout:
            self.get_logger().warn("No recent target position data → stopping robot.")
            self.client.StopMove()              
            self.distance_pid.reset()
            self.yaw_pid.reset()
            return

        # Predict future target
        pred_x, pred_z = self.predict()

        # Compute errors
        dist_err = pred_z - self.desired_distance
        yaw_err  = -pred_x

        # --- Deadband & Anti-Windup for distance ---
        if abs(dist_err) < self.distance_deadband:
            vx = 0.0
            self.distance_pid.reset()
        else:
            vx = self.distance_pid.compute(dist_err, now)

        # --- Deadband & Anti-Windup for yaw ---
        if abs(yaw_err) < self.yaw_deadband:
            yaw_rate = 0.0
            self.yaw_pid.reset()
        else:
            yaw_rate = self.yaw_pid.compute(yaw_err, now)

        self.get_logger().info(
            f"Pred → x: {pred_x:.2f}, z: {pred_z:.2f} | cmd → vx: {vx:.2f}, ω: {yaw_rate:.2f}"
        )

        # Send velocity command
        self.client.Move(vx, 0.0, yaw_rate)      # forward, lateral=0, yaw

    def destroy_node(self):
        # Safely lay down before exit
        self.client.Move(0.0, 0.0, 0.0)
        self.client.StandDown()                
        super().destroy_node()

class PIDController:
    def __init__(self, kp, ki, kd, output_limit):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.limit = output_limit
        self.intg = 0.0
        self.prev_err = 0.0
        self.prev_time = None
        
    def reset(self):
        """Clear integral and derivative history."""
        self.intg = 0.0
        self.prev_err = 0.0
        self.prev_time = None

    def compute(self, err, now):
        dt = now - (self.prev_time or now)
        self.intg += err * dt
        deriv = (err - self.prev_err) / dt if dt > 0 else 0.0
        out = (self.kp*err + self.ki*self.intg + self.kd*deriv)
        out = max(-self.limit, min(self.limit, out))
        self.prev_err, self.prev_time = err, now
        return out

def main(args=None):
    rclpy.init(args=args)
    node = MovementControlNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
