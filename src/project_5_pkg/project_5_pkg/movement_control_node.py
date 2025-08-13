import sys, time, json
from collections import deque
import numpy as np
import math

# ROS2 / TF 
import rclpy
from rclpy.node            import Node
from geometry_msgs.msg     import Point
from tf2_ros               import Buffer, TransformListener, TransformException

# Unitree SDK
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.go2.sport.sport_client import SportClient


# Tuneable Constants
DESIRED_DISTANCE   = 1.20               # [m]
CONTROL_RATE_HZ    = 10.0               # loop frequency
PRED_HORIZON_S     = 0.15               # look-ahead for prediction
DIST_DEADBAND      = 0.10               # [m]  |vx| < this ⇒ 0
YAW_DEADBAND       = math.radians(5)    # [rad]
MAX_FWD_SPEED      = 1.0                # [m/s]
MAX_YAW_SPEED      = math.radians(60)   # [rad/s]
TF_TIMEOUT_S       = 0.50               # [s]     TF older than this → stop

class PIDController:
    def __init__(self, kp, ki, kd, output_limit):
        self.kp, self.ki, self.kd = kp, ki, kd
        self.limit = float(output_limit)
        self.reset()
        
    def reset(self):
        self.intg = 0.0
        self.prev_err = 0.0
        self.prev_time = None

    def compute(self, err, now):
        dt = now - (self.prev_time or now)
        self.intg += err * dt
        deriv = (err - self.prev_err) / dt if dt > 0 else 0.0
        out = (self.kp*err + self.ki*self.intg + self.kd*deriv)
        out = np.clip(out, -self.limit, self.limit)
        self.prev_err, self.prev_time = err, now
        return float(out)
    

class MovementControlNode(Node):
    def __init__(self):
        super().__init__('movement_control_node')

        # === Initialize DDS & SDK client ===
        # Choose ROS_DOMAIN_ID=0 and interface
        # ChannelFactoryInitialize(1, 'lo')                 # local DDS setup
        ChannelFactoryInitialize(0, 'enx3c18a0238600')    # setup DDS participant/interface
        self.client = SportClient()                       # high-level Go2 client    
        self.client.SetTimeout(10.0)       
        self.client.Init()                                # create underlying DDS channels  
        self.client.StandUp()                             # bring robot to standing mode
        self.client.StopMove()  

        # === Controller parameters ===
        self.dt = 1.0 / CONTROL_RATE_HZ
        self.pos_buf = deque(maxlen=int(CONTROL_RATE_HZ * PRED_HORIZON_S))
        
        self.distance_pid = PIDController(kp=0.8, ki=0.00, kd=0.25, output_limit=MAX_FWD_SPEED)
        self.yaw_pid      = PIDController(kp=1.0, ki=0.00, kd=0.5, output_limit=MAX_YAW_SPEED)

        # === TF listener  ===
        self.tf_buffer   = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.clock       = self.get_clock()
        
        # Start loop
        self.create_timer(self.dt, self.control_loop)
        self.get_logger().info("PersonFollowerPIDNode ready. Entering control loop.")
        
        self.tracking = False
        
    def _clip_control_values(self, v, limit_per_sec):
        """
        Convert velocity (m/s or rad/s) → step (m or rad)
        """
        v = float(np.clip(v, -limit_per_sec, limit_per_sec))
        return v #* self.dt

    def predict(self):
        if len(self.pos_buf) < 2:
            return self.pos_buf[-1][1:] if self.pos_buf else (0.0, 0.0)
        t0, x0, z0 = self.pos_buf[0]
        t1, x1, z1 = self.pos_buf[-1]
        dt = t1 - t0
        if dt <= 0:
            return x1, z1
        vx = (x1 - x0) / dt
        vz = (z1 - z0) / dt
        return x1 + vx*PRED_HORIZON_S, z1 + vz*PRED_HORIZON_S

    def control_loop(self):
        now = time.time()
        
        try:
            tf = self.tf_buffer.lookup_transform(
                'camera_orientation',
                'human_frame',
                rclpy.time.Time())
            if self.tracking == False:
                self.tracking = True
                # self.client.Hello()
        except TransformException as e:
            self.get_logger().warn(f"TF lookup failed: {e}")
            self._halt()
            return
        
        # TF timestamp (ROS time) convert to float seconds
        tf_time_ros = tf.header.stamp
        tf_time_sec = tf_time_ros.sec + tf_time_ros.nanosec * 1e-9
        now_ros_sec = self.clock.now().seconds_nanoseconds()[0] + \
                      self.clock.now().seconds_nanoseconds()[1] * 1e-9
                      
        if now_ros_sec - tf_time_sec > TF_TIMEOUT_S:
            self.get_logger().warn("TF data too old → stopping.")
            self._halt()
            self.tracking = False
            return
        
        tx, tz = tf.transform.translation.x, tf.transform.translation.z
        self.pos_buf.append((now, tx, tz))

        # Helps with person changing velocity
        pred_x, pred_z = self.predict()
        # self.get_logger().info(f"{pred_x}, {pred_z}")

        # Compute errors
        dist_err = pred_z - DESIRED_DISTANCE
        yaw_err  = -math.atan2(pred_x, pred_z)

        # Deadband & Anti-Windup for distance
        if abs(dist_err) < DIST_DEADBAND:
            vx = 0.0
            self.distance_pid.reset()
        else:
            vx = self.distance_pid.compute(dist_err, now)

        # Deadband & Anti-Windup for yaw
        if abs(yaw_err) < YAW_DEADBAND:
            yaw_rate = 0.0
            self.yaw_pid.reset()
        else:
            yaw_rate = self.yaw_pid.compute(yaw_err, now)
        
        # Scale forward speed so robot naturally slows while turning
        # vx *= math.cos(yaw_err)
            
        # Velocity -> per-tick displacement
        ctrl_x  = self._clip_control_values(vx,       MAX_FWD_SPEED)
        ctrl_yaw= self._clip_control_values(yaw_rate, MAX_YAW_SPEED)

        self.get_logger().info(
            f"err [dist,yaw]=({dist_err:+.2f},{yaw_err:+.2f}) → "
            f"vel [x,ω]=({vx:+.2f},{yaw_rate:+.2f}) → "
            f"step [Δx,Δyaw]=({ctrl_x:+.3f},{ctrl_yaw:+.3f})")
        
        # Send command
        if ctrl_x == 0 and ctrl_yaw == 0:
            self._halt()
        else:
            pass
            self.client.Move(ctrl_x, 0.0, ctrl_yaw)      # forward, lateral=0, yaw
            
    def _halt(self):
        self.client.StopMove()
        self.distance_pid.reset()
        self.yaw_pid.reset()

    def destroy_node(self):
        # Safely lay down before exit
        self._halt()
        # self.client.StandDown()                
        super().destroy_node()

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
