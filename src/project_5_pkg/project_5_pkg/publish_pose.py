import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Point
from pynput import keyboard
import threading

class HumanPositionPublisher(Node):
    def __init__(self):
        super().__init__('keyboard_position_publisher')
        self.publisher_ = self.create_publisher(Point, '/human_relative_position', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)
        self.x = 0.0
        self.z = 1.0
        self.lock = threading.Lock()

        # Start key listener in background thread
        listener = keyboard.Listener(on_press=self.on_key_press)
        listener.daemon = True
        listener.start()

        self.get_logger().info('Use arrow keys to control (x, z): ← → ↓ ↑')

    def on_key_press(self, key):
        with self.lock:
            try:
                if key == keyboard.Key.up:
                    self.z += 0.1
                elif key == keyboard.Key.down:
                    self.z -= 0.1
                elif key == keyboard.Key.left:
                    self.x -= 0.1
                elif key == keyboard.Key.right:
                    self.x += 0.1
            except Exception as e:
                self.get_logger().warn(f"Key press error: {e}")

    def timer_callback(self):
        with self.lock:
            msg = Point()
            msg.x = self.x
            msg.y = 0.0
            msg.z = self.z
            self.publisher_.publish(msg)
            self.get_logger().info(f'Publishing: (x={self.x:.2f}, z={self.z:.2f})')

def main(args=None):
    rclpy.init(args=args)
    node = HumanPositionPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
