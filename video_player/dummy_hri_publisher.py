#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

class DummyHRIPublisher(Node):
    def __init__(self):
        super().__init__('dummy_hri_publisher')
        # Create a publisher for the /hri_data topic using a valid ROS2 message type.
        self.publisher_ = self.create_publisher(Float32MultiArray, '/hri_data', 10)
        # Publish at 10Hz (0.1 seconds interval)
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

    def timer_callback(self):
        msg = Float32MultiArray()
        # Fill the data array with zeros:
        # data[0] = eye_blink_rate, data[1] = gsr, data[2] = face_temperature
        msg.data = [0.0, 0.0, 0.0]
        self.publisher_.publish(msg)
        self.get_logger().info('Published dummy HRI data: eye_blink_rate=0.0, gsr=0.0, face_temperature=0.0')

def main(args=None):
    rclpy.init(args=args)
    node = DummyHRIPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
