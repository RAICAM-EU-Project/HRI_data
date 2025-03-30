import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import array  # Needed for checking and converting array type

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

# Global variable and a lock to safely share HRI data between threads.
latest_hri_data = None
data_lock = threading.Lock()

class HRIListener(Node):
    """
    ROS2 Node to subscribe to the '/hri/combined_data' topic.
    Expected message type: Float32MultiArray.
    """
    def __init__(self):
        super().__init__('hri_listener')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/hri/combined_data',
            self.listener_callback,
            10)
    
    def listener_callback(self, msg):
        global latest_hri_data
        with data_lock:
            latest_hri_data = msg.data  # Store the latest data (likely an array)
        self.get_logger().info(f"Received HRI data: {msg.data}")

class HRIRequestHandler(BaseHTTPRequestHandler):
    """
    HTTP request handler that returns the latest HRI data as JSON.
    """
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        # Retrieve the data safely from the global variable.
        with data_lock:
            data = latest_hri_data
        if data is None:
            response = {"error": "No HRI data available yet"}
        else:
            # Convert data from a Python array to a list for JSON serialization.
            if isinstance(data, array.array):
                data = list(data)
            response = {"hri_data": data}
        self.wfile.write(json.dumps(response).encode('utf-8'))

def run_http_server(port=8008):
    """
    Launch the HTTP server on the specified port.
    """
    server_address = ('', port)
    httpd = HTTPServer(server_address, HRIRequestHandler)
    print(f"HTTP server running on port {port}.")
    httpd.serve_forever()

def main():
    # Initialize ROS2.
    rclpy.init(args=None)
    hri_node = HRIListener()
    
    # Run ROS2 spin in a separate daemon thread.
    ros_thread = threading.Thread(target=rclpy.spin, args=(hri_node,), daemon=True)
    ros_thread.start()
    
    try:
        # Start the HTTP server (blocking call).
        run_http_server(port=8008)
    except KeyboardInterrupt:
        print("HTTP server interrupted. Shutting down...")
    finally:
        hri_node.destroy_node()
        rclpy.shutdown()
        ros_thread.join()

if __name__ == '__main__':
    main()
