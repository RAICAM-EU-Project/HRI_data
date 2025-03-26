#!/usr/bin/env python3
import os
import sys
import time
import json
import cv2
import queue
import tkinter as tk
from tkinter import Button, Label
from PIL import Image, ImageTk

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

# Create a thread-safe queue for storing HRI data.
hri_data_queue = queue.Queue()

# ROS2 Subscriber Node for /hri_data using Float32MultiArray.
class HRISubscriber(Node):
    def __init__(self):
        super().__init__('hri_subscriber')
        self.subscription = self.create_subscription(
            Float32MultiArray,
            '/hri_data',
            self.listener_callback,
            10
        )
        self.recording = False

    def listener_callback(self, msg):
        if self.recording:
            data = {
                'timestamp': time.time(),
                'eye_blink_rate': msg.data[0] if len(msg.data) > 0 else 0.0,
                'gsr': msg.data[1] if len(msg.data) > 1 else 0.0,
                'face_temperature': msg.data[2] if len(msg.data) > 2 else 0.0
            }
            hri_data_queue.put(data)

    def start_recording(self):
        self.recording = True

    def stop_recording(self):
        self.recording = False

# VideoPlayer class using Tkinter.
class VideoPlayer:
    def __init__(self, root, video_files, hri_subscriber, initial_width):
        self.root = root
        self.video_files = video_files
        self.current_video_index = 0
        self.hri_subscriber = hri_subscriber
        self.recorded_data = []  # List to store HRI data for the current video.
        self.cap = None
        self.video_width = initial_width  # default width; will be updated per video

        # Create the data folder if it doesn't exist.
        self.data_folder = "./data"
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)

        # Create a frame for the video display with height fixed at 1080.
        self.video_frame = tk.Frame(root, width=self.video_width, height=1080)
        self.video_frame.pack(side="top", fill="both", expand=True)

        # Label to display video frames.
        self.video_label = Label(self.video_frame)
        self.video_label.pack(fill="both", expand=True)

        # Play button placed under the video.
        self.play_button = Button(root, text="Play", command=self.play_video)
        self.play_button.pack(side="bottom", pady=10)

        self.update_video_id = None

    def play_video(self):
        if self.current_video_index >= len(self.video_files):
            self.video_label.config(text="No more videos.")
            return

        video_path = self.video_files[self.current_video_index]
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.video_label.config(text=f"Cannot open video: {video_path}")
            return

        # Clear previously recorded data and start recording.
        self.recorded_data = []
        self.hri_subscriber.start_recording()
        self.play_button.config(state="disabled")

        # Read the first frame to determine the appropriate width for height 1080.
        ret, first_frame = self.cap.read()
        if ret:
            orig_h, orig_w = first_frame.shape[:2]
            scale = 1080 / orig_h
            self.video_width = int(orig_w * scale)
            # Update window geometry: video area height=1080 plus extra 120 pixels for the button.
            self.root.geometry(f"{self.video_width}x{1080+120}")
            # Display the first frame.
            resized_frame = cv2.resize(first_frame, (self.video_width, 1080))
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(resized_frame)
            imgtk = ImageTk.PhotoImage(image=image)
            self.video_label.imgtk = imgtk  # Keep a reference.
            self.video_label.config(image=imgtk)
            # Schedule subsequent frame updates.
            self.update_video_id = self.root.after(30, self.update_video)
        else:
            self.video_label.config(text=f"Cannot read first frame: {video_path}")
            self.play_button.config(state="normal")

    def update_video(self):
        # Poll the HRI data queue and append any new data.
        while not hri_data_queue.empty():
            data = hri_data_queue.get()
            self.recorded_data.append(data)

        ret, frame = self.cap.read()
        if ret:
            resized_frame = cv2.resize(frame, (self.video_width, 1080))
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(resized_frame)
            imgtk = ImageTk.PhotoImage(image=image)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
            self.update_video_id = self.root.after(30, self.update_video)
        else:
            if self.update_video_id:
                self.root.after_cancel(self.update_video_id)
            self.cap.release()
            self.hri_subscriber.stop_recording()
            self.save_data()
            self.current_video_index += 1
            self.play_button.config(state="normal")
            self.video_label.config(text="Video ended. Click Play for the next video.")

    def save_data(self):
        video_file = self.video_files[self.current_video_index]
        base_name = os.path.splitext(os.path.basename(video_file))[0]
        json_file = os.path.join(self.data_folder, base_name + "_hri_data.json")
        try:
            with open(json_file, "w") as f:
                json.dump(self.recorded_data, f, indent=4)
            print(f"Saved HRI data to {json_file}")
        except Exception as e:
            print(f"Error saving JSON data: {e}")

# Function to periodically spin the ROS2 node within the Tkinter event loop.
def update_ros(hri_subscriber, root):
    rclpy.spin_once(hri_subscriber, timeout_sec=0.001)
    root.after(1, update_ros, hri_subscriber, root)

def main():
    # List your local video files.
    video_files = [
        "./vids/1.webm",
        "./vids/2.webm",
        "./vids/3.webm"
    ]

    # Use the first video to compute the initial width (scaled to height 1080).
    initial_width = 1920  # default fallback
    if video_files:
        cap = cv2.VideoCapture(video_files[0])
        ret, frame = cap.read()
        if ret:
            orig_h, orig_w = frame.shape[:2]
            scale = 1080 / orig_h
            initial_width = int(orig_w * scale)
        cap.release()

    total_height = 1080 + 120  # 1080 for video and 120 for button area

    rclpy.init(args=sys.argv)
    hri_subscriber = HRISubscriber()
    root = tk.Tk()
    root.title("Video Player with HRI Recording")
    root.geometry(f"{initial_width}x{total_height}")
    player = VideoPlayer(root, video_files, hri_subscriber, initial_width)
    root.after(1, update_ros, hri_subscriber, root)
    root.mainloop()
    hri_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == "__main__":
    main()
