#!/usr/bin/env python3
import os
import sys
import time
import json
import cv2
import queue
import random
import tkinter as tk
from tkinter import Button, Label, Entry, Frame
from PIL import Image, ImageTk
from datetime import datetime
import threading
import requests

# -------------------- Global Queue for HRI Data --------------------
hri_data_queue = queue.Queue()

# -------------------- Function to Poll HRI Data from HTTP Endpoint --------------------
def poll_hri_data():
    """Continuously poll HRI data from http://localhost:8080/data and enqueue samples."""
    url = "http://localhost:8080/data"
    while True:
        try:
            response = requests.get(url, timeout=1)
            if response.status_code == 200:
                data = response.json()
                # Append a local timestamp for reference.
                data['local_timestamp'] = time.time()
                hri_data_queue.put(data)
        except Exception as e:
            print("Error fetching HRI data:", e)
        time.sleep(0.1)

# -------------------- Function to Get Video Files --------------------
def get_video_files(root_dir):
    video_files = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for file in filenames:
            if file.lower().endswith(('.mp4', '.webm')):
                full_path = os.path.join(dirpath, file)
                video_files.append(full_path)
    return sorted(video_files)

# -------------------- User ID Input Window --------------------
class IDInputWindow:
    def __init__(self, root):
        self.root = root
        self.user_id = None
        self.frame = Frame(root)
        self.frame.pack(pady=20)
        Label(self.frame, text="Enter your ID:").pack(side="left", padx=5)
        self.entry = Entry(self.frame)
        self.entry.pack(side="left", padx=5)
        self.submit_button = Button(self.frame, text="Submit", command=self.submit)
        self.submit_button.pack(side="left", padx=5)

    def submit(self):
        entered = self.entry.get().strip()
        if entered:
            self.user_id = entered
            self.frame.destroy()
        else:
            print("Please enter a valid ID.")

# -------------------- VideoPlayer Class --------------------
class VideoPlayer:
    def __init__(self, root, video_files, initial_width, user_id):
        self.root = root
        self.video_files = video_files
        self.current_video_index = 0
        self.user_id = user_id
        self.recorded_data = []  # HRI data for current video
        self.cap = None
        self.video_width = initial_width  # will be updated per video

        # Video display frame (fixed height 1080)
        self.video_frame = Frame(root, width=self.video_width, height=1080)
        self.video_frame.pack(side="top", fill="both", expand=True)

        # Label to show video frames
        self.video_label = Label(self.video_frame)
        self.video_label.pack(fill="both", expand=True)

        # Play button
        self.play_button = Button(root, text="Play", command=self.play_video)
        self.play_button.pack(side="bottom", pady=10)

        self.update_video_id = None

    def play_video(self):
        if self.current_video_index >= len(self.video_files):
            self.video_label.config(text="No more videos.")
            self.root.after(2000, self.root.quit)
            return

        video_path = self.video_files[self.current_video_index]
        if self.cap:
            self.cap.release()
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            self.video_label.config(text=f"Cannot open video: {video_path}")
            return

        # Clear previous recorded data.
        self.recorded_data = []
        self.play_button.config(state="disabled")

        # Read first frame to compute scaling (for height 1080).
        ret, first_frame = self.cap.read()
        if ret:
            orig_h, orig_w = first_frame.shape[:2]
            scale = 1080 / orig_h
            self.video_width = int(orig_w * scale)
            self.root.geometry(f"{self.video_width}x{1080+120}")
            resized_frame = cv2.resize(first_frame, (self.video_width, 1080))
            resized_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(resized_frame)
            imgtk = ImageTk.PhotoImage(image=image)
            self.video_label.imgtk = imgtk  # Keep reference.
            self.video_label.config(image=imgtk)
            # Schedule next frame update.
            self.update_video_id = self.root.after(30, self.update_video)
        else:
            self.video_label.config(text=f"Cannot read first frame: {video_path}")
            self.play_button.config(state="normal")

    def update_video(self):
        # Poll HRI data queue and record new samples.
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
            self.save_data()
            self.current_video_index += 1

            if self.current_video_index < len(self.video_files):
                self.play_button.config(state="normal")
                self.video_label.config(text="Video ended. Click Play for the next video.")
            else:
                self.video_label.config(text="All selected videos played. Exiting...")
                self.root.after(2000, self.root.quit)

    def save_data(self):
        video_file = self.video_files[self.current_video_index]
        base_name = os.path.splitext(os.path.basename(video_file))[0]
        current_dt = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_file = os.path.join(os.path.dirname(video_file),
                                 f"{self.user_id}_{base_name}_{current_dt}.json")
        try:
            with open(json_file, "w") as f:
                json.dump(self.recorded_data, f, indent=4)
            print(f"Saved HRI data to {json_file}")
        except Exception as e:
            print(f"Error saving JSON data: {e}")

# -------------------- Main Function --------------------
def main():
    # Get all video files from the "./new_data" folder.
    all_video_files = get_video_files("./new_data")
    if not all_video_files:
        print("No video files found in ./new_data")
        return

    # Randomly select 5 videos.
    selected_video_files = random.sample(all_video_files, 5)
    print("Selected videos:")
    for vid in selected_video_files:
        print(vid)

    # Use the first video to compute initial width.
    initial_width = 1920  # fallback
    cap = cv2.VideoCapture(selected_video_files[0])
    ret, frame = cap.read()
    if ret:
        orig_h, orig_w = frame.shape[:2]
        scale = 1080 / orig_h
        initial_width = int(orig_w * scale)
    cap.release()

    total_height = 1080 + 120

    # Create the main Tkinter window.
    root = tk.Tk()
    root.title("Video Player with HRI Recording")
    root.geometry(f"{initial_width}x{total_height}")

    # Create and show the user ID input window.
    id_input = IDInputWindow(root)
    # Wait until a user ID is entered.
    def check_id():
        if id_input.user_id:
            user_id = id_input.user_id
            # Create the VideoPlayer now.
            player = VideoPlayer(root, selected_video_files, initial_width, user_id)
        else:
            root.after(100, check_id)
    root.after(100, check_id)

    # Start the HRI data polling thread.
    threading.Thread(target=poll_hri_data, daemon=True).start()

    root.mainloop()

if __name__ == "__main__":
    main()
