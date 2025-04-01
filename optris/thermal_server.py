#!/usr/bin/env python3
import os
import time
import datetime
import threading
import logging
import ctypes as ct
from ctypes.util import find_library
import json
import numpy as np
import cv2
import dlib
from flask import Flask, Response, jsonify

# Set up logging for detailed output.
logging.basicConfig(level=logging.INFO)

# =============================================================================
# Define the structure for frame metadata from the Optris thermal camera.
# =============================================================================
class EvoIRFrameMetadata(ct.Structure):
    _fields_ = [
        ("counter", ct.c_uint),
        ("counterHW", ct.c_uint),
        ("timestamp", ct.c_longlong),
        ("timestampMedia", ct.c_longlong),
        ("flagState", ct.c_int),
        ("tempChip", ct.c_float),
        ("tempFlag", ct.c_float),
        ("tempBox", ct.c_float),
    ]

# =============================================================================
# Global variables for storing the latest image frame, raw image, and facial temperatures.
# =============================================================================
latest_frame = None           # Processed thermal view with ROIs (for /thermal)
latest_raw_frame = None       # Original unprocessed palette image (for /)
latest_temperatures = {}      # Dictionary mapping region names (e.g., "forehead") to temperature values
data_lock = threading.Lock()  # Lock to safely share data between threads

# =============================================================================
# Flask application for broadcasting data over HTTP.
# =============================================================================
app = Flask(__name__)

# -------------------- Streaming Generators --------------------

def generate_raw():
    """Generator that yields the raw palette image as an MJPEG stream."""
    while True:
        with data_lock:
            if latest_raw_frame is None:
                frame = None
            else:
                ret, jpeg = cv2.imencode('.jpg', latest_raw_frame)
                if not ret:
                    frame = None
                else:
                    frame = jpeg.tobytes()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)

def generate_thermal():
    """Generator that yields the processed thermal image (with overlays) as an MJPEG stream."""
    while True:
        with data_lock:
            if latest_frame is None:
                frame = None
            else:
                ret, jpeg = cv2.imencode('.jpg', latest_frame)
                if not ret:
                    frame = None
                else:
                    frame = jpeg.tobytes()
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)

def generate_temp():
    """Generator that yields the latest facial temperatures as server-sent events."""
    while True:
        with data_lock:
            data = latest_temperatures.copy()
        json_data = json.dumps(data)
        yield f"data: {json_data}\n\n"
        time.sleep(0.5)

# -------------------- Flask Routes --------------------

@app.route('/')
def raw_image_endpoint():
    """HTTP endpoint to serve the original (raw) palette image as a real-time MJPEG stream."""
    return Response(generate_raw(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/thermal')
def thermal_image():
    """HTTP endpoint to serve the processed thermal image view (with ROIs) as a real-time MJPEG stream."""
    return Response(generate_thermal(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/temp')
def temperature_data():
    """HTTP endpoint to serve the facial temperatures as real-time server-sent events (SSE)."""
    return Response(generate_temp(),
                    mimetype='text/event-stream')

# =============================================================================
# Define the RegionsOfInterest class to extract facial regions based on landmarks.
# This class uses pre-defined indices from a 68-point facial landmark model.
# =============================================================================
class RegionsOfInterest(object):
    def __init__(self, coords_x, coords_y):
        self.coords_x = coords_x
        self.coords_y = coords_y
        # Calculate eye distance as a reference measurement.
        self.eyes_dist = self.coords_x[45] - self.coords_x[36]
        # Define several regions.
        self.regions = {
            'face': self.define_entire_face(),
            'forehead': self.define_forehead(),
            'left_cheek': self.define_left_cheek(),
            'right_cheek': self.define_right_cheek(),
            'chin': self.define_chin(),
            'nose': self.define_nose()
        }

    def define_forehead(self):
        interm_point = self.coords_x[23] - self.coords_x[20]
        coord_x = self.coords_x[21]
        coord_y = self.coords_y[20] - interm_point / 2
        coord_x1 = self.coords_x[22]
        coord_y1 = self.coords_y[23] - interm_point / 4
        return [coord_x, coord_y, coord_x1, coord_y1]

    def define_left_cheek(self):
        coord_x = self.coords_x[4]
        coord_y = self.coords_y[14]
        coord_x1 = self.coords_x[6]
        coord_y1 = self.coords_y[13]
        return [coord_x, coord_y, coord_x1, coord_y1]

    def define_right_cheek(self):
        coord_x = self.coords_x[10]
        coord_y = self.coords_y[14]
        coord_x1 = self.coords_x[12]
        coord_y1 = self.coords_y[13]
        return [coord_x, coord_y, coord_x1, coord_y1]

    def define_chin(self):
        interm_point_chin = self.coords_x[9] - self.coords_x[7]
        coord_x = self.coords_x[7]
        coord_y = self.coords_y[7] - interm_point_chin / 2
        coord_x1 = self.coords_x[9]
        coord_y1 = self.coords_y[9] - interm_point_chin / 4
        return [coord_x, coord_y, coord_x1, coord_y1]

    def define_nose(self):
        coord_x = self.coords_x[32]
        coord_y = self.coords_y[29]
        coord_x1 = self.coords_x[34]
        coord_y1 = self.coords_y[30]
        return [coord_x, coord_y, coord_x1, coord_y1]

    def define_entire_face(self):
        coord_x = self.coords_x[2]
        coord_y = self.coords_y[0] - self.eyes_dist
        coord_x1 = self.coords_x[5]
        coord_y1 = self.coords_y[10] + self.eyes_dist
        return [coord_x, coord_y, coord_x1, coord_y1]

    def get_multiple_regions(self, regions):
        """Return a dictionary of selected regions."""
        selected = {}
        for reg in regions:
            if reg in self.regions:
                selected[reg] = self.regions[reg]
        return selected

# =============================================================================
# Capture loop: Initializes the camera, acquires images, processes face detection
# and ROI extraction, and updates global variables.
# =============================================================================
def capture_loop():
    global latest_frame, latest_raw_frame, latest_temperatures

    # -------------------------------------------------------------------------
    # Load the Optris thermal camera library.
    # -------------------------------------------------------------------------
    if os.name == 'nt':
        libir = ct.CDLL('.\\libirimager.dll')
    else:
        libir = ct.cdll.LoadLibrary(ct.util.find_library("irdirectsdk"))
    logging.info("Loaded thermal camera library: %s", libir)

    # -------------------------------------------------------------------------
    # Initialize file paths and variables.
    # -------------------------------------------------------------------------
    pathFormat = b'.'
    pathLog = b'logfilename'
    pathXml = b'./16070070.xml'
    palette_width = ct.c_int()
    palette_height = ct.c_int()
    thermal_width = ct.c_int()
    thermal_height = ct.c_int()
    serial = ct.c_ulong()

    # Initialize metadata structure.
    metadata = EvoIRFrameMetadata()

    # Initialize the camera.
    ret = libir.evo_irimager_usb_init(pathXml, pathFormat, pathLog)
    if ret != 0:
        logging.error("Failed to initialize thermal camera library: %s", ret)
        return

    # Retrieve and log the camera serial number.
    ret = libir.evo_irimager_get_serial(ct.byref(serial))
    logging.info("Camera Serial: %d", serial.value)

    # Get thermal image size.
    libir.evo_irimager_get_thermal_image_size(ct.byref(thermal_width), ct.byref(thermal_height))
    logging.info("Thermal Image Size: %d x %d", thermal_width.value, thermal_height.value)

    # Allocate a container for the raw thermal data.
    np_thermal = np.zeros([thermal_width.value * thermal_height.value], dtype=np.uint16)
    npThermalPointer = np_thermal.ctypes.data_as(ct.POINTER(ct.c_ushort))

    # Get palette image size (note that the width might differ due to stride alignment).
    libir.evo_irimager_get_palette_image_size(ct.byref(palette_width), ct.byref(palette_height))
    logging.info("Palette Image Size: %d x %d", palette_width.value, palette_height.value)

    # Allocate a container for the palette (visual) image.
    np_img = np.zeros([palette_width.value * palette_height.value * 3], dtype=np.uint8)
    npImagePointer = np_img.ctypes.data_as(ct.POINTER(ct.c_ubyte))

    # Optionally, control timestamp printing (disabled here).
    show_time_stamp = False

    # -------------------------------------------------------------------------
    # Initialize dlib models for face detection and landmark prediction.
    # Update these file paths as necessary.
    # -------------------------------------------------------------------------
    detector_path = './dlib_files/dlib_face_detector.svm'
    predictor_path = './dlib_files/dlib_landmark_predictor.dat'
    try:
        face_detector = dlib.simple_object_detector(detector_path)
        landmark_predictor = dlib.shape_predictor(predictor_path)
        logging.info("Dlib models loaded successfully.")
    except Exception as e:
        logging.error("Failed to load dlib models: %s", str(e))
        return

    # Variables for managing face detection timing.
    last_detection_time = time.time()
    redetect_interval = 5.0
    current_face_rect = None

    # -------------------------------------------------------------------------
    # Main capture and processing loop.
    # -------------------------------------------------------------------------
    while True:
        if show_time_stamp:
            time_stamp = datetime.datetime.now().strftime("%H:%M:%S %d %B %Y")
            logging.info("Timestamp: %s", time_stamp)

        # Acquire the thermal and palette images along with metadata.
        ret = libir.evo_irimager_get_thermal_palette_image_metadata(
            thermal_width, thermal_height, npThermalPointer,
            palette_width, palette_height, npImagePointer,
            ct.byref(metadata)
        )
        if ret != 0:
            logging.error("Error capturing image: %s", ret)
            continue

        # Process thermal data:
        #   - Reshape the raw data to a 2D array.
        #   - Convert raw values to temperature using the given formula.
        thermal_array = np_thermal.reshape((thermal_height.value, thermal_width.value)).astype(np.float32)
        temperature_map = (thermal_array / 10.0) - 100.0

        # Process palette image:
        #   - Reshape to form the color image.
        palette_image = np_img.reshape((palette_height.value, palette_width.value, 3))
        # Save a copy of the raw palette image (without overlays) for the root endpoint.
        raw_image = palette_image.copy()
        # Convert palette image from RGB to BGR for display and further processing.
        display_image = palette_image[:, :, ::-1].copy()

        # ---------------------------------------------------------------------
        # Face detection:
        # If no face is currently tracked or it has been too long, run detection.
        # ---------------------------------------------------------------------
        current_time = time.time()
        if current_face_rect is None or (current_time - last_detection_time > redetect_interval):
            gray = cv2.cvtColor(display_image, cv2.COLOR_BGR2GRAY)
            detections = face_detector(gray)
            if detections:
                current_face_rect = detections[0]
                last_detection_time = current_time
                logging.info("Face detected.")
            else:
                current_face_rect = None

        # ---------------------------------------------------------------------
        # If a face is detected, run landmark prediction and extract ROIs.
        # ---------------------------------------------------------------------
        temperatures = {}
        if current_face_rect is not None:
            try:
                shape = landmark_predictor(display_image, current_face_rect)
                landmarks_x = [p.x for p in shape.parts()]
                landmarks_y = [p.y for p in shape.parts()]

                # Optionally draw the detected landmarks.
                for (x, y) in zip(landmarks_x, landmarks_y):
                    cv2.circle(display_image, (x, y), 1, (255, 0, 0), 2)

                # Define ROIs using the facial landmarks.
                roi_extractor = RegionsOfInterest(landmarks_x, landmarks_y)
                selected_regions = roi_extractor.get_multiple_regions(['forehead', 'left_cheek', 'right_cheek', 'nose'])

                # Compute scaling factors to map ROI coordinates from the palette image to the thermal image.
                scale_x = thermal_width.value / palette_width.value
                scale_y = thermal_height.value / palette_height.value

                # For each region, draw the ROI and compute the average temperature.
                for region_name, coords in selected_regions.items():
                    x_left, y_top, x_right, y_bottom = map(int, coords)
                    cv2.rectangle(display_image, (x_left, y_top), (x_right, y_bottom), (0, 255, 0), 2)
                    
                    # Map the coordinates to thermal image space.
                    tx_left = int(x_left * scale_x)
                    ty_top = int(y_top * scale_y)
                    tx_right = int(x_right * scale_x)
                    ty_bottom = int(y_bottom * scale_y)
                    
                    # Ensure the ROI is within bounds.
                    tx_left = max(0, min(tx_left, thermal_width.value - 1))
                    ty_top = max(0, min(ty_top, thermal_height.value - 1))
                    tx_right = max(0, min(tx_right, thermal_width.value))
                    ty_bottom = max(0, min(ty_bottom, thermal_height.value))
                    
                    # Extract the temperature data in the ROI and compute its average.
                    if tx_right > tx_left and ty_bottom > ty_top:
                        roi_temp = temperature_map[ty_top:ty_bottom, tx_left:tx_right]
                        avg_temp = float(np.mean(roi_temp))
                        temperatures[region_name] = avg_temp
                        # Annotate the display image with the temperature value.
                        cv2.putText(display_image, f"{avg_temp:.1f}C", (x_left, y_top - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    else:
                        temperatures[region_name] = None
            except Exception as e:
                logging.error("Error in facial temperature extraction: %s", str(e))
                current_face_rect = None  # Reset face detection on error

        # ---------------------------------------------------------------------
        # Update global variables (with thread safety) for the web server.
        # ---------------------------------------------------------------------
        with data_lock:
            latest_raw_frame = raw_image.copy()  # Original palette image (for /)
            latest_frame = display_image.copy()  # Processed image with overlays (for /thermal)
            latest_temperatures = temperatures.copy()

        # Brief sleep to control the loop frequency.
        time.sleep(0.02)

    # Clean shutdown: Terminate the camera (this line is reached only on exit).
    libir.evo_irimager_terminate()
    logging.info("Thermal camera terminated.")

def start_capture_thread():
    """Start the thermal capture and processing loop in a separate daemon thread."""
    thread = threading.Thread(target=capture_loop, daemon=True)
    thread.start()
    return thread

# =============================================================================
# Main entry point: start the capture thread and launch the Flask server.
# =============================================================================
if __name__ == '__main__':
    # Start the background thread for camera acquisition and processing.
    capture_thread = start_capture_thread()
    logging.info("Starting Flask server on http://localhost:8080")
    # Run the Flask app; it will serve the /, /thermal, and /temp endpoints.
    app.run(host='0.0.0.0', port=8080)
