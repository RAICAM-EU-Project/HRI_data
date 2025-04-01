#!/usr/bin/env python3
import os
import time
import datetime
import threading
import logging
import ctypes as ct
from ctypes.util import find_library
import json
import struct
import numpy as np
import cv2
import dlib
import mediapipe as mp
import serial
from flask import Flask, Response, jsonify

# -------------------- Logging --------------------
logging.basicConfig(level=logging.INFO)

# -------------------- Thermal Camera Globals --------------------
latest_frame = None           # Processed thermal view with ROIs (for /thermal)
latest_raw_frame = None       # Original unprocessed palette image (for /)
latest_temperatures = {}      # Dictionary mapping region names to temperature values
last_valid_temperatures = {}  # Stores last non-empty, valid temperature data
data_lock = threading.Lock()  # Lock to safely share thermal camera data

# -------------------- Blink Detection Globals --------------------
latest_blink_frame = None     # Processed blink image (with blink rate overlay)
latest_blink_rate = 0.0       # Weighted blink rate (blinks per minute)
blink_lock = threading.Lock() # Lock to safely share blink detection data

# -------------------- Shimmer Globals --------------------
latest_shimmer_sample = {"timestamp": None, "gsr": None, "ppg": None}
shimmer_lock = threading.Lock()  # Lock to safely share shimmer data
ser = None
stop_flag = False
timestamp_start = None

# =============================================================================
# Thermal Camera: EvoIRFrameMetadata definition
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
# Flask Application Setup
# =============================================================================
app = Flask(__name__)

# -------------------- MJPEG & SSE Generators --------------------
def generate_raw():
    """Generator that yields the raw thermal palette image as MJPEG stream."""
    while True:
        with data_lock:
            if latest_raw_frame is None:
                frame = None
            else:
                ret, jpeg = cv2.imencode('.jpg', latest_raw_frame)
                frame = jpeg.tobytes() if ret else None
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)

def generate_thermal():
    """Generator that yields the processed thermal image (with overlays) as MJPEG stream."""
    while True:
        with data_lock:
            if latest_frame is None:
                frame = None
            else:
                ret, jpeg = cv2.imencode('.jpg', latest_frame)
                frame = jpeg.tobytes() if ret else None
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)

def generate_temp():
    """Generator that yields the latest facial temperatures as SSE events."""
    while True:
        with data_lock:
            data = latest_temperatures.copy()
        json_data = json.dumps(data)
        yield f"data: {json_data}\n\n"
        time.sleep(0.5)

def generate_blink():
    """Generator that yields the blink detection image as MJPEG stream."""
    while True:
        with blink_lock:
            if latest_blink_frame is None:
                frame = None
            else:
                ret, jpeg = cv2.imencode('.jpg', latest_blink_frame)
                frame = jpeg.tobytes() if ret else None
        if frame is not None:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.03)

# -------------------- Flask Routes --------------------
@app.route('/')
def raw_image_endpoint():
    """Serve the original raw thermal palette image as real-time MJPEG."""
    return Response(generate_raw(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/thermal')
def thermal_image():
    """Serve the processed thermal image (with ROIs) as real-time MJPEG."""
    return Response(generate_thermal(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/temp')
def temperature_data():
    """Serve the facial temperature data as real-time SSE."""
    return Response(generate_temp(),
                    mimetype='text/event-stream')

@app.route('/blink')
def blink_image():
    """Serve the blink detection video as real-time MJPEG."""
    return Response(generate_blink(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/shimmer')
def shimmer_endpoint():
    """Serve the latest shimmer (GSR/PPG) data as JSON."""
    with shimmer_lock:
        data = latest_shimmer_sample.copy()
    return jsonify(data)

@app.route('/data')
def combined_data():
    """
    Serve combined data in real time:
      - Shimmer: GSR & PPG
      - Face temperatures (4 regions)
      - Blink rate
    """
    with data_lock, blink_lock, shimmer_lock:
        combined = {
            "shimmer": latest_shimmer_sample,
            "face_temperatures": latest_temperatures,
            "blink_rate": latest_blink_rate
        }
    return jsonify(combined)

# =============================================================================
# RegionsOfInterest Class (for facial ROI extraction)
# =============================================================================
class RegionsOfInterest(object):
    def __init__(self, coords_x, coords_y):
        self.coords_x = coords_x
        self.coords_y = coords_y
        self.eyes_dist = self.coords_x[45] - self.coords_x[36]
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
        selected = {}
        for reg in regions:
            if reg in self.regions:
                selected[reg] = self.regions[reg]
        return selected

# =============================================================================
# Thermal Camera Capture Loop
# =============================================================================
def capture_loop():
    global latest_frame, latest_raw_frame, latest_temperatures, last_valid_temperatures

    if os.name == 'nt':
        libir = ct.CDLL('.\\libirimager.dll')
    else:
        libir = ct.cdll.LoadLibrary(ct.util.find_library("irdirectsdk"))
    logging.info("Loaded thermal camera library: %s", libir)

    pathFormat = b'.'
    pathLog = b'logfilename'
    pathXml = b'./16070070.xml'
    palette_width = ct.c_int()
    palette_height = ct.c_int()
    thermal_width = ct.c_int()
    thermal_height = ct.c_int()
    serial_num = ct.c_ulong()

    metadata = EvoIRFrameMetadata()

    ret = libir.evo_irimager_usb_init(pathXml, pathFormat, pathLog)
    if ret != 0:
        logging.error("Failed to initialize thermal camera library: %s", ret)
        return

    ret = libir.evo_irimager_get_serial(ct.byref(serial_num))
    logging.info("Camera Serial: %d", serial_num.value)

    libir.evo_irimager_get_thermal_image_size(ct.byref(thermal_width), ct.byref(thermal_height))
    logging.info("Thermal Image Size: %d x %d", thermal_width.value, thermal_height.value)

    np_thermal = np.zeros([thermal_width.value * thermal_height.value], dtype=np.uint16)
    npThermalPointer = np_thermal.ctypes.data_as(ct.POINTER(ct.c_ushort))

    libir.evo_irimager_get_palette_image_size(ct.byref(palette_width), ct.byref(palette_height))
    logging.info("Palette Image Size: %d x %d", palette_width.value, palette_height.value)

    np_img = np.zeros([palette_width.value * palette_height.value * 3], dtype=np.uint8)
    npImagePointer = np_img.ctypes.data_as(ct.POINTER(ct.c_ubyte))

    show_time_stamp = False

    detector_path = './dlib_files/dlib_face_detector.svm'
    predictor_path = './dlib_files/dlib_landmark_predictor.dat'
    try:
        face_detector = dlib.simple_object_detector(detector_path)
        landmark_predictor = dlib.shape_predictor(predictor_path)
        logging.info("Dlib models loaded successfully.")
    except Exception as e:
        logging.error("Failed to load dlib models: %s", str(e))
        return

    last_detection_time = time.time()
    redetect_interval = 5.0
    current_face_rect = None

    while True:
        if show_time_stamp:
            time_stamp = datetime.datetime.now().strftime("%H:%M:%S %d %B %Y")
            logging.info("Timestamp: %s", time_stamp)

        ret = libir.evo_irimager_get_thermal_palette_image_metadata(
            thermal_width, thermal_height, npThermalPointer,
            palette_width, palette_height, npImagePointer,
            ct.byref(metadata)
        )
        if ret != 0:
            logging.error("Error capturing image: %s", ret)
            continue

        thermal_array = np_thermal.reshape((thermal_height.value, thermal_width.value)).astype(np.float32)
        temperature_map = (thermal_array / 10.0) - 100.0

        palette_image = np_img.reshape((palette_height.value, palette_width.value, 3))
        raw_image = palette_image.copy()
        display_image = palette_image[:, :, ::-1].copy()

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

        temperatures = {}
        if current_face_rect is not None:
            try:
                shape = landmark_predictor(display_image, current_face_rect)
                landmarks_x = [p.x for p in shape.parts()]
                landmarks_y = [p.y for p in shape.parts()]

                for (x, y) in zip(landmarks_x, landmarks_y):
                    cv2.circle(display_image, (x, y), 1, (255, 0, 0), 2)

                roi_extractor = RegionsOfInterest(landmarks_x, landmarks_y)
                selected_regions = roi_extractor.get_multiple_regions(['forehead', 'left_cheek', 'right_cheek', 'nose'])

                scale_x = thermal_width.value / palette_width.value
                scale_y = thermal_height.value / palette_height.value

                for region_name, coords in selected_regions.items():
                    x_left, y_top, x_right, y_bottom = map(int, coords)
                    cv2.rectangle(display_image, (x_left, y_top), (x_right, y_bottom), (0, 255, 0), 2)

                    tx_left = int(x_left * scale_x)
                    ty_top = int(y_top * scale_y)
                    tx_right = int(x_right * scale_x)
                    ty_bottom = int(y_bottom * scale_y)

                    tx_left = max(0, min(tx_left, thermal_width.value - 1))
                    ty_top = max(0, min(ty_top, thermal_height.value - 1))
                    tx_right = max(0, min(tx_right, thermal_width.value))
                    ty_bottom = max(0, min(ty_bottom, thermal_height.value))

                    if tx_right > tx_left and ty_bottom > ty_top:
                        roi_temp = temperature_map[ty_top:ty_bottom, tx_left:tx_right]
                        avg_temp = float(np.mean(roi_temp))
                        temperatures[region_name] = avg_temp
                        cv2.putText(display_image, f"{avg_temp:.1f}C", (x_left, y_top - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                    else:
                        temperatures[region_name] = None
            except Exception as e:
                logging.error("Error in facial temperature extraction: %s", str(e))
                current_face_rect = None

        with data_lock:
            latest_raw_frame = raw_image.copy()
            latest_frame = display_image.copy()
            # If temperatures dict is empty or has any None, fallback to last valid temperatures.
            valid = True
            if not temperatures:
                valid = False
            else:
                for val in temperatures.values():
                    if val is None:
                        valid = False
                        break
            if valid:
                last_valid_temperatures = temperatures.copy()
            else:
                if last_valid_temperatures:
                    temperatures = last_valid_temperatures.copy()
            latest_temperatures = temperatures.copy()

        time.sleep(0.03)

    libir.evo_irimager_terminate()
    logging.info("Thermal camera terminated.")

# =============================================================================
# Blink Detection Loop (using MediaPipe)
# =============================================================================
def blink_loop():
    global latest_blink_frame, latest_blink_rate
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    ear_threshold = 0.25
    consecutive_frames = 3
    blink_frames = 0
    was_closed = False
    blink_counter = 0
    start_time = time.time()
    blink_timestamps_sliding = []
    window2 = 5.0
    t_last_blink_instant = None
    instant_blink_rate = 0.0

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Cannot open webcam for blink detection.")
        return

    def get_eye_landmarks(landmarks, indices, frame_width, frame_height):
        return np.array([(int(landmarks.landmark[i].x * frame_width),
                          int(landmarks.landmark[i].y * frame_height))
                          for i in indices])

    def calculate_ear(eye):
        A = np.linalg.norm(eye[1] - eye[5])
        B = np.linalg.norm(eye[2] - eye[4])
        C = np.linalg.norm(eye[0] - eye[3])
        return (A + B) / (2.0 * C) if C != 0 else 0

    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to grab blink frame")
            time.sleep(0.033)
            continue
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        current_time_secs = time.time()
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape
                left_eye = get_eye_landmarks(face_landmarks, [33, 160, 158, 133, 153, 144], w, h)
                right_eye = get_eye_landmarks(face_landmarks, [362, 385, 387, 263, 373, 380], w, h)
                left_ear = calculate_ear(left_eye)
                right_ear = calculate_ear(right_eye)
                avg_ear = (left_ear + right_ear) / 2.0

                if avg_ear < ear_threshold:
                    blink_frames += 1
                else:
                    if blink_frames >= consecutive_frames and not was_closed:
                        blink_counter += 1
                        blink_timestamps_sliding.append(current_time_secs)
                        if t_last_blink_instant is not None:
                            dt = current_time_secs - t_last_blink_instant
                            instant_blink_rate = 60.0 / dt if dt > 0 else 0.0
                        t_last_blink_instant = current_time_secs
                        was_closed = True
                    blink_frames = 0
                    was_closed = False

                for pt in left_eye:
                    cv2.circle(frame, pt, 2, (0, 255, 0), -1)
                for pt in right_eye:
                    cv2.circle(frame, pt, 2, (0, 255, 0), -1)
                cv2.putText(frame, f"EAR: {avg_ear:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elapsed_time_secs = current_time_secs - start_time
        elapsed_time_minutes = elapsed_time_secs / 60.0
        blink_rate_option1 = blink_counter / elapsed_time_minutes if elapsed_time_minutes > 0 else 0.0
        blink_timestamps_sliding = [ts for ts in blink_timestamps_sliding if current_time_secs - ts <= window2]
        blink_rate_option2 = len(blink_timestamps_sliding) * (60.0 / window2)
        weighted_blink_rate = (0.1 * blink_rate_option1 +
                               0.7 * blink_rate_option2 +
                               0.2 * instant_blink_rate)
        cv2.putText(frame, f"Weighted Blink Rate: {weighted_blink_rate:.2f} blinks/min", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        with blink_lock:
            latest_blink_frame = frame.copy()
            latest_blink_rate = weighted_blink_rate

        time.sleep(0.033)

    cap.release()
    cv2.destroyAllWindows()

# =============================================================================
# Shimmer Data Functions
# =============================================================================
def wait_for_ack():
    ack = struct.pack('B', 0xff)
    discarded = 0
    while True:
        ddata = ser.read(1)
        if not ddata:
            continue
        if ddata == ack:
            print(f"ACK byte received: {ddata[0]:02x}")
            if discarded > 0:
                print(f"Note: {discarded} non-ACK bytes discarded before ACK.")
            break
        else:
            discarded += 1

def data_read_loop(output_file=None):
    global stop_flag, timestamp_start, latest_shimmer_sample
    FRAMESIZE = 8
    ddata = b""
    numbytes = 0
    while not stop_flag:
        while numbytes < FRAMESIZE and not stop_flag:
            try:
                ddata += ser.read(FRAMESIZE - numbytes)
                numbytes = len(ddata)
            except serial.SerialException:
                print("Serial read error.")
                stop_flag = True
                return
        if len(ddata) < FRAMESIZE:
            continue
        data = ddata[0:FRAMESIZE]
        ddata = ddata[FRAMESIZE:]
        numbytes = len(ddata)
        packettype = data[0]
        t0, t1, t2 = data[1:4]
        timestamp = t0 + t1 * 256 + t2 * 65536
        PPG_raw, GSR_raw = struct.unpack('HH', data[4:8])
        Range = (GSR_raw >> 14) & 0x03
        Rf = [40.2, 287.0, 1000.0, 3300.0][Range]
        gsr_volts = (GSR_raw & 0x3fff) * (3.0 / 4095.0)
        GSR_ohm = Rf / ((gsr_volts / 0.5) - 1.0)
        GSR_muS = 1000000.0 / GSR_ohm
        PPG_mv = PPG_raw * (3000.0 / 4095.0)
        if timestamp_start is None:
            timestamp_start = timestamp
        timestamp_session = timestamp - timestamp_start
        with shimmer_lock:
            latest_shimmer_sample = {
                "timestamp": timestamp_session,
                "gsr": round(GSR_muS, 3),
                "ppg": round(PPG_mv, 3)
            }
        if output_file:
            with open(output_file, 'a') as f:
                f.write(f"{timestamp_session},{GSR_muS},{PPG_mv}\n")

def shimmer_main(shimmer_port, output_file=None):
    global ser, stop_flag
    try:
        ser = serial.Serial(shimmer_port, 115200, timeout=1)
        ser.flushInput()
        print(f"Connected to {shimmer_port}")
    except serial.SerialException as e:
        print(f"Could not open port {shimmer_port}: {e}")
        return None
    print("Configuring Shimmer...")
    ser.write(struct.pack('BBBB', 0x08, 0x04, 0x01, 0x00))
    wait_for_ack()
    ser.write(struct.pack('BB', 0x5E, 0x01))
    wait_for_ack()
    sampling_freq = 200
    clock_wait = int((2 << 14) / sampling_freq)
    ser.write(struct.pack('<BH', 0x05, clock_wait))
    wait_for_ack()
    ser.write(struct.pack('B', 0x07))
    wait_for_ack()
    print("Shimmer streaming started.")
    reader = threading.Thread(target=data_read_loop, args=(output_file,), daemon=True)
    reader.start()
    return reader

# =============================================================================
# Thread Starters
# =============================================================================
def start_capture_thread():
    thread = threading.Thread(target=capture_loop, daemon=True)
    thread.start()
    return thread

def start_blink_thread():
    thread = threading.Thread(target=blink_loop, daemon=True)
    thread.start()
    return thread

# =============================================================================
# Main Entry Point
# =============================================================================
if __name__ == '__main__':
    capture_thread = start_capture_thread()
    blink_thread = start_blink_thread()
    
    # Start shimmer data thread. Adjust shimmer_port as needed.
    shimmer_port = "/dev/rfcomm13"  # Change as appropriate for your system.
    shimmer_thread = shimmer_main(shimmer_port, output_file=None)
    
    logging.info("Starting Flask server on http://localhost:8080")
    app.run(host='0.0.0.0', port=8080)
