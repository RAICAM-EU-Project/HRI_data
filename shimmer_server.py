import sys, struct, serial
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
from threading import Thread
import time
import argparse

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn
import asyncio

# === Constants ===
FRAMESIZE = 8
MAX_POINTS = 200

# === Globals ===
gsr_data = deque(maxlen=MAX_POINTS)
ppg_data = deque(maxlen=MAX_POINTS)
timestamps = deque(maxlen=MAX_POINTS)

latest_sample = {"timestamp": None, "gsr": None, "ppg": None}

ser = None
stop_flag = False
timestamp_start = None

# === FastAPI Setup ===
app = FastAPI()

@app.get("/latest")
async def get_latest():
    return JSONResponse(content=latest_sample)


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
    global stop_flag, timestamp_start, latest_sample
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

        # Extract data
        packettype = data[0]
        t0, t1, t2 = data[1:4]
        timestamp = t0 + t1 * 256 + t2 * 65536
        PPG_raw, GSR_raw = struct.unpack('HH', data[4:8])

        # GSR calculation
        Range = (GSR_raw >> 14) & 0x03
        Rf = [40.2, 287.0, 1000.0, 3300.0][Range]
        gsr_volts = (GSR_raw & 0x3fff) * (3.0 / 4095.0)
        GSR_ohm = Rf / ((gsr_volts / 0.5) - 1.0)
        GSR_muS = 1000000.0 / GSR_ohm

        # PPG calculation
        PPG_mv = PPG_raw * (3000.0 / 4095.0)

        # Time
        if timestamp_start is None:
            timestamp_start = timestamp
        timestamp_session = timestamp - timestamp_start

        # Save data
        timestamps.append(timestamp_session)
        gsr_data.append(GSR_muS)
        ppg_data.append(PPG_mv)

        latest_sample = {
            "timestamp": timestamp_session,
            "gsr": round(GSR_muS, 3),
            "ppg": round(PPG_mv, 3)
        }

        if output_file:
            with open(output_file, 'a') as f:
                f.write(f"{timestamp_session},{GSR_muS},{PPG_mv}\n")


def plot_data():
    fig, (ax1, ax2) = plt.subplots(2, 1)
    line1, = ax1.plot([], [], label='PPG (mV)')
    line2, = ax2.plot([], [], label='GSR (μS)', color='r')

    ax1.set_ylabel('PPG (mV)')
    ax2.set_ylabel('GSR (μS)')
    ax2.set_xlabel('Sample Index')
    ax1.grid(True)
    ax2.grid(True)
    ax1.legend()
    ax2.legend()

    def update(frame):
        if timestamps:
            x = list(range(len(timestamps)))
            line1.set_data(x, list(gsr_data))
            line2.set_data(x, list(ppg_data))
            ax1.set_xlim(0, MAX_POINTS)
            ax1.set_ylim(min(gsr_data, default=0) - 5, max(gsr_data, default=10) + 5)
            ax2.set_xlim(0, MAX_POINTS)
            ax2.set_ylim(min(ppg_data, default=0) - 50, max(ppg_data, default=1000) + 50)
        return line1, line2

    ani = FuncAnimation(fig, update, interval=30, blit=False)
    plt.tight_layout()
    plt.show()


def shimmer_main(shimmer_port, output_file=None):
    global ser, stop_flag

    try:
        ser = serial.Serial(shimmer_port, 115200, timeout=1)
        ser.flushInput()
        print(f"Connected to {shimmer_port}")
    except serial.SerialException as e:
        print(f"Could not open port {shimmer_port}: {e}")
        return

    # Configure Shimmer
    print("Configuring Shimmer...")
    ser.write(struct.pack('BBBB', 0x08, 0x04, 0x01, 0x00))  # Set sensors: GSR and PPG
    wait_for_ack()

    ser.write(struct.pack('BB', 0x5E, 0x01))  # Enable internal expansion board power
    wait_for_ack()

    sampling_freq = 200
    clock_wait = int((2 << 14) / sampling_freq)
    ser.write(struct.pack('<BH', 0x05, clock_wait))  # Set sampling rate
    wait_for_ack()

    ser.write(struct.pack('B', 0x07))  # Start streaming
    wait_for_ack()
    print("Streaming started.")

    # Start read thread
    reader = Thread(target=data_read_loop, args=(output_file,), daemon=True)
    reader.start()

    return reader  # So we can shut it down on exit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Read and broadcast GSR/PPG data from a Shimmer sensor.")
    parser.add_argument("shimmer_port", type=str, default="/dev/rfcomm13", help="Bluetooth serial port of Shimmer")
    parser.add_argument("--output", "-o", type=str, default=None, help="Optional output CSV file")
    parser.add_argument("--port", "-p", type=int, default=8000, help="HTTP server port")
    args = parser.parse_args()

    reader_thread = shimmer_main(args.shimmer_port, args.output)

    # Run the HTTP server (non-blocking)
    try:
        print(f"Starting HTTP server at http://localhost:{args.port}/latest")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        stop_flag = True
        if reader_thread:
            reader_thread.join()
        if ser:
            try:
                ser.write(struct.pack('B', 0x20))  # Stop streaming
                ser.reset_input_buffer()
                wait_for_ack()
                ser.close()
            except Exception:
                pass
        print("Shimmer disconnected.")
