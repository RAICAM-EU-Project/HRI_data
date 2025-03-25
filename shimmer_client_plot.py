
import requests
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import argparse
import time

# === Constants ===
MAX_POINTS = 200
POLL_INTERVAL = 0.05  # seconds (20 Hz polling)

# === Data Buffers ===
gsr_data = deque(maxlen=MAX_POINTS)
ppg_data = deque(maxlen=MAX_POINTS)
timestamps = deque(maxlen=MAX_POINTS)

def fetch_data(server_url):
    try:
        response = requests.get(f"{server_url}/latest", timeout=1.0)
        if response.status_code == 200:
            return response.json()
    except requests.RequestException:
        pass
    return None

def main(server_url):
    fig, (ax1, ax2) = plt.subplots(2, 1)
    line1, = ax1.plot([], [], label='PPG (mV)')
    line2, = ax2.plot([], [], label='GSR (μS)', color='r')

    ax1.set_ylabel('PPG (mV)')
    ax2.set_ylabel('GSR (μS)')
    ax2.set_xlabel('Samples')
    ax1.grid(True)
    ax2.grid(True)
    ax1.legend()
    ax2.legend()

    def update(frame):
        data = fetch_data(server_url)
        if data:
            timestamps.append(data['timestamp'])
            gsr_data.append(data['ppg'])
            ppg_data.append(data['gsr'])

        x = list(range(len(gsr_data)))
        line1.set_data(x, list(ppg_data))
        line2.set_data(x, list(gsr_data))

        ax1.set_xlim(0, MAX_POINTS)
        ax2.set_xlim(0, MAX_POINTS)
        if ppg_data:
            ax1.set_ylim(min(ppg_data) - 50, max(ppg_data) + 50)
        if gsr_data:
            ax2.set_ylim(min(gsr_data) - 5, max(gsr_data) + 5)

        return line1, line2

    ani = FuncAnimation(fig, update, interval=POLL_INTERVAL * 1000, blit=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Receive and plot real-time GSR/PPG data from HTTP endpoint.")
    parser.add_argument("--url", type=str, default="http://localhost:8000", help="HTTP URL of the data server")
    args = parser.parse_args()

    print(f"Connecting to {args.url}/latest for real-time data...")
    main(args.url)
