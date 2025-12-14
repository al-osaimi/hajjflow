# HajjFlow: Edge Crowd Monitoring Dashboard (YOLO + CSRNet)

HajjFlow is a prototype system for real-time crowd detection, counting, and crowd-pressure monitoring using edge devices that stream metrics to a central dashboard for live visualization and decision support.

The project is aligned with **Saudi Vision 2030** and **NSDAI** by demonstrating how **computer vision + edge AI** can improve Hajj crowd safety, reduce reliance on manual CCTV monitoring, and provide actionable insights in real time.

---

## Features

### Edge Client (Python)
- Processes a video stream (file/webcam) and outputs crowd metrics.
- **YOLO** (person detection) for normal density scenes.
- **CSRNet** (density estimation) for extremely dense scenes (fallback mode).
- Sends periodic **Socket.IO heartbeat** messages to the server.
- Supports multiple simulated devices via environment variables (deviceId, zone, lat/lng, etc.).
- Debug mode (optional) for verbose logs and frame sampling.

### Dashboard Server (Node.js + Express + Socket.IO + EJS)
- Receives heartbeats and maintains device registry with online/offline status.
- Live UI: device list, zone grouping, KPIs, charts, and interactive map.
- REST endpoints for snapshot and per-device detail (useful for demos/reports).
- Offline logic (TTL) to mark devices offline when heartbeats stop.

---

## System Design Overview

### High-level Architecture

![High_Level_Design](High_Level_Design.png)

---

## Data Flow (Heartbeat Payload)

![DataFlow](DataFlow.png)


---

## Sequence Diagram (Edge → Server → Dashboard)

![Sequence_Diagram](Sequence_Diagram.png)

---

## Repository Structure (suggested)

```text
.
├─ index.js
├─ package.json
├─ views/
│  └─ dashboard.ejs
├─ public/
│  └─ style.css
└─ edge_client/
   ├─ edge_client.py
   ├─ videos/
   │  ├─ tawaf.mp4
   │  ├─ jamarat.mp4
   │  └─ arafat.mp4
   └─ weights/
      └─ csrnet_shanghaitech.pth
```

---

## Requirements

### OS
- Ubuntu 20.04+ (tested on 22.04/24.04)
- Raspberry Pi / Jetson

### Server
- Node.js 18+ (recommended: 20 LTS)
- npm

### Edge Client
- Python 3.10+ (3.11/3.12 ok)
- Virtual environment (venv)
- For CPU-only testing: no CUDA required

---

## Installation

### 1) Clone and install server dependencies

```bash
git clone https://github.com/al-osaimi/hajjflow
cd hajjflow
npm install
```

### 2) Create Python venv for the edge client

```bash
cd edge_client
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

Install Python packages:

```bash
pip install opencv-python numpy python-socketio requests torch ultralytics
```

> If you’re CPU-only, install CPU Torch from official instructions for your Python version.

---

## Model Weights

### YOLO
Use a known valid Ultralytics model name, e.g.:
- `yolov8n.pt`
- `yolo11.pt`

```bash
YOLO_MODEL=yolov8n.pt
```

### CSRNet weights
Place CSRNet weights at:
```text
edge_client/weights/csrnet_shanghaitech.pth
```

---

## Running the Project

### A) Start the Dashboard Server
```bash
npm start
```
Open: http://localhost:5000

### B) Run One Edge Device (manual)
```bash
cd edge_client
source .venv/bin/activate

BASE_URL=http://localhost:5000 \
DEVICE_ID=EDGE-TAWAF \
DEVICE_NAME="Tawaf Camera" \
ZONE="Masjid Al-Haram (Tawaf)" \
LAT=21.422487 LNG=39.826206 \
VIDEO_PATH=videos/tawaf.mp4 \
YOLO_MODEL=yolov8n.pt \
ENABLE_CSRNET=1 \
CSRNET_WEIGHTS=weights/csrnet_shanghaitech.pth \
python3 edge_client.py
```

### C) Run 3 Simulated Edge Devices (recommended demo)
```bash
npm run dev
npm run edge:all
```

---

## Configuration (Environment Variables)

### Server
| Variable | Default | Meaning |
|---|---:|---|
| PORT | 5000 | Server port |
| OFFLINE_TTL_MS | 30000 | Device offline threshold |
| HISTORY_MAX | 120 | Max history points |

### Edge Client
| Variable | Example | Meaning |
|---|---|---|
| BASE_URL | http://localhost:5000 | Server URL |
| DEVICE_ID | EDGE-TAWAF | Unique device ID |
| DEVICE_NAME | Tawaf Camera | Friendly name |
| ZONE | Masjid Al-Haram (Tawaf) | Location label |
| LAT/LNG | 21.422487/39.826206 | Map location |
| VIDEO_PATH | videos/tawaf.mp4 | Input video |
| YOLO_MODEL | yolov8n.pt | YOLO weights |
| CONF | 0.20 | Detection confidence |
| IMGSZ | 640 | Inference size |
| ENABLE_CSRNET | 1 | Enable CSRNet |
| CSRNET_WEIGHTS | weights/...pth | CSRNet weights |
| HEARTBEAT_SEC | 2.0 | Send interval |

---

## Dashboard Behavior
- Devices appear **Online** if last heartbeat is within `OFFLINE_TTL_MS`.
- Map markers:
  - **LOW** (green), **MEDIUM** (amber), **HIGH** (red), **OFFLINE** (gray)
- Clicking a device card or marker opens the drawer with:
  - Count, pressure, method, YOLO FPS, IoU ratio, dense-scene, CSR time/error, and history chart.

---

## Troubleshooting

### 1) All `count=0` and `iou_ratio=0.0`
- Wrong model name or model not downloaded
- Confidence too high (`CONF=0.15–0.20`)
- Video contains no visible people
- Person class filter enabled (COCO person = class 0)

### 2) CSRNet weights fail to load
- Architecture mismatch. Use CSRNet definition matching the weight source repo.

### 3) Device stuck online after edge stops
- TTL logic handles offline after `OFFLINE_TTL_MS`.
- Optional purge logic can remove devices permanently.

### 4) Socket disconnect / dashboard not updating
- REST fallback polling is enabled.
- Check server logs and browser DevTools.

---

## Ethical & Practical Considerations
- **Privacy**: edge-only processing (no raw video transmission).
- **Bias**: validate models across lighting, angles, and demographics.
- **Security**: use TLS, device authentication, signed telemetry in production.
- **Feasibility**: match hardware to compute cost (YOLO vs CSRNet).

---

## Roadmap (Optional Enhancements)
- Heatmap overlay on the map
- Multi-camera aggregation per zone
- Device authentication
- Persistent storage for analytics
