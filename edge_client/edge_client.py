#!/usr/bin/env python3
"""
HajjFlow Edge Client - Crowd Density Monitoring

This script runs on edge devices (e.g., Jetson, Raspberry Pi, PC) to:
- Detect and count people using YOLO11
- Fall back to CSRNet density estimation in dense/occluded scenes
- Send real-time heartbeats via Socket.IO to central server
- Provide rich debugging tools: overlay, logging, frame/payload saving, live view

Author: Enhanced version with robust debug & graceful shutdown
Date: December 2025
"""

import os
import cv2
import json
import time
import psutil
import signal
import traceback
import requests
import numpy as np
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from ultralytics import YOLO
import socketio

# =========================
# CONFIGURATION (via ENV)
# =========================
BASE_URL = os.getenv("BASE_URL", "http://localhost:5000/")
DEVICE_ID = os.getenv("DEVICE_ID", "EDGE-001")
DEVICE_NAME = os.getenv("DEVICE_NAME", DEVICE_ID)

ZONE = os.getenv("ZONE", "Masjid Al-Haram")
LAT = float(os.getenv("LAT", "21.4225"))
LNG = float(os.getenv("LNG", "39.8262"))

VIDEO_PATH = os.getenv("VIDEO_PATH", "video.mp4")

YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8n.pt")
CONF = float(os.getenv("CONF", "0.15"))
IOU = float(os.getenv("IOU", "0.45"))
IMGSZ = int(os.getenv("IMGSZ", "640"))

# CSRNet Settings
CSRNET_WEIGHTS = os.getenv("CSRNET_WEIGHTS", "edge_client/weights/weights.pth")
ENABLE_CSRNET = os.getenv("ENABLE_CSRNET", "1") == "1"
TORCH_DEVICE = os.getenv("TORCH_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")

# Dense Scene Heuristics
DENSE_COUNT_THR = int(os.getenv("DENSE_COUNT_THR", "15"))           # Many detections → likely crowded
DENSE_IOU_RATIO_THR = float(os.getenv("DENSE_IOU_RATIO_THR", "0.35"))  # High overlap → occlusion

# Crowd Pressure Thresholds (tune per camera view!)
LOW_THR = float(os.getenv("LOW_THR", "20"))
MID_THR = float(os.getenv("MID_THR", "25"))

HEARTBEAT_SEC = float(os.getenv("HEARTBEAT_SEC", "2.0"))

# =========================
# DEBUG SYSTEM (ENHANCED)
# =========================
DEBUG = os.getenv("DEBUG", "0") == "1"
DEBUG_LEVEL = os.getenv("DEBUG_LEVEL", "INFO").upper()  # INFO, DEBUG, TRACE
DEBUG_VIEW = os.getenv("DEBUG_VIEW", "0") == "1"         # Show OpenCV window
DEBUG_LOG = os.getenv("DEBUG_LOG", "")                  # Optional log file path
DEBUG_SAVE_FRAMES = os.getenv("DEBUG_SAVE_FRAMES", "0") == "1"
DEBUG_SAVE_PAYLOADS = os.getenv("DEBUG_SAVE_PAYLOADS", "0") == "1"
DEBUG_OUT_DIR = os.getenv("DEBUG_OUT_DIR", "debug_out")
SHOW_EVERY_N = max(1, int(os.getenv("SHOW_EVERY_N", "1")))  # Show only every Nth frame

# Logging levels
LEVELS = {"TRACE": 5, "DEBUG": 10, "INFO": 20, "WARN": 30, "ERROR": 40}
CUR_LEVEL = LEVELS.get(DEBUG_LEVEL, 20)

# Global shutdown flag for graceful exit
SHUTDOWN_REQUESTED = False

def signal_handler(sig, frame):
    """Handle Ctrl+C gracefully"""
    global SHUTDOWN_REQUESTED
    print("\n[INFO] Shutdown requested (Ctrl+C). Cleaning up...")
    SHUTDOWN_REQUESTED = True

signal.signal(signal.SIGINT, signal_handler)

def _write_log(line: str):
    if DEBUG_LOG:
        try:
            with open(DEBUG_LOG, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            print(f"[ERROR] Failed to write debug log: {e}")

def log(level: str, msg: str):
    """Centralized logging with level control"""
    lv = LEVELS.get(level.upper(), 20)
    if not DEBUG and level.upper() in ("DEBUG", "TRACE"):
        return
    if lv < CUR_LEVEL:
        return
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
    line = f"[{level.upper()}] {timestamp} | {msg}"
    print(line)
    _write_log(line)

def ensure_debug_dir():
    """Create debug output directory if needed"""
    if (DEBUG_SAVE_FRAMES or DEBUG_SAVE_PAYLOADS) and not os.path.exists(DEBUG_OUT_DIR):
        os.makedirs(DEBUG_OUT_DIR, exist_ok=True)
        log("INFO", f"Created debug output directory: {DEBUG_OUT_DIR}")

# =========================
# CSRNet Model Definition
# =========================
class CSRNet(nn.Module):
    """
    Minimal CSRNet implementation: VGG-like frontend + dilated backend for density map regression.
    Outputs a single-channel density map → integrate to get crowd count.
    """
    def __init__(self):
        super().__init__()
        # Frontend: Feature extraction (similar to VGG)
        self.frontend = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1), nn.ReLU(inplace=True),
        )
        # Backend: Dilated convolutions to increase receptive field without downsampling
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
        )
        self.output_layer = nn.Conv2d(64, 1, 1)  # Density map

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x


def load_csrnet(weights_path: str) -> Optional[nn.Module]:
    """Load CSRNet weights with robust error handling and detailed debug feedback"""
    if not os.path.exists(weights_path):
        log("WARN", f"[CSRNet] Weights file not found: {weights_path}")
        return None

    log("INFO", f"[CSRNet] Loading weights from {weights_path} → {TORCH_DEVICE}")
    model = CSRNet().to(TORCH_DEVICE)

    try:
        state_dict = torch.load(weights_path, map_location=TORCH_DEVICE, weights_only=False)
    except Exception as e:
        log("ERROR", f"[CSRNet] Failed to load weights file: {e}")
        if DEBUG:
            log("DEBUG", traceback.format_exc())
        return None

    # Handle wrapped models (DataParallel, checkpoint dicts)
    if isinstance(state_dict, dict) and "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    if isinstance(state_dict, dict):
        # Remove 'module.' prefix if present
        state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    try:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        model.eval()

        if missing:
            log("WARN", f"[CSRNet] {len(missing)} missing keys (partial load)")
            if DEBUG:
                log("DEBUG", f"Missing keys sample: {missing[:8]}")
        if unexpected:
            log("WARN", f"[CSRNet] {len(unexpected)} unexpected keys")
            if DEBUG:
                log("DEBUG", f"Unexpected keys sample: {unexpected[:8]}")

        if not missing and not unexpected:
            log("INFO", "[CSRNet] Weights loaded perfectly (strict match)")
        else:
            log("INFO", "[CSRNet] Partial load completed – verify output quality!")
        return model

    except Exception as e:
        log("ERROR", f"[CSRNet] Failed to apply weights: {e}")
        if DEBUG:
            log("DEBUG", traceback.format_exc())
        return None


# =========================
# Utility Functions
# =========================
def crowd_pressure_label(count: float) -> tuple[str, tuple[int, int, int]]:
    """Return pressure level and corresponding BGR color for visualization"""
    if count < LOW_THR:
        return "LOW", (0, 255, 0)      # Green
    elif count < MID_THR:
        return "MEDIUM", (0, 255, 255) # Yellow
    else:
        return "HIGH", (0, 0, 255)    # Red

def compute_pairwise_iou_ratio(boxes_xyxy: np.ndarray) -> float:
    """
    Estimate scene density via average pairwise IoU among top detections.
    High average IoU → heavy occlusion → YOLO likely undercounts.
    """
    n = boxes_xyxy.shape[0]
    if n < 2:
        return 0.0

    # Limit computation for performance
    max_boxes = min(n, 120)
    b = boxes_xyxy[:max_boxes]
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    areas = np.maximum(x2 - x1, 0) * np.maximum(y2 - y1, 0)

    hits = 0
    total_pairs = 0
    for i in range(max_boxes):
        for j in range(i + 1, max_boxes):
            total_pairs += 1
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])
            inter = max(0, xx2 - xx1) * max(0, yy2 - yy1)
            union = areas[i] + areas[j] - inter + 1e-8
            if inter / union > 0.30:
                hits += 1
            if total_pairs >= 1800:  # Early exit
                return hits / total_pairs

    return hits / max(total_pairs, 1)


def csrnet_count(model: nn.Module, frame_bgr: np.ndarray) -> float:
    """Run CSRNet inference and return estimated crowd count"""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(TORCH_DEVICE)

    with torch.no_grad():
        density_map = model(tensor)
        count = density_map.sum().item()

    return float(count)


# =========================
# Main Loop
# =========================
def main():
    global SHUTDOWN_REQUESTED

    ensure_debug_dir()

    log("INFO", "=== HajjFlow Edge Client Starting ===")
    log("INFO", f"Device: {DEVICE_ID} ({DEVICE_NAME}) | Zone: {ZONE}")
    log("INFO", f"Video: {VIDEO_PATH} | YOLO: {YOLO_MODEL} | CSRNet: {'ENABLED' if ENABLE_CSRNET else 'DISABLED'}")
    log("INFO", f"Torch device: {TORCH_DEVICE}")

    # Socket.IO Client
    sio = socketio.Client(reconnection=True, reconnection_attempts=100, reconnection_delay=2,
                          logger=DEBUG, engineio_logger=(DEBUG and CUR_LEVEL <= 10))

    @sio.event
    def connect():
        log("INFO", f"[Socket.IO] Connected to {BASE_URL}")
        sio.emit("edge:register", {"deviceId": DEVICE_ID, "name": DEVICE_NAME, "zone": ZONE})

    @sio.event
    def disconnect():
        log("WARN", "[Socket.IO] Disconnected – will attempt reconnect")

    @sio.event
    def connect_error(data):
        log("ERROR", f"[Socket.IO] Connection error: {data}")

    # Connect to server
    try:
        sio.connect(BASE_URL, wait_timeout=10)
    except Exception as e:
        log("ERROR", f"Failed to connect to server: {e}")
        return

    # Load YOLO model
    try:
        yolo = YOLO(YOLO_MODEL)
        yolo.to(TORCH_DEVICE)
        log("INFO", f"[YOLO] Loaded {YOLO_MODEL} on {TORCH_DEVICE}")
    except Exception as e:
        log("ERROR", f"[YOLO] Failed to load model: {e}")
        if DEBUG:
            log("DEBUG", traceback.format_exc())
        return

    # Load CSRNet if enabled
    csr_model = load_csrnet(CSRNET_WEIGHTS) if ENABLE_CSRNET else None

    # Open video source
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        log("ERROR", f"Cannot open video source: {VIDEO_PATH}")
        return

    log("INFO", "Processing started. Press 'q' in debug window to quit, 's' to save frame, 'p' to pause.")

    last_heartbeat = 0.0
    seq = 0
    frame_idx = 0
    paused = False

    while not SHUTDOWN_REQUESTED:
        if paused:
            key = cv2.waitKey(100)
            if key == ord('p'):
                paused = False
                log("INFO", "[Debug] Resumed")
            continue

        ret, frame = cap.read()
        if not ret:
            log("INFO", "[Video] End of stream – restarting from beginning")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_idx += 1
        seq += 1

        # YOLO Inference
        inference_start = time.time()
        try:
            results = yolo.predict(
                source=frame,
                imgsz=IMGSZ,
                conf=CONF,
                iou=IOU,
                classes=[0],  # person class
                device=TORCH_DEVICE,
                verbose=False
            )[0]
        except Exception as e:
            log("ERROR", f"[YOLO] Inference failed: {e}")
            if DEBUG:
                log("DEBUG", traceback.format_exc())
            continue

        yolo_time = time.time() - inference_start
        yolo_fps = 1.0 / (yolo_time + 1e-8)

        boxes = results.boxes
        yolo_count = len(boxes) if boxes is not None else 0
        boxes_xyxy = boxes.xyxy.cpu().numpy() if boxes is not None and len(boxes) > 0 else np.zeros((0, 4))

        iou_ratio = compute_pairwise_iou_ratio(boxes_xyxy)
        is_dense = (yolo_count >= DENSE_COUNT_THR) or (iou_ratio >= DENSE_IOU_RATIO_THR)

        # Default: use YOLO count
        final_count = float(yolo_count)
        method_used = "YOLO11"
        csr_time_ms = None
        csr_error = None

        # Fallback to CSRNet in dense scenes
        if csr_model is not None and is_dense:
            try:
                csr_start = time.time()
                final_count = csrnet_count(csr_model, frame)
                method_used = "CSRNet"
                csr_time_ms = round((time.time() - csr_start) * 1000, 1)
            except Exception as e:
                csr_error = str(e)
                method_used = "YOLO11 (CSRNet failed)"
                final_count = float(yolo_count)
                log("WARN", f"[CSRNet] Inference failed: {e}")
                if DEBUG:
                    log("DEBUG", traceback.format_exc())

        pressure_label, pressure_color = crowd_pressure_label(final_count)

        # Send heartbeat
        now = time.time()
        if now - last_heartbeat >= HEARTBEAT_SEC:
            last_heartbeat = now

            payload = {
                "deviceId": DEVICE_ID,
                "name": DEVICE_NAME,
                "zone": ZONE,
                "location": {"lat": LAT, "lng": LNG},
                "ts": int(now),
                "seq": seq,
                "metrics": {
                    "count": round(final_count, 2),
                    "pressure": pressure_label,
                    "method": method_used,
                    "yolo_count": yolo_count,
                    "yolo_fps": round(yolo_fps, 1),
                    "dense_scene": is_dense,
                    "iou_ratio": round(iou_ratio, 3),
                    "csr_time_ms": csr_time_ms,
                    "csr_error": csr_error,
                    "cpu_percent": round(psutil.cpu_percent(), 1),
                    "temperature_c": None  # Optional: get_temp(LAT, LNG)
                }
            }

            try:
                sio.emit("edge:heartbeat", payload)
                log("INFO", f"→ HB | {method_used} | Count: {final_count:.1f} | {pressure_label} | "
                            f"YOLO FPS: {yolo_fps:.1f} | Dense: {is_dense}")
            except Exception as e:
                log("WARN", f"[Socket.IO] Failed to send heartbeat: {e}")

            # Save payload for offline analysis
            if DEBUG_SAVE_PAYLOADS:
                try:
                    path = os.path.join(DEBUG_OUT_DIR, f"payload_{int(now)}_{DEVICE_ID}.json")
                    with open(path, "w", encoding="utf-8") as f:
                        json.dump(payload, f, indent=2)
                    log("DEBUG", f"Payload saved: {path}")
                except Exception as e:
                    log("DEBUG", f"Failed to save payload: {e}")

        # Debug Visualization
        if DEBUG_VIEW and (frame_idx % SHOW_EVERY_N == 0):
            vis_frame = frame.copy()

            # Draw YOLO boxes (limit to avoid clutter)
            for box in boxes_xyxy[:200]:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Enhanced overlay
            overlay_lines = [
                f"Device: {DEVICE_ID} | Zone: {ZONE}",
                f"Method: {method_used} | Pressure: {pressure_label}",
                f"Count: {final_count:.1f} | YOLO: {yolo_count} | IoU Ratio: {iou_ratio:.3f}",
                f"YOLO FPS: {yolo_fps:.1f} | Dense Scene: {is_dense}",
                f"CSRNet Time: {csr_time_ms or '-'} ms | Error: {csr_error or 'None'}",
                f"CPU: {psutil.cpu_percent():.1f}% | Frame: {frame_idx}",
            ]

            # Background bar for readability
            cv2.rectangle(vis_frame, (5, 5), (900, 160), (0, 0, 0), -1)
            cv2.rectangle(vis_frame, (5, 5), (900, 160), pressure_color, 4)

            y_pos = 30
            for line in overlay_lines:
                cv2.putText(vis_frame, line, (15, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            (255, 255, 255), 2, cv2.LINE_AA)
                y_pos += 28

            cv2.imshow("HajjFlow Edge Debug View", vis_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                log("INFO", "[Debug] Quit requested via 'q'")
                break
            elif key == ord('s') and DEBUG_SAVE_FRAMES:
                path = os.path.join(DEBUG_OUT_DIR, f"debug_frame_{DEVICE_ID}_{seq}_{int(now)}.jpg")
                cv2.imwrite(path, vis_frame)
                log("INFO", f"Frame saved: {path}")
            elif key == ord('p'):
                paused = True
                log("INFO", "[Debug] Paused – press 'p' to resume")

        # Small delay to prevent 100% CPU
        time.sleep(0.002)

    # Cleanup
    cap.release()
    if DEBUG_VIEW:
        cv2.destroyAllWindows()
    sio.disconnect()
    log("INFO", "Edge client shutdown complete.")


if __name__ == "__main__":
    main()