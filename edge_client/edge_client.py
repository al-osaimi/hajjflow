#!/usr/bin/env python3
import os
import time
import json
import traceback
from typing import Dict, Any, Optional

import cv2
import numpy as np
import socketio
import torch
import torch.nn as nn
from ultralytics import YOLO

# =========================
# ENV CONFIG
# =========================
BASE_URL = os.getenv("BASE_URL", "http://localhost:5000")
DEVICE_ID = os.getenv("DEVICE_ID", "EDGE-001")
DEVICE_NAME = os.getenv("DEVICE_NAME", DEVICE_ID)

ZONE = os.getenv("ZONE", "Masjid Al-Haram")
LAT = float(os.getenv("LAT", "21.4225"))
LNG = float(os.getenv("LNG", "39.8262"))

VIDEO_PATH = os.getenv("VIDEO_PATH", "video.mp4")

YOLO_MODEL = os.getenv("YOLO_MODEL", "yolo11n.pt")
CONF = float(os.getenv("CONF", "0.15"))
IOU = float(os.getenv("IOU", "0.45"))
IMGSZ = int(os.getenv("IMGSZ", "640"))

# CSRNet
CSRNET_WEIGHTS = os.getenv("CSRNET_WEIGHTS", "edge_client/weihgts/csrnet_shanghaitech.pth")
ENABLE_CSRNET = os.getenv("ENABLE_CSRNET", "1") == "1"
DEVICE = os.getenv("TORCH_DEVICE", "cpu")  # "cpu" or "cuda:0"

# Dense-scene heuristic
DENSE_COUNT_THR = int(os.getenv("DENSE_COUNT_THR", "160"))
DENSE_IOU_RATIO_THR = float(os.getenv("DENSE_IOU_RATIO_THR", "0.35"))

# Crowd pressure thresholds (tune per camera)
LOW_THR = float(os.getenv("LOW_THR", "60"))
MID_THR = float(os.getenv("MID_THR", "140"))

HEARTBEAT_SEC = float(os.getenv("HEARTBEAT_SEC", "2.0"))

# =========================
# DEBUG FEATURES (NEW)
# =========================
DEBUG = os.getenv("DEBUG", "0") == "1"
DEBUG_LEVEL = os.getenv("DEBUG_LEVEL", "INFO").upper()  # INFO | DEBUG | TRACE
DEBUG_VIEW = os.getenv("DEBUG_VIEW", "0") == "1"         # show OpenCV window
DEBUG_LOG = os.getenv("DEBUG_LOG", "")                  # file path optional
DEBUG_SAVE_FRAMES = os.getenv("DEBUG_SAVE_FRAMES", "0") == "1"
DEBUG_SAVE_PAYLOADS = os.getenv("DEBUG_SAVE_PAYLOADS", "0") == "1"
DEBUG_OUT_DIR = os.getenv("DEBUG_OUT_DIR", "debug_out")
SHOW_EVERY_N = int(os.getenv("DEBUG_SHOW_EVERY_N", "1"))

LEVELS = {"INFO": 20, "DEBUG": 10, "TRACE": 5}
CUR_LEVEL = LEVELS.get(DEBUG_LEVEL, 20)

def _write_log(line: str):
    if DEBUG_LOG:
        try:
            with open(DEBUG_LOG, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

def log(level: str, msg: str):
    lv = LEVELS.get(level, 20)
    if (not DEBUG) and level in ("DEBUG", "TRACE"):
        return
    if lv < CUR_LEVEL:
        return
    line = f"[{level}] {time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
    print(line)
    _write_log(line)

def ensure_debug_dir():
    if (DEBUG_SAVE_FRAMES or DEBUG_SAVE_PAYLOADS) and not os.path.exists(DEBUG_OUT_DIR):
        os.makedirs(DEBUG_OUT_DIR, exist_ok=True)

def draw_overlay(frame, text_lines):
    y = 24
    for t in text_lines:
        cv2.putText(frame, t, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 180), 2, cv2.LINE_AA)
        y += 22

# =========================
# CSRNet minimal model (VGG front + dilated backend)
# NOTE: must match your downloaded weights repo
# =========================
class CSRNet(nn.Module):
    def __init__(self):
        super().__init__()
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
        self.backend = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=2, dilation=2), nn.ReLU(inplace=True),
        )
        self.output_layer = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        x = self.frontend(x)
        x = self.backend(x)
        x = self.output_layer(x)
        return x


def load_csrnet(weights_path: str) -> Optional[nn.Module]:
    if not os.path.exists(weights_path):
        log("INFO", f"[CSRNet] weights not found: {weights_path}")
        return None

    model = CSRNet().to(DEVICE)
    log("INFO", f"[CSRNet] loading weights: {weights_path} on {DEVICE}")

    try:
        state = torch.load(weights_path, map_location=DEVICE)
    except Exception as e:
        log("INFO", f"[CSRNet] torch.load failed: {e}")
        if DEBUG:
            log("DEBUG", traceback.format_exc())
        return None

    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    if isinstance(state, dict):
        new_state = {}
        for k, v in state.items():
            new_state[k.replace("module.", "")] = v
        state = new_state

    try:
        missing, unexpected = model.load_state_dict(state, strict=False)
        model.eval()

        # Debug insight
        if missing:
            log("INFO", f"[CSRNet] missing keys: {len(missing)} (architecture mismatch likely)")
            if DEBUG and CUR_LEVEL <= LEVELS["DEBUG"]:
                log("DEBUG", f"[CSRNet] missing sample: {missing[:10]}")
        if unexpected:
            log("INFO", f"[CSRNet] unexpected keys: {len(unexpected)} (weights mismatch)")
            if DEBUG and CUR_LEVEL <= LEVELS["DEBUG"]:
                log("DEBUG", f"[CSRNet] unexpected sample: {unexpected[:10]}")

        if not missing and not unexpected:
            log("INFO", "[CSRNet] loaded weights OK (strict match)")
        else:
            log("INFO", "[CSRNet] loaded with partial match (may still work, verify outputs)")
        return model

    except Exception as e:
        log("INFO", "[CSRNet] FAILED to load weights (strict). Likely architecture mismatch.")
        log("INFO", f"Error: {e}")
        if DEBUG:
            log("DEBUG", traceback.format_exc())
        return None


# =========================
# Helpers
# =========================
def crowd_pressure_label(count: float) -> str:
    if count < LOW_THR:
        return "LOW"
    if count < MID_THR:
        return "MEDIUM"
    return "HIGH"

def compute_pairwise_iou_ratio(boxes_xyxy: np.ndarray) -> float:
    n = boxes_xyxy.shape[0]
    if n < 2:
        return 0.0
    max_n = min(n, 120)
    b = boxes_xyxy[:max_n]
    x1, y1, x2, y2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    area = (x2 - x1).clip(0) * (y2 - y1).clip(0)

    hits, total = 0, 0
    for i in range(max_n):
        for j in range(i + 1, max_n):
            total += 1
            xx1 = max(x1[i], x1[j]); yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j]); yy2 = min(y2[i], y2[j])
            w = max(0.0, xx2 - xx1); h = max(0.0, yy2 - yy1)
            inter = w * h
            union = area[i] + area[j] - inter + 1e-9
            iou = inter / union
            if iou > 0.30:
                hits += 1
            if total > 1800:
                return hits / total
    return hits / max(total, 1)

def csrnet_count(model: nn.Module, frame_bgr: np.ndarray) -> float:
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        dm = model(x)
        return float(dm.sum().item())


# =========================
# Main
# =========================
def main():
    ensure_debug_dir()

    # Socket.IO debug hooks
    sio = socketio.Client(reconnection=True, logger=DEBUG, engineio_logger=DEBUG)

    @sio.event
    def connect():
        log("INFO", f"[socket] connected to {BASE_URL}")
        sio.emit("edge:register", {"deviceId": DEVICE_ID})

    @sio.event
    def disconnect():
        log("INFO", "[socket] disconnected (will auto-reconnect)")

    @sio.event
    def connect_error(data):
        log("INFO", f"[socket] connect_error: {data}")

    log("INFO", f"Starting edge client: id={DEVICE_ID} name={DEVICE_NAME} zone={ZONE}")
    log("INFO", f"Video={VIDEO_PATH} YOLO={YOLO_MODEL} CSRNet={'ON' if ENABLE_CSRNET else 'OFF'} torch_device={DEVICE}")

    sio.connect(BASE_URL, transports=["websocket", "polling"])

    # YOLO11
    try:
        yolo = YOLO(YOLO_MODEL)
        log("INFO", f"[YOLO] loaded {YOLO_MODEL}")
    except Exception as e:
        log("INFO", f"[YOLO] failed to load model: {e}")
        if DEBUG:
            log("DEBUG", traceback.format_exc())
        raise

    PERSON_CLASS_ID = 0

    # CSRNet
    csr = load_csrnet(CSRNET_WEIGHTS) if ENABLE_CSRNET else None

    # Video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open VIDEO_PATH: {VIDEO_PATH}")

    last_send = 0.0
    seq = 0
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            log("INFO", "[video] end-of-file or read failed -> loop to start")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        frame_index += 1
        seq += 1

        # --- YOLO11 detection/count ---
        t0 = time.time()
        try:
            res = yolo.predict(
                source=frame,
                imgsz=IMGSZ,
                conf=CONF,
                iou=IOU,
                classes=[PERSON_CLASS_ID],
                verbose=False,
                device="cpu",
            )[0]
        except Exception as e:
            log("INFO", f"[YOLO] predict error: {e}")
            if DEBUG:
                log("DEBUG", traceback.format_exc())
            continue

        dt = time.time() - t0
        yolo_fps = 1.0 / (dt + 1e-9)

        boxes_xyxy = None
        if res.boxes is None or len(res.boxes) == 0:
            yolo_count = 0
            iou_ratio = 0.0
        else:
            boxes_xyxy = res.boxes.xyxy.cpu().numpy().astype(np.float32)
            yolo_count = int(boxes_xyxy.shape[0])
            iou_ratio = compute_pairwise_iou_ratio(boxes_xyxy)

        dense_scene = (yolo_count >= DENSE_COUNT_THR) or (iou_ratio >= DENSE_IOU_RATIO_THR)

        final_count = float(yolo_count)
        method = "YOLO11"

        # --- CSRNet fallback for dense scenes ---
        csr_err = None
        csr_time_ms = None
        if csr is not None and dense_scene:
            t1 = time.time()
            try:
                final_count = csrnet_count(csr, frame)
                method = "CSRNet"
            except Exception as e:
                csr_err = str(e)
                method = "YOLO11"
                final_count = float(yolo_count)
                if DEBUG:
                    log("DEBUG", f"[CSRNet] error: {csr_err}")
                    log("DEBUG", traceback.format_exc())
            csr_time_ms = round((time.time() - t1) * 1000.0, 2)

        pressure = crowd_pressure_label(final_count)

        # --- Send heartbeat ---
        now = time.time()
        if now - last_send >= HEARTBEAT_SEC:
            last_send = now
            payload = {
                "deviceId": DEVICE_ID,
                "name": DEVICE_NAME,
                "zone": ZONE,
                "location": {"lat": LAT, "lng": LNG},
                "ts": int(now),
                "seq": seq,
                "metrics": {
                    "count": round(final_count, 2),
                    "pressure": pressure,
                    "method": method,
                    "yolo_count": yolo_count,
                    "yolo_fps": round(yolo_fps, 2),
                    "dense_scene": dense_scene,
                    "iou_ratio": round(iou_ratio, 3),
                    "csr_time_ms": csr_time_ms,
                    "csr_error": csr_err,
                }
            }

            try:
                sio.emit("edge:heartbeat", payload)
            except Exception as e:
                log("INFO", f"[socket] emit failed: {e}")
                if DEBUG:
                    log("DEBUG", traceback.format_exc())

            log("INFO", f"[hb] method={method} count={payload['metrics']['count']} "
                        f"pressure={pressure} yolo_fps={payload['metrics']['yolo_fps']} "
                        f"dense={dense_scene} iou_ratio={payload['metrics']['iou_ratio']}")

            if DEBUG_SAVE_PAYLOADS:
                ensure_debug_dir()
                fn = os.path.join(DEBUG_OUT_DIR, f"payload_{int(now)}_{DEVICE_ID}.json")
                try:
                    with open(fn, "w", encoding="utf-8") as f:
                        json.dump(payload, f, ensure_ascii=False, indent=2)
                    log("DEBUG", f"[debug] saved payload -> {fn}")
                except Exception:
                    pass

        # --- Debug overlay / frames ---
        if DEBUG_VIEW and (frame_index % max(1, SHOW_EVERY_N) == 0):
            vis = frame.copy()

            if boxes_xyxy is not None:
                for (x1, y1, x2, y2) in boxes_xyxy[:200]:
                    cv2.rectangle(vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            draw_overlay(vis, [
                f"Device: {DEVICE_ID}  Zone: {ZONE}",
                f"Method: {method}  Pressure: {pressure}",
                f"YOLO count: {yolo_count}  YOLO fps: {round(yolo_fps,2)}",
                f"Dense: {dense_scene}  IoU ratio: {round(iou_ratio,3)}",
                f"Final count: {round(final_count,2)}",
                f"CSR time ms: {csr_time_ms if csr_time_ms is not None else '-'}",
            ])

            cv2.imshow("HajjFlow Edge Debug", vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                log("INFO", "[debug] quit requested")
                break
            if key == ord('s') and DEBUG_SAVE_FRAMES:
                ensure_debug_dir()
                fn = os.path.join(DEBUG_OUT_DIR, f"frame_{DEVICE_ID}_{seq}.jpg")
                cv2.imwrite(fn, vis)
                log("DEBUG", f"[debug] saved frame -> {fn}")

        # optional small sleep
        time.sleep(0.002)

    cap.release()
    if DEBUG_VIEW:
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
