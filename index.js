const express = require("express");
const http = require("http");
const { Server } = require("socket.io");
const path = require("path");

const PORT = process.env.PORT || 5000;

// TTL still used for safety (e.g., edge frozen but socket not closed)
const OFFLINE_TTL_MS = Number(process.env.OFFLINE_TTL_MS || 30_000);

// History points stored per device (for charts)
const HISTORY_MAX = Number(process.env.HISTORY_MAX || 180);

// How often to push dashboard updates (even if no heartbeats)
const PUSH_INTERVAL_MS = Number(process.env.PUSH_INTERVAL_MS || 3000);

// Optional: purge device record after it stays offline this long (0 disables)
const OFFLINE_PURGE_MS = Number(process.env.OFFLINE_PURGE_MS || 0);

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
  cors: { origin: "*" },
  transports: ["websocket", "polling"],
});

app.set("view engine", "ejs");
app.set("views", path.join(__dirname, "views"));
app.use(express.static(path.join(__dirname, "public")));
app.use(express.json({ limit: "2mb" }));

/**
 * devices Map:
 * deviceId => {
 *   deviceId, name, zone,
 *   location: {lat,lng},
 *   metrics: {
 *     count, pressure, method,
 *     yolo_count, yolo_fps, dense_scene, iou_ratio,
 *     csr_time_ms, csr_error,
 *     fps,cpu,temp (legacy/optional)
 *   },
 *   lastSeen, online, disconnectedAt,
 *   ip, userAgent,
 *   lastPayloadTs,
 *   history: [{ts,count,pressure,method,yolo_fps,iou_ratio,dense_scene}]
 * }
 */
const devices = new Map();

// ✅ Track which socket belongs to which deviceId, so we can mark offline on disconnect
const socketToDevice = new Map();   // socket.id -> deviceId
const deviceToSockets = new Map();  // deviceId -> Set(socket.id)

function now() { return Date.now(); }

function toNumber(v, fallback = null) {
  const n = Number(v);
  return Number.isFinite(n) ? n : fallback;
}

function toBool(v, fallback = null) {
  if (typeof v === "boolean") return v;
  if (typeof v === "number") return v !== 0;
  if (typeof v === "string") {
    const s = v.toLowerCase().trim();
    if (["true", "1", "yes", "y"].includes(s)) return true;
    if (["false", "0", "no", "n"].includes(s)) return false;
  }
  return fallback;
}

function toStr(v, fallback = null) {
  if (v == null) return fallback;
  const s = String(v).trim();
  return s.length ? s : fallback;
}

function normalizePayload(payload, meta = {}) {
  if (!payload || typeof payload !== "object") return null;

  const ts = payload.ts
    ? toNumber(payload.ts, Math.floor(now() / 1000))
    : Math.floor(now() / 1000);

  const lat = payload.location?.lat ?? payload.lat;
  const lng = payload.location?.lng ?? payload.lng;

  const deviceId = toStr(payload.deviceId, "");
  if (!deviceId) return null;

  const m = payload.metrics || {};

  const normalized = {
    deviceId,
    name: toStr(payload.name ?? payload.deviceName ?? payload.deviceId, deviceId),
    zone: toStr(payload.zone, "Unassigned"),
    location: (lat != null && lng != null) ? { lat: toNumber(lat), lng: toNumber(lng) } : null,

    metrics: {
      // new fields from updated edge client
      count: toNumber(m.count ?? payload.count, null),
      pressure: toStr(m.pressure, null),        // LOW/MEDIUM/HIGH
      method: toStr(m.method, null),            // YOLO11/CSRNet

      yolo_count: toNumber(m.yolo_count, null),
      yolo_fps: toNumber(m.yolo_fps, null),

      dense_scene: toBool(m.dense_scene, null),
      iou_ratio: toNumber(m.iou_ratio, null),

      csr_time_ms: toNumber(m.csr_time_ms, null),
      csr_error: toStr(m.csr_error, null),

      // legacy/optional
      fps: toNumber(m.fps ?? payload.fps, null),
      cpu: toNumber(m.cpu ?? payload.cpu, null),
      temp: toNumber(m.temp ?? payload.temp, null),
    },

    // server-side meta
    ip: meta.ip || null,
    userAgent: meta.ua || null,

    lastSeen: now(),
    online: true,               // heartbeat means online
    disconnectedAt: null,       // clear if previously disconnected
    lastPayloadTs: ts,
  };

  if (normalized.location && (normalized.location.lat == null || normalized.location.lng == null)) {
    normalized.location = null;
  }

  return normalized;
}

function computeOnline(d) {
  // If we explicitly marked offline on disconnect, respect it
  if (d?.online === false) return false;
  return d && (now() - d.lastSeen) < OFFLINE_TTL_MS;
}

function snapshotList() {
  const arr = Array.from(devices.values()).map(d => ({
    deviceId: d.deviceId,
    name: d.name,
    zone: d.zone,
    location: d.location,
    metrics: d.metrics,
    lastSeen: d.lastSeen,
    online: computeOnline(d),
    ip: d.ip,
    userAgent: d.userAgent,
    lastPayloadTs: d.lastPayloadTs,
  }));

  // stable ordering: online first, then by name
  arr.sort((a, b) => (Number(b.online) - Number(a.online)) || (a.name || "").localeCompare(b.name || ""));
  return arr;
}

function kpiSummary(list) {
  const total = list.length;
  const online = list.filter(d => d.online).length;
  const offline = total - online;

  const avg = (key) => {
    const vals = list.map(d => d.metrics?.[key]).filter(v => Number.isFinite(v));
    if (!vals.length) return null;
    return vals.reduce((a, b) => a + b, 0) / vals.length;
  };

  const sumCount = list.reduce((acc, d) => acc + (Number.isFinite(d.metrics?.count) ? d.metrics.count : 0), 0);

  const pressure = { LOW: 0, MEDIUM: 0, HIGH: 0, UNKNOWN: 0 };
  const method = { YOLO11: 0, CSRNet: 0, UNKNOWN: 0 };

  let denseYes = 0, denseNo = 0;

  for (const d of list) {
    const p = (d.metrics?.pressure || "UNKNOWN").toUpperCase();
    if (pressure[p] == null) pressure.UNKNOWN++;
    else pressure[p]++;

    const m = d.metrics?.method || "UNKNOWN";
    if (method[m] == null) method.UNKNOWN++;
    else method[m]++;

    const dense = d.metrics?.dense_scene;
    if (dense === true) denseYes++;
    else if (dense === false) denseNo++;
  }

  const denseTotal = denseYes + denseNo;
  const denseRatio = denseTotal ? Number((denseYes / denseTotal).toFixed(2)) : null;

  return {
    total, online, offline,
    sumCount: Number.isFinite(sumCount) ? Math.round(sumCount) : null,

    // NEW: use yolo_fps + iou_ratio if available
    avgYoloFps: avg("yolo_fps") ? Number(avg("yolo_fps").toFixed(1)) : null,
    avgIoURatio: avg("iou_ratio") ? Number(avg("iou_ratio").toFixed(2)) : null,

    // legacy optional health stats
    avgCpu: avg("cpu") ? Number(avg("cpu").toFixed(0)) : null,
    avgTemp: avg("temp") ? Number(avg("temp").toFixed(0)) : null,

    pressure,
    method,
    denseRatio,
  };
}

function pushDashboardUpdate() {
  const list = snapshotList();
  io.to("dashboards").emit("dashboard:update", {
    devices: list,
    kpis: kpiSummary(list),
    ts: now(),
  });
}

function bindSocketToDevice(socketId, deviceId) {
  socketToDevice.set(socketId, deviceId);

  if (!deviceToSockets.has(deviceId)) deviceToSockets.set(deviceId, new Set());
  deviceToSockets.get(deviceId).add(socketId);
}

function unbindSocket(socketId) {
  const deviceId = socketToDevice.get(socketId);
  socketToDevice.delete(socketId);

  if (!deviceId) return null;

  const set = deviceToSockets.get(deviceId);
  if (!set) return deviceId;

  set.delete(socketId);
  if (set.size === 0) deviceToSockets.delete(deviceId);

  return deviceId;
}

function markDeviceOffline(deviceId) {
  const d = devices.get(deviceId);
  if (!d) return;

  d.online = false; // immediate offline
  d.disconnectedAt = now();
  // keep lastSeen as-is for display
  devices.set(deviceId, d);
}

// Periodic update push (also refreshes TTL-based online/offline in UI)
setInterval(pushDashboardUpdate, PUSH_INTERVAL_MS);

// Optional purge
if (OFFLINE_PURGE_MS > 0) {
  setInterval(() => {
    const t = now();
    for (const [id, d] of devices.entries()) {
      const online = computeOnline(d);
      if (!online) {
        const offAt = d.disconnectedAt || d.lastSeen || 0;
        if (t - offAt > OFFLINE_PURGE_MS) {
          devices.delete(id);
        }
      }
    }
  }, 60_000);
}

// --- Socket.IO ---
io.on("connection", (socket) => {
  console.log(`Socket connected: ${socket.id}`);
  const ua = socket.handshake.headers["user-agent"] || "";
  const ip = socket.handshake.address;

  // helpful hello
  socket.emit("server:hello", { ts: now() });

  socket.on("dashboard:join", () => {
    socket.join("dashboards");
    const list = snapshotList();
    socket.emit("dashboard:update", { devices: list, kpis: kpiSummary(list), ts: now() });
  });

  socket.on("device:join", ({ deviceId }) => {
    if (deviceId) socket.join(`device:${deviceId}`);
  });

  // Optional explicit registration from edge (recommended)
  socket.on("edge:register", ({ deviceId }) => {
    if (!deviceId) return;
    const id = String(deviceId).trim();
    if (!id) return;
    bindSocketToDevice(socket.id, id);

    const d = devices.get(id);
    if (d) {
      d.online = true;
      d.lastSeen = now();
      d.disconnectedAt = null;
      devices.set(id, d);
      pushDashboardUpdate();
    }
  });

  // Heartbeat from edge device
  socket.on("edge:heartbeat", (payload) => {
    const normalized = normalizePayload(payload, { ip, ua });
    if (!normalized) return;

    // bind socket to deviceId so we can mark offline on disconnect
    bindSocketToDevice(socket.id, normalized.deviceId);

    const prev = devices.get(normalized.deviceId);
    const merged = {
      ...(prev || {}),
      ...normalized,
      history: prev?.history || [],
    };

    // history point (rich)
    merged.history.push({
      ts: normalized.lastPayloadTs ?? Math.floor(now() / 1000),
      count: merged.metrics.count,
      pressure: merged.metrics.pressure,
      method: merged.metrics.method,
      yolo_fps: merged.metrics.yolo_fps,
      iou_ratio: merged.metrics.iou_ratio,
      dense_scene: merged.metrics.dense_scene,
    });

    if (merged.history.length > HISTORY_MAX) {
      merged.history.splice(0, merged.history.length - HISTORY_MAX);
    }

    devices.set(normalized.deviceId, merged);

    // push to dashboards + device room
    pushDashboardUpdate();

    io.to(`device:${normalized.deviceId}`).emit("device:update", {
      device: merged,
      online: computeOnline(merged),
      ts: now(),
    });
  });

  socket.on("disconnect", () => {
    // ✅ immediate offline if last edge socket is gone
    const deviceId = unbindSocket(socket.id);
    if (!deviceId) return;

    const stillConnected = deviceToSockets.has(deviceId);
    if (!stillConnected) {
      markDeviceOffline(deviceId);

      const d = devices.get(deviceId);
      if (d) {
        io.to(`device:${deviceId}`).emit("device:update", {
          device: d,
          online: false,
          ts: now(),
        });
      }
      pushDashboardUpdate();
    }
  });
});

// --- REST endpoints (handy for report/testing) ---
app.get("/", (req, res) => res.render("dashboard"));

app.get("/api/devices", (req, res) => {
  const list = snapshotList();
  res.json({ devices: list, kpis: kpiSummary(list), ts: now() });
});

app.get("/api/devices/:id", (req, res) => {
  const d = devices.get(req.params.id);
  if (!d) return res.status(404).json({ error: "Not found" });
  res.json({ device: d, online: computeOnline(d), ts: now() });
});

server.listen(PORT, () => {
  console.log(`HajjFlow Dashboard: http://localhost:${PORT}`);
});
