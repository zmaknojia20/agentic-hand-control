"""
Camera controller — MediaPipe Tasks hand tracking → Arduino servo angles.
Uses mediapipe ≥ 0.10 Tasks API (solutions namespace was removed in 0.10.x).
Downloads hand_landmarker.task model on first run (~35 MB).
"""

import logging
import math
import os
import threading
import time
import urllib.request
from dataclasses import dataclass, field

log = logging.getLogger("camera")

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "hand_landmarker.task")

# Hard-coded MediaPipe hand skeleton connections (landmark index pairs)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),          # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),          # index
    (0, 9), (9, 10), (10, 11), (11, 12),     # middle
    (0, 13), (13, 14), (14, 15), (15, 16),   # ring
    (0, 17), (17, 18), (18, 19), (19, 20),   # pinky
    (5, 9), (9, 13), (13, 17),               # palm cross-connections
]


@dataclass
class HandState:
    detected: bool = False
    angles: list[int] = field(default_factory=lambda: [0, 0, 0, 0, 0])
    names: tuple = ("thumb", "index", "middle", "ring", "pinky")
    frame_jpeg: bytes | None = None
    error: str | None = None


def _ensure_model() -> str:
    """Download hand_landmarker.task if not already present. Returns local path."""
    if not os.path.exists(MODEL_PATH):
        import ssl
        log.info("Downloading hand_landmarker.task (~35 MB) — one-time setup...")
        tmp = MODEL_PATH + ".part"
        # macOS Python from python.org lacks system CA certs; bypass verification
        # for this single trusted Google Storage URL.
        ctx = ssl.create_default_context()
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        try:
            with urllib.request.urlopen(MODEL_URL, context=ctx) as resp:
                with open(tmp, "wb") as f:
                    f.write(resp.read())
            os.rename(tmp, MODEL_PATH)
        except Exception:
            if os.path.exists(tmp):
                os.remove(tmp)
            raise
        log.info("Model saved to %s", MODEL_PATH)
    return MODEL_PATH


class CameraController:
    def __init__(self, send_angles_fn, camera_index: int = 0):
        self._send = send_angles_fn
        self._cam_idx = camera_index
        self._running = False
        self._thread: threading.Thread | None = None
        self._state = HandState()
        self._lock = threading.Lock()

    # ── public ────────────────────────────────────────────────────────────────

    @property
    def running(self) -> bool:
        return self._running

    @property
    def error(self) -> str | None:
        with self._lock:
            return self._state.error

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True, name="camera")
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=4)
            self._thread = None
        with self._lock:
            self._state = HandState()

    def get_state(self) -> HandState:
        with self._lock:
            s = self._state
            return HandState(detected=s.detected, angles=list(s.angles), error=s.error)

    def get_frame(self) -> bytes | None:
        with self._lock:
            return self._state.frame_jpeg

    # ── private ───────────────────────────────────────────────────────────────

    def _set_error(self, msg: str):
        log.error("Camera error: %s", msg)
        self._running = False
        with self._lock:
            self._state = HandState(error=msg)

    def _loop(self):
        try:
            import cv2
            import mediapipe as mp
        except ImportError as exc:
            self._set_error(f"Missing dependency: {exc}")
            return

        # Download model if needed
        try:
            model_path = _ensure_model()
        except Exception as exc:
            self._set_error(f"Failed to download hand landmarker model: {exc}")
            return

        # Open camera (AVFoundation backend on macOS = lower latency, more reliable)
        log.info("Opening camera index %d", self._cam_idx)
        backend = cv2.CAP_AVFOUNDATION if hasattr(cv2, "CAP_AVFOUNDATION") else cv2.CAP_ANY
        cap = cv2.VideoCapture(self._cam_idx, backend)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            self._set_error(
                f"cv2.VideoCapture({self._cam_idx}) could not open.\n"
                "→ Grant camera access: System Settings → Privacy & Security → Camera → Terminal"
            )
            return

        # Warm-up reads to confirm frames actually arrive (macOS permission check)
        warmed = False
        for _ in range(10):
            ret, _ = cap.read()
            if ret:
                warmed = True
                break
            time.sleep(0.1)

        if not warmed:
            cap.release()
            self._set_error(
                "Camera opened but returned no frames.\n"
                "→ Grant camera access: System Settings → Privacy & Security → Camera → Terminal"
            )
            return

        log.info("Camera OK (index=%d)", self._cam_idx)

        # Build Tasks-API hand landmarker in VIDEO mode
        options = mp.tasks.vision.HandLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.7,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        GREEN = (0, 255, 65)
        DIM   = (0, 136, 42)
        last_send  = 0.0
        start_time = time.time()

        try:
            with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:
                while self._running:
                    ret, frame = cap.read()
                    if not ret:
                        log.warning("cap.read() returned False")
                        time.sleep(0.05)
                        continue

                    frame = cv2.flip(frame, 1)
                    h, w = frame.shape[:2]

                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                    # VIDEO mode requires a monotonically increasing timestamp in ms
                    timestamp_ms = int((time.time() - start_time) * 1000)
                    result = landmarker.detect_for_video(mp_image, timestamp_ms)

                    angles   = [0, 0, 0, 0, 0]
                    detected = False

                    if result.hand_landmarks:
                        lm = result.hand_landmarks[0]   # list of NormalizedLandmark
                        detected = True
                        angles = self._calc_angles(lm)

                        # Draw skeleton
                        for a_idx, b_idx in HAND_CONNECTIONS:
                            ax, ay = int(lm[a_idx].x * w), int(lm[a_idx].y * h)
                            bx, by = int(lm[b_idx].x * w), int(lm[b_idx].y * h)
                            cv2.line(frame, (ax, ay), (bx, by), DIM, 2)
                        for pt in lm:
                            cv2.circle(frame, (int(pt.x * w), int(pt.y * h)), 4, GREEN, -1)

                        now = time.time()
                        if now - last_send >= 0.1:
                            self._send(*angles)
                            last_send = now

                    self._draw_hud(frame, angles, detected, w, h, GREEN, DIM)

                    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
                    with self._lock:
                        self._state = HandState(
                            detected=detected,
                            angles=angles,
                            frame_jpeg=jpeg.tobytes() if ok else None,
                        )

        except Exception as exc:
            self._set_error(f"Camera loop exception: {exc}")
        finally:
            cap.release()
            log.info("Camera released")

    # ── angle math ────────────────────────────────────────────────────────────

    @staticmethod
    def _calc_angles(lm) -> list[int]:
        """
        lm: list of NormalizedLandmark (.x, .y, .z)
        Returns [thumb, index, middle, ring, pinky] servo angles.
        0 = extended, 180 = curled.
        """

        def angle3(a, b, c) -> float:
            """Interior angle at b, given points a-b-c."""
            ax, ay = a.x - b.x, a.y - b.y
            cx, cy = c.x - b.x, c.y - b.y
            dot = ax * cx + ay * cy
            mag = math.hypot(ax, ay) * math.hypot(cx, cy)
            if mag < 1e-8:
                return 0.0
            return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))

        def to_servo(deg: float) -> int:
            # ~175° = straight/extended → 0; ~60° = fully curled → 180
            clamped = max(60.0, min(175.0, deg))
            ratio   = (clamped - 60.0) / (175.0 - 60.0)
            return max(0, min(180, int((1.0 - ratio) * 180)))

        # (a, b, c) triples — angle is measured at b
        # Thumb : CMC(1)-MCP(2)-IP(3)
        # Others: MCP-PIP-DIP  (indices shift by finger)
        joints = [
            (1,  2,  3),   # thumb
            (5,  6,  7),   # index
            (9,  10, 11),  # middle
            (13, 14, 15),  # ring
            (17, 18, 19),  # pinky
        ]
        return [to_servo(angle3(lm[a], lm[b], lm[c])) for a, b, c in joints]

    # ── HUD drawing ───────────────────────────────────────────────────────────

    @staticmethod
    def _draw_hud(frame, angles, detected, w, h, green, dim):
        import cv2
        names = ("THUMB", "INDEX", "MID  ", "RING ", "PINKY")
        bar_w = 120
        x0, y0 = 10, 10

        cv2.rectangle(frame, (x0 - 4, y0 - 4),
                      (x0 + 188, y0 + len(names) * 26 + 10), (0, 15, 0), -1)
        cv2.rectangle(frame, (x0 - 4, y0 - 4),
                      (x0 + 188, y0 + len(names) * 26 + 10), dim, 1)

        for i, (name, angle) in enumerate(zip(names, angles)):
            y    = y0 + i * 26 + 14
            fill = int(bar_w * angle / 180)
            cv2.putText(frame, f"{name} {angle:3d}", (x0, y),
                        cv2.FONT_HERSHEY_PLAIN, 0.9, dim, 1, cv2.LINE_AA)
            cv2.rectangle(frame, (x0 + 93, y - 8),
                          (x0 + 93 + bar_w, y - 2), (0, 30, 0), -1)
            if fill > 0:
                cv2.rectangle(frame, (x0 + 93, y - 8),
                              (x0 + 93 + fill, y - 2), green, -1)

        label = "HAND DETECTED" if detected else "NO HAND"
        cv2.putText(frame, label, (w - 155, 20),
                    cv2.FONT_HERSHEY_PLAIN, 1.0,
                    green if detected else (80, 80, 80), 1, cv2.LINE_AA)
