"""
Agentic Hand Control — FastAPI server
Translates natural language into Arduino servo commands via openai-agents + Ollama.
Camera mode: MediaPipe hand tracking → live servo control.
"""

import asyncio
import json
import os
import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from camera import CameraController

import serial
import serial.tools.list_ports
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agents import Agent, Runner, RunConfig, function_tool
from agents.models.openai_chatcompletions import OpenAIChatCompletionsModel
from openai import AsyncOpenAI

load_dotenv()

ARDUINO_PORT = os.environ.get("ARDUINO_PORT", "")
BAUD_RATE = int(os.environ.get("BAUD_RATE", "9600"))

# DEMO_MODE: no Arduino / no server-side camera. Serial writes are simulated and
# drive an on-screen virtual hand instead. Used for the public cloud showcase.
DEMO_MODE = os.environ.get("DEMO_MODE", "").lower() in ("1", "true", "yes", "on")

# LLM is OpenAI-compatible. Locally this points at Ollama; in the cloud demo it
# points at a hosted provider (e.g. Groq). OLLAMA_URL is kept for back-compat.
LLM_BASE_URL = os.environ.get("LLM_BASE_URL") or os.environ.get("OLLAMA_URL") or "http://localhost:11434/v1"
LLM_API_KEY = os.environ.get("LLM_API_KEY", "ollama")
MODEL_NAME = os.environ.get("MODEL_NAME", "gemma4:e2b-mlx")

_arduino: serial.Serial | None = None

# Latest actuated finger angles — mirrored to the UI's virtual hand. Updated by
# every servo command (real or simulated) so the on-screen hand always matches.
_virtual_angles: list[int] = [0, 0, 0, 0, 0]


def _get_arduino() -> serial.Serial:
    global _arduino
    if _arduino is None or not _arduino.is_open:
        port = ARDUINO_PORT
        if not port:
            for p in serial.tools.list_ports.comports():
                desc = (p.description or "").lower()
                dev = p.device.lower()
                if "usbserial" in dev or "usbmodem" in dev or "arduino" in desc:
                    port = p.device
                    break
            if not port:
                raise RuntimeError(
                    "Arduino not found. Set ARDUINO_PORT env var (e.g. /dev/cu.usbserial-0001)."
                )
        _arduino = serial.Serial(port, BAUD_RATE, timeout=1)
        time.sleep(2)  # wait for Arduino bootloader reset
    return _arduino


def _send_angles(thumb: int, index: int, middle: int, ring: int, pinky: int) -> str:
    global _virtual_angles
    angles = [max(0, min(180, v)) for v in (thumb, index, middle, ring, pinky)]
    _virtual_angles = angles  # mirror to the on-screen hand regardless of backend
    names = ("thumb", "index", "middle", "ring", "pinky")
    summary = "  ".join(f"{n}={a}°" for n, a in zip(names, angles))

    if DEMO_MODE:
        return f"[SIM] {summary}  |  virtual hand actuated"

    cmd = f"{','.join(map(str, angles))}\n"
    try:
        ser = _get_arduino()
        ser.write(cmd.encode())
        reply = ser.readline().decode().strip()
        return f"[OK] {summary}  |  arduino: {reply!r}"
    except Exception as exc:
        return f"[ERROR] {exc}"


# ── Tools ─────────────────────────────────────────────────────────────────────

@function_tool
def set_finger_angles(thumb: int, index: int, middle: int, ring: int, pinky: int) -> str:
    """Set individual servo angles for each finger (0 = fully open/extended, 180 = fully curled/closed)."""
    return _send_angles(thumb, index, middle, ring, pinky)


@function_tool
def open_hand() -> str:
    """Fully open and extend all five fingers."""
    return _send_angles(0, 0, 0, 0, 0)


@function_tool
def close_hand() -> str:
    """Close all five fingers into a tight fist."""
    return _send_angles(180, 180, 180, 180, 180)


@function_tool
def point_index_finger() -> str:
    """Extend the index finger to point while keeping all other fingers curled."""
    return _send_angles(180, 0, 180, 180, 180)


@function_tool
def thumbs_up() -> str:
    """Raise the thumb while curling all other fingers."""
    return _send_angles(0, 180, 180, 180, 180)


@function_tool
def peace_sign() -> str:
    """Extend index and middle fingers (peace/victory sign), curl the rest."""
    return _send_angles(180, 0, 0, 180, 180)


@function_tool
def rock_on() -> str:
    """Extend index and pinky fingers (rock-on / devil horns), curl the rest."""
    return _send_angles(180, 0, 180, 180, 0)


# ── Agent ─────────────────────────────────────────────────────────────────────

_llm_client = AsyncOpenAI(base_url=LLM_BASE_URL, api_key=LLM_API_KEY)
_model = OpenAIChatCompletionsModel(model=MODEL_NAME, openai_client=_llm_client)

agent = Agent(
    name="HandController",
    model=_model,
    instructions=(
        "You are an AI controlling a robotic hand with five servo-driven fingers attached to an Arduino. "
        "Translate the user's natural-language requests into precise servo movements using the available tools. "
        "Always call at least one tool to actuate the hand — never just describe an action without doing it. "
        "After calling a tool, briefly confirm the gesture performed and the angles used. "
        "Finger angle convention: 0° = fully open/extended, 180° = fully curled/closed."
    ),
    tools=[
        set_finger_angles,
        open_hand,
        close_hand,
        point_index_finger,
        thumbs_up,
        peace_sign,
        rock_on,
    ],
)

# ── Camera ────────────────────────────────────────────────────────────────────

_camera: CameraController | None = None

# Latest-angles slot drained by a dedicated serial-writer thread so the
# camera loop never blocks on Arduino I/O (readline can stall up to 1 s).
import threading as _threading

_angle_slot: list[int] | None = None
_angle_event = _threading.Event()
_serial_thread_started = False


def _serial_writer_loop() -> None:
    global _angle_slot
    while True:
        _angle_event.wait()
        _angle_event.clear()
        angles = _angle_slot
        if angles is None:
            continue
        try:
            ser = _get_arduino()
            ser.write(f"{','.join(map(str, angles))}\n".encode())
            try:
                ser.readline()
            except Exception:
                pass
        except Exception:
            # No Arduino / port gone — sleep briefly so we don't spin on
            # repeated port enumeration if frames keep arriving.
            time.sleep(0.5)


def _send_angles_raw(thumb: int, index: int, middle: int, ring: int, pinky: int) -> None:
    """Non-blocking: stash latest angles, wake the serial writer thread."""
    global _angle_slot, _serial_thread_started
    if not _serial_thread_started:
        _threading.Thread(target=_serial_writer_loop, daemon=True, name="serial-writer").start()
        _serial_thread_started = True
    _angle_slot = [max(0, min(180, v)) for v in (thumb, index, middle, ring, pinky)]
    _angle_event.set()


# ── FastAPI ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    if DEMO_MODE:
        print(f"✓ DEMO_MODE — virtual hand, no Arduino. LLM: {MODEL_NAME} @ {LLM_BASE_URL}")
    else:
        try:
            ser = _get_arduino()
            print(f"✓ Arduino connected on {ser.port}")
        except Exception as exc:
            print(f"⚠  Arduino not detected at startup: {exc}")
    yield
    global _camera
    if _camera and _camera.running:
        _camera.stop()
    if _arduino and _arduino.is_open:
        _arduino.close()


app = FastAPI(title="Agentic Hand Control", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")


class ChatRequest(BaseModel):
    message: str


@app.get("/")
async def index():
    return FileResponse("static/index.html")


@app.get("/status")
async def status():
    if DEMO_MODE:
        return {"arduino": "simulated", "port": "virtual", "model": MODEL_NAME, "demo": True}
    try:
        ser = _get_arduino()
        return {"arduino": "connected", "port": ser.port, "model": MODEL_NAME, "demo": False}
    except Exception as exc:
        return {"arduino": "disconnected", "error": str(exc), "model": MODEL_NAME, "demo": False}


@app.get("/hand/state")
async def hand_state():
    """Current actuated finger angles — used by the on-screen virtual hand."""
    return {"angles": _virtual_angles}


@app.get("/ports")
async def list_ports():
    return {
        "ports": [
            {"device": p.device, "description": p.description or ""}
            for p in serial.tools.list_ports.comports()
        ]
    }


# ── Camera endpoints ──────────────────────────────────────────────────────────

@app.post("/camera/start")
async def camera_start(camera_index: int = 0):
    global _camera
    if _camera and _camera.running:
        return {"status": "already_running"}
    _camera = CameraController(_send_angles_raw, camera_index=camera_index)
    _camera.start()

    # Give the thread a moment to warm up, then check for immediate errors
    await asyncio.sleep(1.5)
    err = _camera.error
    if err:
        _camera = None
        return {"status": "error", "error": err}

    return {"status": "started"}


@app.post("/camera/stop")
async def camera_stop():
    global _camera
    if _camera:
        _camera.stop()
        _camera = None
    return {"status": "stopped"}


@app.get("/camera/feed")
async def camera_feed():
    """MJPEG stream of the processed webcam feed."""
    async def generate():
        boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
        # Wait up to 3 s for the first real frame
        deadline = asyncio.get_event_loop().time() + 3.0
        while True:
            frame = _camera.get_frame() if _camera else None
            err   = _camera.error if _camera else "camera not started"
            if err:
                break                           # camera failed; stop stream
            if frame:
                yield boundary + frame + b"\r\n"
            elif asyncio.get_event_loop().time() > deadline:
                break                           # no frame within timeout
            await asyncio.sleep(1 / 30)

    return StreamingResponse(
        generate(),
        media_type="multipart/x-mixed-replace; boundary=frame",
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/camera/state")
async def camera_state():
    """SSE stream of current finger angles, detection status, and errors."""
    async def generate():
        while True:
            if _camera:
                s = _camera.get_state()
                payload = {
                    "running":  _camera.running,
                    "detected": s.detected,
                    "angles":   s.angles,
                    "error":    s.error,
                }
            else:
                payload = {
                    "running":  False,
                    "detected": False,
                    "angles":   [0, 0, 0, 0, 0],
                    "error":    None,
                }
            yield f"data: {json.dumps(payload)}\n\n"
            await asyncio.sleep(0.1)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/chat")
async def chat(req: ChatRequest):
    async def generate() -> AsyncGenerator[str, None]:
        try:
            result = Runner.run_streamed(
                agent,
                req.message,
                run_config=RunConfig(tracing_disabled=True),
            )
            async for event in result.stream_events():
                if event.type == "raw_response_event":
                    data = event.data
                    # Chat-completions stream (Ollama path)
                    if hasattr(data, "choices") and data.choices:
                        delta = data.choices[0].delta
                        content = getattr(delta, "content", None)
                        if content:
                            yield f"data: {json.dumps({'type': 'token', 'content': content})}\n\n"
                    # Responses-API text delta (native OpenAI path)
                    elif getattr(data, "type", None) == "response.output_text.delta":
                        yield f"data: {json.dumps({'type': 'token', 'content': data.delta})}\n\n"

                elif event.type == "run_item_stream_event":
                    item = event.item
                    if item.type == "tool_call_item":
                        name = getattr(item, "tool_name", None)
                        raw = item.raw_item
                        if isinstance(raw, dict):
                            name = name or raw.get("name", "tool")
                            args = raw.get("arguments", raw.get("input", "{}"))
                        else:
                            name = name or getattr(raw, "name", "tool")
                            args = getattr(raw, "arguments", getattr(raw, "input", "{}"))
                        if not isinstance(args, str):
                            args = json.dumps(args)
                        yield f"data: {json.dumps({'type': 'tool_call', 'name': name, 'args': args})}\n\n"
                    elif item.type == "tool_call_output_item":
                        yield f"data: {json.dumps({'type': 'tool_result', 'content': str(item.output)})}\n\n"
                        # Mirror the freshly actuated pose to the on-screen hand.
                        yield f"data: {json.dumps({'type': 'hand', 'angles': _virtual_angles})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'content': str(exc)})}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def start():
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    start()
