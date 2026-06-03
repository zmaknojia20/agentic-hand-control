# Agentic Hand Control

Natural-language control of a 3D-printed, servo-driven robotic hand. An LLM agent
translates requests like *"give me a peace sign"* into Arduino servo commands —
or you can drive the hand live with your own webcam via MediaPipe hand tracking.

Runs two ways from the **same codebase**:

| Mode | Hand | Camera | LLM |
|------|------|--------|-----|
| **Local hardware** | Real Arduino over USB serial | Server-side OpenCV + MediaPipe | Local Ollama |
| **Demo** (`DEMO_MODE=1`) | Animated on-screen SVG hand | In the visitor's browser (MediaPipe Web) | Hosted (Groq) |

> 🔗 **Live demo:** _add your deployed URL here_
> 🎥 **Real hardware video:** _add a link to your demo clip here_

## Architecture

```
                    ┌─────────────── Browser UI (retro terminal) ───────────────┐
                    │  Chat  ·  Animated SVG hand  ·  Webcam hand-tracking       │
                    └───────────────┬───────────────────────┬──────────────────┘
                       /chat (SSE)  │                        │  MediaPipe Web (demo)
                                    ▼                        ▼
        ┌──────────── FastAPI (main.py) ────────────┐   (browser-only,
        │  openai-agents  →  function tools          │    drives SVG hand)
        │  open_hand · close_hand · peace_sign · …   │
        └───────────────┬────────────────────────────┘
                        │ servo angles [thumb,index,middle,ring,pinky]
            ┌───────────┴───────────┐
            ▼                       ▼
   Real Arduino (serial)    Simulated virtual hand (DEMO_MODE)
```

The agent (`openai-agents`) talks to any OpenAI-compatible endpoint — Ollama
locally, Groq in the cloud — and calls typed Python tools that emit servo angles.

## Run locally (real hardware)

Requires [`uv`](https://docs.astral.sh/uv/), an Arduino on USB running the servo
sketch, and a local [Ollama](https://ollama.com) model.

```bash
ollama serve &                 # start Ollama
ollama pull gemma4:e2b-mlx     # or set MODEL_NAME to any local model
uv run hand-control            # → http://localhost:8000
```

The Arduino port is auto-detected; override with `ARDUINO_PORT` if needed.

## Run the demo locally (no hardware)

```bash
DEMO_MODE=1 uv run hand-control
```

Serial writes are simulated and drive the on-screen hand. Point the LLM at any
OpenAI-compatible endpoint via `LLM_BASE_URL` / `LLM_API_KEY` / `MODEL_NAME`
(see `.env.example`).

## Deploy the public demo

The included `Dockerfile` + `render.yaml` deploy the demo to
[Render](https://render.com)'s free tier (no credit card).

1. **Get a free LLM key** — sign up at [console.groq.com](https://console.groq.com)
   and create an API key. Groq is OpenAI-compatible, fast, and supports tool calls.
2. **Push this repo to GitHub.**
3. **Render → New → Blueprint**, connect the repo. It reads `render.yaml`.
4. When prompted, set the **`LLM_API_KEY`** secret to your Groq key.
5. Deploy. Your live URL appears in the dashboard — drop it in your portfolio.

Other hosts (Railway, Fly.io) work too — the `Dockerfile` is platform-agnostic
and honors `$PORT`.

### Demo environment variables

| Var | Demo value |
|-----|-----------|
| `DEMO_MODE` | `1` |
| `LLM_BASE_URL` | `https://api.groq.com/openai/v1` |
| `MODEL_NAME` | `llama-3.3-70b-versatile` |
| `LLM_API_KEY` | _your Groq key (set as a secret)_ |

## Hardware

Five SG90-class servos (one per finger) driven by an Arduino. `0°` = extended,
`180°` = fully curled. See the `Arduino/` sketch for the serial protocol
(`thumb,index,middle,ring,pinky\n`).
