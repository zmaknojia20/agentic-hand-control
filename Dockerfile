# Public cloud demo image — runs the FastAPI app in DEMO_MODE (virtual hand,
# browser-side camera, hosted LLM). No Arduino, no server-side OpenCV/MediaPipe.
FROM python:3.12-slim

WORKDIR /app

# Install demo dependencies first for better layer caching.
COPY requirements-demo.txt .
RUN pip install --no-cache-dir -r requirements-demo.txt

# App code (see .dockerignore for what's excluded).
COPY main.py camera.py ./
COPY static ./static

ENV DEMO_MODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000

EXPOSE 8000

# Honor the platform-provided $PORT (Render/Railway/Fly set this).
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
