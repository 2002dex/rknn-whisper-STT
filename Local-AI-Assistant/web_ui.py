#!/usr/bin/env python3
"""
web_ui.py — Real-time Hindi Voice Assistant Web UI
===================================================
Combines the test_assistant_indic.py recording loop with a browser-based
real-time UI that shows live transcript tokens and the Qwen-2.5 response.

Features:
  • Two activation modes (switchable from the browser):
      1. Wake-word trigger ("hey_jarvis" via openwakeword)
      2. Manual button press in the browser
  • Records audio until silence is detected
  • Streams partial Hindi transcript token-by-token to the browser (WebSocket)
  • Shows final Qwen-2.5 response when ready
  • Handles GPIO LED states (graceful fallback if gpiod unavailable)

Start with:
    uvicorn web_ui:app --host 0.0.0.0 --port 8765
"""

import asyncio
import json
import os
import sys
import time
import wave
import threading
import tempfile
import logging
from pathlib import Path
from typing import Set, Optional

import numpy as np

# ── FastAPI ───────────────────────────────────────────────────
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# ── STT + Qwen core (Updated to import hindi core) ────────────
import assistant_core_hindi as core

logger = logging.getLogger("web_ui")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════

WAKE_WORD         = "hey_jarvis"
WAKE_WORD_ONNX    = "/home/smadmin/assistant/hey_jarvis_v0.1.onnx"
MELSPEC_ONNX      = "/home/smadmin/assistant/melspectrogram.onnx"
EMBEDDING_ONNX    = "/home/smadmin/assistant/embedding_model.onnx"
WAKE_WORD_THRESH  = 0.2

MIC_DEVICE_KEYWORD  = "rt5616"    # substring in PyAudio device name
SAMPLE_RATE         = 16000
CHUNK_SIZE          = 1280
SILENCE_THRESHOLD   = 500         # amplitude units
SILENCE_DURATION    = 5.0         # seconds of silence to stop recording
FOLLOW_UP_TIMEOUT   = 10          # seconds to wait for follow-up after reply

# GPIO LED (CM3588)
GPIO_CHIP = "/dev/gpiochip3"
GPIO_LINE = 1

OUTPUT_WAV = "/tmp/assistant_recording.wav"

# ══════════════════════════════════════════════════════════════
#  GPIO LED  (graceful fallback when gpiod unavailable)
# ══════════════════════════════════════════════════════════════

class _LEDStub:
    """No-op LED when GPIO is unavailable."""
    def set_state(self, s): pass
    def start(self): pass

try:
    import gpiod
    from gpiod.line import Direction, Value as GPIOValue

    class LEDController(threading.Thread):
        def __init__(self):
            super().__init__(daemon=True)
            self.state   = "OFF"
            self.running = True

        def set_state(self, s):
            self.state = s

        def run(self):
            try:
                with gpiod.request_lines(
                    GPIO_CHIP, consumer="assistant-led",
                    config={GPIO_LINE: gpiod.LineSettings(direction=Direction.OUTPUT)}
                ) as req:
                    while self.running:
                        s = self.state
                        if s == "OFF":
                            req.set_value(GPIO_LINE, GPIOValue.ACTIVE); time.sleep(0.1)
                        elif s == "ON":
                            req.set_value(GPIO_LINE, GPIOValue.INACTIVE); time.sleep(0.1)
                        elif s == "BLINK_3":
                            for _ in range(3):
                                req.set_value(GPIO_LINE, GPIOValue.INACTIVE); time.sleep(0.2)
                                req.set_value(GPIO_LINE, GPIOValue.ACTIVE);   time.sleep(0.2)
                            self.state = "ON"
                        elif s == "HEARTBEAT":
                            req.set_value(GPIO_LINE, GPIOValue.INACTIVE); time.sleep(0.1)
                            req.set_value(GPIO_LINE, GPIOValue.ACTIVE);   time.sleep(0.2)
                            req.set_value(GPIO_LINE, GPIOValue.INACTIVE); time.sleep(0.1)
                            req.set_value(GPIO_LINE, GPIOValue.ACTIVE);   time.sleep(0.7)
            except Exception as e:
                logger.warning("GPIO LED error: %s", e)

    led = LEDController()
    led.start()

except ImportError:
    logger.info("gpiod not found — LED control disabled.")
    led = _LEDStub()


# ══════════════════════════════════════════════════════════════
#  AUDIO HELPERS
# ══════════════════════════════════════════════════════════════

def _find_mic():
    """Return PyAudio device index for MIC_DEVICE_KEYWORD, or None."""
    try:
        import pyaudio
        p = pyaudio.PyAudio()
        for i in range(p.get_device_count()):
            info = p.get_device_info_by_index(i)
            if info.get("maxInputChannels", 0) > 0:
                if MIC_DEVICE_KEYWORD.lower() in info["name"].lower():
                    p.terminate()
                    return i
        p.terminate()
    except Exception as e:
        logger.warning("Mic search failed: %s", e)
    return None


def record_until_silence(stream, path: str) -> float:
    """
    Record from PyAudio stream until SILENCE_DURATION seconds of silence.
    Save result as 16kHz mono WAV to *path*.
    Returns duration in seconds.
    """
    led.set_state("BLINK_3")
    frames = []
    silent_chunks    = 0
    max_silent_chunks = int(SILENCE_DURATION * SAMPLE_RATE / CHUNK_SIZE)

    logger.info("🎙 Recording…")
    while True:
        data     = stream.read(CHUNK_SIZE, exception_on_overflow=False)
        frames.append(data)
        vol      = np.abs(np.frombuffer(data, dtype=np.int16)).mean()
        if vol < SILENCE_THRESHOLD:
            silent_chunks += 1
        else:
            silent_chunks  = 0
        if silent_chunks >= max_silent_chunks:
            logger.info("🔇 Silence detected — stopping recording.")
            break

    with wave.open(path, "wb") as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(SAMPLE_RATE)
        wf.writeframes(b"".join(frames))

    dur = len(frames) * CHUNK_SIZE / SAMPLE_RATE
    return dur


# ══════════════════════════════════════════════════════════════
#  WEBSOCKET HUB
# ══════════════════════════════════════════════════════════════

class _WSHub:
    def __init__(self):
        self._clients: Set[WebSocket] = set()
        self._loop:    Optional[asyncio.AbstractEventLoop] = None
        self._q:       Optional[asyncio.Queue] = None

    def set_loop(self, loop):
        self._loop = loop
        self._q    = asyncio.Queue()

    def add(self, ws):    self._clients.add(ws)
    def remove(self, ws): self._clients.discard(ws)

    def push(self, event: dict):
        """Called from sync threads."""
        if self._loop and self._q:
            self._loop.call_soon_threadsafe(self._q.put_nowait, event)

    async def _consumer(self):
        while True:
            try:
                ev   = await self._q.get()
                text = json.dumps(ev, ensure_ascii=False)
                dead = set()
                for ws in list(self._clients):
                    try:
                        await ws.send_text(text)
                    except Exception:
                        dead.add(ws)
                for ws in dead:
                    self._clients.discard(ws)
            except Exception as e:
                logger.error("WebSocket hub error: %s", e)

    async def start(self):
        asyncio.create_task(self._consumer())

hub = _WSHub()
# wire into core's event bus
core._ws_push_fn = hub.push


# ══════════════════════════════════════════════════════════════
#  ASSISTANT SESSION  (runs in background thread)
# ══════════════════════════════════════════════════════════════

_session_lock   = threading.Lock()
_session_active = False


def _run_session(stream=None, wav_path: str = OUTPUT_WAV):
    """
    Full assistant turn:
      record → STT (streaming tokens) → Qwen → push final event.
    Can be triggered by wake-word loop or manual button.
    *stream*: open PyAudio stream to use for recording. If None, opens a new one.
    """
    global _session_active
    import pyaudio

    own_stream = stream is None
    p = None

    try:
        hub.push({"type": "status", "text": "recording is in progress…"})
        led.set_state("BLINK_3")

        if own_stream:
            mic_idx = _find_mic()
            p = pyaudio.PyAudio()
            stream = p.open(
                format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
                input=True,
                input_device_index=mic_idx,
                frames_per_buffer=CHUNK_SIZE,
            )

        record_until_silence(stream, wav_path)

        led.set_state("ON")
        hub.push({"type": "status", "text": "Transcribing…"})

        led.set_state("HEARTBEAT")
        result = core.process_audio(wav_path)

        led.set_state("ON")
        
        # --- NEW LOGIC INTEGRATION LOGS ---
        trans = result.get("translation") or result.get("transcript", "")
        logger.info("Session complete. Input recognized: '%s'", trans[:120])
        
        if result.get("memory_note"):
            logger.info("💾 Database Action Logged: %s", result.get("memory_note"))
        # ----------------------------------

    except Exception as e:
        logger.error("Session error: %s", e)
        hub.push({"type": "error", "message": str(e)})
        led.set_state("OFF")
    finally:
        if own_stream:
            try: stream.stop_stream(); stream.close()
            except Exception: pass
            try: p.terminate()
            except Exception: pass
        if os.path.exists(wav_path):
            try: os.remove(wav_path)
            except Exception: pass
        with _session_lock:
            _session_active = False
        led.set_state("OFF")


def start_session():
    """Start a recording+STT session in a background thread (if not already running)."""
    global _session_active
    with _session_lock:
        if _session_active:
            hub.push({"type": "status", "text": "A recording is already in progress…"})
            return
        _session_active = True
    threading.Thread(target=_run_session, daemon=True, name="assistant-session").start()


# ══════════════════════════════════════════════════════════════
#  WAKE-WORD LISTENER  (background thread, optional)
# ══════════════════════════════════════════════════════════════

_wake_thread: Optional[threading.Thread] = None
_wake_stop   = threading.Event()


def _wake_word_loop():
    """
    Continuously listen for wake word using openwakeword.
    When detected, calls start_session() and then keeps the shared PyAudio
    stream alive for the recording + follow-up cycle.
    """
    import pyaudio
    from openwakeword.model import Model  # type: ignore

    mic_idx = _find_mic()
    if mic_idx is None:
        logger.warning("Wake word loop: mic not found — stopping.")
        hub.push({"type": "status", "text": "Microphone not found."})
        return

    oww = Model(
        wakeword_models=[WAKE_WORD_ONNX],
        inference_framework="onnx",
        melspec_model_path=MELSPEC_ONNX,
        embedding_model_path=EMBEDDING_ONNX,
    )

    p = pyaudio.PyAudio()
    stream = p.open(
        format=pyaudio.paInt16, channels=1, rate=SAMPLE_RATE,
        input=True, input_device_index=mic_idx, frames_per_buffer=CHUNK_SIZE,
    )

    logger.info("🎧 Wake-word listener started — waiting for '%s'…", WAKE_WORD)
    hub.push({"type": "status", "text": f"'{WAKE_WORD}' waiting for…"})

    try:
        while not _wake_stop.is_set():
            led.set_state("OFF")
            chunk = np.frombuffer(stream.read(CHUNK_SIZE, exception_on_overflow=False), dtype=np.int16)
            pred  = oww.predict(chunk)

            # --- PREDICTION SAFETY FIX ---
            score = 0.0
            if pred:
                score = pred.get(WAKE_WORD) or pred.get(WAKE_WORD.replace("_", " "))
                if score is None:
                    _, score = max(pred.items(), key=lambda kv: kv[1])
            # -----------------------------

            if score > WAKE_WORD_THRESH:
                logger.info("🟢 Wake word detected (score=%.3f)", score)
                hub.push({"type": "wake_word", "score": float(score)})

                # run session using the already-open stream
                global _session_active
                with _session_lock:
                    if _session_active:
                        continue
                    _session_active = True

                # run synchronously so we keep the stream
                _run_session(stream=stream)
                oww.reset()

                # brief follow-up listen
                start_wait = time.time()
                while time.time() - start_wait < FOLLOW_UP_TIMEOUT and not _wake_stop.is_set():
                    data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
                    vol  = np.abs(np.frombuffer(data, dtype=np.int16)).mean()
                    if vol > SILENCE_THRESHOLD:
                        hub.push({"type": "status", "text": "Follow-up heard."})
                        with _session_lock:
                            _session_active = True
                        _run_session(stream=stream)
                        oww.reset()
                        start_wait = time.time()   # reset
    finally:
        stream.stop_stream(); stream.close(); p.terminate()


def start_wake_word():
    global _wake_thread
    _wake_stop.clear()
    _wake_thread = threading.Thread(target=_wake_word_loop, daemon=True, name="wake-word")
    _wake_thread.start()


def stop_wake_word():
    _wake_stop.set()


# ══════════════════════════════════════════════════════════════
#  FASTAPI APP
# ══════════════════════════════════════════════════════════════

app = FastAPI(title="AI Voice Assistant", version="1.0")

_static = Path(__file__).parent / "static"
if _static.exists():
    app.mount("/static", StaticFiles(directory=str(_static)), name="static")


@app.on_event("startup")
async def _startup():
    hub.set_loop(asyncio.get_running_loop())
    await hub.start()
    # Pre-load STT model in a thread so the first request is instant
    threading.Thread(target=core.get_stt().load, daemon=True, name="stt-preload").start()
    threading.Thread(target=core.get_llama,       daemon=True, name="llama-preload").start()


# ── HTML SPA ──────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root():
    idx = _static / "assistant.html"
    if idx.exists():
        return HTMLResponse(idx.read_text(encoding="utf-8"))
    return HTMLResponse("<p>static/assistant.html not found.</p>", status_code=404)


# ── WebSocket ─────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    hub.add(ws)
    try:
        while True:
            await ws.receive_text()   # keep alive; client can send pings
    except WebSocketDisconnect:
        pass
    finally:
        hub.remove(ws)


# ── Manual trigger ────────────────────────────────────────────
@app.post("/listen")
async def listen():
    """Start a recording session (manual button trigger)."""
    start_session()
    return {"status": "recording_started"}


# ── Wake-word control ─────────────────────────────────────────
@app.post("/wake/start")
async def wake_start():
    """Start the background wake-word listener."""
    start_wake_word()
    return {"status": "wake_word_listener_started"}


@app.post("/wake/stop")
async def wake_stop():
    """Stop the background wake-word listener."""
    stop_wake_word()
    return {"status": "wake_word_listener_stopped"}


# ── Status ────────────────────────────────────────────────────
@app.get("/status")
async def status():
    return {
        "session_active": _session_active,
        "wake_word_running": _wake_thread is not None and _wake_thread.is_alive(),
        "stt_loaded": core.get_stt().loaded,
    }


# ── Direct run ────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("web_ui:app", host="0.0.0.0", port=8765, reload=False)