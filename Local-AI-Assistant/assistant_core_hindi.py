#!/usr/bin/env python3
# assistant_core_hindi.py
"""
Hindi-forced STT + Qwen-2.5 assistant core.
Based on assistant_core_optimized.py — hardcoded to Hindi forced tokens.

Key additions vs reference:
  - FORCED_IDS locked to Hindi (<|hi|> = 50275)
  - Token-by-token streaming callback: set on_token(text) for real-time UI updates
  - Qwen-2.5 integration (rkllm primary / HTTP fallback) with JSON response parsing
  - process_audio(wav_path) → dict {transcript, qwen_response, memory_note}
  - Thread-safe singleton engine; 30-min idle unload watchdog
"""

import os
import re
import json
import time
import threading
import logging
import sqlite3
import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple, Callable, List

import numpy as np
import librosa
import onnxruntime as ort
import requests
from rknnlite.api import RKNNLite
from transformers import WhisperTokenizer

# ─── Logging ──────────────────────────────────────────────────
LOG_DIR = Path("./logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)
logger = logging.getLogger("assistant_hindi")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    _fh  = logging.FileHandler(LOG_DIR / "assistant_hindi.log", encoding="utf-8")
    _fh.setFormatter(_fmt)
    # Removed stream handler to avoid duplicate logs with web_ui
    logger.addHandler(_fh)

# ══════════════════════════════════════════════════════════════
#  CONFIG
# ══════════════════════════════════════════════════════════════

MODEL_DIR              = Path("/home/smadmin/rknn-model/model/indic-whisper")
ENCODER_RKNN_PATH      = Path("/home/smadmin/rknn-model/model/indic-whisper/encoder_model.rknn")
DECODER_WITH_PAST_ONNX = Path("/home/smadmin/rknn-model/model/indic-whisper/decoder_with_past_model.onnx")
DECODER_ONNX_FALLBACK  = Path("/home/smadmin/rknn-model/model/indic-whisper/decoder_model.onnx")
ENCODER_KV_ONNX        = Path("/home/smadmin/rknn-model/model/indic-whisper/encoder_kv.onnx")

# Hindi forced token sequence: SOT | hi | transcribe | no-timestamps
FORCED_IDS     = [50258, 50275, 50359, 50363]

MAX_DECODING_STEPS = 448
MODEL_IDLE_TIMEOUT = 30 * 60          # seconds
RKNN_CORE_MASK     = RKNNLite.NPU_CORE_ALL
SAMPLE_RATE        = 16000

# ── Qwen-2.5 ─────────────────────────────────────────────────
QWEN_MODEL_PATH   = "/home/smadmin/rkllama/models/Llama3.2"
QWEN_BACKEND      = "rkllm"            # "rkllm" | "http"
QWEN_API_URL      = "http://localhost:8080/v1/chat/completions"
QWEN_API_KEY      = "none"
QWEN_MAX_TOKENS   = 400
QWEN_TEMPERATURE  = 0.4

# ── Regex helpers ─────────────────────────────────────────────
RE_PAST        = re.compile(r"past_key_values\.(\d+)\.(.*)",          re.IGNORECASE)
RE_PRESENT_KV  = re.compile(r"present\.?(\d+)\.?(decoder|encoder)\.(key|value)", re.IGNORECASE)

DEBUG = False
def _dbg(*a): DEBUG and logger.debug(*a)

# ══════════════════════════════════════════════════════════════
#  MEL EXTRACTION
# ══════════════════════════════════════════════════════════════

def get_mel(audio_path: str, duration_secs: int = 30) -> np.ndarray:
    """Whisper-style 80-bin mel spectrogram."""
    audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE)
    audio = librosa.util.fix_length(audio, size=SAMPLE_RATE * duration_secs)
    stft  = librosa.stft(audio, n_fft=400, hop_length=160, window="hann", center=True)
    mag   = np.abs(stft[:, :-1]) ** 2
    filt  = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=400, n_mels=80)
    spec  = filt.dot(mag)
    log   = np.log10(np.maximum(spec, 1e-10))
    log   = np.maximum(log, log.max() - 8.0)
    log   = (log + 4.0) / 4.0
    return log.astype(np.float32)[np.newaxis, :]


# ══════════════════════════════════════════════════════════════
#  STT ENGINE  (Hindi-forced)
# ══════════════════════════════════════════════════════════════

class HindiWhisperSTT:
    """
    Singleton-friendly Whisper STT engine, outputs Hindi only.
    Supports a per-token streaming callback `on_token(partial_text)`.
    """

    def __init__(self):
        self.encoder_rknn    = None
        self.decoder_sess    = None
        self.encoder_kv_sess = None
        self.tokenizer       = None
        self.decoder_input_names  = []
        self.decoder_output_names = []
        self.loaded     = False
        self.lock       = threading.RLock()
        self.last_used  = 0.0
        self.num_layers = self.num_heads = self.head_dim = None

        # Live streaming callback – set by web_ui or caller
        # Signature: on_token(partial_decoded_string: str)
        self.on_token: Optional[Callable[[str], None]] = None

    # ── lifecycle ─────────────────────────────────────────────
    def load(self):
        with self.lock:
            if self.loaded:
                return
            logger.info("🧠 Loading Hindi STT models…")

            try:
                self.tokenizer = WhisperTokenizer.from_pretrained(str(MODEL_DIR))
            except Exception as e:
                logger.warning("Tokenizer load failed: %s", e)

            self.encoder_rknn = RKNNLite()
            self.encoder_rknn.load_rknn(str(ENCODER_RKNN_PATH))
            self.encoder_rknn.init_runtime(core_mask=RKNN_CORE_MASK)
            logger.info("RKNN encoder loaded (NPU).")

            dec = DECODER_WITH_PAST_ONNX if DECODER_WITH_PAST_ONNX.exists() else DECODER_ONNX_FALLBACK
            if not dec.exists():
                raise FileNotFoundError(f"Decoder ONNX not found: {dec}")
            so = ort.SessionOptions()
            so.intra_op_num_threads = 4
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.decoder_sess = ort.InferenceSession(str(dec), so, providers=["CPUExecutionProvider"])
            self.decoder_input_names  = [i.name for i in self.decoder_sess.get_inputs()]
            self.decoder_output_names = [o.name for o in self.decoder_sess.get_outputs()]

            if ENCODER_KV_ONNX.exists():
                try:
                    self.encoder_kv_sess = ort.InferenceSession(
                        str(ENCODER_KV_ONNX), so, providers=["CPUExecutionProvider"])
                except Exception as e:
                    logger.warning("encoder_kv.onnx load failed: %s", e)

            self.num_layers, self.num_heads, self.head_dim = self._infer_kv_layout()
            logger.info("KV layout: layers=%d heads=%d head_dim=%d",
                        self.num_layers, self.num_heads, self.head_dim)
            self.loaded = True
            self.last_used = time.time()
            logger.info("✅ Hindi STT models ready.")

    def unload(self):
        with self.lock:
            if not self.loaded:
                return
            logger.info("🧹 Unloading Hindi STT models (idle).")
            try:
                if self.encoder_rknn:
                    self.encoder_rknn.release()
            except Exception:
                pass
            self.encoder_rknn = self.decoder_sess = self.encoder_kv_sess = self.tokenizer = None
            self.loaded = False

    def check_idle_and_unload(self):
        if self.loaded and (time.time() - self.last_used) > MODEL_IDLE_TIMEOUT:
            self.unload()

    # ── KV layout ─────────────────────────────────────────────
    def _infer_kv_layout(self) -> Tuple[int, int, int]:
        idx = set(); nh = hd = None
        for inp in self.decoder_sess.get_inputs():
            m = RE_PAST.match(inp.name)
            if not m:
                continue
            idx.add(int(m.group(1)))
            shape = getattr(inp, "shape", None)
            if shape:
                ints = [s for s in shape if isinstance(s, int)]
                if len(ints) >= 2 and hd is None:
                    hd = ints[-1]
                    for c in ints:
                        if c not in (1, hd):
                            nh = c; break
        return (max(idx)+1 if idx else 24), (nh or 16), (hd or 64)

    # ── encoder KV ────────────────────────────────────────────
    def _encoder_kv_map(self, enc_hs: np.ndarray) -> Dict[str, np.ndarray]:
        enc_names = [n for n in self.decoder_input_names
                     if ".encoder." in n or "encoder.key" in n or "encoder.value" in n]
        kv: Dict[str, np.ndarray] = {}
        if self.encoder_kv_sess:
            try:
                kv_in = self.encoder_kv_sess.get_inputs()[0].name
                outs  = self.encoder_kv_sess.run(None, {kv_in: enc_hs})
                if len(outs) == len(enc_names):
                    return {n: a.astype(np.float32) for n, a in zip(enc_names, outs)}
            except Exception as e:
                _dbg("encoder_kv failed: %s", e)
        # fallback zeros
        b, seq = int(enc_hs.shape[0]), int(enc_hs.shape[1])
        for name in enc_names:
            inp = next((i for i in self.decoder_sess.get_inputs() if i.name == name), None)
            sh  = getattr(inp, "shape", None) if inp else None
            if sh:
                c = []
                for d in sh:
                    if isinstance(d, int): c.append(d)
                    elif not c: c.append(b)
                    else: c.append(seq)
                kv[name] = np.zeros(tuple(c), dtype=np.float32)
            else:
                kv[name] = np.zeros((b, self.num_heads, seq, self.head_dim), dtype=np.float32)
        return kv

    # ── initial past ──────────────────────────────────────────
    def _build_initial_past(self, enc_hs: np.ndarray) -> Dict[str, np.ndarray]:
        enc_kv = self._encoder_kv_map(enc_hs)
        imap: Dict[str, np.ndarray] = {}
        for inp in self.decoder_sess.get_inputs():
            name = inp.name
            if name == "input_ids":
                continue
            if name in enc_kv:
                imap[name] = enc_kv[name]; continue
            if RE_PAST.match(name):
                sh = getattr(inp, "shape", None)
                if sh:
                    c = []
                    for d in sh:
                        if isinstance(d, int): c.append(d)
                        elif not c: c.append(1)
                        elif len(c) == 2: c.append(0)
                        else: c.append(self.head_dim)
                    imap[name] = np.zeros(tuple(c), dtype=np.float32)
                else:
                    imap[name] = np.zeros((1, self.num_heads, 0, self.head_dim), dtype=np.float32)
                continue
            if name == "encoder_hidden_states" or "encoder_hidden_states" in name:
                imap[name] = enc_hs; continue
            if name == "use_cache_branch":
                imap[name] = np.array([True]); continue
            sh = getattr(inp, "shape", None)
            if sh:
                c = [d if isinstance(d, int) else 1 for d in sh]
                dt = np.int64 if "int64" in str(getattr(inp, "type", "")) else np.float32
                imap[name] = np.zeros(tuple(c), dtype=dt)
            else:
                imap[name] = np.array([], dtype=np.float32)
        return imap

    # ── decode (KV cache, token streaming) ────────────────────
    def _decode(self, enc_hs: np.ndarray) -> str:
        """
        Greedy token-by-token decode forced to Hindi.
        Calls self.on_token(partial_text) after each new token for live streaming.
        """
        tokens   = list(FORCED_IDS)
        imap     = self._build_initial_past(enc_hs)
        out_names = self.decoder_output_names
        if not out_names:
            raise RuntimeError("Decoder has no outputs.")

        logits_name = next((n for n in out_names if "logits" in n.lower()), out_names[0])
        eos_id = getattr(self.tokenizer, "eos_token_id", None) if self.tokenizer else None

        for step in range(MAX_DECODING_STEPS):
            run = {
                "input_ids": (
                    np.array([tokens], dtype=np.int64) if step == 0
                    else np.array([[tokens[-1]]], dtype=np.int64)
                )
            }
            run.update(imap)

            outs    = self.decoder_sess.run(None, run)
            out_map = dict(zip(out_names, outs))
            logits  = out_map.get(logits_name, outs[0])
            nl      = logits[0, -1, :].copy()

            if eos_id is not None:
                nl[eos_id] -= 2.0
            if len(tokens) > 8:
                for t in set(tokens[-8:]):
                    nl[t] -= 0.5

            next_tok = int(np.argmax(nl))
            tokens.append(next_tok)

            # ── streaming callback ─────────────────────────────
            if self.on_token is not None and self.tokenizer:
                try:
                    partial = self.tokenizer.decode(tokens, skip_special_tokens=True)
                    self.on_token(partial)
                except Exception:
                    pass

            # update KV cache
            for name, arr in out_map.items():
                m = RE_PRESENT_KV.match(name)
                if m:
                    pname = f"past_key_values.{m.group(1)}.{m.group(2)}.{m.group(3)}"
                    if pname in imap:
                        imap[pname] = arr.astype(np.float32)
                elif RE_PAST.match(name) and name in imap:
                    imap[name] = arr.astype(np.float32)

            if eos_id is not None and next_tok == eos_id:
                _dbg("EOS at step %d", step)
                break

        if self.tokenizer:
            try:
                return self.tokenizer.decode(tokens, skip_special_tokens=True)
            except Exception as e:
                _dbg("Decode error: %s", e)
        return " ".join(str(t) for t in tokens)

    # ── public API ────────────────────────────────────────────
    def transcribe(self, audio_path: str, duration_secs: int = 30) -> str:
        """Transcribe audio to Hindi text. Thread-safe."""
        with self.lock:
            if not self.loaded:
                self.load()
            self.last_used = time.time()

        mel = get_mel(audio_path, duration_secs=duration_secs)
        if not self.encoder_rknn:
            raise RuntimeError("RKNN encoder not loaded.")
        enc_hs = self.encoder_rknn.inference(inputs=[mel])[0]
        text   = self._decode(enc_hs)
        with self.lock:
            self.last_used = time.time()
        logger.info("📝 Transcript (hi): %s", text[:120])
        return text


# ── global STT singleton ──────────────────────────────────────
_stt: Optional[HindiWhisperSTT] = None
_stt_lock = threading.Lock()

def get_stt() -> HindiWhisperSTT:
    global _stt
    with _stt_lock:
        if _stt is None:
            _stt = HindiWhisperSTT()
        return _stt


# ── watchdog ──────────────────────────────────────────────────
def _watchdog():
    while True:
        time.sleep(60)
        try:
            get_stt().check_idle_and_unload()
        except Exception:
            pass

threading.Thread(target=_watchdog, daemon=True, name="stt-watchdog").start()


# ══════════════════════════════════════════════════════════════
#  QWEN-2.5  SINGLETON
# ══════════════════════════════════════════════════════════════

_SYSTEM_PROMPT = (
    "You are a personal assistant with a memory database. "
    "If the user tells you where something is, you MUST provide a line exactly like this: "
    "STORE: <object> | <location>"
    "Example: STORE: keys | kitchen drawer"
)

_USER_TEMPLATE = """\
TRANSCRIPT: {transcript}

RULES:
1. Identify if the user is storing info or asking for it.
2. If storing, extract 'object' and 'location'. 
3. Provide the 'STORE: <object> | <location>' line at the end.
4. If asking, provide 'REPLY: <answer>'.
"""

_qwen_fn   = None
_qwen_lock = threading.Lock()

def _load_qwen_rkllm():
    try:
        from rkllm.api import RKLLM  # type: ignore
        m = RKLLM()
        if m.load_rkllm(QWEN_MODEL_PATH) != 0:
            raise RuntimeError("rkllm load failed")
        m.init_runtime()
        logger.info("✅ Qwen-2.5 loaded via rkllm (NPU).")
        def _run(prompt: str) -> str:
            r = m.run(prompt)
            return r if isinstance(r, str) else str(r)
        def _unload():
            try:
                m.unload()
                logger.info("✅ Qwen-2.5 unloaded via rkllm.")
            except Exception as e:
                logger.warning("Failed to unload rkllm: %s", e)
        return {'run': _run, 'unload': _unload}
    except Exception as e:
        logger.warning("rkllm Qwen load failed: %s — falling back to HTTP.", e)
        return None

def _load_qwen_http():
    import urllib.request
    def _run(prompt: str) -> str:
        payload = json.dumps({
            "model": "Llama3.2",
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            # OpenAI-style
            "max_tokens": QWEN_MAX_TOKENS,
            "temperature": QWEN_TEMPERATURE,
            "stream": False,
            # Ollama/Open-WebUI-style
            "options": {
                "temperature": QWEN_TEMPERATURE,
                "max_new_tokens": QWEN_MAX_TOKENS,
                "stop": None,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            QWEN_API_URL, data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {QWEN_API_KEY}",
                "Connection": "close",
            },
            method="POST",
        )

        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())

        # Accept both OpenAI and Open-WebUI output formats
        if isinstance(data, dict):
            # OpenAI-style
            if "choices" in data and data["choices"]:
                choice = data["choices"][0]
                if isinstance(choice, dict) and "message" in choice and "content" in choice["message"]:
                    return choice["message"]["content"]
                if isinstance(choice, dict) and "text" in choice:
                    return choice["text"]
            # Ollama/Open-WebUI-style
            if "result" in data and isinstance(data["result"], str):
                return data["result"]

        raise RuntimeError(f"Unexpected response from Qwen HTTP API: {data}")

    def _unload():
        try:
            unload_url = QWEN_API_URL.rstrip('/')
            if unload_url.endswith('/v1/chat/completions'):
                unload_url = unload_url[:unload_url.rfind('/v1/chat/completions')]
            response = requests.post(
                f"{unload_url}/unload_model",
                json={"model_name": "Llama3.2"},
                timeout=30,
            )
            if response.ok:
                logger.info("✅ Qwen-2.5 unloaded via API (unload_model), result=%s", response.text)
            else:
                logger.warning("Qwen-2.5 unload_model failed: %s", response.text)
        except Exception as e:
            logger.warning("Qwen-2.5 unload_model request failed: %s", e)

    logger.info("Qwen-2.5 via HTTP (%s).", QWEN_API_URL)
    return {'run': _run, 'unload': _unload}

def get_qwen():
    global _qwen_fn
    with _qwen_lock:
        if _qwen_fn is None:
            _qwen_fn = (_load_qwen_rkllm() if QWEN_BACKEND == "rkllm" else None) or _load_qwen_http()
        return _qwen_fn


def _parse_qwen_text_response(raw: str, transcript: str) -> dict:
    """Parse non-JSON Qwen output for translation/reply/store fields."""
    text = (raw or "").strip()
    reply_text = text
    action = None
    obj = None
    location = None

    # Check for REPLY
    mr = re.search(r"REPLY:\s*(.*?)(?:\n\nSTORE:|$)", text, flags=re.IGNORECASE | re.DOTALL)
    if mr:
        reply_text = mr.group(1).strip()

    # Check for STORE
    ms = re.search(r"STORE:\s*(.*)", text, flags=re.IGNORECASE)
    if ms:
        action = "store"
        store_val = ms.group(1).strip()
        # Split object and location based on the new prompt rule
        if "|" in store_val:
            obj, location = [x.strip() for x in store_val.split("|", 1)]
        else:
            obj = store_val
        
        # Clean up reply text to remove the STORE line
        reply_text = re.sub(r"\n?STORE:\s*.*", "", reply_text, flags=re.IGNORECASE).strip()

    # Fallback for fetch if the quick-check failed
    if not action and _is_memory_query(transcript):
        action = "fetch"
        obj = transcript

    return {
        "action": action,
        "object": obj,
        "location": location,
        "translation": None,
        "reply_text": reply_text or "Sorry, I am having trouble thinking right now.",
    }


def call_qwen(transcript: str, retries: int = 0) -> dict:
    """Send transcript to Qwen-2.5.

    Returns a dict containing at least:
        {"action", "object", "location", "translation", "reply_text"}
    """
    _ensure_stt_released_for_llm()
    msg = _USER_TEMPLATE.format(transcript=transcript)
    qwen = get_qwen()

    for attempt in range(retries + 1):
        try:
            t0  = time.time()
            # unified API for rkllm/http: expects dict with run/unload
            if isinstance(qwen, dict):
                raw = qwen['run'](msg)
            else:
                raw = qwen(msg)
            raw = "" if raw is None else str(raw).strip()
            logger.info("LLM latency: %.2fs (attempt %d)", time.time() - t0, attempt + 1)

            # Always parse as text response since model never returns JSON
            result = _parse_qwen_text_response(raw, transcript)

            # Unload model after response to free NPU for STT
            if isinstance(qwen, dict) and 'unload' in qwen:
                try:
                    qwen['unload']()
                except Exception as unload_exc:
                    logger.warning("Model unload failed: %s", unload_exc)

            return result

        except Exception as e:
            logger.error("Qwen request error (attempt %d): %s", attempt + 1, e)
            if attempt == retries:
                break
            time.sleep(1)
            continue

    # if we fall through due network error, return safe minimal response
    return {
        "action": None,
        "object": None,
        "location": None,
        "translation": None,
        "reply_text": "Sorry, I am having trouble thinking right now.",
    }

# ══════════════════════════════════════════════════════════════
#   MEMORY STORE  (SQLite)
# ══════════════════════════════════════════════════════════════

MEMORY_DB = Path("./assistant_memory.db")
_MEMORY_DB_LOCK = threading.Lock()

# Minimal stopword list for keyword extraction.
_STOPWORDS = {
    "the", "a", "an", "on", "in", "at", "to", "for", "of", "and", "or", "is", "are", "was",
    "were", "it", "my", "your", "you", "i", "me", "that", "this", "these", "those", "have",
    "has", "had", "will", "would", "could", "should", "them", "then", "there", "here", "from",
}


def _get_db_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(MEMORY_DB), check_same_thread=False)
    # Updated table structure
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory_notes (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            action     TEXT,
            object     TEXT NOT NULL,
            location   TEXT,
            timestamp  TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def _normalize_text(s: str) -> str:
    return re.sub(r"\s+", " ", s.strip()).lower()


def _extract_keywords(text: str) -> List[str]:
    words = re.findall(r"\b\w+\b", text.lower())
    return [w for w in dict.fromkeys(words) if w not in _STOPWORDS]


def save_memory_note(action: str, obj: str, location: str) -> int:
    """Store memory with specific structure: action, object, location."""
    if not obj:
        return 0
    # Format: 23-03-2026 12:23 pm
    ts = datetime.datetime.now().strftime("%d-%m-%Y %I:%M %p")
    
    with _MEMORY_DB_LOCK:
        conn = _get_db_conn()
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO memory_notes (action, object, location, timestamp) VALUES (?, ?, ?, ?)",
            (action, obj, location, ts),
        )
        conn.commit()
        return cur.lastrowid


def query_memory(query: str) -> Optional[dict]:
    """Search for object matches in memory."""
    kws = _extract_keywords(query)
    if not kws: return None
    
    with _MEMORY_DB_LOCK:
        conn = _get_db_conn()
        for kw in kws:
            row = conn.execute(
                "SELECT action, object, location, timestamp FROM memory_notes "
                "WHERE object LIKE ? OR location LIKE ? "
                "ORDER BY id DESC LIMIT 1",
                (f"%{kw}%", f"%{kw}%"),
            ).fetchone()
            if row:
                return {"action": row[0], "object": row[1], "location": row[2], "timestamp": row[3]}
    return None


def _release_llama_and_load_whisper():
    """Unload Llama memory model and ensure Whisper STT model is loaded."""
    global _qwen_fn
    with _qwen_lock:
        if _qwen_fn and isinstance(_qwen_fn, dict) and 'unload' in _qwen_fn:
            try:
                _qwen_fn['unload']()
            except Exception as e:
                logger.warning("Failed to unload Qwen/Llama model: %s", e)
        _qwen_fn = None

    try:
        get_stt().load()
        logger.info("✅ Whisper STT reloaded after memory fetch.")
    except Exception as e:
        logger.warning("Failed to reload Whisper STT: %s", e)


def _ensure_stt_released_for_llm():
    """Release STT resources before calling LLM so NPU/CPU is free."""
    try:
        get_stt().unload()
        time.sleep(0.1)
    except Exception:
        pass


def translate_hindi_to_english(text: str) -> str:
    """Return a best-effort English translation using the LLM backend."""
    if not text or not text.strip():
        return ""
    _ensure_stt_released_for_llm()
    try:
        qwen = get_qwen()
        prompt = (
            "Translate the following Hindi transcript to English. "
            "Output ONLY the English translation (no explanation).\n\n"
            f"{text.strip()}"
        )

        if isinstance(qwen, dict):
            raw = qwen["run"](prompt)
        else:
            raw = qwen(prompt)

        raw_s = "" if raw is None else str(raw).strip()
        # we do not unload here because call_qwen handles unload after structured response
        return raw_s
    except Exception as e:
        logger.warning("Translation call failed: %s", e)
        return ""


def _is_memory_query(text: str) -> bool:
    """Return True if text is likely a memory retrieval query (e.g. "where is my key", "I forgot where...")."""
    t = text.lower().strip()
    # explicit retrieval hints
    if any(w in t for w in ("where", "forgot", "forgotten", "lost", "can't find", "cannot find", "remember")):
        return True
    # fallback by items
    if any(k in t for k in ("key", "keys", "wallet", "phone", "glasses", "bag", "book", "ticket", "passport", "wallet")):
        return True
    return False


# ══════════════════════════════════════════════════════════════════════
#  WebSocket event bus  (set by web_ui.py)
# ══════════════════════════════════════════════════════════════

_ws_push_fn = None   # callable(dict) → None, injected by web_ui

def push_event(event: dict):
    if _ws_push_fn:
        try:
            _ws_push_fn(event)
        except Exception as e:
            logger.warning("WS push error: %s", e)


# ══════════════════════════════════════════════════════════════
#  MAIN CALLABLE
# ══════════════════════════════════════════════════════════════

def process_audio(audio_path: str, duration_secs: int = 30) -> dict:
    """
    Transcribe *audio_path* (Hindi-forced) then get Qwen-2.5 response.

    Returns:
        {
            "transcript":    str,
            "qwen_response": str,
            "memory_note":   str | None,
        }

    Streams partial transcript tokens via push_event({"type":"token","text":"..."})
    and the final result via push_event({"type":"final",...}).
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    logger.info("== process_audio: %s ==", audio_path)

    stt = get_stt()

    # Wire streaming callback → WebSocket
    partial_buf = [""]
    def _on_token(partial: str):
        partial_buf[0] = partial
        push_event({"type": "token", "text": partial})

    stt.on_token = _on_token

    try:
        t0         = time.time()
        transcript = stt.transcribe(audio_path, duration_secs=duration_secs)
        logger.info("STT done in %.2fs", time.time() - t0)

        # Unload Whisper encoders/decoders before passing transcript to Qwen/RKLLM
        # so the NPU is free for language model inference.
        try:
            stt.unload()
            logger.info("Whisper STT model unloaded before LLM call.")
        except Exception as e:
            logger.warning("Failed to unload Whisper STT model: %s", e)
    finally:
        stt.on_token = None   # always detach

    # translate first (fallback to a direct translation call if JSON translation isn't available)
    translation = translate_hindi_to_english(transcript)

    # If this is a direct memory retrieval query, hit local DB first for speed
    if _is_memory_query(translation):
        found = query_memory(translation)
        if found:
            # Reconstruct the sentence from the structured columns
            obj = found.get('object', 'it')
            loc = found.get('location', 'somewhere')
            time_str = found.get('timestamp', 'recently')
            
            # Format the output for the UI
            memory_text = f"{obj} is in {loc} (saved on {time_str})"
            qwen_reply = f"I found this: {memory_text}"
            qwen_reply = f"Translation: {translation}\n{qwen_reply}"
            
            _release_llama_and_load_whisper()
            result = {
                "transcript": transcript,
                "translation": translation,
                "qwen_response": qwen_reply,
                "memory_note": memory_text,
            }
            push_event({"type": "final", **result})
            return result

    # Ask the assistant for structured output
    push_event({"type": "status", "text": "Jarvis is thinking…"})
    t0 = time.time()
    structured = call_qwen(translation)
    logger.info("Qwen done in %.2fs", time.time() - t0)

    # Extract data from Qwen's response
    action = (structured.get("action") or "").lower()
    obj = structured.get("object")
    location = structured.get("location")
    qwen_reply = structured.get("reply_text") or ""
    
    display_note = None  # Prevents UnboundLocalError crash

    # Logic for Storing
    if action == "store" and obj:
        location_str = location or "unknown location"
        display_note = f"{obj} in {location_str}"
        
        # Save to DB with proper schema
        save_memory_note(action, obj, location_str)
        
        # Override reply for UI as requested
        qwen_reply = f"STORE: {display_note}"

    # Logic for LLM-driven Fetching (Fallback if _is_memory_query missed it)
    elif action == "fetch" and obj:
        found = query_memory(obj)
        if found:
            obj_found = found.get('object', 'it')
            loc_found = found.get('location', 'somewhere')
            time_str = found.get('timestamp', 'recently')
            
            memory_text = f"{obj_found} is in {loc_found} (saved on {time_str})"
            qwen_reply = f"I remember! {memory_text}"
            _release_llama_and_load_whisper()
        else:
            qwen_reply = qwen_reply or "I don't have that in memory yet."

    qwen_reply = f"Translation: {translation}\n{qwen_reply}"

    result = {
        "transcript": transcript,
        "translation": translation,
        "qwen_response": qwen_reply,
        "memory_note": display_note,
    }
    push_event({"type": "final", **result})
    return result
