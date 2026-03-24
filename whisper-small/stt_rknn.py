#!/usr/bin/env python3
"""
Offline Speech-to-Text for CM3588 (RK3588 NPU) using Whisper medium RKNN.
Matches the expectations set in prompt.md.

Requirements:
- rknnlite
- onnxruntime
- transformers
- librosa
- pydub (optional, for better silence trimming)
- soundfile
- numpy

Usage:
  python3 stt_rknn.py --file audio.wav
  python3 stt_rknn.py --folder ./audio_samples/
"""

import os
import re
import time
import argparse
import logging
from typing import List, Optional, Tuple
from pathlib import Path

import numpy as np
import librosa
from transformers import WhisperTokenizer

try:
    from rknnlite.api import RKNNLite
    _RKNN_AVAILABLE = True
except ImportError:
    _RKNN_AVAILABLE = False
    print("WARNING: rknnlite not found. NPU inference won't work.")

try:
    from pydub import AudioSegment
    from pydub.silence import detect_nonsilent
    _PYDUB_AVAILABLE = True
except ImportError:
    _PYDUB_AVAILABLE = False
    print("WARNING: pydub not found. Falling back to librosa for silence trimming.")

# -----------------------------------------------------------------------------
# Configuration parameters
# -----------------------------------------------------------------------------

# --- UPDATED: Point to the whisper-small directory ---
MODEL_DIR              = Path("/home/smadmin/rknn-model/model/whisper-small")
ENCODER_RKNN_PATH      = MODEL_DIR / "encoder_model_fp16_rk3588.rknn"
DECODER_MAIN_RKNN_PATH = MODEL_DIR / "decoder_model_fp16_rk3588.rknn"
DECODER_PAST_RKNN_PATH = MODEL_DIR / "decoder_with_past_model_fp16_rk3588.rknn"

RKNN_CORE_MASK = RKNNLite.NPU_CORE_0 if _RKNN_AVAILABLE else None

SAMPLE_RATE = 16000
CHUNK_SEC = 25  
OVERLAP_SEC = 1.0

SILENCE_THRESH_DB = -40
SILENCE_MIN_LEN_MS = 300
MAX_DECODING_STEPS = 448

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Core audio processing functions
# -----------------------------------------------------------------------------

def trim_silence(audio_path: str) -> Tuple[np.ndarray, float]:
    logger.info(f"Loading and trimming silence: {audio_path}")
    t0 = time.time()
    orig_dur = librosa.get_duration(path=audio_path)

    if _PYDUB_AVAILABLE:
        seg = AudioSegment.from_file(audio_path)
        seg = seg.set_channels(1).set_frame_rate(SAMPLE_RATE)
        
        nonsilent = detect_nonsilent(
            seg, min_silence_len=SILENCE_MIN_LEN_MS, silence_thresh=SILENCE_THRESH_DB
        )
        if not nonsilent:
            logger.warning("Entire audio appears silent. Skipping trim.")
            np_audio = np.array(seg.get_array_of_samples(), dtype=np.float32) / 32768.0
            return np_audio, orig_dur
            
        trimmed = seg[nonsilent[0][0]: nonsilent[-1][1]]
        np_audio = np.array(trimmed.get_array_of_samples(), dtype=np.float32) / 32768.0
        
        logger.info(f"Trimmed (pydub): {orig_dur:.1f}s -> {len(trimmed)/1000.0:.1f}s [{time.time() - t0:.2f}s]")
        return np_audio, orig_dur
    else:
        audio, _ = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)
        trimmed_audio, _ = librosa.effects.trim(
            audio, top_db=abs(SILENCE_THRESH_DB), frame_length=512, hop_length=128
        )
        logger.info(f"Trimmed (librosa): {orig_dur:.1f}s -> {len(trimmed_audio)/SAMPLE_RATE:.1f}s [{time.time() - t0:.2f}s]")
        return trimmed_audio, orig_dur

def create_chunks(audio_np: np.ndarray) -> List[np.ndarray]:
    total_samples = len(audio_np)
    chunk_samples = int(CHUNK_SEC * SAMPLE_RATE)
    overlap_samples = int(OVERLAP_SEC * SAMPLE_RATE)
    step = chunk_samples - overlap_samples

    chunks = []
    start = 0
    while start < total_samples:
        end = min(start + chunk_samples, total_samples)
        chunks.append(audio_np[start:end])
        if end == total_samples:
            break
        start += step
        
    logger.info(f"Chunking: Created {len(chunks)} chunk(s).")
    return chunks

def extract_mel_spectrogram(audio_np: np.ndarray) -> np.ndarray:
    desired_length = SAMPLE_RATE * 30
    audio_padded = librosa.util.fix_length(audio_np, size=desired_length)
    
    stft = librosa.stft(audio_padded, n_fft=400, hop_length=160, window="hann", center=True)
    magnitudes = np.abs(stft[:, :-1]) ** 2
    mel_filters = librosa.filters.mel(sr=SAMPLE_RATE, n_fft=400, n_mels=80)
    mel_spec = mel_filters.dot(magnitudes)
    
    log_spec = np.log10(np.maximum(mel_spec, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    
    return log_spec.astype(np.float32)[np.newaxis, :]

def pre_detect_language(audio_path: str) -> str:
    return "auto"

# -----------------------------------------------------------------------------
# RKNN Model Loading and Inference
# -----------------------------------------------------------------------------

class RKNNWhisperSTT:
    def __init__(self):
        self.encoder = None
        self.decoder_main_sess = None
        self.decoder_past_sess = None
        self.tokenizer = None
        
        # --- UPDATED: Whisper Small Architecture ---
        self.num_layers = 12
        self.num_heads = 12
        self.head_dim = 64
        # -------------------------------------------

    def load_models(self):
        logger.info("Loading STT models into RAM/NPU...")
        
        if not MODEL_DIR.exists():
            raise FileNotFoundError(f"MODEL_DIR not found: {MODEL_DIR}. Please check configuration.")
        
        self.tokenizer = WhisperTokenizer.from_pretrained(str(MODEL_DIR))

        # 1. Load Encoder
        if _RKNN_AVAILABLE and ENCODER_RKNN_PATH.exists():
            self.encoder = RKNNLite()
            self.encoder.load_rknn(str(ENCODER_RKNN_PATH))
            self.encoder.init_runtime(core_mask=RKNN_CORE_MASK)
            logger.info("RKNN Encoder initialized.")
        else:
            raise RuntimeError(f"RKNN encoder not found.")

        # 2. Load Decoder Main (Prefill)
        if _RKNN_AVAILABLE and DECODER_MAIN_RKNN_PATH.exists():
            self.decoder_main_sess = RKNNLite()
            self.decoder_main_sess.load_rknn(str(DECODER_MAIN_RKNN_PATH))
            self.decoder_main_sess.init_runtime(core_mask=RKNN_CORE_MASK)
            logger.info("RKNN Decoder Main (Prefill) initialized.")
        else:
            raise RuntimeError(f"RKNN decoder_main not found.")

        # 3. Load Decoder with Past (Generation)
        if _RKNN_AVAILABLE and DECODER_PAST_RKNN_PATH.exists():
            self.decoder_past_sess = RKNNLite()
            self.decoder_past_sess.load_rknn(str(DECODER_PAST_RKNN_PATH))
            self.decoder_past_sess.init_runtime(core_mask=RKNN_CORE_MASK)
            logger.info("RKNN Decoder with Past (Generation) initialized.")
        else:
            raise RuntimeError(f"RKNN decoder_model_fp16 not found.")

    def _decode_loop(self, enc_hs: np.ndarray, initial_tokens: List[int]) -> Tuple[str, str]:
        eos_id = self.tokenizer.eos_token_id if self.tokenizer else 50257
        
        # ==========================================
        # 1. PREFILL PHASE (decoder_model_fp16)
        # ==========================================
        prefill_tokens = list(initial_tokens)
        
        while len(prefill_tokens) < 5:
            prefill_tokens.insert(0, 50258)
        prefill_tokens = prefill_tokens[:5] 
        
        input_ids = np.array([prefill_tokens], dtype=np.int64)
        
        outputs = self.decoder_main_sess.inference(inputs=[input_ids, enc_hs])
        logits = outputs[0]
        
        decoder_past_keys = []
        decoder_past_values = []
        encoder_past_keys = []
        encoder_past_values = []
        
        # --- FIX: Safely rotate array layouts to NHWC for the NPU ---
        def to_nhwc(arr):
            # If shape is (Batch, Heads, Seq, HeadDim) -> change to (Batch, Seq, HeadDim, Heads)
            if len(arr.shape) == 4 and arr.shape[1] == self.num_heads and arr.shape[3] == self.head_dim:
                return np.transpose(arr, (0, 2, 3, 1))
            return arr
        # ------------------------------------------------------------
        
        for i in range(self.num_layers):
            decoder_past_keys.append(to_nhwc(outputs[1 + 4*i]))
            decoder_past_values.append(to_nhwc(outputs[2 + 4*i]))
            encoder_past_keys.append(to_nhwc(outputs[3 + 4*i]))
            encoder_past_values.append(to_nhwc(outputs[4 + 4*i]))
            
        tokens = list(prefill_tokens)
        detected_lang_token = None

        # ==========================================
        # 2. GENERATION PHASE (decoder_with_past_model)
        # ==========================================
        max_allowed_steps = min(MAX_DECODING_STEPS, 64)
        remaining_steps = max_allowed_steps - len(prefill_tokens)
        
        for step in range(remaining_steps):
            next_logits = logits[0, -1, :].copy()
            
            if eos_id is not None:
                next_logits[eos_id] -= 2.0
                
            if len(tokens) > 8:
                for t in set(tokens[-8:]):
                    next_logits[t] -= 0.5
                    
            next_tok = int(np.argmax(next_logits))
            tokens.append(next_tok)
            
            if step == 0:
                for t in reversed(prefill_tokens):
                    if 50259 <= t <= 50357:
                        detected_lang_token = t
                        break
                if not detected_lang_token:
                    detected_lang_token = next_tok
                
            if next_tok == eos_id:
                break
                
            input_ids = np.array([[next_tok]], dtype=np.int64)
            
            run_inputs = [input_ids]
            for i in range(self.num_layers):
                run_inputs.append(decoder_past_keys[i])
                run_inputs.append(decoder_past_values[i])
                run_inputs.append(encoder_past_keys[i])
                run_inputs.append(encoder_past_values[i])
                
            outputs = self.decoder_past_sess.inference(inputs=run_inputs)
            logits = outputs[0]
            
            # Apply the rotation to the newly generated caches too
            for i in range(self.num_layers):
                decoder_past_keys[i] = to_nhwc(outputs[1 + 2*i])
                decoder_past_values[i] = to_nhwc(outputs[2 + 2*i])
                
        lang = "unknown"
        if detected_lang_token is not None:
            decoded_lang = self.tokenizer.decode([detected_lang_token]).replace("<|", "").replace("|>", "")
            lang = decoded_lang.strip()
            
        transcription = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return transcription, lang

    def transcribe_chunk(self, audio_np: np.ndarray, force_lang: Optional[str] = None) -> Tuple[str, str]:
        mel = extract_mel_spectrogram(audio_np)
        enc_out = self.encoder.inference(inputs=[mel])[0]
        
        SOT = 50258
        TRANSCRIBE = 50359
        NOTIMESTAMPS = 50363
        
        if force_lang == "hi":
            tokens = [SOT, 50275, TRANSCRIBE, NOTIMESTAMPS]
        elif force_lang == "en":
            tokens = [SOT, 50259, TRANSCRIBE, NOTIMESTAMPS]
        else:
            tokens = [SOT]

        text, det_lang = self._decode_loop(enc_out, tokens)
        
        if force_lang:
            det_lang = force_lang
            
        return text.strip(), det_lang

# -----------------------------------------------------------------------------
# Stitching helper
# -----------------------------------------------------------------------------

def merge_transcripts(chunks_texts: List[str]) -> str:
    parts = []
    for text in chunks_texts:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            continue
        if parts:
            prev_words = parts[-1].split()
            cur_words  = text.split()
            if prev_words and cur_words and prev_words[-1] == cur_words[0]:
                text = " ".join(cur_words[1:])
        if text:
            parts.append(text)
    return " ".join(parts)

def process_audio(engine: RKNNWhisperSTT, path: str):
    logger.info(f"--- Processing {path} ---")
    
    np_audio, orig_dur = trim_silence(path)
    chunks = create_chunks(np_audio)
    
    chunk_texts = []
    overall_lang = "unknown"
    
    for i, c in enumerate(chunks):
        logger.info(f"Inferencing chunk {i+1}/{len(chunks)}...")
        
        force_lang = overall_lang if overall_lang != "unknown" else None
        text, iter_lang = engine.transcribe_chunk(c, force_lang=force_lang)
        
        if iter_lang != "unknown" and overall_lang == "unknown":
            overall_lang = iter_lang
            logger.info(f"Detected language: {overall_lang}")
            
        chunk_texts.append(text)
        logger.info(f"Chunk {i+1} Output: {text}")
        
    final_text = merge_transcripts(chunk_texts)
    
    logger.info("=====================================")
    logger.info(f"File: {path}")
    logger.info(f"Detected Language: {overall_lang}")
    logger.info(f"Final Transcript: {final_text}")
    logger.info("=====================================\n")

def main():
    parser = argparse.ArgumentParser(description="Offline STT using Whisper RKNN.")
    parser.add_argument("--file", type=str, help="Path to a single audio file to transcribe.")
    parser.add_argument("--folder", type=str, help="Path to a folder containing audio files.")
    
    args = parser.parse_args()
    
    if not args.file and not args.folder:
        parser.print_help()
        return
        
    engine = RKNNWhisperSTT()
    engine.load_models()
    
    if args.file:
        if not os.path.exists(args.file):
            logger.error(f"File not found: {args.file}")
            return
        process_audio(engine, args.file)
        
    elif args.folder:
        if not os.path.exists(args.folder) or not os.path.isdir(args.folder):
            logger.error(f"Directory not found: {args.folder}")
            return
            
        valid_exts = {".wav", ".mp3", ".ogg", ".flac", ".m4a"}
        for f in sorted(os.listdir(args.folder)):
            ext = os.path.splitext(f)[1].lower()
            if ext in valid_exts:
                full_path = os.path.join(args.folder, f)
                process_audio(engine, full_path)

if __name__ == "__main__":
    main()