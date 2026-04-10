# patched_stt_test_v3.py
# Based on your original stt_test_v3.py with KV-cache (decoder_with_past) support.
# Two important notes:
# 1) If your decoder_with_past ONNX requires per-layer encoder key/value tensors,
#    you should export an encoder_kv.onnx (or similar) that takes encoder_hidden_states
#    and returns encoder key/value arrays for every decoder layer. If you provide that
#    file at ENCODER_KV_ONNX_PATH, this script will use it automatically.
# 2) If encoder KV are not provided, this script will fall back to zero-filled encoder KV
#    (which will *not* produce correct transcripts because cross-attention gets no context).
#
# Usage: place your decoder_with_past_model.onnx at DECODER_PATH_WITH_PAST or let the
#       script pick the original decoder (DECODER_PATH) if it still accepts
#       encoder_hidden_states.

import os
import re
import numpy as np
import librosa
import onnxruntime as ort
from rknnlite.api import RKNNLite
from transformers import WhisperTokenizer

# --- Paths (adjust if needed) ---
ENCODER_PATH = "./indic-whisper/encoder_model.rknn"
DECODER_PATH = "./indic-whisper/decoder_model.onnx"                 # original single-call decoder (no past)
DECODER_WITH_PAST_PATH = "./indic-whisper/decoder_with_past_model.onnx"  # your decoder_with_past_model.onnx
ENCODER_KV_ONNX_PATH = "./indic-whisper/encoder_kv.onnx"          # optional helper ONNX to produce per-layer encoder KV
MODEL_DIR = "./indic-whisper"
AUDIO_PATH = "../hindi.mp3"

tokenizer = WhisperTokenizer.from_pretrained(MODEL_DIR)

def get_whisper_mel_v2(audio_path, duration_seconds=30):
    audio, _ = librosa.load(audio_path, sr=16000)
    audio = librosa.util.fix_length(audio, size=16000 * duration_seconds)
    stft = librosa.stft(audio, n_fft=400, hop_length=160, window='hann', center=True)
    magnitudes = np.abs(stft[:, :-1]) ** 2
    mel_filters = librosa.filters.mel(sr=16000, n_fft=400, n_mels=80)
    mel_spec = np.dot(mel_filters, magnitudes)
    log_spec = np.log10(np.maximum(mel_spec, 1e-10))
    log_spec = np.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec.astype(np.float32)[np.newaxis, :]

# --- Helpers to inspect ONNX signature and create past dicts ---
def detect_decoder_mode(decoder_sess):
    """
    Returns 'with_past' if session input names contain past_key_values.*,
    otherwise 'regular' if it accepts 'encoder_hidden_states'.
    """
    input_names = [inp.name for inp in decoder_sess.get_inputs()]
    if any(re.match(r"past_key_values\.\d+\.decoder\.key", n) for n in input_names):
        return "with_past"
    if "encoder_hidden_states" in input_names:
        return "regular"
    # fallback
    return "unknown"

def infer_kv_layout(decoder_sess):
    """
    Inspect input names to infer num_layers, num_heads, head_dim if possible.
    Returns (num_layers, num_heads, head_dim)
    """
    input_names = [inp.name for inp in decoder_sess.get_inputs()]
    layer_indices = set()
    num_heads = None
    head_dim = None
    for inp in decoder_sess.get_inputs():
        m = re.match(r"past_key_values\.(\d+)\.decoder\.key", inp.name)
        if m:
            layer_indices.add(int(m.group(1)))
            # try to parse static shape
            shape = getattr(inp, 'shape', None)
            if shape is not None:
                # shape often like ['batch_size', 16, 'past_decoder_sequence_length', 64]
                # pick the first integer we find after batch dim -> num_heads
                ints = [s for s in shape if isinstance(s, int)]
                if len(ints) >= 2:
                    num_heads = ints[0] if num_heads is None else num_heads
                    head_dim = ints[1] if head_dim is None else head_dim
                elif len(ints) == 1 and num_heads is None:
                    # maybe only one int (head dim) is provided
                    head_dim = ints[0]
    num_layers = max(layer_indices) + 1 if layer_indices else 0
    # fallbacks to common Whisper defaults
    if num_heads is None:
        num_heads = 16
    if head_dim is None:
        head_dim = 64
    return num_layers, int(num_heads), int(head_dim)

def build_initial_past_inputs(decoder_sess, encoder_hidden_states):
    """
    Build the dictionary of input name -> numpy array for the decoder_with_past model.
    If ENCODER_KV_ONNX_PATH exists, use it to compute per-layer encoder key/value arrays.
    Otherwise create zeros for encoder KV (warning: that will likely break decoding).
    """
    num_layers, num_heads, head_dim = infer_kv_layout(decoder_sess)
    batch = int(encoder_hidden_states.shape[0])
    encoder_seq_len = int(encoder_hidden_states.shape[1])

    # create empty decoder past (past_decoder_sequence_length = 0)
    empty_decoder_past = np.zeros((batch, num_heads, 0, head_dim), dtype=np.float32)
    # default encoder kv: zeros until we can build them properly
    default_encoder_kv = np.zeros((batch, num_heads, encoder_seq_len, head_dim), dtype=np.float32)

    # try to compute encoder KV via helper ONNX if available
    encoder_kv_map = None
    if os.path.isfile(ENCODER_KV_ONNX_PATH):
        try:
            kv_sess = ort.InferenceSession(ENCODER_KV_ONNX_PATH, ort.SessionOptions(), providers=['CPUExecutionProvider'])
            # assume this helper accepts encoder_hidden_states input named 'encoder_hidden_states'
            kv_inputs = {kv_sess.get_inputs()[0].name: encoder_hidden_states}
            kv_outputs = kv_sess.run(None, kv_inputs)
            # Expected that kv_outputs is a list matching the past_key_values.*.encoder.key/value ordering
            # We will map outputs to names in decoder_sess.get_inputs() that match '.encoder.key' and '.encoder.value'
            encoder_kv_map = {}
            enc_input_names = [inp.name for inp in decoder_sess.get_inputs() if '.encoder.' in inp.name]
            if len(kv_outputs) == len(enc_input_names):
                for name, arr in zip(enc_input_names, kv_outputs):
                    encoder_kv_map[name] = arr.astype(np.float32)
                print("Loaded encoder KV from encoder_kv.onnx.")
            else:
                print("encoder_kv.onnx ran but returned unexpected number of outputs; falling back to zeros for encoder KV.")
                encoder_kv_map = None
        except Exception as e:
            print("Failed to run encoder_kv.onnx:", e)
            encoder_kv_map = None

    # Build input dict
    input_dict = {}
    for inp in decoder_sess.get_inputs():
        n = inp.name
        if n == 'input_ids':
            # this will be provided per-step
            continue
        if 'decoder.key' in n and 'past_key_values' in n:
            input_dict[n] = empty_decoder_past
        elif 'decoder.value' in n and 'past_key_values' in n:
            input_dict[n] = empty_decoder_past
        elif '.encoder.key' in n or '.encoder.value' in n:
            if encoder_kv_map is not None and n in encoder_kv_map:
                input_dict[n] = encoder_kv_map[n].astype(np.float32)
            else:
                # fallback zeros (not ideal)
                input_dict[n] = default_encoder_kv
        elif n == 'encoder_hidden_states':
            # older decoder expects this instead
            input_dict[n] = encoder_hidden_states
        else:
            # any other input -> try to create a sensible default (zeros)
            # if it's an int64 input like attention_mask or something, leave it alone for now
            # but supply zeros to prevent runtime errors for optional inputs
            shape = getattr(inp, 'shape', None)
            if shape is None:
                input_dict[n] = np.array([], dtype=np.float32)
            else:
                # replace symbolic dims with 1
                concrete_shape = [1 if (not isinstance(d, int) or d is None) else d for d in shape]
                dtype = np.float32
                if inp.type == 'tensor(int64)':
                    dtype = np.int64
                input_dict[n] = np.zeros(concrete_shape, dtype=dtype)
    return input_dict, num_layers, num_heads, head_dim

# --- Inference setup ---
rknn = RKNNLite()
rknn.load_rknn(ENCODER_PATH)
rknn.init_runtime(core_mask=RKNNLite.NPU_CORE_0)

# Prepare ONNX runtime options (CPU)
sess_options = ort.SessionOptions()
sess_options.intra_op_num_threads = 4
sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# Create decoder sessions
decoder_sess = None
use_with_past = False
if os.path.isfile(DECODER_WITH_PAST_PATH):
    print("Loading decoder (with past) ONNX:", DECODER_WITH_PAST_PATH)
    decoder_sess = ort.InferenceSession(DECODER_WITH_PAST_PATH, sess_options, providers=['CPUExecutionProvider'])
    mode = detect_decoder_mode(decoder_sess)
    if mode == 'with_past':
        use_with_past = True
    elif mode == 'regular':
        # some exported decoders expose both interfaces: treat as regular
        use_with_past = False
    else:
        # ambiguous: treat as with_past but be cautious
        use_with_past = True
else:
    print("decoder_with_past.onnx not found; falling back to DECODER_PATH")
    decoder_sess = ort.InferenceSession(DECODER_PATH, sess_options, providers=['CPUExecutionProvider'])
    use_with_past = False

print("Running NPU Encoder (RKNN)...")
mel_input = get_whisper_mel_v2(AUDIO_PATH)
encoder_output = rknn.inference(inputs=[mel_input])[0]    # shape: [batch, encoder_seq_len, hidden_dim]

# Prepare forced tokens (AI4Bharat Hindi example from your script)
tokens = [50258, 50275, 50359, 50363]
max_len = 448

if not use_with_past:
    # Original (no KV-cache) loop (keeps sending the whole token history each step)
    print("Decoding using regular decoder ONNX (no KV-cache)...")
    for i in range(max_len):
        onnx_inputs = {
            'input_ids': np.array([tokens], dtype=np.int64),
            'encoder_hidden_states': encoder_output
        }
        logits = decoder_sess.run(None, onnx_inputs)[0]
        next_token_logits = logits[0, -1, :]
        # repetition penalty
        if len(tokens) > 5:
            for t in set(tokens[-5:]):
                next_token_logits[t] -= 2.0
        next_token_logits[tokenizer.eos_token_id] -= 5.0
        next_token = int(np.argmax(next_token_logits))
        tokens.append(next_token)
        if next_token == tokenizer.eos_token_id:
            print(f"Reached end of speech at token {i}")
            break
else:
    # KV-cache loop: supply past_key_values.* inputs and update them from present.* outputs
    print("Decoding using decoder_with_past ONNX (KV-cache)...")
    # Build initial past inputs
    initial_input_map, num_layers, num_heads, head_dim = build_initial_past_inputs(decoder_sess, encoder_output)
    print(f"Inferred layout: num_layers={num_layers}, num_heads={num_heads}, head_dim={head_dim}")
    # Determine ordered lists of past input names and present output names so we can map them
    past_input_names = [inp.name for inp in decoder_sess.get_inputs() if inp.name != 'input_ids']
    present_output_names = [out.name for out in decoder_sess.get_outputs() if out.name.startswith('present')]

    # For token-by-token, pass only the last token as input_ids (sequence length = 1)
    # Seed input_ids with the forced tokens so far (we will feed the last token only each step)
    # Ensure initial_input_map contains everything except input_ids
    # We'll iteratively update initial_input_map with the 'present.*' values returned by the model
    input_map = {k: v for k, v in initial_input_map.items()}

    # If the decoder expects 'encoder_hidden_states' instead, make sure to provide it
    if 'encoder_hidden_states' in [inp.name for inp in decoder_sess.get_inputs()]:
        input_map['encoder_hidden_states'] = encoder_output

    # Start decoding loop
    for step in range(max_len):
        last_token = np.array([[tokens[-1]]], dtype=np.int64)
        run_inputs = {'input_ids': last_token}
        # attach all past-like inputs
        run_inputs.update(input_map)

        outs = decoder_sess.run(None, run_inputs)
        # map outputs by name
        out_names = [o.name for o in decoder_sess.get_outputs()]
        out_map = {name: arr for name, arr in zip(out_names, outs)}

        # logits is normally named 'logits'
        if 'logits' not in out_map:
            # try index 0 as logits
            logits = outs[0]
            logits_name = out_names[0]
        else:
            logits = out_map['logits']
            logits_name = 'logits'

        next_token_logits = logits[0, -1, :]
        # apply repetition penalty using recent tokens (local)
        if len(tokens) > 5:
            for t in set(tokens[-5:]):
                next_token_logits[t] -= 2.0
        next_token_logits[tokenizer.eos_token_id] -= 5.0
        next_token = int(np.argmax(next_token_logits))
        tokens.append(next_token)

        # update input_map with present.* outputs to be used as past in next step
        # present outputs are named like present.{layer}.decoder.key etc
        for name, arr in out_map.items():
            if name.startswith('present'):
                # use this present as the next past input (names should match)
                # The decoder input names used 'past_key_values.X.decoder.key' etc.
                # We build a mapping: present.N.decoder.key  -> past_key_values.N.decoder.key
                m = re.match(r"present\.(\d+)\.(decoder|encoder)\.(key|value)", name)
                if m:
                    idx, which, kv = m.group(1), m.group(2), m.group(3)
                    past_name = f"past_key_values.{idx}.{which}.{kv}"
                    # store as the next past
                    input_map[past_name] = arr.astype(np.float32)
        if next_token == tokenizer.eos_token_id:
            print(f"Reached end of speech at step {step}")
            break

final_text = tokenizer.decode(tokens, skip_special_tokens=True)
print(f"\n--- Full Transcription ---\n{final_text}")

rknn.release()