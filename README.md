# rknn-whisper-STT
CM3588 whisper-STT inference scripts

## rknn npu load
```bash
watch sudo cat /sys/kernel/debug/rknpu/load
```

### RKNN toolkit setup

```bash
sudo apt-get install -y python3 python3-dev python3-venv python3-pip \
    gcc libxslt1-dev zlib1g zlib1g-dev libglib2.0-0 libsm6 \
    libgl1-mesa-glx libprotobuf-dev

sudo apt-get install -y python3-dev python3-pip python3-numpy python3-opencv gcc portaudio19-dev
```

1. Create a Python Virtual Environment: This helps manage dependencies and avoids conflicts with other projects.

```bash
python3 -m venv rkllm_env
```

- activate the environment
```bash
source rkllm_env/bin/activate
```

- For exiting the virtual environment Deactivate command
```bash
deactivate
```

2. install python dependecies

```bash
python3 -m pip install opencv-python-headless
pip install kaldi_native_fbank onnxruntime sentencepiece soundfile pyyaml "numpy<2"
pip3 install pyaudio openwakeword onnxruntime torch torchvision torchaudio numpy soundfile librosa
```

### Rknn runtime api setup

clone rknn-toolkit2 
```bash
cd rknn-toolkit2/rknn_toolkit_lite2/packages/
```

- install the package wheel according to your python version (install all modules in venv only)

```bash
pip install ./rknn_toolkit_lite2-2.3.2-cp39-cp39-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

cd rknn-toolkit2/rknpu2/runtime/Linux/librknn_api/aarch64

cp ./librknnrt.so /usr/lib/
```

### python pip3 rknn-toolkit-lite2 setup

```bash
pip3 install rknn-toolkit-lite2
```

### verify the python runtime

```bash
python3 - << 'EOF'
try:
    # Correct import for rknn-toolkit-lite2
    from rknnlite.api import RKNNLite
    print("Python runtime OK")

    import ctypes
    # Ensure this library is in your LD_LIBRARY_PATH
    ctypes.CDLL("librknnrt.so")
    print("Native librknnrt loaded OK")

    # Initialize the runtime
    rknn = RKNNLite()
except Exception as e:
    print("ERROR:", e)
EOF
```

## Whisper small

Model Used: https://huggingface.co/danielferr85/whisper-with-past-models-rknn/tree/main/whisper-small

- Download all the model and tokenizer and Transfer it in CM3588.

## Local AI Assistant

Give all the audio/video access to user
```bash
sudo usermod -aG audio,video,render,pulse-access <username>
```

- Download the models used for openwake word

wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_jarvis_v0.1.onnx

wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx

wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx

- Transfer the require indic-whisper model in CM3588.
