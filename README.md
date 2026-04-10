# rknn-whisper-STT
CM3588 whisper-STT inference scripts

## rknn npu load
```bash
watch sudo cat /sys/kernel/debug/rknpu/load
```

Give all the audio/video access to user
```bash
sudo usermod -aG adm,dialout,cdrom,sudo,audio,dip,video,plugdev,staff,systemd-journal,input,render,bluetooth,netdev,pulse-access smadmin
```

---

## Create a disk swap → use **zram + small disk swap**

---

## ✅ Install zram

```bash
sudo apt install zram-tools
```

Edit:

```bash
sudo nano /etc/default/zramswap
```

Set:

```ini
ALGO=lz4
PERCENT=50
```

Restart:

```bash
sudo systemctl restart zramswap
```

Verify:
```bash
htop
```

---

### RKNN toolkit setup

```bash
sudo apt-get install -y python3 python3.11-dev python3-dev portaudio19-dev python3-pip python3-numpy \
    gcc libxslt1-dev zlib1g zlib1g-dev libglib2.0-0 libsm6 \
    libgl1-mesa-glx libprotobuf-dev build-essential
```

Check python3 version. It should be 3.11.X

```bash
python3 -V
```

If version is less than 3.11.x than install 3.11 version, If version already 3.11.x then skip following steps

Build Python 3.11 from source in /usr/local

```bash
cd /usr/src
sudo wget https://www.python.org/ftp/python/3.11.10/Python-3.11.10.tar.xz
sudo tar -xf Python-3.11.10.tar.xz
cd Python-3.11.10

sudo ./configure --prefix=/usr/local/python-3.11.10 --enable-optimizations
sudo make -j"$(nproc)"
sudo make altinstall

# 3) Verify the new interpreter
/usr/local/python-3.11.10/bin/python3.11 --version

# 4) create the rkllama venv with Python 3.11
/usr/local/python-3.11.10/bin/python3.11 -m venv rkllm_env
source rkllm_env/bin/activate
```

---

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
pip install kaldi_native_fbank onnxruntime sentencepiece soundfile pyyaml "numpy<2"
pip3 install pyaudio openwakeword onnxruntime torch torchvision torchaudio numpy soundfile librosa
```

### Rknn runtime setup

Transfer the RKNN runtime and RKLLM runtime library in CM3588 

```bash
sudo mv path/to/librknnrt.so /usr/lib/
sudo mv path/to/librkllmrt.so /usr/lib/
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
---

## Rkllm

## Setup for Rkllama

### Standard Installation (virtual environment like: conda, uv, venv)

```bash
git clone https://github.com/NotPunchnox/rkllama.git

cd rkllama

pip install dotenv huggingface_hub flask flask_cors

#Install RKLLama:
python3 -m pip install .
```

If it gives error:

The rkllama repository you are using is a third-party wrapper. This "Invalid URL" crash often happens when a developer specifies a GitHub link as a dependency in a way that is incompatible with newer pip standards.

**The Fix: Install Dependencies Manually**
Since the automated installation (pip install .) is crashing while parsing the metadata, you can bypass it by installing the core requirements and the package separately.

1. Update your build tools
First, ensure your environment's installer is capable of handling modern metadata:
```bash
pip install --upgrade pip setuptools wheel
```

2. Perform an "Editable" install
Instead of letting pip build a wheel (which is where it's failing), try an editable installation. This often bypasses some of the stricter URL validation checks in the build phase:

```bash
cd ~/rkllama
pip install -e . --no-deps
```

Note: Using `--no-deps` stops pip from looking at the broken dependency list. You will just need to make sure you have numpy installed manually.
A Crucial Note for CM3588
The rkllama library is a wrapper for the RKLLM C API. For it to work on your CM3588:
You must have the `librkllmrt.so` file in your system library path (usually `/usr/lib/` or defined in `LD_LIBRARY_PATH`).
If you haven't already, download the runtime from the official Rockchip RKNN-LLM GitHub.

Additionally, you are trying to install `torch==2.8.0` and `transformers==4.57.6`, which do not exist yet (the latest stable Torch is ~2.5.1 and Transformers is ~4.48).

#### Run Server
Virtualization with venv is started automatically, as well as the NPU frequency setting.

Start the server
```bash
rkllama_server --models <models_dir>
```
To enable debug mode:
```bash
rkllama_server --debug --models <models_dir>
```

### Docker installation

Pull the RKLLama Docker image:

```bash
docker pull ghcr.io/notpunchnox/rkllama:main
```

run server

```bash
docker run -it --privileged -p 8080:8080 -v <local_models_dir>:/opt/rkllama/models ghcr.io/notpunchnox/rkllama:main 
```

### Run Client
Command to start the client

```bash
rkllama_client
#or
rkllama_client help
```

Download the RKLLM model

```bash
rkllama_client pull
```

Download the llama3.2 model from huggingface: https://huggingface.co/jamescallander/Llama-3.2-3B-Instruct_w8a8_g128_rk3588.rkllm


## Download the models used for openwake word

```bash
wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/hey_jarvis_v0.1.onnx

wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx

wget https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx
```

- Transfer the require indic-whisper-medium-hindi model in CM3588.

## Local AI Assistant

Create a folder `/home/smadmin/rknn-model/assistant` in Board and transfer all folder/files in Local-AI-Assistant to that folder.

Make sure rkllama server is running in one SSH session or docker contianer

Run the web server

```bash
cd /home/smadmin/rknn-model/assistant
web_ui:app --host 0.0.0.0 --port 8765
```

Open the link in browser
`http://<CM3588-IP-Address>:8765` or `http://smritimegh.local:8765/`


### For running AI assistant using service

Transfer both service file to `/etc/systemd/system`

```bash
sudo systemctl enable rkllama.service && sudo systemctl start rkllama.service
sudo systemctl enable rkllm-webui.service && sudo systemctl start rkllm-webui.service
```

## Whisper small

Model Used: https://huggingface.co/danielferr85/whisper-with-past-models-rknn/tree/main/whisper-small

- Download all the model files, tokenizer and Transfer it in CM3588.
