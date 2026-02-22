# OR-Symphony — Kaggle Backend Deployment Notebook
#
# This script is intended to be run in a Kaggle Notebook environment
# (with GPU enabled for MedGemma inference). It:
#
#   1. Installs required Python packages
#   2. Clones the OR-Symphony repository
#   3. Downloads model weights (ASR ONNX + MedGemma GGUF)
#   4. Starts the FastAPI backend with real LLM inference
#   5. Opens an ngrok tunnel so your local frontend can connect
#
# INSTRUCTIONS:
#   1. Create a new Kaggle Notebook (Python, GPU T4 x2 or P100)
#   2. Add your ngrok authtoken as a Kaggle Secret named "NGROK_TOKEN"
#   3. Upload this file's content into a code cell
#   4. Run all cells
#   5. Copy the ngrok URL and use it in your local frontend
#
# ============================================================================

# %% [markdown]
# ## Cell 1: Install Dependencies

# %%
import subprocess, sys

def pip_install(*packages):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + list(packages))

# Core backend dependencies
pip_install(
    "fastapi==0.115.0",
    "uvicorn[standard]==0.30.0",
    "pydantic>=2.0",
    "numpy>=1.24",
    "onnxruntime-gpu==1.18.0",  # GPU ONNX for ASR
    "llama-cpp-python==0.3.2",  # MedGemma GGUF inference
    "pyngrok==7.1.6",           # ngrok tunnel
    "torchaudio>=2.0",          # feature extraction
)

print("✅ Core dependencies installed")

# %% [markdown]
# ## Cell 2: Clone Repository & Setup

# %%
import os
from pathlib import Path

# Clone or pull the repo
REPO_URL = "https://github.com/YOUR_USERNAME/OR-Simulation.git"  # ← CHANGE THIS
REPO_DIR = Path("/kaggle/working/OR-Simulation")

if not REPO_DIR.exists():
    subprocess.check_call(["git", "clone", REPO_URL, str(REPO_DIR)])
    print(f"✅ Cloned to {REPO_DIR}")
else:
    subprocess.check_call(["git", "-C", str(REPO_DIR), "pull"])
    print(f"✅ Updated {REPO_DIR}")

os.chdir(REPO_DIR)
sys.path.insert(0, str(REPO_DIR))
print(f"Working directory: {os.getcwd()}")

# %% [markdown]
# ## Cell 3: Download Model Weights
#
# The ASR model (ONNX int8) and MedGemma (GGUF Q3_K_M) need to be available.
# Option A: Upload them as a Kaggle Dataset.
# Option B: Download them here.

# %%
ONNX_DIR = REPO_DIR / "onnx_models" / "medasr"
GGUF_DIR = REPO_DIR / "onnx_models" / "medgemma"

ONNX_DIR.mkdir(parents=True, exist_ok=True)
GGUF_DIR.mkdir(parents=True, exist_ok=True)

# --- Install huggingface_hub for model downloads ---
pip_install("huggingface_hub")
from huggingface_hub import hf_hub_download

# --- ASR Model (MedASR ONNX int8) ---
ASR_MODEL = ONNX_DIR / "model.int8.onnx"
ASR_TOKENS = ONNX_DIR / "tokens.txt"
if not ASR_MODEL.exists():
    print("⬇️  Downloading MedASR ONNX model from HuggingFace...")
    hf_hub_download(
        repo_id="csukuangfj/sherpa-onnx-medasr-ctc-en-int8-2025-12-25",
        filename="model.int8.onnx",
        local_dir=str(ONNX_DIR),
    )
    hf_hub_download(
        repo_id="csukuangfj/sherpa-onnx-medasr-ctc-en-int8-2025-12-25",
        filename="tokens.txt",
        local_dir=str(ONNX_DIR),
    )
    print(f"✅ MedASR model downloaded to {ONNX_DIR}")
else:
    print(f"✅ ASR model found: {ASR_MODEL} ({ASR_MODEL.stat().st_size / 1e6:.1f} MB)")

# --- MedGemma LLM (GGUF Q3_K_M) ---
LLM_MODEL = GGUF_DIR / "medgemma-4b-it-Q3_K_M.gguf"
if not LLM_MODEL.exists():
    print("⬇️  Downloading MedGemma GGUF from HuggingFace (~1.8 GB)...")
    hf_hub_download(
        repo_id="unsloth/medgemma-4b-it-GGUF",
        filename="medgemma-4b-it-Q3_K_M.gguf",
        local_dir=str(GGUF_DIR),
    )
    print(f"✅ MedGemma downloaded to {GGUF_DIR}")
else:
    print(f"✅ MedGemma model found: {LLM_MODEL} ({LLM_MODEL.stat().st_size / 1e9:.2f} GB)")

# %% [markdown]
# ## Cell 4: Configure ngrok Tunnel

# %%
from pyngrok import ngrok, conf
from kaggle_secrets import UserSecretsClient

# Get ngrok token from Kaggle Secrets
try:
    secrets = UserSecretsClient()
    NGROK_TOKEN = secrets.get_secret("NGROK_TOKEN")
except Exception:
    NGROK_TOKEN = os.environ.get("NGROK_TOKEN", "")

if not NGROK_TOKEN:
    print("❌ NGROK_TOKEN not found!")
    print("   Go to https://dashboard.ngrok.com/get-started/your-authtoken")
    print("   Then add it as a Kaggle Secret named 'NGROK_TOKEN'")
    raise ValueError("NGROK_TOKEN required for external access")

ngrok.set_auth_token(NGROK_TOKEN)
print("✅ ngrok configured")

# %% [markdown]
# ## Cell 5: Start the Backend Server

# %%
import threading
import time
import uvicorn

# Set environment variables
os.environ["LLM_STUB"] = "0" if LLM_MODEL.exists() else "1"
os.environ["LLM_REAL"] = "1" if LLM_MODEL.exists() else "0"

# Import the FastAPI app
from src.api.app import app

PORT = 8000

# Start uvicorn in a background thread
def run_server():
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")

server_thread = threading.Thread(target=run_server, daemon=True)
server_thread.start()
print(f"⏳ Waiting for server to start on port {PORT}...")
time.sleep(5)

# Verify server is running
import urllib.request
try:
    resp = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health")
    health = resp.read().decode()
    print(f"✅ Server running! Health: {health}")
except Exception as e:
    print(f"❌ Server health check failed: {e}")

# %% [markdown]
# ## Cell 6: Open ngrok Tunnel — Copy this URL!

# %%
# Open an ngrok tunnel to the backend
tunnel = ngrok.connect(PORT, "http")
public_url = tunnel.public_url

print("=" * 70)
print()
print("  🎉 OR-Symphony Backend is LIVE!")
print()
print(f"  Public URL:  {public_url}")
print()
print("  Use this URL in your local frontend:")
print()
print(f"    http://localhost:3000/?backend={public_url}")
print()
print("  Or set environment variable before building:")
print(f"    VITE_BACKEND_URL={public_url}")
print()
print("=" * 70)
print()
print("API Endpoints:")
print(f"  Health:     {public_url}/health")
print(f"  State:      {public_url}/state")
print(f"  Surgeries:  {public_url}/surgeries")
print(f"  Machines:   {public_url}/machines")
print(f"  WebSocket:  {public_url.replace('http', 'ws')}/ws/state")
print(f"  Audio WS:   {public_url.replace('http', 'ws')}/ws/audio")

# %% [markdown]
# ## Cell 7: Keep Alive
#
# This cell keeps the Kaggle notebook running so the server stays alive.
# The notebook will stay active for up to 12 hours (Kaggle limit).

# %%
print("🔄 Server is running. Keep this notebook tab open.")
print("   Press Stop ⏹ when you want to shut down.")
print()
print(f"   Frontend URL: http://localhost:3000/?backend={public_url}")
print()

# Keep-alive loop
try:
    while True:
        time.sleep(30)
        # Periodic health check
        try:
            resp = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5)
            status = "✅ alive"
        except Exception:
            status = "⚠️ check"
        print(f"[{time.strftime('%H:%M:%S')}] Server {status} | Tunnel: {public_url}")
except KeyboardInterrupt:
    print("\n🛑 Shutting down...")
    ngrok.disconnect(public_url)
    print("ngrok tunnel closed")
