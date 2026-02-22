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
    "faster-whisper>=1.0.0",    # ASR (CTranslate2 backend, GPU-accelerated)
    "soundfile>=0.12.1",        # audio I/O
    "pyngrok==7.1.6",           # ngrok tunnel
    "huggingface_hub",          # model downloads
)
print("✅ Core dependencies installed")

# llama-cpp-python — prebuilt CUDA 12.4 wheel (>= 0.3.8 for Gemma 3 support)
subprocess.check_call([
    sys.executable, "-m", "pip", "install", "-q",
    "llama-cpp-python>=0.3.8",
    "--extra-index-url",
    "https://abetlen.github.io/llama-cpp-python/whl/cu124",
])
print("✅ llama-cpp-python installed with CUDA 12.4 support")

# Verify GPU offload is available
try:
    import llama_cpp
    lib = getattr(llama_cpp, "llama_cpp", None)
    gpu_ok = lib.llama_supports_gpu_offload() if lib else False
    print(f"   GPU offload supported: {gpu_ok}")
    if not gpu_ok:
        print("   ⚠️  llama-cpp-python has no GPU support — LLM will be slow (CPU only)")
except Exception as e:
    print(f"   ⚠️  Could not verify GPU support: {e}")

# %% [markdown]
# ## Cell 2: Clone Repository & Setup

# %%
import os
from pathlib import Path

# Clone or pull the repo
REPO_URL = "https://github.com/Aditya-Lingam-9000/OR-Simulation.git"
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
GGUF_DIR = REPO_DIR / "onnx_models" / "medgemma"

GGUF_DIR.mkdir(parents=True, exist_ok=True)

from huggingface_hub import hf_hub_download

# --- ASR Model (faster-whisper base.en) ---
# faster-whisper auto-downloads from HuggingFace on first use.
# No manual download needed. Model cached at ~/.cache/huggingface/hub/
print("✅ ASR: faster-whisper base.en (auto-downloads on first transcription)")

# --- MedGemma LLM (GGUF Q3_K_M) ---
LLM_MODEL = GGUF_DIR / "medgemma-4b-it-Q3_K_M.gguf"
if not LLM_MODEL.exists():
    print("⬇️  Downloading MedGemma GGUF from HuggingFace (~1.8 GB)...")
    downloaded = hf_hub_download(
        repo_id="unsloth/medgemma-4b-it-GGUF",
        filename="medgemma-4b-it-Q3_K_M.gguf",
        local_dir=str(GGUF_DIR),
    )
    print(f"✅ MedGemma downloaded to: {downloaded}")
    # Verify the file landed in the expected path
    import shutil
    if not LLM_MODEL.exists() and Path(downloaded).exists():
        shutil.copy2(downloaded, LLM_MODEL)
        print(f"   Copied to expected path: {LLM_MODEL}")

if LLM_MODEL.exists():
    sz = LLM_MODEL.stat().st_size
    print(f"✅ MedGemma verified: {LLM_MODEL} ({sz / 1e9:.2f} GB, {sz} bytes)")
    if sz < 100_000_000:
        print(f"⚠️  File suspiciously small ({sz} bytes) — may be corrupt")
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

# Suppress llama.cpp C-level tokenizer warnings (MedGemma 262K vocab)
# Must be set BEFORE llama_cpp is imported anywhere.
os.environ["LLAMA_LOG_LEVEL"] = "ERROR"
os.environ["GGML_LOG_LEVEL"] = "error"

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
