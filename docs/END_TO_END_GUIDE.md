# OR-Symphony — Complete End-to-End Setup Guide

## Live Mic → Kaggle Backend → Local Frontend

This guide walks you through running OR-Symphony end-to-end:
- **Backend** on Kaggle (GPU-accelerated ASR + MedGemma LLM inference)
- **Frontend** on your local machine browser
- **Live microphone** input through the browser — speak surgical commands and watch machines toggle in real-time

---

## Architecture Overview

```
┌─────────────────────── YOUR LOCAL MACHINE ───────────────────────┐
│                                                                   │
│   Browser (Chrome/Edge)                                           │
│   ┌─────────────────────────────────────────────────────────┐     │
│   │  React Frontend (localhost:3000)                         │     │
│   │  ┌──────────┐  ┌──────────┐  ┌────────────────────────┐ │     │
│   │  │ MicRec.  │  │ OPRoom   │  │ SurgerySelector        │ │     │
│   │  │ (AudioW) │  │ (SVG)    │  │ OverrideDialog         │ │     │
│   │  └────┬─────┘  └────▲─────┘  │ StatusBar / MachineList│ │     │
│   │       │ float32      │ JSON   └────────────────────────┘ │     │
│   │       │ PCM 16kHz    │ state                              │     │
│   │       ▼              │                                    │     │
│   │   ws://ngrok/ws/audio  ws://ngrok/ws/state                │     │
│   └─────────────────────────────────────────────────────────┘     │
│                                                                   │
└───────────────────────────────┬───────────────────────────────────┘
                                │ ngrok tunnel (HTTPS/WSS)
                                ▼
┌─────────────────────── KAGGLE NOTEBOOK (GPU) ────────────────────┐
│                                                                   │
│   FastAPI Server (port 8000)                                      │
│   ┌──────────────────────────────────────────────────────────┐    │
│   │  /ws/audio ──→ AudioQueue ──→ ASR Worker (ONNX int8)     │    │
│   │                                     │                     │    │
│   │                              TranscriptQueue              │    │
│   │                                     │                     │    │
│   │                              ┌──────┴──────┐              │    │
│   │                              ▼             ▼              │    │
│   │                        Rule Worker    LLM Dispatcher      │    │
│   │                        (regex/NLP)   (MedGemma 4B)        │    │
│   │                              │             │              │    │
│   │                              ▼             ▼              │    │
│   │                          State Writer (merge + safety)    │    │
│   │                                     │                     │    │
│   │  /ws/state ◄─── Broadcast ◄─────────┘                    │    │
│   └──────────────────────────────────────────────────────────┘    │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### On Your Local Machine
- **Node.js 18+** — [Download](https://nodejs.org/)
- **Git** — to clone the repository
- **Chrome or Edge** browser (Firefox works but Chrome has better AudioWorklet support)
- **A microphone** — built-in laptop mic, headset, USB mic, etc.

### On Kaggle
- A **Kaggle account** — [Sign up](https://www.kaggle.com/)
- **GPU Notebook** enabled (T4 x2 recommended)
- An **ngrok account** (free tier) — [Sign up](https://ngrok.com/)
- Your **ngrok auth token** — get it from [ngrok dashboard](https://dashboard.ngrok.com/get-started/your-authtoken)

---

## Step-by-Step Instructions

### PART A: Set Up the Backend on Kaggle

#### A1. Push your code to GitHub

If you haven't already, push the OR-Simulation repo to GitHub:

```powershell
cd D:\OR-Simulation
git remote add origin https://github.com/YOUR_USERNAME/OR-Simulation.git
git push -u origin main
```

#### A2. Upload Model Weights to Kaggle

You need two model files. Upload them as a **Kaggle Dataset**:

1. Go to [Kaggle Datasets → New Dataset](https://www.kaggle.com/datasets?new=true)
2. Name it `or-symphony-models`
3. Upload these files with this folder structure:
   ```
   or-symphony-models/
   ├── medasr/
   │   └── model.int8.onnx          ← ASR ONNX model (~50-200 MB)
   └── medgemma/
       └── medgemma-4b-it-Q3_K_M.gguf  ← MedGemma LLM (~1.8 GB)
   ```
4. Publish the dataset (can be private)

> **Don't have the models?** The system can still work:
> - Without ASR model → use text input via the `/transcript` endpoint (type commands instead of speaking)
> - Without LLM model → runs in stub mode (rule engine only, no MedGemma confirmation)

#### A3. Create a Kaggle Notebook

1. Go to [Kaggle → New Notebook](https://www.kaggle.com/code)
2. Settings:
   - **Language**: Python
   - **Accelerator**: **GPU T4 x2** (or P100)
   - **Internet**: **ON** (required for ngrok + git clone)
   - **Persistence**: Files only
3. Add your dataset: Click **Add data** → search `or-symphony-models` → Add
4. Add **ngrok Secret**:
   - Click ⋯ menu → **Add-ons** → **Secrets**
   - Add a secret named `NGROK_TOKEN` with your ngrok auth token

#### A4. Run the Deployment Script

Copy the contents of `kaggle/deploy_backend.py` into notebook cells. Or run it as a single cell:

```python
# Cell 1: Install dependencies
import subprocess, sys, os

subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
    "fastapi==0.115.0", "uvicorn[standard]==0.30.0", "pydantic>=2.0",
    "numpy>=1.24", "onnxruntime-gpu==1.18.0", "llama-cpp-python==0.3.2",
    "pyngrok==7.1.6", "torchaudio>=2.0"])
```

```python
# Cell 2: Clone repo
import subprocess
from pathlib import Path

REPO_DIR = Path("/kaggle/working/OR-Simulation")
if not REPO_DIR.exists():
    subprocess.check_call(["git", "clone",
        "https://github.com/YOUR_USERNAME/OR-Simulation.git",   # ← YOUR URL
        str(REPO_DIR)])

os.chdir(REPO_DIR)
sys.path.insert(0, str(REPO_DIR))
```

```python
# Cell 3: Link model weights from Kaggle Dataset
ONNX_DIR = REPO_DIR / "onnx_models" / "medasr"
GGUF_DIR = REPO_DIR / "onnx_models" / "medgemma"
ONNX_DIR.mkdir(parents=True, exist_ok=True)
GGUF_DIR.mkdir(parents=True, exist_ok=True)

# Copy from Kaggle Dataset (adjust path if your dataset name differs)
import shutil
DATASET = Path("/kaggle/input/or-symphony-models")
asr_src = DATASET / "medasr" / "model.int8.onnx"
llm_src = DATASET / "medgemma" / "medgemma-4b-it-Q3_K_M.gguf"

if asr_src.exists():
    shutil.copy2(asr_src, ONNX_DIR / "model.int8.onnx")
    print(f"✅ ASR model linked")
else:
    print("⚠️ ASR model not found — text-only mode")

if llm_src.exists():
    shutil.copy2(llm_src, GGUF_DIR / "medgemma-4b-it-Q3_K_M.gguf")
    print(f"✅ MedGemma model linked")
else:
    print("⚠️ MedGemma not found — LLM stub mode")
```

```python
# Cell 4: Configure environment & start server
import threading, time, uvicorn

LLM_MODEL = GGUF_DIR / "medgemma-4b-it-Q3_K_M.gguf"
os.environ["LLM_STUB"] = "0" if LLM_MODEL.exists() else "1"
os.environ["LLM_REAL"] = "1" if LLM_MODEL.exists() else "0"

from src.api.app import app

PORT = 8000

def run_server():
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")

threading.Thread(target=run_server, daemon=True).start()
time.sleep(8)  # Wait for model loading

# Verify
import urllib.request
resp = urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health")
print(f"✅ Server health: {resp.read().decode()}")
```

```python
# Cell 5: Open ngrok tunnel
from pyngrok import ngrok
from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
ngrok.set_auth_token(secrets.get_secret("NGROK_TOKEN"))

tunnel = ngrok.connect(PORT, "http")
PUBLIC_URL = tunnel.public_url

print("=" * 60)
print(f"🎉  Backend URL: {PUBLIC_URL}")
print(f"    Frontend:    http://localhost:3000/?backend={PUBLIC_URL}")
print("=" * 60)
```

```python
# Cell 6: Keep alive
try:
    while True:
        time.sleep(30)
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5)
            print(f"[{time.strftime('%H:%M:%S')}] ✅ Server alive")
        except: pass
except KeyboardInterrupt:
    ngrok.disconnect(PUBLIC_URL)
    print("Shut down.")
```

5. Run all cells in order. **Copy the PUBLIC_URL** from Cell 5 output.

> The URL looks like: `https://abc1-34-125-67-89.ngrok-free.app`

---

### PART B: Set Up the Frontend Locally

#### B1. Clone the Repository (if not already done)

```powershell
git clone https://github.com/YOUR_USERNAME/OR-Simulation.git
cd OR-Simulation
```

#### B2. Install Frontend Dependencies

```powershell
cd frontend
npm install
```

#### B3. Start the Frontend Dev Server

**Option 1: URL Parameter (recommended — no config files needed)**

```powershell
npm run dev
```

Then open your browser to:
```
http://localhost:3000/?backend=https://YOUR-NGROK-URL.ngrok-free.app
```

Replace `YOUR-NGROK-URL` with the actual URL from Kaggle Cell 5.

**Option 2: Environment Variable**

Create `frontend/.env.local`:
```
VITE_BACKEND_URL=https://YOUR-NGROK-URL.ngrok-free.app
```

Then:
```powershell
npm run dev
```

Open `http://localhost:3000` — it will auto-connect to Kaggle.

---

### PART C: Use the Application

#### C1. Verify Connection

When the frontend loads, check:
- **Status bar** (bottom) shows a green **● Live** dot — WebSocket connected
- **Surgery** dropdown shows PCNL (default)
- **Machines** panel shows M01–M09 with their states

If you see a red **Offline** indicator, check:
- Is the Kaggle notebook still running?
- Is the ngrok URL correct?
- Try refreshing the page

#### C2. Start the Microphone

1. Click the green **Start Mic** button in the header
2. Your browser will ask for microphone permission → **Allow**
3. You should see:
   - A green audio level bar animating
   - **● Live** status next to the mic button
4. The AudioWorklet captures your voice, resamples to 16kHz, and streams to Kaggle

#### C3. Speak Surgical Commands

Speak clearly and naturally. Example commands that trigger machine state changes:

**Phase 1 — Patient Preparation:**
- *"Attach the patient monitor"*
- *"Lights on, let's prep the patient"*
- *"Begin baseline monitoring"*

**Phase 2 — Anesthesia:**
- *"Start anesthesia"* → turns ON Anesthesia Machine (M02)
- *"Intubation complete, start the ventilator"* → turns ON Ventilator (M03)

**Phase 3 — Access:**
- *"Bring in the C-arm, we need fluoroscopy"* → turns ON C-Arm (M04)
- *"Start irrigation"* → turns ON Irrigation Pump (M05)
- *"Camera on, show the scope view"* → turns ON Endoscopic Camera (M08)

**Phase 4 — Stone Management (PCNL):**
- *"Start lithotripsy"* → turns ON Lithotripter (M07)
- *"Suction on, clear the field"* → turns ON Suction (M06)

**Phase 5 — Closure:**
- *"Stop irrigation"* → turns OFF Irrigation Pump
- *"Turn off the lithotripter"* → turns OFF Lithotripter
- *"Turn off fluoroscopy"* → turns OFF C-Arm

**Phase 6 — Emergence:**
- *"Extubation, stop ventilator"* → turns OFF Ventilator
- *"Stop anesthesia"* → turns OFF Anesthesia Machine

#### C4. Watch Real-Time Updates

As you speak:
1. **Audio** streams from browser → Kaggle via WebSocket
2. **ASR** (ONNX model) converts speech to text on Kaggle GPU
3. **Rule Engine** matches keywords → generates machine toggle commands
4. **MedGemma LLM** (if available) confirms or refines the decision
5. **State Writer** merges rule + LLM outputs → broadcasts via WebSocket
6. **Frontend** receives JSON state → updates SVG machine icons in real-time

You'll see:
- Machine icons change colour: **green** (ON) → **grey** (OFF)
- Status bar updates: phase, confidence %, source (rule/medgemma/rule+medgemma)
- Suggestions appear when the pipeline predicts next steps

#### C5. Manual Overrides

Click any machine (in the SVG room or the table) to open the Override dialog:
1. Select **ON**, **OFF**, or **STANDBY**
2. Type a reason (optional)
3. Click **Apply Override**

The override is logged in the audit trail and the state updates immediately.

#### C6. Switch Surgeries

Use the dropdown in the header to switch between:
- **PCNL** — Percutaneous Nephrolithotomy (urology, 9 machines)
- **Partial Hepatectomy** — Liver surgery (9 machines, different set)
- **Lobectomy** — Lung surgery (9 machines, different set)

The room layout, machine icons, and positions change automatically.

---

## Troubleshooting

### Frontend shows "Offline"
- **Cause**: WebSocket can't reach Kaggle backend
- **Fix**: Check the ngrok URL is correct (no trailing slash). Refresh the page.
- **Check**: Open `https://YOUR-NGROK-URL.ngrok-free.app/health` in a browser — you should see JSON.

### ngrok shows "Tunnel not found"
- **Cause**: Kaggle notebook timed out (12h limit) or kernel restarted
- **Fix**: Re-run all cells in the Kaggle notebook, get the new ngrok URL

### Microphone doesn't work
- **Cause**: Browser needs HTTPS for mic access, or permission denied
- **Fix**: `localhost` is exempt from HTTPS requirement. Click the lock icon → allow mic.
- **Chrome**: Settings → Privacy → Site Settings → Microphone → Allow localhost

### No machine toggles when speaking
- **Cause 1**: ASR model not loaded → check Kaggle notebook logs for "ASR model loaded"
- **Cause 2**: Speaking too softly → check the audio level bar is moving
- **Cause 3**: Commands not matching triggers → try exact phrases like "start anesthesia"
- **Fix**: Use the text endpoint as fallback:
  ```
  curl -X POST https://YOUR-NGROK-URL.ngrok-free.app/transcript \
    -H "Content-Type: application/json" \
    -d '{"text": "start anesthesia", "speaker": "surgeon"}'
  ```

### LLM running in stub mode
- **Cause**: MedGemma GGUF model not found on Kaggle
- **Impact**: Pipeline works with rule engine only. Source shows "rule" instead of "rule+medgemma".
- **Fix**: Upload the GGUF file to your Kaggle dataset and re-run Cell 3.

### High latency (>2s for machine toggle)
- **Cause**: ngrok adds ~100-200ms round-trip, ASR inference ~400ms, LLM ~500ms
- **Expected**: Total latency ~1-2s from speech to visual update
- **If worse**: Check Kaggle GPU utilization, reduce audio chunk size

---

## Running Without Kaggle (Local-Only Mode)

If you want to run everything locally (no GPU/LLM needed):

### Option 1: Full backend (rule engine only, LLM stub)

```powershell
cd D:\OR-Simulation

# Terminal 1: Start backend
.venv\Scripts\activate
set LLM_STUB=1
python -m uvicorn src.api.app:app --host 0.0.0.0 --port 8000

# Terminal 2: Start frontend
cd frontend
npm run dev
```

Open `http://localhost:3000` — no `?backend=` needed (Vite proxies to localhost:8000).

### Option 2: Simulator (no real pipeline, fake state updates)

```powershell
# Terminal 1: Start simulator
cd D:\OR-Simulation
.venv\Scripts\activate
python scripts/frontend_simulator.py --port 8000 --interval 2

# Terminal 2: Start frontend
cd frontend
npm run dev
```

Open `http://localhost:3000` — see simulated state cycling through phases.

---

## API Reference (Quick)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/state` | GET | Current surgery state JSON |
| `/surgeries` | GET | List available surgeries |
| `/machines` | GET | Machine definitions for current surgery |
| `/stats` | GET | Pipeline statistics |
| `/select_surgery` | POST | Switch surgery `{"surgery": "PCNL"}` |
| `/transcript` | POST | Feed text `{"text": "start anesthesia", "speaker": "surgeon"}` |
| `/override` | POST | Override `{"machine_id": "M01", "action": "ON", "reason": "..."}` |
| `/ws/state` | WS | Real-time state updates (JSON) |
| `/ws/audio` | WS | Browser microphone audio stream (binary float32 PCM) |

---

## File Overview

```
OR-Simulation/
├── frontend/                          # React + Vite SPA
│   ├── src/
│   │   ├── App.jsx                    # Main shell (reads ?backend= URL param)
│   │   ├── providers/StateProvider.jsx # WebSocket state context
│   │   └── components/
│   │       ├── MicRecorder.jsx        # Browser mic → WebSocket audio
│   │       ├── OPRoom.jsx             # SVG operating room
│   │       ├── MachineIcon.jsx        # Machine SVG icons
│   │       ├── SurgerySelector.jsx    # Surgery dropdown
│   │       ├── OverrideDialog.jsx     # Manual override modal
│   │       ├── AgentOverlay.jsx       # Source indicator
│   │       ├── StatusBar.jsx          # Connection + pipeline status
│   │       └── MachineList.jsx        # Accessible text table
│   ├── public/pcm-processor.js        # AudioWorklet (resample to 16kHz)
│   └── .env.example                   # Backend URL config template
├── src/
│   ├── api/app.py                     # FastAPI server (REST + WS)
│   ├── workers/orchestrator.py        # Pipeline coordinator
│   ├── workers/asr_worker.py          # ONNX ASR inference
│   ├── workers/rule_worker.py         # Rule engine (keyword matching)
│   ├── workers/llm_dispatcher.py      # MedGemma LLM inference
│   └── workers/state_writer.py        # State merge + safety + broadcast
├── configs/
│   ├── machines/{pcnl,partial_hepatectomy,lobectomy}.json
│   └── layouts/{pcnl,partial_hepatectomy,lobectomy}_layout.json
├── kaggle/deploy_backend.py           # Kaggle deployment script
└── scripts/frontend_simulator.py      # Standalone mock backend
```
