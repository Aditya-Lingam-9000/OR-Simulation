# OR-Symphony: Predictive Surgical State Engine

> **⚠️ SAFETY DISCLAIMER**: This is a **simulation and research system ONLY**. It does NOT control real medical devices. All outputs are suggestions requiring human confirmation. No clinical decisions should be made based on this system's output. Human-in-the-loop is mandatory.

---

## Overview

OR-Symphony is a research-grade AI system that:

- Listens to live microphone audio during a **simulated** surgery
- Transcribes speech in real time using MedASR (ONNX-optimized)
- Reasons over transcripts using MedGemma for surgical context understanding
- Tracks surgical phase and machine orchestration
- Outputs strict structured JSON state updates
- Drives a frontend operating-room visualization

## Supported Surgeries

| # | Surgery | Environment |
|---|---------|-------------|
| 1 | PCNL (Percutaneous Nephrolithotomy) | Urology OR |
| 2 | Partial Hepatectomy (Liver Resection) | Hepatobiliary OR |
| 3 | Lobectomy (Lung Surgery) | Thoracic OR |

## Architecture

```
Audio Input → VAD → Chunking → MedASR → Transcript
Transcript → Rule Engine (fast deterministic layer)
Transcript + Context + Surgery Type → MedGemma
MedGemma → Structured JSON
JSON → State Writer → WebSocket → Frontend Animation
```

## Pipeline Components

| Component | Latency Target | Description |
|-----------|---------------|-------------|
| ASR | < 400ms | Speech-to-text via ONNX |
| Rule Engine | < 500ms | Deterministic keyword/regex mapping |
| MedGemma | Best effort | Contextual reasoning + validation |
| State Writer | Atomic | Merges outputs, pushes via WebSocket |

## Project Structure

```
src/
  ingest/       # Mic input, VAD bridge
  asr/          # MedASR wrappers + streaming adapters
  state/        # Rolling buffer, rule engine, schema
  llm/          # MedGemma runner, queue/dispatcher
  api/          # FastAPI + WebSocket endpoints
  workers/      # Orchestrator services
  utils/        # Device helpers, logging, config
tests/          # Unit + integration tests
scripts/        # Utility scripts
configs/        # Machine dictionaries, layouts
schemas/        # JSON schemas
logs/           # Runtime logs (gitignored)
reports/        # Phase reports
data/           # Surgery machine data
docs/           # Documentation
```

## Quick Start

### Prerequisites

- Python 3.10+
- Git

### Setup

```bash
cd d:\OR-Simulation
python -m venv .venv

# Windows
.venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Run API Server (Dev)

```bash
uvicorn src.api.app:app --reload --port 8000
curl http://127.0.0.1:8000/health
```

### Run Tests

```bash
pytest -q
```

## Hardware Support

| Environment | Device | Provider |
|-------------|--------|----------|
| Local Dev | CPU | ONNX CPUExecutionProvider |
| Kaggle | GPU | ONNX CUDAExecutionProvider |

Device is auto-detected at runtime:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Safety & Ethics

- This system is for **simulation and research purposes only**
- **No real medical devices** are controlled
- All suggestions require **human confirmation**
- Immutable audit logs are maintained for all state changes
- See `SAFETY.md` for full safety documentation

## License

MIT License — See `LICENSE` file.

## Development Status

See `plans/MASTER_PROJECT_PLAN.md` for the full development roadmap and phase tracking.
