# Phase 0 Explanation — Repo + Environment Setup

> **Project**: OR-Symphony: Predictive Surgical State Engine  
> **Phase**: 0 — Repo + Environment  
> **Date**: 2026-02-22  

---

## What We Did in This Phase

Phase 0 established the foundational infrastructure for the entire OR-Symphony project:

1. **Git Repository** — Initialized a local git repository with proper `.gitignore` rules to exclude virtual environments, compiled Python files, ONNX model artifacts, temporary files, and logs.

2. **Project Documentation** — Created `README.md` with project overview, architecture diagram, safety disclaimer, quick-start guide, and hardware support table. Created `LICENSE` (MIT) with additional simulation/research-only notice.

3. **Dependency Management** — Created `requirements.txt` listing all baseline Python dependencies across categories:
   - API layer (FastAPI, uvicorn, websockets)
   - ML/AI (PyTorch, Transformers, ONNX Runtime)
   - Audio (sounddevice, soundfile, webrtcvad)
   - Testing (pytest, coverage)
   - Quality (black, ruff)
   - Utilities (pydantic, jsonschema, python-dotenv)

4. **Virtual Environment** — Created `.venv`, installed all dependencies, and verified all critical imports work on CPU (no GPU locally — as expected).

5. **CI Skeleton** — Created GitHub Actions workflow (`.github/workflows/ci.yml`) with two jobs: lint (ruff + black) and unit tests (pytest with coverage).

6. **Directory Scaffolding** — Created all project directories with `__init__.py` (for Python packages) or `.gitkeep` (for non-code directories):
   - `src/` with 7 sub-packages: ingest, asr, state, llm, api, workers, utils
   - `tests/`, `scripts/`, `configs/`, `schemas/`, `logs/`, `reports/`, `data/`, `samples/`, `docs/`, `onnx_models/`

7. **Testing Foundation** — Created `pytest.ini` configuration and initial smoke tests that verify the project structure integrity.

8. **Code Quality Config** — Created `pyproject.toml` with ruff and black settings (line length 100, Python 3.10 target).

---

## Flow in This Phase

```
Create Files → Git Init → Stage → Commit → Create Venv → Install Deps → Verify Imports → Run Tests → Commit Final
```

The flow is linear — no pipeline or workers involved yet. This is pure infrastructure setup.

---

## Flow from Phase 0 (Current) → Looking Ahead

```
Phase 0 (Environment) 
    ↓
Phase 1 (Skeleton Code) — will use the directory structure created here
    ↓
Phase 2 (Audio Capture) — will use the venv and audio packages installed here
    ↓
Phase 3+ — all subsequent phases build on this foundation
```

---

## What the Output of This Phase Feeds Into Phase 1

Phase 1 will use everything created here:

| Phase 0 Output | Phase 1 Usage |
|----------------|---------------|
| `src/ingest/` package | Will add `mic_server.py` |
| `src/asr/` package | Will add `runner.py` (abstract ASR class) |
| `src/state/` package | Will add `rules.py` (rule engine stub) |
| `src/llm/` package | Will add `manager.py` (LLM dispatcher stub) |
| `src/api/` package | Will add `app.py` (FastAPI + WebSocket skeleton) |
| `src/workers/` package | Will add worker stubs (asr, rule, llm, state) |
| `src/utils/` package | Will add `device.py`, `logging_config.py`, `config.py` |
| `tests/` | Will add test stubs for each module |
| `requirements.txt` | All imports available for skeleton code |
| `.github/workflows/ci.yml` | Will run lint + tests on Phase 1 code |
| `data/surgeries_machines.json` | Machine dictionaries ready for rule engine |

---

## Key Decisions Made

1. **Python 3.11.9** — available on the local machine (>= 3.10 requirement met)
2. **CPU-only PyTorch** — no GPU locally; Kaggle/Lightning for GPU testing later
3. **ONNX CPUExecutionProvider** — only provider available locally; CUDA provider will be used on Kaggle
4. **MIT License** — with additional simulation/research disclaimer
5. **Ruff + Black** — for code quality enforcement
6. **pytest** — for all testing (unit, integration, smoke)

---

## Files Count Summary

- **Total files created**: 29
- **Git commits**: 2
- **Tests passing**: 2/2
- **Dependencies installed**: 22+ packages (with transitive deps)
