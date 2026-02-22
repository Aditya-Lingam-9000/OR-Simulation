# Phase 0 — Repo + Environment Setup — Detailed Plan

> **Project**: OR-Symphony: Predictive Surgical State Engine  
> **Phase**: 0 — Repo + Environment  
> **Created**: 2026-02-22  
> **Status**: IN PROGRESS  
> **Gate**: Must PASS before any coding begins  

---

## Objective

Produce a clean, versioned repository with virtual environment, all dependencies installed, and a CI skeleton ready for future phases.

---

## Subphases & Checklist

### 0.1 — Repository Initialization
- [ ] Create project root directory structure
- [ ] Create `.gitignore` (Python + ONNX + models + temp artifacts)
- [ ] Create `README.md` (project overview, safety disclaimer, setup instructions)
- [ ] Create `LICENSE` (MIT or Apache 2.0)
- [ ] Initialize git repository
- [ ] Make initial commit: `chore: init repo skeleton`

### 0.2 — Virtual Environment & Dependencies
- [ ] Create Python virtual environment (`.venv`)
- [ ] Upgrade pip
- [ ] Create `requirements.txt` with all baseline dependencies:
  - fastapi, uvicorn, websockets, aiohttp
  - pytest, pytest-cov, pytest-asyncio
  - black, ruff (linting/formatting)
  - onnxruntime, onnx
  - torch, torchvision, torchaudio
  - transformers>=4.30
  - sounddevice, soundfile, webrtcvad, ffmpeg-python
  - python-dotenv
  - jsonschema
  - pydantic
- [ ] Install all requirements successfully
- [ ] Verify critical imports work

### 0.3 — CI Skeleton
- [ ] Create `.github/workflows/ci.yml`
  - [ ] Lint job (ruff)
  - [ ] Unit test job (pytest)
  - [ ] Triggered on push to main and PRs

### 0.4 — Directory Scaffolding (empty dirs for future phases)
- [ ] `src/` with subdirectories
- [ ] `tests/`
- [ ] `scripts/`
- [ ] `configs/`
- [ ] `schemas/`
- [ ] `logs/`
- [ ] `reports/`
- [ ] `data/`
- [ ] `samples/`
- [ ] `docs/`

### 0.5 — Sanity Checks
- [ ] Activate venv and run import check:
  ```
  python -c "import onnxruntime, torch, transformers, fastapi; print('ok')"
  ```
- [ ] Verify git status is clean after commit
- [ ] Verify `.gitignore` excludes `.venv/`, `__pycache__/`, `*.onnx`, `models/`, `tmp/`

### 0.6 — Phase Report & Sign-off
- [ ] Generate `reports/phase0_report.md`
- [ ] Manual confirmation line added
- [ ] **PHASE 0 PASS: ______ (initials/date/time)**

---

## Artifacts Produced

| Artifact | Path |
|----------|------|
| Git repo | `.git/` |
| Ignore rules | `.gitignore` |
| Project readme | `README.md` |
| License | `LICENSE` |
| Dependencies | `requirements.txt` |
| CI config | `.github/workflows/ci.yml` |
| Phase report | `reports/phase0_report.md` |

---

## Temp Artifacts to Delete After PASS

None — Phase 0 is safe to clean after pass.

---

## Next Phase Preview

**Phase 1** will create the full project skeleton code: module interfaces, API stubs, placeholder workers, device helpers, and config management.
