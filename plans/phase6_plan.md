# Phase 6 — MedGemma Integration Plan

**Date:** 2026-02-22
**Prerequisite:** Phase 5 ✅ (commit 93081c8)
**Model:** MedGemma-4B-IT Q3_K_M (GGUF, 1.95 GB) at `onnx_models/medgemma/`
**Runtime:** llama-cpp-python 0.3.2 (pre-built CPU wheel)

---

## Objectives
1. Full GGUF model loading and inference via llama-cpp-python
2. Structured prompt engineering for surgical phase reasoning
3. Micro-batching with configurable batch size and wait time  
4. Async LLM manager with queue, fallback, and schema validation
5. Local stub/mock path for unit tests (no model required)
6. Integration with Phase 5 rolling buffer & serializer

---

## Files to Create/Modify

### Create
| File | Purpose |
|------|---------|
| `src/llm/prompts.py` | Prompt templates for surgical reasoning |
| `src/llm/batcher.py` | Micro-batcher: collects requests, batches inference |
| `tests/test_llm.py` | ~60 tests for runner, prompts, batcher, manager |

### Modify  
| File | Changes |
|------|---------|
| `src/llm/gguf_runner.py` | Real model load, generate, JSON extraction, retry |
| `src/llm/manager.py` | Full async manager with batcher, fallback, validation |
| `src/llm/__init__.py` | Public exports |

---

## Architecture

```
Transcript → RollingBuffer.get_context_for_llm()
                          ↓
              PromptTemplate.format(surgery, machines, context)
                          ↓
              LLMManager.submit(request) → Queue
                          ↓
              Batcher → batch requests (max_batch=4, max_wait=500ms)
                          ↓
              GGUFRunner.generate(prompt) → raw text → JSON extraction
                          ↓
              StateSerializer.normalize_llm_output() → validated JSON
                          ↓
              Merge with rule engine output → WebSocket
```

---

## Key Design Decisions

1. **Stub mode by default** — Unit tests never load the 2GB model. `GGUFRunner` has a `stub_mode` flag that returns valid mock JSON responses.

2. **JSON extraction** — LLM output is scanned for `{...}` JSON blocks. Multiple strategies: direct parse, regex extraction, fallback to empty state.

3. **Fallback** — If LLM fails (timeout, bad JSON, model not loaded), system falls back to rule-engine-only mode with `source: "rule"` and `meta.reasoning: "degraded"`.

4. **Prompt structure** — System prompt defines the role, schema constraints, and machine dictionary. User prompt provides recent transcript context.

5. **Temperature = 0.1** — Near-deterministic for JSON output consistency.

6. **Micro-batcher** — Collects up to `MAX_BATCH=4` requests over `MAX_WAIT=500ms`, processes them sequentially through the model (GGUF is single-threaded inference).

---

## Deliverables Checklist
- [ ] GGUFRunner with real model loading
- [ ] Prompt templates (system + user)
- [ ] Micro-batcher
- [ ] LLM Manager with queue and fallback
- [ ] ~60 tests (stub mode, prompt formatting, batcher, manager, integration)
- [ ] All tests pass (existing + new)
- [ ] Standalone demo
- [ ] Git commit on main
- [ ] Phase 6 report + explanation
