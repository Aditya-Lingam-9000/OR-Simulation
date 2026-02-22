# Phase 6 Explanation: MedGemma GGUF Integration

## What Was Built

Phase 6 connects the MedGemma-4B-IT language model to the OR-Symphony pipeline,
turning transcript text into structured surgical state updates (JSON).

## The Four Modules

### 1. GGUFRunner (`src/llm/gguf_runner.py`)
- Wraps `llama-cpp-python` to load and run the GGUF model
- `load_model()` → creates a `Llama()` instance from the 1.95GB model file
- `generate(prompt)` → runs completion, extracts JSON from output
- `chat(messages)` → runs chat-style inference with system/user messages
- **Three JSON parsing strategies**: LLMs often wrap JSON in markdown, extra text,
  or code blocks. The parser tries direct JSON parse, brace extraction (`{...}`),
  and ```json block matching in sequence.
- `stub_mode=True` → returns fixed responses without loading the model (for tests)

### 2. PromptBuilder (`src/llm/prompts.py`)
- Loads the surgery-specific machine dictionary from `configs/machines/`
- Builds a **system prompt** that tells MedGemma its role, available machines
  (with IDs, categories, default states), surgical phases, and the exact JSON
  schema it must output
- Builds a **user prompt** with the current transcript context, phase, session
  time, and machine states
- Two output formats: chat-style messages or single completion prompt

### 3. LLMBatcher (`src/llm/batcher.py`)
- Async micro-batcher with configurable `max_batch` (default 4) and `max_wait_ms`
  (default 500ms)
- Collects requests into a queue, then processes them sequentially
- GGUF inference is single-threaded, so "batching" means queuing — but the
  batcher prevents request floods and provides backpressure
- Uses `run_in_executor()` to run blocking inference without blocking the
  event loop
- Supports custom processing functions for testing

### 4. LLMManager (`src/llm/manager.py`)
- Orchestrates the full pipeline: Request → Prompt → Batcher → Runner → Normalize
- On `start()`: loads the GGUF model, starts the batcher
- `submit(request)`: builds prompt, queues for inference, normalizes output,
  optionally validates against the JSON schema
- **Fallback**: if model load fails, inference errors, or batcher times out,
  returns a degraded response with `source: "rule"` so the system continues
  with rule-engine-only mode
- `set_surgery()`: switches the surgery type at runtime (rebuilds prompts)

## How It All Connects

```
Audio → ASR → Transcript
                  ↓
         RollingBuffer.get_context_for_llm()
                  ↓
         LLMRequest (transcript + phase + machines)
                  ↓
         LLMManager.submit()
                  ↓
         PromptBuilder formats system+user prompt
                  ↓
         LLMBatcher queues and dispatches
                  ↓
         GGUFRunner.generate() → raw text → JSON parsing
                  ↓
         StateSerializer.normalize_llm_output()
                  ↓
         LLMResponse (structured JSON)
                  ↓
         build_current_state(rule_result, llm_result)
                  ↓
         WebSocket → Frontend
```

## Why These Design Choices

1. **Pre-built wheel**: No C++ compiler is available on this machine, so
   `llama-cpp-python` was installed from a pre-built CPU wheel index.

2. **Stub mode everywhere**: Every module supports stub mode so the 513 tests
   run in <25 seconds without loading a 2GB model.

3. **Sequential not parallel**: Unlike GPU batch inference, GGUF CPU inference
   is single-threaded. The batcher serializes requests but provides async
   interface, backpressure, and stats.

4. **Fallback over crash**: The system never crashes on LLM failure — it
   degrades to rule-only mode with clear logging and metadata.

5. **Surgery-specific prompts**: Each surgery has different machines, phases,
   and terminology. The prompt builder tailors everything to the active surgery.
