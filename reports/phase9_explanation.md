# Phase 9 Explanation — Local Stress Testing & CI

## What Was Built

Phase 9 validates that the pipeline meets performance requirements under load and establishes continuous integration for the project.

### 1. Stress Test Script (`scripts/stress_local.py`)

A CLI tool that creates an Orchestrator in stub mode (no real LLM) and pushes synthetic surgical transcripts at a configurable rate. It measures:

- **Feed latency**: How long it takes to put a transcript into the queue (should be sub-millisecond)
- **Queue depths**: Snapshots of all four queues over time to detect backpressure
- **Memory**: Process RSS via psutil
- **Throughput**: Actual QPS achieved vs target QPS

The script includes 30 realistic surgical transcript phrases covering machine commands, clinical observations, and status updates.

### 2. Benchmark Runner (`scripts/benchmark_runner.py`)

Two benchmark modes:

**Direct rule engine benchmark** — calls `RuleEngine.process()` synchronously in a tight loop. This isolates the rule matching performance from all async overhead. Results: sub-millisecond median, > 10,000 calls/sec throughput. The rule engine is extremely fast because it's pattern matching against compiled regex + keyword dictionaries.

**Pipeline benchmark** — creates an Orchestrator, feeds transcripts, and measures the time from `feed_transcript()` to the state update callback firing. This captures the full async pipeline: queue put → rule worker processing → state writer merge → callback. Results vary by system but typically complete in < 50ms.

### 3. CI Pipeline

The GitHub Actions workflow now has three stages:

```
lint → test-unit → integration-smoke
```

- **Lint**: `ruff check` + `black --check` across src/, tests/, scripts/
- **Unit tests**: Full pytest suite excluding stress tests (which are slower), with JUnit XML artifact upload
- **Integration smoke**: Runs only acceptance-tagged tests from test_stress.py

Upgraded Python from 3.10 to 3.11 to match the development environment.

### 4. Acceptance Tests

The test file validates the three acceptance criteria from the master plan:

1. **Rule path latency < 500ms**: Multiple test methods (single call, 100-call median, 200-call p95, throughput). All pass with sub-millisecond latencies.

2. **Memory < 2 GB**: Checks process RSS after importing everything and after heavy rule processing. Skipped when psutil isn't installed (CI environments may not have it).

3. **MedGemma p50 < 3s**: The benchmark harness is ready but this requires a Kaggle GPU environment with the actual GGUF model loaded. The `benchmark_runner.py` script handles this measurement when the model is available.

Additionally, stress tests verify pipeline behavior under sustained load: 50 rapid transcripts without crash, overrides during load, surgery switching during load, and queue drain behavior.

## Why This Design

**In-process Orchestrator testing** over HTTP stress testing: By using the Orchestrator directly (not through the API server), we eliminate HTTP overhead from latency measurements. This gives us a clean measurement of the pipeline itself. The API layer adds negligible overhead (tested separately in test_api.py).

**Stub mode for pipeline tests**: The LLM is the slowest component by orders of magnitude (1-5 seconds vs < 1ms for rules). Using `llm_stub=True` lets us stress-test the pipeline architecture (queues, workers, merging) without waiting for model inference each time.

**Separate CI jobs**: Lint failures should fail fast before running expensive tests. Stress tests are separated from unit tests because they take longer and test different concerns (latency bounds vs correctness).
