# Phase 9 Plan — Local Stress Testing & CI

**Date**: 2026-02-22  
**Phase**: 9 of 11  

---

## Requirements (from Master Plan)

1. **Local synthetic load** — `scripts/stress_local.py` pushing transcripts at varying QPS, measuring latencies & queue sizes
2. **Kaggle test** — Re-run micro-batching with concurrent MedASR streams, capture p50/p95 latencies, store in `reports/kaggle_benchmarks/`
3. **Update CI** — Subset of integration tests (lightweight), generate artifacts
4. **Acceptance criteria validation**:
   - Rule path UI update latency median < 500ms (local)
   - MedGemma confirm p50 < 3s (Kaggle quantized)
   - Max memory usage < 90% of target device

## Deliverables

| # | File | Description |
|---|------|-------------|
| 1 | `scripts/stress_local.py` | CLI stress tester — configurable QPS, duration, surgery type |
| 2 | `scripts/benchmark_runner.py` | Benchmark harness — runs pipeline in-process, measures latencies |
| 3 | `reports/kaggle_benchmarks/README.md` | Placeholder + format spec for Kaggle benchmark results |
| 4 | `.github/workflows/ci.yml` | GitHub Actions CI — lint, unit tests, integration smoke |
| 5 | `tests/test_stress.py` | Acceptance criteria tests (latency bounds, memory, throughput) |
| 6 | `reports/phase9_report.md` | Phase completion report |

## Implementation Order

1. Create `scripts/stress_local.py` — async HTTP client pushing transcripts via POST /transcript
2. Create `scripts/benchmark_runner.py` — direct Orchestrator usage, measures per-transcript latency
3. Create `reports/kaggle_benchmarks/` placeholder
4. Create `.github/workflows/ci.yml`
5. Create `tests/test_stress.py` — parameterized acceptance tests
6. Run stress tests, capture benchmarks
7. Run full test suite
8. Update master plan, commit, report

## Notes

- Kaggle benchmarks require GPU environment — we create the harness + placeholder; actual runs happen externally
- Stress tests use `llm_stub=True` to measure pipeline throughput without model load time
- CI uses matrix strategy for Python 3.11 on ubuntu-latest
- Memory measurement via `psutil.Process().memory_info().rss`
