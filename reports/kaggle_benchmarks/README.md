# Kaggle Benchmarks

This directory stores benchmark results from Kaggle GPU environments.

## Format

Each benchmark file is a JSON file with the naming convention:

```
kaggle_benchmark_YYYYMMDD_HHMMSS.json
```

## Expected Structure

```json
{
  "timestamp": "2026-02-22T10:00:00Z",
  "environment": {
    "gpu": "T4",
    "ram_gb": 16,
    "python": "3.11",
    "kaggle_notebook": "<URL>"
  },
  "asr_benchmarks": {
    "concurrent_streams": [1, 2, 4],
    "latency_ms": {
      "p50": 250,
      "p95": 380,
      "p99": 450
    }
  },
  "medgemma_benchmarks": {
    "model": "MedGemma-4B-IT-Q3_K_M",
    "latency_ms": {
      "p50": 2100,
      "p95": 2800
    },
    "tokens_per_sec": 45
  }
}
```

## Acceptance Criteria

| Metric | Target | Source |
|--------|--------|--------|
| Rule path UI update | median < 500ms | `scripts/stress_local.py` |
| MedGemma inference p50 | < 3s | Kaggle quantized |
| Memory usage | < 90% of device | `scripts/benchmark_runner.py` |

## Running

```bash
# Local benchmarks
python -m scripts.benchmark_runner --output reports/kaggle_benchmarks

# Stress test
python -m scripts.stress_local --qps 10 --duration 60
```
