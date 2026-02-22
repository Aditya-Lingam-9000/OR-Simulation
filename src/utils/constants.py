"""
OR-Symphony: Project-wide Constants

All magic values, paths, and defaults are defined here.
No hardcoded values should appear elsewhere in the codebase.
"""

from __future__ import annotations

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
CONFIGS_DIR = PROJECT_ROOT / "configs"
SCHEMAS_DIR = PROJECT_ROOT / "schemas"
LOGS_DIR = PROJECT_ROOT / "logs"
REPORTS_DIR = PROJECT_ROOT / "reports"
SAMPLES_DIR = PROJECT_ROOT / "samples"
ONNX_MODELS_DIR = PROJECT_ROOT / "onnx_models"
TMP_DIR = PROJECT_ROOT / "tmp"

# Model paths
MEDASR_MODEL_PATH = ONNX_MODELS_DIR / "medasr" / "model.int8.onnx"
MEDASR_TOKENS_PATH = ONNX_MODELS_DIR / "medasr" / "tokens.txt"
MEDGEMMA_MODEL_PATH = ONNX_MODELS_DIR / "medgemma" / "medgemma-4b-it-Q3_K_M.gguf"

# Surgery machine data
SURGERIES_MACHINES_PATH = DATA_DIR / "surgeries_machines.json"

# ---------------------------------------------------------------------------
# Audio
# ---------------------------------------------------------------------------

AUDIO_SAMPLE_RATE = 16000  # Hz
AUDIO_CHANNELS = 1  # Mono
AUDIO_BIT_DEPTH = 16
VAD_FRAME_DURATION_MS = 30  # WebRTC VAD frame size
VAD_AGGRESSIVENESS = 2  # 0-3, higher = more aggressive filtering
CHUNK_MIN_DURATION_S = 0.5
CHUNK_MAX_DURATION_S = 2.0
CHUNK_OVERLAP_MS = 200

# ---------------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------------

ASR_CHUNK_LATENCY_TARGET_MS = 400
ASR_MAX_QUEUE_SIZE = 100

# ---------------------------------------------------------------------------
# Rule Engine
# ---------------------------------------------------------------------------

RULE_ENGINE_LATENCY_TARGET_MS = 500
RULE_DEBOUNCE_SECONDS = 3.0

# ---------------------------------------------------------------------------
# LLM (MedGemma)
# ---------------------------------------------------------------------------

LLM_MAX_BATCH_SIZE = 4
LLM_MAX_WAIT_MS = 500
LLM_CONTEXT_WINDOW = 4096  # GGUF context size
LLM_MAX_OUTPUT_TOKENS = 1024
LLM_TEMPERATURE = 0.1  # Low temperature for deterministic JSON output

# ---------------------------------------------------------------------------
# Rolling Buffer
# ---------------------------------------------------------------------------

ROLLING_BUFFER_DURATION_S = 180  # 3 minutes of transcript history

# ---------------------------------------------------------------------------
# API / WebSocket
# ---------------------------------------------------------------------------

API_HOST = "0.0.0.0"
API_PORT = 8000
WS_HEARTBEAT_INTERVAL_S = 5

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = "INFO"

# ---------------------------------------------------------------------------
# Supported Surgeries
# ---------------------------------------------------------------------------

SUPPORTED_SURGERIES = ["PCNL", "Partial Hepatectomy", "Lobectomy"]
DEFAULT_SURGERY = "PCNL"

# ---------------------------------------------------------------------------
# JSON Output Contract Keys
# ---------------------------------------------------------------------------

JSON_OUTPUT_KEYS = ["metadata", "machines", "details", "suggestions", "confidence", "source"]
VALID_SOURCES = ["rule", "medgemma", "rule+medgemma"]

# ---------------------------------------------------------------------------
# Safety
# ---------------------------------------------------------------------------

SAFETY_DISCLAIMER = (
    "OR-Symphony is a SIMULATION and RESEARCH system only. "
    "It does NOT control real medical devices. "
    "All outputs are suggestions requiring human confirmation."
)
