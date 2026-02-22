"""
OR-Symphony: Configuration Management

Uses pydantic Settings for validated, environment-aware configuration.
Loads from .env files and environment variables.

Usage:
    from src.utils.config import get_settings
    settings = get_settings()
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import List, Optional

from pydantic import Field
from pydantic_settings import BaseSettings

from src.utils.constants import (
    API_HOST,
    API_PORT,
    ASR_CHUNK_LATENCY_TARGET_MS,
    AUDIO_SAMPLE_RATE,
    CHUNK_MAX_DURATION_S,
    CHUNK_MIN_DURATION_S,
    CHUNK_OVERLAP_MS,
    DEFAULT_SURGERY,
    LLM_CONTEXT_WINDOW,
    LLM_MAX_BATCH_SIZE,
    LLM_MAX_OUTPUT_TOKENS,
    LLM_MAX_WAIT_MS,
    LLM_TEMPERATURE,
    LOG_LEVEL,
    MEDGEMMA_MODEL_PATH,
    MEDASR_MODEL_PATH,
    MEDASR_TOKENS_PATH,
    PROJECT_ROOT,
    ROLLING_BUFFER_DURATION_S,
    RULE_DEBOUNCE_SECONDS,
    RULE_ENGINE_LATENCY_TARGET_MS,
    VAD_AGGRESSIVENESS,
    VAD_FRAME_DURATION_MS,
)


class Settings(BaseSettings):
    """Application configuration with validation and defaults."""

    # --- General ---
    project_name: str = "OR-Symphony"
    environment: str = Field(default="development", description="development | staging | production")
    debug: bool = True
    log_level: str = LOG_LEVEL
    default_surgery: str = DEFAULT_SURGERY

    # --- API ---
    api_host: str = API_HOST
    api_port: int = API_PORT

    # --- Audio ---
    audio_sample_rate: int = AUDIO_SAMPLE_RATE
    vad_frame_duration_ms: int = VAD_FRAME_DURATION_MS
    vad_aggressiveness: int = VAD_AGGRESSIVENESS
    chunk_min_duration_s: float = CHUNK_MIN_DURATION_S
    chunk_max_duration_s: float = CHUNK_MAX_DURATION_S
    chunk_overlap_ms: int = CHUNK_OVERLAP_MS

    # --- ASR ---
    medasr_model_path: str = str(MEDASR_MODEL_PATH)
    medasr_tokens_path: str = str(MEDASR_TOKENS_PATH)
    asr_latency_target_ms: int = ASR_CHUNK_LATENCY_TARGET_MS

    # --- LLM (MedGemma GGUF) ---
    medgemma_model_path: str = str(MEDGEMMA_MODEL_PATH)
    llm_max_batch_size: int = LLM_MAX_BATCH_SIZE
    llm_max_wait_ms: int = LLM_MAX_WAIT_MS
    llm_context_window: int = LLM_CONTEXT_WINDOW
    llm_max_output_tokens: int = LLM_MAX_OUTPUT_TOKENS
    llm_temperature: float = LLM_TEMPERATURE

    # --- Rule Engine ---
    rule_latency_target_ms: int = RULE_ENGINE_LATENCY_TARGET_MS
    rule_debounce_seconds: float = RULE_DEBOUNCE_SECONDS

    # --- Rolling Buffer ---
    rolling_buffer_duration_s: int = ROLLING_BUFFER_DURATION_S

    model_config = {
        "env_file": str(PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "env_prefix": "OR_SYMPHONY_",
        "case_sensitive": False,
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get cached application settings.

    Returns:
        Settings instance (cached after first call).
    """
    return Settings()


if __name__ == "__main__":
    settings = get_settings()
    print(f"Project: {settings.project_name}")
    print(f"Environment: {settings.environment}")
    print(f"API: {settings.api_host}:{settings.api_port}")
    print(f"MedASR model: {settings.medasr_model_path}")
    print(f"MedGemma model: {settings.medgemma_model_path}")
    print(f"Default surgery: {settings.default_surgery}")
