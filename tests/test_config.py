"""Tests for src.utils.config module."""

from src.utils.config import Settings, get_settings


class TestSettings:
    def test_settings_creates(self):
        settings = Settings()
        assert settings.project_name == "OR-Symphony"

    def test_default_surgery(self):
        settings = Settings()
        assert settings.default_surgery == "PCNL"

    def test_api_defaults(self):
        settings = Settings()
        assert settings.api_host == "0.0.0.0"
        assert settings.api_port == 8000

    def test_audio_defaults(self):
        settings = Settings()
        assert settings.audio_sample_rate == 16000
        assert settings.vad_aggressiveness in range(4)

    def test_llm_defaults(self):
        settings = Settings()
        assert settings.llm_max_batch_size > 0
        assert settings.llm_temperature >= 0.0

    def test_rule_engine_defaults(self):
        settings = Settings()
        assert settings.rule_latency_target_ms > 0
        assert settings.rule_debounce_seconds > 0.0


class TestGetSettings:
    def test_cached_settings(self):
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2  # Should be the same cached instance
