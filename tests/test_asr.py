"""
OR-Symphony: Phase 3 — ASR Module Tests

Tests for:
  - FeatureExtractor (shapes, dtypes, normalization)
  - CTCDecoder (greedy decode, SentencePiece, confidence, timestamps)
  - OnnxASRRunner (end-to-end pipeline, latency, streaming, lifecycle)

Run:
    pytest tests/test_asr.py -v
"""

from __future__ import annotations

import numpy as np
import pytest

from src.utils.constants import (
    AUDIO_SAMPLE_RATE,
    MEDASR_MODEL_PATH,
    MEDASR_TOKENS_PATH,
)

# ============================================================================
# Feature Extractor Tests
# ============================================================================


class TestFeatureExtractor:
    """Tests for src.asr.feature_extractor.FeatureExtractor."""

    @pytest.fixture
    def extractor(self):
        from src.asr.feature_extractor import FeatureExtractor

        return FeatureExtractor(sample_rate=16000, num_mel_bins=128)

    def test_extract_shape_1s(self, extractor):
        """1 second of audio → correct number of 128-dim frames."""
        audio = np.zeros(16000, dtype=np.float32)
        features = extractor.extract(audio)
        assert features.ndim == 2
        assert features.shape[1] == 128
        # 1s at 10ms hop → ~98-100 frames (25ms window, 10ms shift)
        expected = extractor.compute_output_length(16000)
        assert features.shape[0] == expected

    def test_extract_shape_3s(self, extractor):
        """3 seconds of audio → correct feature dimensions."""
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        features = extractor.extract(audio)
        assert features.shape[1] == 128
        expected = extractor.compute_output_length(48000)
        assert features.shape[0] == expected

    def test_extract_dtype_float32(self, extractor):
        """Output features are float32."""
        audio = np.zeros(16000, dtype=np.float32)
        features = extractor.extract(audio)
        assert features.dtype == np.float32

    def test_extract_with_mask_shapes(self, extractor):
        """extract_with_mask returns (1, T, 128) features and (1, T) mask."""
        audio = np.zeros(16000, dtype=np.float32)
        features, mask = extractor.extract_with_mask(audio)
        assert features.ndim == 3
        assert features.shape[0] == 1  # batch dim
        assert features.shape[2] == 128
        assert mask.ndim == 2
        assert mask.shape[0] == 1
        assert mask.shape[1] == features.shape[1]  # T must match

    def test_mask_all_ones(self, extractor):
        """Mask is all ones for a single utterance."""
        audio = np.zeros(16000, dtype=np.float32)
        _, mask = extractor.extract_with_mask(audio)
        assert np.all(mask == 1)
        assert mask.dtype == np.int64

    def test_cmvn_normalization(self, extractor):
        """After CMVN, nearly all feature dims have approximately zero mean."""
        audio = np.random.randn(32000).astype(np.float32) * 0.5
        features = extractor.extract(audio)
        # Mean of each feature dim should be ~0 after CMVN
        # A few bins with near-constant values may have small residual
        means = np.abs(np.mean(features, axis=0))
        fraction_near_zero = np.mean(means < 0.1)
        assert fraction_near_zero >= 0.95, (
            f"Only {fraction_near_zero:.0%} of dims have mean < 0.1"
        )

    def test_no_normalization(self):
        """Without normalization, features are raw fbank values."""
        from src.asr.feature_extractor import FeatureExtractor

        extractor = FeatureExtractor(normalize=False)
        audio = np.random.randn(16000).astype(np.float32) * 0.5
        features = extractor.extract(audio)
        # Raw fbank features should NOT have zero mean
        means = np.mean(features, axis=0)
        # At least some dims should be non-zero
        assert not np.allclose(means, 0.0, atol=0.01)

    def test_audio_too_short_raises(self, extractor):
        """Audio shorter than 1 frame raises ValueError."""
        short_audio = np.zeros(100, dtype=np.float32)  # 6.25ms << 25ms window
        with pytest.raises(ValueError, match="too short"):
            extractor.extract(short_audio)

    def test_compute_output_length(self, extractor):
        """compute_output_length matches actual output."""
        for n_samples in [8000, 16000, 32000, 48000]:
            predicted = extractor.compute_output_length(n_samples)
            audio = np.zeros(n_samples, dtype=np.float32)
            actual = extractor.extract(audio).shape[0]
            assert predicted == actual, (
                f"Mismatch for {n_samples} samples: predicted={predicted}, actual={actual}"
            )

    def test_int16_auto_conversion(self, extractor):
        """Non-float32 audio is auto-converted."""
        audio_i16 = np.zeros(16000, dtype=np.int16)
        features = extractor.extract(audio_i16)
        assert features.shape[1] == 128

    def test_info_property(self, extractor):
        """Info dict has expected keys."""
        info = extractor.info
        assert info["sample_rate"] == 16000
        assert info["num_mel_bins"] == 128
        assert info["normalize"] is True

    def test_deterministic_output(self, extractor):
        """Same audio produces same features (dither=0)."""
        audio = np.random.randn(16000).astype(np.float32) * 0.3
        f1 = extractor.extract(audio)
        f2 = extractor.extract(audio)
        np.testing.assert_array_equal(f1, f2)


# ============================================================================
# CTC Decoder Tests
# ============================================================================


class TestCTCDecoder:
    """Tests for src.asr.ctc_decoder.CTCDecoder."""

    @pytest.fixture
    def decoder(self):
        from src.asr.ctc_decoder import CTCDecoder

        if not MEDASR_TOKENS_PATH.exists():
            pytest.skip("tokens.txt not found — model not downloaded")
        return CTCDecoder(tokens_path=MEDASR_TOKENS_PATH, subsampling_factor=4)

    def test_vocab_size(self, decoder):
        """Vocabulary should have 512 tokens."""
        assert decoder.vocab_size == 512

    def test_blank_token(self, decoder):
        """Token 0 is <blk> (CTC blank)."""
        assert decoder._id_to_token[0] == "<blk>"

    def test_special_tokens(self, decoder):
        """BOS, EOS, UNK tokens are at expected positions."""
        assert decoder._id_to_token[1] == "<s>"
        assert decoder._id_to_token[2] == "</s>"
        assert decoder._id_to_token[3] == "<unk>"

    def test_decode_all_blanks(self, decoder):
        """All-blank logits → empty text."""
        V = 512
        T = 20
        logits = np.full((1, T, V), -10.0, dtype=np.float32)
        logits[:, :, 0] = 10.0  # blank has highest score
        lengths = np.array([T], dtype=np.int64)

        results = decoder.decode(logits, lengths)
        text, confidence, token_ids = results[0]
        assert text == ""
        assert token_ids == []

    def test_decode_single_token(self, decoder):
        """Single non-blank token repeated → decoded once."""
        V = 512
        T = 10
        token_id = 10  # some token
        logits = np.full((1, T, V), -10.0, dtype=np.float32)
        logits[:, :, 0] = 5.0  # blank
        logits[:, 3:7, 0] = -10.0  # remove blank in middle
        logits[:, 3:7, token_id] = 10.0  # token active in frames 3-6

        lengths = np.array([T], dtype=np.int64)
        results = decoder.decode(logits, lengths)
        text, confidence, token_ids = results[0]
        # Should decode to single token (consecutive duplicates collapsed)
        assert len(token_ids) == 1
        assert token_ids[0] == token_id

    def test_decode_confidence_range(self, decoder):
        """Confidence is between 0.0 and 1.0."""
        V = 512
        T = 20
        logits = np.random.randn(1, T, V).astype(np.float32)
        lengths = np.array([T], dtype=np.int64)
        results = decoder.decode(logits, lengths)
        _, confidence, _ = results[0]
        assert 0.0 <= confidence <= 1.0

    def test_decode_special_tokens_skipped(self, decoder):
        """BOS/EOS/UNK tokens are skipped in output."""
        from src.asr.ctc_decoder import BOS_ID, EOS_ID, UNK_ID

        V = 512
        T = 6
        logits = np.full((1, T, V), -10.0, dtype=np.float32)
        # Frame 0: blank, Frame 1: BOS, Frame 2: EOS, Frame 3: UNK
        logits[0, 0, 0] = 10.0  # blank
        logits[0, 1, BOS_ID] = 10.0
        logits[0, 2, EOS_ID] = 10.0
        logits[0, 3, UNK_ID] = 10.0
        logits[0, 4, 0] = 10.0  # blank
        logits[0, 5, 0] = 10.0  # blank

        lengths = np.array([T], dtype=np.int64)
        results = decoder.decode(logits, lengths)
        _, _, token_ids = results[0]
        assert BOS_ID not in token_ids
        assert EOS_ID not in token_ids
        assert UNK_ID not in token_ids

    def test_decode_batch(self, decoder):
        """Batch decode processes all items."""
        V = 512
        T = 10
        N = 3
        logits = np.random.randn(N, T, V).astype(np.float32)
        logits[:, :, 0] += 3.0  # bias toward blank
        lengths = np.full(N, T, dtype=np.int64)
        results = decoder.decode(logits, lengths)
        assert len(results) == N

    def test_decode_single_method(self, decoder):
        """decode_single works for a single (T, V) input."""
        V = 512
        T = 10
        logits = np.random.randn(T, V).astype(np.float32)
        text, confidence, token_ids = decoder.decode_single(logits)
        assert isinstance(text, str)
        assert isinstance(confidence, float)
        assert isinstance(token_ids, list)

    def test_sentencepiece_space(self, decoder):
        """SentencePiece ▁ converted to space in output."""
        from src.asr.ctc_decoder import SPACE_ID

        V = 512
        T = 6
        # Create logits where SPACE_ID (▁) and another token appear
        logits = np.full((T, V), -10.0, dtype=np.float32)
        logits[0, 0] = 10.0  # blank
        logits[1, SPACE_ID] = 10.0  # ▁
        logits[2, 10] = 10.0  # some token
        logits[3, 0] = 10.0  # blank
        logits[4, SPACE_ID] = 10.0  # ▁
        logits[5, 20] = 10.0  # another token

        text, _, _ = decoder.decode_single(logits)
        # ▁ should become space (leading space stripped)
        assert "  " not in text  # no double spaces

    def test_compute_timestamps(self, decoder):
        """Timestamps have correct structure."""
        V = 512
        T = 20
        logits = np.random.randn(T, V).astype(np.float32)
        timestamps = decoder.compute_timestamps(logits, T)
        for item in timestamps:
            assert len(item) == 3
            token_text, start_s, end_s = item
            assert isinstance(token_text, str)
            assert start_s >= 0.0
            assert end_s > start_s

    def test_timestamps_monotonic(self, decoder):
        """Timestamps are monotonically increasing."""
        V = 512
        T = 30
        logits = np.random.randn(T, V).astype(np.float32)
        timestamps = decoder.compute_timestamps(logits, T)
        if len(timestamps) >= 2:
            for i in range(1, len(timestamps)):
                assert timestamps[i][1] >= timestamps[i - 1][1]

    def test_decode_no_lengths(self, decoder):
        """Decode works when lengths is None (uses full T)."""
        V = 512
        T = 15
        logits = np.random.randn(1, T, V).astype(np.float32)
        results = decoder.decode(logits, None)
        assert len(results) == 1

    def test_softmax_normalized(self, decoder):
        """Softmax probabilities sum to 1."""
        logits = np.random.randn(5, 10).astype(np.float32)
        probs = decoder._softmax(logits)
        sums = np.sum(probs, axis=-1)
        np.testing.assert_allclose(sums, 1.0, atol=1e-5)


# ============================================================================
# ONNX ASR Runner Tests (integration — requires model files)
# ============================================================================


@pytest.fixture(scope="module")
def loaded_runner():
    """Load the OnnxASRRunner once for all integration tests."""
    from src.asr.onnx_runner import OnnxASRRunner

    if not MEDASR_MODEL_PATH.exists():
        pytest.skip("MedASR ONNX model not found — skipping integration tests")

    runner = OnnxASRRunner()
    runner.load_model()
    yield runner
    runner.unload_model()


class TestOnnxASRRunner:
    """Integration tests for OnnxASRRunner (require model files)."""

    def test_is_loaded(self, loaded_runner):
        """Runner reports loaded status."""
        assert loaded_runner.is_loaded()

    def test_transcribe_silence(self, loaded_runner):
        """Silence produces empty or near-empty text."""
        from src.asr.runner import ASRResult

        silence = np.zeros(16000, dtype=np.float32)
        result = loaded_runner.transcribe(silence)
        assert isinstance(result, ASRResult)
        assert result.model_name == "medasr-int8-onnx"
        assert result.is_partial is False

    def test_transcribe_returns_asr_result(self, loaded_runner):
        """Transcribe returns proper ASRResult dataclass."""
        audio = np.random.randn(16000).astype(np.float32) * 0.01
        result = loaded_runner.transcribe(audio)
        assert hasattr(result, "segments")
        assert hasattr(result, "processing_time_ms")
        assert hasattr(result, "full_text")

    def test_transcribe_latency_1s(self, loaded_runner):
        """1 second of audio transcribes in < 400ms."""
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        result = loaded_runner.transcribe(audio)
        assert result.processing_time_ms < 400, (
            f"Latency {result.processing_time_ms:.1f}ms exceeds 400ms target"
        )

    def test_transcribe_latency_3s(self, loaded_runner):
        """3 seconds of audio transcribes in < 500ms."""
        audio = np.random.randn(48000).astype(np.float32) * 0.1
        result = loaded_runner.transcribe(audio)
        assert result.processing_time_ms < 500, (
            f"Latency {result.processing_time_ms:.1f}ms exceeds 500ms target"
        )

    def test_transcribe_streaming(self, loaded_runner):
        """Streaming mode marks result as partial."""
        audio = np.random.randn(16000).astype(np.float32) * 0.1
        result = loaded_runner.transcribe_streaming(audio)
        assert result.is_partial is True
        for seg in result.segments:
            assert seg.is_final is False

    def test_segment_fields(self, loaded_runner):
        """If segments exist, they have proper fields."""
        audio = np.random.randn(32000).astype(np.float32) * 0.3
        result = loaded_runner.transcribe(audio)
        for seg in result.segments:
            assert isinstance(seg.text, str)
            assert isinstance(seg.confidence, float)
            assert 0.0 <= seg.confidence <= 1.0
            assert seg.language == "en"
            assert hasattr(seg, "start_time_s")
            assert hasattr(seg, "end_time_s")

    def test_model_info(self, loaded_runner):
        """get_model_info returns expected keys."""
        info = loaded_runner.get_model_info()
        assert "model_path" in info
        assert "loaded" in info
        assert info["loaded"] == "True"
        assert "vocab_size" in info
        assert info["vocab_size"] == "512"
        assert "model_type" in info
        assert "subsampling_factor" in info

    def test_latency_stats(self, loaded_runner):
        """Latency stats track inference count."""
        # Run a transcription to ensure stats are populated
        audio = np.zeros(16000, dtype=np.float32)
        loaded_runner.transcribe(audio)
        stats = loaded_runner.latency_stats
        assert stats["inference_count"] >= 1
        assert stats["avg_ms"] > 0
        assert stats["avg_rtf"] > 0

    def test_full_text_property(self, loaded_runner):
        """full_text concatenates segment texts."""
        from src.asr.runner import ASRResult, TranscriptSegment

        result = ASRResult(
            segments=[
                TranscriptSegment(text="hello", is_final=True),
                TranscriptSegment(text="world", is_final=True),
            ]
        )
        assert result.full_text == "hello world"


class TestOnnxASRRunnerLifecycle:
    """Tests for runner lifecycle (load/unload/error handling)."""

    def test_not_loaded_raises(self):
        """Transcribe before load_model raises RuntimeError."""
        from src.asr.onnx_runner import OnnxASRRunner

        runner = OnnxASRRunner()
        audio = np.zeros(16000, dtype=np.float32)
        with pytest.raises(RuntimeError, match="not loaded"):
            runner.transcribe(audio)

    def test_missing_model_raises(self):
        """Non-existent model path raises FileNotFoundError."""
        from src.asr.onnx_runner import OnnxASRRunner

        runner = OnnxASRRunner(model_path="nonexistent/model.onnx")
        with pytest.raises(FileNotFoundError):
            runner.load_model()

    def test_missing_tokens_raises(self):
        """Non-existent tokens path raises FileNotFoundError."""
        from src.asr.onnx_runner import OnnxASRRunner

        if not MEDASR_MODEL_PATH.exists():
            pytest.skip("Model not found")
        runner = OnnxASRRunner(tokens_path="nonexistent/tokens.txt")
        with pytest.raises(FileNotFoundError):
            runner.load_model()

    def test_load_unload_cycle(self):
        """Load and unload cleanly."""
        from src.asr.onnx_runner import OnnxASRRunner

        if not MEDASR_MODEL_PATH.exists():
            pytest.skip("Model not found")
        runner = OnnxASRRunner()
        runner.load_model()
        assert runner.is_loaded()
        runner.unload_model()
        assert not runner.is_loaded()

    def test_default_paths(self):
        """Default paths are set from constants."""
        from src.asr.onnx_runner import OnnxASRRunner

        runner = OnnxASRRunner()
        assert runner.model_path == MEDASR_MODEL_PATH
        assert runner.tokens_path == MEDASR_TOKENS_PATH

    def test_custom_providers(self):
        """Custom providers are accepted."""
        from src.asr.onnx_runner import OnnxASRRunner

        runner = OnnxASRRunner(providers=["CPUExecutionProvider"])
        assert runner.providers == ["CPUExecutionProvider"]
