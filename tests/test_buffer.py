"""Tests for src.state.rolling_buffer — Enhanced Rolling Transcript Buffer.

Covers:
  - Basic creation, append, eviction
  - Time-based windowing at 180s boundary
  - Speaker tracking and filtering
  - get_context and get_context_for_llm
  - get_entries_since (incremental queries)
  - get_final_entries, get_entries_by_speaker
  - get_summary statistics
  - to_dict / from_dict serialization round-trip
  - save / load file persistence
  - BufferEntry.format_line
  - Edge cases (empty buffer, single entry, etc.)
  - Long simulation (200+ entries)
"""

import json
import time
from pathlib import Path

import pytest

from src.state.rolling_buffer import BufferEntry, RollingBuffer


# ── BufferEntry ──────────────────────────────────────────────────────────


class TestBufferEntry:
    def test_create_defaults(self):
        entry = BufferEntry(text="hello", timestamp=1.0)
        assert entry.text == "hello"
        assert entry.timestamp == 1.0
        assert entry.speaker == "unknown"
        assert entry.is_final is True
        assert entry.source == "asr"

    def test_create_full(self):
        entry = BufferEntry(
            text="incision", timestamp=42.5, speaker="surgeon",
            is_final=False, source="override"
        )
        assert entry.speaker == "surgeon"
        assert entry.is_final is False
        assert entry.source == "override"

    def test_to_dict(self):
        entry = BufferEntry(text="test", timestamp=1.0, speaker="nurse")
        d = entry.to_dict()
        assert d["text"] == "test"
        assert d["timestamp"] == 1.0
        assert d["speaker"] == "nurse"
        assert d["is_final"] is True
        assert d["source"] == "asr"

    def test_from_dict(self):
        d = {"text": "hello", "timestamp": 5.0, "speaker": "surgeon",
             "is_final": False, "source": "system"}
        entry = BufferEntry.from_dict(d)
        assert entry.text == "hello"
        assert entry.timestamp == 5.0
        assert entry.speaker == "surgeon"
        assert entry.is_final is False
        assert entry.source == "system"

    def test_from_dict_defaults(self):
        entry = BufferEntry.from_dict({})
        assert entry.text == ""
        assert entry.timestamp == 0.0
        assert entry.speaker == "unknown"
        assert entry.is_final is True

    def test_format_line_with_timestamp(self):
        entry = BufferEntry(text="needle in", timestamp=42.5, speaker="surgeon")
        line = entry.format_line(include_timestamp=True)
        assert "[42.5s]" in line
        assert "surgeon:" in line
        assert "needle in" in line

    def test_format_line_without_timestamp(self):
        entry = BufferEntry(text="hello", timestamp=1.0, speaker="nurse")
        line = entry.format_line(include_timestamp=False)
        assert "[" not in line
        assert "nurse:" in line
        assert "hello" in line

    def test_format_line_unknown_speaker(self):
        entry = BufferEntry(text="test", timestamp=0.0, speaker="unknown")
        line = entry.format_line()
        assert "unknown:" not in line
        assert "test" in line


# ── RollingBuffer — basic ────────────────────────────────────────────────


class TestRollingBufferBasic:
    def test_create_default(self):
        buf = RollingBuffer()
        assert buf.max_duration_s == 180
        assert buf.entry_count == 0
        assert buf.duration_s == 0.0

    def test_create_custom_duration(self):
        buf = RollingBuffer(max_duration_s=60)
        assert buf.max_duration_s == 60

    def test_append_increments_count(self):
        buf = RollingBuffer()
        buf.append("hello", timestamp=0.0)
        assert buf.entry_count == 1

    def test_append_with_all_params(self):
        buf = RollingBuffer()
        buf.append("test", timestamp=1.0, speaker="surgeon", is_final=False, source="override")
        entries = buf.get_entries()
        assert entries[0].speaker == "surgeon"
        assert entries[0].is_final is False
        assert entries[0].source == "override"

    def test_append_auto_timestamp(self):
        buf = RollingBuffer()
        buf.append("hello")
        assert buf.entry_count == 1
        assert buf.latest_timestamp is not None
        assert buf.latest_timestamp >= 0.0

    def test_properties_empty(self):
        buf = RollingBuffer()
        assert buf.earliest_timestamp is None
        assert buf.latest_timestamp is None
        assert buf.entry_count == 0
        assert buf.duration_s == 0.0

    def test_properties_single_entry(self):
        buf = RollingBuffer()
        buf.append("one", timestamp=5.0)
        assert buf.earliest_timestamp == 5.0
        assert buf.latest_timestamp == 5.0
        assert buf.duration_s == 0.0

    def test_properties_multiple_entries(self):
        buf = RollingBuffer()
        buf.append("a", timestamp=10.0)
        buf.append("b", timestamp=20.0)
        buf.append("c", timestamp=30.0)
        assert buf.earliest_timestamp == 10.0
        assert buf.latest_timestamp == 30.0
        assert buf.duration_s == 20.0

    def test_clear(self):
        buf = RollingBuffer()
        buf.append("x", timestamp=1.0)
        buf.append("y", timestamp=2.0)
        buf.clear()
        assert buf.entry_count == 0
        assert buf.earliest_timestamp is None

    def test_reset_session(self):
        buf = RollingBuffer()
        buf.append("x", timestamp=1.0)
        buf.reset_session()
        assert buf.entry_count == 0
        assert buf._total_appended == 0
        assert buf._total_evicted == 0

    def test_session_elapsed(self):
        buf = RollingBuffer()
        elapsed = buf.session_elapsed_s
        assert elapsed >= 0.0
        assert elapsed < 5.0  # Should be nearly instant in test


# ── Eviction ─────────────────────────────────────────────────────────────


class TestRollingBufferEviction:
    def test_eviction_at_boundary(self):
        buf = RollingBuffer(max_duration_s=10)
        for i in range(15):
            buf.append(f"seg{i}", timestamp=float(i))
        # Window is [5..14] = 10s window
        assert buf.entry_count == 11  # timestamps 4-14, window is latest(14)-10=4 cutoff
        assert buf.earliest_timestamp >= 4.0

    def test_eviction_keeps_within_window(self):
        buf = RollingBuffer(max_duration_s=5)
        for i in range(20):
            buf.append(f"seg{i}", timestamp=float(i))
        # Latest is 19, cutoff is 14
        assert buf.earliest_timestamp >= 14.0
        assert buf.latest_timestamp == 19.0
        assert buf.duration_s <= 5.0

    def test_eviction_single_large_gap(self):
        buf = RollingBuffer(max_duration_s=10)
        buf.append("early", timestamp=0.0)
        buf.append("late", timestamp=100.0)
        # "early" should be evicted (gap > 10s)
        assert buf.entry_count == 1
        assert buf.get_entries()[0].text == "late"

    def test_eviction_counters(self):
        buf = RollingBuffer(max_duration_s=5)
        for i in range(10):
            buf.append(f"seg{i}", timestamp=float(i))
        assert buf._total_appended == 10
        assert buf._total_evicted > 0


# ── get_context ──────────────────────────────────────────────────────────


class TestGetContext:
    def test_basic_context(self):
        buf = RollingBuffer()
        buf.append("hello world", timestamp=0.0)
        buf.append("goodbye world", timestamp=1.0)
        ctx = buf.get_context()
        assert "hello world" in ctx
        assert "goodbye world" in ctx

    def test_context_max_entries(self):
        buf = RollingBuffer()
        for i in range(10):
            buf.append(f"seg{i}", timestamp=float(i))
        ctx = buf.get_context(max_entries=3)
        assert "seg7" in ctx
        assert "seg8" in ctx
        assert "seg9" in ctx
        assert "seg0" not in ctx

    def test_context_empty(self):
        buf = RollingBuffer()
        assert buf.get_context() == ""

    def test_context_skips_blank(self):
        buf = RollingBuffer()
        buf.append("hello", timestamp=0.0)
        buf.append("  ", timestamp=1.0)
        buf.append("world", timestamp=2.0)
        ctx = buf.get_context()
        assert ctx == "hello world"


# ── get_context_for_llm ──────────────────────────────────────────────────


class TestGetContextForLLM:
    def test_empty_returns_placeholder(self):
        buf = RollingBuffer()
        ctx = buf.get_context_for_llm()
        assert ctx == "<no recent transcript>"

    def test_header_includes_surgery(self):
        buf = RollingBuffer()
        buf.append("patient is prepped", timestamp=1.0, speaker="surgeon")
        ctx = buf.get_context_for_llm(surgery="PCNL")
        assert "Surgery: PCNL" in ctx

    def test_header_includes_phase(self):
        buf = RollingBuffer()
        buf.append("starting incision", timestamp=1.0)
        ctx = buf.get_context_for_llm(phase="Phase2")
        assert "Phase: Phase2" in ctx

    def test_contains_transcript_lines(self):
        buf = RollingBuffer()
        buf.append("first line", timestamp=1.0, speaker="surgeon")
        buf.append("second line", timestamp=2.0, speaker="nurse")
        ctx = buf.get_context_for_llm()
        assert "surgeon:" in ctx
        assert "first line" in ctx
        assert "nurse:" in ctx
        assert "second line" in ctx

    def test_includes_timestamps(self):
        buf = RollingBuffer()
        buf.append("test", timestamp=42.5, speaker="surgeon")
        ctx = buf.get_context_for_llm(include_timestamps=True)
        assert "[42.5s]" in ctx

    def test_excludes_timestamps(self):
        buf = RollingBuffer()
        buf.append("test", timestamp=42.5, speaker="surgeon")
        ctx = buf.get_context_for_llm(include_timestamps=False)
        assert "[" not in ctx

    def test_excludes_partial_transcripts(self):
        buf = RollingBuffer()
        buf.append("final text", timestamp=1.0, is_final=True)
        buf.append("partial text", timestamp=2.0, is_final=False)
        ctx = buf.get_context_for_llm()
        assert "final text" in ctx
        assert "partial text" not in ctx

    def test_max_entries(self):
        buf = RollingBuffer()
        for i in range(10):
            buf.append(f"line{i}", timestamp=float(i))
        ctx = buf.get_context_for_llm(max_entries=3)
        assert "line9" in ctx
        assert "line0" not in ctx

    def test_has_delimiters(self):
        buf = RollingBuffer()
        buf.append("test", timestamp=1.0)
        ctx = buf.get_context_for_llm()
        assert "--- Recent OR Transcript ---" in ctx
        assert "--- End Transcript ---" in ctx

    def test_only_partials_returns_placeholder(self):
        buf = RollingBuffer()
        buf.append("partial1", timestamp=1.0, is_final=False)
        buf.append("partial2", timestamp=2.0, is_final=False)
        ctx = buf.get_context_for_llm()
        assert ctx == "<no recent transcript>"


# ── get_entries_since ────────────────────────────────────────────────────


class TestGetEntriesSince:
    def test_returns_entries_at_and_after(self):
        buf = RollingBuffer()
        for i in range(5):
            buf.append(f"seg{i}", timestamp=float(i))
        entries = buf.get_entries_since(3.0)
        assert len(entries) == 2
        assert entries[0].text == "seg3"
        assert entries[1].text == "seg4"

    def test_all_before(self):
        buf = RollingBuffer()
        buf.append("old", timestamp=1.0)
        entries = buf.get_entries_since(100.0)
        assert len(entries) == 0

    def test_all_after(self):
        buf = RollingBuffer()
        for i in range(5):
            buf.append(f"seg{i}", timestamp=float(i + 10))
        entries = buf.get_entries_since(0.0)
        assert len(entries) == 5

    def test_exact_boundary(self):
        buf = RollingBuffer()
        buf.append("a", timestamp=5.0)
        buf.append("b", timestamp=10.0)
        entries = buf.get_entries_since(5.0)
        assert len(entries) == 2

    def test_empty_buffer(self):
        buf = RollingBuffer()
        entries = buf.get_entries_since(0.0)
        assert len(entries) == 0


# ── get_final_entries / get_entries_by_speaker ───────────────────────────


class TestEntryFilters:
    def test_get_final_entries(self):
        buf = RollingBuffer()
        buf.append("final1", timestamp=1.0, is_final=True)
        buf.append("partial", timestamp=2.0, is_final=False)
        buf.append("final2", timestamp=3.0, is_final=True)
        finals = buf.get_final_entries()
        assert len(finals) == 2
        assert finals[0].text == "final1"
        assert finals[1].text == "final2"

    def test_get_entries_by_speaker(self):
        buf = RollingBuffer()
        buf.append("a", timestamp=1.0, speaker="surgeon")
        buf.append("b", timestamp=2.0, speaker="nurse")
        buf.append("c", timestamp=3.0, speaker="surgeon")
        surgeon = buf.get_entries_by_speaker("surgeon")
        assert len(surgeon) == 2

    def test_speaker_case_insensitive(self):
        buf = RollingBuffer()
        buf.append("a", timestamp=1.0, speaker="Surgeon")
        entries = buf.get_entries_by_speaker("surgeon")
        assert len(entries) == 1

    def test_no_matching_speaker(self):
        buf = RollingBuffer()
        buf.append("a", timestamp=1.0, speaker="nurse")
        entries = buf.get_entries_by_speaker("surgeon")
        assert len(entries) == 0


# ── get_summary ──────────────────────────────────────────────────────────


class TestGetSummary:
    def test_summary_empty(self):
        buf = RollingBuffer()
        s = buf.get_summary()
        assert s["entry_count"] == 0
        assert s["duration_s"] == 0.0
        assert s["earliest_timestamp"] is None
        assert s["latest_timestamp"] is None
        assert s["total_appended"] == 0
        assert s["total_evicted"] == 0
        assert s["speakers"] == []
        assert s["sources"] == []

    def test_summary_populated(self):
        buf = RollingBuffer()
        buf.append("a", timestamp=1.0, speaker="surgeon", source="asr")
        buf.append("b", timestamp=5.0, speaker="nurse", source="override")
        s = buf.get_summary()
        assert s["entry_count"] == 2
        assert s["duration_s"] == 4.0
        assert s["earliest_timestamp"] == 1.0
        assert s["latest_timestamp"] == 5.0
        assert s["total_appended"] == 2
        assert s["total_evicted"] == 0
        assert "surgeon" in s["speakers"]
        assert "nurse" in s["speakers"]
        assert "asr" in s["sources"]
        assert "override" in s["sources"]

    def test_summary_max_duration(self):
        buf = RollingBuffer(max_duration_s=60)
        s = buf.get_summary()
        assert s["max_duration_s"] == 60

    def test_summary_session_elapsed(self):
        buf = RollingBuffer()
        s = buf.get_summary()
        assert s["session_elapsed_s"] >= 0.0


# ── Serialization: to_dict / from_dict ───────────────────────────────────


class TestBufferSerialization:
    def test_roundtrip(self):
        buf = RollingBuffer(max_duration_s=60)
        buf.append("hello", timestamp=1.0, speaker="surgeon")
        buf.append("world", timestamp=2.0, speaker="nurse", is_final=False)

        data = buf.to_dict()
        restored = RollingBuffer.from_dict(data)

        assert restored.max_duration_s == 60
        assert restored.entry_count == 2
        entries = restored.get_entries()
        assert entries[0].text == "hello"
        assert entries[0].speaker == "surgeon"
        assert entries[1].text == "world"
        assert entries[1].is_final is False

    def test_to_dict_structure(self):
        buf = RollingBuffer(max_duration_s=30)
        buf.append("test", timestamp=0.0)
        d = buf.to_dict()
        assert "max_duration_s" in d
        assert "session_start" in d
        assert "total_appended" in d
        assert "total_evicted" in d
        assert "entries" in d
        assert len(d["entries"]) == 1
        assert d["entries"][0]["text"] == "test"

    def test_from_dict_empty(self):
        buf = RollingBuffer.from_dict({"entries": []})
        assert buf.entry_count == 0

    def test_roundtrip_preserves_counters(self):
        buf = RollingBuffer(max_duration_s=5)
        for i in range(10):
            buf.append(f"s{i}", timestamp=float(i))
        data = buf.to_dict()
        restored = RollingBuffer.from_dict(data)
        assert restored._total_appended == 10
        assert restored._total_evicted > 0


# ── File persistence: save / load ────────────────────────────────────────


class TestBufferPersistence:
    def test_save_and_load(self, tmp_path):
        buf = RollingBuffer(max_duration_s=30)
        buf.append("hello", timestamp=1.0, speaker="surgeon")
        buf.append("world", timestamp=5.0, speaker="nurse")

        path = tmp_path / "buffer_state.json"
        buf.save(path)

        assert path.exists()
        loaded = RollingBuffer.load(path)
        assert loaded.entry_count == 2
        assert loaded.max_duration_s == 30
        entries = loaded.get_entries()
        assert entries[0].text == "hello"
        assert entries[1].speaker == "nurse"

    def test_save_creates_directory(self, tmp_path):
        buf = RollingBuffer()
        buf.append("test", timestamp=0.0)
        path = tmp_path / "nested" / "dir" / "buf.json"
        buf.save(path)
        assert path.exists()

    def test_saved_file_is_valid_json(self, tmp_path):
        buf = RollingBuffer()
        buf.append("valid json test", timestamp=1.0)
        path = tmp_path / "test.json"
        buf.save(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, dict)
        assert "entries" in data


# ── Long simulation ─────────────────────────────────────────────────────


class TestLongSimulation:
    def test_200_entries_eviction(self):
        """Simulate 200+ transcript entries over a surgery session."""
        buf = RollingBuffer(max_duration_s=30)  # 30s window
        speakers = ["surgeon", "anesthesiologist", "nurse"]

        for i in range(200):
            buf.append(
                f"segment {i}",
                timestamp=float(i),
                speaker=speakers[i % 3],
            )

        # Should keep only ~31 entries (timestamps 169-199)
        assert buf.entry_count <= 32
        assert buf.entry_count >= 30
        assert buf.earliest_timestamp >= 169.0
        assert buf.latest_timestamp == 199.0
        assert buf.duration_s <= 30.0

        # Summary should reflect stats
        s = buf.get_summary()
        assert s["total_appended"] == 200
        assert s["total_evicted"] > 0
        assert len(s["speakers"]) == 3

    def test_500_entries_performance(self):
        """Ensure 500 appends complete quickly."""
        buf = RollingBuffer(max_duration_s=60)
        start = time.time()
        for i in range(500):
            buf.append(f"seg{i}", timestamp=float(i * 0.5))
        elapsed = time.time() - start
        assert elapsed < 1.0  # Should be <100ms in practice
        assert buf.entry_count <= 122  # 60s window at 0.5s intervals

    def test_mixed_final_partial(self):
        """Simulate mix of final and partial transcripts."""
        buf = RollingBuffer(max_duration_s=20)

        for i in range(50):
            is_final = (i % 3 != 0)  # Every 3rd is partial
            buf.append(f"seg{i}", timestamp=float(i), is_final=is_final)

        finals = buf.get_final_entries()
        all_entries = buf.get_entries()
        assert len(finals) < len(all_entries)

        # LLM context should only have final entries
        ctx = buf.get_context_for_llm()
        # Partials should not appear
        for e in all_entries:
            if not e.is_final:
                assert e.text not in ctx
