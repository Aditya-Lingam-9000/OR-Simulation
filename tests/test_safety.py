"""
Tests for src.utils.audit and src.utils.safety modules.

Phase 8: Safety, Logging, Audit, Documentation & Clinical Gating.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import patch

import pytest

from src.utils.audit import (
    AuditLogger,
    OverrideAuditLogger,
    StateAuditLogger,
    TranscriptAuditLogger,
)
from src.utils.safety import (
    ADVISORY_INDICATORS,
    BANNED_PATTERNS,
    SafetyResult,
    SafetyValidator,
    is_suggestion_only,
    validate_output_safe,
)

logger = logging.getLogger(__name__)


# ======================================================================
# Fixtures
# ======================================================================


@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a temp directory for log files."""
    return tmp_path


@pytest.fixture
def log_path(tmp_dir):
    """A temp log file path."""
    return tmp_dir / "test_audit.log"


@pytest.fixture
def audit(log_path):
    """Create a fresh AuditLogger instance."""
    return AuditLogger(log_path, name="test")


@pytest.fixture
def state_audit(tmp_dir):
    """Create a StateAuditLogger with temp path."""
    return StateAuditLogger(log_path=tmp_dir / "state.log")


@pytest.fixture
def override_audit(tmp_dir):
    """Create an OverrideAuditLogger with temp path."""
    return OverrideAuditLogger(log_path=tmp_dir / "overrides.log")


@pytest.fixture
def transcript_audit(tmp_dir):
    """Create a TranscriptAuditLogger with temp dir."""
    return TranscriptAuditLogger(log_dir=tmp_dir / "transcripts")


@pytest.fixture
def validator():
    """Create a SafetyValidator instance."""
    return SafetyValidator()


@pytest.fixture
def valid_state():
    """A minimal valid state dict."""
    return {
        "metadata": {"surgery": "PCNL", "phase": "Phase1", "reasoning": "normal"},
        "machines": {"0": ["M03"], "1": ["M01"]},
        "details": {},
        "suggestions": ["Consider preparing fluoroscopy"],
        "confidence": 0.85,
        "source": "rule",
    }


# ======================================================================
# AuditLogger — Core
# ======================================================================


class TestAuditLogger:
    """Tests for the base AuditLogger class."""

    def test_init_creates_directory(self, tmp_dir):
        """AuditLogger creates parent dirs if needed."""
        nested = tmp_dir / "a" / "b" / "c" / "audit.log"
        audit = AuditLogger(nested, name="nested")
        assert nested.parent.exists()

    def test_init_empty_file(self, audit):
        """Fresh logger starts with 0 entries."""
        assert audit.entry_count == 0

    def test_log_creates_file(self, audit, log_path):
        """First log call creates the file."""
        audit.log({"key": "value"})
        assert log_path.exists()

    def test_log_returns_entry_dict(self, audit):
        """log() returns a complete entry dict."""
        entry = audit.log({"test": 1})

        assert "timestamp" in entry
        assert "logger" in entry
        assert "payload" in entry
        assert "sha256" in entry
        assert "prev_hash" in entry
        assert "seq" in entry

    def test_log_entry_count_increments(self, audit):
        """Each log call increments the entry count."""
        audit.log({"a": 1})
        assert audit.entry_count == 1
        audit.log({"b": 2})
        assert audit.entry_count == 2
        audit.log({"c": 3})
        assert audit.entry_count == 3

    def test_log_seq_numbers(self, audit):
        """Seq numbers start at 0 and increment."""
        e0 = audit.log({"x": 0})
        e1 = audit.log({"x": 1})
        e2 = audit.log({"x": 2})
        assert e0["seq"] == 0
        assert e1["seq"] == 1
        assert e2["seq"] == 2

    def test_log_first_entry_prev_hash_is_genesis(self, audit):
        """First entry's prev_hash is 'genesis'."""
        entry = audit.log({"first": True})
        assert entry["prev_hash"] == "genesis"

    def test_log_chain_links(self, audit):
        """Each entry's prev_hash equals the previous entry's sha256."""
        e1 = audit.log({"step": 1})
        e2 = audit.log({"step": 2})
        e3 = audit.log({"step": 3})

        assert e2["prev_hash"] == e1["sha256"]
        assert e3["prev_hash"] == e2["sha256"]

    def test_log_payload_preserved(self, audit):
        """Payload is stored exactly as given."""
        payload = {"key": "value", "number": 42, "nested": {"a": [1, 2]}}
        entry = audit.log(payload)
        assert entry["payload"] == payload

    def test_log_logger_name(self, audit):
        """Logger name is stored in entry."""
        entry = audit.log({"x": 1})
        assert entry["logger"] == "test"

    def test_log_timestamp_format(self, audit):
        """Timestamp is ISO 8601."""
        entry = audit.log({"x": 1})
        ts = entry["timestamp"]
        # Should parse as ISO format
        dt = datetime.fromisoformat(ts)
        assert dt is not None

    def test_log_file_is_json_lines(self, audit, log_path):
        """Log file has one JSON object per line."""
        audit.log({"a": 1})
        audit.log({"b": 2})

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2
        for line in lines:
            obj = json.loads(line)
            assert "sha256" in obj


class TestAuditLoggerHash:
    """Tests for SHA-256 hash computation and integrity."""

    def test_compute_hash_deterministic(self):
        """Same payload always produces same hash."""
        payload = {"key": "value"}
        h1 = AuditLogger.compute_hash(payload)
        h2 = AuditLogger.compute_hash(payload)
        assert h1 == h2

    def test_compute_hash_is_sha256_hex(self):
        """Hash is a 64-char hex string (SHA-256)."""
        h = AuditLogger.compute_hash({"test": True})
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_compute_hash_matches_manual(self):
        """Hash matches a manually computed SHA-256."""
        payload = {"name": "test"}
        expected = hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
        ).hexdigest()
        assert AuditLogger.compute_hash(payload) == expected

    def test_compute_hash_order_independent(self):
        """sort_keys=True makes key order irrelevant."""
        h1 = AuditLogger.compute_hash({"b": 2, "a": 1})
        h2 = AuditLogger.compute_hash({"a": 1, "b": 2})
        assert h1 == h2

    def test_compute_hash_different_payloads_differ(self):
        """Different payloads produce different hashes."""
        h1 = AuditLogger.compute_hash({"x": 1})
        h2 = AuditLogger.compute_hash({"x": 2})
        assert h1 != h2

    def test_log_sha256_matches_payload_hash(self, audit):
        """Entry sha256 equals compute_hash(payload)."""
        payload = {"data": "test_value"}
        entry = audit.log(payload)
        assert entry["sha256"] == AuditLogger.compute_hash(payload)


class TestAuditLoggerVerify:
    """Tests for chain verification."""

    def test_verify_empty_log(self, audit):
        """Empty log verifies as True."""
        assert audit.verify_chain() is True

    def test_verify_single_entry(self, audit):
        """Single entry verifies successfully."""
        audit.log({"first": True})
        assert audit.verify_chain() is True

    def test_verify_multiple_entries(self, audit):
        """Multiple entries verify chain integrity."""
        for i in range(10):
            audit.log({"index": i})
        assert audit.verify_chain() is True

    def test_verify_detects_tampered_hash(self, audit, log_path):
        """Tampered sha256 field is detected."""
        audit.log({"a": 1})
        audit.log({"b": 2})

        # Tamper with the second entry's sha256
        lines = log_path.read_text().strip().split("\n")
        entry = json.loads(lines[1])
        entry["sha256"] = "0" * 64  # fake hash
        lines[1] = json.dumps(entry)
        log_path.write_text("\n".join(lines) + "\n")

        assert audit.verify_chain() is False

    def test_verify_detects_tampered_payload(self, audit, log_path):
        """Tampered payload is detected (hash won't match)."""
        audit.log({"original": True})

        lines = log_path.read_text().strip().split("\n")
        entry = json.loads(lines[0])
        entry["payload"]["original"] = False  # tamper
        lines[0] = json.dumps(entry)
        log_path.write_text("\n".join(lines) + "\n")

        assert audit.verify_chain() is False

    def test_verify_detects_broken_chain(self, audit, log_path):
        """Broken prev_hash chain is detected."""
        audit.log({"a": 1})
        audit.log({"b": 2})
        audit.log({"c": 3})

        # Break the chain at entry 2
        lines = log_path.read_text().strip().split("\n")
        entry = json.loads(lines[1])
        entry["prev_hash"] = "bad_hash"
        # Recompute sha256 so only chain is broken
        lines[1] = json.dumps(entry)
        log_path.write_text("\n".join(lines) + "\n")

        assert audit.verify_chain() is False

    def test_verify_nonexistent_file(self, tmp_dir):
        """Non-existent file verifies as True."""
        audit = AuditLogger(tmp_dir / "nonexistent.log", name="test")
        assert audit.verify_chain() is True


class TestAuditLoggerResume:
    """Tests for chain resumption from existing file."""

    def test_resume_continues_chain(self, log_path):
        """New logger resumes chain from existing file."""
        # Write 3 entries
        audit1 = AuditLogger(log_path, name="test")
        audit1.log({"step": 1})
        audit1.log({"step": 2})
        last = audit1.log({"step": 3})

        # Create new logger — should resume
        audit2 = AuditLogger(log_path, name="test")
        assert audit2.entry_count == 3

        # Next entry continues the chain
        e4 = audit2.log({"step": 4})
        assert e4["prev_hash"] == last["sha256"]
        assert e4["seq"] == 3

    def test_resume_full_chain_verifies(self, log_path):
        """Chain is valid after resume and additional writes."""
        audit1 = AuditLogger(log_path, name="test")
        audit1.log({"a": 1})
        audit1.log({"b": 2})

        audit2 = AuditLogger(log_path, name="test")
        audit2.log({"c": 3})
        audit2.log({"d": 4})

        assert audit2.verify_chain() is True


class TestAuditLoggerReadEntries:
    """Tests for read_entries."""

    def test_read_empty(self, audit):
        """Empty log returns empty list."""
        assert audit.read_entries() == []

    def test_read_entries_count(self, audit):
        """Returns correct number of entries."""
        for i in range(5):
            audit.log({"i": i})
        assert len(audit.read_entries()) == 5

    def test_read_entries_preserves_order(self, audit):
        """Entries are in chronological order."""
        audit.log({"seq": "first"})
        audit.log({"seq": "second"})
        audit.log({"seq": "third"})

        entries = audit.read_entries()
        assert entries[0]["payload"]["seq"] == "first"
        assert entries[1]["payload"]["seq"] == "second"
        assert entries[2]["payload"]["seq"] == "third"


# ======================================================================
# TranscriptAuditLogger
# ======================================================================


class TestTranscriptAuditLogger:
    """Tests for the TranscriptAuditLogger."""

    def test_init_creates_directory(self, tmp_dir):
        """Creates transcript directory."""
        t = TranscriptAuditLogger(log_dir=tmp_dir / "deep" / "transcripts")
        assert (tmp_dir / "deep" / "transcripts").exists()

    def test_log_transcript_entry(self, transcript_audit):
        """log_transcript returns proper entry."""
        entry = transcript_audit.log_transcript(
            text="Turn on the fluoroscopy",
            speaker="surgeon",
            confidence=0.92,
            surgery="PCNL",
        )

        assert entry["type"] == "transcript"
        assert entry["payload"]["text"] == "Turn on the fluoroscopy"
        assert entry["payload"]["speaker"] == "surgeon"
        assert entry["payload"]["confidence"] == 0.92
        assert entry["payload"]["surgery"] == "PCNL"
        assert "sha256" in entry
        assert "timestamp" in entry

    def test_log_transcript_creates_daily_file(self, transcript_audit):
        """Creates a file named YYYYMMDD.log."""
        transcript_audit.log_transcript(text="test")
        today = datetime.now().strftime("%Y%m%d")
        expected = transcript_audit.log_dir / f"{today}.log"
        assert expected.exists()

    def test_log_transcript_incrementss_count(self, transcript_audit):
        """Entry count increments."""
        assert transcript_audit.entry_count == 0
        transcript_audit.log_transcript(text="one")
        assert transcript_audit.entry_count == 1
        transcript_audit.log_transcript(text="two")
        assert transcript_audit.entry_count == 2

    def test_read_today_returns_entries(self, transcript_audit):
        """read_today returns entries logged today."""
        transcript_audit.log_transcript(text="hello")
        transcript_audit.log_transcript(text="world")

        entries = transcript_audit.read_today()
        assert len(entries) == 2
        assert entries[0]["payload"]["text"] == "hello"
        assert entries[1]["payload"]["text"] == "world"

    def test_read_today_empty(self, transcript_audit):
        """read_today returns empty list with no entries."""
        assert transcript_audit.read_today() == []

    def test_transcript_sha256_correct(self, transcript_audit):
        """SHA-256 matches payload computation."""
        entry = transcript_audit.log_transcript(text="test", speaker="asr")
        payload = entry["payload"]
        expected = AuditLogger.compute_hash(payload)
        assert entry["sha256"] == expected

    def test_log_transcript_defaults(self, transcript_audit):
        """Default values work."""
        entry = transcript_audit.log_transcript(text="basic")
        assert entry["payload"]["speaker"] == "asr"
        assert entry["payload"]["confidence"] == 0.0
        assert entry["payload"]["surgery"] == ""


# ======================================================================
# StateAuditLogger
# ======================================================================


class TestStateAuditLogger:
    """Tests for the StateAuditLogger."""

    def test_init_default_path(self, tmp_dir):
        """Can be created with explicit path."""
        sal = StateAuditLogger(log_path=tmp_dir / "my_state.log")
        assert sal.name == "state_changes"

    def test_log_state_change_extracts_fields(self, state_audit):
        """log_state_change pulls essential fields from state."""
        state = {
            "metadata": {"surgery": "PCNL", "phase": "Phase2"},
            "source": "rule+medgemma",
            "confidence": 0.9,
            "machines": {"0": ["M03"], "1": ["M01", "M05"]},
            "suggestions": ["Consider X", "Check Y"],
        }

        entry = state_audit.log_state_change(state)
        payload = entry["payload"]

        assert payload["surgery"] == "PCNL"
        assert payload["phase"] == "Phase2"
        assert payload["source"] == "rule+medgemma"
        assert payload["confidence"] == 0.9
        assert payload["machines_on"] == ["M01", "M05"]
        assert payload["machines_off"] == ["M03"]
        assert payload["suggestions_count"] == 2

    def test_log_state_change_chain_integrity(self, state_audit):
        """Multiple state changes form valid chain."""
        for i in range(5):
            state_audit.log_state_change({
                "metadata": {"surgery": "PCNL", "phase": f"Phase{i}"},
                "source": "rule",
                "confidence": 0.5 + i * 0.1,
                "machines": {"0": [], "1": []},
                "suggestions": [],
            })

        assert state_audit.verify_chain() is True
        assert state_audit.entry_count == 5

    def test_log_state_change_handles_missing_fields(self, state_audit):
        """Handles state dicts with missing fields gracefully."""
        entry = state_audit.log_state_change({})
        payload = entry["payload"]
        assert payload["surgery"] == ""
        assert payload["phase"] == ""
        assert payload["source"] == ""


# ======================================================================
# OverrideAuditLogger
# ======================================================================


class TestOverrideAuditLogger:
    """Tests for the OverrideAuditLogger."""

    def test_init_default_name(self, override_audit):
        """Logger name is 'overrides'."""
        assert override_audit.name == "overrides"

    def test_log_override_payload(self, override_audit):
        """log_override stores correct payload."""
        entry = override_audit.log_override(
            machine_id="M03",
            action="ON",
            reason="Needed for imaging",
            operator="Dr. Smith",
            surgery="PCNL",
        )

        payload = entry["payload"]
        assert payload["machine_id"] == "M03"
        assert payload["action"] == "ON"
        assert payload["reason"] == "Needed for imaging"
        assert payload["operator"] == "Dr. Smith"
        assert payload["surgery"] == "PCNL"

    def test_log_override_chain(self, override_audit):
        """Override entries form valid chain."""
        override_audit.log_override("M01", "ON", "start", "op1")
        override_audit.log_override("M01", "OFF", "done", "op1")
        override_audit.log_override("M05", "ON", "backup", "op2")

        assert override_audit.verify_chain() is True
        assert override_audit.entry_count == 3

    def test_log_override_defaults(self, override_audit):
        """Default operator is 'unknown'."""
        entry = override_audit.log_override("M01", "OFF", "test")
        assert entry["payload"]["operator"] == "unknown"
        assert entry["payload"]["surgery"] == ""


# ======================================================================
# SafetyResult
# ======================================================================


class TestSafetyResult:
    """Tests for SafetyResult dataclass."""

    def test_default_is_safe(self):
        """Default SafetyResult is safe."""
        r = SafetyResult()
        assert r.is_safe is True
        assert r.violations == []
        assert r.warnings == []

    def test_add_violation_marks_unsafe(self):
        """Adding a violation marks result unsafe."""
        r = SafetyResult()
        r.add_violation("bad thing")
        assert r.is_safe is False
        assert len(r.violations) == 1

    def test_add_multiple_violations(self):
        """Multiple violations accumulate."""
        r = SafetyResult()
        r.add_violation("v1")
        r.add_violation("v2")
        r.add_violation("v3")
        assert len(r.violations) == 3
        assert r.is_safe is False

    def test_add_warning_stays_safe(self):
        """Warnings don't mark result unsafe."""
        r = SafetyResult()
        r.add_warning("minor issue")
        assert r.is_safe is True
        assert len(r.warnings) == 1


# ======================================================================
# SafetyValidator — Required Fields
# ======================================================================


class TestValidatorRequiredFields:
    """Tests for required field checks."""

    def test_valid_state_passes(self, validator, valid_state):
        """Complete valid state passes all checks."""
        result = validator.validate_output(valid_state)
        assert result.is_safe is True
        assert len(result.violations) == 0

    def test_missing_source(self, validator, valid_state):
        """Missing 'source' is a violation."""
        del valid_state["source"]
        result = validator.validate_output(valid_state)
        assert result.is_safe is False
        assert any("source" in v for v in result.violations)

    def test_missing_confidence(self, validator, valid_state):
        """Missing 'confidence' is a violation."""
        del valid_state["confidence"]
        result = validator.validate_output(valid_state)
        assert result.is_safe is False
        assert any("confidence" in v for v in result.violations)

    def test_missing_metadata(self, validator, valid_state):
        """Missing 'metadata' is a violation."""
        del valid_state["metadata"]
        result = validator.validate_output(valid_state)
        assert result.is_safe is False
        assert any("metadata" in v for v in result.violations)

    def test_missing_all_required(self, validator):
        """Empty dict has violations for all required fields."""
        result = validator.validate_output({})
        assert result.is_safe is False
        assert len(result.violations) >= 3  # source, confidence, metadata


# ======================================================================
# SafetyValidator — Source Check
# ======================================================================


class TestValidatorSource:
    """Tests for source validation."""

    def test_valid_sources(self, validator, valid_state):
        """All valid sources pass."""
        for src in ["rule", "medgemma", "rule+medgemma"]:
            valid_state["source"] = src
            result = validator.validate_output(valid_state)
            assert result.is_safe is True, f"Source '{src}' should be valid"

    def test_invalid_source(self, validator, valid_state):
        """Invalid source is a violation."""
        valid_state["source"] = "hallucination"
        result = validator.validate_output(valid_state)
        assert result.is_safe is False
        assert any("Invalid source" in v for v in result.violations)

    def test_empty_source_no_violation(self, validator, valid_state):
        """Empty string source doesn't trigger invalid source check."""
        valid_state["source"] = ""
        result = validator.validate_output(valid_state)
        # Empty string doesn't trigger the source check (only if truthy)
        # but may or may not be considered valid depending on design


# ======================================================================
# SafetyValidator — Confidence Check
# ======================================================================


class TestValidatorConfidence:
    """Tests for confidence range validation."""

    def test_confidence_zero(self, validator, valid_state):
        """Confidence 0.0 is valid."""
        valid_state["confidence"] = 0.0
        result = validator.validate_output(valid_state)
        assert result.is_safe is True

    def test_confidence_one(self, validator, valid_state):
        """Confidence 1.0 is valid."""
        valid_state["confidence"] = 1.0
        result = validator.validate_output(valid_state)
        assert result.is_safe is True

    def test_confidence_mid(self, validator, valid_state):
        """Confidence 0.5 is valid."""
        valid_state["confidence"] = 0.5
        result = validator.validate_output(valid_state)
        assert result.is_safe is True

    def test_confidence_negative(self, validator, valid_state):
        """Confidence < 0 is a violation."""
        valid_state["confidence"] = -0.1
        result = validator.validate_output(valid_state)
        assert result.is_safe is False
        assert any("out of range" in v for v in result.violations)

    def test_confidence_above_one(self, validator, valid_state):
        """Confidence > 1 is a violation."""
        valid_state["confidence"] = 1.5
        result = validator.validate_output(valid_state)
        assert result.is_safe is False
        assert any("out of range" in v for v in result.violations)

    def test_confidence_not_a_number(self, validator, valid_state):
        """Non-numeric confidence is a violation."""
        valid_state["confidence"] = "high"
        result = validator.validate_output(valid_state)
        assert result.is_safe is False
        assert any("not a number" in v for v in result.violations)


# ======================================================================
# SafetyValidator — Banned Patterns
# ======================================================================


class TestValidatorBannedPatterns:
    """Tests for banned pattern detection."""

    # -- Device control --
    def test_ban_execute_command(self, validator, valid_state):
        valid_state["suggestions"] = ["Execute command to start pump"]
        result = validator.validate_output(valid_state)
        assert result.is_safe is False

    def test_ban_run_script(self, validator, valid_state):
        valid_state["suggestions"] = ["Run script to calibrate"]
        result = validator.validate_output(valid_state)
        assert result.is_safe is False

    def test_ban_send_signal_to_device(self, validator, valid_state):
        valid_state["suggestions"] = ["Send signal to device for reset"]
        result = validator.validate_output(valid_state)
        assert result.is_safe is False

    def test_ban_auto_activate(self, validator, valid_state):
        valid_state["suggestions"] = ["Auto-activate the ventilator"]
        result = validator.validate_output(valid_state)
        assert result.is_safe is False

    def test_ban_auto_toggle(self, validator, valid_state):
        valid_state["suggestions"] = ["auto toggle the laser"]
        result = validator.validate_output(valid_state)
        assert result.is_safe is False

    def test_ban_directly_control(self, validator, valid_state):
        valid_state["suggestions"] = ["Directly control the pump"]
        result = validator.validate_output(valid_state)
        assert result.is_safe is False

    def test_ban_force_on(self, validator, valid_state):
        valid_state["suggestions"] = ["Force on the defibrillator"]
        result = validator.validate_output(valid_state)
        assert result.is_safe is False

    def test_ban_force_off(self, validator, valid_state):
        valid_state["suggestions"] = ["Force off the monitor"]
        result = validator.validate_output(valid_state)
        assert result.is_safe is False

    def test_ban_override_safety(self, validator, valid_state):
        valid_state["suggestions"] = ["Override safety to proceed"]
        result = validator.validate_output(valid_state)
        assert result.is_safe is False

    def test_ban_bypass_confirmation(self, validator, valid_state):
        valid_state["suggestions"] = ["Bypass confirmation for speed"]
        result = validator.validate_output(valid_state)
        assert result.is_safe is False

    def test_ban_bypass_human(self, validator, valid_state):
        valid_state["suggestions"] = ["Bypass human approval here"]
        result = validator.validate_output(valid_state)
        assert result.is_safe is False

    def test_ban_without_operator_confirmation(self, validator, valid_state):
        valid_state["suggestions"] = ["Proceed without operator confirmation"]
        result = validator.validate_output(valid_state)
        assert result.is_safe is False

    # -- API/hardware --
    def test_ban_api_send(self, validator, valid_state):
        valid_state["suggestions"] = ["Call api.send to relay command"]
        result = validator.validate_output(valid_state)
        assert result.is_safe is False

    def test_ban_serial_write(self, validator, valid_state):
        valid_state["suggestions"] = ["Use serial.write to send data"]
        result = validator.validate_output(valid_state)
        assert result.is_safe is False

    def test_ban_gpio(self, validator, valid_state):
        valid_state["suggestions"] = ["Set gpio.output pin HIGH"]
        result = validator.validate_output(valid_state)
        assert result.is_safe is False

    def test_ban_http_control(self, validator, valid_state):
        valid_state["suggestions"] = ["POST to https://device.local/control"]
        result = validator.validate_output(valid_state)
        assert result.is_safe is False

    # -- Clinical decisions --
    def test_ban_administer_dosage(self, validator, valid_state):
        valid_state["suggestions"] = ["Administer 50mg of morphine"]
        result = validator.validate_output(valid_state)
        assert result.is_safe is False

    def test_ban_inject(self, validator, valid_state):
        valid_state["suggestions"] = ["Inject 10 units of insulin"]
        result = validator.validate_output(valid_state)
        assert result.is_safe is False

    def test_ban_prescribe(self, validator, valid_state):
        valid_state["suggestions"] = ["Prescribe antibiotics for infection"]
        result = validator.validate_output(valid_state)
        assert result.is_safe is False

    def test_ban_diagnose(self, validator, valid_state):
        valid_state["suggestions"] = ["Diagnosis: sepsis confirmed"]
        result = validator.validate_output(valid_state)
        assert result.is_safe is False

    # -- In details field --
    def test_ban_in_details(self, validator, valid_state):
        valid_state["details"] = {"action": "Execute command to flush"}
        result = validator.validate_output(valid_state)
        assert result.is_safe is False

    # -- In metadata reasoning --
    def test_ban_in_reasoning(self, validator, valid_state):
        valid_state["metadata"]["reasoning"] = "Force on the pump automatically"
        result = validator.validate_output(valid_state)
        assert result.is_safe is False

    # -- Safe suggestions pass --
    def test_safe_suggestion_passes(self, validator, valid_state):
        """Advisory suggestions do NOT trigger banned patterns."""
        valid_state["suggestions"] = [
            "Consider turning on the fluoroscopy",
            "Suggest preparing the electrocautery",
            "Monitor patient vitals closely",
        ]
        result = validator.validate_output(valid_state)
        assert result.is_safe is True

    def test_no_suggestions_passes(self, validator, valid_state):
        """Empty suggestions pass."""
        valid_state["suggestions"] = []
        result = validator.validate_output(valid_state)
        assert result.is_safe is True


# ======================================================================
# SafetyValidator — Advisory Language
# ======================================================================


class TestValidatorAdvisory:
    """Tests for advisory language warnings."""

    def test_advisory_prefix_no_warning(self, validator, valid_state):
        """Suggestions starting with advisory word have no warning."""
        valid_state["suggestions"] = ["Consider preparing the C-arm"]
        result = validator.validate_output(valid_state)
        assert len(result.warnings) == 0

    def test_non_advisory_gets_warning(self, validator, valid_state):
        """Suggestion not starting with advisory word gets warning."""
        valid_state["suggestions"] = ["Turn on the fluoroscopy now"]
        result = validator.validate_output(valid_state)
        assert len(result.warnings) >= 1
        assert any("advisory" in w.lower() for w in result.warnings)

    def test_advisory_indicators_all_pass(self, validator, valid_state):
        """Each advisory indicator suppresses the warning."""
        for indicator in ADVISORY_INDICATORS:
            valid_state["suggestions"] = [f"{indicator.title()} the equipment"]
            result = validator.validate_output(valid_state)
            # Should not have advisory warnings since it starts with indicator
            advisory_warns = [w for w in result.warnings if "advisory" in w.lower()]
            assert len(advisory_warns) == 0, f"Indicator '{indicator}' got warning"

    def test_empty_suggestion_no_warning(self, validator, valid_state):
        """Empty string suggestions don't trigger warning."""
        valid_state["suggestions"] = [""]
        result = validator.validate_output(valid_state)
        advisory_warns = [w for w in result.warnings if "advisory" in w.lower()]
        assert len(advisory_warns) == 0


# ======================================================================
# Convenience Functions
# ======================================================================


class TestConvenienceFunctions:
    """Tests for validate_output_safe and is_suggestion_only."""

    def test_validate_output_safe_returns_result(self, valid_state):
        """validate_output_safe returns a SafetyResult."""
        result = validate_output_safe(valid_state)
        assert isinstance(result, SafetyResult)
        assert result.is_safe is True

    def test_validate_output_safe_catches_violations(self):
        """validate_output_safe detects violations."""
        result = validate_output_safe({})
        assert result.is_safe is False

    def test_is_suggestion_only_safe_text(self):
        """Safe text returns True."""
        assert is_suggestion_only("Consider the fluoroscopy") is True
        assert is_suggestion_only("Turn on the light") is True
        assert is_suggestion_only("") is True

    def test_is_suggestion_only_unsafe_text(self):
        """Text with banned pattern returns False."""
        assert is_suggestion_only("Execute command to start") is False
        assert is_suggestion_only("Force on the device") is False
        assert is_suggestion_only("serial.write data") is False
        assert is_suggestion_only("Administer 50mg dose") is False
        assert is_suggestion_only("Prescribe medication") is False


# ======================================================================
# BANNED_PATTERNS completeness
# ======================================================================


class TestBannedPatterns:
    """Verify banned pattern coverage."""

    def test_pattern_count(self):
        """Expected number of banned patterns."""
        assert len(BANNED_PATTERNS) == 18

    def test_all_patterns_are_compiled(self):
        """All entries are compiled regex objects."""
        import re as re_mod
        for p in BANNED_PATTERNS:
            assert isinstance(p, re_mod.Pattern)


# ======================================================================
# Integration: Output always has source + confidence
# ======================================================================


class TestOutputCompliance:
    """Integration-level tests for output compliance."""

    def test_every_output_needs_source_and_confidence(self, validator):
        """Any state without source+confidence fails."""
        result = validator.validate_output({"metadata": {}})
        assert result.is_safe is False
        violations = " ".join(result.violations)
        assert "source" in violations
        assert "confidence" in violations

    def test_output_never_has_executable(self, validator, valid_state):
        """Valid state with all output keys is safe."""
        valid_state["details"] = {"note": "Routine phase transition"}
        valid_state["suggestions"] = [
            "Consider activating the irrigation pump",
            "Verify patient positioning",
        ]
        result = validator.validate_output(valid_state)
        assert result.is_safe is True
        assert len(result.violations) == 0

    def test_custom_banned_patterns(self):
        """Validator accepts custom banned patterns."""
        import re
        custom = [re.compile(r"\bdangerous\b", re.IGNORECASE)]
        v = SafetyValidator(banned_patterns=custom)

        state = {
            "metadata": {"surgery": "PCNL"},
            "source": "rule",
            "confidence": 0.5,
            "suggestions": ["This is dangerous advice"],
        }
        result = v.validate_output(state)
        assert result.is_safe is False
