"""
OR-Symphony: State Serializer

Normalizes outputs from the rule engine and LLM into the
mandatory JSON output contract format.

Validates against the surgery state JSON schema using jsonschema.

Features:
  - normalize(): Raw dict → JSON contract
  - normalize_rule_output(): RuleEngineResult → JSON contract
  - normalize_llm_output(): Raw LLM response → JSON contract
  - merge(): Combine rule + LLM outputs
  - build_current_state(): Full state with buffer context
  - validate() / validate_schema(): Schema-based validation
  - save_state() / load_state(): Persistence to disk

Usage:
    from src.state.serializer import StateSerializer
    serializer = StateSerializer()
    valid_json = serializer.normalize(raw_output)
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import jsonschema

from src.utils.constants import (
    JSON_OUTPUT_KEYS,
    SCHEMAS_DIR,
    TMP_DIR,
    VALID_SOURCES,
)

logger = logging.getLogger(__name__)

# Path to the JSON schema file
_SCHEMA_PATH = SCHEMAS_DIR / "surgery_state.schema.json"

# Cache loaded schema
_schema_cache: Optional[Dict[str, Any]] = None


def _load_schema() -> Dict[str, Any]:
    """Load and cache the surgery state JSON schema."""
    global _schema_cache
    if _schema_cache is None:
        with open(_SCHEMA_PATH, "r", encoding="utf-8") as f:
            _schema_cache = json.load(f)
        logger.debug("Loaded JSON schema from %s", _SCHEMA_PATH)
    return _schema_cache


class StateSerializer:
    """
    Normalizes and validates surgery state JSON output.

    Ensures all outputs conform to the mandatory JSON contract:
    {
        "metadata": {...},
        "machines": {"0": [...], "1": [...]},
        "details": {...},
        "suggestions": [...],
        "confidence": 0.0-1.0,
        "source": "rule | medgemma | rule+medgemma"
    }
    """

    def __init__(self) -> None:
        """Initialize the serializer and load the schema."""
        self._schema: Optional[Dict[str, Any]] = None
        try:
            self._schema = _load_schema()
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning("Could not load JSON schema: %s — manual validation only", e)

    # ------------------------------------------------------------------
    # Normalize from raw dict
    # ------------------------------------------------------------------

    def normalize(
        self,
        raw: Dict[str, Any],
        surgery: str = "",
        phase: str = "",
    ) -> Dict[str, Any]:
        """
        Normalize a raw output dict into the standard JSON contract.

        Args:
            raw: Raw output from rule engine or LLM.
            surgery: Current surgery name.
            phase: Current surgical phase.

        Returns:
            Normalized dict matching the JSON output contract.

        Raises:
            ValueError: If the output cannot be normalized.
        """
        normalized: Dict[str, Any] = {
            "metadata": dict(raw.get("metadata", {})),
            "machines": {
                "0": list(raw.get("machines", {}).get("0", [])),
                "1": list(raw.get("machines", {}).get("1", [])),
            },
            "details": dict(raw.get("details", {})),
            "suggestions": list(raw.get("suggestions", [])),
            "confidence": self._clamp_confidence(raw.get("confidence", 0.0)),
            "source": self._validate_source(raw.get("source", "rule")),
        }

        # Ensure metadata has required fields
        normalized["metadata"].setdefault("surgery", surgery)
        normalized["metadata"].setdefault("phase", phase)
        normalized["metadata"].setdefault(
            "timestamp", datetime.now(timezone.utc).isoformat()
        )

        return normalized

    # ------------------------------------------------------------------
    # Normalize from RuleEngineResult
    # ------------------------------------------------------------------

    def normalize_rule_output(
        self,
        result: Any,
        surgery: str = "",
        phase: str = "",
        buffer: Any = None,
    ) -> Dict[str, Any]:
        """
        Normalize a RuleEngineResult into the JSON contract.

        Args:
            result: A RuleEngineResult object (from src.state.rules).
            surgery: Current surgery type.
            phase: Current surgical phase.
            buffer: Optional RollingBuffer for context enrichment.

        Returns:
            Normalized state dict.
        """
        # Use the result's built-in to_json_patch()
        raw = result.to_json_patch()

        # Enrich metadata
        raw["metadata"]["surgery"] = surgery
        raw["metadata"]["timestamp"] = datetime.now(timezone.utc).isoformat()

        # Add buffer context if available
        if buffer is not None:
            raw["metadata"]["buffer_entries"] = buffer.entry_count
            raw["metadata"]["buffer_duration_s"] = round(buffer.duration_s, 1)
            raw["metadata"]["session_elapsed_s"] = round(buffer.session_elapsed_s, 1)
            raw["metadata"]["transcript_context"] = buffer.get_context(max_entries=10)

        return self.normalize(raw, surgery=surgery, phase=phase)

    # ------------------------------------------------------------------
    # Normalize from LLM output
    # ------------------------------------------------------------------

    def normalize_llm_output(
        self,
        raw: Dict[str, Any],
        surgery: str = "",
        phase: str = "",
    ) -> Dict[str, Any]:
        """
        Normalize a raw MedGemma LLM response into the JSON contract.

        Handles common LLM output quirks: missing keys, string confidence,
        unstructured machine references, etc.

        Args:
            raw: Raw dict from LLM response parsing.
            surgery: Current surgery type.
            phase: Current surgical phase.

        Returns:
            Normalized state dict.
        """
        # LLM outputs may have different key structures
        metadata = dict(raw.get("metadata", {}))
        metadata.setdefault("surgery", surgery)
        metadata.setdefault("phase", phase)
        metadata.setdefault("timestamp", datetime.now(timezone.utc).isoformat())

        # Extract reasoning
        if "reasoning" in raw and "reasoning" not in metadata:
            metadata["reasoning"] = raw["reasoning"]
        if "next_phase" in raw and "next_phase" not in metadata:
            metadata["next_phase"] = raw["next_phase"]

        # Build machines (LLM may provide them differently)
        machines_raw = raw.get("machines", {"0": [], "1": []})
        if not isinstance(machines_raw, dict):
            machines_raw = {"0": [], "1": []}

        normalized = {
            "metadata": metadata,
            "machines": {
                "0": list(machines_raw.get("0", [])),
                "1": list(machines_raw.get("1", [])),
            },
            "details": dict(raw.get("details", {})),
            "suggestions": list(raw.get("suggestions", [])),
            "confidence": self._clamp_confidence(raw.get("confidence", 0.0)),
            "source": "medgemma",
        }

        return normalized

    # ------------------------------------------------------------------
    # Build full current state
    # ------------------------------------------------------------------

    def build_current_state(
        self,
        rule_result: Any = None,
        llm_result: Optional[Dict[str, Any]] = None,
        buffer: Any = None,
        surgery: str = "",
        phase: str = "",
    ) -> Dict[str, Any]:
        """
        Build a complete state update from rule engine and/or LLM results.

        Args:
            rule_result: RuleEngineResult from the rule engine (or None).
            llm_result: Normalized LLM output dict (or None).
            buffer: RollingBuffer for context enrichment.
            surgery: Current surgery type.
            phase: Current surgical phase.

        Returns:
            Complete, validated state dict.
        """
        ts_start = time.time()

        # Normalize rule output
        if rule_result is not None:
            rule_state = self.normalize_rule_output(
                rule_result, surgery=surgery, phase=phase, buffer=buffer
            )
        else:
            rule_state = self._empty_state(surgery=surgery, phase=phase)

        # Merge with LLM output
        if llm_result is not None:
            state = self.merge(rule_state, llm_result)
        else:
            state = rule_state

        # Add buffer context to metadata
        if buffer is not None:
            state["metadata"]["buffer_entries"] = buffer.entry_count
            state["metadata"]["buffer_duration_s"] = round(buffer.duration_s, 1)
            state["metadata"]["session_elapsed_s"] = round(buffer.session_elapsed_s, 1)

        # Record total processing time
        elapsed_ms = (time.time() - ts_start) * 1000
        state["metadata"]["processing_time_ms"] = round(
            state["metadata"].get("processing_time_ms", 0.0) + elapsed_ms, 1
        )

        return state

    # ------------------------------------------------------------------
    # Merge rule + LLM outputs
    # ------------------------------------------------------------------

    def merge(
        self,
        rule_output: Dict[str, Any],
        llm_output: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Merge rule engine and LLM outputs into a single state update.

        Rule engine output takes priority for machine toggles.
        LLM output adds suggestions and phase reasoning.

        Args:
            rule_output: Output from the rule engine.
            llm_output: Optional output from MedGemma.

        Returns:
            Merged output dict.
        """
        if llm_output is None:
            return rule_output

        merged = dict(rule_output)
        merged["metadata"] = dict(merged.get("metadata", {}))
        merged["source"] = "rule+medgemma"

        # Merge suggestions from LLM
        llm_suggestions = llm_output.get("suggestions", [])
        merged["suggestions"] = list(set(merged.get("suggestions", []) + llm_suggestions))

        # Use LLM confidence if higher
        llm_confidence = llm_output.get("confidence", 0.0)
        if llm_confidence > merged.get("confidence", 0.0):
            merged["confidence"] = llm_confidence

        # Merge metadata (LLM may add phase info)
        llm_meta = llm_output.get("metadata", {})
        for key in ["phase", "reasoning", "next_phase"]:
            if key in llm_meta:
                merged["metadata"][key] = llm_meta[key]

        return merged

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, state: Dict[str, Any]) -> bool:
        """
        Validate that a state dict matches the JSON output contract.

        Uses jsonschema validation if schema is available,
        falls back to manual key checks otherwise.

        Args:
            state: State dictionary to validate.

        Returns:
            True if valid, False otherwise.
        """
        # Try jsonschema validation first
        if self._schema is not None:
            return self.validate_schema(state)

        # Fallback: manual validation
        return self._validate_manual(state)

    def validate_schema(self, state: Dict[str, Any]) -> bool:
        """
        Validate state against the JSON schema using jsonschema.

        Args:
            state: State dictionary to validate.

        Returns:
            True if valid, False otherwise.
        """
        if self._schema is None:
            logger.warning("No schema loaded — falling back to manual validation")
            return self._validate_manual(state)

        try:
            jsonschema.validate(instance=state, schema=self._schema)
            return True
        except jsonschema.ValidationError as e:
            logger.error("Schema validation failed: %s", e.message)
            return False
        except jsonschema.SchemaError as e:
            logger.error("Schema itself is invalid: %s", e.message)
            return False

    def get_validation_errors(self, state: Dict[str, Any]) -> List[str]:
        """
        Get a list of all validation errors for a state dict.

        Args:
            state: State dictionary to validate.

        Returns:
            List of error message strings. Empty if valid.
        """
        if self._schema is None:
            return ["No schema loaded"]

        validator = jsonschema.Draft7Validator(self._schema)
        return [e.message for e in validator.iter_errors(state)]

    def _validate_manual(self, state: Dict[str, Any]) -> bool:
        """Manual (fallback) validation without jsonschema."""
        # Check required keys
        for key in JSON_OUTPUT_KEYS:
            if key not in state:
                logger.error("Missing required key: %s", key)
                return False

        # Check machines format
        machines = state.get("machines", {})
        if not isinstance(machines, dict):
            logger.error("'machines' must be a dict")
            return False
        if "0" not in machines or "1" not in machines:
            logger.error("'machines' must have '0' and '1' keys")
            return False

        # Check confidence range
        confidence = state.get("confidence", -1)
        if not (0.0 <= confidence <= 1.0):
            logger.error("'confidence' must be between 0.0 and 1.0, got %s", confidence)
            return False

        # Check source
        source = state.get("source", "")
        if source not in VALID_SOURCES:
            logger.error("Invalid source: '%s'. Must be one of %s", source, VALID_SOURCES)
            return False

        return True

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_state(
        self, state: Dict[str, Any], path: Optional[Path] = None
    ) -> Path:
        """
        Save a state dict to a JSON file.

        Args:
            state: State dictionary to save.
            path: File path. Defaults to tmp/current_state.json.

        Returns:
            Path where the state was saved.
        """
        if path is None:
            path = TMP_DIR / "current_state.json"

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, default=str)

        logger.info("State saved to %s", path)
        return path

    def load_state(self, path: Optional[Path] = None) -> Optional[Dict[str, Any]]:
        """
        Load a state dict from a JSON file.

        Args:
            path: File path. Defaults to tmp/current_state.json.

        Returns:
            Loaded state dict, or None if file not found.
        """
        if path is None:
            path = TMP_DIR / "current_state.json"

        if not path.exists():
            logger.warning("State file not found: %s", path)
            return None

        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)

        logger.info("State loaded from %s", path)
        return state

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _empty_state(self, surgery: str = "", phase: str = "") -> Dict[str, Any]:
        """Create an empty valid state dict."""
        return {
            "metadata": {
                "surgery": surgery,
                "phase": phase,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "processing_time_ms": 0.0,
                "matched_keywords": [],
                "temporal": "none",
            },
            "machines": {"0": [], "1": []},
            "details": {"toggles": [], "negations": [], "debounced": []},
            "suggestions": [],
            "confidence": 0.0,
            "source": "rule",
        }

    @staticmethod
    def _clamp_confidence(value: Any) -> float:
        """Clamp confidence to [0.0, 1.0]."""
        try:
            v = float(value)
            return max(0.0, min(1.0, v))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _validate_source(source: Any) -> str:
        """Validate and return a valid source string."""
        if isinstance(source, str) and source in VALID_SOURCES:
            return source
        return "rule"


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    serializer = StateSerializer()

    # Test normalization
    raw = {
        "machines": {"0": ["M03"], "1": ["M01", "M09"]},
        "confidence": 0.85,
        "source": "rule",
    }
    normalized = serializer.normalize(raw, surgery="PCNL", phase="Phase2")
    print("Normalized:", json.dumps(normalized, indent=2))
    print("Valid:", serializer.validate(normalized))

    # Test empty state
    empty = serializer._empty_state(surgery="PCNL", phase="Phase1")
    print("\nEmpty state valid:", serializer.validate(empty))

    # Test schema validation errors
    bad = {"metadata": {}, "machines": {"0": [], "1": []}}
    errors = serializer.get_validation_errors(bad)
    print(f"\nValidation errors ({len(errors)}):")
    for err in errors:
        print(f"  - {err}")
