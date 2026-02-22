"""
OR-Symphony: State Serializer

Normalizes outputs from the rule engine and LLM into the
mandatory JSON output contract format.

Validates against the surgery state JSON schema.

Usage:
    from src.state.serializer import StateSerializer
    serializer = StateSerializer()
    valid_json = serializer.normalize(raw_output)
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from src.utils.constants import JSON_OUTPUT_KEYS, VALID_SOURCES

logger = logging.getLogger(__name__)


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
            "metadata": raw.get("metadata", {}),
            "machines": {
                "0": raw.get("machines", {}).get("0", []),
                "1": raw.get("machines", {}).get("1", []),
            },
            "details": raw.get("details", {}),
            "suggestions": raw.get("suggestions", []),
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

    def validate(self, state: Dict[str, Any]) -> bool:
        """
        Validate that a state dict matches the JSON output contract.

        Args:
            state: State dictionary to validate.

        Returns:
            True if valid, False otherwise.
        """
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
