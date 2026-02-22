"""
OR-Symphony: Safety Validation Module

Validates that system outputs are safe — no executable instructions,
mandatory fields present, suggestions are advisory only.

Usage:
    from src.utils.safety import SafetyValidator
    validator = SafetyValidator()
    result = validator.validate_output(state_dict)
    assert result.is_safe
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from src.utils.constants import JSON_OUTPUT_KEYS, VALID_SOURCES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Banned patterns — outputs must NOT contain executable-style language
# ---------------------------------------------------------------------------

BANNED_PATTERNS: List[re.Pattern] = [
    # Direct device control commands
    re.compile(r"\bexecute\s+(command|action|instruction)\b", re.IGNORECASE),
    re.compile(r"\brun\s+(command|script|program)\b", re.IGNORECASE),
    re.compile(r"\bsend\s+(signal|command)\s+to\s+(device|machine|equipment)\b", re.IGNORECASE),
    re.compile(r"\bauto[- ]?activate\b", re.IGNORECASE),
    re.compile(r"\bauto[- ]?toggle\b", re.IGNORECASE),
    re.compile(r"\bdirectly\s+control\b", re.IGNORECASE),
    re.compile(r"\bforce\s+(on|off|shutdown|restart)\b", re.IGNORECASE),
    re.compile(r"\boverride\s+safety\b", re.IGNORECASE),
    re.compile(r"\bbypass\s+(safety|confirmation|human)\b", re.IGNORECASE),
    re.compile(r"\bwithout\s+(human|operator)\s+(confirmation|approval)\b", re.IGNORECASE),
    # API/hardware calls
    re.compile(r"\bapi\s*\.\s*(send|post|put|delete)\b", re.IGNORECASE),
    re.compile(r"\bhttp[s]?://\S+/control\b", re.IGNORECASE),
    re.compile(r"\bserial\.write\b", re.IGNORECASE),
    re.compile(r"\bgpio\.\w+\b", re.IGNORECASE),
    # Clinical decision language
    re.compile(r"\badminister\s+\d+\s*(mg|ml|cc|units)\b", re.IGNORECASE),
    re.compile(r"\binject\s+\d+\b", re.IGNORECASE),
    re.compile(r"\bprescribe\b", re.IGNORECASE),
    re.compile(r"\bdiagnos(e|is)\s*:\s*\w+", re.IGNORECASE),
]

# Suggestion-safe prefixes — suggestions should be advisory
ADVISORY_INDICATORS = [
    "consider", "suggest", "recommend", "may want to", "might need",
    "could", "should consider", "check", "verify", "ensure",
    "prepare", "confirm", "review", "monitor", "note",
]


@dataclass
class SafetyResult:
    """Result of a safety validation check."""
    is_safe: bool = True
    violations: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def add_violation(self, msg: str) -> None:
        self.violations.append(msg)
        self.is_safe = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


class SafetyValidator:
    """
    Validates system outputs for safety compliance.

    Checks:
      1. Required fields present (source, confidence)
      2. Source is a valid value
      3. Confidence is in [0.0, 1.0]
      4. No banned patterns in suggestions or details
      5. Suggestions are advisory (not executable)
    """

    def __init__(self, banned_patterns: Optional[List[re.Pattern]] = None) -> None:
        self.banned_patterns = banned_patterns or BANNED_PATTERNS

    def validate_output(self, state: Dict[str, Any]) -> SafetyResult:
        """
        Run all safety checks on a state output dict.

        Args:
            state: State dictionary to validate.

        Returns:
            SafetyResult with is_safe flag and any violations.
        """
        result = SafetyResult()

        self._check_required_fields(state, result)
        self._check_source(state, result)
        self._check_confidence(state, result)
        self._check_banned_patterns(state, result)
        self._check_suggestions_advisory(state, result)

        if not result.is_safe:
            logger.warning(
                "Safety validation FAILED — %d violations: %s",
                len(result.violations),
                "; ".join(result.violations),
            )

        return result

    def _check_required_fields(self, state: Dict[str, Any], result: SafetyResult) -> None:
        """Check that source and confidence are always present."""
        if "source" not in state:
            result.add_violation("Missing required field: 'source'")
        if "confidence" not in state:
            result.add_violation("Missing required field: 'confidence'")
        if "metadata" not in state:
            result.add_violation("Missing required field: 'metadata'")

    def _check_source(self, state: Dict[str, Any], result: SafetyResult) -> None:
        """Check that source is a valid value."""
        source = state.get("source", "")
        if source and source not in VALID_SOURCES:
            result.add_violation(
                f"Invalid source '{source}' — must be one of {VALID_SOURCES}"
            )

    def _check_confidence(self, state: Dict[str, Any], result: SafetyResult) -> None:
        """Check that confidence is in [0.0, 1.0]."""
        confidence = state.get("confidence")
        if confidence is not None:
            try:
                conf_val = float(confidence)
                if conf_val < 0.0 or conf_val > 1.0:
                    result.add_violation(
                        f"Confidence {conf_val} out of range [0.0, 1.0]"
                    )
            except (TypeError, ValueError):
                result.add_violation(f"Confidence is not a number: {confidence}")

    def _check_banned_patterns(self, state: Dict[str, Any], result: SafetyResult) -> None:
        """Check that no banned patterns appear in suggestions or details."""
        # Check suggestions
        for suggestion in state.get("suggestions", []):
            if not isinstance(suggestion, str):
                continue
            for pattern in self.banned_patterns:
                match = pattern.search(suggestion)
                if match:
                    result.add_violation(
                        f"Banned pattern in suggestion: '{match.group()}' in '{suggestion[:80]}'"
                    )

        # Check details values
        details = state.get("details", {})
        if isinstance(details, dict):
            for key, value in details.items():
                if not isinstance(value, str):
                    continue
                for pattern in self.banned_patterns:
                    match = pattern.search(value)
                    if match:
                        result.add_violation(
                            f"Banned pattern in details[{key}]: '{match.group()}'"
                        )

        # Check metadata reasoning
        reasoning = state.get("metadata", {}).get("reasoning", "")
        if isinstance(reasoning, str):
            for pattern in self.banned_patterns:
                match = pattern.search(reasoning)
                if match:
                    result.add_violation(
                        f"Banned pattern in reasoning: '{match.group()}'"
                    )

    def _check_suggestions_advisory(self, state: Dict[str, Any], result: SafetyResult) -> None:
        """Warn if suggestions don't follow advisory language patterns."""
        for suggestion in state.get("suggestions", []):
            if not isinstance(suggestion, str) or not suggestion.strip():
                continue

            lower = suggestion.lower().strip()
            is_advisory = any(lower.startswith(ind) for ind in ADVISORY_INDICATORS)

            if not is_advisory:
                # Not a violation, just a warning — some valid suggestions
                # may not start with advisory words
                result.add_warning(
                    f"Suggestion may not be advisory: '{suggestion[:60]}'"
                )


def validate_output_safe(state: Dict[str, Any]) -> SafetyResult:
    """
    Convenience function to validate a state dict.

    Args:
        state: State dictionary to validate.

    Returns:
        SafetyResult.
    """
    return SafetyValidator().validate_output(state)


def is_suggestion_only(text: str) -> bool:
    """
    Check if a text looks like a suggestion (not an executable command).

    Args:
        text: Text to check.

    Returns:
        True if the text appears to be a suggestion.
    """
    for pattern in BANNED_PATTERNS:
        if pattern.search(text):
            return False
    return True
