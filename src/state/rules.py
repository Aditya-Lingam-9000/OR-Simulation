"""
OR-Symphony: Deterministic Rule Engine

Fast-path rule engine that maps transcripts to machine state changes.
Uses keyword matching, alias normalization, negation handling,
and debounce logic.

Latency target: < 500ms.

Usage:
    from src.state.rules import RuleEngine
    engine = RuleEngine(surgery="PCNL")
    result = engine.process("start the ventilator")
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from src.utils.constants import (
    RULE_DEBOUNCE_SECONDS,
    RULE_ENGINE_LATENCY_TARGET_MS,
    SURGERIES_MACHINES_PATH,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class MachineToggle:
    """A single machine state change."""

    machine_id: str
    machine_name: str
    action: str  # "ON", "OFF", "STANDBY"
    trigger_text: str = ""
    confidence: float = 1.0


@dataclass
class RuleEngineResult:
    """Result from the rule engine processing."""

    toggles: List[MachineToggle] = field(default_factory=list)
    processing_time_ms: float = 0.0
    source: str = "rule"
    matched_keywords: List[str] = field(default_factory=list)
    negations_detected: List[str] = field(default_factory=list)
    debounced: List[str] = field(default_factory=list)

    def to_json_patch(self) -> Dict[str, Any]:
        """
        Convert to the JSON output contract format.

        Returns:
            Dict matching the mandatory JSON output contract.
        """
        machines_on: List[str] = []
        machines_off: List[str] = []

        for toggle in self.toggles:
            if toggle.action == "ON":
                machines_on.append(toggle.machine_id)
            elif toggle.action in ("OFF", "STANDBY"):
                machines_off.append(toggle.machine_id)

        return {
            "metadata": {
                "processing_time_ms": self.processing_time_ms,
                "matched_keywords": self.matched_keywords,
            },
            "machines": {
                "0": machines_off,
                "1": machines_on,
            },
            "details": {
                "toggles": [
                    {
                        "machine_id": t.machine_id,
                        "name": t.machine_name,
                        "action": t.action,
                        "trigger": t.trigger_text,
                    }
                    for t in self.toggles
                ],
                "negations": self.negations_detected,
                "debounced": self.debounced,
            },
            "suggestions": [],
            "confidence": min((t.confidence for t in self.toggles), default=0.0),
            "source": self.source,
        }


# ---------------------------------------------------------------------------
# Rule Engine
# ---------------------------------------------------------------------------

# Activation keywords
ACTIVATE_KEYWORDS = [
    r"\bturn\s+on\b",
    r"\bstart\b",
    r"\bactivate\b",
    r"\benable\b",
    r"\bpower\s+on\b",
    r"\bswitch\s+on\b",
    r"\bbegin\b",
    r"\binitiate\b",
]

# Deactivation keywords
DEACTIVATE_KEYWORDS = [
    r"\bturn\s+off\b",
    r"\bstop\b",
    r"\bdeactivate\b",
    r"\bdisable\b",
    r"\bpower\s+off\b",
    r"\bswitch\s+off\b",
    r"\bshut\s*(down|off)\b",
    r"\bstandby\b",
]

# Negation patterns
NEGATION_PATTERNS = [
    r"\bdon'?t\b",
    r"\bdo\s+not\b",
    r"\bnot\b",
    r"\bnever\b",
    r"\bhold\s+off\b",
    r"\bwait\b",
    r"\bcancel\b",
]


class RuleEngine:
    """
    Deterministic rule engine for mapping transcripts to machine toggles.

    Features:
    - Keyword/regex matching for machine triggers
    - Alias normalization
    - Negation detection and handling
    - Debounce for repeated commands
    """

    def __init__(self, surgery: str = "PCNL") -> None:
        """
        Initialize the rule engine for a specific surgery.

        Args:
            surgery: Surgery type name.
        """
        self.surgery = surgery
        self.machines: Dict[str, Dict[str, Any]] = {}
        self._trigger_map: Dict[str, Tuple[str, str]] = {}  # trigger -> (machine_id, machine_name)
        self._last_toggle_time: Dict[str, float] = {}  # machine_id -> last toggle timestamp
        self._debounce_seconds = RULE_DEBOUNCE_SECONDS

        self._load_machines()

    def _load_machines(self) -> None:
        """Load machines dictionary for the current surgery."""
        try:
            if not SURGERIES_MACHINES_PATH.exists():
                logger.warning("Machines data file not found: %s", SURGERIES_MACHINES_PATH)
                return

            with open(SURGERIES_MACHINES_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)

            for entry in data:
                if entry.get("surgery") == self.surgery:
                    self.machines = entry.get("machines", {})
                    break

            # Build trigger map
            for machine_id, machine_data in self.machines.items():
                name = machine_data.get("name", "")
                triggers = machine_data.get("triggers", [])
                for trigger in triggers:
                    self._trigger_map[trigger.lower()] = (machine_id, name)
                # Also add the machine name itself as a trigger
                self._trigger_map[name.lower()] = (machine_id, name)

            logger.info(
                "RuleEngine loaded %d machines, %d triggers for '%s'",
                len(self.machines),
                len(self._trigger_map),
                self.surgery,
            )
        except Exception as e:
            logger.error("Failed to load machines for '%s': %s", self.surgery, e)

    def process(self, transcript: str) -> RuleEngineResult:
        """
        Process a transcript and return machine toggles.

        Args:
            transcript: Raw transcript text from ASR.

        Returns:
            RuleEngineResult with matched toggles.
        """
        start_time = time.perf_counter()
        result = RuleEngineResult()
        text = transcript.lower().strip()

        if not text:
            result.processing_time_ms = (time.perf_counter() - start_time) * 1000
            return result

        # Step 1: Detect negations
        has_negation = self._detect_negation(text)
        if has_negation:
            result.negations_detected.append(text)

        # Step 2: Determine intent (activate or deactivate)
        intent = self._detect_intent(text, has_negation)

        # Step 3: Find matching machines
        matched_machines = self._find_machines(text)

        # Step 4: Create toggles with debounce check
        now = time.time()
        for machine_id, machine_name, trigger_text in matched_machines:
            # Debounce check
            last_time = self._last_toggle_time.get(machine_id, 0.0)
            if now - last_time < self._debounce_seconds:
                result.debounced.append(machine_id)
                logger.debug("Debounced toggle for %s (%.1fs since last)", machine_id, now - last_time)
                continue

            action = intent
            toggle = MachineToggle(
                machine_id=machine_id,
                machine_name=machine_name,
                action=action,
                trigger_text=trigger_text,
                confidence=0.9 if not has_negation else 0.7,
            )
            result.toggles.append(toggle)
            result.matched_keywords.append(trigger_text)
            self._last_toggle_time[machine_id] = now

        result.processing_time_ms = (time.perf_counter() - start_time) * 1000

        if result.processing_time_ms > RULE_ENGINE_LATENCY_TARGET_MS:
            logger.warning(
                "Rule engine exceeded latency target: %.1fms > %dms",
                result.processing_time_ms,
                RULE_ENGINE_LATENCY_TARGET_MS,
            )

        return result

    def _detect_negation(self, text: str) -> bool:
        """Check if the text contains negation patterns."""
        for pattern in NEGATION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return True
        return False

    def _detect_intent(self, text: str, has_negation: bool) -> str:
        """
        Determine the intent: ON or OFF.

        Negation flips the intent:
        - "start ventilator" → ON
        - "don't start ventilator" → OFF (negation flips ON → OFF)
        - "stop ventilator" → OFF
        - "don't stop ventilator" → ON (negation flips OFF → ON)
        """
        is_activate = any(re.search(p, text) for p in ACTIVATE_KEYWORDS)
        is_deactivate = any(re.search(p, text) for p in DEACTIVATE_KEYWORDS)

        if is_activate and not is_deactivate:
            return "OFF" if has_negation else "ON"
        elif is_deactivate and not is_activate:
            return "ON" if has_negation else "OFF"
        elif is_activate and is_deactivate:
            # Ambiguous — default to the last keyword found
            return "OFF" if has_negation else "ON"
        else:
            # No explicit intent keyword — default to ON (trigger phrase implies activation)
            return "OFF" if has_negation else "ON"

    def _find_machines(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Find machines mentioned in the text.

        Returns:
            List of (machine_id, machine_name, matched_trigger) tuples.
        """
        matches: List[Tuple[str, str, str]] = []
        seen_ids: Set[str] = set()

        for trigger, (machine_id, machine_name) in self._trigger_map.items():
            if trigger in text and machine_id not in seen_ids:
                matches.append((machine_id, machine_name, trigger))
                seen_ids.add(machine_id)

        return matches

    def switch_surgery(self, surgery: str) -> None:
        """Switch to a different surgery type and reload machines."""
        self.surgery = surgery
        self.machines = {}
        self._trigger_map = {}
        self._last_toggle_time = {}
        self._load_machines()

    def get_machines_summary(self) -> Dict[str, str]:
        """Get a summary of loaded machines."""
        return {
            mid: mdata.get("name", "Unknown")
            for mid, mdata in self.machines.items()
        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    engine = RuleEngine(surgery="PCNL")
    print(f"Loaded machines: {engine.get_machines_summary()}")

    test_phrases = [
        "start the ventilator",
        "turn on the suction",
        "don't start the irrigation pump",
        "stop the lithotripter",
        "camera on",
        "fluoro on",
    ]

    for phrase in test_phrases:
        result = engine.process(phrase)
        patch = result.to_json_patch()
        print(f"\n  Input: '{phrase}'")
        print(f"  ON:  {patch['machines']['1']}")
        print(f"  OFF: {patch['machines']['0']}")
        print(f"  Time: {result.processing_time_ms:.2f}ms")
