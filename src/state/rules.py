"""
OR-Symphony: Deterministic Rule Engine

Production-quality rule engine that maps transcripts to machine state changes.
Features:
  - Regex-based machine matching with word boundaries
  - Rich alias expansion (multiple names per machine)
  - Negation detection and intent flipping
  - Temporal qualifier detection (immediately, after X, when ready)
  - Phase-aware filtering (only match machines valid for current phase)
  - Multi-machine commands ("turn on suction and irrigation")
  - Standby action support
  - Debounce logic for repeated commands
  - Graded confidence scoring

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
# Config path for per-surgery machine files
# ---------------------------------------------------------------------------

CONFIGS_MACHINES_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "machines"

# Surgery name → config filename mapping
_SURGERY_CONFIG_MAP: Dict[str, str] = {
    "PCNL": "pcnl.json",
    "Partial Hepatectomy": "partial_hepatectomy.json",
    "Lobectomy": "lobectomy.json",
}


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
    match_type: str = "trigger"  # "trigger", "alias", "name"


@dataclass
class TemporalQualifier:
    """Temporal context extracted from the transcript."""

    qualifier: str  # "immediately", "after", "when", "in_time", "none"
    raw_text: str = ""
    delay_hint: str = ""  # e.g., "after the incision", "in 5 minutes"


@dataclass
class RuleEngineResult:
    """Result from the rule engine processing."""

    toggles: List[MachineToggle] = field(default_factory=list)
    processing_time_ms: float = 0.0
    source: str = "rule"
    matched_keywords: List[str] = field(default_factory=list)
    negations_detected: List[str] = field(default_factory=list)
    debounced: List[str] = field(default_factory=list)
    temporal: Optional[TemporalQualifier] = None
    current_phase: str = ""

    def to_json_patch(self) -> Dict[str, Any]:
        """
        Convert to the JSON output contract format.

        Returns:
            Dict matching the mandatory JSON output contract.
        """
        machines_on: List[str] = []
        machines_off: List[str] = []
        machines_standby: List[str] = []

        for toggle in self.toggles:
            if toggle.action == "ON":
                machines_on.append(toggle.machine_id)
            elif toggle.action == "STANDBY":
                machines_standby.append(toggle.machine_id)
            else:
                machines_off.append(toggle.machine_id)

        return {
            "metadata": {
                "processing_time_ms": self.processing_time_ms,
                "matched_keywords": self.matched_keywords,
                "phase": self.current_phase,
                "temporal": self.temporal.qualifier if self.temporal else "none",
            },
            "machines": {
                "0": machines_off + machines_standby,
                "1": machines_on,
            },
            "details": {
                "toggles": [
                    {
                        "machine_id": t.machine_id,
                        "name": t.machine_name,
                        "action": t.action,
                        "trigger": t.trigger_text,
                        "confidence": t.confidence,
                        "match_type": t.match_type,
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
# Keyword patterns
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
    r"\bring\s+up\b",
    r"\bfire\s+up\b",
    r"\bplease\s+start\b",
    r"\bgo\s+ahead\s+(with|and)\b",
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
    r"\bend\b",
    r"\bkill\b",
    r"\bcut\b(?!\s+(?:the\s+)?tissue)",  # "cut" but not "cut the tissue"
]

# Standby keywords
STANDBY_KEYWORDS = [
    r"\bstandby\b",
    r"\bstand\s+by\b",
    r"\bpause\b",
    r"\bhold\b(?!\s+off)",
    r"\bidle\b",
    r"\bput\s+on\s+standby\b",
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
    r"\bno\s+need\b",
    r"\bskip\b",
    r"\bforget\b",
    r"\bleave\s+(it\s+)?off\b",
]

# Temporal qualifier patterns
TEMPORAL_PATTERNS = [
    (r"\bimmediately\b|\bright\s+now\b|\bstat\b|\bnow\b", "immediately"),
    (r"\bafter\s+(?:the\s+)?(\w+(?:\s+\w+)?)", "after"),
    (r"\bwhen\s+(?:the\s+)?(\w+(?:\s+\w+)?)", "when"),
    (r"\bin\s+(\d+)\s*(?:min(?:ute)?s?|sec(?:ond)?s?)\b", "in_time"),
    (r"\bbefore\s+(?:the\s+)?(\w+(?:\s+\w+)?)", "before"),
    (r"\bduring\s+(?:the\s+)?(\w+(?:\s+\w+)?)", "during"),
]

# Conjunction patterns for multi-machine commands
CONJUNCTION_PATTERN = re.compile(r"\b(?:and|plus|also|as\s+well\s+as)\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# Rule Engine
# ---------------------------------------------------------------------------


class RuleEngine:
    """
    Deterministic rule engine for mapping transcripts to machine toggles.

    Features:
    - Keyword/regex matching for machine triggers
    - Rich alias expansion
    - Negation detection and handling
    - Temporal qualifier extraction
    - Phase-aware filtering
    - Multi-machine command support
    - Debounce for repeated commands
    - Graded confidence scoring
    """

    def __init__(
        self,
        surgery: str = "PCNL",
        current_phase: str = "",
        phase_filter: bool = False,
    ) -> None:
        """
        Initialize the rule engine for a specific surgery.

        Args:
            surgery: Surgery type name.
            current_phase: Current surgical phase (e.g. "Phase3").
            phase_filter: If True, only match machines valid for current_phase.
        """
        self.surgery = surgery
        self.current_phase = current_phase
        self.phase_filter = phase_filter
        self.machines: Dict[str, Dict[str, Any]] = {}
        self.phases: List[Dict[str, str]] = []
        self._trigger_map: Dict[str, Tuple[str, str, str]] = {}
        # trigger → (machine_id, machine_name, match_type)
        self._trigger_patterns: List[Tuple[re.Pattern, str, str, str]] = []
        # (compiled_regex, machine_id, machine_name, match_type)
        self._last_toggle_time: Dict[str, float] = {}
        self._debounce_seconds = RULE_DEBOUNCE_SECONDS

        self._load_machines()

    def _load_machines(self) -> None:
        """Load machines from per-surgery config file, fallback to legacy data."""
        loaded = False

        # Try per-surgery config first
        config_file = _SURGERY_CONFIG_MAP.get(self.surgery)
        if config_file:
            config_path = CONFIGS_MACHINES_DIR / config_file
            if config_path.exists():
                loaded = self._load_from_config(config_path)

        # Fallback to legacy surgeries_machines.json
        if not loaded:
            self._load_from_legacy()

        # Build regex trigger patterns
        self._build_trigger_patterns()

        logger.info(
            "RuleEngine loaded %d machines, %d trigger patterns for '%s'",
            len(self.machines),
            len(self._trigger_patterns),
            self.surgery,
        )

    def _load_from_config(self, config_path: Path) -> bool:
        """Load from per-surgery config file (configs/machines/*.json)."""
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            self.machines = data.get("machines", {})
            self.phases = data.get("phases", [])
            return True
        except Exception as e:
            logger.error("Failed to load config %s: %s", config_path, e)
            return False

    def _load_from_legacy(self) -> None:
        """Load from legacy data/surgeries_machines.json."""
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
        except Exception as e:
            logger.error("Failed to load machines for '%s': %s", self.surgery, e)

    def _build_trigger_patterns(self) -> None:
        """Build compiled regex patterns for all machine triggers and aliases."""
        self._trigger_patterns = []
        self._trigger_map = {}

        for machine_id, machine_data in self.machines.items():
            name = machine_data.get("name", "")
            triggers = machine_data.get("triggers", [])
            aliases = machine_data.get("aliases", [])

            # Add explicit triggers (highest confidence)
            for trigger in triggers:
                pattern = re.compile(
                    r"\b" + re.escape(trigger.lower()) + r"\b", re.IGNORECASE
                )
                self._trigger_patterns.append((pattern, machine_id, name, "trigger"))
                self._trigger_map[trigger.lower()] = (machine_id, name, "trigger")

            # Add aliases (medium confidence)
            for alias in aliases:
                pattern = re.compile(
                    r"\b" + re.escape(alias.lower()) + r"\b", re.IGNORECASE
                )
                self._trigger_patterns.append((pattern, machine_id, name, "alias"))
                self._trigger_map[alias.lower()] = (machine_id, name, "alias")

            # Add machine name itself (medium confidence)
            pattern = re.compile(
                r"\b" + re.escape(name.lower()) + r"\b", re.IGNORECASE
            )
            self._trigger_patterns.append((pattern, machine_id, name, "name"))
            self._trigger_map[name.lower()] = (machine_id, name, "name")

    def process(
        self,
        transcript: str,
        current_phase: Optional[str] = None,
    ) -> RuleEngineResult:
        """
        Process a transcript and return machine toggles.

        Args:
            transcript: Raw transcript text from ASR.
            current_phase: Override current phase for this call.

        Returns:
            RuleEngineResult with matched toggles.
        """
        start_time = time.perf_counter()
        phase = current_phase or self.current_phase
        result = RuleEngineResult(current_phase=phase)
        text = transcript.lower().strip()

        if not text:
            result.processing_time_ms = (time.perf_counter() - start_time) * 1000
            return result

        # Step 1: Detect negations
        has_negation = self._detect_negation(text)
        if has_negation:
            result.negations_detected.append(text)

        # Step 2: Detect temporal qualifiers
        result.temporal = self._detect_temporal(text)

        # Step 3: Determine intent (activate, deactivate, standby)
        intent = self._detect_intent(text, has_negation)

        # Step 4: Find matching machines (regex-based)
        matched_machines = self._find_machines(text, phase)

        # Step 5: Create toggles with debounce check
        now = time.time()
        for machine_id, machine_name, trigger_text, match_type in matched_machines:
            # Debounce check
            last_time = self._last_toggle_time.get(machine_id, 0.0)
            if now - last_time < self._debounce_seconds:
                result.debounced.append(machine_id)
                logger.debug(
                    "Debounced toggle for %s (%.1fs since last)",
                    machine_id,
                    now - last_time,
                )
                continue

            # Confidence scoring
            confidence = self._compute_confidence(
                match_type, has_negation, intent, result.temporal
            )

            toggle = MachineToggle(
                machine_id=machine_id,
                machine_name=machine_name,
                action=intent,
                trigger_text=trigger_text,
                confidence=confidence,
                match_type=match_type,
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
        Determine the intent: ON, OFF, or STANDBY.

        Logic:
        - "start ventilator" → ON
        - "don't start ventilator" → OFF (negation flips ON → OFF)
        - "stop ventilator" → OFF
        - "don't stop ventilator" → ON (negation flips OFF → ON)
        - "put ventilator on standby" → STANDBY
        - "don't put on standby" → ON (negation flips STANDBY → ON)
        """
        is_standby = any(re.search(p, text) for p in STANDBY_KEYWORDS)
        if is_standby and not has_negation:
            return "STANDBY"
        if is_standby and has_negation:
            return "ON"

        is_activate = any(re.search(p, text) for p in ACTIVATE_KEYWORDS)
        is_deactivate = any(re.search(p, text) for p in DEACTIVATE_KEYWORDS)

        if is_activate and not is_deactivate:
            return "OFF" if has_negation else "ON"
        elif is_deactivate and not is_activate:
            return "ON" if has_negation else "OFF"
        elif is_activate and is_deactivate:
            return "OFF" if has_negation else "ON"
        else:
            # No explicit intent keyword — default to ON
            return "OFF" if has_negation else "ON"

    def _detect_temporal(self, text: str) -> TemporalQualifier:
        """Extract temporal qualifiers from the text."""
        for pattern, qualifier_type in TEMPORAL_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                delay_hint = match.group(1) if match.lastindex and match.lastindex >= 1 else ""
                return TemporalQualifier(
                    qualifier=qualifier_type,
                    raw_text=match.group(0),
                    delay_hint=delay_hint,
                )
        return TemporalQualifier(qualifier="none")

    def _find_machines(
        self,
        text: str,
        phase: str = "",
    ) -> List[Tuple[str, str, str, str]]:
        """
        Find machines mentioned in the text using regex patterns.

        Returns:
            List of (machine_id, machine_name, matched_text, match_type) tuples.
        """
        matches: List[Tuple[str, str, str, str]] = []
        seen_ids: Set[str] = set()

        # Sort patterns: triggers first, then aliases, then names
        # This ensures the highest-confidence match wins per machine
        priority = {"trigger": 0, "alias": 1, "name": 2}
        sorted_patterns = sorted(
            self._trigger_patterns, key=lambda x: priority.get(x[3], 99)
        )

        for pattern, machine_id, machine_name, match_type in sorted_patterns:
            if machine_id in seen_ids:
                continue

            if pattern.search(text):
                # Phase filtering
                if self.phase_filter and phase:
                    valid_phases = self.machines.get(machine_id, {}).get(
                        "phase_usage", []
                    )
                    if valid_phases and phase not in valid_phases:
                        logger.debug(
                            "Skipping %s — not valid in %s (valid: %s)",
                            machine_id,
                            phase,
                            valid_phases,
                        )
                        continue

                matches.append((machine_id, machine_name, pattern.pattern, match_type))
                seen_ids.add(machine_id)

        return matches

    def _compute_confidence(
        self,
        match_type: str,
        has_negation: bool,
        intent: str,
        temporal: Optional[TemporalQualifier],
    ) -> float:
        """
        Compute a graded confidence score for the match.

        Scoring:
        - Trigger match: 0.95
        - Alias match: 0.85
        - Name match: 0.80
        - Negation penalty: -0.10
        - Immediate temporal bonus: +0.05
        """
        base_scores = {"trigger": 0.95, "alias": 0.85, "name": 0.80}
        score = base_scores.get(match_type, 0.70)

        if has_negation:
            score -= 0.10

        if temporal and temporal.qualifier == "immediately":
            score = min(1.0, score + 0.05)

        return round(max(0.0, min(1.0, score)), 2)

    def set_phase(self, phase: str) -> None:
        """Update the current surgical phase."""
        self.current_phase = phase
        logger.info("RuleEngine phase set to: %s", phase)

    def switch_surgery(self, surgery: str) -> None:
        """Switch to a different surgery type and reload machines."""
        self.surgery = surgery
        self.machines = {}
        self.phases = []
        self._trigger_map = {}
        self._trigger_patterns = []
        self._last_toggle_time = {}
        self._load_machines()

    def reset_debounce(self) -> None:
        """Reset all debounce timers."""
        self._last_toggle_time.clear()

    def get_machines_summary(self) -> Dict[str, str]:
        """Get a summary of loaded machines."""
        return {
            mid: mdata.get("name", "Unknown")
            for mid, mdata in self.machines.items()
        }

    def get_machine_aliases(self, machine_id: str) -> List[str]:
        """Get all aliases for a machine."""
        mdata = self.machines.get(machine_id, {})
        aliases = list(mdata.get("aliases", []))
        aliases.extend(mdata.get("triggers", []))
        aliases.append(mdata.get("name", ""))
        return [a for a in aliases if a]

    def get_phase_machines(self, phase: str) -> Dict[str, str]:
        """Get machines valid for a specific phase."""
        result = {}
        for mid, mdata in self.machines.items():
            valid_phases = mdata.get("phase_usage", [])
            if phase in valid_phases:
                result[mid] = mdata.get("name", "Unknown")
        return result


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
        "put the vent on standby",
        "start suction and irrigation immediately",
        "don't stop the ventilator after the incision",
    ]

    for phrase in test_phrases:
        result = engine.process(phrase)
        patch = result.to_json_patch()
        print(f"\n  Input: '{phrase}'")
        print(f"  ON:  {patch['machines']['1']}")
        print(f"  OFF: {patch['machines']['0']}")
        print(f"  Temporal: {patch['metadata']['temporal']}")
        print(f"  Time: {result.processing_time_ms:.2f}ms")
