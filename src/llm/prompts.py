"""
OR-Symphony: LLM Prompt Templates

Structured prompt engineering for MedGemma surgical reasoning.
Provides system and user prompt templates with machine dictionary context,
transcript context, and strict JSON output format instructions.

Usage:
    from src.llm.prompts import PromptBuilder
    builder = PromptBuilder(surgery="PCNL")
    messages = builder.build_messages(transcript_context, current_phase)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.utils.constants import SUPPORTED_SURGERIES

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config path for per-surgery machine files
# ---------------------------------------------------------------------------

CONFIGS_MACHINES_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "machines"

_SURGERY_CONFIG_MAP: Dict[str, str] = {
    "PCNL": "pcnl.json",
    "Partial Hepatectomy": "partial_hepatectomy.json",
    "Lobectomy": "lobectomy.json",
}


# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------

SYSTEM_PROMPT_TEMPLATE = """\
You are OR-Symphony, an AI assistant for real-time surgical state tracking in the operating room.

## Your Role
- Analyze surgical OR transcript segments to determine which machines should be ON, OFF, or in STANDBY.
- Identify the current surgical phase and predict the next phase.
- Provide brief reasoning and suggestions.
- Output ONLY valid JSON matching the schema below.

## Surgery: {surgery_name}
{surgery_description}

## Surgical Phases
{phases_text}

## Available Machines
{machines_text}

## Output JSON Schema
Your response MUST be a single valid JSON object with these exact keys:
```json
{{
  "metadata": {{
    "phase": "<current phase ID, e.g., Phase3>",
    "reasoning": "<brief reasoning for your decisions>",
    "next_phase": "<predicted next phase ID, or empty string>"
  }},
  "machines": {{
    "0": ["<machine IDs that should be OFF or STANDBY>"],
    "1": ["<machine IDs that should be ON>"]
  }},
  "details": {{
    "toggles": [
      {{
        "machine_id": "<M## ID>",
        "name": "<machine name>",
        "action": "<ON|OFF|STANDBY>",
        "trigger": "<keyword that triggered this>",
        "confidence": <0.0-1.0>,
        "match_type": "llm"
      }}
    ]
  }},
  "suggestions": ["<actionable suggestions for OR staff>"],
  "confidence": <0.0-1.0 overall confidence>,
  "source": "medgemma"
}}
```

## Rules
1. Only output JSON — no markdown, explanations, or extra text.
2. Machine IDs must match the format M## (e.g., M01, M09).
3. Only reference machines listed above.
4. Confidence must be between 0.0 and 1.0.
5. If uncertain, set confidence low and keep machines unchanged.
6. Consider phase-appropriate machines — not all machines are relevant in all phases.
"""

# ---------------------------------------------------------------------------
# User prompt template
# ---------------------------------------------------------------------------

USER_PROMPT_TEMPLATE = """\
## Current State
- Surgery: {surgery_name}
- Current Phase: {current_phase}
- Session Time: {session_time}

## Recent OR Transcript
{transcript_context}

## Current Machine States (from rule engine)
{current_machines}

Based on the transcript above, determine:
1. Which machines should change state (ON/OFF/STANDBY)?
2. What is the current surgical phase?
3. What phase comes next?
4. Any suggestions for the OR team?

Respond with JSON only.
"""


# ---------------------------------------------------------------------------
# PromptBuilder
# ---------------------------------------------------------------------------


class PromptBuilder:
    """
    Builds structured prompts for MedGemma surgical reasoning.

    Loads the surgery-specific machine dictionary and formats
    system + user prompts with context.
    """

    def __init__(self, surgery: str = "PCNL") -> None:
        """
        Initialize the prompt builder.

        Args:
            surgery: Surgery type name (must be in SUPPORTED_SURGERIES).
        """
        self.surgery = surgery
        self._config: Optional[Dict[str, Any]] = None
        self._load_config()

    def _load_config(self) -> None:
        """Load surgery-specific machine config."""
        config_file = _SURGERY_CONFIG_MAP.get(self.surgery)
        if config_file is None:
            logger.warning(
                "No config for surgery '%s', using empty config", self.surgery
            )
            self._config = {"surgery": self.surgery, "machines": {}, "phases": []}
            return

        config_path = CONFIGS_MACHINES_DIR / config_file
        if not config_path.exists():
            logger.warning("Config file not found: %s", config_path)
            self._config = {"surgery": self.surgery, "machines": {}, "phases": []}
            return

        with open(config_path, "r", encoding="utf-8") as f:
            self._config = json.load(f)

        logger.debug(
            "Loaded config for %s: %d machines, %d phases",
            self.surgery,
            len(self._config.get("machines", {})),
            len(self._config.get("phases", [])),
        )

    def set_surgery(self, surgery: str) -> None:
        """
        Change the surgery type and reload config.

        Args:
            surgery: New surgery type name.
        """
        if surgery != self.surgery:
            self.surgery = surgery
            self._load_config()
            logger.info("PromptBuilder switched to surgery: %s", surgery)

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def build_system_prompt(self) -> str:
        """
        Build the system prompt with surgery context.

        Returns:
            Formatted system prompt string.
        """
        config = self._config or {}

        # Surgery description
        surgery_desc = f"Full name: {config.get('full_name', self.surgery)}"
        env = config.get("environment", "")
        if env:
            surgery_desc += f"\nEnvironment: {env}"

        # Phases text
        phases = config.get("phases", [])
        if phases:
            phases_text = "\n".join(
                f"- {p['id']}: {p['name']} — {p.get('description', '')}"
                for p in phases
            )
        else:
            phases_text = "No phases defined."

        # Machines text
        machines = config.get("machines", {})
        if machines:
            lines = []
            for mid, mdata in sorted(machines.items()):
                name = mdata.get("name", mid)
                category = mdata.get("category", "")
                desc = mdata.get("description", "")
                default = mdata.get("default_state", "OFF")
                phase_usage = ", ".join(mdata.get("phase_usage", []))
                lines.append(
                    f"- {mid}: {name} [{category}] — {desc} "
                    f"(default: {default}, phases: {phase_usage})"
                )
            machines_text = "\n".join(lines)
        else:
            machines_text = "No machines defined."

        return SYSTEM_PROMPT_TEMPLATE.format(
            surgery_name=self.surgery,
            surgery_description=surgery_desc,
            phases_text=phases_text,
            machines_text=machines_text,
        )

    def build_user_prompt(
        self,
        transcript_context: str,
        current_phase: str = "",
        session_time: str = "",
        current_machines: Optional[Dict[str, List[str]]] = None,
    ) -> str:
        """
        Build the user prompt with transcript context.

        Args:
            transcript_context: Recent transcript text from rolling buffer.
            current_phase: Current surgical phase ID.
            session_time: Human-readable session elapsed time.
            current_machines: Current machine states {"0": [...], "1": [...]}.

        Returns:
            Formatted user prompt string.
        """
        if current_machines is None:
            current_machines = {"0": [], "1": []}

        # Format current machines for display
        machines_on = current_machines.get("1", [])
        machines_off = current_machines.get("0", [])
        if machines_on or machines_off:
            machines_str = f"ON: {', '.join(machines_on) if machines_on else 'none'}\n"
            machines_str += f"OFF/STANDBY: {', '.join(machines_off) if machines_off else 'none'}"
        else:
            machines_str = "No machine states available."

        return USER_PROMPT_TEMPLATE.format(
            surgery_name=self.surgery,
            current_phase=current_phase or "Unknown",
            session_time=session_time or "N/A",
            transcript_context=transcript_context or "<no transcript available>",
            current_machines=machines_str,
        )

    def build_messages(
        self,
        transcript_context: str,
        current_phase: str = "",
        session_time: str = "",
        current_machines: Optional[Dict[str, List[str]]] = None,
    ) -> List[Dict[str, str]]:
        """
        Build a chat-style message list for the model.

        Args:
            transcript_context: Recent transcript from rolling buffer.
            current_phase: Current surgical phase ID.
            session_time: Human-readable session time.
            current_machines: Current machine states.

        Returns:
            List of {"role": ..., "content": ...} message dicts.
        """
        return [
            {"role": "system", "content": self.build_system_prompt()},
            {
                "role": "user",
                "content": self.build_user_prompt(
                    transcript_context=transcript_context,
                    current_phase=current_phase,
                    session_time=session_time,
                    current_machines=current_machines,
                ),
            },
        ]

    def build_completion_prompt(
        self,
        transcript_context: str,
        current_phase: str = "",
        session_time: str = "",
        current_machines: Optional[Dict[str, List[str]]] = None,
    ) -> str:
        """
        Build a single completion-style prompt (concatenated system + user).

        Args:
            transcript_context: Recent transcript from rolling buffer.
            current_phase: Current surgical phase ID.
            session_time: Human-readable session time.
            current_machines: Current machine states.

        Returns:
            Single formatted prompt string.
        """
        system = self.build_system_prompt()
        user = self.build_user_prompt(
            transcript_context=transcript_context,
            current_phase=current_phase,
            session_time=session_time,
            current_machines=current_machines,
        )
        return f"{system}\n\n{user}\n\nJSON Response:\n"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def machine_count(self) -> int:
        """Number of machines in the loaded config."""
        return len((self._config or {}).get("machines", {}))

    @property
    def phase_count(self) -> int:
        """Number of phases in the loaded config."""
        return len((self._config or {}).get("phases", []))

    @property
    def machine_ids(self) -> List[str]:
        """List of machine IDs in the loaded config."""
        return sorted((self._config or {}).get("machines", {}).keys())

    @property
    def config(self) -> Dict[str, Any]:
        """The loaded surgery config."""
        return dict(self._config or {})


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    for surgery in SUPPORTED_SURGERIES:
        builder = PromptBuilder(surgery=surgery)
        print(f"\n{'='*60}")
        print(f"Surgery: {surgery} ({builder.machine_count} machines, {builder.phase_count} phases)")
        print(f"Machine IDs: {builder.machine_ids}")

        # Build sample prompts
        transcript = "[10.0s] surgeon: start the fluoroscopy\n[15.0s] nurse: C-arm is positioned"
        messages = builder.build_messages(
            transcript_context=transcript,
            current_phase="Phase3",
            session_time="2m 30s",
            current_machines={"0": ["M03", "M05"], "1": ["M01", "M02"]},
        )
        print(f"\nSystem prompt: {len(messages[0]['content'])} chars")
        print(f"User prompt: {len(messages[1]['content'])} chars")
        print(f"System prompt preview:\n{messages[0]['content'][:300]}...")
