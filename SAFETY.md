# OR-Symphony: Safety Policy

> **Version:** 1.0  
> **Date:** 2026-02-22  
> **Status:** Active  

---

## 1. Purpose

OR-Symphony is a **simulation and research system** designed to demonstrate
AI-assisted surgical state tracking. It processes audio transcripts and
produces structured suggestions about operating room equipment states.

**This system does NOT control real medical devices.**

---

## 2. Human-in-the-Loop Requirement

### 2.1 Mandatory Human Confirmation

Every output produced by OR-Symphony is a **suggestion only**. No machine state
change takes effect without explicit human confirmation:

- **Rule engine outputs** identify potential equipment commands from speech.
  These are displayed as suggestions, not executed automatically.
- **LLM (MedGemma) outputs** provide contextual reasoning about surgical phase
  and equipment needs. These are advisory and require manual review.
- **Manual overrides** require the operator to provide a reason, which is
  logged to an immutable audit trail.

### 2.2 No Autonomous Actions

The system **MUST NOT**:

- Directly control any medical device, actuator, or safety-critical system
- Send commands to real equipment APIs or hardware interfaces
- Make clinical decisions without human oversight
- Execute any action that could affect patient safety

### 2.3 Operator Responsibilities

The operating room team retains **full authority** over all equipment decisions.
The system's role is limited to:

1. Transcribing speech (informational only)
2. Identifying potential equipment commands (suggestions only)
3. Providing contextual reasoning (advisory only)
4. Displaying current predicted state (visualization only)

---

## 3. Simulation Boundaries

### 3.1 What the System Does

| Capability | Description |
|-----------|-------------|
| Speech-to-text | Transcribes audio using MedASR (ONNX) |
| Rule matching | Maps keywords to equipment identifiers |
| LLM reasoning | Contextual phase and equipment suggestions |
| State tracking | Maintains a JSON state of predicted equipment states |
| Visualization | Displays state via WebSocket to a frontend |
| Audit logging | Records all state changes and overrides |

### 3.2 What the System Does NOT Do

| Boundary | Description |
|----------|-------------|
| Device control | No APIs to real medical equipment |
| Clinical decisions | No diagnosis, treatment, or dosage recommendations |
| Patient data | No access to patient records or PHI |
| Autonomous operation | No action without human confirmation |
| Safety-critical output | All outputs are clearly labeled as suggestions |

### 3.3 Output Format

All system outputs follow a strict JSON schema that includes:

- **`source`**: Indicates whether the output came from the rule engine,
  MedGemma, or both (`"rule"`, `"medgemma"`, `"rule+medgemma"`)
- **`confidence`**: A 0.0–1.0 score indicating the system's certainty
- **`suggestions`**: A list of human-readable suggestion strings
- **`metadata.reasoning`**: Explanation of the reasoning process

No output field contains executable instructions. The `suggestions` array
contains only natural-language advisory text.

---

## 4. Degraded Mode

When the LLM component fails or is unavailable, the system enters
**degraded mode**:

- The rule engine continues to operate independently
- Outputs are marked with `source: "rule"` and `reasoning: "degraded"`
- LLM suggestions are not available
- The system continues to function with reduced capability
- The frontend displays a visual indicator of degraded mode

---

## 5. Audit Trail

### 5.1 Immutable Logging

All system activities are logged to append-only files:

| Log File | Content | Format |
|----------|---------|--------|
| `logs/transcripts/YYYYMMDD.log` | All ASR transcripts | Timestamped text |
| `logs/state_changes.log` | Every state update | JSON with SHA-256 checksum |
| `logs/overrides.log` | Manual overrides | JSON with SHA-256 checksum |
| `logs/app_YYYYMMDD.log` | Application events | Structured log format |

### 5.2 Checksum Integrity

State change and override log entries include:

- **SHA-256 hash** of the entry payload for tamper detection
- **Previous hash** reference for chain integrity verification
- **Timestamp** in ISO 8601 UTC format
- **Operator identifier** (for overrides)

### 5.3 Log Retention

Log files are append-only and should not be modified or deleted during
active use. Log rotation and archival policies should be established
per institutional requirements.

---

## 6. Data Privacy

- The system processes only **audio from the operating room microphone**
- No patient identifying information is stored or transmitted
- Transcripts are stored locally in `logs/transcripts/`
- No data is sent to external services (all inference runs locally)
- The GGUF model runs entirely on the local machine

---

## 7. Regulatory Notice

This system is **not a medical device** and has not been approved by any
regulatory authority (FDA, CE, etc.). It is intended solely for:

- Academic research
- Simulation and training
- Technology demonstration
- Software engineering education

It must NOT be deployed in clinical settings without proper regulatory
review, validation, and approval.

---

## 8. Known Limitations

| Limitation | Impact |
|-----------|--------|
| ASR accuracy | Transcription errors may cause incorrect rule matches |
| LLM hallucination | MedGemma may produce incorrect suggestions |
| Single-threaded GGUF | LLM inference is rate-limited (~2s intervals) |
| No real-time guarantees | System is best-effort, not real-time certified |
| English only | ASR and LLM are trained on English medical speech |
| Limited surgery types | Only 3 surgery types are currently supported |

---

## 9. Contact and Reporting

Issues with this system should be reported through the project's issue
tracker. Safety concerns should be escalated immediately to the project
maintainers.

---

## 10. Sign-Off

By using this system, you acknowledge that:

1. It is a simulation and research tool only
2. It does not control real medical devices
3. All outputs require human confirmation
4. You will not use it for clinical decision-making
5. You understand the limitations described above

**Reviewed by:** ______ (initials/date)
