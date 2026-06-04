# DScribe Discharge Summary Agent

A clinical AI agent that reads patient source-note PDFs and produces a
structured, clinically-safe discharge summary **draft** for clinician review.

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your Anthropic API key
export ANTHROPIC_API_KEY=your-key-here

# 3. Generate synthetic test data
python generate_sample_data.py

# 4. Run on all patients
python main.py --all_patients ./sample_data --output_dir ./outputs

# 5. Run on a single patient
python main.py --patient_folder ./sample_data/patient_001 --patient_id P001
```

Outputs per patient:
- `outputs/{patient_id}_discharge_summary_DRAFT.txt` — the discharge summary
- `outputs/{patient_id}_step_trace.json` — full step-by-step agent trace

---

## Agent Loop Design

The agent runs a structured pipeline with a **hard cap of 40 steps** so it
can never run forever. Each step emits a trace entry:
`reasoning → action → inputs → result → next_step`.

```
PLAN → EXTRACT → VALIDATE → RECONCILE → CONFLICT_CHECK
     → PENDING_CHECK → DRUG_INTERACTION → ESCALATE
     → GENERATE_SUMMARY → REVIEW → DONE
```

### Step descriptions

| Step | What it does |
|------|-------------|
| **PLAN** | Lists PDFs in the patient folder; aborts early if none found |
| **EXTRACT** | Ingests all PDFs (PyMuPDF + OCR fallback); calls LLM to extract structured fields from each note |
| **VALIDATE** | Checks all required fields are present; marks any missing field `[MISSING — CLINICIAN REVIEW REQUIRED]` |
| **RECONCILE** | Diffs admission vs discharge medications; flags any unexplained add/stop/change |
| **CONFLICT_CHECK** | Compares fields across all notes; flags any disagreements (diagnosis, allergy, dates, etc.) |
| **PENDING_CHECK** | Collects structured pending results + scans raw text for pending-keywords |
| **DRUG_INTERACTION** | Calls mock drug-interaction tool on discharge med list; cross-checks allergies for cross-reactivity |
| **ESCALATE** | Collects all escalation flags raised and logs them |
| **GENERATE_SUMMARY** | Passes consolidated data + all flags to LLM; produces the discharge summary draft |
| **REVIEW** | Verifies all required sections exist in the generated summary |

The loop can re-plan: if a step returns `ERROR`, the agent transitions to the
error state rather than continuing silently.

---

## No-Fabrication Guardrail

This is enforced at **three layers**:

1. **LLM system prompt** — The extraction prompt explicitly instructs the model:
   *"Never invent or infer facts not explicitly stated in the text. If a field
   is not found, output null."*

2. **Validation step** — After extraction, every required field is checked. Any
   field that is null/empty is overwritten with the literal string
   `[MISSING — CLINICIAN REVIEW REQUIRED]` before the LLM summary step sees it.

3. **Summary generation prompt** — The summary prompt repeats:
   *"If a field value is [MISSING], reproduce it as-is. Never add plausible
   values, interpretations, or commentary beyond what is sourced."*

The summary is always labelled **DRAFT** and the output never auto-finalises.

---

## Failure & Conflict Handling

### Tool failures
- Every external tool call is wrapped in a retry loop (`MAX_RETRIES = 3`).
- On failure, the agent logs a warning and continues with a `"Tool unavailable"` marker — it never behaves as if a failed call succeeded.
- If PDF ingestion fails for a file, the agent continues with the remaining files and records the failure in the trace.

### Conflicting information
- `conflict_detector.py` compares every scalar field and list field across all source notes.
- Detected conflicts are flagged with `[CONFLICT DETECTED — CLINICIAN REVIEW REQUIRED]` and escalated via the `escalate_to_clinician` tool.
- The agent **never picks a winner** between conflicting values — both values are surfaced.

### Missing/pending data
- Missing fields: marked `[MISSING]`, never filled with a plausible value.
- Pending results: marked `[PENDING]`, surfaced in a dedicated section.
- The `pending_check` step also scans raw text for pending-language patterns to catch anything the LLM extractor missed.

---

## Part 2: Learning Loop

Implemented in `learning_loop.py`.

### Reward signal
A composite of:
- **Normalised edit distance** (1.0 = identical draft/edited, 0.0 = completely rewritten)
- **Section match rate** (fraction of sections with <10% change)
- **Safety penalty** (negative reward if `[MISSING]` or `[CONFLICT]` markers are removed)

### Simulated reviewer
`SimulatedReviewer` applies a consistent hidden editing policy:
- Condenses verbose hospital course paragraphs
- Normalises allergy section format
- Removes redundant blank lines
- Adds a reviewer footer

### Learning mechanism
A **UCB1 contextual bandit** selects among 5 prompt strategies
(default, verbose_narrative, minimal, medication_focused, problem_list).
The bandit updates arm statistics after each (draft, edited) pair.

High-reward (draft, edited) pairs are stored in `CorrectionMemory` and
injected as few-shot examples into future prompts.

### Measured improvement
Run `learning_loop.py` standalone to see before/after metrics comparing
early-iteration rewards to late-iteration rewards.

### Limitations
See `learning_loop.py → LIMITATIONS` for a full discussion, including:
- Cold-start problem with few patients
- Reward gaming risk (vagueness, style mimicry)
- Simulated reviewer fidelity gap
- Single-reviewer overfitting

---

## File Structure

```
discharge_agent/
├── main.py                   # CLI entry point
├── agent_loop.py             # Core agent loop + state machine
├── pdf_ingestion.py          # PyMuPDF + OCR fallback
├── med_reconciliation.py     # Admission vs discharge medication diff
├── conflict_detector.py      # Cross-note conflict detection
├── tools.py                  # Mock tool implementations
├── learning_loop.py          # Part 2: bandit + reviewer + reward signal
├── generate_sample_data.py   # Generates synthetic test PDFs
├── requirements.txt
├── README.md
└── sample_data/
    ├── patient_001/          # CHF case (straightforward)
    └── patient_002/          # SAH case (conflicts + missing + pending)
```

---

## What I Would Do With More Time

1. **Real OCR integration** — test Tesseract/EasyOCR on actual scanned clinical PDFs
2. **Structured output validation** — use Pydantic models to enforce the extracted schema
3. **Named entity recognition** — a clinical NER model (e.g., scispaCy) for more reliable extraction than LLM-only
4. **Retrieval-augmented reconciliation** — drug database lookups (RxNorm, DrugBank) instead of the mock interaction DB
5. **Real learning loop data** — even a small set of real clinician edits would replace the simulated reviewer
6. **Streaming trace UI** — a simple web interface to watch the agent's reasoning in real time
7. **Multi-patient parallelism** — async processing of multiple patient folders simultaneously
