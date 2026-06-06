# DScribe Discharge Summary Agent

A clinical AI agent that reads patient source-note PDFs and generates structured, clinically safe discharge summary drafts for clinician review. The system is designed with strict no-fabrication guardrails, medication reconciliation, conflict detection, pending-result identification, and transparent reasoning traces.

---

## Features

* PDF ingestion and text extraction
* Agentic workflow with planning and validation
* Medication reconciliation
* Conflict detection across source notes
* Pending-result identification
* Drug interaction checking
* Clinician escalation workflow
* No-fabrication safety guardrails
* Step-by-step reasoning trace
* Learning loop for continuous improvement
* FastAPI backend
* Interactive web interface

---

## Quick Start

### 1. Clone the Repository

```bash
git clone <repository-url>
cd discharge_agent
```

### 2. Create a Virtual Environment

```bash
python3 -m venv .venv
```

### 3. Activate the Virtual Environment

Linux / macOS

```bash
source .venv/bin/activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure OpenRouter API Key

```bash
export OPENROUTER_API_KEY=your-api-key
```

### 6. Navigate to Backend Directory

```bash
cd backend
```

### 7. Start the FastAPI Server

```bash
uvicorn server:app --reload --port 8000
```

### 8. Access the Application

Application URL:

```text
http://127.0.0.1:8000
```

API Documentation:

```text
http://127.0.0.1:8000/docs
```

---

## Agent Workflow

The agent follows a structured workflow with a hard iteration limit to prevent infinite execution.

```text
PLAN
 ↓
EXTRACT
 ↓
VALIDATE
 ↓
RECONCILE
 ↓
CONFLICT CHECK
 ↓
PENDING CHECK
 ↓
DRUG INTERACTION
 ↓
ESCALATE
 ↓
GENERATE SUMMARY
 ↓
REVIEW
 ↓
DONE
```

Each step records:

```text
Reasoning
→ Action
→ Inputs
→ Result
→ Next Decision
```

---

## Agent Pipeline

### PLAN

* Identifies uploaded PDFs
* Builds execution plan
* Initializes agent state

### EXTRACT

* Reads all uploaded PDFs
* Extracts clinical information
* Converts unstructured text into structured data

### VALIDATE

* Checks required fields
* Identifies missing information
* Applies safety markers

### RECONCILE

* Compares admission and discharge medications
* Identifies additions, removals, and modifications

### CONFLICT CHECK

* Detects disagreements across documents
* Flags conflicting diagnoses, medications, dates, and allergies

### PENDING CHECK

* Identifies pending laboratory results
* Detects incomplete investigations

### DRUG INTERACTION

* Checks discharge medications
* Identifies potential interaction risks

### ESCALATE

* Creates clinician review flags
* Surfaces safety concerns

### GENERATE SUMMARY

* Produces structured discharge summary draft

### REVIEW

* Verifies required sections exist
* Performs final safety validation

---

## No-Fabrication Guardrails

Clinical safety is enforced at multiple layers.

### Extraction Guardrail

The extraction model is instructed to:

```text
Never invent or infer facts not explicitly stated in the source documents.
Return null for unavailable information.
```

### Validation Guardrail

Missing values are automatically replaced with:

```text
[MISSING — CLINICIAN REVIEW REQUIRED]
```

### Conflict Guardrail

Conflicting values are replaced with:

```text
[CONFLICT DETECTED — CLINICIAN REVIEW REQUIRED]
```

### Pending Result Guardrail

Pending investigations are marked as:

```text
[PENDING RESULT]
```

### Summary Guardrail

The final summary is always generated as:

```text
DRAFT ONLY — FOR CLINICIAN REVIEW
```

The system never auto-finalizes a clinical document.

---

## Failure Handling

### PDF Processing Failures

* Failed documents are logged
* Remaining files continue processing
* Failures are recorded in trace logs

### Tool Failures

* Automatic retry mechanism
* Graceful fallback behavior
* Explicit error logging

### Agent Failures

* Hard iteration cap
* Controlled state transitions
* No silent failures

---

## Medication Reconciliation

The agent compares:

* Admission medication list
* Discharge medication list

Detected changes include:

* Added medications
* Removed medications
* Modified medications
* Unexplained changes

Any unexplained modification is flagged for clinician review.

---

## Conflict Detection

The system compares information across all source notes.

Conflict categories:

* Diagnoses
* Medications
* Allergies
* Admission dates
* Discharge dates
* Follow-up instructions

The agent never chooses between conflicting values. All conflicts are surfaced for clinician review.

---

## Learning Loop

The learning module improves future drafts using reviewer feedback.

### Reward Signal

The reward function combines:

* Normalized edit distance
* Section match rate
* Safety preservation score

### Simulated Reviewer

The reviewer applies:

* Formatting corrections
* Section normalization
* Redundancy removal
* Consistent editing patterns

### Learning Strategy

* Contextual Bandit (UCB1)
* Prompt strategy selection
* Correction memory
* Few-shot feedback injection

---

## Outputs

Generated outputs include:

```text
outputs/
├── discharge_summary.txt
├── step_trace.json
├── conflicts.json
├── escalations.json
└── learning_metrics.json
```

---

## Project Structure

```text
discharge_agent/
│
├── backend/
│   ├── server.py
│   ├── agent_loop.py
│   ├── pdf_ingestion.py
│   ├── med_reconciliation.py
│   ├── conflict_detector.py
│   ├── tools.py
│   ├── prompts.py
│   ├── learning_loop.py
│   ├── models.py
│   └── utils.py
│
├── frontend/
│
├── outputs/
│
├── requirements.txt
├── README.md
└── .env
```

---

## Future Improvements

1. Production-grade OCR integration
2. Clinical NER models
3. Structured schema validation using Pydantic
4. Drug database integration (RxNorm / DrugBank)
5. Human feedback learning
6. Real-time monitoring dashboard
7. Multi-patient parallel processing
8. Enhanced audit logging
9. Clinical terminology normalization

---

## Disclaimer

This system generates draft discharge summaries intended solely for clinician review. It is not a replacement for medical judgment, diagnosis, treatment decisions, or final clinical documentation. All generated outputs must be reviewed and approved by a qualified healthcare professional before use.
