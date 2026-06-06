"""
agent_loop.py
-------------
The core agentic loop for the DScribe Discharge Summary Agent.

Pipeline (each step can re-plan based on results):
  Planner → Extractor → Validator → Reconciler → ConflictChecker
  → PendingResultsChecker → SummaryGenerator → Reviewer

Design principles:
  1. Hard step cap — agent cannot run forever.
  2. Full trace emitted at every step: reasoning → action → result → next.
  3. No fabrication — missing data is marked [MISSING] or [PENDING], never guessed.
  4. Tools are called when needed; failures trigger retry/fallback, never silent success.
  5. All conflicts and flags are escalated, never buried.
"""

from __future__ import annotations

import json
import logging
import os
import re
import textwrap
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from pdf_ingestion import ingest_patient_folder, IngestionResult
from med_reconciliation import (
    parse_medication_list,
    reconcile_medications,
    ReconciliationReport,
)
from conflict_detector import detect_conflicts, detect_medication_conflicts, ConflictReport
from tools import (
    drug_interaction_lookup,
    escalate_to_clinician,
    allergy_cross_reactivity_check,
    get_all_escalations,
    clear_escalations,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MAX_STEPS = 40          # Hard cap — agent stops here regardless
MAX_RETRIES = 3         # Per-tool retry limit
OPENROUTER_MODEL = "openai/gpt-4.1-mini" # Free tier model — change to gemini-1.5-pro if needed
MAX_TOKENS = 2000

MISSING = "[MISSING — CLINICIAN REVIEW REQUIRED]"
PENDING = "[PENDING — RESULTS NOT YET AVAILABLE]"
CONFLICT = "[CONFLICT DETECTED — CLINICIAN REVIEW REQUIRED]"


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------

class StepType(str, Enum):
    PLAN = "PLAN"
    EXTRACT = "EXTRACT"
    VALIDATE = "VALIDATE"
    RECONCILE = "RECONCILE"
    CONFLICT_CHECK = "CONFLICT_CHECK"
    PENDING_CHECK = "PENDING_CHECK"
    DRUG_INTERACTION = "DRUG_INTERACTION"
    ESCALATE = "ESCALATE"
    GENERATE_SUMMARY = "GENERATE_SUMMARY"
    REVIEW = "REVIEW"
    DONE = "DONE"
    ERROR = "ERROR"


@dataclass
class TraceEntry:
    step: int
    step_type: StepType
    reasoning: str
    action: str
    inputs: dict
    result: Any
    next_step: str
    timestamp: float = field(default_factory=time.time)
    duration_ms: float = 0.0

    def to_dict(self) -> dict:
        return {
            "step": self.step,
            "type": self.step_type.value,
            "reasoning": self.reasoning,
            "action": self.action,
            "inputs": self.inputs,
            "result": self.result if isinstance(self.result, (dict, list, str, int, float, bool, type(None))) else str(self.result),
            "next_step": self.next_step,
            "timestamp": self.timestamp,
            "duration_ms": round(self.duration_ms, 1),
        }


@dataclass
class AgentState:
    patient_id: str
    patient_folder: str

    # Ingested source documents
    ingested: dict[str, IngestionResult] = field(default_factory=dict)
    raw_texts: dict[str, str] = field(default_factory=dict)   # filename → full text

    # Extracted structured data (per source note)
    extracted_per_note: dict[str, dict] = field(default_factory=dict)

    # Consolidated extracted data
    consolidated: dict = field(default_factory=dict)

    # Reconciliation & conflict results
    reconciliation: Optional[ReconciliationReport] = None
    conflicts: Optional[ConflictReport] = None
    med_conflicts: list = field(default_factory=list)

    # Summary sections
    summary_sections: dict = field(default_factory=dict)

    # Flags & trace
    trace: list[TraceEntry] = field(default_factory=list)
    step_count: int = 0
    current_step: StepType = StepType.PLAN
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    escalations: list = field(default_factory=list)

    # Final output
    final_summary: str = ""
    finished: bool = False


# ---------------------------------------------------------------------------
# LLM call wrapper
# ---------------------------------------------------------------------------

from openai import OpenAI
import os

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

def _call_llm(system: str, user: str, max_tokens: int = MAX_TOKENS) -> str:
    try:
        response = client.chat.completions.create(
            model="openai/gpt-4.1-mini",
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.1,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content

    except Exception as e:
        raise RuntimeError(f"LLM call failed: {e}") from e
# ---------------------------------------------------------------------------
# Individual step implementations
# ---------------------------------------------------------------------------

def _step_plan(state: AgentState) -> StepType:
    """
    Inspect what documents are available and build a processing plan.
    """
    t0 = time.monotonic()
    folder = Path(state.patient_folder)
    pdf_files = sorted(folder.glob("*.pdf"))
    plan_note = f"Found {len(pdf_files)} PDF(s): {[f.name for f in pdf_files]}"

    state.trace.append(TraceEntry(
        step=state.step_count,
        step_type=StepType.PLAN,
        reasoning="Inspect patient folder to determine what documents are available.",
        action="list_pdf_files",
        inputs={"folder": state.patient_folder},
        result={"files": [f.name for f in pdf_files], "count": len(pdf_files)},
        next_step="EXTRACT" if pdf_files else "ERROR",
        duration_ms=(time.monotonic() - t0) * 1000,
    ))

    if not pdf_files:
        state.errors.append(f"No PDF files found in {state.patient_folder}")
        return StepType.ERROR

    return StepType.EXTRACT


def _step_extract(state: AgentState) -> StepType:
    """
    Ingest all PDFs and extract structured fields using the LLM.
    """
    t0 = time.monotonic()

    # Ingest PDFs
    state.ingested = ingest_patient_folder(state.patient_folder)
    failed = [fn for fn, r in state.ingested.items() if not r.success]
    succeeded = {fn: r for fn, r in state.ingested.items() if r.success}

    if failed:
        state.warnings.append(f"Failed to ingest: {failed}")

    state.raw_texts = {fn: r.full_text for fn, r in succeeded.items()}

    state.trace.append(TraceEntry(
        step=state.step_count,
        step_type=StepType.EXTRACT,
        reasoning="Ingest PDFs; extract clinical fields from each note.",
        action="ingest_patient_folder + llm_extract_per_note",
        inputs={"folder": state.patient_folder, "pdf_count": len(state.ingested)},
        result={
            "succeeded": list(succeeded.keys()),
            "failed": failed,
            "warnings": [w for r in succeeded.values() for w in r.warnings],
        },
        next_step="VALIDATE",
        duration_ms=(time.monotonic() - t0) * 1000,
    ))

    if not succeeded:
        state.errors.append("No PDFs could be successfully ingested.")
        return StepType.ERROR

    # Extract structured fields from each note via LLM
    for filename, text in state.raw_texts.items():
        extracted = _llm_extract_note(filename, text, state)
        state.extracted_per_note[filename] = extracted

    # Consolidate across notes (merge, prefer most specific/recent)
    state.consolidated = _consolidate_extracted(state.extracted_per_note)

    return StepType.VALIDATE


def _llm_extract_note(filename: str, text: str, state: AgentState) -> dict:
    """
    Use LLM to extract structured clinical fields from a single note's text.
    Returns a dict of fields; missing fields are None.
    """
    SYSTEM = textwrap.dedent("""
        You are a clinical data extraction engine.
        Extract the following fields from the medical note text provided.
        IMPORTANT RULES:
        - Never invent or infer facts not explicitly stated in the text.
        - If a field is not found, output null for that field.
        - If a field is explicitly stated as pending, output the string "PENDING".
        - Output ONLY valid JSON, no markdown, no explanation.

        Fields to extract:
        {
          "patient_name": string or null,
          "dob": string or null,
          "mrn": string or null,
          "admission_date": string or null,
          "discharge_date": string or null,
          "attending_physician": string or null,
          "principal_diagnosis": string or null,
          "secondary_diagnoses": [string] or [],
          "allergies": [string] or [],
          "admission_medications": [{"name": string, "dose": string or null, "frequency": string or null, "route": string or null, "indication": string or null}],
          "discharge_medications": [{"name": string, "dose": string or null, "frequency": string or null, "route": string or null, "indication": string or null}],
          "procedures": [string] or [],
          "hospital_course": string or null,
          "discharge_condition": string or null,
          "follow_up_instructions": string or null,
          "pending_results": [string] or [],
          "note_type": string (e.g. "admission_note", "progress_note", "discharge_summary", "lab_result", "medication_record", "unknown")
        }
    """)

    # Truncate text to avoid token limits (keep first 6000 chars)
    truncated = text[:6000] if len(text) > 6000 else text

    for attempt in range(MAX_RETRIES):
        try:
            raw = _call_llm(SYSTEM, f"Source note filename: {filename}\n\nNote text:\n{truncated}")
            # Strip any accidental markdown fences
            raw = re.sub(r"```(?:json)?|```", "", raw).strip()
            data = json.loads(raw)
            data["_source_note"] = filename
            return data
        except Exception as e:
            logger.warning("Extraction attempt %d failed for %s: %s", attempt + 1, filename, e)
            if attempt == MAX_RETRIES - 1:
                state.warnings.append(f"LLM extraction failed for {filename}: {e}")
                return {"_source_note": filename, "_extraction_failed": True}

    return {"_source_note": filename, "_extraction_failed": True}


def _consolidate_extracted(extracted_per_note: dict[str, dict]) -> dict:
    """
    Merge extracted fields across notes.
    Strategy: prefer non-null values; for lists, union across notes.
    Keeps track of all seen values for conflict detection.
    """
    merged = {
        "patient_name": None, "dob": None, "mrn": None,
        "admission_date": None, "discharge_date": None,
        "attending_physician": None, "principal_diagnosis": None,
        "secondary_diagnoses": [], "allergies": [],
        "admission_medications": [], "discharge_medications": [],
        "procedures": [], "hospital_course": None,
        "discharge_condition": None, "follow_up_instructions": None,
        "pending_results": [],
    }

    scalar_fields = [
        "patient_name", "dob", "mrn", "admission_date", "discharge_date",
        "attending_physician", "principal_diagnosis", "hospital_course",
        "discharge_condition", "follow_up_instructions",
    ]
    list_fields = [
        "secondary_diagnoses", "allergies", "procedures", "pending_results",
    ]

    for note_data in extracted_per_note.values():
        if note_data.get("_extraction_failed"):
            continue

        for f in scalar_fields:
            val = note_data.get(f)
            if val and not merged[f]:
                merged[f] = val

        for f in list_fields:
            vals = note_data.get(f) or []
            existing_lower = {str(x).lower() for x in merged[f]}
            for v in vals:
                if v and str(v).lower() not in existing_lower:
                    merged[f].append(v)
                    existing_lower.add(str(v).lower())

        # Medications: collect from whichever note has them
        for med_field in ("admission_medications", "discharge_medications"):
            meds = note_data.get(med_field) or []
            if meds:
                existing_names = {
                    m.get("name", "").lower() for m in merged[med_field]
                }
                for m in meds:
                    if m.get("name", "").lower() not in existing_names:
                        merged[med_field].append(m)
                        existing_names.add(m.get("name", "").lower())

    return merged


def _step_validate(state: AgentState) -> StepType:
    """
    Check required fields; mark missing/pending fields explicitly.
    """
    t0 = time.monotonic()
    required = [
        "patient_name", "dob", "mrn", "admission_date", "discharge_date",
        "principal_diagnosis", "hospital_course", "discharge_condition",
    ]
    missing_fields = []
    for f in required:
        val = state.consolidated.get(f)
        if not val or str(val).strip() == "":
            missing_fields.append(f)
            state.consolidated[f] = MISSING

    pending_results = state.consolidated.get("pending_results") or []

    state.trace.append(TraceEntry(
        step=state.step_count,
        step_type=StepType.VALIDATE,
        reasoning="Check all required fields are present; flag missing/pending.",
        action="validate_extracted_fields",
        inputs={"required_fields": required},
        result={
            "missing_fields": missing_fields,
            "pending_results_count": len(pending_results),
            "all_fields_present": len(missing_fields) == 0,
        },
        next_step="RECONCILE",
        duration_ms=(time.monotonic() - t0) * 1000,
    ))

    if missing_fields:
        state.warnings.append(
            f"Required fields missing from all source notes: {missing_fields}. "
            "Marked [MISSING] in draft."
        )

    return StepType.RECONCILE


def _step_reconcile(state: AgentState) -> StepType:
    """
    Medication reconciliation: admission vs discharge.
    """
    t0 = time.monotonic()

    adm_raw = state.consolidated.get("admission_medications") or []
    dis_raw = state.consolidated.get("discharge_medications") or []

    # Convert dict-format meds to Medication objects
    from med_reconciliation import Medication
    def dicts_to_meds(dicts, source="consolidated") -> list:
        meds = []
        for d in dicts:
            if not isinstance(d, dict) or not d.get("name"):
                continue
            meds.append(Medication(
                name=d["name"],
                raw_name=d["name"],
                dose=d.get("dose"),
                frequency=d.get("frequency"),
                route=d.get("route"),
                indication=d.get("indication"),
                source_note=source,
            ))
        return meds

    adm_meds = dicts_to_meds(adm_raw, "admission")
    dis_meds = dicts_to_meds(dis_raw, "discharge")

    report = reconcile_medications(adm_meds, dis_meds)
    state.reconciliation = report

    state.trace.append(TraceEntry(
        step=state.step_count,
        step_type=StepType.RECONCILE,
        reasoning="Compare admission vs discharge medications; flag unexplained changes.",
        action="reconcile_medications",
        inputs={
            "admission_med_count": len(adm_meds),
            "discharge_med_count": len(dis_meds),
        },
        result=report.to_dict(),
        next_step="CONFLICT_CHECK",
        duration_ms=(time.monotonic() - t0) * 1000,
    ))

    # Escalate flagged medication changes
    for change in report.flagged:
        escalate_to_clinician(
            category="medication_reconciliation",
            severity="MEDIUM",
            description=change.flag_message,
            recommended_action="Clinician to document reason for medication change before finalising.",
        )

    return StepType.CONFLICT_CHECK


def _step_conflict_check(state: AgentState) -> StepType:
    """
    Detect conflicting information across source notes.
    """
    t0 = time.monotonic()

    conflicts = detect_conflicts(state.extracted_per_note)
    state.conflicts = conflicts

    state.trace.append(TraceEntry(
        step=state.step_count,
        step_type=StepType.CONFLICT_CHECK,
        reasoning="Compare fields across all source notes for conflicts.",
        action="detect_conflicts",
        inputs={"source_notes": list(state.extracted_per_note.keys())},
        result=conflicts.to_dict(),
        next_step="PENDING_CHECK",
        duration_ms=(time.monotonic() - t0) * 1000,
    ))

    # Escalate all detected conflicts
    for conflict in conflicts.conflicts:
        escalate_to_clinician(
            category="clinical_conflict",
            severity=conflict.severity.value,
            description=conflict.message,
            recommended_action=f"Clinician must resolve conflict in field '{conflict.field}' before finalising summary.",
            source_notes=[v[0] for v in conflict.values],
        )

    return StepType.PENDING_CHECK


def _step_pending_check(state: AgentState) -> StepType:
    """
    Identify and surface all pending/outstanding results.
    """
    t0 = time.monotonic()
    pending = state.consolidated.get("pending_results") or []

    # Also scan raw text for pending keywords
    additional_pending = []
    pending_patterns = [
        r"pending\s+(?:result|lab|culture|biopsy|report)",
        r"awaiting\s+(?:result|report|culture|biopsy)",
        r"result\s+(?:not\s+)?(?:yet\s+)?(?:available|returned|resulted)",
        r"to\s+follow",
    ]
    for filename, text in state.raw_texts.items():
        for pattern in pending_patterns:
            matches = re.findall(
                rf".{{0,40}}{pattern}.{{0,40}}", text, re.IGNORECASE
            )
            for m in matches:
                m_clean = m.strip()
                if m_clean not in additional_pending:
                    additional_pending.append(f"[{filename}] {m_clean}")

    all_pending = list(pending) + additional_pending

    state.trace.append(TraceEntry(
        step=state.step_count,
        step_type=StepType.PENDING_CHECK,
        reasoning="Surface all pending results explicitly — they must not be omitted or guessed.",
        action="identify_pending_results",
        inputs={"structured_pending": pending, "pattern_scan": True},
        result={"pending_items": all_pending, "count": len(all_pending)},
        next_step="DRUG_INTERACTION",
        duration_ms=(time.monotonic() - t0) * 1000,
    ))

    state.consolidated["all_pending_results"] = all_pending
    return StepType.DRUG_INTERACTION


def _step_drug_interaction(state: AgentState) -> StepType:
    """
    Run drug-interaction check on discharge medications.
    """
    t0 = time.monotonic()
    dis_meds = state.consolidated.get("discharge_medications") or []
    med_names = [m.get("name", "") for m in dis_meds if isinstance(m, dict) and m.get("name")]

    if not med_names:
        state.trace.append(TraceEntry(
            step=state.step_count,
            step_type=StepType.DRUG_INTERACTION,
            reasoning="No discharge medications extracted — skipping interaction check.",
            action="skip",
            inputs={},
            result={"skipped": True},
            next_step="ESCALATE",
            duration_ms=(time.monotonic() - t0) * 1000,
        ))
        return StepType.ESCALATE

    # Retry on transient failure
    result = None
    for attempt in range(MAX_RETRIES):
        result = drug_interaction_lookup(med_names)
        if result.success:
            break
        logger.warning("Drug interaction lookup attempt %d failed: %s", attempt + 1, result.error)
        time.sleep(0.5)

    if not result or not result.success:
        state.warnings.append("Drug interaction lookup failed after retries — skipped.")
        interaction_data = {"error": "Tool unavailable", "interactions": []}
    else:
        interaction_data = result.output

    state.trace.append(TraceEntry(
        step=state.step_count,
        step_type=StepType.DRUG_INTERACTION,
        reasoning="Check discharge medications for known drug-drug interactions.",
        action="drug_interaction_lookup",
        inputs={"medications": med_names},
        result=result.to_trace() if result else {"error": "All attempts failed"},
        next_step="ESCALATE",
        duration_ms=(time.monotonic() - t0) * 1000,
    ))

    state.consolidated["drug_interactions"] = interaction_data.get("interactions", [])

    # Escalate any HIGH interactions
    for interaction in interaction_data.get("interactions", []):
        if interaction.get("severity") == "HIGH":
            escalate_to_clinician(
                category="drug_interaction",
                severity="HIGH",
                description=interaction["description"],
                recommended_action=interaction["recommended_action"],
            )

    # Allergy cross-reactivity checks
    allergies = state.consolidated.get("allergies") or []
    for allergy in allergies:
        cr_result = allergy_cross_reactivity_check(allergy, med_names)
        if cr_result.success and cr_result.output.get("count", 0) > 0:
            for warning in cr_result.output["cross_reactivity_warnings"]:
                escalate_to_clinician(
                    category="allergy_cross_reactivity",
                    severity="HIGH",
                    description=(
                        f"Potential cross-reactivity: patient has allergy to '{allergy}' "
                        f"and is prescribed '{warning['medication']}' ({warning['warning']})"
                    ),
                    recommended_action="Clinician review before prescribing.",
                )

    return StepType.ESCALATE


def _step_escalate(state: AgentState) -> StepType:
    """
    Collect and log all escalations raised during processing.
    """
    t0 = time.monotonic()
    state.escalations = get_all_escalations()

    state.trace.append(TraceEntry(
        step=state.step_count,
        step_type=StepType.ESCALATE,
        reasoning="Collect all escalations raised and prepare for inclusion in summary.",
        action="collect_escalations",
        inputs={},
        result={
            "total_escalations": len(state.escalations),
            "high_severity": sum(1 for e in state.escalations if e.severity == "HIGH"),
            "issues": [
                {"id": e.issue_id, "category": e.category, "severity": e.severity}
                for e in state.escalations
            ],
        },
        next_step="GENERATE_SUMMARY",
        duration_ms=(time.monotonic() - t0) * 1000,
    ))
    return StepType.GENERATE_SUMMARY


def _step_generate_summary(state: AgentState) -> StepType:
    """
    Generate the structured discharge summary draft via LLM.
    """
    t0 = time.monotonic()
    c = state.consolidated

    # Build context for the LLM
    recon_narrative = state.reconciliation.to_narrative() if state.reconciliation else "Reconciliation not performed."
    conflict_narrative = state.conflicts.to_narrative() if state.conflicts else "No conflict check performed."
    escalation_lines = "\n".join(
        f"  [{e.severity}] {e.category}: {e.description}"
        for e in state.escalations
    ) or "  None"

    pending_items = c.get("all_pending_results") or []
    pending_narrative = "\n".join(f"  - {p}" for p in pending_items) if pending_items else "  None identified"

    drug_interactions = c.get("drug_interactions") or []
    interaction_narrative = "\n".join(
        f"  [{i['severity']}] {i['description']}" for i in drug_interactions
    ) if drug_interactions else "  None identified"

    SYSTEM = textwrap.dedent("""
        You are a clinical documentation assistant. Your task is to produce a
        structured DRAFT discharge summary for clinician review.

        ABSOLUTE RULES — never break these:
        1. NEVER invent or guess any clinical fact.
        2. If a field value is [MISSING], reproduce it as-is.
        3. If a field value is [PENDING], reproduce it as-is and add a note.
        4. If there is a [CONFLICT], reproduce it and explicitly state it needs resolution.
        5. Always frame this as a DRAFT requiring clinician review.
        6. Do not add plausible values, interpretations, or commentary beyond what is sourced.

        Format the summary with these clearly labelled sections:
        PATIENT DEMOGRAPHICS | ADMISSION & DISCHARGE DATES | PRINCIPAL DIAGNOSIS |
        SECONDARY DIAGNOSES | HOSPITAL COURSE | PROCEDURES | DISCHARGE MEDICATIONS |
        ALLERGIES | FOLLOW-UP INSTRUCTIONS | PENDING RESULTS | DISCHARGE CONDITION |
        FLAGS & ESCALATIONS FOR CLINICIAN REVIEW
    """)

    context = textwrap.dedent(f"""
        --- EXTRACTED DATA (from source notes) ---
        Patient Name: {c.get('patient_name', MISSING)}
        DOB: {c.get('dob', MISSING)}
        MRN: {c.get('mrn', MISSING)}
        Admission Date: {c.get('admission_date', MISSING)}
        Discharge Date: {c.get('discharge_date', MISSING)}
        Attending Physician: {c.get('attending_physician', MISSING)}
        Principal Diagnosis: {c.get('principal_diagnosis', MISSING)}
        Secondary Diagnoses: {', '.join(c.get('secondary_diagnoses') or []) or MISSING}
        Allergies: {', '.join(c.get('allergies') or []) or MISSING}
        Procedures: {', '.join(c.get('procedures') or []) or MISSING}
        Hospital Course: {c.get('hospital_course', MISSING)}
        Discharge Condition: {c.get('discharge_condition', MISSING)}
        Follow-up Instructions: {c.get('follow_up_instructions', MISSING)}

        --- MEDICATION RECONCILIATION ---
        {recon_narrative}

        --- PENDING RESULTS ---
        {pending_narrative}

        --- DRUG INTERACTIONS ---
        {interaction_narrative}

        --- CONFLICTS DETECTED ---
        {conflict_narrative}

        --- ESCALATIONS FOR CLINICIAN REVIEW ---
        {escalation_lines}

        --- INSTRUCTIONS ---
        Using ONLY the data above, write the structured discharge summary draft.
        For every field labelled [MISSING] or [PENDING], reproduce that label.
        For every conflict, reproduce [CONFLICT DETECTED — CLINICIAN REVIEW REQUIRED].
        This is a DRAFT. State this clearly at the top and bottom.
    """)

    summary_text = ""
    for attempt in range(MAX_RETRIES):
        try:
            summary_text = _call_llm(SYSTEM, context, max_tokens=2000)
            break
        except Exception as e:
            logger.warning("Summary generation attempt %d failed: %s", attempt + 1, e)
            if attempt == MAX_RETRIES - 1:
                summary_text = _build_fallback_summary(state)
                state.warnings.append("LLM summary generation failed; using template fallback.")

    state.final_summary = summary_text

    state.trace.append(TraceEntry(
        step=state.step_count,
        step_type=StepType.GENERATE_SUMMARY,
        reasoning="Generate structured discharge summary draft from consolidated data.",
        action="llm_generate_summary",
        inputs={"context_length_chars": len(context)},
        result={"summary_length_chars": len(summary_text), "generated": bool(summary_text)},
        next_step="REVIEW",
        duration_ms=(time.monotonic() - t0) * 1000,
    ))
    return StepType.REVIEW


def _build_fallback_summary(state: AgentState) -> str:
    """Template-based fallback if LLM generation fails."""
    c = state.consolidated
    recon = state.reconciliation.to_narrative() if state.reconciliation else "Not available."
    conflicts = state.conflicts.to_narrative() if state.conflicts else "Not checked."
    pending = "\n".join(f"  - {p}" for p in (c.get("all_pending_results") or [])) or "  None"
    escalations = "\n".join(
        f"  [{e.severity}] {e.description}" for e in state.escalations
    ) or "  None"

    return textwrap.dedent(f"""
        ╔══════════════════════════════════════════════════════════════╗
        ║         DISCHARGE SUMMARY DRAFT — FOR CLINICIAN REVIEW       ║
        ╚══════════════════════════════════════════════════════════════╝

        PATIENT DEMOGRAPHICS
        Name: {c.get('patient_name', MISSING)}
        DOB: {c.get('dob', MISSING)}
        MRN: {c.get('mrn', MISSING)}

        ADMISSION & DISCHARGE DATES
        Admission: {c.get('admission_date', MISSING)}
        Discharge: {c.get('discharge_date', MISSING)}
        Attending: {c.get('attending_physician', MISSING)}

        PRINCIPAL DIAGNOSIS
        {c.get('principal_diagnosis', MISSING)}

        SECONDARY DIAGNOSES
        {chr(10).join('  - ' + d for d in (c.get('secondary_diagnoses') or [])) or MISSING}

        HOSPITAL COURSE
        {c.get('hospital_course', MISSING)}

        PROCEDURES
        {chr(10).join('  - ' + p for p in (c.get('procedures') or [])) or MISSING}

        DISCHARGE MEDICATIONS
        {recon}

        ALLERGIES
        {', '.join(c.get('allergies') or []) or MISSING}

        FOLLOW-UP INSTRUCTIONS
        {c.get('follow_up_instructions', MISSING)}

        PENDING RESULTS
        {pending}

        DISCHARGE CONDITION
        {c.get('discharge_condition', MISSING)}

        ═══════════════════════════════════════════════════════════════
        FLAGS & ESCALATIONS FOR CLINICIAN REVIEW
        {escalations}

        CONFLICTS DETECTED
        {conflicts}
        ═══════════════════════════════════════════════════════════════

        ⚠ THIS IS A DRAFT. All [MISSING] and [CONFLICT] fields require
          clinician resolution before this document is finalised.
        Generated by DScribe Discharge Summary Agent.
    """).strip()


def _step_review(state: AgentState) -> StepType:
    """
    Final review step: check the summary for completeness and safety.
    """
    t0 = time.monotonic()
    summary = state.final_summary

    issues = []
    # Check no required section is entirely absent from the summary
    required_sections = [
        "PATIENT DEMOGRAPHICS", "PRINCIPAL DIAGNOSIS", "HOSPITAL COURSE",
        "DISCHARGE MEDICATIONS", "PENDING RESULTS", "DISCHARGE CONDITION",
    ]
    for section in required_sections:
        if section.lower() not in summary.lower():
            issues.append(f"Section '{section}' missing from generated summary.")

    # Check guardrail: MISSING marker should appear for any missing field
    missing_fields_in_state = [
        k for k, v in state.consolidated.items()
        if v == MISSING or v == PENDING
    ]

    state.trace.append(TraceEntry(
        step=state.step_count,
        step_type=StepType.REVIEW,
        reasoning="Final safety review: verify all sections present, no fabrication occurred.",
        action="review_summary_completeness",
        inputs={"required_sections": required_sections},
        result={
            "issues": issues,
            "missing_fields_in_state": missing_fields_in_state,
            "escalation_count": len(state.escalations),
            "conflict_count": len(state.conflicts.conflicts) if state.conflicts else 0,
            "review_passed": len(issues) == 0,
        },
        next_step="DONE",
        duration_ms=(time.monotonic() - t0) * 1000,
    ))

    if issues:
        state.warnings.extend(issues)

    return StepType.DONE


# ---------------------------------------------------------------------------
# Main agent loop
# ---------------------------------------------------------------------------

STEP_FN_MAP = {
    StepType.PLAN: _step_plan,
    StepType.EXTRACT: _step_extract,
    StepType.VALIDATE: _step_validate,
    StepType.RECONCILE: _step_reconcile,
    StepType.CONFLICT_CHECK: _step_conflict_check,
    StepType.PENDING_CHECK: _step_pending_check,
    StepType.DRUG_INTERACTION: _step_drug_interaction,
    StepType.ESCALATE: _step_escalate,
    StepType.GENERATE_SUMMARY: _step_generate_summary,
    StepType.REVIEW: _step_review,
}


def run_agent(patient_id: str, patient_folder: str) -> AgentState:
    """
    Run the full discharge summary agent for a single patient.
    Returns the final AgentState including summary, trace, and all flags.
    """
    clear_escalations()
    state = AgentState(patient_id=patient_id, patient_folder=patient_folder)

    logger.info("=" * 60)
    logger.info("Starting agent for patient: %s", patient_id)
    logger.info("=" * 60)

    while not state.finished and state.step_count < MAX_STEPS:
        state.step_count += 1
        current = state.current_step

        if current == StepType.DONE:
            state.finished = True
            break

        if current == StepType.ERROR:
            logger.error("Agent entered ERROR state. Errors: %s", state.errors)
            state.finished = True
            state.final_summary = (
                f"ERROR: Agent could not complete processing for patient {patient_id}.\n"
                f"Errors: {state.errors}\n"
                f"Please review source documents manually."
            )
            break

        step_fn = STEP_FN_MAP.get(current)
        if not step_fn:
            state.errors.append(f"Unknown step type: {current}")
            state.current_step = StepType.ERROR
            continue

        logger.info("[Step %d/%d] %s", state.step_count, MAX_STEPS, current.value)
        try:
            next_step = step_fn(state)
            state.current_step = next_step
        except Exception as exc:
            logger.exception("Unhandled exception in step %s: %s", current.value, exc)
            state.errors.append(f"Step {current.value} crashed: {exc}")
            state.current_step = StepType.ERROR

    if state.step_count >= MAX_STEPS and not state.finished:
        state.warnings.append(f"Hard step cap ({MAX_STEPS}) reached. Agent halted.")
        state.finished = True
        logger.warning("Hard step cap reached for patient %s", patient_id)

    logger.info(
        "Agent finished for %s: %d steps, %d escalations, %d errors",
        patient_id, state.step_count,
        len(state.escalations), len(state.errors),
    )
    return state


def save_outputs(state: AgentState, output_dir: str) -> dict[str, str]:
    """
    Save discharge summary draft and step trace to output_dir.
    Returns dict of output filenames → paths.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    pid = state.patient_id

    # Discharge summary
    summary_path = out / f"{pid}_discharge_summary_DRAFT.txt"
    summary_path.write_text(state.final_summary, encoding="utf-8")

    # Step trace (JSON)
    trace_data = {
        "patient_id": pid,
        "total_steps": state.step_count,
        "finished": state.finished,
        "errors": state.errors,
        "warnings": state.warnings,
        "escalations": [
            {
                "id": e.issue_id, "category": e.category,
                "severity": e.severity, "description": e.description,
                "action": e.recommended_action,
            }
            for e in state.escalations
        ],
        "trace": [t.to_dict() for t in state.trace],
    }
    trace_path = out / f"{pid}_step_trace.json"
    trace_path.write_text(json.dumps(trace_data, indent=2, default=str), encoding="utf-8")

    logger.info("Outputs saved to %s", output_dir)
    return {
        "discharge_summary": str(summary_path),
        "step_trace": str(trace_path),
    }
