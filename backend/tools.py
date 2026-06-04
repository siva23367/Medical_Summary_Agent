"""
tools.py
--------
Mock external tool implementations used by the agent loop.

Tools provided:
  1. drug_interaction_lookup  — checks for known drug-drug interactions
  2. escalate_to_clinician    — flags an issue for mandatory clinician review
  3. lab_reference_lookup     — checks if a lab value is within normal range
  4. allergy_cross_reactivity — checks for cross-reactive allergens

In production these would call real APIs/services.  Here they are implemented
as rule-based mocks with a realistic set of known interactions and reference
ranges so the agent can be demonstrated end-to-end without live external calls.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Return types
# ---------------------------------------------------------------------------

@dataclass
class ToolResult:
    success: bool
    tool_name: str
    input_params: dict
    output: dict
    error: Optional[str] = None
    latency_ms: float = 0.0

    def to_trace(self) -> dict:
        return {
            "tool": self.tool_name,
            "success": self.success,
            "inputs": self.input_params,
            "output": self.output,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 1),
        }


@dataclass
class Escalation:
    issue_id: str
    category: str       # "medication_conflict" | "missing_data" | "interaction" | "conflict"
    severity: str       # "HIGH" | "MEDIUM" | "LOW"
    description: str
    recommended_action: str
    source_notes: list[str] = field(default_factory=list)
    acknowledged: bool = False


# ---------------------------------------------------------------------------
# Shared state (in-memory; would be a DB in production)
# ---------------------------------------------------------------------------

_escalations: list[Escalation] = []
_escalation_counter = 0


# ---------------------------------------------------------------------------
# Known drug interactions database (subset — representative mock)
# ---------------------------------------------------------------------------

_INTERACTIONS: list[dict] = [
    {
        "drugs": {"warfarin", "aspirin"},
        "severity": "HIGH",
        "description": "Warfarin + Aspirin: increased bleeding risk. Monitor INR closely.",
        "action": "Consider GI protection; monitor INR; confirm intent with prescriber.",
    },
    {
        "drugs": {"warfarin", "ibuprofen"},
        "severity": "HIGH",
        "description": "Warfarin + Ibuprofen: significantly increased bleeding risk.",
        "action": "Avoid combination if possible. Use paracetamol as alternative.",
    },
    {
        "drugs": {"metformin", "contrast"},
        "severity": "HIGH",
        "description": "Metformin + IV contrast: risk of lactic acidosis.",
        "action": "Hold metformin 48h before and after iodinated contrast.",
    },
    {
        "drugs": {"ssri", "tramadol"},
        "severity": "HIGH",
        "description": "SSRI + Tramadol: risk of serotonin syndrome.",
        "action": "Avoid combination; if required, use lowest effective doses and monitor.",
    },
    {
        "drugs": {"ace inhibitor", "potassium"},
        "severity": "MEDIUM",
        "description": "ACE inhibitor + Potassium supplements: risk of hyperkalaemia.",
        "action": "Monitor serum potassium levels.",
    },
    {
        "drugs": {"simvastatin", "amiodarone"},
        "severity": "HIGH",
        "description": "Simvastatin + Amiodarone: increased risk of myopathy/rhabdomyolysis.",
        "action": "Limit simvastatin dose to 20 mg/day or switch statin.",
    },
    {
        "drugs": {"methotrexate", "nsaid"},
        "severity": "HIGH",
        "description": "Methotrexate + NSAID: reduced methotrexate clearance, toxicity risk.",
        "action": "Avoid concurrent use; if necessary, monitor methotrexate levels closely.",
    },
    {
        "drugs": {"digoxin", "amiodarone"},
        "severity": "HIGH",
        "description": "Digoxin + Amiodarone: elevated digoxin levels, risk of toxicity.",
        "action": "Reduce digoxin dose by ~50%; monitor levels and ECG.",
    },
    {
        "drugs": {"ciprofloxacin", "theophylline"},
        "severity": "MEDIUM",
        "description": "Ciprofloxacin + Theophylline: increased theophylline levels.",
        "action": "Monitor theophylline levels; consider dose reduction.",
    },
    {
        "drugs": {"lithium", "nsaid"},
        "severity": "HIGH",
        "description": "Lithium + NSAID: reduced lithium clearance, toxicity risk.",
        "action": "Monitor lithium levels closely; avoid NSAIDs if possible.",
    },
]

# NSAID synonyms for matching
_NSAID_NAMES = {"ibuprofen", "naproxen", "diclofenac", "indomethacin", "celecoxib", "aspirin"}
_SSRI_NAMES = {"fluoxetine", "sertraline", "escitalopram", "citalopram", "paroxetine", "fluvoxamine"}
_ACE_NAMES = {"lisinopril", "ramipril", "enalapril", "perindopril", "captopril", "quinapril"}


def _normalise_drug_for_lookup(name: str) -> set[str]:
    """Return a set of labels this drug matches in the interaction DB."""
    n = name.strip().lower()
    labels = {n}
    if n in _NSAID_NAMES:
        labels.add("nsaid")
    if n in _SSRI_NAMES:
        labels.add("ssri")
    if n in _ACE_NAMES:
        labels.add("ace inhibitor")
    return labels


# ---------------------------------------------------------------------------
# Tool 1: Drug interaction lookup
# ---------------------------------------------------------------------------

def drug_interaction_lookup(
    medications: list[str],
    simulate_failure_rate: float = 0.05,   # 5% chance of transient failure for realism
) -> ToolResult:
    """
    Check a list of medication names for known interactions.
    Returns a ToolResult with any interactions found.
    """
    start = time.monotonic()
    params = {"medications": medications}

    # Simulate occasional transient failures
    if random.random() < simulate_failure_rate:
        return ToolResult(
            success=False,
            tool_name="drug_interaction_lookup",
            input_params=params,
            output={},
            error="Service temporarily unavailable (simulated transient failure).",
            latency_ms=(time.monotonic() - start) * 1000,
        )

    # Build set of normalised labels per medication
    med_labels: list[set[str]] = [_normalise_drug_for_lookup(m) for m in medications]

    found_interactions = []
    for interaction in _INTERACTIONS:
        interaction_drugs = interaction["drugs"]
        # Check if any pair of provided meds matches this interaction
        matched_pair = []
        for labels in med_labels:
            for idrg in interaction_drugs:
                if idrg in labels:
                    matched_pair.append(idrg)
                    break
        if len(matched_pair) >= 2:
            found_interactions.append({
                "drugs_involved": list(interaction_drugs),
                "severity": interaction["severity"],
                "description": interaction["description"],
                "recommended_action": interaction["action"],
            })

    latency = (time.monotonic() - start) * 1000
    return ToolResult(
        success=True,
        tool_name="drug_interaction_lookup",
        input_params=params,
        output={
            "interactions_found": len(found_interactions),
            "interactions": found_interactions,
            "checked_medications": medications,
        },
        latency_ms=latency,
    )


# ---------------------------------------------------------------------------
# Tool 2: Escalate to clinician
# ---------------------------------------------------------------------------

def escalate_to_clinician(
    category: str,
    severity: str,
    description: str,
    recommended_action: str,
    source_notes: Optional[list[str]] = None,
) -> ToolResult:
    """
    Record an escalation flag for clinician review.
    In production this would push a notification / create a task in the EMR.
    """
    global _escalation_counter
    start = time.monotonic()

    _escalation_counter += 1
    issue_id = f"ESC-{_escalation_counter:04d}"

    escalation = Escalation(
        issue_id=issue_id,
        category=category,
        severity=severity,
        description=description,
        recommended_action=recommended_action,
        source_notes=source_notes or [],
    )
    _escalations.append(escalation)

    logger.warning(
        "[ESCALATION %s] %s — %s: %s",
        issue_id, severity, category, description
    )

    latency = (time.monotonic() - start) * 1000
    return ToolResult(
        success=True,
        tool_name="escalate_to_clinician",
        input_params={
            "category": category,
            "severity": severity,
            "description": description,
        },
        output={
            "issue_id": issue_id,
            "status": "flagged_for_review",
            "message": f"Issue {issue_id} recorded for clinician review.",
        },
        latency_ms=latency,
    )


def get_all_escalations() -> list[Escalation]:
    """Return all escalations raised so far (for summary generation)."""
    return list(_escalations)


def clear_escalations() -> None:
    """Reset escalation list (call between patients)."""
    global _escalations, _escalation_counter
    _escalations = []
    _escalation_counter = 0


# ---------------------------------------------------------------------------
# Tool 3: Lab reference ranges
# ---------------------------------------------------------------------------

_LAB_RANGES = {
    "haemoglobin":  {"male": (130, 175), "female": (115, 155), "unit": "g/L"},
    "hemoglobin":   {"male": (130, 175), "female": (115, 155), "unit": "g/L"},
    "wbc":          {"low": 4.0,  "high": 11.0,  "unit": "x10^9/L"},
    "white blood cell count": {"low": 4.0, "high": 11.0, "unit": "x10^9/L"},
    "platelets":    {"low": 150,  "high": 400,   "unit": "x10^9/L"},
    "sodium":       {"low": 135,  "high": 145,   "unit": "mmol/L"},
    "potassium":    {"low": 3.5,  "high": 5.0,   "unit": "mmol/L"},
    "creatinine":   {"male": (60, 110), "female": (45, 90), "unit": "umol/L"},
    "urea":         {"low": 2.5,  "high": 7.8,   "unit": "mmol/L"},
    "bun":          {"low": 7.0,  "high": 20.0,  "unit": "mg/dL"},
    "glucose":      {"low": 4.0,  "high": 7.8,   "unit": "mmol/L"},
    "hba1c":        {"low": 4.0,  "high": 5.6,   "unit": "%"},
    "inr":          {"low": 0.9,  "high": 1.2,   "unit": "ratio"},
    "alt":          {"low": 7.0,  "high": 56.0,  "unit": "U/L"},
    "ast":          {"low": 10.0, "high": 40.0,  "unit": "U/L"},
    "troponin":     {"low": 0.0,  "high": 0.04,  "unit": "ng/mL"},
    "bnp":          {"low": 0.0,  "high": 100.0, "unit": "pg/mL"},
    "crp":          {"low": 0.0,  "high": 10.0,  "unit": "mg/L"},
}

def lab_reference_lookup(
    test_name: str,
    value: float,
    sex: Optional[str] = None,   # "male" | "female" | None
) -> ToolResult:
    """
    Check whether a lab value is within the reference range.
    Returns a ToolResult with normal/abnormal classification.
    """
    start = time.monotonic()
    params = {"test_name": test_name, "value": value, "sex": sex}
    key = test_name.strip().lower()

    if key not in _LAB_RANGES:
        return ToolResult(
            success=True,
            tool_name="lab_reference_lookup",
            input_params=params,
            output={
                "test": test_name,
                "value": value,
                "status": "UNKNOWN",
                "message": f"No reference range available for '{test_name}'.",
            },
            latency_ms=(time.monotonic() - start) * 1000,
        )

    ref = _LAB_RANGES[key]
    unit = ref.get("unit", "")

    # Sex-specific ranges
    if sex and sex.lower() in ref:
        low, high = ref[sex.lower()]
    elif "low" in ref and "high" in ref:
        low, high = ref["low"], ref["high"]
    else:
        # Sex required but not provided
        return ToolResult(
            success=True,
            tool_name="lab_reference_lookup",
            input_params=params,
            output={
                "test": test_name,
                "value": value,
                "status": "UNKNOWN",
                "message": f"Sex required to interpret '{test_name}' — not provided.",
            },
            latency_ms=(time.monotonic() - start) * 1000,
        )

    if value < low:
        status = "LOW"
        message = f"{test_name}: {value} {unit} is BELOW normal range ({low}–{high} {unit})."
    elif value > high:
        status = "HIGH"
        message = f"{test_name}: {value} {unit} is ABOVE normal range ({low}–{high} {unit})."
    else:
        status = "NORMAL"
        message = f"{test_name}: {value} {unit} is within normal range ({low}–{high} {unit})."

    return ToolResult(
        success=True,
        tool_name="lab_reference_lookup",
        input_params=params,
        output={
            "test": test_name,
            "value": value,
            "unit": unit,
            "status": status,
            "normal_range": f"{low}–{high}",
            "message": message,
        },
        latency_ms=(time.monotonic() - start) * 1000,
    )


# ---------------------------------------------------------------------------
# Tool 4: Allergy cross-reactivity check
# ---------------------------------------------------------------------------

_CROSS_REACTIVITY = {
    "penicillin":        ["amoxicillin", "ampicillin", "piperacillin", "cephalosporins (10% cross-reactivity)"],
    "sulfonamides":      ["trimethoprim-sulfamethoxazole", "furosemide (weak)", "thiazides (weak)"],
    "codeine":           ["morphine", "hydrocodone", "oxycodone (cross-sensitivity possible)"],
    "aspirin":           ["nsaids (ibuprofen, naproxen, diclofenac) — cross-sensitivity in aspirin-sensitive patients"],
    "cephalosporins":    ["penicillin (10% cross-reactivity)"],
    "iodine":            ["contrast media (if true iodine allergy — verify with allergy specialist)"],
}

def allergy_cross_reactivity_check(
    allergy: str,
    prescribed_medications: list[str],
) -> ToolResult:
    """
    Check whether any prescribed medication may cross-react with a known allergy.
    """
    start = time.monotonic()
    params = {"allergy": allergy, "medications": prescribed_medications}
    key = allergy.strip().lower()

    warnings = []
    cross_reactions = _CROSS_REACTIVITY.get(key, [])

    for med in prescribed_medications:
        med_lower = med.strip().lower()
        for cr in cross_reactions:
            if med_lower in cr or any(m in cr for m in med_lower.split()):
                warnings.append({
                    "medication": med,
                    "allergy": allergy,
                    "warning": cr,
                    "severity": "HIGH",
                })

    latency = (time.monotonic() - start) * 1000
    return ToolResult(
        success=True,
        tool_name="allergy_cross_reactivity_check",
        input_params=params,
        output={
            "allergy_checked": allergy,
            "cross_reactivity_warnings": warnings,
            "count": len(warnings),
        },
        latency_ms=latency,
    )
