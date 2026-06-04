"""
med_reconciliation.py
---------------------
Compares admission medications vs. discharge medications and surfaces:
  - Newly added medications (with/without documented reason)
  - Stopped medications (with/without documented reason)
  - Dose/frequency changes (with/without documented reason)
  - Medications whose reason is documented vs. unexplained (flagged for clinician)

All output is structured for the agent loop to embed directly into the
discharge summary draft.  Nothing is silently resolved — unexplained
changes are always flagged.
"""

from __future__ import annotations

import re
import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class ChangeType(str, Enum):
    ADDED = "ADDED"
    STOPPED = "STOPPED"
    DOSE_CHANGED = "DOSE_CHANGED"
    FREQUENCY_CHANGED = "FREQUENCY_CHANGED"
    UNCHANGED = "UNCHANGED"
    UNKNOWN = "UNKNOWN"


@dataclass
class Medication:
    name: str                        # normalised lowercase name
    raw_name: str                    # as found in the document
    dose: Optional[str] = None       # e.g. "10 mg"
    frequency: Optional[str] = None  # e.g. "twice daily"
    route: Optional[str] = None      # e.g. "oral"
    indication: Optional[str] = None # documented reason if present
    source_note: str = ""            # which document it came from


@dataclass
class MedChange:
    change_type: ChangeType
    medication_name: str
    admission_med: Optional[Medication]
    discharge_med: Optional[Medication]
    reason_documented: bool
    flag_for_review: bool
    flag_message: str = ""
    detail: str = ""

    def to_dict(self) -> dict:
        return {
            "change_type": self.change_type.value,
            "medication": self.medication_name,
            "admission": _med_summary(self.admission_med),
            "discharge": _med_summary(self.discharge_med),
            "reason_documented": self.reason_documented,
            "flag_for_review": self.flag_for_review,
            "flag_message": self.flag_message,
            "detail": self.detail,
        }


@dataclass
class ReconciliationReport:
    unchanged: list[MedChange] = field(default_factory=list)
    added: list[MedChange] = field(default_factory=list)
    stopped: list[MedChange] = field(default_factory=list)
    changed: list[MedChange] = field(default_factory=list)
    flagged: list[MedChange] = field(default_factory=list)   # subset needing review
    warnings: list[str] = field(default_factory=list)

    @property
    def all_changes(self) -> list[MedChange]:
        return self.added + self.stopped + self.changed

    def to_dict(self) -> dict:
        return {
            "summary": {
                "unchanged": len(self.unchanged),
                "added": len(self.added),
                "stopped": len(self.stopped),
                "changed": len(self.changed),
                "flagged_for_review": len(self.flagged),
            },
            "added": [c.to_dict() for c in self.added],
            "stopped": [c.to_dict() for c in self.stopped],
            "changed": [c.to_dict() for c in self.changed],
            "flagged_for_review": [c.to_dict() for c in self.flagged],
            "unchanged": [c.to_dict() for c in self.unchanged],
            "warnings": self.warnings,
        }

    def to_narrative(self) -> str:
        """Human-readable narrative for the discharge summary."""
        lines: list[str] = []

        if self.added:
            lines.append("NEWLY STARTED MEDICATIONS:")
            for c in self.added:
                reason = f" (Reason: {c.discharge_med.indication})" if c.reason_documented else " [⚠ REASON NOT DOCUMENTED — FLAG FOR REVIEW]"
                lines.append(f"  + {c.medication_name.title()} {_dose_str(c.discharge_med)}{reason}")

        if self.stopped:
            lines.append("\nSTOPPED MEDICATIONS:")
            for c in self.stopped:
                reason = f" (Reason: {c.admission_med.indication})" if c.reason_documented else " [⚠ REASON NOT DOCUMENTED — FLAG FOR REVIEW]"
                lines.append(f"  - {c.medication_name.title()} {_dose_str(c.admission_med)}{reason}")

        if self.changed:
            lines.append("\nCHANGED MEDICATIONS:")
            for c in self.changed:
                reason = f" (Reason: {c.detail})" if c.reason_documented else " [⚠ REASON NOT DOCUMENTED — FLAG FOR REVIEW]"
                adm = _dose_str(c.admission_med)
                dis = _dose_str(c.discharge_med)
                lines.append(f"  ~ {c.medication_name.title()}: {adm} → {dis}{reason}")

        if self.unchanged:
            lines.append("\nCONTINUED MEDICATIONS:")
            for c in self.unchanged:
                lines.append(f"  = {c.medication_name.title()} {_dose_str(c.discharge_med)}")

        if not lines:
            lines.append("No medications to reconcile.")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Common dose keywords for normalisation
_FREQ_PATTERNS = [
    (r"\bonce\s*daily\b|\bqd\b|\bq24h\b", "once daily"),
    (r"\btwice\s*daily\b|\bbid\b|\bb\.i\.d\b", "twice daily"),
    (r"\bthree\s*times\s*daily\b|\btid\b|\bt\.i\.d\b", "three times daily"),
    (r"\bfour\s*times\s*daily\b|\bqid\b", "four times daily"),
    (r"\bevery\s*(\d+)\s*hours?\b|\bq(\d+)h\b", None),   # kept as-is
    (r"\bas\s*needed\b|\bprn\b", "as needed"),
    (r"\bat\s*bedtime\b|\bqhs\b", "at bedtime"),
]

def _normalise_name(name: str) -> str:
    """Lowercase, strip trailing punctuation/whitespace."""
    return re.sub(r"[^\w\s\-]", "", name).strip().lower()

def _normalise_freq(freq: Optional[str]) -> Optional[str]:
    if not freq:
        return None
    f = freq.strip().lower()
    for pattern, replacement in _FREQ_PATTERNS:
        if re.search(pattern, f):
            return replacement if replacement else f
    return f

def _med_summary(med: Optional[Medication]) -> Optional[str]:
    if med is None:
        return None
    parts = [med.raw_name]
    if med.dose:
        parts.append(med.dose)
    if med.frequency:
        parts.append(med.frequency)
    if med.route:
        parts.append(f"({med.route})")
    return " ".join(parts)

def _dose_str(med: Optional[Medication]) -> str:
    if med is None:
        return ""
    parts = []
    if med.dose:
        parts.append(med.dose)
    if med.frequency:
        parts.append(med.frequency)
    return " ".join(parts)

def _doses_differ(a: Optional[str], b: Optional[str]) -> bool:
    if a is None and b is None:
        return False
    if a is None or b is None:
        return True
    return a.strip().lower() != b.strip().lower()


# ---------------------------------------------------------------------------
# Core reconciliation logic
# ---------------------------------------------------------------------------

def reconcile_medications(
    admission_meds: list[Medication],
    discharge_meds: list[Medication],
) -> ReconciliationReport:
    """
    Compare admission vs discharge medication lists and produce a
    ReconciliationReport.  All unexplained changes are flagged.
    """
    report = ReconciliationReport()

    if not admission_meds and not discharge_meds:
        report.warnings.append("Both admission and discharge medication lists are empty.")
        return report

    if not admission_meds:
        report.warnings.append("Admission medication list is missing — cannot perform full reconciliation.")
    if not discharge_meds:
        report.warnings.append("Discharge medication list is missing — cannot perform full reconciliation.")

    # Build lookup dicts keyed by normalised name
    adm_map: dict[str, Medication] = {_normalise_name(m.name): m for m in admission_meds}
    dis_map: dict[str, Medication] = {_normalise_name(m.name): m for m in discharge_meds}

    all_names = set(adm_map) | set(dis_map)

    for name in sorted(all_names):
        adm = adm_map.get(name)
        dis = dis_map.get(name)

        # --- Added at discharge ---
        if adm is None and dis is not None:
            reason_doc = bool(dis.indication)
            change = MedChange(
                change_type=ChangeType.ADDED,
                medication_name=name,
                admission_med=None,
                discharge_med=dis,
                reason_documented=reason_doc,
                flag_for_review=not reason_doc,
                flag_message="" if reason_doc else (
                    f"⚠ {dis.raw_name} was ADDED at discharge with no documented reason. "
                    "Clinician review required."
                ),
            )
            report.added.append(change)
            if change.flag_for_review:
                report.flagged.append(change)

        # --- Stopped at discharge ---
        elif adm is not None and dis is None:
            reason_doc = bool(adm.indication)
            change = MedChange(
                change_type=ChangeType.STOPPED,
                medication_name=name,
                admission_med=adm,
                discharge_med=None,
                reason_documented=reason_doc,
                flag_for_review=not reason_doc,
                flag_message="" if reason_doc else (
                    f"⚠ {adm.raw_name} was STOPPED at discharge with no documented reason. "
                    "Clinician review required."
                ),
            )
            report.stopped.append(change)
            if change.flag_for_review:
                report.flagged.append(change)

        # --- Present in both — check for changes ---
        elif adm is not None and dis is not None:
            dose_changed = _doses_differ(adm.dose, dis.dose)
            freq_changed = _doses_differ(
                _normalise_freq(adm.frequency),
                _normalise_freq(dis.frequency),
            )

            if not dose_changed and not freq_changed:
                report.unchanged.append(MedChange(
                    change_type=ChangeType.UNCHANGED,
                    medication_name=name,
                    admission_med=adm,
                    discharge_med=dis,
                    reason_documented=True,
                    flag_for_review=False,
                ))
            else:
                change_desc_parts = []
                if dose_changed:
                    change_desc_parts.append(
                        f"dose {adm.dose or 'unknown'} → {dis.dose or 'unknown'}"
                    )
                if freq_changed:
                    change_desc_parts.append(
                        f"frequency {adm.frequency or 'unknown'} → {dis.frequency or 'unknown'}"
                    )
                detail = "; ".join(change_desc_parts)

                reason_doc = bool(dis.indication or adm.indication)
                change = MedChange(
                    change_type=ChangeType.DOSE_CHANGED if dose_changed else ChangeType.FREQUENCY_CHANGED,
                    medication_name=name,
                    admission_med=adm,
                    discharge_med=dis,
                    reason_documented=reason_doc,
                    flag_for_review=not reason_doc,
                    flag_message="" if reason_doc else (
                        f"⚠ {adm.raw_name} was CHANGED ({detail}) with no documented reason. "
                        "Clinician review required."
                    ),
                    detail=detail,
                )
                report.changed.append(change)
                if change.flag_for_review:
                    report.flagged.append(change)

    return report


# ---------------------------------------------------------------------------
# Lightweight parser for raw medication text blocks
# ---------------------------------------------------------------------------

def parse_medication_list(raw_text: str, source_note: str = "") -> list[Medication]:
    """
    Best-effort parser that extracts medications from a raw text block.
    Handles common formats:
      "Metformin 500 mg twice daily"
      "1. Lisinopril 10mg PO QD (for hypertension)"
      "- Aspirin 81 mg daily"
    Returns a list of Medication objects.  Unrecognised lines are skipped
    with a warning in the log.
    """
    meds: list[Medication] = []

    # Patterns
    dose_re = re.compile(
        r"(\d+\.?\d*)\s*(mg|mcg|g|ml|mmol|units?|iu|meq|mg/kg|mg/dl)\b",
        re.IGNORECASE,
    )
    freq_re = re.compile(
        r"\b(once\s*daily|twice\s*daily|three\s*times\s*daily|four\s*times\s*daily"
        r"|qd|bid|tid|qid|q\d+h|every\s*\d+\s*hours?|as\s*needed|prn|at\s*bedtime|qhs"
        r"|daily|weekly|monthly)\b",
        re.IGNORECASE,
    )
    route_re = re.compile(
        r"\b(oral|po|iv|intravenous|subcutaneous|sc|im|intramuscular"
        r"|topical|inhaled|sublingual|sl|rectal|pr|transdermal)\b",
        re.IGNORECASE,
    )
    indication_re = re.compile(
        r"(?:for|indication:|reason:)\s*([^;\n\)]{3,60})",
        re.IGNORECASE,
    )

    for raw_line in raw_text.splitlines():
        line = raw_line.strip()
        # Skip empty lines, headers, separator lines
        if not line or len(line) < 4:
            continue
        if re.match(r"^[-=*#]{3,}$", line):
            continue
        # Strip leading list markers
        line_clean = re.sub(r"^\d+[\.\)]\s*|^[-•*]\s*", "", line).strip()
        if not line_clean:
            continue

        # Extract components
        dose_match = dose_re.search(line_clean)
        freq_match = freq_re.search(line_clean)
        route_match = route_re.search(line_clean)
        indication_match = indication_re.search(line_clean)

        # Derive the medication name: everything before the first dose/number
        if dose_match:
            name_raw = line_clean[:dose_match.start()].strip()
        elif freq_match:
            name_raw = line_clean[:freq_match.start()].strip()
        else:
            # Take up to the first parenthesis or comma as the name
            name_raw = re.split(r"[,(]", line_clean)[0].strip()

        # Clean up the name
        name_raw = re.sub(r"\s+", " ", name_raw).strip(" -•:")
        if len(name_raw) < 2:
            logger.debug("Skipping unrecognisable medication line: %r", raw_line)
            continue

        med = Medication(
            name=_normalise_name(name_raw),
            raw_name=name_raw,
            dose=f"{dose_match.group(1)} {dose_match.group(2)}" if dose_match else None,
            frequency=freq_match.group(0).strip() if freq_match else None,
            route=route_match.group(0).strip() if route_match else None,
            indication=indication_match.group(1).strip() if indication_match else None,
            source_note=source_note,
        )
        meds.append(med)

    logger.debug("Parsed %d medications from %s", len(meds), source_note or "text block")
    return meds
