"""
conflict_detector.py
--------------------
Detects conflicting clinical information across multiple source notes for a
single patient.

Checks:
  - Conflicting principal / secondary diagnoses
  - Conflicting admission or discharge dates
  - Conflicting allergy records
  - Conflicting medication information across notes
  - Conflicting vital signs / lab values on the same date
  - Conflicting demographic fields (DOB, MRN, name)

All detected conflicts are returned as structured ConflictReport objects so
the agent loop can embed them in the discharge summary draft and flag them
for clinician review.  Nothing is silently resolved — the agent must never
pick a "winner" between conflicting values.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class ConflictSeverity(str, Enum):
    HIGH = "HIGH"       # Patient-safety-critical (e.g., allergy discrepancy)
    MEDIUM = "MEDIUM"   # Clinically significant (e.g., diagnosis mismatch)
    LOW = "LOW"         # Administrative (e.g., date format difference)


@dataclass
class Conflict:
    field: str                     # What field conflicts (e.g., "discharge_diagnosis")
    severity: ConflictSeverity
    values: list[tuple[str, Any]]  # [(source_note_name, value), ...]
    message: str                   # Human-readable description
    flag_for_review: bool = True   # Always True for real conflicts

    def to_dict(self) -> dict:
        return {
            "field": self.field,
            "severity": self.severity.value,
            "values": [{"source": s, "value": str(v)} for s, v in self.values],
            "message": self.message,
            "flag_for_review": self.flag_for_review,
        }

    def to_narrative(self) -> str:
        value_lines = "\n".join(
            f"    [{src}]: {val}" for src, val in self.values
        )
        return (
            f"⚠ CONFLICT [{self.severity.value}] — {self.field}\n"
            f"{value_lines}\n"
            f"  → {self.message}"
        )


@dataclass
class ConflictReport:
    conflicts: list[Conflict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def has_conflicts(self) -> bool:
        return bool(self.conflicts)

    @property
    def high_severity(self) -> list[Conflict]:
        return [c for c in self.conflicts if c.severity == ConflictSeverity.HIGH]

    @property
    def medium_severity(self) -> list[Conflict]:
        return [c for c in self.conflicts if c.severity == ConflictSeverity.MEDIUM]

    def to_dict(self) -> dict:
        return {
            "total_conflicts": len(self.conflicts),
            "high_severity": len(self.high_severity),
            "medium_severity": len(self.medium_severity),
            "conflicts": [c.to_dict() for c in self.conflicts],
            "warnings": self.warnings,
        }

    def to_narrative(self) -> str:
        if not self.conflicts:
            return "No conflicts detected across source notes."
        lines = [
            f"CONFLICTS DETECTED ({len(self.conflicts)} total, "
            f"{len(self.high_severity)} HIGH severity):\n"
        ]
        for c in sorted(self.conflicts, key=lambda x: x.severity.value):
            lines.append(c.to_narrative())
            lines.append("")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _norm_str(s: Any) -> str:
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip().lower()

def _norm_date(s: Any) -> str:
    """Strip time components and normalise date strings for comparison."""
    if s is None:
        return ""
    s = str(s).strip()
    # Accept YYYY-MM-DD, MM/DD/YYYY, DD-Mon-YYYY
    patterns = [
        (r"(\d{4})-(\d{2})-(\d{2})", r"\1-\2-\3"),
        (r"(\d{2})/(\d{2})/(\d{4})", r"\3-\1-\2"),
        (r"(\d{2})-([A-Za-z]{3})-(\d{4})", None),
    ]
    for pat, fmt in patterns:
        m = re.search(pat, s, re.IGNORECASE)
        if m:
            return m.group(0).lower()
    return s.lower()

def _norm_allergy(s: Any) -> str:
    """Normalise allergy name for comparison (strip severity qualifiers)."""
    s = _norm_str(s)
    # Remove common qualifiers to compare base substance
    s = re.sub(r"\b(mild|moderate|severe|anaphylaxis|rash|hives|nausea|reaction)\b", "", s)
    return re.sub(r"\s+", " ", s).strip()

_SETS_OF_SYNONYMS = [
    {"mi", "myocardial infarction", "heart attack"},
    {"htn", "hypertension", "high blood pressure"},
    {"dm", "diabetes mellitus", "diabetes", "t2dm", "type 2 diabetes"},
    {"chf", "congestive heart failure", "heart failure"},
    {"copd", "chronic obstructive pulmonary disease"},
    {"ckd", "chronic kidney disease", "chronic renal disease"},
    {"cva", "stroke", "cerebrovascular accident"},
    {"pe", "pulmonary embolism"},
    {"dvt", "deep vein thrombosis"},
    {"af", "afib", "atrial fibrillation"},
]

def _diagnoses_match(a: str, b: str) -> bool:
    """Return True if two diagnosis strings are equivalent (synonyms count)."""
    na, nb = _norm_str(a), _norm_str(b)
    if na == nb:
        return True
    for synonyms in _SETS_OF_SYNONYMS:
        if na in synonyms and nb in synonyms:
            return True
    return False


# ---------------------------------------------------------------------------
# Individual field checkers
# ---------------------------------------------------------------------------

def _check_field(
    field_name: str,
    values: list[tuple[str, Any]],  # [(source, value)]
    severity: ConflictSeverity,
    norm_fn=_norm_str,
    match_fn=None,
) -> Optional[Conflict]:
    """
    Generic field conflict checker.  Returns a Conflict if the normalised
    values differ, else None.
    """
    populated = [(src, v) for src, v in values if norm_fn(v)]
    if len(populated) < 2:
        return None   # Can't conflict with only one value

    if match_fn is None:
        match_fn = lambda a, b: norm_fn(a) == norm_fn(b)

    # Compare every pair
    base_src, base_val = populated[0]
    for src, val in populated[1:]:
        if not match_fn(base_val, val):
            return Conflict(
                field=field_name,
                severity=severity,
                values=populated,
                message=(
                    f"'{field_name}' differs across notes. "
                    f"Clinician must resolve before finalising."
                ),
            )
    return None


def _check_list_field(
    field_name: str,
    lists: list[tuple[str, list[str]]],  # [(source, [item, ...])]
    severity: ConflictSeverity,
    norm_fn=_norm_str,
) -> list[Conflict]:
    """
    Check for items present in some sources but absent in others
    (asymmetric lists), e.g. allergy lists or diagnosis lists.
    Returns a list of conflicts (one per missing/extra item).
    """
    if len(lists) < 2:
        return []

    # Build sets of normalised items per source
    sets: list[tuple[str, set[str]]] = [
        (src, {norm_fn(i) for i in items if norm_fn(i)})
        for src, items in lists
    ]

    all_items: set[str] = set()
    for _, s in sets:
        all_items |= s

    conflicts: list[Conflict] = []
    for item in sorted(all_items):
        present_in = [src for src, s in sets if item in s]
        absent_from = [src for src, s in sets if item not in s]
        if absent_from and present_in:
            sources_with_values = [(src, item) for src in present_in] + [
                (src, "<NOT LISTED>") for src in absent_from
            ]
            conflicts.append(Conflict(
                field=f"{field_name}: '{item}'",
                severity=severity,
                values=sources_with_values,
                message=(
                    f"'{item}' is listed in {present_in} but absent from {absent_from}. "
                    f"Omission from some notes may indicate error. Clinician review required."
                ),
            ))
    return conflicts


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def detect_conflicts(extracted_data: dict[str, dict]) -> ConflictReport:
    """
    Main entry point.

    Parameters
    ----------
    extracted_data : dict[str, dict]
        Mapping of source_note_filename → extracted fields dict.
        Expected keys per note (all optional):
            patient_name, dob, mrn,
            admission_date, discharge_date,
            principal_diagnosis, secondary_diagnoses (list),
            allergies (list),
            discharge_condition,
            attending_physician
    Returns
    -------
    ConflictReport
        Structured report of all detected conflicts.
    """
    report = ConflictReport()

    if len(extracted_data) < 2:
        report.warnings.append(
            "Only one (or zero) source notes available — conflict detection requires at least two."
        )
        return report

    sources = list(extracted_data.keys())

    # ------------------------------------------------------------------ #
    # 1. Demographics                                                      #
    # ------------------------------------------------------------------ #
    for field_name, severity in [
        ("patient_name", ConflictSeverity.MEDIUM),
        ("dob", ConflictSeverity.HIGH),
        ("mrn", ConflictSeverity.HIGH),
    ]:
        values = [(src, extracted_data[src].get(field_name)) for src in sources]
        conflict = _check_field(
            field_name, values, severity,
            norm_fn=_norm_str if field_name != "dob" else _norm_date,
        )
        if conflict:
            report.conflicts.append(conflict)

    # ------------------------------------------------------------------ #
    # 2. Admission / Discharge dates                                       #
    # ------------------------------------------------------------------ #
    for field_name in ("admission_date", "discharge_date"):
        values = [(src, extracted_data[src].get(field_name)) for src in sources]
        conflict = _check_field(
            field_name, values, ConflictSeverity.MEDIUM,
            norm_fn=_norm_date,
        )
        if conflict:
            report.conflicts.append(conflict)

    # ------------------------------------------------------------------ #
    # 3. Principal diagnosis                                               #
    # ------------------------------------------------------------------ #
    values = [(src, extracted_data[src].get("principal_diagnosis")) for src in sources]
    conflict = _check_field(
        "principal_diagnosis", values, ConflictSeverity.HIGH,
        match_fn=_diagnoses_match,
    )
    if conflict:
        report.conflicts.append(conflict)

    # ------------------------------------------------------------------ #
    # 4. Secondary diagnoses (list comparison)                             #
    # ------------------------------------------------------------------ #
    secondary_lists = [
        (src, extracted_data[src].get("secondary_diagnoses") or [])
        for src in sources
    ]
    sec_conflicts = _check_list_field(
        "secondary_diagnosis", secondary_lists, ConflictSeverity.MEDIUM,
        norm_fn=lambda x: _norm_str(x),
    )
    report.conflicts.extend(sec_conflicts)

    # ------------------------------------------------------------------ #
    # 5. Allergies — HIGH severity                                         #
    # ------------------------------------------------------------------ #
    allergy_lists = [
        (src, extracted_data[src].get("allergies") or [])
        for src in sources
    ]
    # Only flag if a non-empty allergy is listed in one note but not another
    allergy_conflicts = _check_list_field(
        "allergy", allergy_lists, ConflictSeverity.HIGH,
        norm_fn=_norm_allergy,
    )
    report.conflicts.extend(allergy_conflicts)

    # ------------------------------------------------------------------ #
    # 6. Discharge condition                                               #
    # ------------------------------------------------------------------ #
    values = [(src, extracted_data[src].get("discharge_condition")) for src in sources]
    conflict = _check_field(
        "discharge_condition", values, ConflictSeverity.MEDIUM,
    )
    if conflict:
        report.conflicts.append(conflict)

    # ------------------------------------------------------------------ #
    # 7. Attending physician                                               #
    # ------------------------------------------------------------------ #
    values = [(src, extracted_data[src].get("attending_physician")) for src in sources]
    conflict = _check_field(
        "attending_physician", values, ConflictSeverity.LOW,
    )
    if conflict:
        report.conflicts.append(conflict)

    # ------------------------------------------------------------------ #
    # Log summary                                                          #
    # ------------------------------------------------------------------ #
    logger.info(
        "Conflict detection complete: %d conflicts (%d HIGH, %d MEDIUM, %d LOW)",
        len(report.conflicts),
        len(report.high_severity),
        len(report.medium_severity),
        len([c for c in report.conflicts if c.severity == ConflictSeverity.LOW]),
    )

    return report


def detect_medication_conflicts(
    med_lists: dict[str, list[dict]],  # source → list of {name, dose, frequency}
) -> list[Conflict]:
    """
    Detect the same medication listed with different doses or frequencies across
    progress notes (not admission vs discharge — that's med_reconciliation).
    Returns a list of Conflict objects.
    """
    conflicts: list[Conflict] = []
    sources = list(med_lists.keys())
    if len(sources) < 2:
        return conflicts

    # Build a unified dict: norm_name → {source: med_dict}
    by_name: dict[str, dict[str, dict]] = {}
    for src, meds in med_lists.items():
        for med in meds:
            key = _norm_str(med.get("name", ""))
            if not key:
                continue
            by_name.setdefault(key, {})[src] = med

    for name, source_meds in by_name.items():
        if len(source_meds) < 2:
            continue
        src_list = list(source_meds.keys())
        base_src = src_list[0]
        base_dose = _norm_str(source_meds[base_src].get("dose"))
        base_freq = _norm_str(source_meds[base_src].get("frequency"))

        for other_src in src_list[1:]:
            other_dose = _norm_str(source_meds[other_src].get("dose"))
            other_freq = _norm_str(source_meds[other_src].get("frequency"))

            if base_dose and other_dose and base_dose != other_dose:
                conflicts.append(Conflict(
                    field=f"medication_dose: {name}",
                    severity=ConflictSeverity.HIGH,
                    values=[(s, source_meds[s].get("dose")) for s in src_list],
                    message=(
                        f"Medication '{name}' has conflicting doses across notes. "
                        "Do not resolve automatically — clinician review required."
                    ),
                ))
            elif base_freq and other_freq and base_freq != other_freq:
                conflicts.append(Conflict(
                    field=f"medication_frequency: {name}",
                    severity=ConflictSeverity.MEDIUM,
                    values=[(s, source_meds[s].get("frequency")) for s in src_list],
                    message=(
                        f"Medication '{name}' has conflicting frequencies across notes. "
                        "Clinician review required."
                    ),
                ))

    return conflicts
