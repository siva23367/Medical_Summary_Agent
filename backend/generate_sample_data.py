"""
generate_sample_data.py
-----------------------
Generates synthetic patient note PDFs for testing the agent.
Creates two patients:
  patient_001: Straightforward case (CHF exacerbation)
  patient_002: Complex case with conflicts, missing data, pending results

Run: python generate_sample_data.py
Outputs to ./sample_data/patient_001/ and ./sample_data/patient_002/
"""

from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.units import inch
from reportlab.lib import colors


def make_pdf(output_path: str, title: str, content_sections: list[tuple[str, str]]) -> None:
    """Create a simple PDF with heading + body sections."""
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        rightMargin=inch,
        leftMargin=inch,
        topMargin=inch,
        bottomMargin=inch,
    )
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "CustomTitle",
        parent=styles["Title"],
        fontSize=14,
        spaceAfter=12,
    )
    heading_style = ParagraphStyle(
        "CustomHeading",
        parent=styles["Heading2"],
        fontSize=11,
        spaceAfter=6,
        spaceBefore=12,
        textColor=colors.darkblue,
    )
    body_style = ParagraphStyle(
        "CustomBody",
        parent=styles["Normal"],
        fontSize=10,
        spaceAfter=6,
        leading=14,
    )

    story = [Paragraph(title, title_style), Spacer(1, 0.2 * inch)]
    for heading, body in content_sections:
        if heading:
            story.append(Paragraph(heading, heading_style))
        # Replace newlines with <br/> for ReportLab
        body_html = body.replace("\n", "<br/>")
        story.append(Paragraph(body_html, body_style))
        story.append(Spacer(1, 0.1 * inch))

    doc.build(story)
    print(f"  Created: {output_path}")


# ==========================================================================
# PATIENT 001: Straightforward CHF case
# ==========================================================================

def create_patient_001(base_dir: Path) -> None:
    folder = base_dir / "patient_001"
    folder.mkdir(parents=True, exist_ok=True)

    # --- Admission Note ---
    make_pdf(
        str(folder / "admission_note.pdf"),
        "ADMISSION NOTE — City General Hospital",
        [
            ("Patient Information", (
                "Name: John Arthur Doe\n"
                "DOB: 15-Mar-1952  |  MRN: CGH-00123456\n"
                "Admission Date: 2025-03-10\n"
                "Attending Physician: Dr. Sarah Mitchell\n"
                "Ward: Cardiology 4B"
            )),
            ("Chief Complaint", "Increasing shortness of breath and bilateral leg swelling x 5 days."),
            ("History of Presenting Illness", (
                "Mr Doe is a 72-year-old male with known ischaemic cardiomyopathy (EF 35%) presenting "
                "with worsening dyspnoea on exertion and at rest over the past 5 days. He reports 4 kg "
                "weight gain over 2 weeks. He has been poorly compliant with his fluid restriction. "
                "He denies chest pain, syncope, or palpitations. No recent infections."
            )),
            ("Past Medical History", (
                "1. Congestive heart failure (EF 35%, ischaemic aetiology)\n"
                "2. Hypertension\n"
                "3. Type 2 diabetes mellitus\n"
                "4. Chronic kidney disease stage 3\n"
                "5. Hyperlipidaemia"
            )),
            ("Allergies", "Penicillin — anaphylaxis\nSulfonamides — rash"),
            ("Admission Medications", (
                "1. Furosemide 40 mg oral once daily\n"
                "2. Carvedilol 12.5 mg oral twice daily (for heart failure)\n"
                "3. Ramipril 5 mg oral once daily (for heart failure/hypertension)\n"
                "4. Atorvastatin 40 mg oral at bedtime\n"
                "5. Metformin 500 mg oral twice daily (for type 2 diabetes)\n"
                "6. Aspirin 75 mg oral once daily\n"
                "7. Spironolactone 25 mg oral once daily"
            )),
            ("Examination", (
                "BP: 158/96 mmHg  |  HR: 88 bpm  |  RR: 22  |  SpO2: 94% on air\n"
                "JVP elevated at 5 cm above sternal angle. Bilateral basal crepitations. "
                "Bilateral pitting oedema to knees."
            )),
            ("Plan", (
                "1. IV furosemide 80 mg BD for diuresis\n"
                "2. Daily weights and strict fluid balance\n"
                "3. Hold metformin (CKD + diuresis)\n"
                "4. Echocardiogram requested\n"
                "5. BNP, renal function, electrolytes, FBC"
            )),
        ],
    )

    # --- Progress Note ---
    make_pdf(
        str(folder / "progress_note_day3.pdf"),
        "PROGRESS NOTE — Day 3",
        [
            ("Patient", "John Arthur Doe  |  MRN: CGH-00123456  |  Date: 2025-03-13"),
            ("Subjective", "Patient reports improved breathlessness. Still some ankle swelling. Tolerating oral fluids."),
            ("Objective", (
                "BP: 132/82 mmHg  |  HR: 74 bpm  |  SpO2: 97% on air\n"
                "Weight: 88 kg (down 3.5 kg from admission)\n"
                "Reduced basal crepitations. Oedema improved to ankles only."
            )),
            ("Labs", (
                "Na: 138 mmol/L  |  K: 3.8 mmol/L  |  Creatinine: 142 umol/L (baseline 130)\n"
                "BNP: 850 pg/mL (down from 2100 on admission)\n"
                "FBC: WBC 7.2, Hb 118 g/L, Platelets 224"
            )),
            ("Echocardiogram", "EF 35%, moderate mitral regurgitation, dilated LV. No new wall motion abnormalities."),
            ("Assessment & Plan", (
                "Improving CHF. Continue IV diuresis today, switch to oral furosemide 80 mg tomorrow.\n"
                "Uptitrate carvedilol to 25 mg twice daily when euvolaemic.\n"
                "Renal function stable — will reassess metformin restart at discharge."
            )),
        ],
    )

    # --- Lab Results ---
    make_pdf(
        str(folder / "lab_results.pdf"),
        "LABORATORY RESULTS",
        [
            ("Patient", "John Arthur Doe  |  MRN: CGH-00123456"),
            ("Admission Labs (2025-03-10)", (
                "Sodium: 136 mmol/L  |  Potassium: 4.1 mmol/L\n"
                "Creatinine: 138 umol/L  |  Urea: 9.2 mmol/L\n"
                "BNP: 2100 pg/mL (HIGH)\n"
                "HbA1c: 7.8% (above target)\n"
                "Troponin I: 0.02 ng/mL (normal)\n"
                "INR: 1.1  |  ALT: 28 U/L  |  AST: 32 U/L"
            )),
            ("Day 3 Labs (2025-03-13)", (
                "Sodium: 138 mmol/L  |  Potassium: 3.8 mmol/L\n"
                "Creatinine: 142 umol/L  |  Urea: 8.1 mmol/L\n"
                "BNP: 850 pg/mL"
            )),
            ("Pending Results", (
                "Urine microalbumin: specimen sent 2025-03-13 — result pending\n"
                "Thyroid function tests: sent 2025-03-12 — result pending"
            )),
        ],
    )

    # --- Medication Record ---
    make_pdf(
        str(folder / "medication_record.pdf"),
        "MEDICATION RECORD — DISCHARGE",
        [
            ("Patient", "John Arthur Doe  |  MRN: CGH-00123456  |  Discharge Date: 2025-03-16"),
            ("Discharge Medications", (
                "1. Furosemide 80 mg oral once daily (increased from 40 mg — ongoing diuresis for CHF)\n"
                "2. Carvedilol 25 mg oral twice daily (uptitrated from 12.5 mg — heart failure optimisation)\n"
                "3. Ramipril 5 mg oral once daily\n"
                "4. Atorvastatin 40 mg oral at bedtime\n"
                "5. Aspirin 75 mg oral once daily\n"
                "6. Spironolactone 25 mg oral once daily\n"
                "7. Empagliflozin 10 mg oral once daily (new — added for HFrEF and diabetes)\n"
                "NOTE: Metformin HELD at discharge — to be restarted by GP once renal function stable"
            )),
            ("Discharge Condition", "Improved. Haemodynamically stable. Euvolaemic."),
            ("Follow-up", (
                "1. Heart failure clinic in 2 weeks (2025-03-30) — Dr Mitchell\n"
                "2. GP within 1 week for renal function check and metformin review\n"
                "3. Daily weights — attend ED if weight gain >2 kg in 24 hours\n"
                "4. Fluid restriction: 1.5 L per day\n"
                "5. Collect pending lab results from GP"
            )),
        ],
    )


# ==========================================================================
# PATIENT 002: Complex case — conflicts, missing data, pending results
# ==========================================================================

def create_patient_002(base_dir: Path) -> None:
    folder = base_dir / "patient_002"
    folder.mkdir(parents=True, exist_ok=True)

    # Admission note — has some info but some is missing/inconsistent
    make_pdf(
        str(folder / "admission_note.pdf"),
        "ADMISSION NOTE — Riverside Medical Centre",
        [
            ("Patient Information", (
                "Name: Maria Elena Santos\n"
                "DOB: 22-Sep-1968  |  MRN: RMC-78234\n"
                "Admission Date: 2025-04-02\n"
                "Attending: Dr. James Okonkwo"
            )),
            ("Chief Complaint", "Sudden onset severe headache and confusion."),
            ("History", (
                "Ms Santos is a 56-year-old woman presenting with sudden onset severe headache "
                "('worst headache of my life') and acute confusion. Onset 3 hours prior. "
                "No focal neurological deficit on initial assessment. PMHx: hypertension, "
                "known epilepsy on levetiracetam. Smoker (20 pack-years)."
            )),
            ("Allergies", "Codeine — vomiting and drowsiness"),
            ("Admission Medications", (
                "1. Amlodipine 10 mg oral once daily (hypertension)\n"
                "2. Levetiracetam 500 mg oral twice daily (epilepsy)\n"
                "3. Atorvastatin 20 mg at bedtime"
            )),
            ("Initial Assessment", (
                "Rule out subarachnoid haemorrhage. CT head ordered urgently.\n"
                "Neurology consulted."
            )),
        ],
    )

    # Progress note — has conflicting diagnosis
    make_pdf(
        str(folder / "progress_note_day2.pdf"),
        "PROGRESS NOTE — Day 2",
        [
            ("Patient", "Maria Santos  |  MRN: RMC-78234  |  Date: 2025-04-03"),
            ("Update", (
                "CT head: NO subarachnoid haemorrhage. LP performed — xanthochromia POSITIVE.\n"
                "Diagnosis revised to: SUBARACHNOID HAEMORRHAGE (SAH) confirmed on LP.\n"
                "Transferred to neurosurgery."
            )),
            ("Medications on Transfer", (
                "1. Nimodipine 60 mg oral every 4 hours (new — SAH neuroprotection)\n"
                "2. Levetiracetam 1000 mg oral twice daily (dose doubled — seizure prophylaxis in SAH)\n"
                "3. Amlodipine HELD — BP management now via nimodipine\n"
                "4. Atorvastatin 20 mg continued\n"
                "5. Dexamethasone 4 mg IV every 6 hours (new — reason not documented)"
            )),
            ("Attending Note", "Primary attending now: Dr. Priya Nair (Neurosurgery)"),
        ],
    )

    # Neurosurgery note — CONFLICTS with previous notes
    make_pdf(
        str(folder / "neurosurgery_note.pdf"),
        "NEUROSURGERY CONSULTATION NOTE",
        [
            ("Patient", "Maria E. Santos  |  MRN: RMC-78234"),
            ("Date", "2025-04-03"),
            (
                "NOTE — CONFLICT WITH PREVIOUS DOCUMENTATION",
                (
                    "Reviewing previous notes: CT head report actually reads 'Cannot exclude SAH — "
                    "recommend LP for confirmation'. LP xanthochromia result is borderline positive "
                    "and may represent traumatic tap. Primary diagnosis remains UNCERTAIN.\n\n"
                    "Neurosurgery assessment: NO operative intervention indicated at this time. "
                    "Watchful waiting. MRI brain with contrast ordered."
                ),
            ),
            ("Allergies noted", (
                "Patient reports allergy to PENICILLIN (rash) — NOT documented in admission note. "
                "Please update allergy record."
            )),
            ("Discharge plan (tentative)", "Discharge home when pain controlled and no neurological deterioration."),
            ("Pending", (
                "MRI brain with contrast: PENDING — booked for 2025-04-05\n"
                "Neurovascular CTA: PENDING — result awaited\n"
                "Formal LP analysis (cytology): PENDING"
            )),
        ],
    )

    # Discharge medication record — has unexplained changes
    make_pdf(
        str(folder / "medication_record.pdf"),
        "MEDICATION RECORD — DISCHARGE",
        [
            ("Patient", "Maria E. Santos  |  MRN: RMC-78234"),
            ("Discharge Date", "2025-04-07"),  # Conflicts with admission date math
            ("Discharge Medications", (
                "1. Levetiracetam 1000 mg oral twice daily\n"
                "2. Nimodipine 60 mg oral every 4 hours (for 21 days total)\n"
                "3. Atorvastatin 40 mg at bedtime\n"
                "4. Paracetamol 1 g oral as needed (max 4 g/day)\n"
                "NOTE: Amlodipine and Dexamethasone NOT on discharge list — no reason documented"
            )),
            ("Discharge Condition", "Stable but diagnosis unresolved — see neurosurgery note."),
            ("Follow-up", (
                "1. Urgent neurosurgery outpatient: 2025-04-10 — Dr Nair\n"
                "2. Collect MRI and CTA results — PENDING at discharge\n"
                "3. Return to ED immediately if: severe headache recurrence, seizure, "
                "   focal weakness, visual changes, reduced consciousness"
            )),
        ],
    )


# ==========================================================================
# Main
# ==========================================================================

if __name__ == "__main__":
    base = Path("./sample_data")
    base.mkdir(exist_ok=True)

    print("Generating sample patient data...")
    print("\nPatient 001 (straightforward CHF case):")
    create_patient_001(base)

    print("\nPatient 002 (complex — conflicts, missing data, pending results):")
    create_patient_002(base)

    print("\nDone. Run the agent with:")
    print("  python main.py --all_patients ./sample_data --output_dir ./outputs")
