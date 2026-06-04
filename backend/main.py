
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env file
load_dotenv()
print("ENV FILE:", os.path.exists(".env"))
print("KEY:", os.getenv("OPENROUTER_API_KEY"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("dscribe")


def _check_api_key() -> None:
    

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        logger.error(
            "\n"
            "OPENROUTER_API_KEY not found.\n\n"
            "Create a .env file in the project root:\n\n"
            "OPENROUTER_API_KEY=your_api_key_here\n"
        )
        sys.exit(1)


def run_single_patient(
    patient_id: str,
    patient_folder: str,
    output_dir: str,
) -> None:

    from agent_loop import run_agent, save_outputs

    logger.info(
        "Processing patient %s from %s",
        patient_id,
        patient_folder,
    )

    state = run_agent(
        patient_id=patient_id,
        patient_folder=patient_folder,
    )

    paths = save_outputs(
        state=state,
        output_dir=output_dir,
    )

    print("\n" + "=" * 70)
    print(f"PATIENT: {patient_id}")
    print(f"Steps executed: {state.step_count}")
    print(
        f"Escalations: {len(state.escalations)} "
        f"({sum(1 for e in state.escalations if e.severity == 'HIGH')} HIGH)"
    )
    print(
        f"Conflicts: "
        f"{len(state.conflicts.conflicts) if state.conflicts else 0}"
    )
    print(f"Errors: {len(state.errors)}")
    print(f"Draft saved: {paths['discharge_summary']}")
    print(f"Trace saved: {paths['step_trace']}")
    print("=" * 70)

    if state.errors:
        print("\nERRORS:")
        for error in state.errors:
            print(f" - {error}")

    if state.warnings:
        print("\nWARNINGS:")
        for warning in state.warnings:
            print(f" - {warning}")

    print("\n--- DISCHARGE SUMMARY PREVIEW ---")

    preview = state.final_summary[:2000]
    print(preview)

    if len(state.final_summary) > 2000:
        print(
            f"\n... [truncated - see "
            f"{paths['discharge_summary']} for full text]"
        )


def run_all_patients(
    patients_root: str,
    output_dir: str,
) -> None:

    root = Path(patients_root)

    patient_folders = sorted(
        [
            folder
            for folder in root.iterdir()
            if folder.is_dir()
        ]
    )

    if not patient_folders:
        logger.error(
            "No patient subfolders found in %s",
            patients_root,
        )
        sys.exit(1)

    logger.info(
        "Found %d patient folder(s): %s",
        len(patient_folders),
        [folder.name for folder in patient_folders],
    )

    for folder in patient_folders:
        run_single_patient(
            patient_id=folder.name,
            patient_folder=str(folder),
            output_dir=output_dir,
        )


def main() -> None:

    parser = argparse.ArgumentParser(
        description="DScribe Discharge Summary Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--patient_folder",
        help="Path to a single patient's PDF folder",
    )

    group.add_argument(
        "--all_patients",
        help="Path containing patient subfolders",
    )

    parser.add_argument(
        "--patient_id",
        default=None,
        help="Patient ID",
    )

    parser.add_argument(
        "--output_dir",
        default="./outputs",
        help="Output directory",
    )

    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=[
            "DEBUG",
            "INFO",
            "WARNING",
            "ERROR",
        ],
    )

    args = parser.parse_args()

    logging.getLogger().setLevel(args.log_level)

    _check_api_key()

    if args.patient_folder:

        patient_id = (
            args.patient_id
            or Path(args.patient_folder).name
        )

        run_single_patient(
            patient_id=patient_id,
            patient_folder=args.patient_folder,
            output_dir=args.output_dir,
        )

    else:

        run_all_patients(
            patients_root=args.all_patients,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()