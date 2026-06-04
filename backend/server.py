"""
server.py — DScribe FastAPI Backend
Handles PDF uploads, runs the discharge summary agent, streams progress,
and serves the frontend static files.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dscribe.server")

app = FastAPI(title="DScribe Discharge Summary Agent", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory job store  { job_id: { status, result, events, error } }
_jobs: dict[str, dict] = {}

UPLOAD_DIR = Path(tempfile.gettempdir()) / "dscribe_uploads"
UPLOAD_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Helper: run agent in a thread and push SSE events
# ---------------------------------------------------------------------------

def _run_agent_sync(job_id: str, patient_folder: str, patient_id: str) -> None:
    """Runs inside a thread-pool executor."""
    job = _jobs[job_id]
    try:
        # Lazy import so the module path resolves correctly
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from agent_loop import run_agent, save_outputs

        job["events"].append({"type": "status", "message": "Starting agent…"})

        state = run_agent(patient_id=patient_id, patient_folder=patient_folder)

        # Save outputs to a temp dir
        out_dir = UPLOAD_DIR / job_id / "outputs"
        paths = save_outputs(state=state, output_dir=str(out_dir))

        # Build result payload
        trace_data = json.loads(Path(paths["step_trace"]).read_text())
        summary_text = Path(paths["discharge_summary"]).read_text()

        job["result"] = {
            "patient_id": patient_id,
            "summary": summary_text,
            "trace": trace_data,
            "escalations": [
                {
                    "id": e.issue_id,
                    "category": e.category,
                    "severity": e.severity,
                    "description": e.description,
                    "action": e.recommended_action,
                }
                for e in state.escalations
            ],
            "conflicts": state.conflicts.to_dict() if state.conflicts else {"total_conflicts": 0, "conflicts": []},
            "warnings": state.warnings,
            "errors": state.errors,
            "steps": state.step_count,
        }
        job["status"] = "done"
        job["events"].append({"type": "done", "message": "Agent finished."})

    except Exception as exc:
        logger.exception("Agent failed for job %s: %s", job_id, exc)
        job["status"] = "error"
        job["error"] = str(exc)
        job["events"].append({"type": "error", "message": str(exc)})
    finally:
        # Clean up uploaded PDFs
        folder = Path(patient_folder)
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"status": "ok", "service": "DScribe"}


@app.post("/api/run")
async def run_agent_endpoint(
    files: list[UploadFile] = File(...),
    gemini_key: str = "",
):
    """
    Accept one or more PDF uploads for a single patient, run the agent,
    return a job_id immediately.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")

    # Use key from form, or fall back to env
    effective_key = gemini_key.strip() or os.getenv("GEMINI_API_KEY", "")
    if not effective_key:
        raise HTTPException(status_code=400, detail="GEMINI_API_KEY is required.")
    os.environ["GEMINI_API_KEY"] = effective_key

    # Validate all are PDFs
    for f in files:
        if not (f.filename or "").lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"Only PDF files accepted. Got: {f.filename}")

    job_id = str(uuid.uuid4())
    patient_folder = UPLOAD_DIR / job_id / "pdfs"
    patient_folder.mkdir(parents=True)

    # Save uploaded PDFs
    for f in files:
        dest = patient_folder / (f.filename or f"file_{uuid.uuid4()}.pdf")
        content = await f.read()
        dest.write_bytes(content)

    # Create job entry
    _jobs[job_id] = {
        "status": "running",
        "result": None,
        "events": [{"type": "status", "message": f"Uploaded {len(files)} PDF(s). Processing…"}],
        "error": None,
    }

    # Run agent in background thread
    patient_id = f"P-{job_id[:8].upper()}"
    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_agent_sync, job_id, str(patient_folder), patient_id)

    return {"job_id": job_id, "patient_id": patient_id, "file_count": len(files)}


@app.get("/api/status/{job_id}")
async def get_status(job_id: str):
    """Poll for job status and result."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "job_id": job_id,
        "status": job["status"],
        "result": job["result"],
        "error": job["error"],
    }


@app.get("/api/events/{job_id}")
async def event_stream(job_id: str):
    """SSE stream for live progress updates."""
    job = _jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    async def generate():
        sent = 0
        while True:
            events = job["events"]
            while sent < len(events):
                ev = events[sent]
                yield f"data: {json.dumps(ev)}\n\n"
                sent += 1
            if job["status"] in ("done", "error"):
                break
            await asyncio.sleep(0.3)

    return StreamingResponse(generate(), media_type="text/event-stream")


@app.post("/api/demo")
async def run_demo():
    """
    Run agent on the built-in P001 sample data (CHF case).
    Generates sample PDFs on the fly and runs the agent.
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent))

    job_id = str(uuid.uuid4())
    demo_folder = UPLOAD_DIR / job_id / "pdfs"
    demo_folder.mkdir(parents=True)

    _jobs[job_id] = {
        "status": "running",
        "result": None,
        "events": [{"type": "status", "message": "Generating demo patient data…"}],
        "error": None,
    }

    def _run_demo():
        try:
            from generate_sample_data import create_patient_001
            from pathlib import Path as P
            create_patient_001(demo_folder.parent)
            # Move generated PDFs into pdfs/
            src = demo_folder.parent / "patient_001"
            if src.exists():
                for pdf in src.glob("*.pdf"):
                    shutil.copy(pdf, demo_folder / pdf.name)
                shutil.rmtree(src, ignore_errors=True)
            _run_agent_sync(job_id, str(demo_folder), "P001-DEMO")
        except Exception as exc:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(exc)

    loop = asyncio.get_event_loop()
    loop.run_in_executor(None, _run_demo)
    return {"job_id": job_id, "patient_id": "P001-DEMO"}


# ---------------------------------------------------------------------------
# Serve frontend
# ---------------------------------------------------------------------------

frontend_dir = Path(__file__).parent.parent / "frontend"
if (frontend_dir / "static").exists():
    app.mount("/static", StaticFiles(directory=str(frontend_dir / "static")), name="static")

@app.get("/", include_in_schema=False)
@app.get("/{full_path:path}", include_in_schema=False)
async def serve_frontend(full_path: str = ""):
    index = frontend_dir / "index.html"
    if index.exists():
        return FileResponse(str(index))
    return JSONResponse({"message": "DScribe API running. Frontend not found."})
