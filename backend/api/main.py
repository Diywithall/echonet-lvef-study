from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import shutil
import os
import uuid
import time
import traceback
from models.echonet_wrapper import EchoNetAnalyzer
import uvicorn

# ── Configuration ──────────────────────────────────────────────────────────
UPLOAD_DIR  = "temp_uploads"
EF_WEIGHTS  = "models/r2plus1d_18_32_2_pretrained.pt"

os.makedirs(UPLOAD_DIR, exist_ok=True)

# ── App setup ──────────────────────────────────────────────────────────────
app = FastAPI(title="EchoNet Cardiology Analysis API")

app.mount("/uploads", StaticFiles(directory=UPLOAD_DIR), name="uploads")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Lazy-loaded analyser (only R(2+1)D-18, no segmentation model) ──────────
_analyzer: EchoNetAnalyzer | None = None

def get_analyzer() -> EchoNetAnalyzer:
    global _analyzer
    if _analyzer is None:
        _analyzer = EchoNetAnalyzer(EF_WEIGHTS)
    return _analyzer

# ── In-memory task store ───────────────────────────────────────────────────
tasks: dict = {}

# ── Routes ────────────────────────────────────────────────────────────────

@app.get("/")
def read_root():
    return {"message": "EchoNet Cardiology API is running"}


@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".avi")):
        raise HTTPException(status_code=400, detail="Only .mp4 and .avi videos are supported")

    file_id   = str(uuid.uuid4())
    extension = file.filename.rsplit(".", 1)[-1]
    file_path = os.path.join(UPLOAD_DIR, f"{file_id}.{extension}")

    with open(file_path, "wb") as buf:
        shutil.copyfileobj(file.file, buf)

    return {"file_id": file_id, "file_path": file_path}


def _run_analysis_task(task_id: str, file_path: str) -> None:
    """Background worker: runs analyze() and writes result (or error) into tasks[]."""
    try:
        tasks[task_id]["status"] = "processing"
        ana = get_analyzer()
        result = ana.analyze(file_path)
        tasks[task_id]["status"]       = "completed"
        tasks[task_id]["result"]       = result
        tasks[task_id]["completed_at"] = time.time()
    except Exception as exc:
        tasks[task_id]["status"] = "failed"
        tasks[task_id]["error"]  = (
            f"{type(exc).__name__}: {exc}\n"
            f"{traceback.format_exc()}"
        )


@app.post("/analyze")
async def analyze_video(file_id: str, background_tasks: BackgroundTasks):
    # Locate the uploaded file
    file_path = None
    for name in os.listdir(UPLOAD_DIR):
        if name.startswith(file_id):
            file_path = os.path.join(UPLOAD_DIR, name)
            break

    if file_path is None:
        raise HTTPException(status_code=404, detail="File not found — upload first")

    task_id = str(uuid.uuid4())
    tasks[task_id] = {
        "status":     "queued",
        "created_at": time.time(),
        "file_id":    file_id,
    }

    background_tasks.add_task(_run_analysis_task, task_id, file_path)
    return {"task_id": task_id}


@app.get("/results/{task_id}")
async def get_results(task_id: str):
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    return tasks[task_id]


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
