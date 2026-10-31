from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import io
import uuid
import pandas as pd
from backend.services import (
    register_dataset, dataset_preview, dataset_quality, analyze_question,
    dataset_anomalies
)

app = FastAPI(title="AI-Powered Data Analyst API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    dataset_id: str
    question: str = Field(min_length=1, max_length=2000)
    history: list[dict[str, str]] = Field(default_factory=list)

@app.get("/health")
def health():
    return {"status": "ok", "service": "ai-data-analyst-api"}

@app.post("/datasets/upload")
async def upload_csv(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="Upload at least one CSV file.")
    parsed, total_bytes = [], 0
    for file in files:
        if not file.filename or not file.filename.lower().endswith(".csv"):
            raise HTTPException(status_code=400, detail="Only .csv files are supported.")
        content = await file.read()
        total_bytes += len(content)
        if total_bytes > 25 * 1024 * 1024:
            raise HTTPException(status_code=413, detail="Combined upload size must be under 25 MB.")
        try:
            df = pd.read_csv(io.BytesIO(content))
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Could not parse {file.filename} as CSV: {exc}")
        if df.empty:
            raise HTTPException(status_code=400, detail=f"{file.filename} has no data rows.")
        df.columns = [str(c).strip() for c in df.columns]
        if len(df.columns) > 200 or any(not col for col in df.columns):
            raise HTTPException(status_code=400, detail=f"{file.filename} has invalid or too many columns.")
        if df.columns.duplicated().any():
            raise HTTPException(status_code=400, detail=f"{file.filename} has duplicate column names.")
        parsed.append((file.filename, df))
    dataset_id = str(uuid.uuid4())
    return {"dataset_id": dataset_id, "files": register_dataset(dataset_id, parsed)}

@app.get("/datasets/{dataset_id}/preview")
def preview(dataset_id: str):
    try:
        return {"files": dataset_preview(dataset_id)}
    except KeyError:
        raise HTTPException(status_code=404, detail="Dataset not found. Upload CSV files again.")

@app.get("/datasets/{dataset_id}/quality")
def quality(dataset_id: str):
    try:
        return {"files": dataset_quality(dataset_id)}
    except KeyError:
        raise HTTPException(status_code=404, detail="Dataset not found. Upload CSV files again.")

@app.get("/datasets/{dataset_id}/anomalies")
def anomalies(dataset_id: str):
    try:
        return {"files": dataset_anomalies(dataset_id)}
    except KeyError:
        raise HTTPException(status_code=404, detail="Dataset not found. Upload CSV files again.")

@app.post("/analyze")
def analyze(request: AnalyzeRequest):
    try:
        return analyze_question(request.dataset_id, request.question, request.history)
    except KeyError:
        raise HTTPException(status_code=404, detail="Dataset not found. Please upload CSV files again.")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception:
        raise HTTPException(status_code=500, detail="Analysis failed. Check backend logs and input data.")
