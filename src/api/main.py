from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Literal, Optional
import time
import hashlib
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from src.models.summarizer import TextSummarizer, ExtractiveSummarizer

app = FastAPI(title="Text Summarizer API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

extractive = ExtractiveSummarizer()
abstractive: Optional[TextSummarizer] = None

def get_abstractive():
    global abstractive
    if abstractive is None:
        abstractive = TextSummarizer("bart")
    return abstractive

class SummarizeRequest(BaseModel):
    text: str = Field(..., min_length=50, max_length=10000)
    method: Literal["extractive", "abstractive"] = "extractive"
    length: Literal["short", "medium", "long"] = "medium"
    num_sentences: int = Field(3, ge=1, le=10)

    @validator("text")
    def clean(cls, v):
        return v.strip()

class SummarizeResponse(BaseModel):
    success: bool
    summary: str
    method_used: str
    original_length: int
    summary_length: int
    compression_ratio: float
    processing_time_ms: float
    cache_hit: bool = False

_cache = {}

@app.get("/")
def root():
    return {"message": "Text Summarizer API!", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "healthy",
            "abstractive_loaded": abstractive is not None}

@app.post("/summarize", response_model=SummarizeResponse)
def summarize(req: SummarizeRequest):
    t0 = time.time()
    key = hashlib.md5(
        f"{req.text}{req.method}{req.length}".encode()
    ).hexdigest()

    if key in _cache:
        r = _cache[key].copy()
        r["cache_hit"] = True
        r["processing_time_ms"] = round((time.time()-t0)*1000, 2)
        return SummarizeResponse(**r)

    try:
        if req.method == "extractive":
            res = extractive.summarize(req.text, req.num_sentences)
            data = dict(
                success=True,
                summary=res["summary"],
                method_used="extractive (TF-IDF)",
                original_length=len(req.text.split()),
                summary_length=len(res["summary"].split()),
                compression_ratio=res.get("compression_ratio", 0.0),
                processing_time_ms=0,
                cache_hit=False
            )
        else:
            res = get_abstractive().summarize(req.text, req.length)
            data = dict(
                success=True,
                summary=res["summary"],
                method_used=f"abstractive ({res['model_used']})",
                original_length=res["original_length"],
                summary_length=res["summary_length"],
                compression_ratio=res["compression_ratio"],
                processing_time_ms=0,
                cache_hit=False
            )
    except ValueError as e:
        raise HTTPException(422, str(e))
    except Exception as e:
        raise HTTPException(500, f"Error: {e}")

    data["processing_time_ms"] = round((time.time()-t0)*1000, 2)
    if len(_cache) < 100:
        _cache[key] = data.copy()
    return SummarizeResponse(**data)

@app.delete("/cache")
def clear_cache():
    _cache.clear()
    return {"message": "Cache cleared"}

@app.get("/app")
def frontend():
    return FileResponse("templates/index.html")