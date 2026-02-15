from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .schemas import TranslateRequest, TranslateResponse
from .translate import RuntimeResources, load_runtime_resources


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.resources = None
    app.state.startup_error = None
    try:
        resources: RuntimeResources = load_runtime_resources(settings)
        app.state.resources = resources
    except Exception as exc:
        app.state.startup_error = str(exc)
    yield


app = FastAPI(
    title="Vexa API",
    version="1.0.0",
    description="Production-ready FastAPI service for German → English neural machine translation.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root() -> dict:
    return {
        "service": "Vexa API",
        "status": "running",
        "docs": "/docs",
        "health": "/health",
        "translate": "/translate",
    }


@app.get("/health")
def health_check() -> dict:
    startup_error: Optional[str] = getattr(app.state, "startup_error", None)
    ready = startup_error is None and getattr(app.state, "resources", None) is not None

    return {
        "status": "ok" if ready else "degraded",
        "ready": ready,
        "model_path": str(settings.model_path),
        "tokenizer_path": str(settings.tokenizer_path),
        "device": settings.device,
        "error": startup_error,
    }


@app.post("/translate", response_model=TranslateResponse)
def translate(payload: TranslateRequest) -> TranslateResponse:
    startup_error: Optional[str] = getattr(app.state, "startup_error", None)
    resources: Optional[RuntimeResources] = getattr(app.state, "resources", None)
    if startup_error or resources is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Translation service is not ready. "
                f"Model/tokenizer load error: {startup_error or 'unknown error'}"
            ),
        )

    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input text must not be empty.")

    try:
        translation = resources.translator.translate_text(text)
        return TranslateResponse(translation=translation)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Translation failed: {exc}") from exc
