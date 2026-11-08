"""FastAPI web server for the Agentic RAG pipeline.

Run with:
    uvicorn api:app --reload
then open http://127.0.0.1:8000 in your browser.
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

load_dotenv()

from src.data_loader import init_collections  # noqa: E402
from src.graph import build_graph  # noqa: E402

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent / "static"
REQUIRED_KEYS = ("OPENAI_API_KEY", "TAVILY_API_KEY")


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)


class ChatResponse(BaseModel):
    answer: str
    source: str
    context: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Check the API keys, build the vector store and compile the graph once at startup."""
    missing = [key for key in REQUIRED_KEYS if not os.getenv(key)]
    if missing:
        raise RuntimeError(f"Missing keys in .env: {', '.join(missing)}. Fill them in and restart.")

    init_collections()
    app.state.graph = build_graph()
    logger.info("Agentic RAG graph is ready.")
    yield


app = FastAPI(title="Agentic RAG", lifespan=lifespan)


@app.get("/api/health")
def health() -> dict:
    """Simple check that the server is up."""
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Run one question through the graph and return the answer, its source and the context used.

    Declared with plain `def` so FastAPI runs the blocking graph call in a worker thread.
    """
    query = request.query.strip()
    if not query:
        raise HTTPException(status_code=422, detail="The question is empty.")

    try:
        result = app.state.graph.invoke({"query": query, "iteration_count": 0})
    except Exception as exc:  # surface a readable error to the page instead of a bare 500
        logger.exception("The pipeline failed for query: %s", query)
        raise HTTPException(status_code=500, detail=f"The pipeline failed: {exc}") from exc

    return ChatResponse(
        answer=result.get("response", ""),
        source=result.get("source", "Unknown"),
        context=result.get("context", ""),
    )


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
