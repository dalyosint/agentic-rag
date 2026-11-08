FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN useradd --create-home app

WORKDIR /app

# Install dependencies first so code changes don't trigger a full reinstall
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY api.py main.py ./
COPY src ./src
COPY static ./static

USER app

# Download ChromaDB's embedding model now, so it isn't fetched on the first question
RUN python -c "from chromadb.utils.embedding_functions import ONNXMiniLM_L6_V2; ONNXMiniLM_L6_V2()(['warmup'])"

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health', timeout=3)"

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
