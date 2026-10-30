FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

ARG INSTALL_RAG=false

COPY pyproject.toml README.md ./
COPY agents ./agents
COPY api ./api
COPY core ./core
COPY evidence ./evidence
COPY frontend ./frontend
COPY graph ./graph
COPY guardrail ./guardrail
COPY memory ./memory
COPY rag ./rag
COPY tools ./tools

RUN python -m pip install --upgrade pip && \
    if [ "$INSTALL_RAG" = "true" ]; then \
        python -m pip install ".[rag]"; \
    else \
        python -m pip install .; \
    fi

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=3)"

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
