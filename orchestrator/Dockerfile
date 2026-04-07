FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --timeout=300 \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    pydantic-settings \
    httpx \
    langchain \
    langchain-google-genai \
    langchain-core \
    langsmith \
    langgraph \
    python-dotenv

COPY shared/ /app/shared/
COPY orchestrator/ /app/orchestrator/

WORKDIR /app/orchestrator

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]