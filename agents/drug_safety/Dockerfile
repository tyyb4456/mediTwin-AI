FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --timeout=300 \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    pydantic-settings \
    httpx \
    "redis[hiredis]" \
    langchain \
    langchain-google-genai \
    langchain-core \
    langsmith \
    python-dotenv \
    "mcp[cli]" \
    fastmcp \
    starlette

COPY shared/ /app/shared/
COPY agents/drug_safety/ /app/agent/

WORKDIR /app/agent

EXPOSE 8004

CMD ["uvicorn", "main:combined_app", "--host", "0.0.0.0", "--port", "8004"]