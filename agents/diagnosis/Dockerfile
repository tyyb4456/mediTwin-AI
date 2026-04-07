FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --timeout=300\
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    pydantic-settings \
    httpx \
    langchain \
    langchain-google-genai \
    langchain-chroma \
    langchain-text-splitters \
    langgraph \
    langsmith \
    chromadb \
    python-dotenv

COPY shared/ /app/shared/
COPY knowledge_base/ /app/knowledge_base/
COPY agents/diagnosis/ /app/agent/

ENV PYTHONPATH=/app  

WORKDIR /app/agent

EXPOSE 8002

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8002"]