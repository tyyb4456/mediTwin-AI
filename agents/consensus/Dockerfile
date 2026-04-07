FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --timeout=300 \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    pydantic-settings \
    langchain \
    langchain-google-genai \
    langchain-chroma \
    langchain-core \
    langsmith \
    langgraph \
    chromadb \
    python-dotenv

COPY shared/ /app/shared/
COPY agents/consensus/ /app/agent/

WORKDIR /app/agent

EXPOSE 8007

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8007"]