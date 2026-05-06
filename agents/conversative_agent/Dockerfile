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
    langchain-core \
    langchain-google-genai \
    langchain-community \
    langsmith \
    langgraph \
    langgraph-checkpoint-postgres \
    "psycopg[binary,pool]" \
    python-dotenv

COPY shared/ /app/shared/
COPY agents/tool_agent/ /app/agents/tool_agent/

WORKDIR /app/agents/tool_agent

EXPOSE 8010

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8010"]