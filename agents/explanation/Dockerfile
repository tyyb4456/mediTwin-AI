FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN pip install --timeout=300 \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    pydantic-settings \
    asyncpg \
    scikit-learn \
    xgboost-cpu \
    langchain \
    langchain-google-genai \
    langchain-core \
    langsmith \
    textstat \
    python-dotenv

COPY shared/ /app/shared/
COPY agents/digital_twin/ /app/agents/digital_twin/
COPY agents/explanation/ /app/agent/

ENV PYTHONPATH=/app

WORKDIR /app/agent

EXPOSE 8009

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8009"]