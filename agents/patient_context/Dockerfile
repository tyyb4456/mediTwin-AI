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
    langgraph \
    langsmith \
    python-dotenv \
    "fhirclient==4.2.1"

COPY shared/ /app/shared/
COPY agents/patient_context/ /app/agent/

ENV PYTHONPATH=/app 

WORKDIR /app/agent

EXPOSE 8001

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]