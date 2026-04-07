FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --timeout=300 \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    pydantic-settings \
    numpy \
    pandas \
    scikit-learn \
    xgboost \
    langchain \
    langchain-google-genai \
    langchain-core \
    langsmith \
    python-dotenv

COPY shared/ /app/shared/
COPY agents/digital_twin/ /app/agent/

WORKDIR /app/agent

RUN python train_models.py

EXPOSE 8006

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8006"]