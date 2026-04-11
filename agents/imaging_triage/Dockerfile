FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    libgl1 \
    libglib2.0-0 \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --timeout=300 \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    pydantic-settings \
    # tensorflow-cpu \
    Pillow \
    numpy \
    python-dotenv

COPY shared/ /app/shared/
COPY agents/imaging_triage/ /app/agent/

ENV PYTHONPATH=/app

WORKDIR /app/agent

EXPOSE 8005

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005"]