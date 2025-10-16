FROM python:3.11-slim as builder
WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY src/ ./src/
COPY alembic/ ./alembic/
COPY alembic.ini ./
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONPATH=/app/src

HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "jeffrey.interfaces.bridge.api:app", "--host", "0.0.0.0", "--port", "8000"]