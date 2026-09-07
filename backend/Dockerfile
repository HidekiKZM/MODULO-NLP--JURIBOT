# backend/Dockerfile
# ==========================================
# Juribot - FastAPI + Qdrant + Gemini
# Dockerfile multi-stage (builder + runtime)
# ==========================================

# ---------- Estágio 1: Builder ----------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Isolate dependencies from packages preinstalled in the base image.
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN python -m venv /opt/venv

# Dependências do sistema (para compilar libs Python)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Instala dependências Python (com cache em layer separado)
COPY requirements*.txt ./
ARG TORCH_VERSION=2.10.0
RUN pip install --no-cache-dir --upgrade pip setuptools wheel \
 && pip install --no-cache-dir torch==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir -r requirements.txt


# ---------- Estágio 2: Runtime ----------
FROM python:3.11-slim AS runtime

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Dependências de runtime (OCR + PDF + TLS)
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-por \
 && rm -rf /var/lib/apt/lists/*

# Copia as libs do builder (Python já com dependências)
COPY --from=builder /opt/venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copia o código do backend
COPY . ./backend

# Define usuário não-root (boa prática de segurança)
RUN useradd -m appuser && mkdir -p /home/appuser/.cache \
 && chown -R appuser /app /home/appuser/.cache
USER appuser

# Porta exposta (FastAPI)
EXPOSE 8000

# CMD padrão (docker-compose pode sobrescrever)
CMD ["uvicorn", "backend.app_main:app", "--host", "0.0.0.0", "--port", "8000"]
