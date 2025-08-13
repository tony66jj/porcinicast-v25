# Dockerfile — PorciniCast v2.5.x (Render-ready)

FROM python:3.11-slim

# Env pulite
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Librerie di sistema per wheels scientifici + healthcheck via curl
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gfortran libopenblas-dev liblapack-dev pkg-config curl \
 && rm -rf /var/lib/apt/lists/*

# Dipendenze Python
COPY requirements.txt .
RUN python -m pip install --upgrade pip \
 && pip install --no-cache-dir -r requirements.txt

# Codice applicativo
COPY . .

# cartelle utili (se usate)
RUN mkdir -p data logs

# Render espone una PORT dinamica
EXPOSE $PORT

# Healthcheck (niente dipendenze Python)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
  CMD curl -fsS "http://localhost:${PORT}/api/health" || exit 1

# Avvio FastAPI
CMD uvicorn main:app --host 0.0.0.0 --port $PORT

