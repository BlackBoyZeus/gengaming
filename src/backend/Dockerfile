# Stage 1: Builder
FROM freebsd:13.2 as builder

# Set build environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    POETRY_VERSION=1.4.0 \
    POETRY_HOME=/opt/poetry \
    POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_CREATE=false \
    CUDA_HOME=/usr/local/cuda-12.1 \
    LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64 \
    PATH="/opt/poetry/bin:$PATH"

# Install FreeBSD system dependencies
RUN pkg update && pkg install -y \
    python39 \
    py39-pip \
    py39-wheel \
    py39-setuptools \
    curl \
    git \
    gcc \
    gmake \
    pkgconf \
    nvidia-gpu-toolkit-12.1 \
    && rm -rf /var/cache/pkg/*

# Install Poetry
RUN curl -sSL https://install.python-poetry.org | python3.9 - \
    && poetry config virtualenvs.create false

# Set up working directory
WORKDIR /build

# Copy dependency files
COPY pyproject.toml poetry.lock requirements.txt ./

# Install Python packages
RUN poetry install --no-dev --no-root \
    && pip3.9 install --no-cache-dir -r requirements.txt

# Stage 2: Final
FROM freebsd:13.2

# Set runtime environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    CUDA_HOME=/usr/local/cuda-12.1 \
    LD_LIBRARY_PATH=/usr/local/cuda-12.1/lib64 \
    MAX_WORKERS=4 \
    MEMORY_LIMIT=8g \
    TZ=UTC

# Install runtime dependencies
RUN pkg update && pkg install -y \
    python39 \
    py39-pip \
    nvidia-gpu-toolkit-12.1 \
    curl \
    && rm -rf /var/cache/pkg/*

# Create virtual environment
RUN python3.9 -m venv /opt/venv

# Copy Python packages from builder
COPY --from=builder /usr/local/lib/python3.9/site-packages /opt/venv/lib/python3.9/site-packages
COPY --from=builder /usr/local/bin /opt/venv/bin

# Set up working directory
WORKDIR /app

# Create non-root user
RUN pw adduser -n gamegen -d /app -s /usr/sbin/nologin -c "GameGen-X Service Account" \
    && chown -R gamegen:gamegen /app

# Set up logging directory
RUN mkdir -p /app/logs \
    && chown -R gamegen:gamegen /app/logs

# Copy application code
COPY --chown=gamegen:gamegen . .

# Configure security
RUN chmod -R 550 /app \
    && chmod -R 770 /app/logs

# Expose port
EXPOSE 8000

# Set up volumes
VOLUME ["/app/models", "/app/weights", "/app/logs"]

# Set up healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Switch to non-root user
USER gamegen

# Set resource limits
LABEL org.freebsd.resource-limits.cpus="4" \
      org.freebsd.resource-limits.memory="8g" \
      org.freebsd.resource-limits.pids="1000"

# Add metadata labels
LABEL maintainer="GameGen-X Team" \
      version="1.0.0" \
      description="GameGen-X Backend Service" \
      org.opencontainers.image.source="https://github.com/gamegen-x/backend" \
      org.opencontainers.image.licenses="MIT"

# Start FastAPI application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4", "--limit-concurrency", "1000", "--limit-max-requests", "10000", "--timeout-keep-alive", "120", "--log-config", "logging.conf"]