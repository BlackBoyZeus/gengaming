# GameGen-X Backend Service

Enterprise-grade backend service for real-time game video generation and control using advanced AI models, optimized for FreeBSD-based Orbis OS and non-NVIDIA GPUs.

## System Requirements

- FreeBSD-based Orbis OS 13+
- Python 3.9+
- Non-NVIDIA GPU with minimum 24GB VRAM
- 64GB RAM (512GB recommended)
- 100Gbps network for distributed training
- 50TB NVMe storage for datasets/checkpoints
- Podman 4.0+ for containerization

## Quick Start

1. Clone the repository:
```bash
git clone https://github.com/gamegen-x/backend
cd backend
```

2. Copy environment configuration:
```bash
cp .env.example .env
```

3. Install dependencies using Poetry:
```bash
poetry install
```

4. Initialize database:
```bash
poetry run alembic upgrade head
```

5. Start development server:
```bash
poetry run uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## Development Setup

### FreeBSD Environment Configuration

1. Install system dependencies:
```bash
pkg update && pkg install -y \
    python39 \
    py39-pip \
    py39-wheel \
    py39-setuptools \
    curl \
    git \
    gcc \
    gmake \
    pkgconf \
    nvidia-gpu-toolkit-12.1
```

2. Configure GPU drivers:
```bash
# Load GPU kernel module
kldload nvidia-gpu
# Verify GPU detection
nvidia-smi
```

3. Configure Python environment:
```bash
python3.9 -m venv venv
source venv/bin/activate
pip install --upgrade pip poetry
```

### Development Tools

- **Code Formatting**: Black and isort
```bash
poetry run black .
poetry run isort .
```

- **Type Checking**: MyPy
```bash
poetry run mypy app/
```

- **Testing**: Pytest
```bash
poetry run pytest --cov=app tests/
```

## Production Deployment

### Container Setup

1. Build production container:
```bash
podman build -t gamegen-x-backend:latest .
```

2. Deploy services:
```bash
podman-compose -f docker-compose.yml up -d
```

### Environment Configuration

Required environment variables:
- `ENVIRONMENT`: Runtime environment (development/staging/production)
- `GPU_ENABLED`: Enable GPU acceleration
- `MODEL_PATH`: Path to model weights directory
- `REDIS_URL`: Redis cache connection URL
- `DATABASE_URL`: PostgreSQL connection URL
- `JWT_SECRET`: JWT token secret key
- `LOG_LEVEL`: Logging level

### Resource Allocation

Container resource limits (per service):
- API Service: 4 CPUs, 8GB RAM
- Cache Service: 2 CPUs, 4GB RAM
- Database: 2 CPUs, 4GB RAM

### Monitoring Setup

1. Configure Prometheus metrics:
```bash
poetry run python scripts/setup_monitoring.py
```

2. Enable health checks:
```bash
curl http://localhost:8000/health
```

## API Documentation

### Authentication

- JWT-based authentication
- Token expiration: 1 hour
- Refresh token validity: 7 days

### Core Endpoints

- `POST /api/v1/generate`: Generate game video
- `POST /api/v1/control`: Send control signals
- `GET /api/v1/status`: Check generation status
- `WS /ws/stream`: Real-time frame streaming

### WebSocket Interface

Connect to WebSocket endpoint:
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/stream');
```

### Rate Limiting

| Endpoint | Rate Limit |
|----------|------------|
| /generate | 10/minute |
| /control | 60/minute |
| /status | 120/minute |

## Security Guidelines

### Best Practices

1. Keep dependencies updated:
```bash
poetry update
```

2. Run security checks:
```bash
poetry run safety check
```

3. Enable security headers:
```python
app.add_middleware(SecurityMiddleware)
```

### Data Protection

- All API communications over TLS 1.3
- Database connections using SSL
- Model weights encrypted at rest
- Regular security audits

## Maintenance

### Database Migrations

Create new migration:
```bash
poetry run alembic revision --autogenerate -m "description"
```

Apply migrations:
```bash
poetry run alembic upgrade head
```

### Backup Procedures

1. Database backup:
```bash
pg_dump -Fc gamegen_x > backup.dump
```

2. Model weights backup:
```bash
tar -czf weights_backup.tar.gz weights/
```

## Support

- GitHub Issues: [gamegen-x/backend/issues](https://github.com/gamegen-x/backend/issues)
- Documentation: [docs.gamegen-x.com](https://docs.gamegen-x.com)
- Security: security@gamegen-x.com

## License

MIT License - see [LICENSE](LICENSE) for details