# Local Development Guide

This guide covers setting up a local development environment for the Advanced Federated Pipeline system.

## Prerequisites

### System Requirements

**Minimum Requirements:**
- 8GB RAM (16GB recommended)
- 50GB free disk space
- Multi-core CPU (4+ cores recommended)
- Internet connection for downloading dependencies

**Supported Operating Systems:**
- Windows 10/11
- macOS 10.15+
- Ubuntu 18.04+ / Debian 10+
- CentOS 8+ / RHEL 8+

### Required Software

- **Docker Desktop** 4.0+ with Docker Compose
- **Python** 3.8+
- **Node.js** 16+ (for web interfaces)
- **Git** 2.20+
- **Visual Studio Code** (recommended IDE)

## Quick Start

### 1. Clone Repository

```bash
# Clone the repository
git clone <repository-url>
cd advanced-federated-pipeline

# Verify directory structure
ls -la
```

### 2. Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit environment variables
nano .env
```

```bash
# .env
# Server Configuration
AGGREGATION_SERVER_HOST=localhost
AGGREGATION_SERVER_PORT=8000
EDGE_COORDINATOR_PORT=8001

# Database Configuration
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=federated_pipeline
POSTGRES_USER=federated_user
POSTGRES_PASSWORD=secure_password

# Redis Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# Development Settings
DEBUG=true
LOG_LEVEL=DEBUG
PYTHONPATH=./src

# SDR Configuration (for testing)
SDR_MOCK_MODE=true
SDR_SAMPLE_RATE=2400000
SDR_FREQUENCY=100000000

# Security (development keys - change for production)
JWT_SECRET=dev-jwt-secret-key
ENCRYPTION_KEY=dev-encryption-key-32-chars-long
```

### 3. Start Development Environment

```bash
# Start all services
docker-compose up -d

# Verify services are running
docker-compose ps

# Check logs
docker-compose logs -f
```

### 4. Verify Installation

```bash
# Test aggregation server
curl http://localhost:8000/health

# Test edge coordinator
curl http://localhost:8001/health

# Check database connection
docker-compose exec postgres psql -U federated_user -d federated_pipeline -c "SELECT version();"
```

## Detailed Setup

### Python Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements/dev.txt

# Install in development mode
pip install -e .
```

### Database Setup

```bash
# Initialize database
docker-compose exec aggregation-server python -m alembic upgrade head

# Create test data
docker-compose exec aggregation-server python scripts/create_test_data.py

# Verify database setup
docker-compose exec postgres psql -U federated_user -d federated_pipeline -c "\dt"
```

### SDR Hardware Setup (Optional)

```bash
# Install SDR drivers (Linux/macOS)
# For RTL-SDR
sudo apt install rtl-sdr librtlsdr-dev  # Ubuntu/Debian
brew install librtlsdr  # macOS

# For HackRF
sudo apt install hackrf libhackrf-dev  # Ubuntu/Debian
brew install hackrf  # macOS

# Test SDR hardware
rtl_test -t  # RTL-SDR test
hackrf_info  # HackRF test

# If no hardware available, use mock mode
export SDR_MOCK_MODE=true
```

## Development Workflow

### Running Individual Components

```bash
# Run aggregation server only
cd src/aggregation_server
python main.py

# Run edge coordinator only
cd src/edge_coordinator
python service.py

# Run SDR client only
cd src/sdr_client
python sdr_client.py

# Run mobile client (simulation)
cd src/mobile_client
python mobile_client.py
```

### Running Tests

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/unit/test_aggregation_server.py -v
```

### Code Quality

```bash
# Format code
black src/ tests/
isort src/ tests/

# Lint code
flake8 src/ tests/
pylint src/

# Type checking
mypy src/

# Security scanning
bandit -r src/
```

## IDE Configuration

### Visual Studio Code

Create `.vscode/settings.json`:

```json
{
    "python.defaultInterpreterPath": "./venv/bin/python",
    "python.linting.enabled": true,
    "python.linting.pylintEnabled": true,
    "python.linting.flake8Enabled": true,
    "python.formatting.provider": "black",
    "python.sortImports.args": ["--profile", "black"],
    "python.testing.pytestEnabled": true,
    "python.testing.pytestArgs": [
        "tests"
    ],
    "files.exclude": {
        "**/__pycache__": true,
        "**/*.pyc": true,
        ".pytest_cache": true,
        ".coverage": true,
        "htmlcov": true
    }
}
```

Create `.vscode/launch.json`:

```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Aggregation Server",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/src/aggregation_server/main.py",
            "console": "integratedTerminal",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src"
            }
        },
        {
            "name": "Edge Coordinator",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/src/edge_coordinator/service.py",
            "console": "integratedTerminal",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src"
            }
        },
        {
            "name": "SDR Client",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/src/sdr_client/sdr_client.py",
            "console": "integratedTerminal",
            "env": {
                "PYTHONPATH": "${workspaceFolder}/src",
                "SDR_MOCK_MODE": "true"
            }
        },
        {
            "name": "Run Tests",
            "type": "python",
            "request": "launch",
            "module": "pytest",
            "args": ["tests/", "-v"],
            "console": "integratedTerminal"
        }
    ]
}
```

### PyCharm Configuration

1. Open project in PyCharm
2. Configure Python interpreter: `File > Settings > Project > Python Interpreter`
3. Add content root: `File > Settings > Project > Project Structure`
4. Configure test runner: `File > Settings > Tools > Python Integrated Tools`

## Development Services

### Database Management

```bash
# Access PostgreSQL
docker-compose exec postgres psql -U federated_user -d federated_pipeline

# Common database operations
\dt                    # List tables
\d+ clients           # Describe clients table
SELECT * FROM models LIMIT 5;  # Query data

# Reset database
docker-compose down -v
docker-compose up -d postgres
docker-compose exec aggregation-server python -m alembic upgrade head
```

### Redis Management

```bash
# Access Redis CLI
docker-compose exec redis redis-cli

# Common Redis operations
KEYS *                 # List all keys
GET client:123         # Get value
FLUSHALL              # Clear all data
```

### Monitoring Stack

```bash
# Start monitoring services
docker-compose -f docker-compose.monitoring.yml up -d

# Access services
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
# Jaeger: http://localhost:16686
```

## Development Tools

### API Testing

```bash
# Install HTTPie for API testing
pip install httpie

# Test aggregation server endpoints
http GET localhost:8000/health
http POST localhost:8000/api/clients name=test-client type=sdr
http GET localhost:8000/api/models

# Test with authentication
http POST localhost:8000/api/auth/login username=admin password=admin
# Use returned token in subsequent requests
http GET localhost:8000/api/clients Authorization:"Bearer <token>"
```

### Load Testing

```bash
# Install locust for load testing
pip install locust

# Run load tests
cd tests/performance
locust -f load_test.py --host=http://localhost:8000
```

### Mock Data Generation

```bash
# Generate mock signal data
python scripts/generate_mock_data.py --samples 1000 --output data/mock_signals.json

# Generate mock clients
python scripts/generate_mock_clients.py --count 50 --output data/mock_clients.json

# Load mock data into system
python scripts/load_test_data.py
```

## Debugging

### Application Debugging

```python
# Add breakpoints in code
import pdb; pdb.set_trace()

# Or use ipdb for better debugging
import ipdb; ipdb.set_trace()

# Remote debugging with debugpy
import debugpy
debugpy.listen(5678)
debugpy.wait_for_client()
```

### Container Debugging

```bash
# Access container shell
docker-compose exec aggregation-server bash

# View container logs
docker-compose logs -f aggregation-server

# Debug container startup issues
docker-compose up aggregation-server  # Without -d flag

# Check container resource usage
docker stats
```

### Network Debugging

```bash
# Test container networking
docker-compose exec aggregation-server ping edge-coordinator
docker-compose exec aggregation-server nslookup postgres

# Check port bindings
docker-compose port aggregation-server 8000
netstat -tulpn | grep 8000
```

## Performance Profiling

### Python Profiling

```python
# Profile with cProfile
import cProfile
import pstats

profiler = cProfile.Profile()
profiler.enable()

# Your code here

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)
```

### Memory Profiling

```bash
# Install memory profiler
pip install memory-profiler

# Profile memory usage
python -m memory_profiler src/aggregation_server/main.py

# Line-by-line memory profiling
@profile
def your_function():
    # Your code here
    pass
```

## Hot Reloading

### Python Hot Reloading

```bash
# Install watchdog for file watching
pip install watchdog

# Use uvicorn with reload for FastAPI
uvicorn src.aggregation_server.main:app --reload --host 0.0.0.0 --port 8000

# Or use nodemon-like tool
pip install nodemon
nodemon --exec python src/aggregation_server/main.py
```

### Docker Hot Reloading

```yaml
# docker-compose.dev.yml
version: '3.8'
services:
  aggregation-server:
    build:
      context: .
      dockerfile: docker/Dockerfile.aggregation-server
      target: development
    volumes:
      - ./src:/app/src
      - ./config:/app/config
    environment:
      - PYTHONPATH=/app/src
      - DEBUG=true
    command: uvicorn src.aggregation_server.main:app --reload --host 0.0.0.0 --port 8000
```

## Troubleshooting

### Common Issues

1. **Port Already in Use**
   ```bash
   # Find process using port
   lsof -i :8000  # macOS/Linux
   netstat -ano | findstr :8000  # Windows
   
   # Kill process
   kill -9 <PID>  # macOS/Linux
   taskkill /PID <PID> /F  # Windows
   ```

2. **Docker Issues**
   ```bash
   # Clean up Docker
   docker system prune -a
   docker volume prune
   
   # Rebuild containers
   docker-compose down
   docker-compose build --no-cache
   docker-compose up -d
   ```

3. **Python Import Issues**
   ```bash
   # Check PYTHONPATH
   echo $PYTHONPATH
   
   # Add to PYTHONPATH
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   ```

4. **Database Connection Issues**
   ```bash
   # Check database status
   docker-compose exec postgres pg_isready
   
   # Reset database
   docker-compose down postgres
   docker volume rm advanced-federated-pipeline_postgres_data
   docker-compose up -d postgres
   ```

### Performance Issues

```bash
# Monitor system resources
htop  # Linux/macOS
# Or use built-in tools
top
iostat 1
vmstat 1

# Monitor Docker resources
docker stats

# Check disk space
df -h
du -sh ./*
```

### Logging Configuration

```python
# logging_config.py
import logging
import sys

def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('logs/app.log')
        ]
    )
    
    # Reduce noise from third-party libraries
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('docker').setLevel(logging.WARNING)
```

## Development Best Practices

### Code Organization

```
src/
├── common/              # Shared utilities and interfaces
├── aggregation_server/  # Central server implementation
├── edge_coordinator/    # Edge coordination service
├── sdr_client/         # SDR hardware client
├── mobile_client/      # Mobile device client
└── monitoring/         # Monitoring and metrics

tests/
├── unit/               # Unit tests
├── integration/        # Integration tests
├── performance/        # Performance tests
└── fixtures/           # Test data and fixtures
```

### Git Workflow

```bash
# Create feature branch
git checkout -b feature/new-aggregation-algorithm

# Make changes and commit
git add .
git commit -m "feat: implement Byzantine-fault-tolerant aggregation"

# Push and create pull request
git push origin feature/new-aggregation-algorithm
```

### Environment Management

```bash
# Use different environments
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d  # Development
docker-compose -f docker-compose.yml -f docker-compose.test.yml up -d  # Testing
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d  # Production
```

This completes the local development guide with comprehensive setup instructions, development workflows, debugging techniques, and best practices.