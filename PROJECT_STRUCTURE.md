# Project Structure

## Directory Layout

```
federated-pipeline/
├── src/                          # Source code
│   ├── aggregation_server/       # Central aggregation service
│   ├── edge_coordinator/          # Edge coordination service
│   ├── sdr_client/               # SDR hardware client
│   ├── mobile_client/            # Mobile device client
│   ├── infrastructure/           # Infrastructure services
│   └── common/                   # Shared components
│       ├── interfaces.py         # Base interfaces and data models
│       └── config.py            # Configuration management
├── config/                       # Configuration files
│   ├── base.yaml                # Base configuration
│   ├── development.yaml         # Development environment
│   └── production.yaml          # Production environment
├── docker/                       # Docker configurations
│   ├── Dockerfile.aggregation-server
│   ├── Dockerfile.edge-coordinator
│   ├── Dockerfile.sdr-client
│   └── Dockerfile.mobile-client
├── requirements/                 # Python dependencies
│   ├── base.txt                 # Common requirements
│   ├── aggregation-server.txt   # Server-specific requirements
│   ├── edge-coordinator.txt     # Edge coordinator requirements
│   ├── sdr-client.txt          # SDR client requirements
│   ├── mobile-client.txt       # Mobile client requirements
│   └── dev.txt                 # Development requirements
├── monitoring/                   # Monitoring configuration
│   └── prometheus.yml          # Prometheus configuration
├── tests/                       # Test files
├── docs/                        # Documentation
└── docker-compose.yml          # Container orchestration
```

## Component Architecture

### Aggregation Server
- Central coordination and model aggregation
- REST API and gRPC interfaces
- Byzantine-fault-tolerant aggregation
- Model storage and versioning

### Edge Coordinator
- Regional coordination for edge devices
- Local model aggregation
- Network partition handling
- Resource management

### SDR Client
- Hardware integration (RTL-SDR, HackRF, USRP)
- Real-time signal processing
- Feature extraction
- Local model training

### Mobile Client
- Cross-platform mobile support
- Battery optimization
- Network-aware training
- Simulated signal generation

### Infrastructure Services
- Configuration management
- Monitoring and metrics
- Logging and alerting
- Storage services

## Configuration Management

The system uses a hierarchical configuration system:
1. Base configuration (`config/base.yaml`)
2. Environment-specific overrides (`config/{environment}.yaml`)
3. Environment variable overrides

## Container Strategy

Multi-stage Docker builds for:
- Development: Hot-reloading and debugging tools
- Production: Optimized, secure containers
- Health checks and resource limits
- Hardware access for SDR clients

## Getting Started

1. Install Docker and Docker Compose
2. Clone the repository
3. Run `docker-compose up -d` to start all services
4. Access services:
   - Aggregation Server: http://localhost:8000
   - Edge Coordinator: http://localhost:8001
   - Prometheus: http://localhost:9091
   - Grafana: http://localhost:3000