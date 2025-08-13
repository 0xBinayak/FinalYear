# Deployment Guide

This directory contains deployment guides for the Advanced Federated Pipeline system across different environments.

## Available Guides

- [Cloud Deployment](cloud-deployment.md) - Deploy to AWS, Azure, GCP
- [Edge Deployment](edge-deployment.md) - Deploy to edge devices and Raspberry Pi
- [Mobile Deployment](mobile-deployment.md) - Deploy mobile clients
- [Local Development](local-development.md) - Set up local development environment

## Quick Start

For a quick local deployment:

```bash
# Clone the repository
git clone <repository-url>
cd advanced-federated-pipeline

# Start with Docker Compose
docker-compose up -d

# Verify deployment
curl http://localhost:8000/health
```

## Architecture Overview

The system consists of:
- **Aggregation Server**: Central coordination hub
- **Edge Coordinators**: Regional coordination nodes
- **SDR Clients**: Hardware-based signal processing
- **Mobile Clients**: Software-based mobile devices
- **Infrastructure Services**: Monitoring, configuration, storage

## Prerequisites

- Docker and Docker Compose
- Python 3.8+
- SDR hardware (for SDR clients)
- Kubernetes cluster (for production deployment)

## Support

For deployment issues, see [Troubleshooting Guide](../troubleshooting/README.md)