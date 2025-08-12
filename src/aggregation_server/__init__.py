"""
Aggregation Server Package

This package implements the central aggregation service for federated learning.
It provides:
- FastAPI-based REST server with health checks
- Client registration and authentication system
- Basic FedAvg aggregation algorithm
- Model storage and versioning system
"""

from .server import AggregationServer
from .main import app
from .models import (
    ClientRegistrationRequest,
    ClientRegistrationResponse,
    ModelUpdateRequest,
    ModelUpdateResponse,
    GlobalModelResponse,
    HealthResponse
)
from .auth import generate_client_token, verify_client_token
from .storage import ModelStorage
from .aggregation import FedAvgAggregator, AggregatorFactory

__all__ = [
    'AggregationServer',
    'app',
    'ClientRegistrationRequest',
    'ClientRegistrationResponse', 
    'ModelUpdateRequest',
    'ModelUpdateResponse',
    'GlobalModelResponse',
    'HealthResponse',
    'generate_client_token',
    'verify_client_token',
    'ModelStorage',
    'FedAvgAggregator',
    'AggregatorFactory'
]