# Aggregation Server API

The Aggregation Server provides the central coordination hub for federated learning operations.

## Base URL

```
Production: https://api.federated-pipeline.com
Development: http://localhost:8000
```

## Authentication

All endpoints require JWT authentication unless otherwise specified.

```bash
# Get authentication token
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "secure_password"}'

# Use token in subsequent requests
curl -H "Authorization: Bearer <token>" http://localhost:8000/api/clients
```

## Endpoints

### Health Check

#### GET /health

Check server health status.

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-13T10:30:00Z",
  "version": "1.0.0",
  "components": {
    "database": "healthy",
    "redis": "healthy",
    "storage": "healthy"
  }
}
```

**Example:**
```bash
curl http://localhost:8000/health
```

### Client Management

#### POST /api/clients

Register a new client in the federated learning system.

**Request Body:**
```json
{
  "client_id": "sdr-client-001",
  "client_type": "sdr",
  "capabilities": {
    "cpu_cores": 4,
    "memory_gb": 8,
    "gpu_available": false,
    "sdr_hardware": "rtlsdr"
  },
  "location": {
    "latitude": 37.7749,
    "longitude": -122.4194,
    "region": "us-west-1"
  },
  "network_info": {
    "connection_type": "wifi",
    "bandwidth_mbps": 100,
    "latency_ms": 20
  }
}
```

**Response:**
```json
{
  "client_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "client_id": "sdr-client-001",
  "registration_time": "2025-01-13T10:30:00Z",
  "status": "registered"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/clients \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "client_id": "sdr-client-001",
    "client_type": "sdr",
    "capabilities": {
      "cpu_cores": 4,
      "memory_gb": 8,
      "sdr_hardware": "rtlsdr"
    }
  }'
```

#### GET /api/clients

List all registered clients.

**Query Parameters:**
- `client_type` (optional): Filter by client type (sdr, mobile, edge)
- `status` (optional): Filter by status (active, inactive, training)
- `region` (optional): Filter by geographic region
- `limit` (optional): Number of results to return (default: 50)
- `offset` (optional): Pagination offset (default: 0)

**Response:**
```json
{
  "clients": [
    {
      "client_id": "sdr-client-001",
      "client_type": "sdr",
      "status": "active",
      "last_seen": "2025-01-13T10:25:00Z",
      "reputation_score": 0.95,
      "location": {
        "region": "us-west-1"
      }
    }
  ],
  "total": 1,
  "limit": 50,
  "offset": 0
}
```

**Example:**
```bash
curl "http://localhost:8000/api/clients?client_type=sdr&limit=10" \
  -H "Authorization: Bearer <token>"
```

#### GET /api/clients/{client_id}

Get detailed information about a specific client.

**Response:**
```json
{
  "client_id": "sdr-client-001",
  "client_type": "sdr",
  "status": "active",
  "capabilities": {
    "cpu_cores": 4,
    "memory_gb": 8,
    "gpu_available": false,
    "sdr_hardware": "rtlsdr"
  },
  "location": {
    "latitude": 37.7749,
    "longitude": -122.4194,
    "region": "us-west-1"
  },
  "statistics": {
    "total_training_rounds": 150,
    "successful_updates": 148,
    "average_training_time": 45.2,
    "data_quality_score": 0.92
  },
  "last_seen": "2025-01-13T10:25:00Z",
  "reputation_score": 0.95
}
```

**Example:**
```bash
curl http://localhost:8000/api/clients/sdr-client-001 \
  -H "Authorization: Bearer <token>"
```

### Model Management

#### POST /api/models

Upload a new global model or model update.

**Request Body (multipart/form-data):**
- `model_file`: Binary model file
- `metadata`: JSON metadata

**Metadata JSON:**
```json
{
  "model_version": "1.0.0",
  "architecture": "cnn",
  "framework": "pytorch",
  "input_shape": [1, 2, 1024],
  "num_classes": 11,
  "compression": "gzip"
}
```

**Response:**
```json
{
  "model_id": "model_123456",
  "version": "1.0.0",
  "upload_time": "2025-01-13T10:30:00Z",
  "size_bytes": 2048576,
  "checksum": "sha256:abc123..."
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/models \
  -H "Authorization: Bearer <token>" \
  -F "model_file=@model.pth" \
  -F 'metadata={"model_version":"1.0.0","architecture":"cnn"}'
```

#### GET /api/models

List available models.

**Query Parameters:**
- `version` (optional): Filter by model version
- `architecture` (optional): Filter by model architecture
- `limit` (optional): Number of results (default: 20)

**Response:**
```json
{
  "models": [
    {
      "model_id": "model_123456",
      "version": "1.0.0",
      "architecture": "cnn",
      "size_bytes": 2048576,
      "created_at": "2025-01-13T10:30:00Z",
      "performance_metrics": {
        "accuracy": 0.94,
        "f1_score": 0.92
      }
    }
  ],
  "total": 1
}
```

#### GET /api/models/{model_id}

Download a specific model.

**Response:** Binary model file with appropriate headers.

**Example:**
```bash
curl -o model.pth http://localhost:8000/api/models/model_123456 \
  -H "Authorization: Bearer <token>"
```

### Training Management

#### POST /api/training/start

Start a new federated learning round.

**Request Body:**
```json
{
  "model_id": "model_123456",
  "training_config": {
    "num_clients": 10,
    "min_clients": 5,
    "client_selection_strategy": "random",
    "aggregation_algorithm": "fedavg",
    "privacy_budget": 1.0,
    "max_training_time": 3600
  },
  "hyperparameters": {
    "learning_rate": 0.001,
    "batch_size": 32,
    "local_epochs": 5
  }
}
```

**Response:**
```json
{
  "training_round_id": "round_789",
  "status": "started",
  "selected_clients": [
    "sdr-client-001",
    "mobile-client-002"
  ],
  "start_time": "2025-01-13T10:30:00Z",
  "estimated_completion": "2025-01-13T11:30:00Z"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/training/start \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <token>" \
  -d '{
    "model_id": "model_123456",
    "training_config": {
      "num_clients": 10,
      "aggregation_algorithm": "fedavg"
    }
  }'
```

#### GET /api/training/rounds

List training rounds.

**Query Parameters:**
- `status` (optional): Filter by status (started, completed, failed)
- `limit` (optional): Number of results (default: 20)

**Response:**
```json
{
  "rounds": [
    {
      "training_round_id": "round_789",
      "status": "completed",
      "model_id": "model_123456",
      "start_time": "2025-01-13T10:30:00Z",
      "end_time": "2025-01-13T11:15:00Z",
      "participating_clients": 8,
      "successful_updates": 7,
      "aggregated_accuracy": 0.94
    }
  ],
  "total": 1
}
```

#### GET /api/training/rounds/{round_id}

Get detailed information about a training round.

**Response:**
```json
{
  "training_round_id": "round_789",
  "status": "completed",
  "model_id": "model_123456",
  "start_time": "2025-01-13T10:30:00Z",
  "end_time": "2025-01-13T11:15:00Z",
  "client_updates": [
    {
      "client_id": "sdr-client-001",
      "status": "completed",
      "training_time": 45.2,
      "data_samples": 1000,
      "local_accuracy": 0.92,
      "update_size_bytes": 102400
    }
  ],
  "aggregation_results": {
    "algorithm": "fedavg",
    "global_accuracy": 0.94,
    "convergence_metric": 0.001,
    "privacy_budget_used": 0.5
  }
}
```

### Model Updates

#### POST /api/updates

Submit a model update from a client.

**Request Body (multipart/form-data):**
- `update_file`: Binary model update file
- `metadata`: JSON metadata

**Metadata JSON:**
```json
{
  "client_id": "sdr-client-001",
  "training_round_id": "round_789",
  "training_metrics": {
    "local_accuracy": 0.92,
    "training_loss": 0.15,
    "training_time": 45.2,
    "data_samples": 1000
  },
  "data_statistics": {
    "class_distribution": [100, 150, 200, 120, 80],
    "signal_quality": 0.88
  },
  "privacy_metrics": {
    "differential_privacy_epsilon": 0.1,
    "noise_scale": 0.01
  }
}
```

**Response:**
```json
{
  "update_id": "update_456",
  "status": "received",
  "timestamp": "2025-01-13T10:45:00Z",
  "validation_status": "passed"
}
```

**Example:**
```bash
curl -X POST http://localhost:8000/api/updates \
  -H "Authorization: Bearer <token>" \
  -F "update_file=@model_update.pth" \
  -F 'metadata={"client_id":"sdr-client-001","training_round_id":"round_789"}'
```

### Metrics and Monitoring

#### GET /api/metrics

Get system-wide metrics.

**Query Parameters:**
- `start_time` (optional): Start time for metrics (ISO 8601)
- `end_time` (optional): End time for metrics (ISO 8601)
- `metric_type` (optional): Type of metrics (performance, privacy, security)

**Response:**
```json
{
  "metrics": {
    "system_performance": {
      "active_clients": 25,
      "training_rounds_completed": 150,
      "average_round_time": 1800,
      "global_model_accuracy": 0.94
    },
    "privacy_metrics": {
      "differential_privacy_budget_used": 0.5,
      "secure_aggregation_success_rate": 0.98
    },
    "security_metrics": {
      "byzantine_attacks_detected": 2,
      "anomalous_updates_rejected": 5
    }
  },
  "timestamp": "2025-01-13T10:30:00Z"
}
```

#### GET /api/metrics/clients/{client_id}

Get metrics for a specific client.

**Response:**
```json
{
  "client_id": "sdr-client-001",
  "metrics": {
    "participation_rate": 0.85,
    "average_accuracy": 0.92,
    "data_quality_score": 0.88,
    "reliability_score": 0.95,
    "communication_efficiency": 0.91
  },
  "recent_activity": [
    {
      "timestamp": "2025-01-13T10:30:00Z",
      "action": "model_update_submitted",
      "round_id": "round_789"
    }
  ]
}
```

## WebSocket API

### Real-time Updates

Connect to WebSocket for real-time system updates:

```javascript
const ws = new WebSocket('ws://localhost:8000/ws');

ws.onmessage = function(event) {
  const data = JSON.parse(event.data);
  console.log('Received:', data);
};

// Subscribe to specific events
ws.send(JSON.stringify({
  action: 'subscribe',
  events: ['training_started', 'client_connected', 'model_updated']
}));
```

**Event Types:**
- `training_started`: New training round started
- `training_completed`: Training round completed
- `client_connected`: New client registered
- `client_disconnected`: Client disconnected
- `model_updated`: Global model updated
- `anomaly_detected`: Security anomaly detected

## Error Codes

| Code | Description |
|------|-------------|
| `INVALID_REQUEST` | Request format is invalid |
| `AUTHENTICATION_FAILED` | Invalid or expired token |
| `AUTHORIZATION_DENIED` | Insufficient permissions |
| `CLIENT_NOT_FOUND` | Client ID not found |
| `MODEL_NOT_FOUND` | Model ID not found |
| `TRAINING_ROUND_NOT_FOUND` | Training round ID not found |
| `VALIDATION_ERROR` | Request validation failed |
| `RESOURCE_LIMIT_EXCEEDED` | Rate limit or quota exceeded |
| `INTERNAL_ERROR` | Internal server error |

## Rate Limits

| Endpoint Category | Limit |
|-------------------|-------|
| Authentication | 10 requests/minute |
| Client Registration | 5 requests/minute |
| Model Upload | 2 requests/minute |
| Model Updates | 100 requests/minute |
| Metrics | 1000 requests/minute |
| General API | 500 requests/minute |

## SDK Examples

### Python SDK

```python
from federated_pipeline import AggregationServerClient

client = AggregationServerClient(
    base_url="http://localhost:8000",
    api_key="your-api-key"
)

# Register a client
response = client.register_client({
    "client_id": "sdr-client-001",
    "client_type": "sdr",
    "capabilities": {
        "cpu_cores": 4,
        "memory_gb": 8
    }
})

# Start training round
training_round = client.start_training({
    "model_id": "model_123456",
    "num_clients": 10
})

# Submit model update
client.submit_update(
    update_file="model_update.pth",
    metadata={
        "client_id": "sdr-client-001",
        "training_round_id": training_round["training_round_id"]
    }
)
```

### JavaScript SDK

```javascript
import { AggregationServerClient } from '@federated-pipeline/js-sdk';

const client = new AggregationServerClient({
  baseUrl: 'http://localhost:8000',
  apiKey: 'your-api-key'
});

// Register a client
const response = await client.registerClient({
  clientId: 'mobile-client-001',
  clientType: 'mobile',
  capabilities: {
    cpuCores: 2,
    memoryGb: 4
  }
});

// Get training configuration
const config = await client.getTrainingConfig();

// Submit model update
await client.submitUpdate({
  updateFile: updateBlob,
  metadata: {
    clientId: 'mobile-client-001',
    trainingRoundId: 'round_789'
  }
});
```

This completes the comprehensive Aggregation Server API documentation with interactive examples and SDK usage.