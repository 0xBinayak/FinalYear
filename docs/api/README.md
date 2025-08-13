# API Documentation

This directory contains comprehensive API documentation for the Advanced Federated Pipeline system.

## Available Documentation

- [Aggregation Server API](aggregation-server-api.md) - Central coordination and model aggregation
- [Edge Coordinator API](edge-coordinator-api.md) - Regional coordination and edge management
- [SDR Client API](sdr-client-api.md) - Hardware-based signal processing client
- [Mobile Client API](mobile-client-api.md) - Software-based mobile client
- [Monitoring API](monitoring-api.md) - System monitoring and metrics

## Interactive Documentation

When running the system locally, interactive API documentation is available at:

- **Aggregation Server**: http://localhost:8000/docs (Swagger UI)
- **Edge Coordinator**: http://localhost:8001/docs (Swagger UI)
- **Monitoring Service**: http://localhost:9090/docs (Swagger UI)

## Authentication

All API endpoints require authentication using JWT tokens. See [Authentication Guide](authentication.md) for details.

## Rate Limiting

API endpoints are rate-limited to prevent abuse:
- Public endpoints: 100 requests/minute
- Authenticated endpoints: 1000 requests/minute
- Admin endpoints: 10000 requests/minute

## Error Handling

All APIs follow consistent error response format:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request parameters",
    "details": {
      "field": "client_id",
      "reason": "Required field missing"
    },
    "timestamp": "2025-01-13T10:30:00Z",
    "request_id": "req_123456789"
  }
}
```

## SDKs and Client Libraries

- [Python SDK](../sdks/python/README.md)
- [JavaScript SDK](../sdks/javascript/README.md)
- [Mobile SDK](../sdks/mobile/README.md)

## Examples

See the [examples](examples/) directory for complete integration examples in various programming languages.