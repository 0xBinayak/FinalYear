# Federated Learning System Test Suite

This directory contains comprehensive tests for the advanced federated learning pipeline system. The test suite is designed to validate all aspects of the system including functionality, performance, security, and reliability.

## Test Structure

```
tests/
├── conftest.py                 # Shared fixtures and configuration
├── unit/                       # Unit tests for individual components
│   ├── test_aggregation_server.py
│   ├── test_sdr_client.py
│   ├── test_edge_coordinator.py
│   ├── test_mobile_client.py
│   ├── test_common_components.py
│   ├── test_monitoring.py
│   └── test_mock_sdr_hardware.py
├── integration/                # Integration tests for workflows
│   └── test_federated_workflows.py
├── security/                   # Security and privacy tests
│   ├── test_byzantine_fault_tolerance.py
│   └── test_privacy_mechanisms.py
├── performance/                # Performance and stress tests
│   ├── test_scalability.py
│   └── test_stress_testing.py
└── README.md                   # This file
```

## Test Categories

### Unit Tests (`tests/unit/`)

Test individual components in isolation with >90% code coverage target.

**Key Test Files:**
- `test_aggregation_server.py`: Tests for central aggregation server
- `test_sdr_client.py`: Tests for SDR hardware clients
- `test_edge_coordinator.py`: Tests for edge coordination services
- `test_mobile_client.py`: Tests for mobile device clients
- `test_common_components.py`: Tests for shared utilities and data structures
- `test_monitoring.py`: Tests for monitoring and metrics collection
- `test_mock_sdr_hardware.py`: Tests for mock SDR hardware simulation

**Coverage Areas:**
- Component initialization and configuration
- Core functionality and algorithms
- Error handling and edge cases
- Data validation and processing
- Hardware abstraction layers

### Integration Tests (`tests/integration/`)

Test complete workflows and component interactions.

**Key Scenarios:**
- End-to-end federated learning rounds
- Hierarchical federated learning with edge coordinators
- Mixed client types (SDR, Mobile, Simulated)
- Network resilience and partition recovery
- Long-running stability tests

### Security Tests (`tests/security/`)

Validate security mechanisms and attack resistance.

**Security Areas:**
- **Byzantine Fault Tolerance**: Detection and mitigation of malicious clients
- **Privacy Mechanisms**: Differential privacy, secure aggregation
- **Attack Defense**: Model poisoning, gradient inversion, membership inference
- **Encryption**: Secure communication and data protection
- **Compliance**: GDPR, HIPAA, and audit trail validation

### Performance Tests (`tests/performance/`)

Measure system performance under various load conditions.

**Performance Areas:**
- **Scalability**: Client registration, model aggregation with varying client numbers
- **Network Conditions**: High latency, bandwidth limitations, intermittent connectivity
- **Resource Usage**: Memory consumption, CPU utilization, disk I/O
- **Concurrent Operations**: Multiple simultaneous federated learning rounds

### Stress Tests (`tests/performance/test_stress_testing.py`)

Test system behavior under extreme conditions.

**Stress Scenarios:**
- **Extreme Load**: Maximum client capacity, rapid client churn
- **Large Models**: Very large model updates and memory pressure
- **Resource Exhaustion**: Memory, CPU, and disk I/O stress
- **Failure Recovery**: Cascading failures, data corruption recovery

## Test Markers

Tests are organized using pytest markers for selective execution:

- `@pytest.mark.unit`: Unit tests
- `@pytest.mark.integration`: Integration tests
- `@pytest.mark.security`: Security tests
- `@pytest.mark.privacy`: Privacy-specific tests
- `@pytest.mark.performance`: Performance tests
- `@pytest.mark.stress`: Stress tests
- `@pytest.mark.slow`: Long-running tests
- `@pytest.mark.mock_sdr`: Tests using mock SDR hardware

## Running Tests

### Using the Test Runner Script

The recommended way to run tests is using the provided test runner script:

```bash
# Run all unit tests
python scripts/run_tests.py unit

# Run integration tests
python scripts/run_tests.py integration

# Run security tests
python scripts/run_tests.py security

# Run performance tests (may take a while)
python scripts/run_tests.py performance

# Run stress tests (resource intensive)
python scripts/run_tests.py stress

# Run fast tests only (excludes slow, performance, stress)
python scripts/run_tests.py fast

# Run all tests
python scripts/run_tests.py all

# Generate coverage report
python scripts/run_tests.py coverage --html-report
```

### Using pytest Directly

```bash
# Run specific test categories
pytest -m unit                    # Unit tests only
pytest -m "integration and not slow"  # Fast integration tests
pytest -m security               # Security tests
pytest -m "not stress"          # All tests except stress tests

# Run specific test files
pytest tests/unit/test_aggregation_server.py
pytest tests/security/test_privacy_mechanisms.py

# Run with coverage
pytest --cov=src --cov-report=html

# Run in parallel (requires pytest-xdist)
pytest -n auto

# Verbose output
pytest -v

# Stop on first failure
pytest -x
```

## Test Configuration

### Environment Variables

- `PYTEST_CURRENT_TEST`: Automatically set by pytest
- `FEDERATED_TEST_MODE`: Set to enable test-specific configurations

### Configuration Files

- `pytest.ini`: Main pytest configuration
- `tests/conftest.py`: Shared fixtures and test utilities

### Mock Hardware

The test suite includes comprehensive mock SDR hardware simulation:

- **Mock Devices**: RTL-SDR, HackRF, USRP simulation
- **Signal Generation**: Various modulation types and channel effects
- **Hardware Impairments**: IQ imbalance, DC offset, phase noise
- **Error Simulation**: Connection failures, read errors, calibration

## Performance Benchmarks

The test suite includes performance benchmarks for:

- **Client Registration**: < 0.1 seconds per client
- **Model Update Processing**: < 0.05 seconds per update
- **Aggregation**: < 0.02 seconds per client
- **Memory Usage**: < 0.5 MB per registered client
- **Concurrent Clients**: Support for 500+ clients
- **Model Size**: Support for models up to 50MB

## Security Validation

Security tests validate:

- **Byzantine Tolerance**: Up to 33% malicious clients
- **Privacy Protection**: Differential privacy with configurable ε
- **Attack Resistance**: Model poisoning, gradient inversion, membership inference
- **Secure Communication**: Encryption, digital signatures, integrity verification
- **Compliance**: Audit trails, data protection, regulatory compliance

## Continuous Integration

The test suite is designed for CI/CD integration:

- **Fast Tests**: Complete in < 5 minutes for quick feedback
- **Full Test Suite**: Comprehensive validation in < 30 minutes
- **Coverage Reports**: Automated coverage tracking with 90% target
- **Performance Regression**: Automated performance monitoring
- **Security Scanning**: Continuous security validation

## Test Data and Fixtures

### Shared Fixtures (`conftest.py`)

- `test_config`: Test-safe configuration
- `aggregation_server`: Initialized server instance
- `sample_client_info`: Mock client information
- `sample_model_update`: Mock model update
- `mock_sdr_hardware`: Mock SDR devices
- `byzantine_client_updates`: Malicious client updates for security testing
- `performance_benchmarks`: Performance comparison baselines

### Test Data Generation

- **Synthetic Signals**: Generated with realistic characteristics
- **Model Weights**: Various sizes and structures for testing
- **Network Conditions**: Simulated network scenarios
- **Attack Scenarios**: Comprehensive security test cases

## Debugging and Troubleshooting

### Common Issues

1. **Async Test Failures**: Ensure proper cleanup of async tasks
2. **Resource Leaks**: Use resource monitors to track memory/CPU usage
3. **Timing Issues**: Add appropriate delays for async operations
4. **Mock Hardware**: Verify mock device configuration

### Debug Tools

- **Verbose Output**: Use `-v` flag for detailed test output
- **Resource Monitoring**: Built-in resource usage tracking
- **Coverage Reports**: Identify untested code paths
- **Performance Profiling**: Detailed timing and resource metrics

### Test Isolation

- Each test runs in isolation with fresh fixtures
- Automatic cleanup of async tasks and resources
- Mock external dependencies to avoid side effects
- Temporary directories for file-based tests

## Contributing to Tests

### Adding New Tests

1. **Choose Appropriate Category**: Unit, integration, security, or performance
2. **Use Existing Fixtures**: Leverage shared fixtures from `conftest.py`
3. **Add Proper Markers**: Use pytest markers for categorization
4. **Include Documentation**: Add docstrings explaining test purpose
5. **Validate Coverage**: Ensure new code is covered by tests

### Test Quality Guidelines

- **Clear Test Names**: Descriptive test function names
- **Single Responsibility**: Each test should validate one specific behavior
- **Proper Assertions**: Use specific assertions with clear error messages
- **Resource Cleanup**: Ensure proper cleanup of resources
- **Performance Awareness**: Consider test execution time

### Mock Guidelines

- **Realistic Mocks**: Mock behavior should match real components
- **Comprehensive Coverage**: Mock all external dependencies
- **Error Scenarios**: Include failure modes in mocks
- **Configuration**: Make mocks configurable for different test scenarios

## Test Metrics and Reporting

### Coverage Metrics

- **Line Coverage**: Target >90% line coverage
- **Branch Coverage**: Validate all code paths
- **Function Coverage**: Ensure all functions are tested
- **Missing Coverage**: Identify untested code areas

### Performance Metrics

- **Execution Time**: Track test execution duration
- **Resource Usage**: Monitor memory and CPU consumption
- **Scalability**: Validate performance with increasing load
- **Regression Detection**: Compare against historical baselines

### Security Metrics

- **Attack Detection Rate**: Percentage of attacks detected
- **False Positive Rate**: Legitimate clients incorrectly flagged
- **Privacy Budget Tracking**: Differential privacy budget consumption
- **Compliance Validation**: Regulatory requirement adherence

This comprehensive test suite ensures the federated learning system is robust, secure, performant, and ready for production deployment.