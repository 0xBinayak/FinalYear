# Demonstrations

This directory contains demonstration scripts and examples for the Advanced Federated Pipeline system.

## Directory Structure

```
demos/
├── field_testing/          # Field testing demonstrations
├── real_world/            # Real-world scenario demonstrations
├── signal_processing/     # Signal processing demonstrations
├── integration/           # Integration demonstrations
├── performance/           # Performance demonstrations
├── security/              # Security demonstrations
└── dashboards/           # Dashboard demonstrations
```

## Running Demonstrations

### Prerequisites
- Python 3.8+
- Required dependencies installed (`pip install -r requirements/dev.txt`)
- System components running (see deployment guides)

### Quick Start

```bash
# Run basic field testing demo
python demos/field_testing/basic_field_test.py

# Run real-world federated learning demo
python demos/real_world/federated_learning_demo.py

# Run signal processing demo
python demos/signal_processing/signal_processing_demo.py

# Run full system integration demo
python demos/integration/full_system_demo.py

# Run security and privacy demo
python demos/security/privacy_security_demo.py
```

## Available Demonstrations

### 1. Field Testing (`field_testing/`)
- **basic_field_test.py**: Basic field testing with SDR data collection
- **advanced_field_test.py**: Advanced multi-environment testing
- **performance_benchmark.py**: Performance benchmarking against research

**Features:**
- Live SDR data collection
- Multi-environment validation
- Performance benchmarking
- Over-the-air testing

### 2. Real World (`real_world/`)
- **federated_learning_demo.py**: Complete federated learning workflow
- **dataset_integration_demo.py**: Real dataset integration (RadioML)
- **concept_drift_demo.py**: Concept drift detection and adaptation

**Features:**
- Real signal dataset integration
- Multi-location federated learning
- Model convergence analysis
- Concept drift handling

### 3. Signal Processing (`signal_processing/`)
- **signal_processing_demo.py**: Comprehensive signal processing pipeline
- **modulation_classification_demo.py**: Modulation classification showcase
- **adaptive_processing_demo.py**: Adaptive signal processing

**Features:**
- Feature extraction
- Channel simulation
- Modulation classification
- Wideband processing

### 4. Integration (`integration/`)
- **full_system_demo.py**: End-to-end system integration
- **component_interaction_demo.py**: Component interaction testing
- **workflow_demo.py**: Complete workflow demonstration

**Features:**
- Client registration
- Model updates and aggregation
- Global model distribution
- System monitoring

### 5. Performance (`performance/`)
- **scalability_demo.py**: System scalability testing
- **load_testing_demo.py**: Load testing scenarios
- **resource_utilization_demo.py**: Resource usage analysis

**Features:**
- Multi-client scalability
- Network performance testing
- Resource optimization
- Bottleneck identification

### 6. Security (`security/`)
- **privacy_security_demo.py**: Privacy and security features
- **byzantine_tolerance_demo.py**: Byzantine fault tolerance
- **audit_compliance_demo.py**: Audit logging and compliance

**Features:**
- Differential privacy
- Anomaly detection
- Byzantine fault tolerance
- Audit logging and compliance

### 7. Dashboards (`dashboards/`)
- Pre-configured dashboard templates
- Real-time monitoring visualizations
- Performance analytics dashboards
- Security monitoring dashboards

## Configuration

### Environment Variables
```bash
# Set demo configuration
export DEMO_CONFIG_PATH="config/demo.yaml"
export DEMO_OUTPUT_DIR="demo_output"
export DEMO_LOG_LEVEL="INFO"
```

### Configuration Files
Each demonstration can be configured using YAML files in the `config/` directory:
- `config/demo.yaml` - General demo configuration
- `config/field_testing.yaml` - Field testing specific settings
- `config/security_demo.yaml` - Security demo settings

### Example Configuration
```yaml
# config/demo.yaml
demo:
  duration_minutes: 30
  num_clients: 5
  enable_visualization: true
  save_results: true
  output_directory: "demo_output"

logging:
  level: "INFO"
  file: "demo.log"

network:
  mock_latency: true
  latency_range: [10, 50]  # ms
```

## Output and Results

### Output Directories
- `demo_output/` - Demonstration results and reports
- `demo_logs/` - Execution logs
- `demo_visualizations/` - Generated plots and charts

### Result Files
- **JSON Reports**: Structured results and metrics
- **CSV Data**: Time-series data for analysis
- **PNG/SVG Plots**: Visualization outputs
- **Log Files**: Detailed execution logs

### Example Output Structure
```
demo_output/
├── field_testing_results_20250113.json
├── performance_metrics.csv
├── visualizations/
│   ├── accuracy_over_time.png
│   ├── client_distribution_map.png
│   └── network_topology.svg
└── logs/
    ├── field_testing.log
    └── system_events.log
```

## Advanced Usage

### Custom Demonstrations
Create custom demonstrations by extending base classes:

```python
from demos.base import BaseDemonstration

class CustomDemo(BaseDemonstration):
    def __init__(self, config):
        super().__init__(config)
    
    async def run_demo(self):
        # Your custom demo logic
        pass
```

### Batch Execution
Run multiple demonstrations in sequence:

```bash
# Run all demonstrations
python demos/run_all_demos.py

# Run specific category
python demos/run_all_demos.py --category security

# Run with custom config
python demos/run_all_demos.py --config config/custom_demo.yaml
```

### Integration with CI/CD
Demonstrations can be integrated into CI/CD pipelines:

```yaml
# .github/workflows/demo-tests.yml
- name: Run Demonstrations
  run: |
    python demos/field_testing/basic_field_test.py --no-viz
    python demos/integration/full_system_demo.py --duration 5
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   # Ensure PYTHONPATH includes src directory
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   ```

2. **Missing Dependencies**
   ```bash
   # Install all demo dependencies
   pip install -r requirements/dev.txt
   pip install -r requirements/demo.txt
   ```

3. **SDR Hardware Issues**
   ```bash
   # Use mock mode if no hardware available
   export SDR_MOCK_MODE=true
   ```

4. **Network Connectivity**
   ```bash
   # Use local mode for offline testing
   export DEMO_LOCAL_MODE=true
   ```

### Getting Help

- Check the troubleshooting guide: `docs/troubleshooting/README.md`
- Review component-specific documentation in `docs/`
- Open an issue on GitHub for bugs or feature requests

## Contributing

To add new demonstrations:

1. Create a new directory under the appropriate category
2. Follow the naming convention: `*_demo.py`
3. Include comprehensive docstrings and comments
4. Add configuration options and error handling
5. Update this README with the new demonstration

### Demo Template
```python
#!/usr/bin/env python3
"""
Template for new demonstrations.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

def print_banner():
    """Print demonstration banner."""
    print("Your Demo Banner Here")

async def main():
    """Main demonstration function."""
    print_banner()
    
    try:
        # Your demo logic here
        print("✅ Demo completed successfully!")
        return True
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False

if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
```