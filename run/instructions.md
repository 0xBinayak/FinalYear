# Advanced Federated Pipeline - Running Instructions

This comprehensive guide explains how to run the Advanced Federated Pipeline system after the recent restructuring.

## 🚀 Quick Start Guide

### 1. **Prerequisites Setup**

First, ensure you have the required dependencies:

```bash
# Install Python dependencies
pip install -r requirements/dev.txt

# Or install specific component requirements
pip install -r requirements/base.txt
pip install -r requirements/aggregation-server.txt
pip install -r requirements/edge-coordinator.txt
```

### 2. **Environment Setup**

```bash
# Set up environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Copy and configure environment file
cp .env.example .env
# Edit .env with your specific settings
```

## 🎯 Running Options

### **Option 1: Quick Demonstrations (Recommended for First Time)**

Run individual demonstrations to see different aspects of the system:

```bash
# 1. Basic field testing (5 minutes)
python demos/field_testing/basic_field_test.py

# 2. Signal processing pipeline (2 minutes)
python demos/signal_processing/signal_processing_demo.py

# 3. Quick real-world demo (3 minutes)
python demos/real_world/federated_learning_demo.py --quick

# 4. System integration demo (5 minutes)
python demos/integration/full_system_demo.py

# 5. Security and privacy demo (7 minutes)
python demos/security/privacy_security_demo.py
```

### **Option 2: Run All Demonstrations**

Execute the complete demonstration suite:

```bash
# Run all demos automatically
python demos/run_all_demos.py

# Run specific categories
python demos/run_all_demos.py --category integration security

# Run with custom timeout and output
python demos/run_all_demos.py --timeout 600 --output my_results.json
```

### **Option 3: Docker Environment (Full System)**

For a complete system deployment:

```bash
# Start the full system with Docker
docker-compose up -d

# Verify services are running
docker-compose ps

# Check logs
docker-compose logs -f

# Test the system
curl http://localhost:8000/health
curl http://localhost:8001/health
```

### **Option 4: Individual Components**

Run specific components separately:

```bash
# 1. Start Aggregation Server
cd src/aggregation_server
python main.py

# 2. Start Edge Coordinator (in another terminal)
cd src/edge_coordinator
python service.py

# 3. Start SDR Client (in another terminal)
cd src/sdr_client
python sdr_client.py

# 4. Start Mobile Client (in another terminal)
cd src/mobile_client
python mobile_client.py
```

## 🧪 Testing the System

### **Run Tests**

```bash
# Run all tests
pytest

# Run specific test categories
pytest tests/unit/          # Unit tests
pytest tests/integration/   # Integration tests
pytest tests/performance/   # Performance tests
pytest tests/security/      # Security tests

# Run with coverage
pytest --cov=src --cov-report=html
```

### **Run Specific Tests**

```bash
# Test aggregation server
pytest tests/unit/test_aggregation_server.py -v

# Test signal processing
pytest tests/unit/test_sdr_client.py -v

# Test security features
pytest tests/security/ -v
```

## 📊 Monitoring and Visualization

### **Start Monitoring Stack**

```bash
# Start monitoring services
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d

# Access dashboards
# Grafana: http://localhost:3000 (admin/admin)
# Prometheus: http://localhost:9090
# Jaeger: http://localhost:16686
```

## 🔧 Configuration Options

### **Development Configuration**

```bash
# Use development settings
export ENVIRONMENT=development

# Enable debug mode
export DEBUG=true
export LOG_LEVEL=DEBUG

# Use mock SDR hardware (if no physical SDR)
export SDR_MOCK_MODE=true
```

### **Production Configuration**

```bash
# Use production settings
export ENVIRONMENT=production

# Configure for production
cp config/production.yaml.template config/production.yaml
# Edit config/production.yaml with your settings
```

## 🎮 Interactive Usage Examples

### **Example 1: Quick System Test**

```bash
# 1. Start the system
docker-compose up -d

# 2. Run a quick integration test
python demos/integration/full_system_demo.py

# 3. Check results
curl http://localhost:8000/api/clients
```

### **Example 2: Signal Processing Demo**

```bash
# 1. Run signal processing demo
python demos/signal_processing/signal_processing_demo.py

# 2. Check generated visualizations (if enabled)
ls demo_output/visualizations/
```

### **Example 3: Security Testing**

```bash
# 1. Run security demonstration
python demos/security/privacy_security_demo.py

# 2. Check security reports
ls demo_output/security_reports/
```

## 🐛 Troubleshooting

### **Common Issues**

1. **Import Errors**
   ```bash
   # Fix Python path
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   ```

2. **Port Already in Use**
   ```bash
   # Check what's using the port
   lsof -i :8000  # macOS/Linux
   netstat -ano | findstr :8000  # Windows
   
   # Kill the process or use different ports
   ```

3. **Missing Dependencies**
   ```bash
   # Install all dependencies
   pip install -r requirements/dev.txt
   
   # For SDR hardware support
   pip install pyrtlsdr hackrf
   ```

4. **Docker Issues**
   ```bash
   # Clean up Docker
   docker-compose down
   docker system prune -f
   docker-compose up -d
   ```

### **Getting Help**

- Check the troubleshooting guide: `docs/troubleshooting/README.md`
- Review logs in `demo_output/logs/`
- Check component-specific documentation in `docs/`

## 📈 What to Expect

### **Demo Outputs**

When you run demonstrations, you'll see:

- **Console Output**: Real-time progress and results
- **JSON Reports**: Detailed results in `demo_output/`
- **Visualizations**: Charts and graphs (if enabled)
- **Log Files**: Detailed execution logs

### **Success Indicators**

Look for these signs that the system is working:

- ✅ Green checkmarks in demo output
- 🎉 "Completed successfully" messages
- HTTP 200 responses from health checks
- Generated result files in `demo_output/`

## 🎯 Recommended First Run

For your first time running the project, I recommend this sequence:

```bash
# 1. Quick setup
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
export SDR_MOCK_MODE=true

# 2. Run a simple demo first
python demos/signal_processing/signal_processing_demo.py

# 3. If that works, try the integration demo
python demos/integration/full_system_demo.py

# 4. Finally, run all demos
python demos/run_all_demos.py --category integration signal_processing
```

## 📋 Detailed Component Instructions

### **Aggregation Server**

The central coordination hub for federated learning:

```bash
# Start aggregation server
cd src/aggregation_server
python main.py

# Or with custom config
python main.py --config ../../config/production.yaml

# Test the server
curl http://localhost:8000/health
curl http://localhost:8000/api/clients
```

**Key Features:**
- Client registration and management
- Model aggregation using various algorithms
- Privacy-preserving mechanisms
- Security and anomaly detection

### **Edge Coordinator**

Regional coordination nodes for edge computing:

```bash
# Start edge coordinator
cd src/edge_coordinator
python service.py

# Or with specific region
python service.py --region us-west-1

# Test the coordinator
curl http://localhost:8001/health
```

**Key Features:**
- Local client coordination
- Offline operation support
- Resource management
- Data quality assurance

### **SDR Client**

Hardware-based signal processing client:

```bash
# Start SDR client (with real hardware)
cd src/sdr_client
python sdr_client.py --device rtlsdr

# Or with mock hardware
export SDR_MOCK_MODE=true
python sdr_client.py

# Test signal processing
python demo_signal_processing.py
```

**Key Features:**
- Real-time signal processing
- Multiple SDR hardware support
- Adaptive algorithms
- Hardware abstraction

### **Mobile Client**

Software-based mobile device client:

```bash
# Start mobile client
cd src/mobile_client
python mobile_client.py

# Or with specific configuration
python mobile_client.py --config mobile_config.yaml

# Test mobile optimizations
python test_mobile_optimizations.py
```

**Key Features:**
- Battery-aware processing
- Network optimization
- Privacy protection
- Cross-platform support

## 🔍 Advanced Usage

### **Custom Configuration**

Create custom configuration files:

```yaml
# config/custom.yaml
aggregation_server:
  host: "0.0.0.0"
  port: 8000
  max_clients: 100

federated_learning:
  aggregation_strategy: "fedavg"
  min_clients: 5
  privacy_budget: 1.0

privacy:
  enable_differential_privacy: true
  epsilon: 1.0
  delta: 1e-5
```

Use custom configuration:

```bash
python src/aggregation_server/main.py --config config/custom.yaml
```

### **Development Mode**

Enable development features:

```bash
# Set development environment
export ENVIRONMENT=development
export DEBUG=true
export LOG_LEVEL=DEBUG

# Enable hot reloading
export ENABLE_HOT_RELOAD=true

# Use mock hardware
export SDR_MOCK_MODE=true
export MOCK_NETWORK_CONDITIONS=true
```

### **Performance Monitoring**

Monitor system performance:

```bash
# Start with performance monitoring
python demos/performance/performance_monitor.py

# Generate performance reports
python scripts/generate_performance_report.py

# Analyze bottlenecks
python scripts/analyze_bottlenecks.py
```

### **Security Testing**

Test security features:

```bash
# Run security tests
pytest tests/security/ -v

# Test privacy mechanisms
python demos/security/privacy_security_demo.py

# Simulate attacks
python tests/security/test_byzantine_fault_tolerance.py
```

## 🚀 Deployment Options

### **Local Development**

```bash
# Quick local setup
docker-compose up -d
python demos/run_all_demos.py --category integration
```

### **Production Deployment**

```bash
# Use production configuration
cp config/production.yaml.template config/production.yaml
# Edit production.yaml with your settings

# Deploy with Docker
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Or deploy to Kubernetes
kubectl apply -f k8s/
```

### **Cloud Deployment**

```bash
# AWS deployment
scripts/deploy.sh --provider aws --region us-west-2

# Azure deployment
scripts/deploy.ps1 -Provider azure -Region eastus

# GCP deployment
scripts/deploy.sh --provider gcp --region us-central1
```

## 📊 Monitoring and Observability

### **Metrics Collection**

```bash
# Start monitoring stack
docker-compose -f docker-compose.monitoring.yml up -d

# Access monitoring dashboards
open http://localhost:3000  # Grafana
open http://localhost:9090  # Prometheus
open http://localhost:16686 # Jaeger
```

### **Log Analysis**

```bash
# View aggregated logs
docker-compose logs -f

# Analyze specific component logs
docker-compose logs aggregation-server
docker-compose logs edge-coordinator

# Search logs
grep "ERROR" logs/*.log
grep "federated_learning" logs/*.log
```

### **Performance Analysis**

```bash
# Generate performance report
python scripts/performance_analysis.py

# Monitor resource usage
python scripts/resource_monitor.py

# Analyze network performance
python scripts/network_analysis.py
```

## 🔒 Security and Privacy

### **Security Configuration**

```bash
# Enable security features
export ENABLE_SECURITY=true
export ENABLE_DIFFERENTIAL_PRIVACY=true
export ENABLE_SECURE_AGGREGATION=true

# Configure privacy parameters
export PRIVACY_EPSILON=1.0
export PRIVACY_DELTA=1e-5
```

### **Privacy Testing**

```bash
# Test differential privacy
python tests/security/test_privacy_mechanisms.py

# Test secure aggregation
python tests/security/test_secure_aggregation.py

# Generate privacy report
python scripts/generate_privacy_report.py
```

## 🎓 Learning and Exploration

### **Understanding the System**

1. **Start with Documentation**
   - Read `docs/user-guides/README.md`
   - Review `docs/deployment/local-development.md`
   - Check `docs/api/README.md`

2. **Run Demonstrations**
   - Begin with `demos/signal_processing/signal_processing_demo.py`
   - Progress to `demos/integration/full_system_demo.py`
   - Explore `demos/security/privacy_security_demo.py`

3. **Examine Code**
   - Study `src/common/interfaces.py` for data structures
   - Review `src/aggregation_server/server.py` for main logic
   - Explore `src/sdr_client/signal_processing.py` for algorithms

### **Experimentation**

```bash
# Experiment with different algorithms
python demos/real_world/federated_learning_demo.py --algorithm fedavg
python demos/real_world/federated_learning_demo.py --algorithm krum

# Test different privacy levels
python demos/security/privacy_security_demo.py --epsilon 0.1
python demos/security/privacy_security_demo.py --epsilon 1.0

# Simulate different network conditions
python demos/performance/network_simulation_demo.py --latency 100
python demos/performance/network_simulation_demo.py --bandwidth 10
```

## 🆘 Support and Resources

### **Documentation**
- **API Documentation**: `docs/api/`
- **Deployment Guides**: `docs/deployment/`
- **User Guides**: `docs/user-guides/`
- **Troubleshooting**: `docs/troubleshooting/`

### **Community and Support**
- **GitHub Issues**: Report bugs and request features
- **Documentation**: Comprehensive guides in `docs/`
- **Examples**: Working examples in `demos/`
- **Tests**: Reference implementations in `tests/`

### **Development Resources**
- **Source Code**: Well-documented code in `src/`
- **Configuration**: Templates in `config/`
- **Scripts**: Utility scripts in `scripts/`
- **Docker**: Container configurations in `docker/`

## 🎉 Success Checklist

After running the system, you should be able to:

- [ ] Start individual components successfully
- [ ] Run demonstrations without errors
- [ ] See green checkmarks in demo outputs
- [ ] Access monitoring dashboards
- [ ] Generate result reports
- [ ] View system health endpoints
- [ ] Execute tests successfully
- [ ] Understand system architecture

If you can check all these boxes, congratulations! You have successfully set up and run the Advanced Federated Pipeline system.

---

**Need Help?** 
- Check `docs/troubleshooting/common-issues.md`
- Review component logs in `demo_output/logs/`
- Examine test results for debugging information
- Consult the comprehensive documentation in `docs/`