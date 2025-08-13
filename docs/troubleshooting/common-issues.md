# Common Issues and Solutions

This guide covers the most frequently encountered issues and their solutions.

## System Won't Start

### Issue: Docker Compose Fails to Start

**Symptoms:**
- `docker-compose up` fails with errors
- Containers exit immediately
- Port binding errors

**Diagnosis:**
```bash
# Check Docker status
docker --version
docker-compose --version

# Check for port conflicts
netstat -tulpn | grep :8000
lsof -i :8000  # macOS/Linux

# Check Docker logs
docker-compose logs
```

**Solutions:**

1. **Port Already in Use**
   ```bash
   # Find and kill process using port
   sudo lsof -i :8000
   sudo kill -9 <PID>
   
   # Or change port in docker-compose.yml
   ports:
     - "8080:8000"  # Use different external port
   ```

2. **Insufficient Resources**
   ```bash
   # Check available resources
   docker system df
   docker system prune -a  # Clean up unused resources
   
   # Increase Docker memory limit (Docker Desktop)
   # Settings > Resources > Memory > 4GB+
   ```

3. **Permission Issues**
   ```bash
   # Fix Docker permissions (Linux)
   sudo usermod -aG docker $USER
   newgrp docker
   
   # Fix file permissions
   sudo chown -R $USER:$USER .
   ```

### Issue: Database Connection Failures

**Symptoms:**
- "Connection refused" errors
- Database timeout errors
- Migration failures

**Diagnosis:**
```bash
# Check PostgreSQL status
docker-compose exec postgres pg_isready

# Test connection
docker-compose exec postgres psql -U federated_user -d federated_pipeline -c "SELECT 1;"

# Check database logs
docker-compose logs postgres
```

**Solutions:**

1. **Database Not Ready**
   ```bash
   # Wait for database to be ready
   docker-compose up -d postgres
   sleep 30
   docker-compose up -d
   ```

2. **Wrong Credentials**
   ```bash
   # Check environment variables
   docker-compose exec aggregation-server env | grep POSTGRES
   
   # Update .env file
   POSTGRES_USER=federated_user
   POSTGRES_PASSWORD=secure_password
   POSTGRES_DB=federated_pipeline
   ```

3. **Database Corruption**
   ```bash
   # Reset database
   docker-compose down -v
   docker volume rm advanced-federated-pipeline_postgres_data
   docker-compose up -d postgres
   
   # Run migrations
   docker-compose exec aggregation-server python -m alembic upgrade head
   ```

## Client Connection Problems

### Issue: Clients Cannot Connect to Server

**Symptoms:**
- Client registration fails
- Connection timeout errors
- Authentication failures

**Diagnosis:**
```bash
# Test server connectivity
curl -v http://localhost:8000/health

# Check firewall rules
sudo ufw status  # Ubuntu
sudo iptables -L  # General Linux

# Test from client machine
telnet server-ip 8000
```

**Solutions:**

1. **Network Connectivity**
   ```bash
   # Check server is listening
   netstat -tulpn | grep :8000
   
   # Test with curl
   curl -I http://server-ip:8000/health
   
   # Check DNS resolution
   nslookup server-hostname
   ```

2. **Firewall Issues**
   ```bash
   # Open required ports
   sudo ufw allow 8000/tcp
   sudo ufw allow 8001/tcp
   
   # Or disable firewall temporarily for testing
   sudo ufw disable
   ```

3. **SSL/TLS Issues**
   ```bash
   # Test SSL connection
   openssl s_client -connect server-hostname:443
   
   # Check certificate validity
   curl -vI https://server-hostname/health
   ```

### Issue: Authentication Failures

**Symptoms:**
- "Invalid token" errors
- "Authentication required" errors
- Token expiration issues

**Diagnosis:**
```bash
# Test authentication endpoint
curl -X POST http://localhost:8000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin"}'

# Decode JWT token
echo "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9..." | base64 -d
```

**Solutions:**

1. **Invalid Credentials**
   ```bash
   # Reset admin password
   docker-compose exec aggregation-server python scripts/reset_admin_password.py
   
   # Create new user
   docker-compose exec aggregation-server python scripts/create_user.py --username newuser --password newpass
   ```

2. **Token Expiration**
   ```bash
   # Check token expiration in config
   JWT_EXPIRATION_HOURS=24
   
   # Get new token
   curl -X POST http://localhost:8000/api/auth/login \
     -H "Content-Type: application/json" \
     -d '{"username": "admin", "password": "admin"}'
   ```

3. **Clock Synchronization**
   ```bash
   # Sync system clock
   sudo ntpdate -s time.nist.gov
   
   # Or use systemd-timesyncd
   sudo systemctl enable systemd-timesyncd
   sudo systemctl start systemd-timesyncd
   ```

## Training Failures

### Issue: Training Rounds Fail to Complete

**Symptoms:**
- Training rounds stuck in "started" state
- Clients not participating
- Model aggregation failures

**Diagnosis:**
```bash
# Check training status
curl http://localhost:8000/api/training/rounds \
  -H "Authorization: Bearer <token>"

# Check client status
curl http://localhost:8000/api/clients \
  -H "Authorization: Bearer <token>"

# Check aggregation server logs
docker-compose logs aggregation-server | grep -i training
```

**Solutions:**

1. **Insufficient Clients**
   ```bash
   # Check minimum client requirement
   # Reduce min_clients in training config
   {
     "training_config": {
       "min_clients": 2,  # Reduced from 5
       "num_clients": 5
     }
   }
   ```

2. **Client Resource Issues**
   ```bash
   # Check client resources
   docker stats
   
   # Reduce model complexity
   {
     "hyperparameters": {
       "batch_size": 16,  # Reduced from 32
       "local_epochs": 3   # Reduced from 5
     }
   }
   ```

3. **Network Issues**
   ```bash
   # Check client connectivity
   curl http://client-ip:8002/health
   
   # Increase timeout values
   TRAINING_TIMEOUT=7200  # 2 hours
   CLIENT_TIMEOUT=300     # 5 minutes
   ```

### Issue: Model Convergence Problems

**Symptoms:**
- Accuracy not improving
- Loss increasing
- Divergent training

**Diagnosis:**
```bash
# Check training metrics
curl http://localhost:8000/api/metrics \
  -H "Authorization: Bearer <token>"

# Analyze model performance
python scripts/analyze_training_metrics.py --round-id round_123
```

**Solutions:**

1. **Learning Rate Issues**
   ```json
   {
     "hyperparameters": {
       "learning_rate": 0.0001,  // Reduced from 0.001
       "learning_rate_decay": 0.95,
       "adaptive_lr": true
     }
   }
   ```

2. **Data Quality Issues**
   ```bash
   # Check data distribution
   python scripts/analyze_data_distribution.py
   
   # Enable data quality filtering
   DATA_QUALITY_THRESHOLD=0.8
   ENABLE_DATA_VALIDATION=true
   ```

3. **Byzantine Clients**
   ```json
   {
     "aggregation_algorithm": "krum",  // More robust than fedavg
     "byzantine_tolerance": 0.33,
     "enable_anomaly_detection": true
   }
   ```

## Performance Issues

### Issue: Slow Training Performance

**Symptoms:**
- Long training round times
- High CPU/memory usage
- Network bottlenecks

**Diagnosis:**
```bash
# Monitor system resources
htop
iotop
nethogs

# Check Docker stats
docker stats

# Profile application
python -m cProfile -o profile.stats src/aggregation_server/main.py
```

**Solutions:**

1. **Resource Optimization**
   ```yaml
   # docker-compose.yml
   services:
     aggregation-server:
       deploy:
         resources:
           limits:
             memory: 4G
             cpus: '2.0'
           reservations:
             memory: 2G
             cpus: '1.0'
   ```

2. **Database Optimization**
   ```sql
   -- Add database indexes
   CREATE INDEX idx_clients_status ON clients(status);
   CREATE INDEX idx_training_rounds_timestamp ON training_rounds(start_time);
   
   -- Optimize queries
   EXPLAIN ANALYZE SELECT * FROM clients WHERE status = 'active';
   ```

3. **Network Optimization**
   ```bash
   # Enable compression
   ENABLE_COMPRESSION=true
   COMPRESSION_LEVEL=6
   
   # Use connection pooling
   DATABASE_POOL_SIZE=20
   REDIS_POOL_SIZE=10
   ```

### Issue: Memory Leaks

**Symptoms:**
- Gradually increasing memory usage
- Out of memory errors
- Container restarts

**Diagnosis:**
```bash
# Monitor memory usage over time
while true; do
  docker stats --no-stream | grep aggregation-server
  sleep 60
done

# Use memory profiler
pip install memory-profiler
python -m memory_profiler src/aggregation_server/main.py
```

**Solutions:**

1. **Python Memory Management**
   ```python
   import gc
   
   # Force garbage collection
   gc.collect()
   
   # Monitor memory usage
   import psutil
   process = psutil.Process()
   print(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")
   ```

2. **Container Limits**
   ```yaml
   services:
     aggregation-server:
       deploy:
         resources:
           limits:
             memory: 2G
       restart: unless-stopped
   ```

3. **Connection Management**
   ```python
   # Close database connections properly
   try:
       # Database operations
       pass
   finally:
       connection.close()
   
   # Use connection pooling
   from sqlalchemy.pool import QueuePool
   engine = create_engine(
       DATABASE_URL,
       poolclass=QueuePool,
       pool_size=10,
       max_overflow=20
   )
   ```

## SDR Hardware Issues

### Issue: SDR Device Not Detected

**Symptoms:**
- "No SDR device found" errors
- Device initialization failures
- Permission denied errors

**Diagnosis:**
```bash
# Check USB devices
lsusb | grep -i rtl
lsusb | grep -i hackrf

# Check device permissions
ls -la /dev/bus/usb/

# Test SDR tools
rtl_test -t
hackrf_info
```

**Solutions:**

1. **Driver Installation**
   ```bash
   # Install RTL-SDR drivers
   sudo apt install rtl-sdr librtlsdr-dev
   
   # Install HackRF drivers
   sudo apt install hackrf libhackrf-dev
   
   # Blacklist conflicting drivers
   echo 'blacklist dvb_usb_rtl28xxu' | sudo tee -a /etc/modprobe.d/blacklist-rtl.conf
   ```

2. **USB Permissions**
   ```bash
   # Add udev rules
   sudo tee /etc/udev/rules.d/20-rtlsdr.rules << EOF
   SUBSYSTEM=="usb", ATTRS{idVendor}=="0bda", ATTRS{idProduct}=="2838", GROUP="adm", MODE="0666", SYMLINK+="rtl_sdr"
   EOF
   
   # Reload udev rules
   sudo udevadm control --reload-rules
   sudo udevadm trigger
   ```

3. **Docker USB Access**
   ```yaml
   # docker-compose.yml
   services:
     sdr-client:
       devices:
         - /dev/bus/usb:/dev/bus/usb
       privileged: true  # Only if necessary
   ```

### Issue: Poor Signal Quality

**Symptoms:**
- Low SNR readings
- High error rates
- Inconsistent signal detection

**Diagnosis:**
```bash
# Test signal strength
rtl_power -f 100M:200M:1M -i 10 -g 40

# Check antenna connection
rtl_test -s 2400000 -d 0

# Monitor signal quality
python scripts/signal_quality_monitor.py
```

**Solutions:**

1. **Antenna Optimization**
   ```bash
   # Use appropriate antenna for frequency range
   # FM: Telescopic antenna
   # GSM: 900/1800 MHz antenna
   # WiFi: 2.4 GHz antenna
   
   # Check antenna connections
   # Ensure proper grounding
   ```

2. **Gain Settings**
   ```python
   # Automatic gain control
   sdr.set_gain_mode(True)
   
   # Manual gain optimization
   for gain in range(0, 50, 5):
       sdr.set_gain(gain)
       # Test signal quality
   ```

3. **Interference Mitigation**
   ```python
   # Frequency hopping
   frequencies = [100e6, 200e6, 300e6]
   
   # Notch filtering
   from scipy import signal
   b, a = signal.iirnotch(60, 30, fs=sample_rate)  # Remove 60Hz noise
   ```

## Database Issues

### Issue: Database Performance Degradation

**Symptoms:**
- Slow query responses
- High CPU usage on database
- Connection timeouts

**Diagnosis:**
```sql
-- Check slow queries
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;

-- Check database size
SELECT pg_size_pretty(pg_database_size('federated_pipeline'));

-- Check table sizes
SELECT schemaname, tablename, pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) as size
FROM pg_tables
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

**Solutions:**

1. **Query Optimization**
   ```sql
   -- Add missing indexes
   CREATE INDEX CONCURRENTLY idx_model_updates_timestamp ON model_updates(created_at);
   CREATE INDEX CONCURRENTLY idx_clients_last_seen ON clients(last_seen);
   
   -- Analyze query plans
   EXPLAIN (ANALYZE, BUFFERS) SELECT * FROM clients WHERE status = 'active';
   ```

2. **Database Maintenance**
   ```sql
   -- Update statistics
   ANALYZE;
   
   -- Vacuum tables
   VACUUM ANALYZE clients;
   VACUUM ANALYZE model_updates;
   
   -- Reindex if necessary
   REINDEX INDEX CONCURRENTLY idx_clients_status;
   ```

3. **Configuration Tuning**
   ```bash
   # postgresql.conf optimizations
   shared_buffers = 256MB
   effective_cache_size = 1GB
   work_mem = 4MB
   maintenance_work_mem = 64MB
   checkpoint_completion_target = 0.9
   wal_buffers = 16MB
   ```

This covers the most common issues and their solutions. For more specific problems, refer to the component-specific troubleshooting guides.