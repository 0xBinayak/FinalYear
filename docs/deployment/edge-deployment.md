# Edge Deployment Guide

This guide covers deploying the Advanced Federated Pipeline system to edge devices including Raspberry Pi, industrial IoT gateways, and embedded systems.

## Supported Edge Platforms

- Raspberry Pi 4 (4GB+ RAM recommended)
- NVIDIA Jetson Nano/Xavier
- Intel NUC
- Industrial IoT gateways
- ARM64-based edge devices

## Prerequisites

### Hardware Requirements

**Minimum Requirements:**
- ARM64 or x86_64 processor
- 2GB RAM (4GB recommended)
- 16GB storage (32GB recommended)
- Network connectivity (WiFi or Ethernet)
- USB port for SDR hardware (optional)

**Recommended SDR Hardware:**
- RTL-SDR dongles (RTL2832U chipset)
- HackRF One
- USRP B200/B210
- BladeRF

### Software Requirements

- Docker Engine 20.10+
- Python 3.8+
- Git

## Raspberry Pi Deployment

### Step 1: Prepare Raspberry Pi

```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt install docker-compose -y

# Install Python dependencies
sudo apt install python3-pip python3-venv -y
```

### Step 2: Install SDR Drivers

```bash
# Install RTL-SDR drivers
sudo apt install rtl-sdr librtlsdr-dev -y

# Install HackRF drivers
sudo apt install hackrf libhackrf-dev -y

# Install UHD for USRP
sudo apt install uhd-host libuhd-dev -y

# Test SDR hardware
rtl_test -t  # For RTL-SDR
hackrf_info  # For HackRF
uhd_find_devices  # For USRP
```

### Step 3: Deploy Edge Coordinator

```bash
# Clone repository
git clone <repository-url>
cd advanced-federated-pipeline

# Create edge-specific configuration
cp config/edge.yaml.example config/edge.yaml

# Edit configuration
nano config/edge.yaml
```

```yaml
# config/edge.yaml
edge_coordinator:
  region: "edge-site-1"
  max_local_clients: 5
  aggregation_server_url: "https://your-cloud-server.com"
  offline_mode: true
  sync_interval: 300  # 5 minutes

sdr_client:
  enabled: true
  sdr_type: "rtlsdr"  # or "hackrf", "usrp"
  frequency_range: [88e6, 108e6]  # FM band
  sample_rate: 2.4e6

resources:
  cpu_limit: "1.0"
  memory_limit: "1Gi"
  storage_limit: "8Gi"

logging:
  level: "INFO"
  file: "/var/log/federated-pipeline.log"
```

### Step 4: Start Services

```bash
# Start edge coordinator
docker-compose -f docker-compose.edge.yml up -d

# Verify deployment
docker ps
curl http://localhost:8001/health
```

## NVIDIA Jetson Deployment

### Step 1: Prepare Jetson Device

```bash
# Install JetPack SDK
sudo apt update
sudo apt install nvidia-jetpack -y

# Install Docker (if not included)
sudo apt install docker.io -y
sudo systemctl enable docker
sudo usermod -aG docker $USER
```

### Step 2: GPU-Accelerated Configuration

```yaml
# docker-compose.jetson.yml
version: '3.8'
services:
  edge-coordinator:
    image: federated-pipeline/edge-coordinator:arm64
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - CUDA_VISIBLE_DEVICES=0
    volumes:
      - ./config:/app/config
      - /dev/bus/usb:/dev/bus/usb
    ports:
      - "8001:8001"
    
  sdr-client:
    image: federated-pipeline/sdr-client:arm64-cuda
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - SDR_TYPE=rtlsdr
      - GPU_ACCELERATION=true
    devices:
      - /dev/bus/usb:/dev/bus/usb
    volumes:
      - ./data:/app/data
```

### Step 3: Deploy with GPU Support

```bash
# Start with GPU acceleration
docker-compose -f docker-compose.jetson.yml up -d

# Monitor GPU usage
nvidia-smi
```

## Industrial IoT Gateway Deployment

### Step 1: Gateway Configuration

```bash
# For industrial gateways with limited resources
# Create minimal configuration
cat > config/industrial.yaml << EOF
edge_coordinator:
  region: "factory-floor-1"
  max_local_clients: 3
  resource_optimization: true
  low_power_mode: true

sdr_client:
  enabled: true
  processing_mode: "lightweight"
  batch_size: 32
  model_compression: true

monitoring:
  metrics_interval: 60
  log_rotation: true
  max_log_size: "10MB"
EOF
```

### Step 2: Lightweight Deployment

```yaml
# docker-compose.industrial.yml
version: '3.8'
services:
  edge-coordinator:
    image: federated-pipeline/edge-coordinator:alpine
    restart: unless-stopped
    environment:
      - CONFIG_FILE=/app/config/industrial.yaml
      - PYTHONUNBUFFERED=1
    volumes:
      - ./config:/app/config
      - ./data:/app/data
    ports:
      - "8001:8001"
    deploy:
      resources:
        limits:
          memory: 512M
          cpus: '0.5'
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

## Network Configuration

### Offline Operation Setup

```bash
# Configure offline synchronization
cat > scripts/sync-offline.sh << 'EOF'
#!/bin/bash

# Check internet connectivity
if ping -c 1 google.com &> /dev/null; then
    echo "Online - syncing with central server"
    
    # Upload local models
    curl -X POST http://central-server.com/api/models \
         -H "Content-Type: application/json" \
         -d @/app/data/local_model.json
    
    # Download global model
    curl -o /app/data/global_model.json \
         http://central-server.com/api/global-model
    
    echo "Sync completed"
else
    echo "Offline - continuing local operation"
fi
EOF

chmod +x scripts/sync-offline.sh

# Add to crontab for periodic sync
echo "*/5 * * * * /path/to/scripts/sync-offline.sh" | crontab -
```

### VPN Configuration

```bash
# Install WireGuard for secure connectivity
sudo apt install wireguard -y

# Configure VPN
sudo nano /etc/wireguard/wg0.conf
```

```ini
# /etc/wireguard/wg0.conf
[Interface]
PrivateKey = <edge-device-private-key>
Address = 10.0.0.2/24

[Peer]
PublicKey = <central-server-public-key>
Endpoint = central-server.com:51820
AllowedIPs = 10.0.0.0/24
PersistentKeepalive = 25
```

```bash
# Start VPN
sudo wg-quick up wg0
sudo systemctl enable wg-quick@wg0
```

## Resource Management

### CPU and Memory Optimization

```bash
# Create resource monitoring script
cat > scripts/resource-monitor.sh << 'EOF'
#!/bin/bash

# Monitor system resources
CPU_USAGE=$(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | cut -d'%' -f1)
MEM_USAGE=$(free | grep Mem | awk '{printf("%.1f", $3/$2 * 100.0)}')

# Adjust container resources based on usage
if (( $(echo "$CPU_USAGE > 80" | bc -l) )); then
    echo "High CPU usage detected: $CPU_USAGE%"
    docker update --cpus="0.5" federated-pipeline_edge-coordinator_1
fi

if (( $(echo "$MEM_USAGE > 80" | bc -l) )); then
    echo "High memory usage detected: $MEM_USAGE%"
    docker update --memory="512m" federated-pipeline_edge-coordinator_1
fi
EOF

chmod +x scripts/resource-monitor.sh

# Add to crontab
echo "*/1 * * * * /path/to/scripts/resource-monitor.sh" | crontab -
```

### Storage Management

```bash
# Create log rotation configuration
sudo nano /etc/logrotate.d/federated-pipeline
```

```
/var/log/federated-pipeline/*.log {
    daily
    rotate 7
    compress
    delaycompress
    missingok
    notifempty
    create 644 root root
    postrotate
        docker kill -s USR1 $(docker ps -q --filter name=federated-pipeline)
    endscript
}
```

## Security Hardening

### Firewall Configuration

```bash
# Configure UFW firewall
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow necessary ports
sudo ufw allow 8001/tcp  # Edge coordinator
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 51820/udp # WireGuard VPN

# Allow from specific networks only
sudo ufw allow from 10.0.0.0/24 to any port 8001
```

### Container Security

```yaml
# Security-hardened docker-compose
version: '3.8'
services:
  edge-coordinator:
    image: federated-pipeline/edge-coordinator:alpine
    user: "1000:1000"
    read_only: true
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    tmpfs:
      - /tmp:noexec,nosuid,size=100m
    volumes:
      - ./config:/app/config:ro
      - ./data:/app/data
```

## Monitoring and Maintenance

### Health Monitoring

```bash
# Create health check script
cat > scripts/health-check.sh << 'EOF'
#!/bin/bash

# Check container health
CONTAINERS=("edge-coordinator" "sdr-client")

for container in "${CONTAINERS[@]}"; do
    if ! docker ps | grep -q $container; then
        echo "Container $container is not running"
        docker-compose restart $container
    fi
done

# Check disk space
DISK_USAGE=$(df / | tail -1 | awk '{print $5}' | sed 's/%//')
if [ $DISK_USAGE -gt 80 ]; then
    echo "Disk usage is high: $DISK_USAGE%"
    # Clean up old logs and data
    find /var/log -name "*.log" -mtime +7 -delete
    docker system prune -f
fi

# Check memory usage
MEM_USAGE=$(free | grep Mem | awk '{printf("%.0f", $3/$2 * 100.0)}')
if [ $MEM_USAGE -gt 85 ]; then
    echo "Memory usage is high: $MEM_USAGE%"
    # Restart services to free memory
    docker-compose restart
fi
EOF

chmod +x scripts/health-check.sh

# Add to crontab for regular checks
echo "*/5 * * * * /path/to/scripts/health-check.sh" | crontab -
```

### Remote Management

```bash
# Install SSH for remote access
sudo apt install openssh-server -y
sudo systemctl enable ssh

# Configure SSH key authentication
mkdir -p ~/.ssh
echo "your-public-key" >> ~/.ssh/authorized_keys
chmod 600 ~/.ssh/authorized_keys
chmod 700 ~/.ssh

# Disable password authentication
sudo sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo systemctl restart ssh
```

## Troubleshooting

### Common Issues

1. **SDR Hardware Not Detected**
   ```bash
   # Check USB devices
   lsusb
   
   # Check SDR drivers
   rtl_test -t
   dmesg | grep rtl
   ```

2. **Container Startup Issues**
   ```bash
   # Check container logs
   docker logs edge-coordinator
   
   # Check system resources
   free -h
   df -h
   ```

3. **Network Connectivity Issues**
   ```bash
   # Test connectivity to central server
   ping central-server.com
   curl -I https://central-server.com/health
   
   # Check VPN status
   sudo wg show
   ```

### Performance Optimization

```bash
# Optimize for ARM devices
echo 'vm.swappiness=10' | sudo tee -a /etc/sysctl.conf
echo 'vm.vfs_cache_pressure=50' | sudo tee -a /etc/sysctl.conf

# Apply changes
sudo sysctl -p
```

## Backup and Recovery

### Automated Backup

```bash
# Create backup script
cat > scripts/backup-edge.sh << 'EOF'
#!/bin/bash

BACKUP_DIR="/backup/$(date +%Y%m%d)"
mkdir -p $BACKUP_DIR

# Backup configuration
cp -r config/ $BACKUP_DIR/

# Backup local models and data
docker exec edge-coordinator tar -czf /tmp/data-backup.tar.gz /app/data
docker cp edge-coordinator:/tmp/data-backup.tar.gz $BACKUP_DIR/

# Upload to central storage (if online)
if ping -c 1 central-server.com &> /dev/null; then
    rsync -avz $BACKUP_DIR/ backup-server:/backups/edge-devices/$(hostname)/
fi

# Clean old backups (keep 7 days)
find /backup -type d -mtime +7 -exec rm -rf {} +
EOF

chmod +x scripts/backup-edge.sh

# Schedule daily backups
echo "0 2 * * * /path/to/scripts/backup-edge.sh" | crontab -
```

This completes the edge deployment guide with comprehensive coverage of different edge platforms, security, monitoring, and maintenance procedures.