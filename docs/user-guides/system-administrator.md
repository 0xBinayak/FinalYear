# System Administrator Guide

This guide is designed for system administrators responsible for installing, configuring, and maintaining the Advanced Federated Pipeline system.

## Overview

As a system administrator, you'll be responsible for:
- System installation and configuration
- User and access management
- System monitoring and maintenance
- Backup and disaster recovery
- Security hardening and compliance
- Performance optimization

## Prerequisites

### Technical Skills
- Linux/Unix system administration
- Docker and container orchestration
- Database administration (PostgreSQL)
- Network configuration and troubleshooting
- Security best practices

### System Requirements
- **Production**: 16GB RAM, 8 CPU cores, 500GB storage
- **Development**: 8GB RAM, 4 CPU cores, 100GB storage
- **Network**: Stable internet connection, open ports 8000, 8001, 443

## Installation

### 1. Initial System Setup

```bash
# Update system packages
sudo apt update && sudo apt upgrade -y

# Install required packages
sudo apt install -y docker.io docker-compose git curl wget

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Verify installation
docker --version
docker-compose --version
```

### 2. Clone and Configure

```bash
# Clone repository
git clone <repository-url>
cd advanced-federated-pipeline

# Copy configuration template
cp .env.example .env

# Edit configuration
nano .env
```

### 3. Environment Configuration

```bash
# .env - Production Configuration
# Server Configuration
AGGREGATION_SERVER_HOST=0.0.0.0
AGGREGATION_SERVER_PORT=8000
EDGE_COORDINATOR_PORT=8001

# Database Configuration
POSTGRES_HOST=postgres
POSTGRES_PORT=5432
POSTGRES_DB=federated_pipeline
POSTGRES_USER=federated_user
POSTGRES_PASSWORD=<generate-secure-password>

# Redis Configuration
REDIS_HOST=redis
REDIS_PORT=6379
REDIS_PASSWORD=<generate-secure-password>

# Security Configuration
JWT_SECRET=<generate-32-char-secret>
ENCRYPTION_KEY=<generate-32-char-key>
SSL_CERT_PATH=/etc/ssl/certs/federated-pipeline.crt
SSL_KEY_PATH=/etc/ssl/private/federated-pipeline.key

# Production Settings
DEBUG=false
LOG_LEVEL=INFO
ENVIRONMENT=production

# Monitoring
PROMETHEUS_ENABLED=true
GRAFANA_ENABLED=true
JAEGER_ENABLED=true
```

### 4. SSL Certificate Setup

```bash
# Generate self-signed certificate (development)
openssl req -x509 -newkey rsa:4096 -keyout ssl/private.key -out ssl/cert.crt -days 365 -nodes

# Or use Let's Encrypt (production)
sudo apt install certbot
sudo certbot certonly --standalone -d your-domain.com

# Copy certificates
sudo cp /etc/letsencrypt/live/your-domain.com/fullchain.pem ssl/cert.crt
sudo cp /etc/letsencrypt/live/your-domain.com/privkey.pem ssl/private.key
sudo chown $USER:$USER ssl/*
```

### 5. Deploy System

```bash
# Start production deployment
docker-compose -f docker-compose.yml -f docker-compose.prod.yml up -d

# Verify deployment
docker-compose ps
curl https://your-domain.com/health
```

## User Management

### 1. Create Admin User

```bash
# Create initial admin user
docker-compose exec aggregation-server python scripts/create_admin.py \
  --username admin \
  --password <secure-password> \
  --email admin@your-domain.com
```

### 2. User Roles and Permissions

```bash
# Available roles
# - admin: Full system access
# - operator: System monitoring and basic operations
# - developer: API access and development tools
# - viewer: Read-only access to dashboards

# Create users with specific roles
docker-compose exec aggregation-server python scripts/create_user.py \
  --username operator1 \
  --password <password> \
  --role operator \
  --email operator1@your-domain.com

docker-compose exec aggregation-server python scripts/create_user.py \
  --username dev1 \
  --password <password> \
  --role developer \
  --email dev1@your-domain.com
```

### 3. API Key Management

```bash
# Generate API keys for service accounts
docker-compose exec aggregation-server python scripts/generate_api_key.py \
  --name "monitoring-service" \
  --permissions "read:metrics,read:clients"

docker-compose exec aggregation-server python scripts/generate_api_key.py \
  --name "backup-service" \
  --permissions "read:all"

# List API keys
docker-compose exec aggregation-server python scripts/list_api_keys.py

# Revoke API key
docker-compose exec aggregation-server python scripts/revoke_api_key.py \
  --key-id <key-id>
```

## System Configuration

### 1. Database Configuration

```bash
# Database tuning for production
cat > config/postgresql.conf << EOF
# Memory settings
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB

# Checkpoint settings
checkpoint_completion_target = 0.9
wal_buffers = 16MB
checkpoint_timeout = 10min

# Connection settings
max_connections = 100
listen_addresses = '*'

# Logging
log_statement = 'mod'
log_min_duration_statement = 1000
log_line_prefix = '%t [%p]: [%l-1] user=%u,db=%d,app=%a,client=%h '
EOF

# Apply configuration
docker-compose restart postgres
```

### 2. Redis Configuration

```bash
# Redis tuning
cat > config/redis.conf << EOF
# Memory management
maxmemory 512mb
maxmemory-policy allkeys-lru

# Persistence
save 900 1
save 300 10
save 60 10000

# Security
requirepass <redis-password>
rename-command FLUSHDB ""
rename-command FLUSHALL ""

# Network
bind 0.0.0.0
port 6379
timeout 300
EOF

# Apply configuration
docker-compose restart redis
```

### 3. Application Configuration

```yaml
# config/production.yaml
server:
  host: "0.0.0.0"
  port: 8000
  workers: 4
  max_connections: 1000
  timeout: 30

database:
  pool_size: 20
  max_overflow: 30
  pool_timeout: 30
  pool_recycle: 3600

redis:
  pool_size: 10
  timeout: 5

logging:
  level: "INFO"
  format: "json"
  file: "/var/log/federated-pipeline/app.log"
  max_size: "100MB"
  backup_count: 5

security:
  jwt_expiration: 24  # hours
  password_min_length: 12
  max_login_attempts: 5
  lockout_duration: 300  # seconds

monitoring:
  metrics_interval: 60
  health_check_interval: 30
  alert_thresholds:
    cpu_usage: 80
    memory_usage: 85
    disk_usage: 90
    response_time: 5000  # ms
```

## Monitoring and Alerting

### 1. Prometheus Configuration

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "federated_learning_rules.yml"

scrape_configs:
  - job_name: 'aggregation-server'
    static_configs:
      - targets: ['aggregation-server:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'edge-coordinator'
    static_configs:
      - targets: ['edge-coordinator:8001']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093
```

### 2. Grafana Dashboard Setup

```bash
# Import pre-built dashboards
curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @monitoring/dashboards/system-overview.json

curl -X POST http://admin:admin@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @monitoring/dashboards/federated-learning.json

# Configure data sources
curl -X POST http://admin:admin@localhost:3000/api/datasources \
  -H "Content-Type: application/json" \
  -d '{
    "name": "Prometheus",
    "type": "prometheus",
    "url": "http://prometheus:9090",
    "access": "proxy",
    "isDefault": true
  }'
```

### 3. Alert Configuration

```yaml
# monitoring/alertmanager.yml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@your-domain.com'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
- name: 'web.hook'
  email_configs:
  - to: 'admin@your-domain.com'
    subject: 'Federated Pipeline Alert: {{ .GroupLabels.alertname }}'
    body: |
      {{ range .Alerts }}
      Alert: {{ .Annotations.summary }}
      Description: {{ .Annotations.description }}
      {{ end }}

  slack_configs:
  - api_url: 'YOUR_SLACK_WEBHOOK_URL'
    channel: '#alerts'
    title: 'Federated Pipeline Alert'
    text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
```

## Backup and Recovery

### 1. Automated Backup Script

```bash
#!/bin/bash
# backup.sh

BACKUP_DIR="/backup/federated-pipeline"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="federated-pipeline-backup-$DATE"

# Create backup directory
mkdir -p $BACKUP_DIR

# Database backup
docker-compose exec -T postgres pg_dump -U federated_user federated_pipeline | gzip > $BACKUP_DIR/$BACKUP_NAME-db.sql.gz

# Redis backup
docker-compose exec -T redis redis-cli --rdb - | gzip > $BACKUP_DIR/$BACKUP_NAME-redis.rdb.gz

# Configuration backup
tar -czf $BACKUP_DIR/$BACKUP_NAME-config.tar.gz config/ .env

# Model storage backup
docker-compose exec -T aggregation-server tar -czf - /app/models | gzip > $BACKUP_DIR/$BACKUP_NAME-models.tar.gz

# Upload to cloud storage (optional)
if [ "$CLOUD_BACKUP_ENABLED" = "true" ]; then
    aws s3 sync $BACKUP_DIR s3://your-backup-bucket/federated-pipeline/
fi

# Clean old backups (keep 30 days)
find $BACKUP_DIR -name "federated-pipeline-backup-*" -mtime +30 -delete

echo "Backup completed: $BACKUP_NAME"
```

### 2. Schedule Backups

```bash
# Add to crontab
crontab -e

# Daily backup at 2 AM
0 2 * * * /path/to/backup.sh >> /var/log/backup.log 2>&1

# Weekly full backup at 3 AM on Sundays
0 3 * * 0 /path/to/full-backup.sh >> /var/log/backup.log 2>&1
```

### 3. Recovery Procedures

```bash
#!/bin/bash
# restore.sh

BACKUP_NAME=$1
BACKUP_DIR="/backup/federated-pipeline"

if [ -z "$BACKUP_NAME" ]; then
    echo "Usage: $0 <backup-name>"
    exit 1
fi

# Stop services
docker-compose down

# Restore database
gunzip -c $BACKUP_DIR/$BACKUP_NAME-db.sql.gz | docker-compose exec -T postgres psql -U federated_user -d federated_pipeline

# Restore Redis
gunzip -c $BACKUP_DIR/$BACKUP_NAME-redis.rdb.gz | docker-compose exec -T redis redis-cli --pipe

# Restore configuration
tar -xzf $BACKUP_DIR/$BACKUP_NAME-config.tar.gz

# Restore models
gunzip -c $BACKUP_DIR/$BACKUP_NAME-models.tar.gz | docker-compose exec -T aggregation-server tar -xzf - -C /

# Start services
docker-compose up -d

echo "Restore completed from: $BACKUP_NAME"
```

## Security Hardening

### 1. System Security

```bash
# Update system packages regularly
sudo apt update && sudo apt upgrade -y

# Configure firewall
sudo ufw enable
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow 22/tcp    # SSH
sudo ufw allow 80/tcp    # HTTP
sudo ufw allow 443/tcp   # HTTPS
sudo ufw allow from 10.0.0.0/8 to any port 8000  # Internal network only

# Disable root login
sudo sed -i 's/PermitRootLogin yes/PermitRootLogin no/' /etc/ssh/sshd_config
sudo systemctl restart ssh

# Configure fail2ban
sudo apt install fail2ban
sudo systemctl enable fail2ban
sudo systemctl start fail2ban
```

### 2. Container Security

```yaml
# docker-compose.security.yml
version: '3.8'
services:
  aggregation-server:
    security_opt:
      - no-new-privileges:true
    read_only: true
    user: "1000:1000"
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE
    tmpfs:
      - /tmp:noexec,nosuid,size=100m
      - /var/run:noexec,nosuid,size=100m

  postgres:
    security_opt:
      - no-new-privileges:true
    user: "999:999"
    cap_drop:
      - ALL
    cap_add:
      - CHOWN
      - DAC_OVERRIDE
      - FOWNER
      - SETGID
      - SETUID
```

### 3. Network Security

```bash
# Create isolated Docker network
docker network create --driver bridge \
  --subnet=172.20.0.0/16 \
  --ip-range=172.20.240.0/20 \
  federated-pipeline-network

# Use network in docker-compose
networks:
  default:
    external:
      name: federated-pipeline-network
```

## Performance Optimization

### 1. System Tuning

```bash
# Kernel parameters
cat >> /etc/sysctl.conf << EOF
# Network optimization
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 87380 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216

# File descriptor limits
fs.file-max = 65536

# Memory management
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5
EOF

# Apply changes
sudo sysctl -p
```

### 2. Docker Optimization

```json
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "storage-driver": "overlay2",
  "storage-opts": [
    "overlay2.override_kernel_check=true"
  ],
  "default-ulimits": {
    "nofile": {
      "Name": "nofile",
      "Hard": 64000,
      "Soft": 64000
    }
  }
}
```

### 3. Application Tuning

```bash
# Environment variables for performance
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
export MALLOC_ARENA_MAX=2

# Gunicorn configuration
cat > config/gunicorn.conf.py << EOF
bind = "0.0.0.0:8000"
workers = 4
worker_class = "uvicorn.workers.UvicornWorker"
worker_connections = 1000
max_requests = 1000
max_requests_jitter = 100
timeout = 30
keepalive = 2
preload_app = True
EOF
```

## Maintenance Tasks

### 1. Daily Tasks

```bash
#!/bin/bash
# daily-maintenance.sh

# Check system health
curl -f http://localhost:8000/health || echo "Health check failed"

# Check disk space
df -h | awk '$5 > 80 {print "Disk usage warning: " $0}'

# Check container status
docker-compose ps | grep -v "Up" && echo "Container status warning"

# Clean Docker system
docker system prune -f

# Rotate logs
logrotate /etc/logrotate.d/federated-pipeline

# Update system metrics
python scripts/update_system_metrics.py
```

### 2. Weekly Tasks

```bash
#!/bin/bash
# weekly-maintenance.sh

# Database maintenance
docker-compose exec postgres psql -U federated_user -d federated_pipeline -c "VACUUM ANALYZE;"

# Update statistics
docker-compose exec postgres psql -U federated_user -d federated_pipeline -c "ANALYZE;"

# Check for security updates
sudo apt list --upgradable | grep -i security

# Generate system report
python scripts/generate_system_report.py --output /var/log/weekly-report-$(date +%Y%m%d).json
```

### 3. Monthly Tasks

```bash
#!/bin/bash
# monthly-maintenance.sh

# Full database backup
/path/to/full-backup.sh

# Security audit
python scripts/security_audit.py

# Performance analysis
python scripts/performance_analysis.py --period 30days

# Update documentation
git pull origin main
```

## Troubleshooting

### 1. Common Issues

**Service Won't Start:**
```bash
# Check logs
docker-compose logs service-name

# Check configuration
docker-compose config

# Restart service
docker-compose restart service-name
```

**Database Connection Issues:**
```bash
# Test connection
docker-compose exec postgres pg_isready

# Check credentials
docker-compose exec aggregation-server env | grep POSTGRES

# Reset database
docker-compose down postgres
docker volume rm advanced-federated-pipeline_postgres_data
docker-compose up -d postgres
```

**Performance Issues:**
```bash
# Monitor resources
docker stats
htop

# Check slow queries
docker-compose exec postgres psql -U federated_user -d federated_pipeline -c "SELECT query, mean_time FROM pg_stat_statements ORDER BY mean_time DESC LIMIT 10;"

# Analyze logs
grep -i "slow\|error\|timeout" /var/log/federated-pipeline/*.log
```

### 2. Emergency Procedures

**System Down:**
1. Check system status
2. Review recent changes
3. Check resource availability
4. Restart services
5. Escalate if needed

**Data Corruption:**
1. Stop all services immediately
2. Assess damage scope
3. Restore from backup
4. Verify data integrity
5. Resume operations

**Security Incident:**
1. Isolate affected systems
2. Preserve evidence
3. Notify security team
4. Follow incident response plan
5. Document lessons learned

This guide provides comprehensive coverage of system administration tasks. For specific technical issues, refer to the troubleshooting guides and operational runbooks.