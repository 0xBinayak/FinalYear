#!/bin/bash

# Production Backup Script for Federated Pipeline
# This script creates comprehensive backups of all system components

set -euo pipefail

# Configuration
BACKUP_BASE_DIR="${BACKUP_DIR:-/backup/federated-pipeline}"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_NAME="federated-pipeline-backup-$DATE"
BACKUP_DIR="$BACKUP_BASE_DIR/$BACKUP_NAME"
RETENTION_DAYS="${BACKUP_RETENTION_DAYS:-30}"
CLOUD_BACKUP_ENABLED="${CLOUD_BACKUP_ENABLED:-false}"
CLOUD_BUCKET="${CLOUD_BUCKET:-}"
ENCRYPTION_ENABLED="${BACKUP_ENCRYPTION_ENABLED:-true}"
ENCRYPTION_KEY="${BACKUP_ENCRYPTION_KEY:-}"

# Logging
LOG_FILE="/var/log/federated-pipeline/backup-$DATE.log"
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

error() {
    log "ERROR: $1"
    exit 1
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check if Docker is running
    if ! docker info >/dev/null 2>&1; then
        error "Docker is not running"
    fi
    
    # Check if docker-compose is available
    if ! command -v docker-compose >/dev/null 2>&1; then
        error "docker-compose is not installed"
    fi
    
    # Check backup directory
    mkdir -p "$BACKUP_DIR"
    if [[ ! -w "$BACKUP_DIR" ]]; then
        error "Backup directory is not writable: $BACKUP_DIR"
    fi
    
    # Check encryption key if encryption is enabled
    if [[ "$ENCRYPTION_ENABLED" == "true" && -z "$ENCRYPTION_KEY" ]]; then
        error "Encryption is enabled but no encryption key provided"
    fi
    
    log "Prerequisites check passed"
}

# Create database backup
backup_database() {
    log "Starting database backup..."
    
    local db_backup_file="$BACKUP_DIR/database.sql"
    
    # Create PostgreSQL dump
    docker-compose exec -T postgres pg_dump \
        -U "${POSTGRES_USER:-federated_user}" \
        -d "${POSTGRES_DB:-federated_pipeline}" \
        --verbose \
        --no-owner \
        --no-privileges \
        --clean \
        --if-exists > "$db_backup_file"
    
    if [[ $? -eq 0 ]]; then
        log "Database backup completed: $(du -h "$db_backup_file" | cut -f1)"
    else
        error "Database backup failed"
    fi
    
    # Compress database backup
    gzip "$db_backup_file"
    log "Database backup compressed"
}

# Create Redis backup
backup_redis() {
    log "Starting Redis backup..."
    
    local redis_backup_file="$BACKUP_DIR/redis.rdb"
    
    # Create Redis dump
    docker-compose exec -T redis redis-cli --rdb - > "$redis_backup_file"
    
    if [[ $? -eq 0 ]]; then
        log "Redis backup completed: $(du -h "$redis_backup_file" | cut -f1)"
    else
        error "Redis backup failed"
    fi
    
    # Compress Redis backup
    gzip "$redis_backup_file"
    log "Redis backup compressed"
}

# Backup configuration files
backup_configuration() {
    log "Starting configuration backup..."
    
    local config_backup_file="$BACKUP_DIR/configuration.tar.gz"
    
    # Create configuration archive
    tar -czf "$config_backup_file" \
        config/ \
        .env \
        docker-compose*.yml \
        monitoring/ \
        ssl/ \
        2>/dev/null || true
    
    if [[ -f "$config_backup_file" ]]; then
        log "Configuration backup completed: $(du -h "$config_backup_file" | cut -f1)"
    else
        error "Configuration backup failed"
    fi
}

# Backup model storage
backup_models() {
    log "Starting model storage backup..."
    
    local models_backup_file="$BACKUP_DIR/models.tar.gz"
    
    # Create models archive from container
    docker-compose exec -T aggregation-server tar -czf - /app/models 2>/dev/null > "$models_backup_file" || {
        log "Warning: Model storage backup failed or no models found"
        touch "$models_backup_file"
    }
    
    if [[ -f "$models_backup_file" ]]; then
        log "Model storage backup completed: $(du -h "$models_backup_file" | cut -f1)"
    fi
}

# Backup application data
backup_application_data() {
    log "Starting application data backup..."
    
    local app_data_backup_file="$BACKUP_DIR/application-data.tar.gz"
    
    # Create application data archive
    docker-compose exec -T aggregation-server tar -czf - \
        /app/data \
        /app/logs \
        /var/log/federated-pipeline \
        2>/dev/null > "$app_data_backup_file" || {
        log "Warning: Application data backup failed or no data found"
        touch "$app_data_backup_file"
    }
    
    if [[ -f "$app_data_backup_file" ]]; then
        log "Application data backup completed: $(du -h "$app_data_backup_file" | cut -f1)"
    fi
}

# Backup SSL certificates
backup_ssl_certificates() {
    log "Starting SSL certificates backup..."
    
    local ssl_backup_file="$BACKUP_DIR/ssl-certificates.tar.gz"
    
    # Create SSL certificates archive
    if [[ -d "ssl" ]]; then
        tar -czf "$ssl_backup_file" ssl/ 2>/dev/null || {
            log "Warning: SSL certificates backup failed"
            touch "$ssl_backup_file"
        }
        log "SSL certificates backup completed: $(du -h "$ssl_backup_file" | cut -f1)"
    else
        log "No SSL certificates directory found"
        touch "$ssl_backup_file"
    fi
}

# Create system metadata
create_metadata() {
    log "Creating backup metadata..."
    
    local metadata_file="$BACKUP_DIR/metadata.json"
    
    cat > "$metadata_file" << EOF
{
    "backup_name": "$BACKUP_NAME",
    "backup_date": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "backup_type": "full",
    "system_info": {
        "hostname": "$(hostname)",
        "os": "$(uname -s)",
        "kernel": "$(uname -r)",
        "docker_version": "$(docker --version)",
        "docker_compose_version": "$(docker-compose --version)"
    },
    "application_info": {
        "version": "$(docker-compose exec -T aggregation-server python -c 'import sys; print(sys.version)' 2>/dev/null || echo 'unknown')",
        "containers": $(docker-compose ps --format json 2>/dev/null || echo '[]')
    },
    "backup_components": [
        "database",
        "redis",
        "configuration",
        "models",
        "application_data",
        "ssl_certificates"
    ],
    "encryption_enabled": $ENCRYPTION_ENABLED,
    "retention_days": $RETENTION_DAYS
}
EOF
    
    log "Backup metadata created"
}

# Encrypt backup if enabled
encrypt_backup() {
    if [[ "$ENCRYPTION_ENABLED" != "true" ]]; then
        return 0
    fi
    
    log "Starting backup encryption..."
    
    local encrypted_backup_file="$BACKUP_BASE_DIR/$BACKUP_NAME.tar.gz.enc"
    
    # Create encrypted archive
    tar -czf - -C "$BACKUP_BASE_DIR" "$BACKUP_NAME" | \
        openssl enc -aes-256-cbc -salt -k "$ENCRYPTION_KEY" > "$encrypted_backup_file"
    
    if [[ $? -eq 0 ]]; then
        log "Backup encryption completed: $(du -h "$encrypted_backup_file" | cut -f1)"
        # Remove unencrypted backup
        rm -rf "$BACKUP_DIR"
        log "Unencrypted backup removed"
    else
        error "Backup encryption failed"
    fi
}

# Upload to cloud storage
upload_to_cloud() {
    if [[ "$CLOUD_BACKUP_ENABLED" != "true" || -z "$CLOUD_BUCKET" ]]; then
        return 0
    fi
    
    log "Starting cloud upload..."
    
    local backup_file
    if [[ "$ENCRYPTION_ENABLED" == "true" ]]; then
        backup_file="$BACKUP_BASE_DIR/$BACKUP_NAME.tar.gz.enc"
    else
        backup_file="$BACKUP_BASE_DIR/$BACKUP_NAME.tar.gz"
        # Create compressed archive if not encrypted
        tar -czf "$backup_file" -C "$BACKUP_BASE_DIR" "$BACKUP_NAME"
    fi
    
    # Upload based on cloud provider
    case "${CLOUD_PROVIDER:-aws}" in
        "aws")
            aws s3 cp "$backup_file" "s3://$CLOUD_BUCKET/backups/$(basename "$backup_file")"
            ;;
        "gcp")
            gsutil cp "$backup_file" "gs://$CLOUD_BUCKET/backups/$(basename "$backup_file")"
            ;;
        "azure")
            az storage blob upload --file "$backup_file" --container-name backups --name "$(basename "$backup_file")"
            ;;
        *)
            log "Warning: Unknown cloud provider: ${CLOUD_PROVIDER:-aws}"
            ;;
    esac
    
    if [[ $? -eq 0 ]]; then
        log "Cloud upload completed"
    else
        log "Warning: Cloud upload failed"
    fi
}

# Clean old backups
cleanup_old_backups() {
    log "Cleaning up old backups..."
    
    # Remove local backups older than retention period
    find "$BACKUP_BASE_DIR" -name "federated-pipeline-backup-*" -type d -mtime +$RETENTION_DAYS -exec rm -rf {} + 2>/dev/null || true
    find "$BACKUP_BASE_DIR" -name "federated-pipeline-backup-*.tar.gz*" -type f -mtime +$RETENTION_DAYS -delete 2>/dev/null || true
    
    # Clean cloud backups if enabled
    if [[ "$CLOUD_BACKUP_ENABLED" == "true" && -n "$CLOUD_BUCKET" ]]; then
        case "${CLOUD_PROVIDER:-aws}" in
            "aws")
                aws s3 ls "s3://$CLOUD_BUCKET/backups/" | \
                    awk '$1 < "'$(date -d "$RETENTION_DAYS days ago" +%Y-%m-%d)'" {print $4}' | \
                    xargs -I {} aws s3 rm "s3://$CLOUD_BUCKET/backups/{}" 2>/dev/null || true
                ;;
            *)
                log "Cloud cleanup not implemented for ${CLOUD_PROVIDER:-aws}"
                ;;
        esac
    fi
    
    log "Old backups cleaned up"
}

# Verify backup integrity
verify_backup() {
    log "Verifying backup integrity..."
    
    local backup_file
    if [[ "$ENCRYPTION_ENABLED" == "true" ]]; then
        backup_file="$BACKUP_BASE_DIR/$BACKUP_NAME.tar.gz.enc"
        # Test decryption
        openssl enc -aes-256-cbc -d -salt -k "$ENCRYPTION_KEY" -in "$backup_file" | tar -tzf - >/dev/null
    else
        backup_file="$BACKUP_BASE_DIR/$BACKUP_NAME.tar.gz"
        if [[ ! -f "$backup_file" ]]; then
            tar -czf "$backup_file" -C "$BACKUP_BASE_DIR" "$BACKUP_NAME"
        fi
        # Test archive
        tar -tzf "$backup_file" >/dev/null
    fi
    
    if [[ $? -eq 0 ]]; then
        log "Backup integrity verification passed"
    else
        error "Backup integrity verification failed"
    fi
}

# Send notification
send_notification() {
    local status=$1
    local message=$2
    
    if [[ -n "${NOTIFICATION_WEBHOOK:-}" ]]; then
        curl -X POST "$NOTIFICATION_WEBHOOK" \
            -H "Content-Type: application/json" \
            -d "{\"text\":\"Backup $status: $message\"}" \
            >/dev/null 2>&1 || true
    fi
    
    if [[ -n "${NOTIFICATION_EMAIL:-}" ]]; then
        echo "$message" | mail -s "Federated Pipeline Backup $status" "$NOTIFICATION_EMAIL" 2>/dev/null || true
    fi
}

# Main backup function
main() {
    local start_time=$(date +%s)
    
    log "Starting production backup: $BACKUP_NAME"
    
    trap 'error "Backup interrupted"' INT TERM
    
    check_prerequisites
    backup_database
    backup_redis
    backup_configuration
    backup_models
    backup_application_data
    backup_ssl_certificates
    create_metadata
    encrypt_backup
    verify_backup
    upload_to_cloud
    cleanup_old_backups
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    log "Backup completed successfully in ${duration}s"
    
    # Calculate backup size
    local backup_size
    if [[ "$ENCRYPTION_ENABLED" == "true" ]]; then
        backup_size=$(du -h "$BACKUP_BASE_DIR/$BACKUP_NAME.tar.gz.enc" | cut -f1)
    else
        backup_size=$(du -sh "$BACKUP_DIR" | cut -f1)
    fi
    
    local success_message="Backup $BACKUP_NAME completed successfully. Size: $backup_size, Duration: ${duration}s"
    log "$success_message"
    send_notification "SUCCESS" "$success_message"
}

# Error handling
handle_error() {
    local error_message="Backup $BACKUP_NAME failed: $1"
    log "$error_message"
    send_notification "FAILED" "$error_message"
    exit 1
}

trap 'handle_error "Unexpected error"' ERR

# Run main function
main "$@"