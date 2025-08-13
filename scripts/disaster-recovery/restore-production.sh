#!/bin/bash

# Production Restore Script for Federated Pipeline
# This script restores system from backup

set -euo pipefail

# Configuration
BACKUP_BASE_DIR="${BACKUP_DIR:-/backup/federated-pipeline}"
RESTORE_COMPONENTS="${RESTORE_COMPONENTS:-all}"
FORCE_RESTORE="${FORCE_RESTORE:-false}"
ENCRYPTION_ENABLED="${BACKUP_ENCRYPTION_ENABLED:-true}"
ENCRYPTION_KEY="${BACKUP_ENCRYPTION_KEY:-}"

# Logging
LOG_FILE="/var/log/federated-pipeline/restore-$(date +%Y%m%d_%H%M%S).log"
mkdir -p "$(dirname "$LOG_FILE")"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

error() {
    log "ERROR: $1"
    exit 1
}

usage() {
    cat << EOF
Usage: $0 [OPTIONS] BACKUP_NAME

Restore Federated Pipeline from backup.

OPTIONS:
    -c, --components COMPONENTS    Components to restore (all,database,redis,config,models,data,ssl)
    -f, --force                   Force restore without confirmation
    -h, --help                    Show this help message

EXAMPLES:
    $0 federated-pipeline-backup-20250113_120000
    $0 --components database,redis federated-pipeline-backup-20250113_120000
    $0 --force federated-pipeline-backup-20250113_120000

EOF
}

# Parse command line arguments
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -c|--components)
                RESTORE_COMPONENTS="$2"
                shift 2
                ;;
            -f|--force)
                FORCE_RESTORE="true"
                shift
                ;;
            -h|--help)
                usage
                exit 0
                ;;
            -*)
                error "Unknown option: $1"
                ;;
            *)
                if [[ -z "${BACKUP_NAME:-}" ]]; then
                    BACKUP_NAME="$1"
                else
                    error "Multiple backup names specified"
                fi
                shift
                ;;
        esac
    done
    
    if [[ -z "${BACKUP_NAME:-}" ]]; then
        error "Backup name is required"
    fi
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
    
    # Check backup exists
    local backup_file
    if [[ "$ENCRYPTION_ENABLED" == "true" ]]; then
        backup_file="$BACKUP_BASE_DIR/$BACKUP_NAME.tar.gz.enc"
    else
        backup_file="$BACKUP_BASE_DIR/$BACKUP_NAME.tar.gz"
        if [[ ! -f "$backup_file" ]]; then
            backup_file="$BACKUP_BASE_DIR/$BACKUP_NAME"
        fi
    fi
    
    if [[ ! -e "$backup_file" ]]; then
        error "Backup not found: $backup_file"
    fi
    
    # Check encryption key if needed
    if [[ "$ENCRYPTION_ENABLED" == "true" && -z "$ENCRYPTION_KEY" ]]; then
        error "Encryption key required for encrypted backup"
    fi
    
    log "Prerequisites check passed"
}

# Extract backup
extract_backup() {
    log "Extracting backup: $BACKUP_NAME"
    
    local backup_file
    local extract_dir="$BACKUP_BASE_DIR/restore-$(date +%Y%m%d_%H%M%S)"
    
    mkdir -p "$extract_dir"
    
    if [[ "$ENCRYPTION_ENABLED" == "true" ]]; then
        backup_file="$BACKUP_BASE_DIR/$BACKUP_NAME.tar.gz.enc"
        log "Decrypting and extracting encrypted backup..."
        openssl enc -aes-256-cbc -d -salt -k "$ENCRYPTION_KEY" -in "$backup_file" | \
            tar -xzf - -C "$extract_dir"
    else
        backup_file="$BACKUP_BASE_DIR/$BACKUP_NAME.tar.gz"
        if [[ -f "$backup_file" ]]; then
            log "Extracting compressed backup..."
            tar -xzf "$backup_file" -C "$extract_dir"
        else
            # Backup is already a directory
            backup_file="$BACKUP_BASE_DIR/$BACKUP_NAME"
            log "Using uncompressed backup directory..."
            cp -r "$backup_file" "$extract_dir/"
            BACKUP_NAME=$(basename "$backup_file")
        fi
    fi
    
    BACKUP_EXTRACT_DIR="$extract_dir/$BACKUP_NAME"
    
    if [[ ! -d "$BACKUP_EXTRACT_DIR" ]]; then
        error "Backup extraction failed or invalid backup structure"
    fi
    
    log "Backup extracted to: $BACKUP_EXTRACT_DIR"
}

# Verify backup integrity
verify_backup() {
    log "Verifying backup integrity..."
    
    local metadata_file="$BACKUP_EXTRACT_DIR/metadata.json"
    
    if [[ ! -f "$metadata_file" ]]; then
        log "Warning: No metadata file found, proceeding with basic verification"
        return 0
    fi
    
    # Check metadata
    local backup_date=$(jq -r '.backup_date' "$metadata_file" 2>/dev/null || echo "unknown")
    local backup_type=$(jq -r '.backup_type' "$metadata_file" 2>/dev/null || echo "unknown")
    local components=$(jq -r '.backup_components[]' "$metadata_file" 2>/dev/null || echo "")
    
    log "Backup date: $backup_date"
    log "Backup type: $backup_type"
    log "Components: $(echo $components | tr '\n' ' ')"
    
    # Verify component files exist
    for component in $components; do
        case $component in
            "database")
                [[ -f "$BACKUP_EXTRACT_DIR/database.sql.gz" ]] || log "Warning: Database backup file missing"
                ;;
            "redis")
                [[ -f "$BACKUP_EXTRACT_DIR/redis.rdb.gz" ]] || log "Warning: Redis backup file missing"
                ;;
            "configuration")
                [[ -f "$BACKUP_EXTRACT_DIR/configuration.tar.gz" ]] || log "Warning: Configuration backup file missing"
                ;;
            "models")
                [[ -f "$BACKUP_EXTRACT_DIR/models.tar.gz" ]] || log "Warning: Models backup file missing"
                ;;
            "application_data")
                [[ -f "$BACKUP_EXTRACT_DIR/application-data.tar.gz" ]] || log "Warning: Application data backup file missing"
                ;;
            "ssl_certificates")
                [[ -f "$BACKUP_EXTRACT_DIR/ssl-certificates.tar.gz" ]] || log "Warning: SSL certificates backup file missing"
                ;;
        esac
    done
    
    log "Backup integrity verification completed"
}

# Confirm restore operation
confirm_restore() {
    if [[ "$FORCE_RESTORE" == "true" ]]; then
        return 0
    fi
    
    echo
    echo "WARNING: This will restore the Federated Pipeline system from backup."
    echo "Current data will be replaced with backup data."
    echo
    echo "Backup: $BACKUP_NAME"
    echo "Components to restore: $RESTORE_COMPONENTS"
    echo
    read -p "Are you sure you want to continue? (yes/no): " -r
    echo
    
    if [[ ! $REPLY =~ ^[Yy][Ee][Ss]$ ]]; then
        log "Restore cancelled by user"
        exit 0
    fi
}

# Stop services
stop_services() {
    log "Stopping services..."
    
    docker-compose down || {
        log "Warning: Failed to stop some services"
    }
    
    # Wait for services to stop
    sleep 10
    
    log "Services stopped"
}

# Restore database
restore_database() {
    if [[ "$RESTORE_COMPONENTS" != "all" && "$RESTORE_COMPONENTS" != *"database"* ]]; then
        return 0
    fi
    
    log "Restoring database..."
    
    local db_backup_file="$BACKUP_EXTRACT_DIR/database.sql.gz"
    
    if [[ ! -f "$db_backup_file" ]]; then
        log "Warning: Database backup file not found, skipping database restore"
        return 0
    fi
    
    # Start only PostgreSQL
    docker-compose up -d postgres
    
    # Wait for PostgreSQL to be ready
    log "Waiting for PostgreSQL to be ready..."
    for i in {1..30}; do
        if docker-compose exec postgres pg_isready -U "${POSTGRES_USER:-federated_user}" >/dev/null 2>&1; then
            break
        fi
        sleep 2
    done
    
    # Drop and recreate database
    docker-compose exec postgres psql -U "${POSTGRES_USER:-federated_user}" -d postgres -c "DROP DATABASE IF EXISTS ${POSTGRES_DB:-federated_pipeline};" || true
    docker-compose exec postgres psql -U "${POSTGRES_USER:-federated_user}" -d postgres -c "CREATE DATABASE ${POSTGRES_DB:-federated_pipeline};"
    
    # Restore database
    gunzip -c "$db_backup_file" | docker-compose exec -T postgres psql -U "${POSTGRES_USER:-federated_user}" -d "${POSTGRES_DB:-federated_pipeline}"
    
    if [[ $? -eq 0 ]]; then
        log "Database restore completed"
    else
        error "Database restore failed"
    fi
}

# Restore Redis
restore_redis() {
    if [[ "$RESTORE_COMPONENTS" != "all" && "$RESTORE_COMPONENTS" != *"redis"* ]]; then
        return 0
    fi
    
    log "Restoring Redis..."
    
    local redis_backup_file="$BACKUP_EXTRACT_DIR/redis.rdb.gz"
    
    if [[ ! -f "$redis_backup_file" ]]; then
        log "Warning: Redis backup file not found, skipping Redis restore"
        return 0
    fi
    
    # Start only Redis
    docker-compose up -d redis
    
    # Wait for Redis to be ready
    log "Waiting for Redis to be ready..."
    for i in {1..30}; do
        if docker-compose exec redis redis-cli ping >/dev/null 2>&1; then
            break
        fi
        sleep 2
    done
    
    # Flush existing data
    docker-compose exec redis redis-cli FLUSHALL
    
    # Restore Redis data
    gunzip -c "$redis_backup_file" | docker-compose exec -T redis redis-cli --pipe
    
    if [[ $? -eq 0 ]]; then
        log "Redis restore completed"
    else
        error "Redis restore failed"
    fi
}

# Restore configuration
restore_configuration() {
    if [[ "$RESTORE_COMPONENTS" != "all" && "$RESTORE_COMPONENTS" != *"config"* ]]; then
        return 0
    fi
    
    log "Restoring configuration..."
    
    local config_backup_file="$BACKUP_EXTRACT_DIR/configuration.tar.gz"
    
    if [[ ! -f "$config_backup_file" ]]; then
        log "Warning: Configuration backup file not found, skipping configuration restore"
        return 0
    fi
    
    # Backup current configuration
    if [[ -d "config" ]]; then
        mv config "config.backup.$(date +%Y%m%d_%H%M%S)" || true
    fi
    
    # Extract configuration
    tar -xzf "$config_backup_file" -C .
    
    if [[ $? -eq 0 ]]; then
        log "Configuration restore completed"
    else
        error "Configuration restore failed"
    fi
}

# Restore models
restore_models() {
    if [[ "$RESTORE_COMPONENTS" != "all" && "$RESTORE_COMPONENTS" != *"models"* ]]; then
        return 0
    fi
    
    log "Restoring models..."
    
    local models_backup_file="$BACKUP_EXTRACT_DIR/models.tar.gz"
    
    if [[ ! -f "$models_backup_file" ]]; then
        log "Warning: Models backup file not found, skipping models restore"
        return 0
    fi
    
    # Start aggregation server temporarily
    docker-compose up -d aggregation-server
    
    # Wait for service to be ready
    sleep 10
    
    # Remove existing models and restore
    docker-compose exec aggregation-server rm -rf /app/models/* || true
    gunzip -c "$models_backup_file" | docker-compose exec -T aggregation-server tar -xzf - -C /
    
    if [[ $? -eq 0 ]]; then
        log "Models restore completed"
    else
        log "Warning: Models restore failed"
    fi
}

# Restore application data
restore_application_data() {
    if [[ "$RESTORE_COMPONENTS" != "all" && "$RESTORE_COMPONENTS" != *"data"* ]]; then
        return 0
    fi
    
    log "Restoring application data..."
    
    local app_data_backup_file="$BACKUP_EXTRACT_DIR/application-data.tar.gz"
    
    if [[ ! -f "$app_data_backup_file" ]]; then
        log "Warning: Application data backup file not found, skipping application data restore"
        return 0
    fi
    
    # Ensure aggregation server is running
    docker-compose up -d aggregation-server
    
    # Wait for service to be ready
    sleep 10
    
    # Restore application data
    gunzip -c "$app_data_backup_file" | docker-compose exec -T aggregation-server tar -xzf - -C /
    
    if [[ $? -eq 0 ]]; then
        log "Application data restore completed"
    else
        log "Warning: Application data restore failed"
    fi
}

# Restore SSL certificates
restore_ssl_certificates() {
    if [[ "$RESTORE_COMPONENTS" != "all" && "$RESTORE_COMPONENTS" != *"ssl"* ]]; then
        return 0
    fi
    
    log "Restoring SSL certificates..."
    
    local ssl_backup_file="$BACKUP_EXTRACT_DIR/ssl-certificates.tar.gz"
    
    if [[ ! -f "$ssl_backup_file" ]]; then
        log "Warning: SSL certificates backup file not found, skipping SSL certificates restore"
        return 0
    fi
    
    # Backup current SSL certificates
    if [[ -d "ssl" ]]; then
        mv ssl "ssl.backup.$(date +%Y%m%d_%H%M%S)" || true
    fi
    
    # Extract SSL certificates
    tar -xzf "$ssl_backup_file" -C .
    
    if [[ $? -eq 0 ]]; then
        log "SSL certificates restore completed"
    else
        error "SSL certificates restore failed"
    fi
}

# Start all services
start_services() {
    log "Starting all services..."
    
    docker-compose up -d
    
    # Wait for services to be ready
    log "Waiting for services to be ready..."
    sleep 30
    
    # Check service health
    local max_attempts=30
    local attempt=1
    
    while [[ $attempt -le $max_attempts ]]; do
        if curl -f http://localhost:8000/health >/dev/null 2>&1; then
            log "Services are ready"
            return 0
        fi
        
        log "Waiting for services... (attempt $attempt/$max_attempts)"
        sleep 10
        ((attempt++))
    done
    
    log "Warning: Services may not be fully ready"
}

# Verify restore
verify_restore() {
    log "Verifying restore..."
    
    # Check service health
    if curl -f http://localhost:8000/health >/dev/null 2>&1; then
        log "Aggregation server is healthy"
    else
        log "Warning: Aggregation server health check failed"
    fi
    
    # Check database connectivity
    if docker-compose exec postgres pg_isready -U "${POSTGRES_USER:-federated_user}" >/dev/null 2>&1; then
        log "Database is accessible"
    else
        log "Warning: Database connectivity check failed"
    fi
    
    # Check Redis connectivity
    if docker-compose exec redis redis-cli ping >/dev/null 2>&1; then
        log "Redis is accessible"
    else
        log "Warning: Redis connectivity check failed"
    fi
    
    log "Restore verification completed"
}

# Cleanup
cleanup() {
    log "Cleaning up temporary files..."
    
    if [[ -n "${BACKUP_EXTRACT_DIR:-}" && -d "$BACKUP_EXTRACT_DIR" ]]; then
        rm -rf "$(dirname "$BACKUP_EXTRACT_DIR")"
    fi
    
    log "Cleanup completed"
}

# Send notification
send_notification() {
    local status=$1
    local message=$2
    
    if [[ -n "${NOTIFICATION_WEBHOOK:-}" ]]; then
        curl -X POST "$NOTIFICATION_WEBHOOK" \
            -H "Content-Type: application/json" \
            -d "{\"text\":\"Restore $status: $message\"}" \
            >/dev/null 2>&1 || true
    fi
    
    if [[ -n "${NOTIFICATION_EMAIL:-}" ]]; then
        echo "$message" | mail -s "Federated Pipeline Restore $status" "$NOTIFICATION_EMAIL" 2>/dev/null || true
    fi
}

# Main restore function
main() {
    local start_time=$(date +%s)
    
    log "Starting restore: $BACKUP_NAME"
    log "Components to restore: $RESTORE_COMPONENTS"
    
    trap 'error "Restore interrupted"' INT TERM
    trap 'cleanup' EXIT
    
    check_prerequisites
    extract_backup
    verify_backup
    confirm_restore
    stop_services
    restore_database
    restore_redis
    restore_configuration
    restore_models
    restore_application_data
    restore_ssl_certificates
    start_services
    verify_restore
    
    local end_time=$(date +%s)
    local duration=$((end_time - start_time))
    
    local success_message="Restore from $BACKUP_NAME completed successfully in ${duration}s"
    log "$success_message"
    send_notification "SUCCESS" "$success_message"
}

# Error handling
handle_error() {
    local error_message="Restore from $BACKUP_NAME failed: $1"
    log "$error_message"
    send_notification "FAILED" "$error_message"
    cleanup
    exit 1
}

trap 'handle_error "Unexpected error"' ERR

# Parse arguments and run
parse_args "$@"
main