#!/bin/bash

# Federated Pipeline Deployment Script
# Supports multiple environments and deployment targets

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
DEFAULT_ENVIRONMENT="development"
DEFAULT_TARGET="docker-compose"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Help function
show_help() {
    cat << EOF
Federated Pipeline Deployment Script

Usage: $0 [OPTIONS] COMMAND

Commands:
    deploy          Deploy the application
    update          Update existing deployment
    rollback        Rollback to previous version
    status          Check deployment status
    logs            Show application logs
    cleanup         Clean up resources
    test            Run deployment tests

Options:
    -e, --environment ENV    Target environment (development|staging|production)
    -t, --target TARGET      Deployment target (docker-compose|kubernetes|helm)
    -v, --version VERSION    Application version to deploy
    -c, --config FILE        Custom configuration file
    -n, --namespace NS       Kubernetes namespace
    -d, --dry-run           Show what would be done without executing
    -h, --help              Show this help message

Examples:
    $0 deploy -e production -t kubernetes
    $0 update -e staging -v v1.2.3
    $0 rollback -e production -t helm
    $0 status -e development
    $0 logs -e staging --follow

Environment Variables:
    REGISTRY_URL            Container registry URL
    REGISTRY_USERNAME       Registry username
    REGISTRY_PASSWORD       Registry password
    KUBECONFIG             Kubernetes config file path
    HELM_CHART_VERSION     Helm chart version to use

EOF
}

# Parse command line arguments
parse_args() {
    ENVIRONMENT="$DEFAULT_ENVIRONMENT"
    TARGET="$DEFAULT_TARGET"
    VERSION=""
    CONFIG_FILE=""
    NAMESPACE=""
    DRY_RUN=false
    COMMAND=""
    FOLLOW_LOGS=false

    while [[ $# -gt 0 ]]; do
        case $1 in
            -e|--environment)
                ENVIRONMENT="$2"
                shift 2
                ;;
            -t|--target)
                TARGET="$2"
                shift 2
                ;;
            -v|--version)
                VERSION="$2"
                shift 2
                ;;
            -c|--config)
                CONFIG_FILE="$2"
                shift 2
                ;;
            -n|--namespace)
                NAMESPACE="$2"
                shift 2
                ;;
            -d|--dry-run)
                DRY_RUN=true
                shift
                ;;
            --follow)
                FOLLOW_LOGS=true
                shift
                ;;
            -h|--help)
                show_help
                exit 0
                ;;
            deploy|update|rollback|status|logs|cleanup|test)
                COMMAND="$1"
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_help
                exit 1
                ;;
        esac
    done

    if [[ -z "$COMMAND" ]]; then
        log_error "No command specified"
        show_help
        exit 1
    fi
}

# Validate environment
validate_environment() {
    case "$ENVIRONMENT" in
        development|staging|production)
            log_info "Deploying to environment: $ENVIRONMENT"
            ;;
        *)
            log_error "Invalid environment: $ENVIRONMENT"
            exit 1
            ;;
    esac
}

# Check prerequisites
check_prerequisites() {
    local missing_tools=()

    case "$TARGET" in
        docker-compose)
            if ! command -v docker-compose &> /dev/null; then
                missing_tools+=("docker-compose")
            fi
            ;;
        kubernetes)
            if ! command -v kubectl &> /dev/null; then
                missing_tools+=("kubectl")
            fi
            ;;
        helm)
            if ! command -v helm &> /dev/null; then
                missing_tools+=("helm")
            fi
            if ! command -v kubectl &> /dev/null; then
                missing_tools+=("kubectl")
            fi
            ;;
    esac

    if [[ ${#missing_tools[@]} -gt 0 ]]; then
        log_error "Missing required tools: ${missing_tools[*]}"
        exit 1
    fi
}

# Build images
build_images() {
    log_info "Building Docker images..."
    
    local components=("aggregation-server" "edge-coordinator" "sdr-client" "mobile-client" "metrics-collector")
    local registry_url="${REGISTRY_URL:-registry.company.com}"
    local image_tag="${VERSION:-latest}"
    
    for component in "${components[@]}"; do
        log_info "Building $component..."
        
        if [[ "$DRY_RUN" == "true" ]]; then
            log_info "[DRY RUN] Would build: $registry_url/federated-pipeline/$component:$image_tag"
        else
            docker build \
                -f "docker/Dockerfile.$component" \
                -t "$registry_url/federated-pipeline/$component:$image_tag" \
                --target production \
                .
            
            if [[ "$ENVIRONMENT" != "development" ]]; then
                docker push "$registry_url/federated-pipeline/$component:$image_tag"
            fi
        fi
    done
}

# Deploy with Docker Compose
deploy_docker_compose() {
    log_info "Deploying with Docker Compose..."
    
    local compose_file="docker-compose.yml"
    case "$ENVIRONMENT" in
        production)
            compose_file="docker-compose.prod.yml"
            ;;
        staging)
            compose_file="docker-compose.staging.yml"
            ;;
    esac
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would run: docker-compose -f $compose_file up -d"
        return
    fi
    
    # Set environment variables
    export ENVIRONMENT
    export VERSION="${VERSION:-latest}"
    
    # Deploy
    docker-compose -f "$compose_file" up -d
    
    # Wait for services to be healthy
    log_info "Waiting for services to be healthy..."
    docker-compose -f "$compose_file" ps
    
    # Run health checks
    sleep 30
    if curl -f http://localhost:8000/health &> /dev/null; then
        log_success "Aggregation server is healthy"
    else
        log_error "Aggregation server health check failed"
        return 1
    fi
}

# Deploy with Kubernetes
deploy_kubernetes() {
    log_info "Deploying with Kubernetes..."
    
    local namespace="${NAMESPACE:-federated-pipeline-$ENVIRONMENT}"
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would deploy to namespace: $namespace"
        return
    fi
    
    # Create namespace if it doesn't exist
    kubectl create namespace "$namespace" --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply configurations
    kubectl apply -f k8s/namespace.yaml -n "$namespace"
    kubectl apply -f k8s/configmap.yaml -n "$namespace"
    kubectl apply -f k8s/secrets.yaml -n "$namespace"
    kubectl apply -f k8s/postgres.yaml -n "$namespace"
    kubectl apply -f k8s/aggregation-server.yaml -n "$namespace"
    kubectl apply -f k8s/edge-coordinator.yaml -n "$namespace"
    
    # Wait for deployment
    kubectl rollout status deployment/aggregation-server -n "$namespace" --timeout=600s
    
    log_success "Kubernetes deployment completed"
}

# Deploy with Helm
deploy_helm() {
    log_info "Deploying with Helm..."
    
    local namespace="${NAMESPACE:-federated-pipeline-$ENVIRONMENT}"
    local release_name="federated-pipeline-$ENVIRONMENT"
    local values_file="helm/federated-pipeline/values-$ENVIRONMENT.yaml"
    
    if [[ ! -f "$values_file" ]]; then
        values_file="helm/federated-pipeline/values.yaml"
    fi
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would deploy Helm release: $release_name"
        helm template "$release_name" helm/federated-pipeline \
            --namespace "$namespace" \
            --values "$values_file" \
            ${VERSION:+--set image.tag="$VERSION"}
        return
    fi
    
    # Create namespace
    kubectl create namespace "$namespace" --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy with Helm
    helm upgrade --install "$release_name" helm/federated-pipeline \
        --namespace "$namespace" \
        --values "$values_file" \
        ${VERSION:+--set image.tag="$VERSION"} \
        --wait --timeout=600s
    
    log_success "Helm deployment completed"
}

# Update deployment
update_deployment() {
    log_info "Updating deployment..."
    
    case "$TARGET" in
        docker-compose)
            deploy_docker_compose
            ;;
        kubernetes)
            deploy_kubernetes
            ;;
        helm)
            deploy_helm
            ;;
    esac
}

# Rollback deployment
rollback_deployment() {
    log_info "Rolling back deployment..."
    
    case "$TARGET" in
        docker-compose)
            log_warning "Docker Compose rollback not implemented"
            ;;
        kubernetes)
            local namespace="${NAMESPACE:-federated-pipeline-$ENVIRONMENT}"
            kubectl rollout undo deployment/aggregation-server -n "$namespace"
            ;;
        helm)
            local release_name="federated-pipeline-$ENVIRONMENT"
            local namespace="${NAMESPACE:-federated-pipeline-$ENVIRONMENT}"
            helm rollback "$release_name" -n "$namespace"
            ;;
    esac
}

# Check deployment status
check_status() {
    log_info "Checking deployment status..."
    
    case "$TARGET" in
        docker-compose)
            docker-compose ps
            ;;
        kubernetes)
            local namespace="${NAMESPACE:-federated-pipeline-$ENVIRONMENT}"
            kubectl get pods -n "$namespace"
            kubectl get services -n "$namespace"
            ;;
        helm)
            local release_name="federated-pipeline-$ENVIRONMENT"
            local namespace="${NAMESPACE:-federated-pipeline-$ENVIRONMENT}"
            helm status "$release_name" -n "$namespace"
            ;;
    esac
}

# Show logs
show_logs() {
    log_info "Showing application logs..."
    
    local follow_flag=""
    if [[ "$FOLLOW_LOGS" == "true" ]]; then
        follow_flag="-f"
    fi
    
    case "$TARGET" in
        docker-compose)
            docker-compose logs $follow_flag
            ;;
        kubernetes)
            local namespace="${NAMESPACE:-federated-pipeline-$ENVIRONMENT}"
            kubectl logs -l app.kubernetes.io/name=aggregation-server -n "$namespace" $follow_flag
            ;;
        helm)
            local namespace="${NAMESPACE:-federated-pipeline-$ENVIRONMENT}"
            kubectl logs -l app.kubernetes.io/name=aggregation-server -n "$namespace" $follow_flag
            ;;
    esac
}

# Cleanup resources
cleanup_resources() {
    log_info "Cleaning up resources..."
    
    if [[ "$DRY_RUN" == "true" ]]; then
        log_info "[DRY RUN] Would clean up resources"
        return
    fi
    
    case "$TARGET" in
        docker-compose)
            docker-compose down -v
            docker system prune -f
            ;;
        kubernetes)
            local namespace="${NAMESPACE:-federated-pipeline-$ENVIRONMENT}"
            kubectl delete namespace "$namespace" --ignore-not-found=true
            ;;
        helm)
            local release_name="federated-pipeline-$ENVIRONMENT"
            local namespace="${NAMESPACE:-federated-pipeline-$ENVIRONMENT}"
            helm uninstall "$release_name" -n "$namespace" || true
            kubectl delete namespace "$namespace" --ignore-not-found=true
            ;;
    esac
}

# Run deployment tests
run_tests() {
    log_info "Running deployment tests..."
    
    # Health check tests
    local base_url="http://localhost:8000"
    case "$TARGET" in
        kubernetes|helm)
            local namespace="${NAMESPACE:-federated-pipeline-$ENVIRONMENT}"
            base_url="http://$(kubectl get service aggregation-server-service -n "$namespace" -o jsonpath='{.status.loadBalancer.ingress[0].ip}'):8000"
            ;;
    esac
    
    # Test health endpoint
    if curl -f "$base_url/health" &> /dev/null; then
        log_success "Health check passed"
    else
        log_error "Health check failed"
        return 1
    fi
    
    # Test metrics endpoint
    if curl -f "$base_url:9090/metrics" &> /dev/null; then
        log_success "Metrics endpoint accessible"
    else
        log_warning "Metrics endpoint not accessible"
    fi
    
    log_success "All tests passed"
}

# Main execution
main() {
    parse_args "$@"
    validate_environment
    check_prerequisites
    
    cd "$PROJECT_ROOT"
    
    case "$COMMAND" in
        deploy)
            build_images
            case "$TARGET" in
                docker-compose)
                    deploy_docker_compose
                    ;;
                kubernetes)
                    deploy_kubernetes
                    ;;
                helm)
                    deploy_helm
                    ;;
            esac
            ;;
        update)
            update_deployment
            ;;
        rollback)
            rollback_deployment
            ;;
        status)
            check_status
            ;;
        logs)
            show_logs
            ;;
        cleanup)
            cleanup_resources
            ;;
        test)
            run_tests
            ;;
    esac
    
    log_success "Command '$COMMAND' completed successfully"
}

# Run main function
main "$@"