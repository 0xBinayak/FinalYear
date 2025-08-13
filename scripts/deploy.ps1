# Federated Pipeline Deployment Script (PowerShell)
# Supports multiple environments and deployment targets

param(
    [Parameter(Mandatory=$true)]
    [ValidateSet("deploy", "update", "rollback", "status", "logs", "cleanup", "test")]
    [string]$Command,
    
    [Parameter()]
    [ValidateSet("development", "staging", "production")]
    [string]$Environment = "development",
    
    [Parameter()]
    [ValidateSet("docker-compose", "kubernetes", "helm")]
    [string]$Target = "docker-compose",
    
    [Parameter()]
    [string]$Version = "",
    
    [Parameter()]
    [string]$ConfigFile = "",
    
    [Parameter()]
    [string]$Namespace = "",
    
    [Parameter()]
    [switch]$DryRun,
    
    [Parameter()]
    [switch]$Follow,
    
    [Parameter()]
    [switch]$Help
)

# Configuration
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Split-Path -Parent $ScriptDir

# Colors for output
$Colors = @{
    Red = "Red"
    Green = "Green"
    Yellow = "Yellow"
    Blue = "Blue"
    White = "White"
}

# Logging functions
function Write-Info {
    param([string]$Message)
    Write-Host "[INFO] $Message" -ForegroundColor $Colors.Blue
}

function Write-Success {
    param([string]$Message)
    Write-Host "[SUCCESS] $Message" -ForegroundColor $Colors.Green
}

function Write-Warning {
    param([string]$Message)
    Write-Host "[WARNING] $Message" -ForegroundColor $Colors.Yellow
}

function Write-Error {
    param([string]$Message)
    Write-Host "[ERROR] $Message" -ForegroundColor $Colors.Red
}

# Help function
function Show-Help {
    @"
Federated Pipeline Deployment Script (PowerShell)

Usage: .\deploy.ps1 -Command <COMMAND> [OPTIONS]

Commands:
    deploy          Deploy the application
    update          Update existing deployment
    rollback        Rollback to previous version
    status          Check deployment status
    logs            Show application logs
    cleanup         Clean up resources
    test            Run deployment tests

Options:
    -Environment ENV        Target environment (development|staging|production)
    -Target TARGET          Deployment target (docker-compose|kubernetes|helm)
    -Version VERSION        Application version to deploy
    -ConfigFile FILE        Custom configuration file
    -Namespace NS           Kubernetes namespace
    -DryRun                 Show what would be done without executing
    -Follow                 Follow logs (for logs command)
    -Help                   Show this help message

Examples:
    .\deploy.ps1 -Command deploy -Environment production -Target kubernetes
    .\deploy.ps1 -Command update -Environment staging -Version v1.2.3
    .\deploy.ps1 -Command rollback -Environment production -Target helm
    .\deploy.ps1 -Command status -Environment development
    .\deploy.ps1 -Command logs -Environment staging -Follow

Environment Variables:
    REGISTRY_URL            Container registry URL
    REGISTRY_USERNAME       Registry username
    REGISTRY_PASSWORD       Registry password
    KUBECONFIG             Kubernetes config file path
    HELM_CHART_VERSION     Helm chart version to use

"@
}

# Check prerequisites
function Test-Prerequisites {
    $missingTools = @()
    
    switch ($Target) {
        "docker-compose" {
            if (-not (Get-Command docker-compose -ErrorAction SilentlyContinue)) {
                $missingTools += "docker-compose"
            }
        }
        "kubernetes" {
            if (-not (Get-Command kubectl -ErrorAction SilentlyContinue)) {
                $missingTools += "kubectl"
            }
        }
        "helm" {
            if (-not (Get-Command helm -ErrorAction SilentlyContinue)) {
                $missingTools += "helm"
            }
            if (-not (Get-Command kubectl -ErrorAction SilentlyContinue)) {
                $missingTools += "kubectl"
            }
        }
    }
    
    if ($missingTools.Count -gt 0) {
        Write-Error "Missing required tools: $($missingTools -join ', ')"
        exit 1
    }
}

# Build images
function Build-Images {
    Write-Info "Building Docker images..."
    
    $components = @("aggregation-server", "edge-coordinator", "sdr-client", "mobile-client", "metrics-collector")
    $registryUrl = $env:REGISTRY_URL ?? "registry.company.com"
    $imageTag = $Version ?? "latest"
    
    foreach ($component in $components) {
        Write-Info "Building $component..."
        
        if ($DryRun) {
            Write-Info "[DRY RUN] Would build: $registryUrl/federated-pipeline/$component`:$imageTag"
        } else {
            $dockerArgs = @(
                "build",
                "-f", "docker/Dockerfile.$component",
                "-t", "$registryUrl/federated-pipeline/$component`:$imageTag",
                "--target", "production",
                "."
            )
            
            & docker @dockerArgs
            
            if ($LASTEXITCODE -ne 0) {
                Write-Error "Failed to build $component"
                exit 1
            }
            
            if ($Environment -ne "development") {
                & docker push "$registryUrl/federated-pipeline/$component`:$imageTag"
                
                if ($LASTEXITCODE -ne 0) {
                    Write-Error "Failed to push $component"
                    exit 1
                }
            }
        }
    }
}

# Deploy with Docker Compose
function Deploy-DockerCompose {
    Write-Info "Deploying with Docker Compose..."
    
    $composeFile = "docker-compose.yml"
    switch ($Environment) {
        "production" { $composeFile = "docker-compose.prod.yml" }
        "staging" { $composeFile = "docker-compose.staging.yml" }
    }
    
    if ($DryRun) {
        Write-Info "[DRY RUN] Would run: docker-compose -f $composeFile up -d"
        return
    }
    
    # Set environment variables
    $env:ENVIRONMENT = $Environment
    $env:VERSION = $Version ?? "latest"
    
    # Deploy
    & docker-compose -f $composeFile up -d
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Docker Compose deployment failed"
        exit 1
    }
    
    # Wait for services to be healthy
    Write-Info "Waiting for services to be healthy..."
    & docker-compose -f $composeFile ps
    
    # Run health checks
    Start-Sleep -Seconds 30
    try {
        $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 10
        if ($response.StatusCode -eq 200) {
            Write-Success "Aggregation server is healthy"
        } else {
            Write-Error "Aggregation server health check failed"
            return 1
        }
    } catch {
        Write-Error "Aggregation server health check failed: $($_.Exception.Message)"
        return 1
    }
}

# Deploy with Kubernetes
function Deploy-Kubernetes {
    Write-Info "Deploying with Kubernetes..."
    
    $namespace = $Namespace ?? "federated-pipeline-$Environment"
    
    if ($DryRun) {
        Write-Info "[DRY RUN] Would deploy to namespace: $namespace"
        return
    }
    
    # Create namespace if it doesn't exist
    & kubectl create namespace $namespace --dry-run=client -o yaml | kubectl apply -f -
    
    # Apply configurations
    & kubectl apply -f k8s/namespace.yaml -n $namespace
    & kubectl apply -f k8s/configmap.yaml -n $namespace
    & kubectl apply -f k8s/secrets.yaml -n $namespace
    & kubectl apply -f k8s/postgres.yaml -n $namespace
    & kubectl apply -f k8s/aggregation-server.yaml -n $namespace
    & kubectl apply -f k8s/edge-coordinator.yaml -n $namespace
    
    # Wait for deployment
    & kubectl rollout status deployment/aggregation-server -n $namespace --timeout=600s
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Kubernetes deployment completed"
    } else {
        Write-Error "Kubernetes deployment failed"
        exit 1
    }
}

# Deploy with Helm
function Deploy-Helm {
    Write-Info "Deploying with Helm..."
    
    $namespace = $Namespace ?? "federated-pipeline-$Environment"
    $releaseName = "federated-pipeline-$Environment"
    $valuesFile = "helm/federated-pipeline/values-$Environment.yaml"
    
    if (-not (Test-Path $valuesFile)) {
        $valuesFile = "helm/federated-pipeline/values.yaml"
    }
    
    if ($DryRun) {
        Write-Info "[DRY RUN] Would deploy Helm release: $releaseName"
        $helmArgs = @(
            "template", $releaseName, "helm/federated-pipeline",
            "--namespace", $namespace,
            "--values", $valuesFile
        )
        if ($Version) {
            $helmArgs += @("--set", "image.tag=$Version")
        }
        & helm @helmArgs
        return
    }
    
    # Create namespace
    & kubectl create namespace $namespace --dry-run=client -o yaml | kubectl apply -f -
    
    # Deploy with Helm
    $helmArgs = @(
        "upgrade", "--install", $releaseName, "helm/federated-pipeline",
        "--namespace", $namespace,
        "--values", $valuesFile,
        "--wait", "--timeout=600s"
    )
    if ($Version) {
        $helmArgs += @("--set", "image.tag=$Version")
    }
    
    & helm @helmArgs
    
    if ($LASTEXITCODE -eq 0) {
        Write-Success "Helm deployment completed"
    } else {
        Write-Error "Helm deployment failed"
        exit 1
    }
}

# Update deployment
function Update-Deployment {
    Write-Info "Updating deployment..."
    
    switch ($Target) {
        "docker-compose" { Deploy-DockerCompose }
        "kubernetes" { Deploy-Kubernetes }
        "helm" { Deploy-Helm }
    }
}

# Rollback deployment
function Rollback-Deployment {
    Write-Info "Rolling back deployment..."
    
    switch ($Target) {
        "docker-compose" {
            Write-Warning "Docker Compose rollback not implemented"
        }
        "kubernetes" {
            $namespace = $Namespace ?? "federated-pipeline-$Environment"
            & kubectl rollout undo deployment/aggregation-server -n $namespace
        }
        "helm" {
            $releaseName = "federated-pipeline-$Environment"
            $namespace = $Namespace ?? "federated-pipeline-$Environment"
            & helm rollback $releaseName -n $namespace
        }
    }
}

# Check deployment status
function Get-Status {
    Write-Info "Checking deployment status..."
    
    switch ($Target) {
        "docker-compose" {
            & docker-compose ps
        }
        "kubernetes" {
            $namespace = $Namespace ?? "federated-pipeline-$Environment"
            & kubectl get pods -n $namespace
            & kubectl get services -n $namespace
        }
        "helm" {
            $releaseName = "federated-pipeline-$Environment"
            $namespace = $Namespace ?? "federated-pipeline-$Environment"
            & helm status $releaseName -n $namespace
        }
    }
}

# Show logs
function Show-Logs {
    Write-Info "Showing application logs..."
    
    $followFlag = if ($Follow) { "-f" } else { "" }
    
    switch ($Target) {
        "docker-compose" {
            if ($Follow) {
                & docker-compose logs -f
            } else {
                & docker-compose logs
            }
        }
        "kubernetes" {
            $namespace = $Namespace ?? "federated-pipeline-$Environment"
            $kubectlArgs = @("logs", "-l", "app.kubernetes.io/name=aggregation-server", "-n", $namespace)
            if ($Follow) {
                $kubectlArgs += "-f"
            }
            & kubectl @kubectlArgs
        }
        "helm" {
            $namespace = $Namespace ?? "federated-pipeline-$Environment"
            $kubectlArgs = @("logs", "-l", "app.kubernetes.io/name=aggregation-server", "-n", $namespace)
            if ($Follow) {
                $kubectlArgs += "-f"
            }
            & kubectl @kubectlArgs
        }
    }
}

# Cleanup resources
function Remove-Resources {
    Write-Info "Cleaning up resources..."
    
    if ($DryRun) {
        Write-Info "[DRY RUN] Would clean up resources"
        return
    }
    
    switch ($Target) {
        "docker-compose" {
            & docker-compose down -v
            & docker system prune -f
        }
        "kubernetes" {
            $namespace = $Namespace ?? "federated-pipeline-$Environment"
            & kubectl delete namespace $namespace --ignore-not-found=true
        }
        "helm" {
            $releaseName = "federated-pipeline-$Environment"
            $namespace = $Namespace ?? "federated-pipeline-$Environment"
            & helm uninstall $releaseName -n $namespace
            & kubectl delete namespace $namespace --ignore-not-found=true
        }
    }
}

# Run deployment tests
function Invoke-Tests {
    Write-Info "Running deployment tests..."
    
    # Health check tests
    $baseUrl = "http://localhost:8000"
    switch ($Target) {
        { $_ -in @("kubernetes", "helm") } {
            $namespace = $Namespace ?? "federated-pipeline-$Environment"
            try {
                $serviceIp = & kubectl get service aggregation-server-service -n $namespace -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
                if ($serviceIp) {
                    $baseUrl = "http://$serviceIp:8000"
                }
            } catch {
                Write-Warning "Could not get service IP, using localhost"
            }
        }
    }
    
    # Test health endpoint
    try {
        $response = Invoke-WebRequest -Uri "$baseUrl/health" -UseBasicParsing -TimeoutSec 10
        if ($response.StatusCode -eq 200) {
            Write-Success "Health check passed"
        } else {
            Write-Error "Health check failed"
            return 1
        }
    } catch {
        Write-Error "Health check failed: $($_.Exception.Message)"
        return 1
    }
    
    # Test metrics endpoint
    try {
        $response = Invoke-WebRequest -Uri "$baseUrl`:9090/metrics" -UseBasicParsing -TimeoutSec 10
        if ($response.StatusCode -eq 200) {
            Write-Success "Metrics endpoint accessible"
        } else {
            Write-Warning "Metrics endpoint not accessible"
        }
    } catch {
        Write-Warning "Metrics endpoint not accessible: $($_.Exception.Message)"
    }
    
    Write-Success "All tests passed"
}

# Main execution
function Main {
    if ($Help) {
        Show-Help
        exit 0
    }
    
    Write-Info "Deploying to environment: $Environment"
    Test-Prerequisites
    
    Set-Location $ProjectRoot
    
    switch ($Command) {
        "deploy" {
            Build-Images
            switch ($Target) {
                "docker-compose" { Deploy-DockerCompose }
                "kubernetes" { Deploy-Kubernetes }
                "helm" { Deploy-Helm }
            }
        }
        "update" { Update-Deployment }
        "rollback" { Rollback-Deployment }
        "status" { Get-Status }
        "logs" { Show-Logs }
        "cleanup" { Remove-Resources }
        "test" { Invoke-Tests }
    }
    
    Write-Success "Command '$Command' completed successfully"
}

# Run main function
Main