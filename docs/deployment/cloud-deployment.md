# Cloud Deployment Guide

This guide covers deploying the Advanced Federated Pipeline system to major cloud providers.

## AWS Deployment

### Prerequisites
- AWS CLI configured
- kubectl installed
- Helm 3.x installed
- EKS cluster running

### Step 1: Create EKS Cluster

```bash
# Create EKS cluster
eksctl create cluster --name federated-pipeline --region us-west-2 --nodes 3

# Configure kubectl
aws eks update-kubeconfig --region us-west-2 --name federated-pipeline
```

### Step 2: Deploy with Helm

```bash
# Add Helm repository (if using external charts)
helm repo add federated-pipeline ./helm/federated-pipeline

# Install the system
helm install federated-pipeline ./helm/federated-pipeline \
  --set aggregationServer.replicas=2 \
  --set edgeCoordinator.enabled=true \
  --set monitoring.enabled=true \
  --set storage.type=aws-ebs
```

### Step 3: Configure Load Balancer

```yaml
# aws-load-balancer.yaml
apiVersion: v1
kind: Service
metadata:
  name: aggregation-server-lb
  annotations:
    service.beta.kubernetes.io/aws-load-balancer-type: nlb
spec:
  type: LoadBalancer
  ports:
  - port: 80
    targetPort: 8000
  selector:
    app: aggregation-server
```

```bash
kubectl apply -f aws-load-balancer.yaml
```

### Step 4: Configure Monitoring

```bash
# Install Prometheus and Grafana
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install monitoring prometheus-community/kube-prometheus-stack

# Import custom dashboards
kubectl apply -f monitoring/grafana-dashboards.yaml
```

## Azure Deployment

### Prerequisites
- Azure CLI installed
- AKS cluster running

### Step 1: Create AKS Cluster

```bash
# Create resource group
az group create --name federated-pipeline-rg --location eastus

# Create AKS cluster
az aks create \
  --resource-group federated-pipeline-rg \
  --name federated-pipeline-aks \
  --node-count 3 \
  --enable-addons monitoring \
  --generate-ssh-keys

# Get credentials
az aks get-credentials --resource-group federated-pipeline-rg --name federated-pipeline-aks
```

### Step 2: Deploy Application

```bash
# Deploy using Helm
helm install federated-pipeline ./helm/federated-pipeline \
  --set aggregationServer.replicas=2 \
  --set storage.type=azure-disk \
  --set monitoring.enabled=true
```

## Google Cloud Platform (GCP) Deployment

### Prerequisites
- gcloud CLI configured
- GKE cluster running

### Step 1: Create GKE Cluster

```bash
# Create GKE cluster
gcloud container clusters create federated-pipeline \
  --zone us-central1-a \
  --num-nodes 3 \
  --enable-monitoring \
  --enable-logging

# Get credentials
gcloud container clusters get-credentials federated-pipeline --zone us-central1-a
```

### Step 2: Deploy Application

```bash
# Deploy using Helm
helm install federated-pipeline ./helm/federated-pipeline \
  --set aggregationServer.replicas=2 \
  --set storage.type=gce-pd \
  --set monitoring.enabled=true
```

## Configuration

### Environment Variables

```yaml
# values-production.yaml
aggregationServer:
  replicas: 3
  resources:
    requests:
      memory: "1Gi"
      cpu: "500m"
    limits:
      memory: "2Gi"
      cpu: "1000m"
  env:
    - name: AGGREGATION_STRATEGY
      value: "byzantine_robust"
    - name: MAX_CLIENTS
      value: "1000"
    - name: LOG_LEVEL
      value: "INFO"

edgeCoordinator:
  enabled: true
  replicas: 2
  env:
    - name: LOCAL_CLIENTS_MAX
      value: "50"

monitoring:
  enabled: true
  prometheus:
    retention: "30d"
  grafana:
    adminPassword: "secure-password"

storage:
  type: "aws-ebs"  # or "azure-disk", "gce-pd"
  size: "100Gi"
  storageClass: "gp2"
```

### Secrets Management

```bash
# Create secrets
kubectl create secret generic federated-pipeline-secrets \
  --from-literal=database-password=your-secure-password \
  --from-literal=jwt-secret=your-jwt-secret \
  --from-literal=encryption-key=your-encryption-key
```

## Scaling

### Horizontal Pod Autoscaling

```yaml
# hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: aggregation-server-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: aggregation-server
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

### Cluster Autoscaling

```bash
# Enable cluster autoscaling (AWS)
eksctl create nodegroup --cluster=federated-pipeline \
  --name=autoscaling-nodes \
  --nodes-min=1 \
  --nodes-max=10 \
  --node-type=m5.large \
  --enable-autoscaling
```

## Monitoring and Logging

### CloudWatch (AWS)

```yaml
# cloudwatch-config.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: cloudwatch-config
data:
  cwagentconfig.json: |
    {
      "logs": {
        "metrics_collected": {
          "kubernetes": {
            "cluster_name": "federated-pipeline",
            "metrics_collection_interval": 60
          }
        }
      }
    }
```

### Application Insights (Azure)

```yaml
# Add to deployment
env:
- name: APPINSIGHTS_INSTRUMENTATIONKEY
  valueFrom:
    secretKeyRef:
      name: app-insights
      key: instrumentation-key
```

## Security

### Network Policies

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: federated-pipeline-network-policy
spec:
  podSelector:
    matchLabels:
      app: federated-pipeline
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: federated-pipeline
    ports:
    - protocol: TCP
      port: 8000
```

### Pod Security Policies

```yaml
# pod-security-policy.yaml
apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: federated-pipeline-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
```

## Backup and Disaster Recovery

### Database Backup

```bash
# Automated backup script
#!/bin/bash
BACKUP_NAME="federated-pipeline-backup-$(date +%Y%m%d-%H%M%S)"

# Create backup
kubectl exec -it postgres-0 -- pg_dump -U postgres federated_pipeline > $BACKUP_NAME.sql

# Upload to cloud storage
aws s3 cp $BACKUP_NAME.sql s3://your-backup-bucket/database-backups/
```

### Model Storage Backup

```bash
# Backup model storage
kubectl exec -it aggregation-server-0 -- tar -czf /tmp/models-backup.tar.gz /app/models
kubectl cp aggregation-server-0:/tmp/models-backup.tar.gz ./models-backup-$(date +%Y%m%d).tar.gz
```

## Troubleshooting

### Common Issues

1. **Pod Startup Issues**
   ```bash
   kubectl describe pod <pod-name>
   kubectl logs <pod-name>
   ```

2. **Resource Constraints**
   ```bash
   kubectl top nodes
   kubectl top pods
   ```

3. **Network Connectivity**
   ```bash
   kubectl exec -it <pod-name> -- nslookup aggregation-server
   ```

### Health Checks

```bash
# Check system health
curl http://<load-balancer-ip>/health

# Check individual components
kubectl get pods -l app=federated-pipeline
kubectl get services
```

## Cost Optimization

### Resource Right-Sizing

```bash
# Monitor resource usage
kubectl top pods --sort-by=cpu
kubectl top pods --sort-by=memory

# Adjust resource requests/limits based on usage
```

### Spot Instances (AWS)

```yaml
# Use spot instances for non-critical workloads
nodeGroups:
  - name: spot-nodes
    instanceTypes: ["m5.large", "m5.xlarge"]
    spot: true
    minSize: 0
    maxSize: 10
```

This completes the cloud deployment guide covering AWS, Azure, and GCP with comprehensive configuration, scaling, monitoring, and troubleshooting information.