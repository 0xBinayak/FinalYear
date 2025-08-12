"""
Real-time performance monitoring dashboards for federated learning system.
Provides web-based dashboards using Grafana-compatible JSON configurations.
"""

import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta


@dataclass
class DashboardPanel:
    """Configuration for a dashboard panel."""
    id: int
    title: str
    type: str
    targets: List[Dict[str, Any]]
    gridPos: Dict[str, int]
    options: Optional[Dict[str, Any]] = None
    fieldConfig: Optional[Dict[str, Any]] = None


@dataclass
class Dashboard:
    """Complete dashboard configuration."""
    id: Optional[int]
    title: str
    tags: List[str]
    timezone: str
    panels: List[DashboardPanel]
    time: Dict[str, str]
    refresh: str


class FederatedLearningDashboards:
    """Generator for federated learning monitoring dashboards."""
    
    def __init__(self):
        self.base_time_range = {
            "from": "now-1h",
            "to": "now"
        }
    
    def create_system_overview_dashboard(self) -> Dict[str, Any]:
        """Create system overview dashboard."""
        panels = [
            DashboardPanel(
                id=1,
                title="Active Clients",
                type="stat",
                targets=[{
                    "expr": "sum(fl_active_clients)",
                    "legendFormat": "Total Active Clients"
                }],
                gridPos={"h": 8, "w": 6, "x": 0, "y": 0},
                options={
                    "colorMode": "value",
                    "graphMode": "area",
                    "justifyMode": "center"
                }
            ),
            DashboardPanel(
                id=2,
                title="Training Rounds Completed",
                type="stat",
                targets=[{
                    "expr": "sum(rate(fl_training_rounds_total[5m]))",
                    "legendFormat": "Rounds/min"
                }],
                gridPos={"h": 8, "w": 6, "x": 6, "y": 0}
            ),
            DashboardPanel(
                id=3,
                title="Model Accuracy",
                type="timeseries",
                targets=[{
                    "expr": "fl_model_accuracy",
                    "legendFormat": "{{model_version}} - {{dataset}}"
                }],
                gridPos={"h": 8, "w": 12, "x": 12, "y": 0}
            ),
            DashboardPanel(
                id=4,
                title="Request Rate by Component",
                type="timeseries",
                targets=[{
                    "expr": "sum(rate(fl_requests_total[5m])) by (component)",
                    "legendFormat": "{{component}}"
                }],
                gridPos={"h": 8, "w": 12, "x": 0, "y": 8}
            ),
            DashboardPanel(
                id=5,
                title="Memory Usage",
                type="timeseries",
                targets=[{
                    "expr": "fl_memory_usage_bytes / 1024 / 1024",
                    "legendFormat": "{{component}} - {{instance_id}} (MB)"
                }],
                gridPos={"h": 8, "w": 12, "x": 12, "y": 8}
            ),
            DashboardPanel(
                id=6,
                title="Network Latency",
                type="heatmap",
                targets=[{
                    "expr": "fl_network_latency_seconds",
                    "legendFormat": "{{source}} -> {{destination}}"
                }],
                gridPos={"h": 8, "w": 24, "x": 0, "y": 16}
            )
        ]
        
        dashboard = Dashboard(
            id=None,
            title="Federated Learning - System Overview",
            tags=["federated-learning", "overview"],
            timezone="browser",
            panels=panels,
            time=self.base_time_range,
            refresh="5s"
        )
        
        return asdict(dashboard)
    
    def create_privacy_dashboard(self) -> Dict[str, Any]:
        """Create privacy monitoring dashboard."""
        panels = [
            DashboardPanel(
                id=1,
                title="Privacy Budget Usage",
                type="timeseries",
                targets=[{
                    "expr": "fl_privacy_budget_used",
                    "legendFormat": "{{client_id}}"
                }],
                gridPos={"h": 8, "w": 12, "x": 0, "y": 0}
            ),
            DashboardPanel(
                id=2,
                title="Privacy Epsilon Values",
                type="stat",
                targets=[{
                    "expr": "fl_privacy_epsilon",
                    "legendFormat": "{{component}}"
                }],
                gridPos={"h": 8, "w": 12, "x": 12, "y": 0}
            ),
            DashboardPanel(
                id=3,
                title="Privacy Violations",
                type="timeseries",
                targets=[{
                    "expr": "sum(rate(fl_privacy_violations_total[5m])) by (violation_type)",
                    "legendFormat": "{{violation_type}}"
                }],
                gridPos={"h": 8, "w": 24, "x": 0, "y": 8}
            )
        ]
        
        dashboard = Dashboard(
            id=None,
            title="Federated Learning - Privacy Monitoring",
            tags=["federated-learning", "privacy"],
            timezone="browser",
            panels=panels,
            time=self.base_time_range,
            refresh="10s"
        )
        
        return asdict(dashboard)
    
    def create_fairness_dashboard(self) -> Dict[str, Any]:
        """Create fairness monitoring dashboard."""
        panels = [
            DashboardPanel(
                id=1,
                title="Fairness Scores by Group",
                type="timeseries",
                targets=[{
                    "expr": "fl_fairness_score",
                    "legendFormat": "{{group_type}} - {{metric_type}}"
                }],
                gridPos={"h": 8, "w": 12, "x": 0, "y": 0}
            ),
            DashboardPanel(
                id=2,
                title="Client Contribution Variance",
                type="stat",
                targets=[{
                    "expr": "fl_client_contribution_variance",
                    "legendFormat": "Variance"
                }],
                gridPos={"h": 8, "w": 12, "x": 12, "y": 0}
            ),
            DashboardPanel(
                id=3,
                title="Client Participation Rate",
                type="heatmap",
                targets=[{
                    "expr": "fl_client_participation_rate",
                    "legendFormat": "{{region}}"
                }],
                gridPos={"h": 8, "w": 24, "x": 0, "y": 8}
            )
        ]
        
        dashboard = Dashboard(
            id=None,
            title="Federated Learning - Fairness Analysis",
            tags=["federated-learning", "fairness"],
            timezone="browser",
            panels=panels,
            time=self.base_time_range,
            refresh="30s"
        )
        
        return asdict(dashboard)
    
    def create_signal_processing_dashboard(self) -> Dict[str, Any]:
        """Create signal processing monitoring dashboard."""
        panels = [
            DashboardPanel(
                id=1,
                title="Signal Quality (SNR)",
                type="timeseries",
                targets=[{
                    "expr": "fl_signal_quality_snr_db",
                    "legendFormat": "{{client_id}} - {{frequency_band}}"
                }],
                gridPos={"h": 8, "w": 12, "x": 0, "y": 0}
            ),
            DashboardPanel(
                id=2,
                title="Classification Confidence",
                type="histogram",
                targets=[{
                    "expr": "fl_classification_confidence",
                    "legendFormat": "{{signal_type}}"
                }],
                gridPos={"h": 8, "w": 12, "x": 12, "y": 0}
            ),
            DashboardPanel(
                id=3,
                title="Model Size Distribution",
                type="histogram",
                targets=[{
                    "expr": "fl_model_size_bytes / 1024",
                    "legendFormat": "{{compression_type}} (KB)"
                }],
                gridPos={"h": 8, "w": 24, "x": 0, "y": 8}
            )
        ]
        
        dashboard = Dashboard(
            id=None,
            title="Federated Learning - Signal Processing",
            tags=["federated-learning", "signal-processing"],
            timezone="browser",
            panels=panels,
            time=self.base_time_range,
            refresh="5s"
        )
        
        return asdict(dashboard)
    
    def create_anomaly_detection_dashboard(self) -> Dict[str, Any]:
        """Create anomaly detection dashboard."""
        panels = [
            DashboardPanel(
                id=1,
                title="Anomaly Scores",
                type="timeseries",
                targets=[{
                    "expr": "fl_anomaly_score",
                    "legendFormat": "{{component}} - {{anomaly_type}}"
                }],
                gridPos={"h": 8, "w": 12, "x": 0, "y": 0}
            ),
            DashboardPanel(
                id=2,
                title="Anomalies Detected",
                type="stat",
                targets=[{
                    "expr": "sum(fl_anomalies_detected_total)",
                    "legendFormat": "Total Anomalies"
                }],
                gridPos={"h": 8, "w": 6, "x": 12, "y": 0}
            ),
            DashboardPanel(
                id=3,
                title="Anomaly Severity Distribution",
                type="piechart",
                targets=[{
                    "expr": "sum(fl_anomalies_detected_total) by (severity)",
                    "legendFormat": "{{severity}}"
                }],
                gridPos={"h": 8, "w": 6, "x": 18, "y": 0}
            ),
            DashboardPanel(
                id=4,
                title="Anomaly Timeline",
                type="timeseries",
                targets=[{
                    "expr": "sum(rate(fl_anomalies_detected_total[1m])) by (anomaly_type)",
                    "legendFormat": "{{anomaly_type}}"
                }],
                gridPos={"h": 8, "w": 24, "x": 0, "y": 8}
            )
        ]
        
        dashboard = Dashboard(
            id=None,
            title="Federated Learning - Anomaly Detection",
            tags=["federated-learning", "anomaly-detection"],
            timezone="browser",
            panels=panels,
            time=self.base_time_range,
            refresh="10s"
        )
        
        return asdict(dashboard)
    
    def export_all_dashboards(self, output_dir: str = "dashboards"):
        """Export all dashboards to JSON files."""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        dashboards = {
            "system_overview": self.create_system_overview_dashboard(),
            "privacy_monitoring": self.create_privacy_dashboard(),
            "fairness_analysis": self.create_fairness_dashboard(),
            "signal_processing": self.create_signal_processing_dashboard(),
            "anomaly_detection": self.create_anomaly_detection_dashboard()
        }
        
        for name, dashboard in dashboards.items():
            filepath = os.path.join(output_dir, f"{name}.json")
            with open(filepath, 'w') as f:
                json.dump(dashboard, f, indent=2)
        
        return list(dashboards.keys())


def create_grafana_datasource_config() -> Dict[str, Any]:
    """Create Grafana datasource configuration for Prometheus."""
    return {
        "name": "Federated Learning Prometheus",
        "type": "prometheus",
        "url": "http://prometheus:9090",
        "access": "proxy",
        "isDefault": True,
        "jsonData": {
            "timeInterval": "5s",
            "queryTimeout": "60s"
        }
    }