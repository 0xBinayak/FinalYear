"""
Edge Coordinator Module

Provides edge coordination functionality for federated learning including:
- Local client management and registration
- Hierarchical aggregation for bandwidth optimization
- Network partition detection and handling
- Offline operation with eventual consistency
"""

from .coordinator import EdgeCoordinator, CoordinatorState, ClientStatus, LocalClient
from .network_partition import (
    NetworkPartitionDetector, PartitionType, RecoveryStrategy, 
    PartitionEvent, ConnectivityTest
)
from .offline_operation import (
    OfflineOperationManager, OfflineMode, SyncStrategy, 
    OfflineUpdate, ModelVersion
)
from .resource_manager import (
    ResourceManager, ResourceProfile, TrainingTask, 
    SchedulingStrategy, ResourceType
)
from .data_quality import (
    DataQualityValidator, DataQualityReport, DataQualityIssue,
    ValidationSeverity
)
from .service import EdgeCoordinatorService, create_edge_coordinator_service

__all__ = [
    # Main coordinator
    'EdgeCoordinator',
    'CoordinatorState', 
    'ClientStatus',
    'LocalClient',
    
    # Network partition detection
    'NetworkPartitionDetector',
    'PartitionType',
    'RecoveryStrategy',
    'PartitionEvent',
    'ConnectivityTest',
    
    # Offline operation
    'OfflineOperationManager',
    'OfflineMode',
    'SyncStrategy',
    'OfflineUpdate',
    'ModelVersion',
    
    # Resource management
    'ResourceManager',
    'ResourceProfile',
    'TrainingTask',
    'SchedulingStrategy',
    'ResourceType',
    
    # Data quality
    'DataQualityValidator',
    'DataQualityReport',
    'DataQualityIssue',
    'ValidationSeverity',
    
    # Service
    'EdgeCoordinatorService',
    'create_edge_coordinator_service'
]