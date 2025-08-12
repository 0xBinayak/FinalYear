"""
Base interfaces and abstract classes for all components
"""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import numpy as np


@dataclass
class ClientInfo:
    """Client information model"""
    client_id: str
    client_type: str  # SDR, Mobile, Simulated
    capabilities: Dict[str, Any]
    location: Optional[Dict[str, float]]  # GPS coordinates
    network_info: Dict[str, Any]
    hardware_specs: Dict[str, Any]
    reputation_score: float = 1.0


@dataclass
class ModelUpdate:
    """Model update from client"""
    client_id: str
    model_weights: bytes
    training_metrics: Dict[str, float]
    data_statistics: Dict[str, Any]
    computation_time: float
    network_conditions: Dict[str, Any]
    privacy_budget_used: float = 0.0


@dataclass
class TrainingConfig:
    """Training configuration"""
    model_architecture: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    privacy_settings: Dict[str, Any]
    resource_constraints: Dict[str, Any]
    aggregation_strategy: str
    client_selection_criteria: Dict[str, Any]


@dataclass
class SignalSample:
    """Signal data model"""
    timestamp: datetime
    frequency: float
    sample_rate: float
    iq_data: np.ndarray
    modulation_type: str
    snr: float
    location: Optional[Dict[str, float]]
    device_id: str
    metadata: Dict[str, Any]


class BaseClient(ABC):
    """Abstract base class for all client types"""
    
    @abstractmethod
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the client with configuration"""
        pass
    
    @abstractmethod
    def train_local_model(self, training_data: Any) -> ModelUpdate:
        """Train local model and return update"""
        pass
    
    @abstractmethod
    def receive_global_model(self, model_weights: bytes) -> bool:
        """Receive and apply global model update"""
        pass
    
    @abstractmethod
    def get_client_info(self) -> ClientInfo:
        """Get client information"""
        pass


class BaseAggregator(ABC):
    """Abstract base class for aggregation services"""
    
    @abstractmethod
    def register_client(self, client_info: ClientInfo) -> str:
        """Register a new client and return token"""
        pass
    
    @abstractmethod
    def receive_model_update(self, client_id: str, model_update: ModelUpdate) -> bool:
        """Receive model update from client"""
        pass
    
    @abstractmethod
    def aggregate_models(self, updates: List[ModelUpdate]) -> bytes:
        """Aggregate model updates"""
        pass
    
    @abstractmethod
    def distribute_global_model(self, client_ids: List[str]) -> bool:
        """Distribute global model to clients"""
        pass


class BaseCoordinator(ABC):
    """Abstract base class for coordination services"""
    
    @abstractmethod
    def manage_local_clients(self, clients: List[BaseClient]) -> bool:
        """Manage local client cluster"""
        pass
    
    @abstractmethod
    def aggregate_local_models(self, updates: List[ModelUpdate]) -> ModelUpdate:
        """Aggregate local model updates"""
        pass
    
    @abstractmethod
    def sync_with_global_server(self, server_endpoint: str) -> bool:
        """Synchronize with global aggregation server"""
        pass


class BaseDataLoader(ABC):
    """Abstract base class for data loading"""
    
    @abstractmethod
    def load_signal_data(self, file_path: str) -> List[SignalSample]:
        """Load signal data from file"""
        pass
    
    @abstractmethod
    def preprocess_data(self, data: List[SignalSample]) -> Any:
        """Preprocess signal data for training"""
        pass
    
    @abstractmethod
    def validate_data_quality(self, data: List[SignalSample]) -> Dict[str, Any]:
        """Validate data quality and return metrics"""
        pass


class BaseMonitor(ABC):
    """Abstract base class for monitoring services"""
    
    @abstractmethod
    def collect_metrics(self, component_id: str, metrics: Dict[str, Any]) -> bool:
        """Collect metrics from component"""
        pass
    
    @abstractmethod
    def detect_anomalies(self, metrics: Dict[str, Any]) -> List[str]:
        """Detect anomalies in metrics"""
        pass
    
    @abstractmethod
    def generate_alerts(self, anomalies: List[str]) -> bool:
        """Generate alerts for detected anomalies"""
        pass