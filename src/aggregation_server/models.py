"""
Pydantic models for API request/response validation
"""
from typing import Dict, List, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ClientRegistrationRequest(BaseModel):
    """Client registration request model"""
    client_id: str = Field(..., description="Unique client identifier")
    client_type: str = Field(..., description="Type of client (SDR, Mobile, Simulated)")
    capabilities: Dict[str, Any] = Field(..., description="Client capabilities")
    location: Optional[Dict[str, float]] = Field(None, description="GPS coordinates")
    network_info: Dict[str, Any] = Field(..., description="Network information")
    hardware_specs: Dict[str, Any] = Field(..., description="Hardware specifications")


class ClientRegistrationResponse(BaseModel):
    """Client registration response model"""
    client_id: str
    token: str
    status: str
    message: Optional[str] = None


class ModelUpdateRequest(BaseModel):
    """Model update request model"""
    model_weights: bytes = Field(..., description="Serialized model weights")
    training_metrics: Dict[str, float] = Field(..., description="Training metrics")
    data_statistics: Dict[str, Any] = Field(..., description="Data statistics")
    computation_time: float = Field(..., description="Training computation time")
    network_conditions: Dict[str, Any] = Field(..., description="Network conditions")
    privacy_budget_used: float = Field(0.0, description="Privacy budget consumed")


class ModelUpdateResponse(BaseModel):
    """Model update response model"""
    client_id: str
    status: str
    round_number: int
    message: Optional[str] = None


class GlobalModelResponse(BaseModel):
    """Global model response model"""
    model_weights: bytes
    version: str
    round_number: int
    training_config: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class TrainingConfigResponse(BaseModel):
    """Training configuration response model"""
    model_architecture: Dict[str, Any]
    hyperparameters: Dict[str, Any]
    privacy_settings: Dict[str, Any]
    resource_constraints: Dict[str, Any]
    aggregation_strategy: str
    client_selection_criteria: Dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: datetime
    version: str
    uptime: float
    active_clients: int
    current_round: int
    last_aggregation: Optional[datetime] = None


class MetricsRequest(BaseModel):
    """Client metrics request model"""
    metrics: Dict[str, Any] = Field(..., description="Client metrics")
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ServerStatusResponse(BaseModel):
    """Server status response model"""
    status: str
    uptime: float
    total_clients: int
    active_clients: int
    current_round: int
    total_rounds: int
    last_aggregation: Optional[datetime]
    aggregation_strategy: str
    model_version: str
    performance_metrics: Dict[str, Any]


class ClientMetrics(BaseModel):
    """Client metrics model"""
    client_id: str
    timestamp: datetime
    training_accuracy: Optional[float] = None
    training_loss: Optional[float] = None
    data_size: Optional[int] = None
    computation_time: Optional[float] = None
    network_latency: Optional[float] = None
    battery_level: Optional[float] = None
    memory_usage: Optional[float] = None
    cpu_usage: Optional[float] = None


class AggregationResult(BaseModel):
    """Aggregation result model"""
    round_number: int
    participating_clients: List[str]
    aggregation_strategy: str
    convergence_metrics: Dict[str, float]
    model_version: str
    timestamp: datetime
    performance_improvement: Optional[float] = None


class ClientSelectionCriteria(BaseModel):
    """Client selection criteria model"""
    min_clients: int = Field(2, description="Minimum number of clients")
    max_clients: int = Field(100, description="Maximum number of clients")
    selection_strategy: str = Field("random", description="Selection strategy")
    quality_threshold: float = Field(0.5, description="Minimum quality threshold")
    reputation_weight: float = Field(0.3, description="Reputation weight in selection")
    resource_weight: float = Field(0.3, description="Resource weight in selection")
    geographic_diversity: bool = Field(False, description="Ensure geographic diversity")


class PrivacyBudget(BaseModel):
    """Privacy budget tracking model"""
    client_id: str
    total_budget: float
    used_budget: float
    remaining_budget: float
    last_update: datetime
    epsilon_per_round: float


class NetworkConditions(BaseModel):
    """Network conditions model"""
    latency: float = Field(..., description="Network latency in ms")
    bandwidth: float = Field(..., description="Available bandwidth in Mbps")
    packet_loss: float = Field(0.0, description="Packet loss percentage")
    jitter: float = Field(0.0, description="Network jitter in ms")
    connection_type: str = Field("unknown", description="Connection type")


class DeviceCapabilities(BaseModel):
    """Device capabilities model"""
    cpu_cores: int
    memory_gb: float
    storage_gb: float
    gpu_available: bool = False
    gpu_memory_gb: Optional[float] = None
    battery_capacity: Optional[float] = None
    network_interfaces: List[str] = []
    sdr_hardware: Optional[Dict[str, Any]] = None


class TrainingProgress(BaseModel):
    """Training progress tracking model"""
    client_id: str
    round_number: int
    epoch: int
    batch: int
    loss: float
    accuracy: Optional[float] = None
    timestamp: datetime
    estimated_completion: Optional[datetime] = None