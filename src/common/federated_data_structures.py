"""
Federated learning data structures with real-world constraints and privacy-preserving features
"""
import numpy as np
import json
import pickle
import gzip
import hashlib
import time
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import struct

from .interfaces import ModelUpdate
from .signal_models import EnhancedSignalSample, SignalQualityMetrics


class CompressionType(Enum):
    """Compression algorithms for model updates"""
    NONE = "none"
    GZIP = "gzip"
    QUANTIZATION = "quantization"
    SPARSIFICATION = "sparsification"
    FEDAVG_COMPRESSION = "fedavg_compression"
    GRADIENT_COMPRESSION = "gradient_compression"


class PrivacyMechanism(Enum):
    """Privacy-preserving mechanisms"""
    NONE = "none"
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    SECURE_AGGREGATION = "secure_aggregation"
    HOMOMORPHIC_ENCRYPTION = "homomorphic_encryption"
    FEDERATED_DROPOUT = "federated_dropout"


class AggregationStrategy(Enum):
    """Model aggregation strategies"""
    FEDAVG = "fedavg"
    FEDPROX = "fedprox"
    SCAFFOLD = "scaffold"
    FEDNOVA = "fednova"
    BYZANTINE_ROBUST = "byzantine_robust"
    KRUM = "krum"
    TRIMMED_MEAN = "trimmed_mean"
    MEDIAN = "median"


@dataclass
class NetworkConditions:
    """Network conditions for realistic federated learning"""
    bandwidth_mbps: float
    latency_ms: float
    packet_loss_rate: float
    jitter_ms: float
    connection_stability: float  # 0.0 to 1.0
    is_mobile: bool = False
    is_metered: bool = False
    signal_strength_dbm: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def get_transmission_time_estimate(self, data_size_bytes: int) -> float:
        """Estimate transmission time in seconds"""
        # Account for overhead and packet loss
        effective_bandwidth = self.bandwidth_mbps * (1 - self.packet_loss_rate) * 0.8  # 80% efficiency
        transmission_time = (data_size_bytes * 8) / (effective_bandwidth * 1e6)  # Convert to seconds
        
        # Add latency and jitter
        total_time = transmission_time + (self.latency_ms + self.jitter_ms) / 1000
        
        # Account for connection stability
        stability_factor = 1.0 / max(0.1, self.connection_stability)
        
        return total_time * stability_factor


@dataclass
class ComputeResources:
    """Compute resource constraints for edge devices"""
    cpu_cores: int
    cpu_frequency_ghz: float
    memory_gb: float
    available_memory_gb: float
    gpu_available: bool = False
    gpu_memory_gb: float = 0.0
    battery_level: Optional[float] = None  # 0.0 to 1.0
    thermal_state: str = "normal"  # normal, warm, hot, critical
    power_source: str = "battery"  # battery, plugged
    
    def can_handle_training(self, model_size_mb: float, batch_size: int) -> bool:
        """Check if device can handle training with given parameters"""
        # Rough memory requirement estimation
        memory_needed = model_size_mb * 4 + batch_size * 0.1  # GB
        
        if memory_needed > self.available_memory_gb:
            return False
        
        # Battery constraint
        if self.battery_level is not None and self.battery_level < 0.2:
            return False
        
        # Thermal constraint
        if self.thermal_state in ["hot", "critical"]:
            return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PrivacyBudget:
    """Differential privacy budget tracking"""
    epsilon: float
    delta: float
    consumed_epsilon: float = 0.0
    consumed_delta: float = 0.0
    mechanism: str = "gaussian"
    sensitivity: float = 1.0
    
    def can_spend(self, epsilon_cost: float, delta_cost: float = 0.0) -> bool:
        """Check if privacy budget allows spending"""
        return (self.consumed_epsilon + epsilon_cost <= self.epsilon and
                self.consumed_delta + delta_cost <= self.delta)
    
    def spend(self, epsilon_cost: float, delta_cost: float = 0.0):
        """Spend privacy budget"""
        if not self.can_spend(epsilon_cost, delta_cost):
            raise ValueError("Insufficient privacy budget")
        
        self.consumed_epsilon += epsilon_cost
        self.consumed_delta += delta_cost
    
    def remaining_budget(self) -> Tuple[float, float]:
        """Get remaining privacy budget"""
        return (self.epsilon - self.consumed_epsilon, 
                self.delta - self.consumed_delta)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EnhancedModelUpdate:
    """Enhanced model update with real-world constraints and privacy features"""
    # Core model data
    client_id: str
    model_weights: bytes
    model_size_bytes: int
    
    # Training metadata
    training_rounds: int
    local_epochs: int
    batch_size: int
    learning_rate: float
    samples_used: int
    training_time_seconds: float
    
    # Quality metrics
    training_loss: float
    validation_loss: Optional[float] = None
    training_accuracy: Optional[float] = None
    validation_accuracy: Optional[float] = None
    convergence_metric: Optional[float] = None
    
    # Data characteristics
    data_distribution: Dict[str, int] = field(default_factory=dict)  # class counts
    signal_quality_stats: Optional[Dict[str, float]] = None
    modulation_diversity: Optional[Dict[str, int]] = None
    snr_distribution: Optional[Dict[str, float]] = None
    
    # System constraints
    network_conditions: NetworkConditions = field(default_factory=lambda: NetworkConditions(10.0, 100.0, 0.01, 10.0, 0.9))
    compute_resources: ComputeResources = field(default_factory=lambda: ComputeResources(4, 2.0, 8.0, 6.0))
    
    # Privacy and security
    privacy_budget_used: PrivacyBudget = field(default_factory=lambda: PrivacyBudget(1.0, 1e-5))
    privacy_mechanism: PrivacyMechanism = PrivacyMechanism.NONE
    noise_added: bool = False
    secure_aggregation_ready: bool = False
    
    # Compression and optimization
    compression_type: CompressionType = CompressionType.NONE
    compression_ratio: float = 1.0
    original_size_bytes: Optional[int] = None
    
    # Timestamps and versioning
    created_timestamp: datetime = field(default_factory=datetime.now)
    model_version: str = "1.0"
    client_version: str = "1.0"
    
    # Validation and integrity
    checksum: Optional[str] = None
    signature: Optional[str] = None
    
    # Custom metadata
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing"""
        if self.checksum is None:
            self.checksum = self._calculate_checksum()
        
        if self.original_size_bytes is None:
            self.original_size_bytes = self.model_size_bytes
    
    def _calculate_checksum(self) -> str:
        """Calculate checksum for integrity verification"""
        hasher = hashlib.sha256()
        hasher.update(self.model_weights)
        hasher.update(self.client_id.encode())
        hasher.update(str(self.training_rounds).encode())
        return hasher.hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify model update integrity"""
        return self.checksum == self._calculate_checksum()
    
    def apply_compression(self, compression_type: CompressionType, **kwargs) -> 'EnhancedModelUpdate':
        """Apply compression to model weights"""
        if compression_type == CompressionType.GZIP:
            compressed_weights = gzip.compress(self.model_weights)
            compression_ratio = len(compressed_weights) / len(self.model_weights)
        
        elif compression_type == CompressionType.QUANTIZATION:
            # Simple quantization example
            weights_array = np.frombuffer(self.model_weights, dtype=np.float32)
            bits = kwargs.get('bits', 8)
            
            # Quantize to specified bits
            min_val, max_val = weights_array.min(), weights_array.max()
            scale = (max_val - min_val) / (2**bits - 1)
            quantized = np.round((weights_array - min_val) / scale).astype(np.uint8)
            
            # Store quantization parameters
            quant_params = struct.pack('ff', min_val, scale)
            compressed_weights = quant_params + quantized.tobytes()
            compression_ratio = len(compressed_weights) / len(self.model_weights)
        
        elif compression_type == CompressionType.SPARSIFICATION:
            # Sparsification by magnitude
            weights_array = np.frombuffer(self.model_weights, dtype=np.float32)
            threshold = kwargs.get('threshold', 0.01)
            
            # Keep only weights above threshold
            mask = np.abs(weights_array) > threshold
            sparse_weights = weights_array * mask
            
            # Simple sparse representation (could be optimized)
            compressed_weights = sparse_weights.tobytes()
            compression_ratio = np.sum(mask) / len(weights_array)
        
        else:
            compressed_weights = self.model_weights
            compression_ratio = 1.0
        
        # Create new update with compressed weights
        new_update = EnhancedModelUpdate(
            client_id=self.client_id,
            model_weights=compressed_weights,
            model_size_bytes=len(compressed_weights),
            training_rounds=self.training_rounds,
            local_epochs=self.local_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            samples_used=self.samples_used,
            training_time_seconds=self.training_time_seconds,
            training_loss=self.training_loss,
            validation_loss=self.validation_loss,
            training_accuracy=self.training_accuracy,
            validation_accuracy=self.validation_accuracy,
            convergence_metric=self.convergence_metric,
            data_distribution=self.data_distribution.copy(),
            signal_quality_stats=self.signal_quality_stats,
            modulation_diversity=self.modulation_diversity,
            snr_distribution=self.snr_distribution,
            network_conditions=self.network_conditions,
            compute_resources=self.compute_resources,
            privacy_budget_used=self.privacy_budget_used,
            privacy_mechanism=self.privacy_mechanism,
            noise_added=self.noise_added,
            secure_aggregation_ready=self.secure_aggregation_ready,
            compression_type=compression_type,
            compression_ratio=compression_ratio,
            original_size_bytes=self.original_size_bytes,
            created_timestamp=self.created_timestamp,
            model_version=self.model_version,
            client_version=self.client_version,
            custom_metadata=self.custom_metadata.copy()
        )
        
        return new_update
    
    def apply_differential_privacy(self, epsilon: float, delta: float = 1e-5, 
                                 sensitivity: float = 1.0) -> 'EnhancedModelUpdate':
        """Apply differential privacy to model update"""
        if not self.privacy_budget_used.can_spend(epsilon, delta):
            raise ValueError("Insufficient privacy budget")
        
        # Add Gaussian noise for differential privacy
        weights_array = np.frombuffer(self.model_weights, dtype=np.float32)
        
        # Calculate noise scale
        sigma = np.sqrt(2 * np.log(1.25 / delta)) * sensitivity / epsilon
        noise = np.random.normal(0, sigma, weights_array.shape)
        
        # Add noise to weights
        noisy_weights = weights_array + noise
        noisy_weights_bytes = noisy_weights.astype(np.float32).tobytes()
        
        # Update privacy budget
        new_privacy_budget = PrivacyBudget(
            epsilon=self.privacy_budget_used.epsilon,
            delta=self.privacy_budget_used.delta,
            consumed_epsilon=self.privacy_budget_used.consumed_epsilon + epsilon,
            consumed_delta=self.privacy_budget_used.consumed_delta + delta,
            mechanism="gaussian",
            sensitivity=sensitivity
        )
        
        # Create new update with noisy weights
        new_update = EnhancedModelUpdate(
            client_id=self.client_id,
            model_weights=noisy_weights_bytes,
            model_size_bytes=len(noisy_weights_bytes),
            training_rounds=self.training_rounds,
            local_epochs=self.local_epochs,
            batch_size=self.batch_size,
            learning_rate=self.learning_rate,
            samples_used=self.samples_used,
            training_time_seconds=self.training_time_seconds,
            training_loss=self.training_loss,
            validation_loss=self.validation_loss,
            training_accuracy=self.training_accuracy,
            validation_accuracy=self.validation_accuracy,
            convergence_metric=self.convergence_metric,
            data_distribution=self.data_distribution.copy(),
            signal_quality_stats=self.signal_quality_stats,
            modulation_diversity=self.modulation_diversity,
            snr_distribution=self.snr_distribution,
            network_conditions=self.network_conditions,
            compute_resources=self.compute_resources,
            privacy_budget_used=new_privacy_budget,
            privacy_mechanism=PrivacyMechanism.DIFFERENTIAL_PRIVACY,
            noise_added=True,
            secure_aggregation_ready=self.secure_aggregation_ready,
            compression_type=self.compression_type,
            compression_ratio=self.compression_ratio,
            original_size_bytes=self.original_size_bytes,
            created_timestamp=self.created_timestamp,
            model_version=self.model_version,
            client_version=self.client_version,
            custom_metadata=self.custom_metadata.copy()
        )
        
        return new_update
    
    def estimate_transmission_time(self) -> float:
        """Estimate time to transmit this update"""
        return self.network_conditions.get_transmission_time_estimate(self.model_size_bytes)
    
    def can_train_locally(self, model_size_mb: float) -> bool:
        """Check if client can handle local training"""
        return self.compute_resources.can_handle_training(model_size_mb, self.batch_size)
    
    def to_legacy_format(self) -> ModelUpdate:
        """Convert to legacy ModelUpdate format for compatibility"""
        training_metrics = {
            'training_loss': self.training_loss,
            'validation_loss': self.validation_loss,
            'training_accuracy': self.training_accuracy,
            'validation_accuracy': self.validation_accuracy,
            'convergence_metric': self.convergence_metric,
            'local_epochs': self.local_epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'samples_used': self.samples_used
        }
        
        data_statistics = {
            'data_distribution': self.data_distribution,
            'signal_quality_stats': self.signal_quality_stats,
            'modulation_diversity': self.modulation_diversity,
            'snr_distribution': self.snr_distribution
        }
        
        network_conditions = self.network_conditions.to_dict()
        
        return ModelUpdate(
            client_id=self.client_id,
            model_weights=self.model_weights,
            training_metrics=training_metrics,
            data_statistics=data_statistics,
            computation_time=self.training_time_seconds,
            network_conditions=network_conditions,
            privacy_budget_used=self.privacy_budget_used.consumed_epsilon
        )
    
    def serialize(self, format: str = 'pickle') -> bytes:
        """Serialize the model update"""
        if format == 'pickle':
            return pickle.dumps(self)
        elif format == 'json':
            # Convert to JSON-serializable format
            json_data = {
                'client_id': self.client_id,
                'model_weights': self.model_weights.hex(),  # Convert bytes to hex string
                'model_size_bytes': self.model_size_bytes,
                'training_rounds': self.training_rounds,
                'local_epochs': self.local_epochs,
                'batch_size': self.batch_size,
                'learning_rate': self.learning_rate,
                'samples_used': self.samples_used,
                'training_time_seconds': self.training_time_seconds,
                'training_loss': self.training_loss,
                'validation_loss': self.validation_loss,
                'training_accuracy': self.training_accuracy,
                'validation_accuracy': self.validation_accuracy,
                'convergence_metric': self.convergence_metric,
                'data_distribution': self.data_distribution,
                'signal_quality_stats': self.signal_quality_stats,
                'modulation_diversity': self.modulation_diversity,
                'snr_distribution': self.snr_distribution,
                'network_conditions': self.network_conditions.to_dict(),
                'compute_resources': self.compute_resources.to_dict(),
                'privacy_budget_used': self.privacy_budget_used.to_dict(),
                'privacy_mechanism': self.privacy_mechanism.value,
                'noise_added': self.noise_added,
                'secure_aggregation_ready': self.secure_aggregation_ready,
                'compression_type': self.compression_type.value,
                'compression_ratio': self.compression_ratio,
                'original_size_bytes': self.original_size_bytes,
                'created_timestamp': self.created_timestamp.isoformat(),
                'model_version': self.model_version,
                'client_version': self.client_version,
                'checksum': self.checksum,
                'signature': self.signature,
                'custom_metadata': self.custom_metadata
            }
            return json.dumps(json_data).encode('utf-8')
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    @classmethod
    def deserialize(cls, data: bytes, format: str = 'pickle') -> 'EnhancedModelUpdate':
        """Deserialize model update"""
        if format == 'pickle':
            return pickle.loads(data)
        elif format == 'json':
            json_data = json.loads(data.decode('utf-8'))
            
            # Reconstruct objects
            network_conditions = NetworkConditions(**json_data['network_conditions'])
            compute_resources = ComputeResources(**json_data['compute_resources'])
            privacy_budget = PrivacyBudget(**json_data['privacy_budget_used'])
            
            return cls(
                client_id=json_data['client_id'],
                model_weights=bytes.fromhex(json_data['model_weights']),
                model_size_bytes=json_data['model_size_bytes'],
                training_rounds=json_data['training_rounds'],
                local_epochs=json_data['local_epochs'],
                batch_size=json_data['batch_size'],
                learning_rate=json_data['learning_rate'],
                samples_used=json_data['samples_used'],
                training_time_seconds=json_data['training_time_seconds'],
                training_loss=json_data['training_loss'],
                validation_loss=json_data['validation_loss'],
                training_accuracy=json_data['training_accuracy'],
                validation_accuracy=json_data['validation_accuracy'],
                convergence_metric=json_data['convergence_metric'],
                data_distribution=json_data['data_distribution'],
                signal_quality_stats=json_data['signal_quality_stats'],
                modulation_diversity=json_data['modulation_diversity'],
                snr_distribution=json_data['snr_distribution'],
                network_conditions=network_conditions,
                compute_resources=compute_resources,
                privacy_budget_used=privacy_budget,
                privacy_mechanism=PrivacyMechanism(json_data['privacy_mechanism']),
                noise_added=json_data['noise_added'],
                secure_aggregation_ready=json_data['secure_aggregation_ready'],
                compression_type=CompressionType(json_data['compression_type']),
                compression_ratio=json_data['compression_ratio'],
                original_size_bytes=json_data['original_size_bytes'],
                created_timestamp=datetime.fromisoformat(json_data['created_timestamp']),
                model_version=json_data['model_version'],
                client_version=json_data['client_version'],
                checksum=json_data['checksum'],
                signature=json_data['signature'],
                custom_metadata=json_data['custom_metadata']
            )
        else:
            raise ValueError(f"Unsupported format: {format}")


@dataclass
class AggregationResult:
    """Result of model aggregation with comprehensive metadata"""
    # Aggregated model
    aggregated_weights: bytes
    model_size_bytes: int
    
    # Aggregation metadata
    aggregation_strategy: AggregationStrategy
    participating_clients: List[str]
    total_samples_used: int
    aggregation_round: int
    
    # Quality metrics
    convergence_metrics: Dict[str, float] = field(default_factory=dict)
    client_contributions: Dict[str, float] = field(default_factory=dict)  # Contribution weights
    data_diversity_metrics: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    aggregation_time_seconds: float = 0.0
    communication_overhead_bytes: int = 0
    computation_overhead_flops: int = 0
    
    # Privacy and security
    privacy_guarantees: Dict[str, float] = field(default_factory=dict)
    byzantine_clients_detected: List[str] = field(default_factory=list)
    security_violations: List[str] = field(default_factory=list)
    
    # Network and system metrics
    average_network_conditions: Optional[NetworkConditions] = None
    resource_utilization: Dict[str, float] = field(default_factory=dict)
    
    # Timestamps
    aggregation_timestamp: datetime = field(default_factory=datetime.now)
    next_round_deadline: Optional[datetime] = None
    
    # Validation
    checksum: Optional[str] = None
    
    # Custom metadata
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.checksum is None:
            self.checksum = self._calculate_checksum()
    
    def _calculate_checksum(self) -> str:
        """Calculate checksum for integrity verification"""
        hasher = hashlib.sha256()
        hasher.update(self.aggregated_weights)
        hasher.update(str(self.aggregation_round).encode())
        hasher.update(str(sorted(self.participating_clients)).encode())
        return hasher.hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify aggregation result integrity"""
        return self.checksum == self._calculate_checksum()
    
    def get_client_selection_metrics(self) -> Dict[str, Any]:
        """Get metrics for client selection in next round"""
        return {
            'successful_clients': len(self.participating_clients),
            'byzantine_rate': len(self.byzantine_clients_detected) / max(1, len(self.participating_clients)),
            'average_contribution': np.mean(list(self.client_contributions.values())) if self.client_contributions else 0.0,
            'contribution_variance': np.var(list(self.client_contributions.values())) if self.client_contributions else 0.0,
            'data_diversity_score': self.data_diversity_metrics.get('diversity_score', 0.0)
        }
    
    def serialize(self, format: str = 'pickle') -> bytes:
        """Serialize aggregation result"""
        if format == 'pickle':
            return pickle.dumps(self)
        elif format == 'json':
            json_data = {
                'aggregated_weights': self.aggregated_weights.hex(),
                'model_size_bytes': self.model_size_bytes,
                'aggregation_strategy': self.aggregation_strategy.value,
                'participating_clients': self.participating_clients,
                'total_samples_used': self.total_samples_used,
                'aggregation_round': self.aggregation_round,
                'convergence_metrics': self.convergence_metrics,
                'client_contributions': self.client_contributions,
                'data_diversity_metrics': self.data_diversity_metrics,
                'aggregation_time_seconds': self.aggregation_time_seconds,
                'communication_overhead_bytes': self.communication_overhead_bytes,
                'computation_overhead_flops': self.computation_overhead_flops,
                'privacy_guarantees': self.privacy_guarantees,
                'byzantine_clients_detected': self.byzantine_clients_detected,
                'security_violations': self.security_violations,
                'average_network_conditions': self.average_network_conditions.to_dict() if self.average_network_conditions else None,
                'resource_utilization': self.resource_utilization,
                'aggregation_timestamp': self.aggregation_timestamp.isoformat(),
                'next_round_deadline': self.next_round_deadline.isoformat() if self.next_round_deadline else None,
                'checksum': self.checksum,
                'custom_metadata': self.custom_metadata
            }
            return json.dumps(json_data).encode('utf-8')
        else:
            raise ValueError(f"Unsupported format: {format}")


@dataclass
class StreamingDataBatch:
    """Batch of streaming signal data for real-time processing"""
    batch_id: str
    samples: List[EnhancedSignalSample]
    batch_timestamp: datetime
    sequence_number: int
    
    # Streaming metadata
    source_device: str
    stream_id: str
    buffer_size: int
    processing_latency_ms: float
    
    # Quality metrics
    samples_dropped: int = 0
    out_of_order_samples: int = 0
    duplicate_samples: int = 0
    
    # Processing status
    processed: bool = False
    processing_time_ms: Optional[float] = None
    
    def get_batch_statistics(self) -> Dict[str, Any]:
        """Get batch statistics"""
        if not self.samples:
            return {'empty_batch': True}
        
        modulations = [s.modulation_type.value for s in self.samples]
        snr_values = [s.quality_metrics.snr_db for s in self.samples]
        frequencies = [s.rf_params.center_frequency for s in self.samples]
        
        return {
            'sample_count': len(self.samples),
            'modulation_distribution': {mod: modulations.count(mod) for mod in set(modulations)},
            'snr_range': [min(snr_values), max(snr_values)],
            'frequency_range': [min(frequencies), max(frequencies)],
            'avg_snr': np.mean(snr_values),
            'total_duration': sum(s.duration for s in self.samples),
            'samples_dropped': self.samples_dropped,
            'quality_issues': sum(1 for s in self.samples if s.quality_metrics.snr_db < -10)
        }
    
    def serialize_for_streaming(self) -> bytes:
        """Serialize for efficient streaming"""
        # Use compressed pickle for streaming
        return gzip.compress(pickle.dumps(self))
    
    @classmethod
    def deserialize_from_stream(cls, data: bytes) -> 'StreamingDataBatch':
        """Deserialize from streaming data"""
        return pickle.loads(gzip.decompress(data))