"""
Federated Learning Client Functionality

Implements local model training with PyTorch, model compression, differential updates,
network-aware communication with retry mechanisms, and privacy-preserving training.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pickle
import gzip
import hashlib
import time
import threading
import queue
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime, timedelta
import requests
import json

from .signal_processing import FeatureVector, ModulationType
from .hardware_abstraction import SignalBuffer

logger = logging.getLogger(__name__)


class CompressionMethod(Enum):
    """Model compression methods"""
    NONE = "none"
    QUANTIZATION = "quantization"
    SPARSIFICATION = "sparsification"
    FEDAVG_DIFF = "fedavg_diff"
    GRADIENT_COMPRESSION = "gradient_compression"


class PrivacyMethod(Enum):
    """Privacy-preserving methods"""
    NONE = "none"
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    SECURE_AGGREGATION = "secure_aggregation"
    HOMOMORPHIC_ENCRYPTION = "homomorphic_encryption"


@dataclass
class TrainingConfig:
    """Configuration for federated learning training"""
    model_architecture: str = "signal_classifier"
    learning_rate: float = 0.001
    batch_size: int = 32
    local_epochs: int = 5
    optimizer: str = "adam"
    loss_function: str = "cross_entropy"
    
    # Compression settings
    compression_method: CompressionMethod = CompressionMethod.QUANTIZATION
    compression_ratio: float = 0.1
    quantization_bits: int = 8
    
    # Privacy settings
    privacy_method: PrivacyMethod = PrivacyMethod.DIFFERENTIAL_PRIVACY
    privacy_epsilon: float = 1.0
    privacy_delta: float = 1e-5
    noise_multiplier: float = 1.0
    
    # Communication settings
    max_retries: int = 3
    retry_delay: float = 1.0
    timeout: float = 30.0
    
    # Resource constraints
    max_memory_mb: int = 512
    max_cpu_percent: float = 80.0
    battery_threshold: float = 0.2


@dataclass
class ModelUpdate:
    """Federated learning model update"""
    client_id: str
    round_number: int
    model_weights: bytes
    model_metadata: Dict[str, Any]
    training_metrics: Dict[str, float]
    compression_info: Dict[str, Any]
    privacy_info: Dict[str, Any]
    timestamp: datetime
    signature: Optional[str] = None


@dataclass
class NetworkConditions:
    """Current network conditions"""
    bandwidth_mbps: float
    latency_ms: float
    packet_loss: float
    connection_type: str
    signal_strength: float


class SignalDataset(Dataset):
    """PyTorch dataset for signal classification"""
    
    def __init__(self, features: List[FeatureVector], labels: List[int]):
        self.features = features
        self.labels = labels
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        feature_vector = self._extract_feature_tensor(self.features[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return feature_vector, label
    
    def _extract_feature_tensor(self, features: FeatureVector) -> torch.Tensor:
        """Extract numerical features for training"""
        # Combine various features into a single tensor
        feature_list = []
        
        # Spectral features
        if len(features.power_spectral_density) > 0:
            # Take first 64 PSD bins
            psd_features = features.power_spectral_density[:64]
            if len(psd_features) < 64:
                psd_features = np.pad(psd_features, (0, 64 - len(psd_features)), 'constant')
            feature_list.extend(psd_features)
        else:
            feature_list.extend([0.0] * 64)
        
        # Time domain features
        feature_list.extend([
            features.rms_power,
            features.peak_power,
            features.papr,
            features.zero_crossing_rate
        ])
        
        # Statistical features
        feature_list.extend([
            features.mean_amplitude,
            features.std_amplitude,
            features.skewness,
            features.kurtosis
        ])
        
        # Constellation features
        feature_list.append(features.evm)
        
        # Cumulant features (real parts)
        if len(features.cumulants) > 0:
            cumulant_features = [np.real(c) for c in features.cumulants[:5]]
            while len(cumulant_features) < 5:
                cumulant_features.append(0.0)
            feature_list.extend(cumulant_features)
        else:
            feature_list.extend([0.0] * 5)
        
        # Convert to tensor
        return torch.tensor(feature_list, dtype=torch.float32)


class SignalClassifierModel(nn.Module):
    """Neural network for signal classification"""
    
    def __init__(self, input_size: int = 78, hidden_size: int = 128, 
                 num_classes: int = 10, dropout: float = 0.3):
        super(SignalClassifierModel, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        # Feature extraction layers
        self.feature_layers = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        
        # Classification head
        self.classifier = nn.Linear(hidden_size // 2, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x):
        features = self.feature_layers(x)
        output = self.classifier(features)
        return output
    
    def _initialize_weights(self):
        """Initialize model weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class ModelCompressor:
    """Model compression utilities"""
    
    def __init__(self, method: CompressionMethod = CompressionMethod.QUANTIZATION):
        self.method = method
        
    def compress_model(self, model: nn.Module, config: TrainingConfig) -> Tuple[bytes, Dict[str, Any]]:
        """Compress model weights"""
        if self.method == CompressionMethod.NONE:
            return self._serialize_model(model), {"method": "none", "compression_ratio": 1.0}
        
        elif self.method == CompressionMethod.QUANTIZATION:
            return self._quantize_model(model, config.quantization_bits)
        
        elif self.method == CompressionMethod.SPARSIFICATION:
            return self._sparsify_model(model, config.compression_ratio)
        
        elif self.method == CompressionMethod.GRADIENT_COMPRESSION:
            return self._compress_gradients(model, config.compression_ratio)
        
        else:
            logger.warning(f"Unknown compression method: {self.method}")
            return self._serialize_model(model), {"method": "none", "compression_ratio": 1.0}
    
    def decompress_model(self, compressed_data: bytes, metadata: Dict[str, Any], 
                        model_template: nn.Module) -> nn.Module:
        """Decompress model weights"""
        method = metadata.get("method", "none")
        
        if method == "none":
            return self._deserialize_model(compressed_data, model_template)
        elif method == "quantization":
            return self._dequantize_model(compressed_data, metadata, model_template)
        elif method == "sparsification":
            return self._desparsify_model(compressed_data, metadata, model_template)
        else:
            logger.warning(f"Unknown decompression method: {method}")
            return self._deserialize_model(compressed_data, model_template)
    
    def _serialize_model(self, model: nn.Module) -> bytes:
        """Serialize model to bytes"""
        return pickle.dumps(model.state_dict())
    
    def _deserialize_model(self, data: bytes, model_template: nn.Module) -> nn.Module:
        """Deserialize model from bytes"""
        state_dict = pickle.loads(data)
        model_template.load_state_dict(state_dict)
        return model_template
    
    def _quantize_model(self, model: nn.Module, bits: int) -> Tuple[bytes, Dict[str, Any]]:
        """Quantize model weights"""
        state_dict = model.state_dict()
        quantized_dict = {}
        
        for name, param in state_dict.items():
            # Quantize to specified bits
            param_np = param.cpu().numpy()
            
            # Calculate quantization parameters
            min_val = np.min(param_np)
            max_val = np.max(param_np)
            scale = (max_val - min_val) / (2**bits - 1)
            
            # Quantize
            quantized = np.round((param_np - min_val) / scale).astype(np.uint8)
            
            quantized_dict[name] = {
                'data': quantized,
                'min_val': min_val,
                'max_val': max_val,
                'scale': scale,
                'shape': param_np.shape
            }
        
        compressed_data = gzip.compress(pickle.dumps(quantized_dict))
        
        metadata = {
            "method": "quantization",
            "bits": bits,
            "compression_ratio": len(compressed_data) / len(pickle.dumps(state_dict)),
            "original_size": len(pickle.dumps(state_dict)),
            "compressed_size": len(compressed_data)
        }
        
        return compressed_data, metadata
    
    def _dequantize_model(self, compressed_data: bytes, metadata: Dict[str, Any], 
                         model_template: nn.Module) -> nn.Module:
        """Dequantize model weights"""
        quantized_dict = pickle.loads(gzip.decompress(compressed_data))
        state_dict = {}
        
        for name, quant_data in quantized_dict.items():
            # Dequantize
            quantized = quant_data['data']
            min_val = quant_data['min_val']
            scale = quant_data['scale']
            shape = quant_data['shape']
            
            dequantized = quantized.astype(np.float32) * scale + min_val
            dequantized = dequantized.reshape(shape)
            
            state_dict[name] = torch.tensor(dequantized)
        
        model_template.load_state_dict(state_dict)
        return model_template
    
    def _sparsify_model(self, model: nn.Module, sparsity_ratio: float) -> Tuple[bytes, Dict[str, Any]]:
        """Apply sparsification to model"""
        state_dict = model.state_dict()
        sparse_dict = {}
        
        for name, param in state_dict.items():
            param_np = param.cpu().numpy()
            
            # Calculate threshold for sparsification
            threshold = np.percentile(np.abs(param_np), sparsity_ratio * 100)
            
            # Create sparse representation
            mask = np.abs(param_np) > threshold
            sparse_values = param_np[mask]
            sparse_indices = np.where(mask)
            
            sparse_dict[name] = {
                'values': sparse_values,
                'indices': sparse_indices,
                'shape': param_np.shape,
                'threshold': threshold
            }
        
        compressed_data = gzip.compress(pickle.dumps(sparse_dict))
        
        metadata = {
            "method": "sparsification",
            "sparsity_ratio": sparsity_ratio,
            "compression_ratio": len(compressed_data) / len(pickle.dumps(state_dict)),
            "original_size": len(pickle.dumps(state_dict)),
            "compressed_size": len(compressed_data)
        }
        
        return compressed_data, metadata
    
    def _desparsify_model(self, compressed_data: bytes, metadata: Dict[str, Any], 
                         model_template: nn.Module) -> nn.Module:
        """Reconstruct model from sparse representation"""
        sparse_dict = pickle.loads(gzip.decompress(compressed_data))
        state_dict = {}
        
        for name, sparse_data in sparse_dict.items():
            # Reconstruct dense tensor
            shape = sparse_data['shape']
            dense_param = np.zeros(shape)
            
            values = sparse_data['values']
            indices = sparse_data['indices']
            
            dense_param[indices] = values
            state_dict[name] = torch.tensor(dense_param)
        
        model_template.load_state_dict(state_dict)
        return model_template
    
    def _compress_gradients(self, model: nn.Module, compression_ratio: float) -> Tuple[bytes, Dict[str, Any]]:
        """Compress gradients using top-k sparsification"""
        gradients = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_np = param.grad.cpu().numpy()
                
                # Top-k compression
                k = int(grad_np.size * compression_ratio)
                flat_grad = grad_np.flatten()
                
                # Get top-k indices
                top_k_indices = np.argpartition(np.abs(flat_grad), -k)[-k:]
                top_k_values = flat_grad[top_k_indices]
                
                gradients[name] = {
                    'values': top_k_values,
                    'indices': top_k_indices,
                    'shape': grad_np.shape
                }
        
        compressed_data = gzip.compress(pickle.dumps(gradients))
        
        metadata = {
            "method": "gradient_compression",
            "compression_ratio": compression_ratio,
            "compressed_size": len(compressed_data)
        }
        
        return compressed_data, metadata


class PrivacyEngine:
    """Privacy-preserving training utilities"""
    
    def __init__(self, method: PrivacyMethod = PrivacyMethod.DIFFERENTIAL_PRIVACY):
        self.method = method
        
    def apply_privacy(self, model: nn.Module, config: TrainingConfig) -> Dict[str, Any]:
        """Apply privacy-preserving mechanisms"""
        if self.method == PrivacyMethod.NONE:
            return {"method": "none", "privacy_cost": 0.0}
        
        elif self.method == PrivacyMethod.DIFFERENTIAL_PRIVACY:
            return self._apply_differential_privacy(model, config)
        
        elif self.method == PrivacyMethod.SECURE_AGGREGATION:
            return self._apply_secure_aggregation(model, config)
        
        else:
            logger.warning(f"Unknown privacy method: {self.method}")
            return {"method": "none", "privacy_cost": 0.0}
    
    def _apply_differential_privacy(self, model: nn.Module, config: TrainingConfig) -> Dict[str, Any]:
        """Apply differential privacy noise to model parameters"""
        epsilon = config.privacy_epsilon
        delta = config.privacy_delta
        noise_multiplier = config.noise_multiplier
        
        # Add Gaussian noise to model parameters
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    # Calculate noise scale
                    sensitivity = 2.0  # L2 sensitivity (simplified)
                    noise_scale = noise_multiplier * sensitivity / epsilon
                    
                    # Add noise
                    noise = torch.normal(0, noise_scale, size=param.shape)
                    param.add_(noise)
        
        # Calculate privacy cost (simplified)
        privacy_cost = epsilon
        
        return {
            "method": "differential_privacy",
            "epsilon": epsilon,
            "delta": delta,
            "noise_multiplier": noise_multiplier,
            "privacy_cost": privacy_cost
        }
    
    def _apply_secure_aggregation(self, model: nn.Module, config: TrainingConfig) -> Dict[str, Any]:
        """Apply secure aggregation (simplified implementation)"""
        # In a real implementation, this would involve cryptographic protocols
        # For now, we'll just add some randomization
        
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    # Add small random perturbation
                    noise = torch.normal(0, 0.01, size=param.shape)
                    param.add_(noise)
        
        return {
            "method": "secure_aggregation",
            "privacy_cost": 0.1  # Minimal privacy cost
        }


class NetworkManager:
    """Network-aware communication with retry mechanisms"""
    
    def __init__(self, server_url: str, client_id: str):
        self.server_url = server_url
        self.client_id = client_id
        self.session = requests.Session()
        
    def estimate_network_conditions(self) -> NetworkConditions:
        """Estimate current network conditions"""
        try:
            # Simple ping test
            start_time = time.time()
            response = self.session.get(f"{self.server_url}/ping", timeout=5.0)
            latency = (time.time() - start_time) * 1000  # ms
            
            # Estimate bandwidth (simplified)
            bandwidth = 10.0  # Mbps (placeholder)
            
            return NetworkConditions(
                bandwidth_mbps=bandwidth,
                latency_ms=latency,
                packet_loss=0.01,  # 1% (placeholder)
                connection_type="wifi",
                signal_strength=0.8
            )
            
        except Exception as e:
            logger.warning(f"Failed to estimate network conditions: {e}")
            return NetworkConditions(
                bandwidth_mbps=1.0,
                latency_ms=1000.0,
                packet_loss=0.1,
                connection_type="unknown",
                signal_strength=0.1
            )
    
    def send_model_update(self, update: ModelUpdate, config: TrainingConfig) -> bool:
        """Send model update with retry mechanism"""
        for attempt in range(config.max_retries):
            try:
                # Prepare payload
                payload = {
                    'client_id': update.client_id,
                    'round_number': update.round_number,
                    'model_weights': update.model_weights.hex(),  # Convert bytes to hex
                    'model_metadata': update.model_metadata,
                    'training_metrics': update.training_metrics,
                    'compression_info': update.compression_info,
                    'privacy_info': update.privacy_info,
                    'timestamp': update.timestamp.isoformat()
                }
                
                # Send request
                response = self.session.post(
                    f"{self.server_url}/model_update",
                    json=payload,
                    timeout=config.timeout
                )
                
                if response.status_code == 200:
                    logger.info(f"Model update sent successfully (attempt {attempt + 1})")
                    return True
                else:
                    logger.warning(f"Server returned status {response.status_code} (attempt {attempt + 1})")
                    
            except Exception as e:
                logger.warning(f"Failed to send model update (attempt {attempt + 1}): {e}")
            
            # Wait before retry
            if attempt < config.max_retries - 1:
                time.sleep(config.retry_delay * (2 ** attempt))  # Exponential backoff
        
        logger.error("Failed to send model update after all retries")
        return False
    
    def receive_global_model(self, config: TrainingConfig) -> Optional[bytes]:
        """Receive global model from server"""
        try:
            response = self.session.get(
                f"{self.server_url}/global_model",
                params={'client_id': self.client_id},
                timeout=config.timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                model_weights = bytes.fromhex(data['model_weights'])
                logger.info("Global model received successfully")
                return model_weights
            else:
                logger.warning(f"Failed to receive global model: status {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Error receiving global model: {e}")
            return None


class FederatedLearningClient:
    """Main federated learning client"""
    
    def __init__(self, client_id: str, server_url: str, config: TrainingConfig):
        self.client_id = client_id
        self.server_url = server_url
        self.config = config
        
        # Initialize components
        self.model = SignalClassifierModel()
        self.compressor = ModelCompressor(config.compression_method)
        self.privacy_engine = PrivacyEngine(config.privacy_method)
        self.network_manager = NetworkManager(server_url, client_id)
        
        # Training state
        self.current_round = 0
        self.training_history = []
        self.is_training = False
        
        # Setup optimizer and loss function
        self._setup_training()
        
    def _setup_training(self):
        """Setup optimizer and loss function"""
        if self.config.optimizer == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer == "sgd":
            self.optimizer = optim.SGD(self.model.parameters(), lr=self.config.learning_rate)
        else:
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
        
        if self.config.loss_function == "cross_entropy":
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = nn.CrossEntropyLoss()
    
    def train_local_model(self, features: List[FeatureVector], 
                         labels: List[int]) -> ModelUpdate:
        """Train local model on client data"""
        try:
            self.is_training = True
            start_time = time.time()
            
            # Create dataset and dataloader
            dataset = SignalDataset(features, labels)
            dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=True)
            
            # Training loop
            self.model.train()
            epoch_losses = []
            
            for epoch in range(self.config.local_epochs):
                epoch_loss = 0.0
                num_batches = 0
                
                for batch_features, batch_labels in dataloader:
                    # Forward pass
                    outputs = self.model(batch_features)
                    loss = self.criterion(outputs, batch_labels)
                    
                    # Backward pass
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    epoch_loss += loss.item()
                    num_batches += 1
                
                avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
                epoch_losses.append(avg_epoch_loss)
                
                logger.info(f"Epoch {epoch + 1}/{self.config.local_epochs}, Loss: {avg_epoch_loss:.4f}")
            
            # Apply privacy mechanisms
            privacy_info = self.privacy_engine.apply_privacy(self.model, self.config)
            
            # Compress model
            compressed_weights, compression_info = self.compressor.compress_model(
                self.model, self.config
            )
            
            # Calculate training metrics
            training_time = time.time() - start_time
            training_metrics = {
                'training_time': training_time,
                'num_samples': len(features),
                'num_epochs': self.config.local_epochs,
                'final_loss': epoch_losses[-1] if epoch_losses else 0.0,
                'avg_loss': np.mean(epoch_losses) if epoch_losses else 0.0
            }
            
            # Create model update
            model_update = ModelUpdate(
                client_id=self.client_id,
                round_number=self.current_round,
                model_weights=compressed_weights,
                model_metadata={
                    'model_architecture': self.config.model_architecture,
                    'input_size': self.model.input_size,
                    'hidden_size': self.model.hidden_size,
                    'num_classes': self.model.num_classes
                },
                training_metrics=training_metrics,
                compression_info=compression_info,
                privacy_info=privacy_info,
                timestamp=datetime.now()
            )
            
            # Add to training history
            self.training_history.append({
                'round': self.current_round,
                'metrics': training_metrics,
                'timestamp': datetime.now()
            })
            
            self.is_training = False
            logger.info(f"Local training completed in {training_time:.2f} seconds")
            
            return model_update
            
        except Exception as e:
            self.is_training = False
            logger.error(f"Error in local training: {e}")
            raise
    
    def update_global_model(self, global_model_weights: bytes) -> bool:
        """Update local model with global model weights"""
        try:
            # Create temporary model for decompression
            temp_model = SignalClassifierModel(
                input_size=self.model.input_size,
                hidden_size=self.model.hidden_size,
                num_classes=self.model.num_classes
            )
            
            # Decompress global model
            # Note: In a real implementation, we'd need compression metadata
            global_model = self.compressor.decompress_model(
                global_model_weights, 
                {"method": "none"},  # Simplified
                temp_model
            )
            
            # Update local model
            self.model.load_state_dict(global_model.state_dict())
            
            logger.info("Global model update applied successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error updating global model: {e}")
            return False
    
    def participate_in_round(self, features: List[FeatureVector], 
                           labels: List[int]) -> bool:
        """Participate in a federated learning round"""
        try:
            logger.info(f"Starting federated learning round {self.current_round}")
            
            # Check network conditions
            network_conditions = self.network_manager.estimate_network_conditions()
            logger.info(f"Network conditions: {network_conditions.bandwidth_mbps:.1f} Mbps, "
                       f"{network_conditions.latency_ms:.1f} ms latency")
            
            # Train local model
            model_update = self.train_local_model(features, labels)
            
            # Send model update to server
            success = self.network_manager.send_model_update(model_update, self.config)
            
            if success:
                # Receive updated global model
                global_model_weights = self.network_manager.receive_global_model(self.config)
                
                if global_model_weights:
                    self.update_global_model(global_model_weights)
                    self.current_round += 1
                    logger.info(f"Federated learning round {self.current_round - 1} completed successfully")
                    return True
                else:
                    logger.warning("Failed to receive global model")
                    return False
            else:
                logger.error("Failed to send model update")
                return False
                
        except Exception as e:
            logger.error(f"Error in federated learning round: {e}")
            return False
    
    def evaluate_model(self, features: List[FeatureVector], 
                      labels: List[int]) -> Dict[str, float]:
        """Evaluate model performance"""
        try:
            dataset = SignalDataset(features, labels)
            dataloader = DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)
            
            self.model.eval()
            total_loss = 0.0
            correct_predictions = 0
            total_samples = 0
            
            with torch.no_grad():
                for batch_features, batch_labels in dataloader:
                    outputs = self.model(batch_features)
                    loss = self.criterion(outputs, batch_labels)
                    
                    total_loss += loss.item()
                    
                    # Calculate accuracy
                    _, predicted = torch.max(outputs.data, 1)
                    total_samples += batch_labels.size(0)
                    correct_predictions += (predicted == batch_labels).sum().item()
            
            avg_loss = total_loss / len(dataloader)
            accuracy = correct_predictions / total_samples if total_samples > 0 else 0.0
            
            metrics = {
                'loss': avg_loss,
                'accuracy': accuracy,
                'num_samples': total_samples
            }
            
            logger.info(f"Model evaluation: Loss={avg_loss:.4f}, Accuracy={accuracy:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {'loss': float('inf'), 'accuracy': 0.0, 'num_samples': 0}
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        return {
            'client_id': self.client_id,
            'current_round': self.current_round,
            'is_training': self.is_training,
            'training_history': self.training_history[-10:],  # Last 10 rounds
            'model_parameters': sum(p.numel() for p in self.model.parameters()),
            'config': {
                'learning_rate': self.config.learning_rate,
                'batch_size': self.config.batch_size,
                'local_epochs': self.config.local_epochs,
                'compression_method': self.config.compression_method.value,
                'privacy_method': self.config.privacy_method.value
            }
        }
    
    def save_model(self, filepath: str) -> bool:
        """Save model to file"""
        try:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'current_round': self.current_round,
                'config': self.config,
                'training_history': self.training_history
            }, filepath)
            
            logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load model from file"""
        try:
            checkpoint = torch.load(filepath)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.current_round = checkpoint['current_round']
            self.training_history = checkpoint['training_history']
            
            logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False