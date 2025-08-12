"""
Federated learning aggregation algorithms
"""
import pickle
import logging
import numpy as np
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

from ..common.interfaces import ModelUpdate
from ..common.config import AppConfig

logger = logging.getLogger(__name__)


class BaseAggregationAlgorithm(ABC):
    """Abstract base class for aggregation algorithms"""
    
    def __init__(self, config: AppConfig):
        self.config = config
    
    @abstractmethod
    async def aggregate(self, updates: List[ModelUpdate]) -> bytes:
        """Aggregate model updates"""
        pass
    
    @abstractmethod
    def get_algorithm_name(self) -> str:
        """Get algorithm name"""
        pass


class FedAvgAggregator(BaseAggregationAlgorithm):
    """Federated Averaging (FedAvg) aggregation algorithm"""
    
    def __init__(self, config: AppConfig):
        super().__init__(config)
        self.algorithm_name = "FedAvg"
    
    async def aggregate(self, updates: List[ModelUpdate]) -> bytes:
        """Perform FedAvg aggregation"""
        if not updates:
            raise ValueError("No updates to aggregate")
        
        logger.info(f"Aggregating {len(updates)} model updates using FedAvg")
        
        try:
            # Deserialize all model weights
            client_weights = []
            client_data_sizes = []
            
            for update in updates:
                weights = pickle.loads(update.model_weights)
                data_size = update.data_statistics.get('num_samples', 1)
                
                client_weights.append(weights)
                client_data_sizes.append(data_size)
            
            # Perform weighted averaging
            aggregated_weights = self._weighted_average(client_weights, client_data_sizes)
            
            # Serialize aggregated weights
            return pickle.dumps(aggregated_weights)
        
        except Exception as e:
            logger.error(f"FedAvg aggregation failed: {e}")
            raise
    
    def get_algorithm_name(self) -> str:
        return self.algorithm_name
    
    def _weighted_average(self, weights_list: List[Any], data_sizes: List[int]) -> Any:
        """Perform weighted average of model weights"""
        total_samples = sum(data_sizes)
        
        if total_samples == 0:
            # Fallback to simple average
            return self._simple_average(weights_list)
        
        # Handle different weight formats
        if isinstance(weights_list[0], dict):
            return self._weighted_average_dict(weights_list, data_sizes, total_samples)
        elif isinstance(weights_list[0], list):
            return self._weighted_average_list(weights_list, data_sizes, total_samples)
        elif isinstance(weights_list[0], np.ndarray):
            return self._weighted_average_array(weights_list, data_sizes, total_samples)
        else:
            raise ValueError(f"Unsupported weight format: {type(weights_list[0])}")
    
    def _weighted_average_dict(self, weights_list: List[Dict], data_sizes: List[int], total_samples: int) -> Dict:
        """Weighted average for dictionary-based weights (PyTorch state_dict format)"""
        aggregated = {}
        
        # Get all parameter names from first model
        param_names = weights_list[0].keys()
        
        for param_name in param_names:
            # Collect parameter values from all clients
            param_values = [weights[param_name] for weights in weights_list]
            
            # Convert to numpy arrays if needed
            if hasattr(param_values[0], 'numpy'):
                param_values = [p.numpy() for p in param_values]
            
            # Weighted average
            weighted_sum = np.zeros_like(param_values[0])
            for param_value, data_size in zip(param_values, data_sizes):
                weight = data_size / total_samples
                weighted_sum += weight * param_value
            
            aggregated[param_name] = weighted_sum
        
        return aggregated
    
    def _weighted_average_list(self, weights_list: List[List], data_sizes: List[int], total_samples: int) -> List:
        """Weighted average for list-based weights"""
        aggregated = []
        
        for i in range(len(weights_list[0])):
            # Collect i-th layer weights from all clients
            layer_weights = [weights[i] for weights in weights_list]
            
            # Convert to numpy arrays
            layer_weights = [np.array(w) for w in layer_weights]
            
            # Weighted average
            weighted_sum = np.zeros_like(layer_weights[0])
            for layer_weight, data_size in zip(layer_weights, data_sizes):
                weight = data_size / total_samples
                weighted_sum += weight * layer_weight
            
            aggregated.append(weighted_sum)
        
        return aggregated
    
    def _weighted_average_array(self, weights_list: List[np.ndarray], data_sizes: List[int], total_samples: int) -> np.ndarray:
        """Weighted average for numpy array weights"""
        weighted_sum = np.zeros_like(weights_list[0])
        
        for weights, data_size in zip(weights_list, data_sizes):
            weight = data_size / total_samples
            weighted_sum += weight * weights
        
        return weighted_sum
    
    def _simple_average(self, weights_list: List[Any]) -> Any:
        """Simple average fallback"""
        if isinstance(weights_list[0], dict):
            return self._simple_average_dict(weights_list)
        elif isinstance(weights_list[0], list):
            return self._simple_average_list(weights_list)
        elif isinstance(weights_list[0], np.ndarray):
            return self._simple_average_array(weights_list)
        else:
            raise ValueError(f"Unsupported weight format: {type(weights_list[0])}")
    
    def _simple_average_dict(self, weights_list: List[Dict]) -> Dict:
        """Simple average for dictionary weights"""
        aggregated = {}
        param_names = weights_list[0].keys()
        
        for param_name in param_names:
            param_values = [weights[param_name] for weights in weights_list]
            
            if hasattr(param_values[0], 'numpy'):
                param_values = [p.numpy() for p in param_values]
            
            aggregated[param_name] = np.mean(param_values, axis=0)
        
        return aggregated
    
    def _simple_average_list(self, weights_list: List[List]) -> List:
        """Simple average for list weights"""
        aggregated = []
        
        for i in range(len(weights_list[0])):
            layer_weights = [np.array(weights[i]) for weights in weights_list]
            aggregated.append(np.mean(layer_weights, axis=0))
        
        return aggregated
    
    def _simple_average_array(self, weights_list: List[np.ndarray]) -> np.ndarray:
        """Simple average for array weights"""
        return np.mean(weights_list, axis=0)


class KrumAggregator(BaseAggregationAlgorithm):
    """Krum Byzantine-fault-tolerant aggregation algorithm"""
    
    def __init__(self, config: AppConfig):
        super().__init__(config)
        self.algorithm_name = "Krum"
        self.byzantine_ratio = 0.33  # Assume up to 33% Byzantine clients
    
    async def initialize(self):
        """Initialize the aggregator"""
        pass
    
    async def shutdown(self):
        """Shutdown the aggregator"""
        pass
    
    async def aggregate(self, updates: List[ModelUpdate]) -> bytes:
        """Perform Krum aggregation"""
        if not updates:
            raise ValueError("No updates to aggregate")
        
        logger.info(f"Aggregating {len(updates)} model updates using Krum")
        
        try:
            # Deserialize all model weights
            client_weights = []
            for update in updates:
                weights = pickle.loads(update.model_weights)
                client_weights.append(weights)
            
            # Calculate number of Byzantine clients to tolerate
            n = len(client_weights)
            f = int(n * self.byzantine_ratio)
            
            if n < 2 * f + 3:
                logger.warning(f"Not enough clients for Byzantine tolerance: {n} clients, need at least {2 * f + 3}")
                # Fallback to FedAvg
                fedavg_aggregator = FedAvgAggregator(self.config)
                return await fedavg_aggregator.aggregate(updates)
            
            # Select best client using Krum algorithm
            selected_weights = self._krum_selection(client_weights, f)
            
            return pickle.dumps(selected_weights)
        
        except Exception as e:
            logger.error(f"Krum aggregation failed: {e}")
            raise
    
    def get_algorithm_name(self) -> str:
        return self.algorithm_name
    
    def _krum_selection(self, weights_list: List[Any], f: int) -> Any:
        """Select the best client weights using Krum algorithm"""
        n = len(weights_list)
        
        # Convert weights to flat vectors for distance calculation
        flat_weights = []
        for weights in weights_list:
            if isinstance(weights, dict):
                flat = np.concatenate([np.array(w).flatten() for w in weights.values()])
            elif isinstance(weights, list):
                flat = np.concatenate([np.array(w).flatten() for w in weights])
            else:
                flat = np.array(weights).flatten()
            flat_weights.append(flat)
        
        # Calculate pairwise distances
        distances = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.linalg.norm(flat_weights[i] - flat_weights[j])
                distances[i, j] = distances[j, i] = dist
        
        # Calculate Krum scores
        scores = []
        for i in range(n):
            # Sum of distances to n-f-2 closest clients
            sorted_distances = np.sort(distances[i])
            score = np.sum(sorted_distances[1:n-f-1])  # Exclude self (distance 0)
            scores.append(score)
        
        # Select client with minimum score
        best_client_idx = np.argmin(scores)
        return weights_list[best_client_idx]


class TrimmedMeanAggregator(BaseAggregationAlgorithm):
    """Trimmed Mean Byzantine-fault-tolerant aggregation algorithm"""
    
    def __init__(self, config: AppConfig):
        super().__init__(config)
        self.algorithm_name = "TrimmedMean"
        self.trim_ratio = 0.2  # Trim 20% from each end
    
    async def initialize(self):
        """Initialize the aggregator"""
        pass
    
    async def shutdown(self):
        """Shutdown the aggregator"""
        pass
    
    async def aggregate(self, updates: List[ModelUpdate]) -> bytes:
        """Perform Trimmed Mean aggregation"""
        if not updates:
            raise ValueError("No updates to aggregate")
        
        logger.info(f"Aggregating {len(updates)} model updates using Trimmed Mean")
        
        try:
            # Deserialize all model weights
            client_weights = []
            for update in updates:
                weights = pickle.loads(update.model_weights)
                client_weights.append(weights)
            
            # Perform trimmed mean aggregation
            aggregated_weights = self._trimmed_mean(client_weights)
            
            return pickle.dumps(aggregated_weights)
        
        except Exception as e:
            logger.error(f"Trimmed Mean aggregation failed: {e}")
            raise
    
    def get_algorithm_name(self) -> str:
        return self.algorithm_name
    
    def _trimmed_mean(self, weights_list: List[Any]) -> Any:
        """Perform trimmed mean aggregation"""
        if isinstance(weights_list[0], dict):
            return self._trimmed_mean_dict(weights_list)
        elif isinstance(weights_list[0], list):
            return self._trimmed_mean_list(weights_list)
        elif isinstance(weights_list[0], np.ndarray):
            return self._trimmed_mean_array(weights_list)
        else:
            raise ValueError(f"Unsupported weight format: {type(weights_list[0])}")
    
    def _trimmed_mean_dict(self, weights_list: List[Dict]) -> Dict:
        """Trimmed mean for dictionary weights"""
        aggregated = {}
        param_names = weights_list[0].keys()
        
        for param_name in param_names:
            param_values = [weights[param_name] for weights in weights_list]
            
            if hasattr(param_values[0], 'numpy'):
                param_values = [p.numpy() for p in param_values]
            
            # Convert to numpy arrays and stack
            param_array = np.stack([np.array(p) for p in param_values])
            
            # Calculate trimmed mean along client axis
            aggregated[param_name] = self._calculate_trimmed_mean(param_array, axis=0)
        
        return aggregated
    
    def _trimmed_mean_list(self, weights_list: List[List]) -> List:
        """Trimmed mean for list weights"""
        aggregated = []
        
        for i in range(len(weights_list[0])):
            layer_weights = [np.array(weights[i]) for weights in weights_list]
            layer_array = np.stack(layer_weights)
            aggregated.append(self._calculate_trimmed_mean(layer_array, axis=0))
        
        return aggregated
    
    def _trimmed_mean_array(self, weights_list: List[np.ndarray]) -> np.ndarray:
        """Trimmed mean for array weights"""
        weights_array = np.stack(weights_list)
        return self._calculate_trimmed_mean(weights_array, axis=0)
    
    def _calculate_trimmed_mean(self, data: np.ndarray, axis: int) -> np.ndarray:
        """Calculate trimmed mean along specified axis"""
        n = data.shape[axis]
        trim_count = int(n * self.trim_ratio)
        
        if trim_count == 0:
            return np.mean(data, axis=axis)
        
        # Sort along the specified axis
        sorted_data = np.sort(data, axis=axis)
        
        # Trim from both ends
        if axis == 0:
            trimmed_data = sorted_data[trim_count:-trim_count]
        else:
            # Handle other axes if needed
            trimmed_data = sorted_data
        
        return np.mean(trimmed_data, axis=axis)


class WeightedAggregator(BaseAggregationAlgorithm):
    """Weighted aggregation based on client data quality and size"""
    
    def __init__(self, config: AppConfig):
        super().__init__(config)
        self.algorithm_name = "Weighted"
        self.quality_weight = 0.3
        self.size_weight = 0.7
    
    async def initialize(self):
        """Initialize the aggregator"""
        pass
    
    async def shutdown(self):
        """Shutdown the aggregator"""
        pass
    
    async def aggregate(self, updates: List[ModelUpdate]) -> bytes:
        """Perform weighted aggregation"""
        if not updates:
            raise ValueError("No updates to aggregate")
        
        logger.info(f"Aggregating {len(updates)} model updates using Weighted aggregation")
        
        try:
            # Deserialize all model weights
            client_weights = []
            weights = []
            
            for update in updates:
                model_weights = pickle.loads(update.model_weights)
                client_weights.append(model_weights)
                
                # Calculate weight based on data quality and size
                data_size = update.data_statistics.get('num_samples', 1)
                accuracy = update.training_metrics.get('accuracy', 0.5)
                loss = update.training_metrics.get('loss', 1.0)
                
                # Quality score (higher is better)
                quality_score = accuracy / (1.0 + loss)
                
                # Combined weight
                weight = self.size_weight * data_size + self.quality_weight * quality_score * 1000
                weights.append(weight)
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            else:
                weights = [1.0 / len(weights)] * len(weights)
            
            # Perform weighted aggregation
            aggregated_weights = self._weighted_aggregation(client_weights, weights)
            
            return pickle.dumps(aggregated_weights)
        
        except Exception as e:
            logger.error(f"Weighted aggregation failed: {e}")
            raise
    
    def get_algorithm_name(self) -> str:
        return self.algorithm_name
    
    def _weighted_aggregation(self, weights_list: List[Any], weights: List[float]) -> Any:
        """Perform weighted aggregation"""
        if isinstance(weights_list[0], dict):
            return self._weighted_aggregation_dict(weights_list, weights)
        elif isinstance(weights_list[0], list):
            return self._weighted_aggregation_list(weights_list, weights)
        elif isinstance(weights_list[0], np.ndarray):
            return self._weighted_aggregation_array(weights_list, weights)
        else:
            raise ValueError(f"Unsupported weight format: {type(weights_list[0])}")
    
    def _weighted_aggregation_dict(self, weights_list: List[Dict], weights: List[float]) -> Dict:
        """Weighted aggregation for dictionary weights"""
        aggregated = {}
        param_names = weights_list[0].keys()
        
        for param_name in param_names:
            param_values = [w[param_name] for w in weights_list]
            
            if hasattr(param_values[0], 'numpy'):
                param_values = [p.numpy() for p in param_values]
            
            # Weighted sum
            weighted_sum = np.zeros_like(param_values[0])
            for param_value, weight in zip(param_values, weights):
                weighted_sum += weight * np.array(param_value)
            
            aggregated[param_name] = weighted_sum
        
        return aggregated
    
    def _weighted_aggregation_list(self, weights_list: List[List], weights: List[float]) -> List:
        """Weighted aggregation for list weights"""
        aggregated = []
        
        for i in range(len(weights_list[0])):
            layer_weights = [w[i] for w in weights_list]
            
            # Weighted sum
            weighted_sum = np.zeros_like(layer_weights[0])
            for layer_weight, weight in zip(layer_weights, weights):
                weighted_sum += weight * np.array(layer_weight)
            
            aggregated.append(weighted_sum)
        
        return aggregated
    
    def _weighted_aggregation_array(self, weights_list: List[np.ndarray], weights: List[float]) -> np.ndarray:
        """Weighted aggregation for array weights"""
        weighted_sum = np.zeros_like(weights_list[0])
        
        for weight_array, weight in zip(weights_list, weights):
            weighted_sum += weight * weight_array
        
        return weighted_sum


class AggregatorFactory:
    """Factory for creating aggregation algorithms"""
    
    _algorithms = {
        'fedavg': FedAvgAggregator,
        'krum': KrumAggregator,
        'trimmed_mean': TrimmedMeanAggregator,
        'weighted': WeightedAggregator,
    }
    
    @classmethod
    def create_aggregator(cls, algorithm_name: str, config: AppConfig) -> BaseAggregationAlgorithm:
        """Create aggregator instance"""
        algorithm_name = algorithm_name.lower()
        
        if algorithm_name not in cls._algorithms:
            raise ValueError(f"Unknown aggregation algorithm: {algorithm_name}")
        
        return cls._algorithms[algorithm_name](config)
    
    @classmethod
    def get_available_algorithms(cls) -> List[str]:
        """Get list of available algorithms"""
        return list(cls._algorithms.keys())


class FedAvgAggregator:
    """Main aggregator class that uses the selected algorithm"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.algorithm = AggregatorFactory.create_aggregator(
            config.federated_learning.aggregation_strategy,
            config
        )
    
    async def initialize(self):
        """Initialize aggregator"""
        logger.info(f"Initialized aggregator with algorithm: {self.algorithm.get_algorithm_name()}")
    
    async def shutdown(self):
        """Shutdown aggregator"""
        logger.info("Aggregator shutdown complete")
    
    async def aggregate_models(self, updates: List[ModelUpdate]) -> bytes:
        """Aggregate model updates"""
        if not updates:
            raise ValueError("No model updates to aggregate")
        
        logger.info(f"Aggregating {len(updates)} model updates")
        
        # Validate updates
        self._validate_updates(updates)
        
        # Perform aggregation
        aggregated_weights = await self.algorithm.aggregate(updates)
        
        logger.info("Model aggregation completed successfully")
        return aggregated_weights
    
    def _validate_updates(self, updates: List[ModelUpdate]):
        """Validate model updates before aggregation"""
        if not updates:
            raise ValueError("No updates provided")
        
        # Check all updates have weights
        for update in updates:
            if not update.model_weights:
                raise ValueError(f"Empty model weights from client {update.client_id}")
        
        # Validate weight compatibility (basic check)
        try:
            first_weights = pickle.loads(updates[0].model_weights)
            
            for i, update in enumerate(updates[1:], 1):
                weights = pickle.loads(update.model_weights)
                
                if type(weights) != type(first_weights):
                    raise ValueError(f"Incompatible weight types: client 0 has {type(first_weights)}, client {i} has {type(weights)}")
                
                if isinstance(weights, dict):
                    if set(weights.keys()) != set(first_weights.keys()):
                        raise ValueError(f"Incompatible parameter names between clients")
                
                elif isinstance(weights, list):
                    if len(weights) != len(first_weights):
                        raise ValueError(f"Incompatible number of layers between clients")
        
        except Exception as e:
            logger.error(f"Update validation failed: {e}")
            raise ValueError(f"Invalid model updates: {e}")
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """Get information about current algorithm"""
        return {
            'name': self.algorithm.get_algorithm_name(),
            'available_algorithms': AggregatorFactory.get_available_algorithms()
        }