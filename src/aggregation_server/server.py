"""
Core aggregation server implementation
"""
import asyncio
import logging
import pickle
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from collections import defaultdict
import numpy as np

from .auth import generate_client_token, TokenManager
from .storage import ModelStorage
from .aggregation import FedAvgAggregator
from .privacy import PrivacySecurityManager
from ..common.interfaces import BaseAggregator, ClientInfo, ModelUpdate, TrainingConfig
from ..common.config import AppConfig

logger = logging.getLogger(__name__)


class AggregationServer:
    """Main aggregation server implementation"""
    
    def __init__(self, config: AppConfig):
        self.config = config
        self.start_time = datetime.utcnow()
        
        # Core components
        self.model_storage = ModelStorage(config)
        self.aggregator = FedAvgAggregator(config)
        self.token_manager = TokenManager()
        self.privacy_security_manager = PrivacySecurityManager(config)
        
        # State management
        self.registered_clients: Dict[str, ClientInfo] = {}
        self.pending_updates: Dict[int, List[ModelUpdate]] = defaultdict(list)
        self.current_round = 0
        self.global_model_version = "1.0.0"
        self.last_aggregation_time: Optional[datetime] = None
        
        # Metrics and monitoring
        self.client_metrics: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self.round_metrics: Dict[int, Dict[str, Any]] = {}
        
        # Background tasks
        self.background_tasks: List[asyncio.Task] = []
        self.is_running = False
        
        # Advanced aggregation features
        self.convergence_history: List[Dict[str, float]] = []
        self.convergence_threshold = 0.001
        self.convergence_patience = 3
        self.selected_clients_history: List[List[str]] = []
    
    async def initialize(self):
        """Initialize the aggregation server"""
        logger.info("Initializing aggregation server...")
        
        # Initialize storage
        await self.model_storage.initialize()
        
        # Initialize aggregator
        await self.aggregator.initialize()
        
        # Start background tasks
        self.is_running = True
        self.background_tasks = [
            asyncio.create_task(self._cleanup_expired_tokens()),
            asyncio.create_task(self._monitor_training_progress()),
            asyncio.create_task(self._periodic_aggregation()),
            asyncio.create_task(self._monitor_convergence())
        ]
        
        logger.info("Aggregation server initialized successfully")
    
    async def shutdown(self):
        """Shutdown the aggregation server"""
        logger.info("Shutting down aggregation server...")
        
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Shutdown components
        await self.model_storage.shutdown()
        await self.aggregator.shutdown()
        
        logger.info("Aggregation server shutdown complete")
    
    async def register_client(self, client_info: ClientInfo) -> str:
        """Register a new client and return authentication token"""
        logger.info(f"Registering client: {client_info.client_id}")
        
        # Validate client information
        if not client_info.client_id:
            raise ValueError("Client ID is required")
        
        if client_info.client_type not in ["SDR", "Mobile", "Simulated"]:
            raise ValueError(f"Invalid client type: {client_info.client_type}")
        
        # Check if client already registered
        if client_info.client_id in self.registered_clients:
            logger.warning(f"Client {client_info.client_id} already registered, updating info")
        
        # Store client information
        self.registered_clients[client_info.client_id] = client_info
        
        # Generate authentication token
        token = generate_client_token(
            client_info.client_id,
            {
                'client_type': client_info.client_type,
                'capabilities': client_info.capabilities,
                'reputation_score': client_info.reputation_score
            }
        )
        
        logger.info(f"Client {client_info.client_id} registered successfully")
        return token
    
    async def receive_model_update(self, client_id: str, model_update: ModelUpdate) -> bool:
        """Receive model update from client"""
        logger.info(f"Received model update from client: {client_id}")
        
        # Validate client
        if client_id not in self.registered_clients:
            logger.error(f"Unknown client: {client_id}")
            return False
        
        # Validate model update
        if not self._validate_model_update(model_update):
            logger.error(f"Invalid model update from client: {client_id}")
            return False
        
        # Store update for current round
        self.pending_updates[self.current_round].append(model_update)
        
        # Update client metrics
        self.client_metrics[client_id].update({
            'last_update': datetime.utcnow(),
            'training_metrics': model_update.training_metrics,
            'computation_time': model_update.computation_time,
            'data_statistics': model_update.data_statistics
        })
        
        logger.info(f"Model update from {client_id} accepted for round {self.current_round}")
        
        # Check if we have enough updates to trigger aggregation
        await self._check_aggregation_trigger()
        
        return True
    
    async def get_global_model(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get the latest global model for client"""
        if client_id not in self.registered_clients:
            logger.error(f"Unknown client requesting global model: {client_id}")
            return None
        
        # Get latest model from storage
        model_data = await self.model_storage.get_latest_model()
        
        if model_data:
            return {
                'weights': model_data['weights'],
                'version': self.global_model_version,
                'round': self.current_round,
                'config': await self.get_training_configuration(client_id)
            }
        
        return None
    
    async def get_training_configuration(self, client_id: str) -> Dict[str, Any]:
        """Get training configuration for client"""
        client_info = self.registered_clients.get(client_id)
        if not client_info:
            raise ValueError(f"Unknown client: {client_id}")
        
        # Base configuration
        base_config = {
            'model_architecture': {
                'type': 'cnn',
                'layers': [64, 128, 256],
                'dropout': 0.5,
                'activation': 'relu'
            },
            'hyperparameters': {
                'learning_rate': self.config.federated_learning.learning_rate,
                'batch_size': self.config.federated_learning.batch_size,
                'local_epochs': self.config.federated_learning.local_epochs
            },
            'privacy_settings': {
                'enable_differential_privacy': self.config.privacy.enable_differential_privacy,
                'epsilon': self.config.privacy.epsilon,
                'delta': self.config.privacy.delta
            },
            'resource_constraints': {
                'max_memory_mb': 1024,
                'max_cpu_percent': 80,
                'max_training_time': 300
            },
            'aggregation_strategy': self.config.federated_learning.aggregation_strategy,
            'client_selection_criteria': {
                'min_reputation': 0.5,
                'max_clients_per_round': self.config.federated_learning.max_clients,
                'adaptive_selection': True,
                'quality_weight': 0.4,
                'reputation_weight': 0.2,
                'data_weight': 0.2,
                'network_weight': 0.2
            },
            'convergence_settings': {
                'threshold': self.convergence_threshold,
                'patience': self.convergence_patience,
                'early_stopping': True
            }
        }
        
        # Customize based on client capabilities
        if client_info.capabilities.get('gpu_available'):
            base_config['hyperparameters']['batch_size'] *= 2
            base_config['resource_constraints']['max_memory_mb'] *= 4
        
        if client_info.client_type == 'Mobile':
            base_config['hyperparameters']['local_epochs'] = min(3, base_config['hyperparameters']['local_epochs'])
            base_config['resource_constraints']['max_cpu_percent'] = 60
        
        return base_config
    
    async def report_client_metrics(self, client_id: str, metrics: Dict[str, Any]):
        """Report client metrics"""
        if client_id not in self.registered_clients:
            raise ValueError(f"Unknown client: {client_id}")
        
        self.client_metrics[client_id].update({
            'timestamp': datetime.utcnow(),
            **metrics
        })
        
        logger.debug(f"Updated metrics for client {client_id}")
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get server health status"""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        active_clients = len(self.token_manager.get_active_clients())
        
        return {
            'status': 'healthy' if self.is_running else 'unhealthy',
            'timestamp': datetime.utcnow(),
            'version': '1.0.0',
            'uptime': uptime,
            'active_clients': active_clients,
            'current_round': self.current_round,
            'last_aggregation': self.last_aggregation_time
        }
    
    async def get_server_status(self) -> Dict[str, Any]:
        """Get detailed server status"""
        uptime = (datetime.utcnow() - self.start_time).total_seconds()
        active_clients = len(self.token_manager.get_active_clients())
        
        return {
            'status': 'running' if self.is_running else 'stopped',
            'uptime': uptime,
            'total_clients': len(self.registered_clients),
            'active_clients': active_clients,
            'current_round': self.current_round,
            'total_rounds': len(self.round_metrics),
            'last_aggregation': self.last_aggregation_time,
            'aggregation_strategy': self.config.federated_learning.aggregation_strategy,
            'model_version': self.global_model_version,
            'performance_metrics': self._calculate_performance_metrics()
        }
    
    async def get_current_round(self) -> int:
        """Get current training round number"""
        return self.current_round
    
    async def get_available_strategies(self) -> List[str]:
        """Get available aggregation strategies"""
        return self.aggregator.get_algorithm_info()['available_algorithms']
    
    async def get_convergence_history(self) -> List[Dict[str, Any]]:
        """Get convergence history"""
        return self.convergence_history
    
    async def update_aggregation_strategy(self, strategy: str) -> bool:
        """Update aggregation strategy"""
        try:
            from .aggregation import AggregatorFactory
            
            # Validate strategy
            available_strategies = AggregatorFactory.get_available_algorithms()
            if strategy not in available_strategies:
                logger.error(f"Invalid aggregation strategy: {strategy}")
                return False
            
            # Create new aggregator
            new_aggregator = AggregatorFactory.create_aggregator(strategy, self.config)
            await new_aggregator.initialize()
            
            # Replace current aggregator
            await self.aggregator.shutdown()
            self.aggregator = new_aggregator
            
            # Update config
            self.config.federated_learning.aggregation_strategy = strategy
            
            logger.info(f"Aggregation strategy updated to: {strategy}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to update aggregation strategy: {e}")
            return False
    
    async def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        return self.privacy_security_manager.get_security_status()
    
    async def get_client_privacy_budget(self, client_id: str) -> Dict[str, Any]:
        """Get privacy budget status for specific client"""
        budget_status = self.privacy_security_manager.dp_mechanism.get_privacy_budget_status()
        
        return {
            'client_id': client_id,
            'client_budget_used': budget_status['client_budgets'].get(client_id, 0.0),
            'client_budget_remaining': 5.0 - budget_status['client_budgets'].get(client_id, 0.0),
            'global_budget_used': budget_status['global_budget_used'],
            'global_budget_remaining': budget_status['global_budget_remaining'],
            'epsilon': budget_status['epsilon'],
            'delta': budget_status['delta']
        }
    
    async def generate_compliance_report(self, days: int = 30) -> Dict[str, Any]:
        """Generate compliance report"""
        return self.privacy_security_manager.generate_compliance_report(days)
    
    async def block_client(self, client_id: str, reason: str) -> bool:
        """Block a client from participating"""
        try:
            if client_id not in self.registered_clients:
                logger.error(f"Cannot block unknown client: {client_id}")
                return False
            
            # Set reputation to 0 to effectively block the client
            self.registered_clients[client_id].reputation_score = 0.0
            
            # Revoke all tokens for this client
            revoked_count = self.token_manager.revoke_client_tokens(client_id)
            
            # Log the blocking event
            self.privacy_security_manager.audit_logger.log_event(
                'client_blocked',
                client_id,
                {'reason': reason, 'tokens_revoked': revoked_count}
            )
            
            logger.info(f"Client {client_id} blocked: {reason}")
            return True
        
        except Exception as e:
            logger.error(f"Failed to block client {client_id}: {e}")
            return False
    
    def _validate_model_update(self, model_update: ModelUpdate) -> bool:
        """Validate model update"""
        try:
            # Check required fields
            if not model_update.client_id:
                return False
            
            if not model_update.model_weights:
                return False
            
            # Validate weights can be deserialized
            try:
                weights = pickle.loads(model_update.model_weights)
                if not isinstance(weights, (dict, list, np.ndarray)):
                    return False
            except Exception:
                return False
            
            # Validate metrics
            if not isinstance(model_update.training_metrics, dict):
                return False
            
            # Check computation time is reasonable
            if model_update.computation_time < 0 or model_update.computation_time > 3600:
                return False
            
            return True
        
        except Exception as e:
            logger.error(f"Model update validation error: {e}")
            return False
    
    async def _check_aggregation_trigger(self):
        """Check if aggregation should be triggered"""
        current_updates = len(self.pending_updates[self.current_round])
        min_clients = self.config.federated_learning.min_clients
        
        if current_updates >= min_clients:
            logger.info(f"Triggering aggregation with {current_updates} updates")
            await self._perform_aggregation()
    
    async def _perform_aggregation(self):
        """Perform model aggregation with adaptive client selection"""
        if not self.pending_updates[self.current_round]:
            return
        
        logger.info(f"Starting aggregation for round {self.current_round}")
        
        try:
            # Get updates for current round
            all_updates = self.pending_updates[self.current_round]
            
            # Apply security and privacy protection
            protected_updates = self.privacy_security_manager.apply_privacy_protection(all_updates)
            
            # Detect and filter anomalies
            filtered_updates = self.privacy_security_manager.detect_and_filter_anomalies(protected_updates)
            
            # Apply adaptive client selection
            selected_updates = self._adaptive_client_selection(filtered_updates)
            
            logger.info(f"Security pipeline: {len(all_updates)} → {len(protected_updates)} → {len(filtered_updates)} → {len(selected_updates)} updates")
            logger.info(f"Selected {len(selected_updates)} out of {len(all_updates)} clients for aggregation")
            
            # Perform aggregation
            aggregated_weights = await self.aggregator.aggregate_models(selected_updates)
            
            # Store aggregated model
            model_data = {
                'weights': aggregated_weights,
                'version': self.global_model_version,
                'round': self.current_round,
                'timestamp': datetime.utcnow(),
                'participating_clients': [u.client_id for u in selected_updates]
            }
            
            await self.model_storage.store_model(model_data)
            
            # Calculate convergence metrics
            convergence_metrics = self._calculate_convergence_metrics(selected_updates)
            
            # Update metrics
            self.round_metrics[self.current_round] = {
                'participating_clients': len(selected_updates),
                'total_available_clients': len(all_updates),
                'aggregation_time': datetime.utcnow(),
                'convergence_metrics': convergence_metrics
            }
            
            # Update convergence history
            self.convergence_history.append(convergence_metrics)
            
            # Store selected clients history
            self.selected_clients_history.append([u.client_id for u in selected_updates])
            
            # Check for early stopping
            if self._should_early_stop():
                logger.info("Early stopping triggered due to convergence")
                # Could implement early stopping logic here
            
            # Move to next round
            self.current_round += 1
            self.last_aggregation_time = datetime.utcnow()
            
            logger.info(f"Aggregation completed for round {self.current_round - 1}")
        
        except Exception as e:
            logger.error(f"Aggregation failed: {e}")
    
    def _calculate_convergence_metrics(self, updates: List[ModelUpdate]) -> Dict[str, float]:
        """Calculate convergence metrics"""
        if not updates:
            return {}
        
        # Calculate average training metrics
        avg_loss = np.mean([u.training_metrics.get('loss', 0) for u in updates])
        avg_accuracy = np.mean([u.training_metrics.get('accuracy', 0) for u in updates])
        
        return {
            'average_loss': float(avg_loss),
            'average_accuracy': float(avg_accuracy),
            'participating_clients': len(updates),
            'total_data_samples': sum(u.data_statistics.get('num_samples', 0) for u in updates)
        }
    
    def _calculate_performance_metrics(self) -> Dict[str, Any]:
        """Calculate overall performance metrics"""
        if not self.round_metrics:
            return {}
        
        recent_rounds = list(self.round_metrics.keys())[-5:]  # Last 5 rounds
        
        avg_clients = np.mean([
            self.round_metrics[r]['participating_clients'] 
            for r in recent_rounds
        ])
        
        return {
            'average_clients_per_round': float(avg_clients),
            'total_rounds_completed': len(self.round_metrics),
            'last_round_clients': self.round_metrics.get(self.current_round - 1, {}).get('participating_clients', 0)
        }
    
    async def _cleanup_expired_tokens(self):
        """Background task to cleanup expired tokens"""
        while self.is_running:
            try:
                expired_count = self.token_manager.cleanup_expired_tokens()
                if expired_count > 0:
                    logger.info(f"Cleaned up {expired_count} expired tokens")
                
                await asyncio.sleep(300)  # Run every 5 minutes
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Token cleanup error: {e}")
                await asyncio.sleep(60)
    
    async def _monitor_training_progress(self):
        """Background task to monitor training progress"""
        while self.is_running:
            try:
                # Monitor client activity
                active_clients = self.token_manager.get_active_clients()
                logger.debug(f"Active clients: {len(active_clients)}")
                
                # Check for stalled training
                current_time = datetime.utcnow()
                stalled_threshold = timedelta(minutes=30)
                
                for client_id, metrics in self.client_metrics.items():
                    last_update = metrics.get('last_update')
                    if last_update and (current_time - last_update) > stalled_threshold:
                        logger.warning(f"Client {client_id} appears stalled")
                
                await asyncio.sleep(60)  # Run every minute
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Training progress monitoring error: {e}")
                await asyncio.sleep(60)
    
    async def _periodic_aggregation(self):
        """Background task for periodic aggregation"""
        while self.is_running:
            try:
                # Check if we should trigger aggregation based on time
                if self.last_aggregation_time:
                    time_since_last = datetime.utcnow() - self.last_aggregation_time
                    if time_since_last > timedelta(minutes=10):  # 10 minutes timeout
                        current_updates = len(self.pending_updates[self.current_round])
                        if current_updates > 0:
                            logger.info("Triggering periodic aggregation")
                            await self._perform_aggregation()
                
                await asyncio.sleep(60)  # Check every minute
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Periodic aggregation error: {e}")
                await asyncio.sleep(60)
    
    def _adaptive_client_selection(self, updates: List[ModelUpdate]) -> List[ModelUpdate]:
        """Select clients adaptively based on quality and diversity"""
        if len(updates) <= self.config.federated_learning.max_clients:
            return updates
        
        # Calculate client scores
        client_scores = []
        for update in updates:
            client_info = self.registered_clients.get(update.client_id)
            if not client_info:
                continue
            
            # Quality score based on training metrics
            accuracy = update.training_metrics.get('accuracy', 0.5)
            loss = update.training_metrics.get('loss', 1.0)
            quality_score = accuracy / (1.0 + loss)
            
            # Reputation score
            reputation_score = client_info.reputation_score
            
            # Data size score (normalized)
            data_size = update.data_statistics.get('num_samples', 1)
            data_score = min(data_size / 1000.0, 1.0)  # Normalize to [0, 1]
            
            # Network quality score
            latency = update.network_conditions.get('latency', 100)
            bandwidth = update.network_conditions.get('bandwidth', 10)
            network_score = min(bandwidth / 100.0, 1.0) * max(0, 1.0 - latency / 1000.0)
            
            # Combined score
            combined_score = (
                0.4 * quality_score +
                0.2 * reputation_score +
                0.2 * data_score +
                0.2 * network_score
            )
            
            client_scores.append((update, combined_score))
        
        # Sort by score (descending)
        client_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Select top clients
        max_clients = min(self.config.federated_learning.max_clients, len(client_scores))
        selected_updates = [score[0] for score in client_scores[:max_clients]]
        
        # Update client reputation based on selection
        for update, score in client_scores[:max_clients]:
            client_info = self.registered_clients.get(update.client_id)
            if client_info:
                # Slightly increase reputation for selected clients
                client_info.reputation_score = min(1.0, client_info.reputation_score + 0.01)
        
        return selected_updates
    
    def _should_early_stop(self) -> bool:
        """Check if training should stop early due to convergence"""
        if len(self.convergence_history) < self.convergence_patience + 1:
            return False
        
        # Check if loss improvement is below threshold for patience rounds
        recent_losses = [
            metrics.get('average_loss', float('inf'))
            for metrics in self.convergence_history[-self.convergence_patience-1:]
        ]
        
        improvements = []
        for i in range(1, len(recent_losses)):
            if recent_losses[i-1] > 0:
                improvement = (recent_losses[i-1] - recent_losses[i]) / recent_losses[i-1]
                improvements.append(improvement)
        
        # Check if all recent improvements are below threshold
        if improvements and all(imp < self.convergence_threshold for imp in improvements):
            return True
        
        return False
    
    async def _monitor_convergence(self):
        """Background task to monitor convergence"""
        while self.is_running:
            try:
                if len(self.convergence_history) >= 2:
                    current_metrics = self.convergence_history[-1]
                    previous_metrics = self.convergence_history[-2]
                    
                    current_loss = current_metrics.get('average_loss', float('inf'))
                    previous_loss = previous_metrics.get('average_loss', float('inf'))
                    
                    if previous_loss > 0:
                        improvement = (previous_loss - current_loss) / previous_loss
                        logger.info(f"Convergence monitoring: Loss improvement = {improvement:.4f}")
                        
                        if improvement < self.convergence_threshold:
                            logger.info("Slow convergence detected")
                
                await asyncio.sleep(30)  # Check every 30 seconds
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Convergence monitoring error: {e}")
                await asyncio.sleep(60)