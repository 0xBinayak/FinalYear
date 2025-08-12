"""
Edge Coordinator - Base functionality for local client management and hierarchical aggregation
"""
import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor
import hashlib
import pickle

from ..common.interfaces import BaseCoordinator, BaseClient, ClientInfo, ModelUpdate
from ..common.federated_data_structures import (
    EnhancedModelUpdate, AggregationResult, NetworkConditions, 
    ComputeResources, AggregationStrategy
)


class CoordinatorState(Enum):
    """Edge coordinator operational states"""
    INITIALIZING = "initializing"
    ONLINE = "online"
    OFFLINE = "offline"
    PARTITIONED = "partitioned"
    SYNCING = "syncing"
    ERROR = "error"


class ClientStatus(Enum):
    """Client status within edge coordinator"""
    REGISTERED = "registered"
    ACTIVE = "active"
    INACTIVE = "inactive"
    TRAINING = "training"
    FAILED = "failed"
    QUARANTINED = "quarantined"


@dataclass
class LocalClient:
    """Local client information and status"""
    client_info: ClientInfo
    status: ClientStatus = ClientStatus.REGISTERED
    last_seen: datetime = field(default_factory=datetime.now)
    last_update: Optional[datetime] = None
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    failure_count: int = 0
    reputation_score: float = 1.0
    current_model_version: Optional[str] = None
    
    def update_reputation(self, success: bool, quality_score: float = 1.0):
        """Update client reputation based on performance"""
        if success:
            self.reputation_score = min(1.0, self.reputation_score + 0.1 * quality_score)
            self.failure_count = max(0, self.failure_count - 1)
        else:
            self.reputation_score = max(0.0, self.reputation_score - 0.2)
            self.failure_count += 1
        
        # Quarantine clients with poor reputation
        if self.reputation_score < 0.3 or self.failure_count > 5:
            self.status = ClientStatus.QUARANTINED


@dataclass
class NetworkPartition:
    """Network partition information"""
    partition_id: str
    detected_at: datetime
    affected_clients: List[str]
    global_server_reachable: bool
    estimated_duration: Optional[timedelta] = None
    recovery_strategy: str = "wait_and_retry"


class EdgeCoordinator(BaseCoordinator):
    """
    Edge Coordinator for managing local client clusters with hierarchical aggregation
    and network partition handling
    """
    
    def __init__(self, coordinator_id: str, region: str, config: Dict[str, Any]):
        self.coordinator_id = coordinator_id
        self.region = region
        self.config = config
        
        # State management
        self.state = CoordinatorState.INITIALIZING
        self.local_clients: Dict[str, LocalClient] = {}
        self.active_partitions: Dict[str, NetworkPartition] = {}
        
        # Global server connection
        self.global_server_endpoint = config.get('global_server_endpoint')
        self.global_server_connected = False
        self.last_global_sync = None
        
        # Local aggregation
        self.local_model_cache: Dict[str, bytes] = {}
        self.pending_updates: List[EnhancedModelUpdate] = []
        self.aggregation_strategy = AggregationStrategy(config.get('aggregation_strategy', 'fedavg'))
        
        # Configuration
        self.max_local_clients = config.get('max_local_clients', 20)
        self.heartbeat_interval = config.get('heartbeat_interval', 30)  # seconds
        self.sync_interval = config.get('sync_interval', 300)  # seconds
        self.partition_timeout = config.get('partition_timeout', 600)  # seconds
        
        # Threading
        self.executor = ThreadPoolExecutor(max_workers=4)
        self.running = False
        self.background_tasks = []
        
        # Logging
        self.logger = logging.getLogger(f"EdgeCoordinator-{coordinator_id}")
        
        # Offline operation support
        self.offline_updates: List[EnhancedModelUpdate] = []
        self.offline_model_versions: Dict[str, bytes] = {}
        
        self.logger.info(f"Edge Coordinator {coordinator_id} initialized for region {region}")
    
    async def start(self):
        """Start the edge coordinator"""
        self.running = True
        self.state = CoordinatorState.ONLINE
        
        # Start background tasks
        self.background_tasks = [
            asyncio.create_task(self._heartbeat_loop()),
            asyncio.create_task(self._sync_loop()),
            asyncio.create_task(self._partition_detection_loop()),
            asyncio.create_task(self._cleanup_loop())
        ]
        
        self.logger.info(f"Edge Coordinator {self.coordinator_id} started")
    
    async def stop(self):
        """Stop the edge coordinator"""
        self.running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.executor.shutdown(wait=True)
        
        self.logger.info(f"Edge Coordinator {self.coordinator_id} stopped")
    
    def register_local_client(self, client_info: ClientInfo) -> str:
        """Register a new local client"""
        if len(self.local_clients) >= self.max_local_clients:
            raise ValueError("Maximum local clients reached")
        
        client_id = client_info.client_id
        
        if client_id in self.local_clients:
            # Update existing client
            self.local_clients[client_id].client_info = client_info
            self.local_clients[client_id].last_seen = datetime.now()
            self.local_clients[client_id].status = ClientStatus.ACTIVE
        else:
            # Register new client
            local_client = LocalClient(
                client_info=client_info,
                status=ClientStatus.REGISTERED
            )
            self.local_clients[client_id] = local_client
        
        self.logger.info(f"Client {client_id} registered with edge coordinator")
        return f"edge-token-{client_id}-{int(time.time())}"
    
    def unregister_local_client(self, client_id: str) -> bool:
        """Unregister a local client"""
        if client_id in self.local_clients:
            del self.local_clients[client_id]
            self.logger.info(f"Client {client_id} unregistered from edge coordinator")
            return True
        return False
    
    def receive_local_update(self, client_id: str, model_update: EnhancedModelUpdate) -> bool:
        """Receive model update from local client"""
        if client_id not in self.local_clients:
            self.logger.warning(f"Received update from unregistered client {client_id}")
            return False
        
        local_client = self.local_clients[client_id]
        
        # Validate update integrity
        if not model_update.verify_integrity():
            self.logger.warning(f"Invalid update from client {client_id}")
            local_client.update_reputation(False)
            return False
        
        # Update client status
        local_client.last_update = datetime.now()
        local_client.status = ClientStatus.ACTIVE
        local_client.current_model_version = model_update.model_version
        
        # Store update for local aggregation
        self.pending_updates.append(model_update)
        
        # Update reputation based on update quality
        quality_score = self._assess_update_quality(model_update)
        local_client.update_reputation(True, quality_score)
        
        self.logger.info(f"Received update from client {client_id}, quality: {quality_score:.2f}")
        return True
    
    def _assess_update_quality(self, update: EnhancedModelUpdate) -> float:
        """Assess the quality of a model update"""
        quality_score = 1.0
        
        # Check training metrics
        if update.training_loss is not None:
            # Lower loss is better (up to a point)
            if update.training_loss < 0.1:
                quality_score += 0.2
            elif update.training_loss > 2.0:
                quality_score -= 0.3
        
        # Check data diversity
        if update.data_distribution:
            # More diverse data is better
            diversity = len(update.data_distribution) / max(1, sum(update.data_distribution.values()))
            quality_score += diversity * 0.2
        
        # Check network conditions
        if update.network_conditions.connection_stability < 0.5:
            quality_score -= 0.2
        
        # Check computation resources
        if update.compute_resources.battery_level is not None:
            if update.compute_resources.battery_level < 0.2:
                quality_score -= 0.1
        
        return max(0.0, min(1.0, quality_score))
    
    async def aggregate_local_models(self) -> Optional[EnhancedModelUpdate]:
        """Aggregate pending local model updates"""
        if not self.pending_updates:
            return None
        
        start_time = time.time()
        
        try:
            # Filter out quarantined clients
            valid_updates = [
                update for update in self.pending_updates
                if (update.client_id in self.local_clients and 
                    self.local_clients[update.client_id].status != ClientStatus.QUARANTINED)
            ]
            
            if not valid_updates:
                self.logger.warning("No valid updates for local aggregation")
                self.pending_updates.clear()
                return None
            
            # Perform aggregation based on strategy
            if self.aggregation_strategy == AggregationStrategy.FEDAVG:
                aggregated_update = await self._fedavg_aggregation(valid_updates)
            elif self.aggregation_strategy == AggregationStrategy.BYZANTINE_ROBUST:
                aggregated_update = await self._byzantine_robust_aggregation(valid_updates)
            else:
                # Default to FedAvg
                aggregated_update = await self._fedavg_aggregation(valid_updates)
            
            # Clear pending updates
            self.pending_updates.clear()
            
            aggregation_time = time.time() - start_time
            self.logger.info(f"Local aggregation completed in {aggregation_time:.2f}s with {len(valid_updates)} updates")
            
            return aggregated_update
            
        except Exception as e:
            self.logger.error(f"Local aggregation failed: {e}")
            return None
    
    async def _fedavg_aggregation(self, updates: List[EnhancedModelUpdate]) -> EnhancedModelUpdate:
        """Perform FedAvg aggregation"""
        if not updates:
            raise ValueError("No updates to aggregate")
        
        # Calculate weights based on number of samples
        total_samples = sum(update.samples_used for update in updates)
        weights = [update.samples_used / total_samples for update in updates]
        
        # Aggregate model weights
        first_update = updates[0]
        aggregated_weights = self._weighted_average_weights(
            [update.model_weights for update in updates], weights
        )
        
        # Aggregate metrics
        avg_training_loss = sum(update.training_loss * w for update, w in zip(updates, weights))
        avg_training_time = sum(update.training_time_seconds * w for update, w in zip(updates, weights))
        
        # Create aggregated update
        aggregated_update = EnhancedModelUpdate(
            client_id=f"edge-{self.coordinator_id}",
            model_weights=aggregated_weights,
            model_size_bytes=len(aggregated_weights),
            training_rounds=max(update.training_rounds for update in updates),
            local_epochs=int(sum(update.local_epochs * w for update, w in zip(updates, weights))),
            batch_size=int(sum(update.batch_size * w for update, w in zip(updates, weights))),
            learning_rate=sum(update.learning_rate * w for update, w in zip(updates, weights)),
            samples_used=total_samples,
            training_time_seconds=avg_training_time,
            training_loss=avg_training_loss,
            validation_loss=None,  # Will be computed later if needed
            training_accuracy=None,  # Will be computed later if needed
            validation_accuracy=None,
            convergence_metric=None,
            data_distribution=self._merge_data_distributions([update.data_distribution for update in updates]),
            network_conditions=self._average_network_conditions([update.network_conditions for update in updates]),
            compute_resources=self._average_compute_resources([update.compute_resources for update in updates])
        )
        
        return aggregated_update
    
    async def _byzantine_robust_aggregation(self, updates: List[EnhancedModelUpdate]) -> EnhancedModelUpdate:
        """Perform Byzantine-robust aggregation using coordinate-wise median"""
        if len(updates) < 3:
            # Fall back to FedAvg for small number of updates
            return await self._fedavg_aggregation(updates)
        
        # Convert weights to numpy arrays for easier manipulation
        import numpy as np
        
        weight_arrays = []
        for update in updates:
            weights = np.frombuffer(update.model_weights, dtype=np.float32)
            weight_arrays.append(weights)
        
        # Stack weights and compute coordinate-wise median
        stacked_weights = np.stack(weight_arrays, axis=0)
        median_weights = np.median(stacked_weights, axis=0)
        
        # Convert back to bytes
        aggregated_weights = median_weights.astype(np.float32).tobytes()
        
        # Use median client's metadata as base
        median_idx = len(updates) // 2
        base_update = updates[median_idx]
        
        aggregated_update = EnhancedModelUpdate(
            client_id=f"edge-{self.coordinator_id}",
            model_weights=aggregated_weights,
            model_size_bytes=len(aggregated_weights),
            training_rounds=base_update.training_rounds,
            local_epochs=base_update.local_epochs,
            batch_size=base_update.batch_size,
            learning_rate=base_update.learning_rate,
            samples_used=sum(update.samples_used for update in updates),
            training_time_seconds=base_update.training_time_seconds,
            training_loss=base_update.training_loss,
            data_distribution=self._merge_data_distributions([update.data_distribution for update in updates]),
            network_conditions=base_update.network_conditions,
            compute_resources=base_update.compute_resources
        )
        
        return aggregated_update
    
    def _weighted_average_weights(self, weight_lists: List[bytes], weights: List[float]) -> bytes:
        """Compute weighted average of model weights"""
        import numpy as np
        
        # Convert bytes to numpy arrays
        weight_arrays = [np.frombuffer(weights_bytes, dtype=np.float32) for weights_bytes in weight_lists]
        
        # Compute weighted average
        weighted_sum = np.zeros_like(weight_arrays[0])
        for weight_array, weight in zip(weight_arrays, weights):
            weighted_sum += weight_array * weight
        
        return weighted_sum.astype(np.float32).tobytes()
    
    def _merge_data_distributions(self, distributions: List[Dict[str, int]]) -> Dict[str, int]:
        """Merge data distributions from multiple clients"""
        merged = {}
        for dist in distributions:
            if dist:
                for key, value in dist.items():
                    merged[key] = merged.get(key, 0) + value
        return merged
    
    def _average_network_conditions(self, conditions_list: List[NetworkConditions]) -> NetworkConditions:
        """Average network conditions from multiple clients"""
        if not conditions_list:
            return NetworkConditions(10.0, 100.0, 0.01, 10.0, 0.9)
        
        avg_bandwidth = sum(c.bandwidth_mbps for c in conditions_list) / len(conditions_list)
        avg_latency = sum(c.latency_ms for c in conditions_list) / len(conditions_list)
        avg_packet_loss = sum(c.packet_loss_rate for c in conditions_list) / len(conditions_list)
        avg_jitter = sum(c.jitter_ms for c in conditions_list) / len(conditions_list)
        avg_stability = sum(c.connection_stability for c in conditions_list) / len(conditions_list)
        
        return NetworkConditions(
            bandwidth_mbps=avg_bandwidth,
            latency_ms=avg_latency,
            packet_loss_rate=avg_packet_loss,
            jitter_ms=avg_jitter,
            connection_stability=avg_stability,
            is_mobile=any(c.is_mobile for c in conditions_list),
            is_metered=any(c.is_metered for c in conditions_list)
        )
    
    def _average_compute_resources(self, resources_list: List[ComputeResources]) -> ComputeResources:
        """Average compute resources from multiple clients"""
        if not resources_list:
            return ComputeResources(4, 2.0, 8.0, 6.0)
        
        avg_cores = int(sum(r.cpu_cores for r in resources_list) / len(resources_list))
        avg_freq = sum(r.cpu_frequency_ghz for r in resources_list) / len(resources_list)
        avg_memory = sum(r.memory_gb for r in resources_list) / len(resources_list)
        avg_available = sum(r.available_memory_gb for r in resources_list) / len(resources_list)
        
        return ComputeResources(
            cpu_cores=avg_cores,
            cpu_frequency_ghz=avg_freq,
            memory_gb=avg_memory,
            available_memory_gb=avg_available,
            gpu_available=any(r.gpu_available for r in resources_list),
            battery_level=None  # Don't average battery levels
        )
    
    async def _heartbeat_loop(self):
        """Background task for client heartbeat monitoring"""
        while self.running:
            try:
                current_time = datetime.now()
                timeout_threshold = current_time - timedelta(seconds=self.heartbeat_interval * 3)
                
                # Check for inactive clients
                for client_id, local_client in list(self.local_clients.items()):
                    if local_client.last_seen < timeout_threshold:
                        if local_client.status == ClientStatus.ACTIVE:
                            local_client.status = ClientStatus.INACTIVE
                            self.logger.warning(f"Client {client_id} marked as inactive")
                        
                        # Remove very old inactive clients
                        if local_client.last_seen < current_time - timedelta(hours=1):
                            del self.local_clients[client_id]
                            self.logger.info(f"Removed inactive client {client_id}")
                
                await asyncio.sleep(self.heartbeat_interval)
                
            except Exception as e:
                self.logger.error(f"Heartbeat loop error: {e}")
                await asyncio.sleep(self.heartbeat_interval)
    
    async def _sync_loop(self):
        """Background task for synchronizing with global server"""
        while self.running:
            try:
                await self._attempt_global_sync()
                await asyncio.sleep(self.sync_interval)
                
            except Exception as e:
                self.logger.error(f"Sync loop error: {e}")
                await asyncio.sleep(self.sync_interval)
    
    async def _partition_detection_loop(self):
        """Background task for network partition detection"""
        while self.running:
            try:
                await self._detect_network_partitions()
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Partition detection error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_loop(self):
        """Background task for cleanup operations"""
        while self.running:
            try:
                # Clean up old model cache entries
                current_time = datetime.now()
                cache_timeout = timedelta(hours=2)
                
                # This would clean up old cached models (implementation depends on storage)
                
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(3600)
    
    async def _attempt_global_sync(self):
        """Attempt to synchronize with global aggregation server"""
        try:
            # Check if we have local aggregated updates to send
            local_aggregate = await self.aggregate_local_models()
            
            if local_aggregate and self.global_server_endpoint:
                # This would send the update to global server
                # Implementation depends on communication protocol
                success = await self._send_to_global_server(local_aggregate)
                
                if success:
                    self.global_server_connected = True
                    self.last_global_sync = datetime.now()
                    self.state = CoordinatorState.ONLINE
                    self.logger.info("Successfully synced with global server")
                else:
                    self._handle_global_sync_failure()
            
        except Exception as e:
            self.logger.error(f"Global sync failed: {e}")
            self._handle_global_sync_failure()
    
    async def _send_to_global_server(self, update: EnhancedModelUpdate) -> bool:
        """Send update to global aggregation server"""
        # This is a placeholder - actual implementation would use HTTP/gRPC
        # For now, just simulate success/failure
        import random
        return random.random() > 0.1  # 90% success rate
    
    def _handle_global_sync_failure(self):
        """Handle failure to sync with global server"""
        self.global_server_connected = False
        
        if self.state == CoordinatorState.ONLINE:
            self.state = CoordinatorState.PARTITIONED
            self.logger.warning("Lost connection to global server, entering partitioned mode")
        
        # Store updates for later sync
        if self.pending_updates:
            self.offline_updates.extend(self.pending_updates)
            self.pending_updates.clear()
    
    async def _detect_network_partitions(self):
        """Detect network partitions among local clients"""
        # Simple partition detection based on client connectivity patterns
        current_time = datetime.now()
        
        # Group clients by their last seen time
        recent_clients = []
        old_clients = []
        
        for client_id, local_client in self.local_clients.items():
            if current_time - local_client.last_seen < timedelta(minutes=5):
                recent_clients.append(client_id)
            else:
                old_clients.append(client_id)
        
        # If we have a significant split, it might indicate a partition
        if len(old_clients) > len(recent_clients) * 0.3:  # More than 30% are old
            partition_id = f"partition-{int(time.time())}"
            partition = NetworkPartition(
                partition_id=partition_id,
                detected_at=current_time,
                affected_clients=old_clients,
                global_server_reachable=self.global_server_connected
            )
            
            self.active_partitions[partition_id] = partition
            self.logger.warning(f"Detected potential network partition: {partition_id}")
    
    # Implementation of BaseCoordinator abstract methods
    def manage_local_clients(self, clients: List[BaseClient]) -> bool:
        """Manage local client cluster"""
        try:
            for client in clients:
                client_info = client.get_client_info()
                self.register_local_client(client_info)
            return True
        except Exception as e:
            self.logger.error(f"Failed to manage local clients: {e}")
            return False
    
    def aggregate_local_models(self, updates: List[ModelUpdate]) -> ModelUpdate:
        """Aggregate local model updates (legacy interface)"""
        # Convert to enhanced updates and use async aggregation
        enhanced_updates = []
        for update in updates:
            enhanced_update = EnhancedModelUpdate(
                client_id=update.client_id,
                model_weights=update.model_weights,
                model_size_bytes=len(update.model_weights),
                training_rounds=1,
                local_epochs=1,
                batch_size=32,
                learning_rate=0.01,
                samples_used=100,  # Default values
                training_time_seconds=update.computation_time,
                training_loss=update.training_metrics.get('training_loss', 1.0),
                data_distribution=update.data_statistics.get('data_distribution', {}),
                network_conditions=NetworkConditions(**update.network_conditions)
            )
            enhanced_updates.append(enhanced_update)
        
        # Run async aggregation in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result = loop.run_until_complete(self._fedavg_aggregation(enhanced_updates))
            return result.to_legacy_format()
        finally:
            loop.close()
    
    def sync_with_global_server(self, server_endpoint: str) -> bool:
        """Synchronize with global aggregation server (legacy interface)"""
        self.global_server_endpoint = server_endpoint
        
        # Run async sync in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._attempt_global_sync())
            return self.global_server_connected
        finally:
            loop.close()
    
    def get_coordinator_status(self) -> Dict[str, Any]:
        """Get comprehensive coordinator status"""
        return {
            'coordinator_id': self.coordinator_id,
            'region': self.region,
            'state': self.state.value,
            'local_clients_count': len(self.local_clients),
            'active_clients': len([c for c in self.local_clients.values() if c.status == ClientStatus.ACTIVE]),
            'pending_updates': len(self.pending_updates),
            'offline_updates': len(self.offline_updates),
            'global_server_connected': self.global_server_connected,
            'last_global_sync': self.last_global_sync.isoformat() if self.last_global_sync else None,
            'active_partitions': len(self.active_partitions),
            'uptime_seconds': time.time() - (self.last_global_sync.timestamp() if self.last_global_sync else time.time())
        }