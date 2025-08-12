"""
Offline Operation and Eventual Consistency for Edge Coordinator
"""
import asyncio
import logging
import json
import pickle
import gzip
import hashlib
import time
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
from pathlib import Path

from ..common.federated_data_structures import (
    EnhancedModelUpdate, AggregationResult, AggregationStrategy
)


class OfflineMode(Enum):
    """Offline operation modes"""
    NORMAL = "normal"           # Connected to global server
    DEGRADED = "degraded"       # Partial connectivity
    ISOLATED = "isolated"       # No global connectivity
    RECOVERY = "recovery"       # Recovering from partition


class SyncStrategy(Enum):
    """Synchronization strategies"""
    IMMEDIATE = "immediate"     # Sync as soon as connection is restored
    BATCHED = "batched"        # Batch updates before syncing
    COMPRESSED = "compressed"   # Compress updates before syncing
    SELECTIVE = "selective"     # Only sync most important updates


@dataclass
class OfflineUpdate:
    """Update stored during offline operation"""
    update_id: str
    update: EnhancedModelUpdate
    created_at: datetime
    priority: float = 1.0
    attempts: int = 0
    last_attempt: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_age_hours(self) -> float:
        """Get age of update in hours"""
        return (datetime.now() - self.created_at).total_seconds() / 3600
    
    def should_retry(self, max_attempts: int = 3, backoff_hours: float = 1.0) -> bool:
        """Check if update should be retried"""
        if self.attempts >= max_attempts:
            return False
        
        if self.last_attempt is None:
            return True
        
        time_since_attempt = datetime.now() - self.last_attempt
        required_backoff = timedelta(hours=backoff_hours * (2 ** self.attempts))
        
        return time_since_attempt >= required_backoff


@dataclass
class ModelVersion:
    """Model version information"""
    version_id: str
    model_weights: bytes
    metadata: Dict[str, Any]
    created_at: datetime
    parent_version: Optional[str] = None
    is_global: bool = False
    is_local: bool = False
    
    def get_checksum(self) -> str:
        """Get model checksum"""
        hasher = hashlib.sha256()
        hasher.update(self.model_weights)
        hasher.update(self.version_id.encode())
        return hasher.hexdigest()


class OfflineOperationManager:
    """
    Manages offline operation and eventual consistency for edge coordinator
    """
    
    def __init__(self, coordinator_id: str, storage_path: str, config: Dict[str, Any]):
        self.coordinator_id = coordinator_id
        self.storage_path = Path(storage_path)
        self.config = config
        self.logger = logging.getLogger(f"OfflineManager-{coordinator_id}")
        
        # Ensure storage directory exists
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Operation state
        self.mode = OfflineMode.NORMAL
        self.offline_since: Optional[datetime] = None
        self.last_global_sync: Optional[datetime] = None
        
        # Offline storage
        self.offline_updates: Dict[str, OfflineUpdate] = {}
        self.model_versions: Dict[str, ModelVersion] = {}
        self.pending_sync_queue: List[str] = []  # Update IDs to sync
        
        # Configuration
        self.max_offline_updates = config.get('max_offline_updates', 1000)
        self.max_offline_hours = config.get('max_offline_hours', 24)
        self.sync_strategy = SyncStrategy(config.get('sync_strategy', 'batched'))
        self.compression_enabled = config.get('compression_enabled', True)
        
        # Background tasks
        self.running = False
        self.background_tasks: List[asyncio.Task] = []
        
        # Load persisted state
        self._load_offline_state()
    
    async def start(self):
        """Start offline operation manager"""
        self.running = True
        
        self.background_tasks = [
            asyncio.create_task(self._cleanup_loop()),
            asyncio.create_task(self._persistence_loop()),
            asyncio.create_task(self._sync_retry_loop())
        ]
        
        self.logger.info("Offline operation manager started")
    
    async def stop(self):
        """Stop offline operation manager"""
        self.running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Persist final state
        self._persist_offline_state()
        
        self.logger.info("Offline operation manager stopped")
    
    def enter_offline_mode(self, reason: str = "network_partition"):
        """Enter offline operation mode"""
        if self.mode == OfflineMode.NORMAL:
            self.mode = OfflineMode.ISOLATED
            self.offline_since = datetime.now()
            
            self.logger.warning(f"Entering offline mode: {reason}")
            
            # Persist state immediately
            self._persist_offline_state()
    
    def exit_offline_mode(self):
        """Exit offline operation mode"""
        if self.mode in [OfflineMode.ISOLATED, OfflineMode.DEGRADED]:
            self.mode = OfflineMode.RECOVERY
            
            offline_duration = None
            if self.offline_since:
                offline_duration = datetime.now() - self.offline_since
                self.logger.info(f"Exiting offline mode after {offline_duration}")
            
            # Trigger synchronization
            asyncio.create_task(self._initiate_recovery_sync())
    
    def store_offline_update(self, update: EnhancedModelUpdate, priority: float = 1.0) -> str:
        """Store update during offline operation"""
        update_id = f"offline-{self.coordinator_id}-{int(time.time())}-{len(self.offline_updates)}"
        
        offline_update = OfflineUpdate(
            update_id=update_id,
            update=update,
            created_at=datetime.now(),
            priority=priority,
            metadata={
                'coordinator_id': self.coordinator_id,
                'offline_mode': self.mode.value,
                'client_count': 1  # Will be updated for aggregated updates
            }
        )
        
        # Check storage limits
        if len(self.offline_updates) >= self.max_offline_updates:
            self._cleanup_old_updates()
        
        self.offline_updates[update_id] = offline_update
        self.pending_sync_queue.append(update_id)
        
        self.logger.info(f"Stored offline update {update_id} (priority: {priority})")
        return update_id
    
    def store_model_version(self, version_id: str, model_weights: bytes, 
                           metadata: Dict[str, Any], is_global: bool = False) -> bool:
        """Store a model version"""
        try:
            model_version = ModelVersion(
                version_id=version_id,
                model_weights=model_weights,
                metadata=metadata,
                created_at=datetime.now(),
                is_global=is_global,
                is_local=not is_global
            )
            
            self.model_versions[version_id] = model_version
            
            # Persist to disk
            self._persist_model_version(model_version)
            
            self.logger.info(f"Stored model version {version_id} ({'global' if is_global else 'local'})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to store model version {version_id}: {e}")
            return False
    
    def get_latest_model_version(self, prefer_global: bool = True) -> Optional[ModelVersion]:
        """Get the latest model version"""
        if not self.model_versions:
            return None
        
        # Sort by creation time
        sorted_versions = sorted(
            self.model_versions.values(),
            key=lambda v: v.created_at,
            reverse=True
        )
        
        if prefer_global:
            # Try to find latest global version first
            global_versions = [v for v in sorted_versions if v.is_global]
            if global_versions:
                return global_versions[0]
        
        # Return latest version (global or local)
        return sorted_versions[0]
    
    async def sync_with_global_server(self, server_endpoint: str) -> Dict[str, Any]:
        """Synchronize offline updates with global server"""
        if not self.pending_sync_queue:
            return {'status': 'no_updates', 'synced_count': 0}
        
        sync_start = time.time()
        synced_count = 0
        failed_count = 0
        
        try:
            # Prepare updates for sync based on strategy
            updates_to_sync = self._prepare_updates_for_sync()
            
            for update_batch in updates_to_sync:
                try:
                    success = await self._sync_update_batch(server_endpoint, update_batch)
                    
                    if success:
                        synced_count += len(update_batch)
                        # Remove synced updates from pending queue
                        for update_id in update_batch:
                            if update_id in self.pending_sync_queue:
                                self.pending_sync_queue.remove(update_id)
                            # Mark as synced but keep for a while
                            if update_id in self.offline_updates:
                                self.offline_updates[update_id].metadata['synced'] = True
                                self.offline_updates[update_id].metadata['synced_at'] = datetime.now().isoformat()
                    else:
                        failed_count += len(update_batch)
                        # Update retry information
                        for update_id in update_batch:
                            if update_id in self.offline_updates:
                                offline_update = self.offline_updates[update_id]
                                offline_update.attempts += 1
                                offline_update.last_attempt = datetime.now()
                
                except Exception as e:
                    self.logger.error(f"Failed to sync batch: {e}")
                    failed_count += len(update_batch)
            
            sync_duration = time.time() - sync_start
            
            # Update sync status
            if synced_count > 0:
                self.last_global_sync = datetime.now()
                
                if self.mode == OfflineMode.RECOVERY:
                    self.mode = OfflineMode.NORMAL
                    self.offline_since = None
            
            result = {
                'status': 'completed',
                'synced_count': synced_count,
                'failed_count': failed_count,
                'sync_duration_seconds': sync_duration,
                'remaining_updates': len(self.pending_sync_queue)
            }
            
            self.logger.info(f"Sync completed: {synced_count} synced, {failed_count} failed")
            return result
            
        except Exception as e:
            self.logger.error(f"Sync with global server failed: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _prepare_updates_for_sync(self) -> List[List[str]]:
        """Prepare updates for synchronization based on strategy"""
        if not self.pending_sync_queue:
            return []
        
        # Get updates that should be retried
        retry_updates = [
            update_id for update_id in self.pending_sync_queue
            if (update_id in self.offline_updates and 
                self.offline_updates[update_id].should_retry())
        ]
        
        if self.sync_strategy == SyncStrategy.IMMEDIATE:
            # Sync one at a time
            return [[update_id] for update_id in retry_updates[:10]]  # Limit to 10
        
        elif self.sync_strategy == SyncStrategy.BATCHED:
            # Group into batches of 5
            batch_size = 5
            batches = []
            for i in range(0, len(retry_updates), batch_size):
                batch = retry_updates[i:i + batch_size]
                batches.append(batch)
            return batches
        
        elif self.sync_strategy == SyncStrategy.SELECTIVE:
            # Sort by priority and age, sync top 10
            prioritized_updates = sorted(
                retry_updates,
                key=lambda uid: (
                    self.offline_updates[uid].priority,
                    -self.offline_updates[uid].get_age_hours()  # Negative for descending
                ),
                reverse=True
            )
            return [prioritized_updates[:10]]
        
        else:
            # Default batched approach
            return [retry_updates[:10]]
    
    async def _sync_update_batch(self, server_endpoint: str, update_ids: List[str]) -> bool:
        """Sync a batch of updates to global server"""
        try:
            # Prepare batch data
            batch_updates = []
            for update_id in update_ids:
                if update_id in self.offline_updates:
                    offline_update = self.offline_updates[update_id]
                    
                    # Apply compression if enabled
                    update_data = offline_update.update
                    if self.compression_enabled:
                        update_data = self._compress_update(update_data)
                    
                    batch_updates.append({
                        'update_id': update_id,
                        'update': update_data,
                        'metadata': offline_update.metadata
                    })
            
            if not batch_updates:
                return True
            
            # This would send to actual global server
            # For now, simulate success/failure
            import random
            success = random.random() > 0.2  # 80% success rate
            
            if success:
                self.logger.info(f"Successfully synced batch of {len(batch_updates)} updates")
            else:
                self.logger.warning(f"Failed to sync batch of {len(batch_updates)} updates")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Batch sync failed: {e}")
            return False
    
    def _compress_update(self, update: EnhancedModelUpdate) -> bytes:
        """Compress update for efficient transmission"""
        try:
            # Serialize and compress
            serialized = pickle.dumps(update)
            compressed = gzip.compress(serialized)
            
            compression_ratio = len(compressed) / len(serialized)
            self.logger.debug(f"Compressed update: {compression_ratio:.2f} ratio")
            
            return compressed
            
        except Exception as e:
            self.logger.error(f"Update compression failed: {e}")
            return pickle.dumps(update)
    
    def _cleanup_old_updates(self):
        """Clean up old offline updates"""
        current_time = datetime.now()
        max_age = timedelta(hours=self.max_offline_hours)
        
        # Find updates to remove
        updates_to_remove = []
        for update_id, offline_update in self.offline_updates.items():
            age = current_time - offline_update.created_at
            
            # Remove if too old or already synced and old enough
            if (age > max_age or 
                (offline_update.metadata.get('synced') and age > timedelta(hours=2))):
                updates_to_remove.append(update_id)
        
        # Remove old updates
        for update_id in updates_to_remove:
            del self.offline_updates[update_id]
            if update_id in self.pending_sync_queue:
                self.pending_sync_queue.remove(update_id)
        
        if updates_to_remove:
            self.logger.info(f"Cleaned up {len(updates_to_remove)} old updates")
    
    def _persist_offline_state(self):
        """Persist offline state to disk"""
        try:
            state_file = self.storage_path / "offline_state.json"
            
            state_data = {
                'coordinator_id': self.coordinator_id,
                'mode': self.mode.value,
                'offline_since': self.offline_since.isoformat() if self.offline_since else None,
                'last_global_sync': self.last_global_sync.isoformat() if self.last_global_sync else None,
                'pending_sync_queue': self.pending_sync_queue,
                'update_count': len(self.offline_updates),
                'model_version_count': len(self.model_versions)
            }
            
            with open(state_file, 'w') as f:
                json.dump(state_data, f, indent=2)
            
            # Persist updates separately (they're large)
            updates_file = self.storage_path / "offline_updates.pkl"
            with open(updates_file, 'wb') as f:
                pickle.dump(self.offline_updates, f)
            
        except Exception as e:
            self.logger.error(f"Failed to persist offline state: {e}")
    
    def _load_offline_state(self):
        """Load offline state from disk"""
        try:
            state_file = self.storage_path / "offline_state.json"
            updates_file = self.storage_path / "offline_updates.pkl"
            
            if state_file.exists():
                with open(state_file, 'r') as f:
                    state_data = json.load(f)
                
                self.mode = OfflineMode(state_data.get('mode', 'normal'))
                
                if state_data.get('offline_since'):
                    self.offline_since = datetime.fromisoformat(state_data['offline_since'])
                
                if state_data.get('last_global_sync'):
                    self.last_global_sync = datetime.fromisoformat(state_data['last_global_sync'])
                
                self.pending_sync_queue = state_data.get('pending_sync_queue', [])
            
            if updates_file.exists():
                with open(updates_file, 'rb') as f:
                    self.offline_updates = pickle.load(f)
            
            # Load model versions
            self._load_model_versions()
            
            self.logger.info(f"Loaded offline state: {len(self.offline_updates)} updates, {len(self.model_versions)} model versions")
            
        except Exception as e:
            self.logger.error(f"Failed to load offline state: {e}")
    
    def _persist_model_version(self, model_version: ModelVersion):
        """Persist a model version to disk"""
        try:
            version_dir = self.storage_path / "model_versions"
            version_dir.mkdir(exist_ok=True)
            
            version_file = version_dir / f"{model_version.version_id}.pkl"
            with open(version_file, 'wb') as f:
                pickle.dump(model_version, f)
            
        except Exception as e:
            self.logger.error(f"Failed to persist model version {model_version.version_id}: {e}")
    
    def _load_model_versions(self):
        """Load model versions from disk"""
        try:
            version_dir = self.storage_path / "model_versions"
            if not version_dir.exists():
                return
            
            for version_file in version_dir.glob("*.pkl"):
                try:
                    with open(version_file, 'rb') as f:
                        model_version = pickle.load(f)
                    
                    self.model_versions[model_version.version_id] = model_version
                    
                except Exception as e:
                    self.logger.error(f"Failed to load model version from {version_file}: {e}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model versions: {e}")
    
    async def _cleanup_loop(self):
        """Background cleanup task"""
        while self.running:
            try:
                self._cleanup_old_updates()
                await asyncio.sleep(3600)  # Run every hour
                
            except Exception as e:
                self.logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(3600)
    
    async def _persistence_loop(self):
        """Background persistence task"""
        while self.running:
            try:
                self._persist_offline_state()
                await asyncio.sleep(300)  # Persist every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Persistence loop error: {e}")
                await asyncio.sleep(300)
    
    async def _sync_retry_loop(self):
        """Background sync retry task"""
        while self.running:
            try:
                if self.mode == OfflineMode.RECOVERY and self.pending_sync_queue:
                    # Attempt to sync pending updates
                    await self._initiate_recovery_sync()
                
                await asyncio.sleep(600)  # Try every 10 minutes
                
            except Exception as e:
                self.logger.error(f"Sync retry loop error: {e}")
                await asyncio.sleep(600)
    
    async def _initiate_recovery_sync(self):
        """Initiate recovery synchronization"""
        if not self.pending_sync_queue:
            return
        
        # This would use the actual global server endpoint
        # For now, simulate the sync process
        server_endpoint = "http://global-server:8000"
        
        result = await self.sync_with_global_server(server_endpoint)
        
        if result['status'] == 'completed' and result['synced_count'] > 0:
            self.logger.info(f"Recovery sync successful: {result['synced_count']} updates synced")
        else:
            self.logger.warning(f"Recovery sync incomplete: {result}")
    
    def get_offline_status(self) -> Dict[str, Any]:
        """Get comprehensive offline operation status"""
        offline_duration = None
        if self.offline_since:
            offline_duration = (datetime.now() - self.offline_since).total_seconds()
        
        return {
            'mode': self.mode.value,
            'offline_since': self.offline_since.isoformat() if self.offline_since else None,
            'offline_duration_seconds': offline_duration,
            'last_global_sync': self.last_global_sync.isoformat() if self.last_global_sync else None,
            'offline_updates_count': len(self.offline_updates),
            'pending_sync_count': len(self.pending_sync_queue),
            'model_versions_count': len(self.model_versions),
            'storage_path': str(self.storage_path),
            'sync_strategy': self.sync_strategy.value,
            'compression_enabled': self.compression_enabled
        }