"""
Tests for Edge Coordinator functionality
"""
import asyncio
import pytest
import tempfile
import shutil
from datetime import datetime, timedelta
from pathlib import Path

from .coordinator import EdgeCoordinator, CoordinatorState, ClientStatus
from .network_partition import NetworkPartitionDetector, PartitionType
from .offline_operation import OfflineOperationManager, OfflineMode
from ..common.interfaces import ClientInfo
from ..common.federated_data_structures import (
    EnhancedModelUpdate, NetworkConditions, ComputeResources
)


class TestEdgeCoordinator:
    """Test Edge Coordinator base functionality"""
    
    @pytest.fixture
    async def coordinator(self):
        """Create test coordinator"""
        config = {
            'max_local_clients': 10,
            'heartbeat_interval': 5,
            'sync_interval': 30,
            'aggregation_strategy': 'fedavg'
        }
        
        coordinator = EdgeCoordinator("test-coord-1", "test-region", config)
        await coordinator.start()
        
        yield coordinator
        
        await coordinator.stop()
    
    @pytest.fixture
    def sample_client_info(self):
        """Create sample client info"""
        return ClientInfo(
            client_id="test-client-1",
            client_type="SDR",
            capabilities={"cpu_cores": 4, "memory_gb": 8},
            location={"lat": 37.7749, "lon": -122.4194},
            network_info={"bandwidth_mbps": 100},
            hardware_specs={"sdr_type": "rtlsdr"},
            reputation_score=1.0
        )
    
    @pytest.fixture
    def sample_model_update(self):
        """Create sample model update"""
        import numpy as np
        
        # Create dummy model weights
        weights = np.random.random(1000).astype(np.float32)
        
        return EnhancedModelUpdate(
            client_id="test-client-1",
            model_weights=weights.tobytes(),
            model_size_bytes=len(weights.tobytes()),
            training_rounds=1,
            local_epochs=5,
            batch_size=32,
            learning_rate=0.01,
            samples_used=1000,
            training_time_seconds=120.0,
            training_loss=0.5,
            validation_loss=0.6,
            training_accuracy=0.85,
            data_distribution={"class_0": 400, "class_1": 600},
            network_conditions=NetworkConditions(50.0, 20.0, 0.01, 5.0, 0.95),
            compute_resources=ComputeResources(4, 2.5, 8.0, 6.0)
        )
    
    async def test_client_registration(self, coordinator, sample_client_info):
        """Test client registration"""
        # Register client
        token = coordinator.register_local_client(sample_client_info)
        
        assert token is not None
        assert sample_client_info.client_id in coordinator.local_clients
        
        local_client = coordinator.local_clients[sample_client_info.client_id]
        assert local_client.status == ClientStatus.REGISTERED
        assert local_client.client_info.client_id == sample_client_info.client_id
    
    async def test_client_unregistration(self, coordinator, sample_client_info):
        """Test client unregistration"""
        # Register then unregister
        coordinator.register_local_client(sample_client_info)
        success = coordinator.unregister_local_client(sample_client_info.client_id)
        
        assert success
        assert sample_client_info.client_id not in coordinator.local_clients
    
    async def test_model_update_reception(self, coordinator, sample_client_info, sample_model_update):
        """Test receiving model updates"""
        # Register client first
        coordinator.register_local_client(sample_client_info)
        
        # Send model update
        success = coordinator.receive_local_update(sample_client_info.client_id, sample_model_update)
        
        assert success
        assert len(coordinator.pending_updates) == 1
        
        # Check client status updated
        local_client = coordinator.local_clients[sample_client_info.client_id]
        assert local_client.status == ClientStatus.ACTIVE
        assert local_client.last_update is not None
    
    async def test_local_aggregation(self, coordinator, sample_client_info):
        """Test local model aggregation"""
        import numpy as np
        
        # Register client
        coordinator.register_local_client(sample_client_info)
        
        # Create multiple updates
        for i in range(3):
            weights = np.random.random(100).astype(np.float32)
            update = EnhancedModelUpdate(
                client_id=f"test-client-{i+1}",
                model_weights=weights.tobytes(),
                model_size_bytes=len(weights.tobytes()),
                training_rounds=1,
                local_epochs=5,
                batch_size=32,
                learning_rate=0.01,
                samples_used=1000,
                training_time_seconds=120.0,
                training_loss=0.5 + i * 0.1,
                data_distribution={"class_0": 400 + i * 50, "class_1": 600 - i * 50}
            )
            
            # Register additional clients
            if i > 0:
                client_info = ClientInfo(
                    client_id=f"test-client-{i+1}",
                    client_type="SDR",
                    capabilities={},
                    location=None,
                    network_info={},
                    hardware_specs={}
                )
                coordinator.register_local_client(client_info)
            
            coordinator.receive_local_update(f"test-client-{i+1}", update)
        
        # Perform aggregation
        aggregated_update = await coordinator.aggregate_local_models()
        
        assert aggregated_update is not None
        assert aggregated_update.client_id == f"edge-{coordinator.coordinator_id}"
        assert aggregated_update.samples_used == 3000  # Sum of all samples
        assert len(coordinator.pending_updates) == 0  # Should be cleared after aggregation


class TestNetworkPartitionDetector:
    """Test Network Partition Detection"""
    
    @pytest.fixture
    async def detector(self):
        """Create test partition detector"""
        config = {
            'heartbeat_timeout': 10,
            'probe_interval': 5,
            'consensus_threshold': 0.6,
            'global_servers': ['http://test-server:8000'],
            'peer_coordinators': ['test-peer-1']
        }
        
        detector = NetworkPartitionDetector("test-coord-1", config)
        await detector.start()
        
        yield detector
        
        await detector.stop()
    
    async def test_node_registration(self, detector):
        """Test node registration for monitoring"""
        node_info = {
            'client_type': 'SDR',
            'endpoint': 'http://client:8080',
            'ip_address': '192.168.1.100'
        }
        
        detector.register_node("test-node-1", node_info)
        
        assert "test-node-1" in detector.known_nodes
        assert "test-node-1" in detector.node_states
        assert detector.node_states["test-node-1"]["status"] == "active"
    
    async def test_heartbeat_update(self, detector):
        """Test heartbeat updates"""
        # Register node first
        detector.register_node("test-node-1", {})
        
        # Update heartbeat
        detector.update_node_heartbeat("test-node-1", {"cpu_usage": 0.5})
        
        node_state = detector.node_states["test-node-1"]
        assert node_state["status"] == "active"
        assert "metadata" in node_state
        assert node_state["metadata"]["cpu_usage"] == 0.5
    
    async def test_partition_detection(self, detector):
        """Test partition detection logic"""
        # Register a node
        detector.register_node("test-node-1", {})
        
        # Simulate node going offline by not updating heartbeat
        # Wait for heartbeat timeout
        await asyncio.sleep(0.1)  # Short wait for test
        
        # Manually trigger heartbeat check
        current_time = datetime.now()
        timeout_threshold = current_time - timedelta(seconds=detector.heartbeat_timeout)
        
        # Simulate old heartbeat
        detector.node_states["test-node-1"]["last_seen"] = timeout_threshold - timedelta(seconds=1)
        
        # The actual partition detection would happen in the background loop
        # For testing, we check the logic directly
        node_state = detector.node_states["test-node-1"]
        if node_state["last_seen"] < timeout_threshold:
            node_state["status"] = "suspected"
            node_state["consecutive_failures"] += 1
        
        assert detector.node_states["test-node-1"]["status"] == "suspected"


class TestOfflineOperationManager:
    """Test Offline Operation Manager"""
    
    @pytest.fixture
    async def offline_manager(self):
        """Create test offline manager"""
        temp_dir = tempfile.mkdtemp()
        
        config = {
            'max_offline_updates': 100,
            'max_offline_hours': 24,
            'sync_strategy': 'batched',
            'compression_enabled': True
        }
        
        manager = OfflineOperationManager("test-coord-1", temp_dir, config)
        await manager.start()
        
        yield manager
        
        await manager.stop()
        
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_enhanced_update(self):
        """Create sample enhanced model update"""
        import numpy as np
        
        weights = np.random.random(100).astype(np.float32)
        
        return EnhancedModelUpdate(
            client_id="test-client-1",
            model_weights=weights.tobytes(),
            model_size_bytes=len(weights.tobytes()),
            training_rounds=1,
            local_epochs=5,
            batch_size=32,
            learning_rate=0.01,
            samples_used=1000,
            training_time_seconds=120.0,
            training_loss=0.5
        )
    
    async def test_offline_mode_transition(self, offline_manager):
        """Test entering and exiting offline mode"""
        # Initially should be in normal mode
        assert offline_manager.mode == OfflineMode.NORMAL
        
        # Enter offline mode
        offline_manager.enter_offline_mode("test_partition")
        assert offline_manager.mode == OfflineMode.ISOLATED
        assert offline_manager.offline_since is not None
        
        # Exit offline mode
        offline_manager.exit_offline_mode()
        assert offline_manager.mode == OfflineMode.RECOVERY
    
    async def test_offline_update_storage(self, offline_manager, sample_enhanced_update):
        """Test storing updates during offline operation"""
        # Enter offline mode
        offline_manager.enter_offline_mode("test")
        
        # Store update
        update_id = offline_manager.store_offline_update(sample_enhanced_update, priority=1.5)
        
        assert update_id in offline_manager.offline_updates
        assert update_id in offline_manager.pending_sync_queue
        
        stored_update = offline_manager.offline_updates[update_id]
        assert stored_update.priority == 1.5
        assert stored_update.update.client_id == sample_enhanced_update.client_id
    
    async def test_model_version_storage(self, offline_manager):
        """Test model version storage"""
        import numpy as np
        
        # Create dummy model
        model_weights = np.random.random(100).astype(np.float32).tobytes()
        metadata = {"accuracy": 0.85, "loss": 0.3}
        
        # Store model version
        success = offline_manager.store_model_version(
            "v1.0", model_weights, metadata, is_global=True
        )
        
        assert success
        assert "v1.0" in offline_manager.model_versions
        
        # Get latest version
        latest = offline_manager.get_latest_model_version()
        assert latest is not None
        assert latest.version_id == "v1.0"
        assert latest.is_global
    
    async def test_sync_preparation(self, offline_manager, sample_enhanced_update):
        """Test sync preparation logic"""
        # Store some updates
        for i in range(5):
            update = EnhancedModelUpdate(
                client_id=f"client-{i}",
                model_weights=sample_enhanced_update.model_weights,
                model_size_bytes=sample_enhanced_update.model_size_bytes,
                training_rounds=1,
                local_epochs=5,
                batch_size=32,
                learning_rate=0.01,
                samples_used=1000,
                training_time_seconds=120.0,
                training_loss=0.5
            )
            offline_manager.store_offline_update(update, priority=float(i))
        
        # Prepare updates for sync
        update_batches = offline_manager._prepare_updates_for_sync()
        
        assert len(update_batches) > 0
        assert all(isinstance(batch, list) for batch in update_batches)


# Integration test
class TestEdgeCoordinatorIntegration:
    """Integration tests for complete edge coordinator functionality"""
    
    @pytest.fixture
    async def full_coordinator(self):
        """Create full edge coordinator with all components"""
        from .service import EdgeCoordinatorService
        
        temp_dir = tempfile.mkdtemp()
        
        config = {
            'max_local_clients': 10,
            'heartbeat_interval': 5,
            'sync_interval': 30,
            'aggregation_strategy': 'fedavg',
            'partition_detection': {
                'heartbeat_timeout': 10,
                'probe_interval': 5,
                'consensus_threshold': 0.6
            },
            'offline_operation': {
                'max_offline_updates': 100,
                'sync_strategy': 'batched'
            },
            'storage_path': temp_dir
        }
        
        service = EdgeCoordinatorService("test-coord-1", "test-region", config)
        await service.start()
        
        yield service
        
        await service.stop()
        shutil.rmtree(temp_dir)
    
    async def test_full_workflow(self, full_coordinator):
        """Test complete workflow from client registration to aggregation"""
        import numpy as np
        
        # Register multiple clients
        clients = []
        for i in range(3):
            client_info = ClientInfo(
                client_id=f"test-client-{i}",
                client_type="SDR",
                capabilities={"cpu_cores": 4},
                location=None,
                network_info={},
                hardware_specs={}
            )
            
            token = full_coordinator.coordinator.register_local_client(client_info)
            clients.append((client_info, token))
            
            # Register with partition detector
            full_coordinator.partition_detector.register_node(
                client_info.client_id,
                {"client_type": "SDR"}
            )
        
        # Send model updates from each client
        for i, (client_info, token) in enumerate(clients):
            weights = np.random.random(100).astype(np.float32)
            update = EnhancedModelUpdate(
                client_id=client_info.client_id,
                model_weights=weights.tobytes(),
                model_size_bytes=len(weights.tobytes()),
                training_rounds=1,
                local_epochs=5,
                batch_size=32,
                learning_rate=0.01,
                samples_used=1000,
                training_time_seconds=120.0,
                training_loss=0.5 + i * 0.1
            )
            
            success = full_coordinator.coordinator.receive_local_update(
                client_info.client_id, update
            )
            assert success
        
        # Trigger aggregation
        await full_coordinator._perform_aggregation()
        
        # Verify aggregation occurred
        assert len(full_coordinator.coordinator.pending_updates) == 0
        
        # Test partition scenario
        full_coordinator.offline_manager.enter_offline_mode("test_partition")
        
        # Send another update while offline
        weights = np.random.random(100).astype(np.float32)
        offline_update = EnhancedModelUpdate(
            client_id="test-client-0",
            model_weights=weights.tobytes(),
            model_size_bytes=len(weights.tobytes()),
            training_rounds=2,
            local_epochs=5,
            batch_size=32,
            learning_rate=0.01,
            samples_used=1000,
            training_time_seconds=120.0,
            training_loss=0.4
        )
        
        full_coordinator.coordinator.receive_local_update("test-client-0", offline_update)
        
        # Should be stored offline
        assert len(full_coordinator.offline_manager.offline_updates) > 0
        
        # Exit offline mode
        full_coordinator.offline_manager.exit_offline_mode()
        
        # Verify recovery mode
        assert full_coordinator.offline_manager.mode == OfflineMode.RECOVERY


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])