#!/usr/bin/env python3
"""
Basic test for edge coordinator functionality
"""
import sys
import os
import asyncio
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from common.interfaces import ClientInfo
from common.federated_data_structures import EnhancedModelUpdate, NetworkConditions, ComputeResources
from edge_coordinator.coordinator import EdgeCoordinator, CoordinatorState, ClientStatus


async def test_edge_coordinator():
    """Test basic edge coordinator functionality"""
    print("Testing Edge Coordinator...")
    
    # Create coordinator
    config = {
        'max_local_clients': 10,
        'heartbeat_interval': 5,
        'sync_interval': 30,
        'aggregation_strategy': 'fedavg'
    }
    
    coordinator = EdgeCoordinator("test-coord-1", "test-region", config)
    await coordinator.start()
    
    try:
        # Test 1: Client registration
        print("\n1. Testing client registration...")
        client_info = ClientInfo(
            client_id="test-client-1",
            client_type="SDR",
            capabilities={"cpu_cores": 4, "memory_gb": 8},
            location={"lat": 37.7749, "lon": -122.4194},
            network_info={"bandwidth_mbps": 100},
            hardware_specs={"sdr_type": "rtlsdr"},
            reputation_score=1.0
        )
        
        token = coordinator.register_local_client(client_info)
        print(f"✓ Client registered with token: {token}")
        print(f"✓ Local clients count: {len(coordinator.local_clients)}")
        
        # Verify client is registered
        assert client_info.client_id in coordinator.local_clients
        local_client = coordinator.local_clients[client_info.client_id]
        assert local_client.status == ClientStatus.REGISTERED
        print("✓ Client registration verified")
        
        # Test 2: Model update reception
        print("\n2. Testing model update reception...")
        import numpy as np
        
        weights = np.random.random(100).astype(np.float32)
        model_update = EnhancedModelUpdate(
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
        
        success = coordinator.receive_local_update(client_info.client_id, model_update)
        print(f"✓ Model update received: {success}")
        print(f"✓ Pending updates count: {len(coordinator.pending_updates)}")
        
        # Verify update was received
        assert success
        assert len(coordinator.pending_updates) == 1
        local_client = coordinator.local_clients[client_info.client_id]
        assert local_client.status == ClientStatus.ACTIVE
        print("✓ Model update reception verified")
        
        # Test 3: Register multiple clients and test aggregation
        print("\n3. Testing multiple clients and aggregation...")
        
        for i in range(2, 4):  # Add 2 more clients
            client_info_i = ClientInfo(
                client_id=f"test-client-{i}",
                client_type="SDR",
                capabilities={"cpu_cores": 4},
                location=None,
                network_info={},
                hardware_specs={}
            )
            
            coordinator.register_local_client(client_info_i)
            
            # Send update from this client
            weights_i = np.random.random(100).astype(np.float32)
            update_i = EnhancedModelUpdate(
                client_id=f"test-client-{i}",
                model_weights=weights_i.tobytes(),
                model_size_bytes=len(weights_i.tobytes()),
                training_rounds=1,
                local_epochs=5,
                batch_size=32,
                learning_rate=0.01,
                samples_used=1000,
                training_time_seconds=120.0,
                training_loss=0.5 + i * 0.1,
                data_distribution={"class_0": 400 + i * 50, "class_1": 600 - i * 50}
            )
            
            coordinator.receive_local_update(f"test-client-{i}", update_i)
        
        print(f"✓ Total clients registered: {len(coordinator.local_clients)}")
        print(f"✓ Total pending updates: {len(coordinator.pending_updates)}")
        
        # Test aggregation
        print("\n4. Testing local aggregation...")
        aggregated_update = await coordinator.aggregate_local_models()
        
        if aggregated_update:
            print(f"✓ Aggregation successful")
            print(f"✓ Aggregated client ID: {aggregated_update.client_id}")
            print(f"✓ Total samples used: {aggregated_update.samples_used}")
            print(f"✓ Pending updates after aggregation: {len(coordinator.pending_updates)}")
            
            # Verify aggregation
            assert aggregated_update.client_id == f"edge-{coordinator.coordinator_id}"
            assert aggregated_update.samples_used == 3000  # 3 clients * 1000 samples each
            assert len(coordinator.pending_updates) == 0  # Should be cleared
            print("✓ Local aggregation verified")
        else:
            print("✗ Aggregation failed")
        
        # Test 4: Coordinator status
        print("\n5. Testing coordinator status...")
        status = coordinator.get_coordinator_status()
        print(f"✓ Coordinator ID: {status['coordinator_id']}")
        print(f"✓ Region: {status['region']}")
        print(f"✓ State: {status['state']}")
        print(f"✓ Local clients count: {status['local_clients_count']}")
        print(f"✓ Active clients: {status['active_clients']}")
        
        # Test 5: Client unregistration
        print("\n6. Testing client unregistration...")
        success = coordinator.unregister_local_client("test-client-1")
        print(f"✓ Client unregistration: {success}")
        print(f"✓ Remaining clients: {len(coordinator.local_clients)}")
        
        assert success
        assert "test-client-1" not in coordinator.local_clients
        print("✓ Client unregistration verified")
        
        print("\n🎉 All edge coordinator tests passed!")
        
    finally:
        await coordinator.stop()


async def test_network_partition_detector():
    """Test network partition detection"""
    print("\n" + "="*50)
    print("Testing Network Partition Detector...")
    
    from edge_coordinator.network_partition import NetworkPartitionDetector
    
    config = {
        'heartbeat_timeout': 10,
        'probe_interval': 5,
        'consensus_threshold': 0.6,
        'global_servers': ['http://test-server:8000'],
        'peer_coordinators': ['test-peer-1']
    }
    
    detector = NetworkPartitionDetector("test-coord-1", config)
    await detector.start()
    
    try:
        # Test node registration
        print("\n1. Testing node registration...")
        node_info = {
            'client_type': 'SDR',
            'endpoint': 'http://client:8080',
            'ip_address': '192.168.1.100'
        }
        
        detector.register_node("test-node-1", node_info)
        print(f"✓ Node registered: test-node-1")
        print(f"✓ Known nodes: {len(detector.known_nodes)}")
        
        assert "test-node-1" in detector.known_nodes
        assert "test-node-1" in detector.node_states
        print("✓ Node registration verified")
        
        # Test heartbeat update
        print("\n2. Testing heartbeat updates...")
        detector.update_node_heartbeat("test-node-1", {"cpu_usage": 0.5})
        
        node_state = detector.node_states["test-node-1"]
        print(f"✓ Node status: {node_state['status']}")
        print(f"✓ Metadata updated: {node_state.get('metadata', {})}")
        
        assert node_state["status"] == "active"
        print("✓ Heartbeat update verified")
        
        # Test partition status
        print("\n3. Testing partition status...")
        status = detector.get_partition_status()
        print(f"✓ Active partitions: {status['active_partitions']}")
        print(f"✓ Node states count: {len(status['node_states'])}")
        
        print("✓ Network partition detector tests passed!")
        
    finally:
        await detector.stop()


async def test_offline_operation_manager():
    """Test offline operation manager"""
    print("\n" + "="*50)
    print("Testing Offline Operation Manager...")
    
    import tempfile
    import shutil
    from edge_coordinator.offline_operation import OfflineOperationManager, OfflineMode
    
    temp_dir = tempfile.mkdtemp()
    
    try:
        config = {
            'max_offline_updates': 100,
            'max_offline_hours': 24,
            'sync_strategy': 'batched',
            'compression_enabled': True
        }
        
        manager = OfflineOperationManager("test-coord-1", temp_dir, config)
        await manager.start()
        
        try:
            # Test mode transitions
            print("\n1. Testing offline mode transitions...")
            print(f"✓ Initial mode: {manager.mode.value}")
            assert manager.mode == OfflineMode.NORMAL
            
            manager.enter_offline_mode("test_partition")
            print(f"✓ Offline mode: {manager.mode.value}")
            assert manager.mode == OfflineMode.ISOLATED
            
            manager.exit_offline_mode()
            print(f"✓ Recovery mode: {manager.mode.value}")
            assert manager.mode == OfflineMode.RECOVERY
            print("✓ Mode transitions verified")
            
            # Test offline update storage
            print("\n2. Testing offline update storage...")
            import numpy as np
            
            weights = np.random.random(100).astype(np.float32)
            update = EnhancedModelUpdate(
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
            
            update_id = manager.store_offline_update(update, priority=1.5)
            print(f"✓ Update stored with ID: {update_id}")
            print(f"✓ Offline updates count: {len(manager.offline_updates)}")
            print(f"✓ Pending sync count: {len(manager.pending_sync_queue)}")
            
            assert update_id in manager.offline_updates
            assert update_id in manager.pending_sync_queue
            print("✓ Offline update storage verified")
            
            # Test model version storage
            print("\n3. Testing model version storage...")
            model_weights = np.random.random(100).astype(np.float32).tobytes()
            metadata = {"accuracy": 0.85, "loss": 0.3}
            
            success = manager.store_model_version("v1.0", model_weights, metadata, is_global=True)
            print(f"✓ Model version stored: {success}")
            print(f"✓ Model versions count: {len(manager.model_versions)}")
            
            latest = manager.get_latest_model_version()
            print(f"✓ Latest version: {latest.version_id if latest else None}")
            
            assert success
            assert "v1.0" in manager.model_versions
            assert latest is not None
            assert latest.version_id == "v1.0"
            print("✓ Model version storage verified")
            
            # Test offline status
            print("\n4. Testing offline status...")
            status = manager.get_offline_status()
            print(f"✓ Mode: {status['mode']}")
            print(f"✓ Offline updates: {status['offline_updates_count']}")
            print(f"✓ Model versions: {status['model_versions_count']}")
            
            print("✓ Offline operation manager tests passed!")
            
        finally:
            await manager.stop()
    
    finally:
        shutil.rmtree(temp_dir)


async def main():
    """Run all tests"""
    print("🚀 Starting Edge Coordinator Tests")
    print("="*50)
    
    try:
        await test_edge_coordinator()
        await test_network_partition_detector()
        await test_offline_operation_manager()
        
        print("\n" + "="*50)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("Edge Coordinator implementation is working correctly!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)