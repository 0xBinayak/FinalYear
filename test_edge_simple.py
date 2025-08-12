#!/usr/bin/env python3
"""
Simple test for edge coordinator functionality without relative imports
"""
import sys
import os
import asyncio
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import directly without using the __init__.py
import common.interfaces as interfaces
import common.federated_data_structures as fed_structs

# Import the coordinator module directly
import importlib.util
import types

# Load coordinator module manually to avoid relative import issues
coord_path = os.path.join(os.path.dirname(__file__), 'src', 'edge_coordinator', 'coordinator.py')
spec = importlib.util.spec_from_file_location("coordinator", coord_path)
coordinator_module = importlib.util.module_from_spec(spec)

# Set up the module's namespace to resolve relative imports
coordinator_module.interfaces = interfaces
coordinator_module.fed_structs = fed_structs

# Patch the imports in the coordinator module
sys.modules['edge_coordinator.coordinator'] = coordinator_module

# Now execute the module
spec.loader.exec_module(coordinator_module)

# Extract the classes we need
EdgeCoordinator = coordinator_module.EdgeCoordinator
CoordinatorState = coordinator_module.CoordinatorState
ClientStatus = coordinator_module.ClientStatus


async def test_basic_functionality():
    """Test basic edge coordinator functionality"""
    print("🚀 Testing Edge Coordinator Basic Functionality")
    print("="*50)
    
    try:
        # Create coordinator
        config = {
            'max_local_clients': 10,
            'heartbeat_interval': 5,
            'sync_interval': 30,
            'aggregation_strategy': 'fedavg'
        }
        
        coordinator = EdgeCoordinator("test-coord-1", "test-region", config)
        print("✓ EdgeCoordinator created successfully")
        
        # Test client registration
        client_info = interfaces.ClientInfo(
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
        
        # Verify registration
        assert client_info.client_id in coordinator.local_clients
        local_client = coordinator.local_clients[client_info.client_id]
        assert local_client.status == ClientStatus.REGISTERED
        print("✓ Client registration verified")
        
        # Test model update
        import numpy as np
        
        weights = np.random.random(100).astype(np.float32)
        model_update = fed_structs.EnhancedModelUpdate(
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
            data_distribution={"class_0": 400, "class_1": 600}
        )
        
        success = coordinator.receive_local_update(client_info.client_id, model_update)
        print(f"✓ Model update received: {success}")
        print(f"✓ Pending updates count: {len(coordinator.pending_updates)}")
        
        # Verify update
        assert success
        assert len(coordinator.pending_updates) == 1
        local_client = coordinator.local_clients[client_info.client_id]
        assert local_client.status == ClientStatus.ACTIVE
        print("✓ Model update reception verified")
        
        # Test coordinator status
        status = coordinator.get_coordinator_status()
        print(f"✓ Coordinator status retrieved:")
        print(f"  - ID: {status['coordinator_id']}")
        print(f"  - Region: {status['region']}")
        print(f"  - State: {status['state']}")
        print(f"  - Local clients: {status['local_clients_count']}")
        print(f"  - Active clients: {status['active_clients']}")
        
        # Test client unregistration
        success = coordinator.unregister_local_client("test-client-1")
        print(f"✓ Client unregistration: {success}")
        print(f"✓ Remaining clients: {len(coordinator.local_clients)}")
        
        assert success
        assert "test-client-1" not in coordinator.local_clients
        print("✓ Client unregistration verified")
        
        print("\n🎉 All basic functionality tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_data_structures():
    """Test federated data structures"""
    print("\n" + "="*50)
    print("Testing Federated Data Structures...")
    
    try:
        # Test NetworkConditions
        network_conditions = fed_structs.NetworkConditions(
            bandwidth_mbps=50.0,
            latency_ms=20.0,
            packet_loss_rate=0.01,
            jitter_ms=5.0,
            connection_stability=0.95
        )
        
        transmission_time = network_conditions.get_transmission_time_estimate(1024 * 1024)  # 1MB
        print(f"✓ NetworkConditions created, transmission time: {transmission_time:.2f}s")
        
        # Test ComputeResources
        compute_resources = fed_structs.ComputeResources(
            cpu_cores=4,
            cpu_frequency_ghz=2.5,
            memory_gb=8.0,
            available_memory_gb=6.0
        )
        
        can_train = compute_resources.can_handle_training(100.0, 32)  # 100MB model, batch size 32
        print(f"✓ ComputeResources created, can handle training: {can_train}")
        
        # Test EnhancedModelUpdate
        import numpy as np
        weights = np.random.random(100).astype(np.float32)
        
        model_update = fed_structs.EnhancedModelUpdate(
            client_id="test-client",
            model_weights=weights.tobytes(),
            model_size_bytes=len(weights.tobytes()),
            training_rounds=1,
            local_epochs=5,
            batch_size=32,
            learning_rate=0.01,
            samples_used=1000,
            training_time_seconds=120.0,
            training_loss=0.5,
            network_conditions=network_conditions,
            compute_resources=compute_resources
        )
        
        print(f"✓ EnhancedModelUpdate created")
        print(f"  - Client ID: {model_update.client_id}")
        print(f"  - Model size: {model_update.model_size_bytes} bytes")
        print(f"  - Training loss: {model_update.training_loss}")
        
        # Test integrity verification
        integrity_ok = model_update.verify_integrity()
        print(f"✓ Integrity verification: {integrity_ok}")
        
        # Test serialization
        serialized = model_update.serialize('pickle')
        print(f"✓ Serialization successful, size: {len(serialized)} bytes")
        
        # Test deserialization
        deserialized = fed_structs.EnhancedModelUpdate.deserialize(serialized, 'pickle')
        print(f"✓ Deserialization successful")
        print(f"  - Client ID matches: {deserialized.client_id == model_update.client_id}")
        print(f"  - Training loss matches: {deserialized.training_loss == model_update.training_loss}")
        
        print("✓ Federated data structures tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Data structures test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("🚀 Starting Edge Coordinator Simple Tests")
    
    success = True
    
    # Test data structures first (synchronous)
    success &= test_data_structures()
    
    # Test basic functionality (asynchronous)
    success &= asyncio.run(test_basic_functionality())
    
    if success:
        print("\n" + "="*50)
        print("🎉 ALL TESTS PASSED! 🎉")
        print("Edge Coordinator basic implementation is working!")
        return 0
    else:
        print("\n" + "="*50)
        print("❌ SOME TESTS FAILED")
        return 1


if __name__ == "__main__":
    import sys
    exit_code = main()
    sys.exit(exit_code)