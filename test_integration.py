#!/usr/bin/env python3
"""
Integration test for aggregation server
"""
import asyncio
import sys
import os
import json
import pickle
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.aggregation_server.server import AggregationServer
from src.common.interfaces import ClientInfo, ModelUpdate
from src.common.config import get_config


async def test_full_integration():
    """Test full integration scenario"""
    print("Testing full integration scenario...")
    
    config = get_config()
    server = AggregationServer(config)
    
    try:
        # Initialize server
        await server.initialize()
        print("✓ Server initialized")
        
        # Register multiple clients
        clients = []
        for i in range(3):
            client_info = ClientInfo(
                client_id=f"client_{i}",
                client_type="Simulated",
                capabilities={"cpu_cores": 4, "memory_gb": 8, "gpu_available": i % 2 == 0},
                location={"lat": 37.7749 + i, "lon": -122.4194 + i},
                network_info={"bandwidth": 100 + i * 10, "latency": 10 + i},
                hardware_specs={"gpu": i % 2 == 0},
                reputation_score=1.0
            )
            
            token = await server.register_client(client_info)
            clients.append((client_info.client_id, token))
            print(f"✓ Registered client {client_info.client_id}")
        
        # Submit model updates from all clients
        for i, (client_id, token) in enumerate(clients):
            # Create dummy model weights
            weights = {
                "layer1": np.random.randn(10, 5).astype(np.float32),
                "layer2": np.random.randn(5, 1).astype(np.float32)
            }
            
            model_update = ModelUpdate(
                client_id=client_id,
                model_weights=pickle.dumps(weights),
                training_metrics={"loss": 0.5 - i * 0.1, "accuracy": 0.7 + i * 0.05},
                data_statistics={"num_samples": 1000 + i * 100},
                computation_time=30.0 + i * 5,
                network_conditions={"latency": 10 + i, "bandwidth": 100 + i * 10},
                privacy_budget_used=0.1
            )
            
            success = await server.receive_model_update(client_id, model_update)
            print(f"✓ Model update from {client_id}: {success}")
        
        # Wait a moment for aggregation to complete
        await asyncio.sleep(1)
        
        # Test global model retrieval
        for client_id, _ in clients:
            global_model = await server.get_global_model(client_id)
            if global_model:
                print(f"✓ Global model retrieved for {client_id}: version {global_model['version']}")
            else:
                print(f"⚠ No global model available for {client_id}")
        
        # Test training configuration
        for client_id, _ in clients:
            config_data = await server.get_training_configuration(client_id)
            print(f"✓ Training config for {client_id}: {config_data['aggregation_strategy']}")
        
        # Test server status
        status = await server.get_server_status()
        print(f"✓ Server status: {status['status']}")
        print(f"✓ Total clients: {status['total_clients']}")
        print(f"✓ Current round: {status['current_round']}")
        
        # Test client metrics reporting
        for client_id, _ in clients:
            metrics = {
                "cpu_usage": 50.0 + np.random.randn() * 10,
                "memory_usage": 60.0 + np.random.randn() * 15,
                "network_latency": 10.0 + np.random.randn() * 2
            }
            await server.report_client_metrics(client_id, metrics)
            print(f"✓ Metrics reported for {client_id}")
        
        print("\n✅ Full integration test passed!")
        return True
        
    except Exception as e:
        print(f"❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        await server.shutdown()
        print("✓ Server shutdown complete")


if __name__ == "__main__":
    success = asyncio.run(test_full_integration())
    sys.exit(0 if success else 1)