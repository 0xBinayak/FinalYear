"""
Basic tests for aggregation server functionality
"""
import asyncio
import pickle
import numpy as np
from datetime import datetime

from .server import AggregationServer
from ..common.interfaces import ClientInfo, ModelUpdate
from ..common.config import get_config


async def test_basic_functionality():
    """Test basic aggregation server functionality"""
    print("Testing basic aggregation server functionality...")
    
    # Get configuration
    config = get_config()
    
    # Create server instance
    server = AggregationServer(config)
    
    try:
        # Initialize server
        await server.initialize()
        print("✓ Server initialized successfully")
        
        # Test client registration
        client_info = ClientInfo(
            client_id="test_client_1",
            client_type="Simulated",
            capabilities={"cpu_cores": 4, "memory_gb": 8},
            location={"lat": 37.7749, "lon": -122.4194},
            network_info={"bandwidth": 100, "latency": 10},
            hardware_specs={"gpu": False},
            reputation_score=1.0
        )
        
        token = await server.register_client(client_info)
        print(f"✓ Client registered with token: {token[:16]}...")
        
        # Test model update
        dummy_weights = pickle.dumps({"layer1": np.random.randn(10, 5), "layer2": np.random.randn(5, 1)})
        
        model_update = ModelUpdate(
            client_id="test_client_1",
            model_weights=dummy_weights,
            training_metrics={"loss": 0.5, "accuracy": 0.8},
            data_statistics={"num_samples": 1000},
            computation_time=30.0,
            network_conditions={"latency": 10, "bandwidth": 100},
            privacy_budget_used=0.1
        )
        
        success = await server.receive_model_update("test_client_1", model_update)
        print(f"✓ Model update received: {success}")
        
        # Test health status
        health = await server.get_health_status()
        print(f"✓ Health status: {health['status']}")
        
        # Test server status
        status = await server.get_server_status()
        print(f"✓ Server status: {status['status']}, Active clients: {status['active_clients']}")
        
        # Test training configuration
        config_data = await server.get_training_configuration("test_client_1")
        print(f"✓ Training config retrieved: {config_data['aggregation_strategy']}")
        
        print("\n✅ All basic tests passed!")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        raise
    
    finally:
        # Cleanup
        await server.shutdown()
        print("✓ Server shutdown complete")


if __name__ == "__main__":
    asyncio.run(test_basic_functionality())