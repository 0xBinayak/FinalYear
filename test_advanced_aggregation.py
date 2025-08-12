#!/usr/bin/env python3
"""
Test advanced aggregation strategies
"""
import asyncio
import sys
import os
import pickle
import numpy as np
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.aggregation_server.server import AggregationServer
from src.common.interfaces import ClientInfo, ModelUpdate
from src.common.config import get_config


async def test_advanced_aggregation():
    """Test advanced aggregation strategies"""
    print("Testing advanced aggregation strategies...")
    
    config = get_config()
    
    # Test different strategies
    strategies = ['fedavg', 'krum', 'trimmed_mean', 'weighted']
    
    for strategy in strategies:
        print(f"\n--- Testing {strategy.upper()} strategy ---")
        
        # Update config for this strategy
        config.federated_learning.aggregation_strategy = strategy
        config.federated_learning.min_clients = 1  # Allow single client for testing
        
        server = AggregationServer(config)
        
        try:
            await server.initialize()
            print(f"✓ Server initialized with {strategy} strategy")
            
            # Register clients with different quality profiles
            clients = []
            for i in range(5):
                client_info = ClientInfo(
                    client_id=f"client_{i}",
                    client_type="Simulated",
                    capabilities={"cpu_cores": 4 + i, "memory_gb": 8 + i * 2, "gpu_available": i % 2 == 0},
                    location={"lat": 37.7749 + i, "lon": -122.4194 + i},
                    network_info={"bandwidth": 50 + i * 20, "latency": 20 - i * 2},
                    hardware_specs={"gpu": i % 2 == 0},
                    reputation_score=0.6 + i * 0.08  # Varying reputation
                )
                
                token = await server.register_client(client_info)
                clients.append((client_info.client_id, token))
            
            print(f"✓ Registered {len(clients)} clients")
            
            # Submit model updates with varying quality
            for i, (client_id, token) in enumerate(clients):
                # Create model weights with some variation
                base_weights = {
                    "layer1": np.random.randn(10, 5).astype(np.float32) * (1 + i * 0.1),
                    "layer2": np.random.randn(5, 1).astype(np.float32) * (1 + i * 0.1)
                }
                
                # Add some "Byzantine" behavior for testing robustness
                if strategy in ['krum', 'trimmed_mean'] and i == 4:
                    # Make last client Byzantine (extreme values)
                    base_weights["layer1"] *= 10
                    base_weights["layer2"] *= 10
                
                model_update = ModelUpdate(
                    client_id=client_id,
                    model_weights=pickle.dumps(base_weights),
                    training_metrics={
                        "loss": 1.0 - i * 0.15,  # Decreasing loss
                        "accuracy": 0.5 + i * 0.08  # Increasing accuracy
                    },
                    data_statistics={"num_samples": 800 + i * 200},
                    computation_time=25.0 + i * 5,
                    network_conditions={
                        "latency": 20 - i * 2,
                        "bandwidth": 50 + i * 20,
                        "packet_loss": i * 0.01
                    },
                    privacy_budget_used=0.1
                )
                
                success = await server.receive_model_update(client_id, model_update)
                print(f"✓ Model update from {client_id}: {success}")
            
            # Wait for aggregation
            await asyncio.sleep(1)
            
            # Check results
            status = await server.get_server_status()
            print(f"✓ Aggregation completed: Round {status['current_round']}")
            
            # Test convergence history
            convergence_history = await server.get_convergence_history()
            if convergence_history:
                latest_metrics = convergence_history[-1]
                print(f"✓ Latest convergence metrics: Loss={latest_metrics.get('average_loss', 'N/A'):.4f}, "
                      f"Accuracy={latest_metrics.get('average_accuracy', 'N/A'):.4f}")
            
            # Test strategy switching
            available_strategies = await server.get_available_strategies()
            print(f"✓ Available strategies: {available_strategies}")
            
        except Exception as e:
            print(f"❌ Test failed for {strategy}: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            await server.shutdown()
            print(f"✓ Server shutdown for {strategy}")
    
    print("\n✅ Advanced aggregation strategy tests completed!")


async def test_client_selection():
    """Test adaptive client selection"""
    print("\nTesting adaptive client selection...")
    
    config = get_config()
    config.federated_learning.aggregation_strategy = 'weighted'
    config.federated_learning.max_clients = 3  # Limit to test selection
    config.federated_learning.min_clients = 1
    
    server = AggregationServer(config)
    
    try:
        await server.initialize()
        print("✓ Server initialized for client selection test")
        
        # Register many clients with different profiles
        clients = []
        for i in range(8):
            client_info = ClientInfo(
                client_id=f"select_client_{i}",
                client_type="Simulated",
                capabilities={"cpu_cores": 2 + i, "memory_gb": 4 + i},
                location={"lat": 37.0 + i, "lon": -122.0 + i},
                network_info={"bandwidth": 20 + i * 15, "latency": 50 - i * 5},
                hardware_specs={"gpu": i > 4},
                reputation_score=0.3 + i * 0.1  # Varying reputation
            )
            
            token = await server.register_client(client_info)
            clients.append((client_info.client_id, token))
        
        print(f"✓ Registered {len(clients)} clients for selection test")
        
        # Submit updates with varying quality
        for i, (client_id, token) in enumerate(clients):
            weights = {
                "layer1": np.random.randn(5, 3).astype(np.float32),
                "layer2": np.random.randn(3, 1).astype(np.float32)
            }
            
            model_update = ModelUpdate(
                client_id=client_id,
                model_weights=pickle.dumps(weights),
                training_metrics={
                    "loss": 2.0 - i * 0.2,  # Better clients have lower loss
                    "accuracy": 0.3 + i * 0.08  # Better clients have higher accuracy
                },
                data_statistics={"num_samples": 500 + i * 100},
                computation_time=20.0 + i * 3,
                network_conditions={
                    "latency": 50 - i * 5,
                    "bandwidth": 20 + i * 15
                },
                privacy_budget_used=0.05
            )
            
            success = await server.receive_model_update(client_id, model_update)
            print(f"✓ Update from {client_id} (quality rank ~{i+1}): {success}")
        
        # Wait for aggregation
        await asyncio.sleep(1)
        
        # Check which clients were selected
        status = await server.get_server_status()
        print(f"✓ Client selection completed: Round {status['current_round']}")
        
        # Check round metrics to see selection
        if hasattr(server, 'round_metrics') and server.round_metrics:
            latest_round = max(server.round_metrics.keys())
            round_info = server.round_metrics[latest_round]
            selected_count = round_info['participating_clients']
            total_count = round_info['total_available_clients']
            print(f"✓ Selected {selected_count} out of {total_count} available clients")
        
    except Exception as e:
        print(f"❌ Client selection test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await server.shutdown()
        print("✓ Client selection test completed")


if __name__ == "__main__":
    async def run_all_tests():
        await test_advanced_aggregation()
        await test_client_selection()
    
    asyncio.run(run_all_tests())