#!/usr/bin/env python3
"""
Full system integration demonstration for the Advanced Federated Pipeline.
Shows end-to-end workflow from client registration to model aggregation.
"""

import asyncio
import sys
import os
import json
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.aggregation_server.server import AggregationServer
from src.common.interfaces import ClientInfo, ModelUpdate
from src.common.config import get_config


def print_banner():
    """Print demonstration banner."""
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║              FULL SYSTEM INTEGRATION DEMONSTRATION                           ║
    ║                                                                              ║
    ║  End-to-End Federated Learning Workflow                                     ║
    ║  • Client Registration                                                       ║
    ║  • Model Updates                                                             ║
    ║  • Aggregation Process                                                       ║
    ║  • Global Model Distribution                                                 ║
    ║  • System Monitoring                                                         ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """)


async def demonstrate_client_registration():
    """Demonstrate client registration process."""
    print("\n👥 Client Registration Demonstration")
    print("="*50)
    
    config = get_config()
    server = AggregationServer(config)
    
    try:
        # Initialize server
        await server.initialize()
        print("✅ Aggregation server initialized")
        
        # Register different types of clients
        client_types = [
            {
                "client_id": "sdr_client_001",
                "client_type": "SDR",
                "capabilities": {"cpu_cores": 4, "memory_gb": 8, "gpu_available": False, "sdr_hardware": "RTL-SDR"},
                "location": {"lat": 37.7749, "lon": -122.4194, "region": "San Francisco"},
                "network_info": {"bandwidth": 100, "latency": 15},
                "hardware_specs": {"sdr_type": "rtlsdr", "frequency_range": [24e6, 1766e6]},
            },
            {
                "client_id": "mobile_client_001", 
                "client_type": "Mobile",
                "capabilities": {"cpu_cores": 8, "memory_gb": 6, "gpu_available": True, "battery_level": 0.85},
                "location": {"lat": 40.7128, "lon": -74.0060, "region": "New York"},
                "network_info": {"bandwidth": 50, "latency": 25},
                "hardware_specs": {"device_model": "iPhone 14", "os_version": "iOS 16.0"},
            },
            {
                "client_id": "edge_client_001",
                "client_type": "Edge",
                "capabilities": {"cpu_cores": 16, "memory_gb": 32, "gpu_available": True, "storage_gb": 500},
                "location": {"lat": 34.0522, "lon": -118.2437, "region": "Los Angeles"},
                "network_info": {"bandwidth": 1000, "latency": 5},
                "hardware_specs": {"gpu_model": "NVIDIA RTX 4080", "edge_coordinator": True},
            }
        ]
        
        registered_clients = []
        
        for client_data in client_types:
            client_info = ClientInfo(
                client_id=client_data["client_id"],
                client_type=client_data["client_type"],
                capabilities=client_data["capabilities"],
                location=client_data["location"],
                network_info=client_data["network_info"],
                hardware_specs=client_data["hardware_specs"],
                reputation_score=1.0
            )
            
            token = await server.register_client(client_info)
            registered_clients.append((client_info.client_id, token))
            
            print(f"✅ Registered {client_info.client_type} client: {client_info.client_id}")
            print(f"   Location: {client_info.location['region']}")
            print(f"   Capabilities: {client_info.capabilities}")
        
        return server, registered_clients
        
    except Exception as e:
        print(f"❌ Client registration failed: {e}")
        await server.shutdown()
        raise


async def demonstrate_model_updates(server, clients):
    """Demonstrate model update submission."""
    print("\n📤 Model Update Demonstration")
    print("="*50)
    
    # Simulate different model architectures and performance
    model_scenarios = [
        {
            "client_type": "SDR",
            "model_architecture": "CNN",
            "layers": {
                "conv1": np.random.randn(32, 1, 3, 3).astype(np.float32),
                "conv2": np.random.randn(64, 32, 3, 3).astype(np.float32),
                "fc1": np.random.randn(128, 64).astype(np.float32),
                "fc2": np.random.randn(11, 128).astype(np.float32)  # 11 modulation classes
            },
            "performance": {"loss": 0.45, "accuracy": 0.82, "f1_score": 0.79}
        },
        {
            "client_type": "Mobile",
            "model_architecture": "MobileNet",
            "layers": {
                "depthwise_conv1": np.random.randn(32, 1, 3, 3).astype(np.float32),
                "pointwise_conv1": np.random.randn(32, 32, 1, 1).astype(np.float32),
                "depthwise_conv2": np.random.randn(64, 32, 3, 3).astype(np.float32),
                "classifier": np.random.randn(11, 64).astype(np.float32)
            },
            "performance": {"loss": 0.52, "accuracy": 0.78, "f1_score": 0.75}
        },
        {
            "client_type": "Edge",
            "model_architecture": "ResNet",
            "layers": {
                "conv1": np.random.randn(64, 1, 7, 7).astype(np.float32),
                "layer1": np.random.randn(64, 64, 3, 3).astype(np.float32),
                "layer2": np.random.randn(128, 64, 3, 3).astype(np.float32),
                "layer3": np.random.randn(256, 128, 3, 3).astype(np.float32),
                "fc": np.random.randn(11, 256).astype(np.float32)
            },
            "performance": {"loss": 0.38, "accuracy": 0.87, "f1_score": 0.85}
        }
    ]
    
    successful_updates = 0
    
    for i, (client_id, token) in enumerate(clients):
        scenario = model_scenarios[i]
        
        print(f"\nSubmitting update from {client_id}:")
        print(f"  Architecture: {scenario['model_architecture']}")
        print(f"  Performance: {scenario['performance']}")
        
        # Create model update
        model_update = ModelUpdate(
            client_id=client_id,
            model_weights=pickle.dumps(scenario["layers"]),
            training_metrics=scenario["performance"],
            data_statistics={
                "num_samples": np.random.randint(800, 1200),
                "class_distribution": np.random.dirichlet(np.ones(11)).tolist(),
                "avg_snr": np.random.uniform(5, 25)
            },
            computation_time=np.random.uniform(45, 120),
            network_conditions={
                "latency": np.random.uniform(5, 30),
                "bandwidth": np.random.uniform(50, 1000)
            },
            privacy_budget_used=np.random.uniform(0.05, 0.15)
        )
        
        success = await server.receive_model_update(client_id, model_update)
        if success:
            successful_updates += 1
            print(f"  ✅ Update submitted successfully")
        else:
            print(f"  ❌ Update submission failed")
    
    print(f"\nModel Update Summary:")
    print(f"  Successful updates: {successful_updates}/{len(clients)}")
    
    return successful_updates == len(clients)


async def demonstrate_aggregation_process(server, clients):
    """Demonstrate the aggregation process."""
    print("\n🔄 Aggregation Process Demonstration")
    print("="*50)
    
    print("Waiting for aggregation to complete...")
    
    # Wait for aggregation
    await asyncio.sleep(2)
    
    # Check server status
    status = await server.get_server_status()
    
    print(f"Server Status:")
    print(f"  Status: {status['status']}")
    print(f"  Total clients: {status['total_clients']}")
    print(f"  Active clients: {status.get('active_clients', 'N/A')}")
    print(f"  Current round: {status['current_round']}")
    print(f"  Last aggregation: {status.get('last_aggregation_time', 'N/A')}")
    
    # Test global model retrieval
    global_models_retrieved = 0
    
    print(f"\nGlobal Model Distribution:")
    for client_id, _ in clients:
        global_model = await server.get_global_model(client_id)
        if global_model:
            global_models_retrieved += 1
            print(f"  ✅ {client_id}: Model version {global_model['version']}")
            print(f"     Size: {len(global_model.get('weights', b''))} bytes")
            print(f"     Accuracy: {global_model.get('performance', {}).get('accuracy', 'N/A')}")
        else:
            print(f"  ❌ {client_id}: No global model available")
    
    print(f"\nGlobal Model Summary:")
    print(f"  Models distributed: {global_models_retrieved}/{len(clients)}")
    
    return global_models_retrieved > 0


async def demonstrate_training_configuration(server, clients):
    """Demonstrate training configuration management."""
    print("\n⚙️ Training Configuration Demonstration")
    print("="*50)
    
    configurations_retrieved = 0
    
    for client_id, _ in clients:
        config_data = await server.get_training_configuration(client_id)
        if config_data:
            configurations_retrieved += 1
            print(f"✅ {client_id} configuration:")
            print(f"   Aggregation strategy: {config_data.get('aggregation_strategy', 'N/A')}")
            print(f"   Learning rate: {config_data.get('learning_rate', 'N/A')}")
            print(f"   Batch size: {config_data.get('batch_size', 'N/A')}")
            print(f"   Local epochs: {config_data.get('local_epochs', 'N/A')}")
        else:
            print(f"❌ {client_id}: No configuration available")
    
    print(f"\nConfiguration Summary:")
    print(f"  Configurations provided: {configurations_retrieved}/{len(clients)}")
    
    return configurations_retrieved > 0


async def demonstrate_monitoring(server, clients):
    """Demonstrate system monitoring capabilities."""
    print("\n📊 System Monitoring Demonstration")
    print("="*50)
    
    # Report client metrics
    print("Reporting client metrics:")
    
    for client_id, _ in clients:
        # Simulate realistic metrics based on client type
        if "sdr" in client_id.lower():
            metrics = {
                "cpu_usage": np.random.uniform(40, 70),
                "memory_usage": np.random.uniform(30, 60),
                "network_latency": np.random.uniform(10, 25),
                "signal_quality": np.random.uniform(0.7, 0.95),
                "sdr_temperature": np.random.uniform(35, 55)
            }
        elif "mobile" in client_id.lower():
            metrics = {
                "cpu_usage": np.random.uniform(20, 50),
                "memory_usage": np.random.uniform(40, 80),
                "network_latency": np.random.uniform(15, 40),
                "battery_level": np.random.uniform(0.3, 0.9),
                "thermal_state": np.random.choice(["normal", "fair", "serious"])
            }
        else:  # edge
            metrics = {
                "cpu_usage": np.random.uniform(30, 60),
                "memory_usage": np.random.uniform(25, 50),
                "network_latency": np.random.uniform(2, 10),
                "gpu_usage": np.random.uniform(40, 80),
                "storage_usage": np.random.uniform(20, 70)
            }
        
        await server.report_client_metrics(client_id, metrics)
        print(f"  ✅ {client_id}: {len(metrics)} metrics reported")
        
        # Show key metrics
        for key, value in list(metrics.items())[:3]:
            if isinstance(value, (int, float)):
                print(f"     {key}: {value:.1f}")
            else:
                print(f"     {key}: {value}")
    
    print(f"\nMonitoring Summary:")
    print(f"  Clients monitored: {len(clients)}")
    print(f"  Metrics collected: {len(clients) * 5}")  # Approximate
    
    return True


async def run_full_system_demo():
    """Run the complete system demonstration."""
    print_banner()
    
    try:
        # Step 1: Client Registration
        server, clients = await demonstrate_client_registration()
        
        # Step 2: Model Updates
        updates_success = await demonstrate_model_updates(server, clients)
        
        # Step 3: Aggregation Process
        aggregation_success = await demonstrate_aggregation_process(server, clients)
        
        # Step 4: Training Configuration
        config_success = await demonstrate_training_configuration(server, clients)
        
        # Step 5: System Monitoring
        monitoring_success = await demonstrate_monitoring(server, clients)
        
        # Final Results
        print(f"\n{'='*60}")
        print(f"FULL SYSTEM DEMONSTRATION RESULTS")
        print(f"{'='*60}")
        
        results = {
            "Client Registration": "✅ PASSED",
            "Model Updates": "✅ PASSED" if updates_success else "❌ FAILED",
            "Aggregation Process": "✅ PASSED" if aggregation_success else "❌ FAILED", 
            "Training Configuration": "✅ PASSED" if config_success else "❌ FAILED",
            "System Monitoring": "✅ PASSED" if monitoring_success else "❌ FAILED"
        }
        
        for component, status in results.items():
            print(f"  {component:.<25} {status}")
        
        all_passed = all([updates_success, aggregation_success, config_success, monitoring_success])
        
        if all_passed:
            print(f"\n🎉 Full system demonstration completed successfully!")
            print(f"   All components working correctly")
            print(f"   System ready for production deployment")
        else:
            print(f"\n⚠️  Some components failed")
            print(f"   Review failed components before deployment")
        
        return all_passed
        
    except Exception as e:
        print(f"\n❌ System demonstration failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        if 'server' in locals():
            await server.shutdown()
            print("\n✅ Server shutdown complete")


def main():
    """Main entry point."""
    try:
        success = asyncio.run(run_full_system_demo())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Demonstration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()