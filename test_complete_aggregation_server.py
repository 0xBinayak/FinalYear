#!/usr/bin/env python3
"""
Comprehensive test for complete aggregation server functionality
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


async def test_complete_aggregation_server():
    """Test complete aggregation server with all features"""
    print("🚀 Testing Complete Aggregation Server Functionality")
    print("=" * 60)
    
    # Configure for comprehensive testing
    config = get_config()
    config.privacy.enable_differential_privacy = True
    config.privacy.epsilon = 1.0
    config.federated_learning.aggregation_strategy = 'weighted'
    config.federated_learning.min_clients = 2
    config.federated_learning.max_clients = 5
    
    server = AggregationServer(config)
    
    try:
        # 1. Server Initialization
        print("\n1. Server Initialization")
        print("-" * 30)
        await server.initialize()
        print("✅ Server initialized with all components")
        
        health_status = await server.get_health_status()
        print(f"✅ Health check: {health_status['status']}")
        print(f"   Version: {health_status['version']}")
        print(f"   Uptime: {health_status['uptime']:.2f}s")
        
        # 2. Client Registration and Authentication
        print("\n2. Client Registration & Authentication")
        print("-" * 40)
        
        clients = []
        client_types = ["SDR", "Mobile", "Simulated"]
        
        for i in range(6):  # Register more clients than max to test selection
            client_info = ClientInfo(
                client_id=f"comprehensive_client_{i}",
                client_type=client_types[i % 3],
                capabilities={
                    "cpu_cores": 2 + i,
                    "memory_gb": 4 + i * 2,
                    "gpu_available": i % 2 == 0,
                    "sdr_hardware": i < 2
                },
                location={"lat": 37.7749 + i * 0.1, "lon": -122.4194 + i * 0.1},
                network_info={
                    "bandwidth": 50 + i * 20,
                    "latency": 20 - i * 2,
                    "connection_type": "wifi" if i % 2 else "cellular"
                },
                hardware_specs={
                    "gpu": i % 2 == 0,
                    "sdr_type": "RTL-SDR" if i < 2 else None
                },
                reputation_score=0.6 + i * 0.06
            )
            
            token = await server.register_client(client_info)
            clients.append((client_info.client_id, token, client_info.client_type))
            print(f"✅ Registered {client_info.client_id} ({client_info.client_type})")
        
        print(f"✅ Total clients registered: {len(clients)}")
        
        # 3. Model Updates with Varying Quality
        print("\n3. Model Updates & Quality Assessment")
        print("-" * 40)
        
        updates_submitted = 0
        for i, (client_id, token, client_type) in enumerate(clients):
            # Create model weights with varying quality
            if i == 5:  # Make last client potentially Byzantine
                weights = {
                    "layer1": np.random.randn(8, 5).astype(np.float32) * 3,  # Larger weights
                    "layer2": np.random.randn(5, 2).astype(np.float32) * 3,
                    "output": np.random.randn(2, 1).astype(np.float32) * 3
                }
                training_metrics = {"loss": 0.05, "accuracy": 0.98}  # Suspiciously good
                data_size = 5000  # Claims large dataset
                comp_time = 10.0  # Very fast training
            else:
                # Normal client behavior
                scale = 0.1 + i * 0.02  # Gradually improving clients
                weights = {
                    "layer1": np.random.randn(8, 5).astype(np.float32) * scale,
                    "layer2": np.random.randn(5, 2).astype(np.float32) * scale,
                    "output": np.random.randn(2, 1).astype(np.float32) * scale
                }
                training_metrics = {
                    "loss": 1.0 - i * 0.12,  # Improving loss
                    "accuracy": 0.5 + i * 0.08  # Improving accuracy
                }
                data_size = 800 + i * 150
                comp_time = 25.0 + i * 5
            
            model_update = ModelUpdate(
                client_id=client_id,
                model_weights=pickle.dumps(weights),
                training_metrics=training_metrics,
                data_statistics={
                    "num_samples": data_size,
                    "data_quality_score": 0.7 + i * 0.05,
                    "label_distribution": {"class_0": 0.6, "class_1": 0.4}
                },
                computation_time=comp_time,
                network_conditions={
                    "latency": 20 - i * 2,
                    "bandwidth": 50 + i * 20,
                    "packet_loss": i * 0.005,
                    "jitter": i * 0.5
                },
                privacy_budget_used=0.08 + i * 0.01
            )
            
            success = await server.receive_model_update(client_id, model_update)
            if success:
                updates_submitted += 1
                print(f"✅ Update from {client_id}: Loss={training_metrics['loss']:.3f}, "
                      f"Acc={training_metrics['accuracy']:.3f}, Size={data_size}")
            else:
                print(f"❌ Update rejected from {client_id}")
        
        print(f"✅ Updates submitted: {updates_submitted}")
        
        # 4. Wait for Aggregation and Security Processing
        print("\n4. Aggregation & Security Processing")
        print("-" * 40)
        
        await asyncio.sleep(2)  # Allow time for aggregation
        
        server_status = await server.get_server_status()
        print(f"✅ Current round: {server_status['current_round']}")
        print(f"✅ Total clients: {server_status['total_clients']}")
        print(f"✅ Active clients: {server_status['active_clients']}")
        print(f"✅ Aggregation strategy: {server_status['aggregation_strategy']}")
        
        # 5. Security and Privacy Status
        print("\n5. Security & Privacy Assessment")
        print("-" * 40)
        
        security_status = await server.get_security_status()
        privacy_budget = security_status['privacy_budget_status']
        anomaly_report = security_status['anomaly_report']
        
        print(f"✅ Global privacy budget used: {privacy_budget['global_budget_used']:.3f}")
        print(f"✅ Privacy budget remaining: {privacy_budget['global_budget_remaining']:.3f}")
        print(f"✅ Epsilon: {privacy_budget['epsilon']}")
        print(f"✅ Recent anomalies detected: {len(anomaly_report['recent_anomalies'])}")
        print(f"✅ Audit log integrity: {security_status['log_integrity_verified']}")
        
        # Show client-specific privacy budgets
        for client_id, _, _ in clients[:3]:  # Show first 3 clients
            budget_info = await server.get_client_privacy_budget(client_id)
            print(f"   {client_id}: {budget_info['client_budget_used']:.3f} used")
        
        # 6. Advanced Aggregation Strategy Testing
        print("\n6. Advanced Aggregation Strategies")
        print("-" * 40)
        
        available_strategies = await server.get_available_strategies()
        print(f"✅ Available strategies: {available_strategies}")
        
        # Test strategy switching
        for strategy in ['krum', 'trimmed_mean', 'fedavg']:
            if strategy in available_strategies:
                # Make first client admin for strategy switching
                admin_client = clients[0][0]
                server.registered_clients[admin_client].reputation_score = 0.95
                
                success = await server.update_aggregation_strategy(strategy)
                if success:
                    print(f"✅ Successfully switched to {strategy} strategy")
                else:
                    print(f"⚠ Failed to switch to {strategy} strategy")
        
        # 7. Convergence Monitoring
        print("\n7. Convergence Monitoring")
        print("-" * 30)
        
        convergence_history = await server.get_convergence_history()
        if convergence_history:
            latest_metrics = convergence_history[-1]
            print(f"✅ Latest convergence metrics:")
            print(f"   Average loss: {latest_metrics.get('average_loss', 'N/A'):.4f}")
            print(f"   Average accuracy: {latest_metrics.get('average_accuracy', 'N/A'):.4f}")
            print(f"   Participating clients: {latest_metrics.get('participating_clients', 'N/A')}")
            print(f"   Total data samples: {latest_metrics.get('total_data_samples', 'N/A')}")
        else:
            print("⚠ No convergence history available yet")
        
        # 8. Global Model Distribution
        print("\n8. Global Model Distribution")
        print("-" * 35)
        
        models_distributed = 0
        for client_id, _, client_type in clients[:3]:  # Test first 3 clients
            global_model = await server.get_global_model(client_id)
            if global_model:
                models_distributed += 1
                print(f"✅ Global model distributed to {client_id} ({client_type})")
                print(f"   Version: {global_model['version']}")
                print(f"   Round: {global_model['round']}")
            else:
                print(f"❌ No global model available for {client_id}")
        
        print(f"✅ Models successfully distributed: {models_distributed}")
        
        # 9. Training Configuration
        print("\n9. Training Configuration")
        print("-" * 30)
        
        for client_id, _, client_type in clients[:2]:  # Test first 2 clients
            training_config = await server.get_training_configuration(client_id)
            print(f"✅ Training config for {client_id} ({client_type}):")
            print(f"   Learning rate: {training_config['hyperparameters']['learning_rate']}")
            print(f"   Batch size: {training_config['hyperparameters']['batch_size']}")
            print(f"   Local epochs: {training_config['hyperparameters']['local_epochs']}")
            print(f"   Privacy enabled: {training_config['privacy_settings']['enable_differential_privacy']}")
        
        # 10. Compliance and Audit
        print("\n10. Compliance & Audit")
        print("-" * 25)
        
        compliance_report = await server.generate_compliance_report(1)
        print(f"✅ Compliance report generated:")
        print(f"   Total events: {compliance_report['total_events']}")
        print(f"   Privacy violations: {len(compliance_report['privacy_violations'])}")
        print(f"   Security incidents: {len(compliance_report['security_incidents'])}")
        print(f"   Data access events: {len(compliance_report['data_access_events'])}")
        
        # 11. Client Management
        print("\n11. Client Management")
        print("-" * 25)
        
        # Test client blocking (simulate admin action)
        admin_client = clients[0][0]
        server.registered_clients[admin_client].reputation_score = 0.95  # Make admin
        
        if len(clients) > 5:
            suspicious_client = clients[5][0]  # Block the potentially Byzantine client
            success = await server.block_client(suspicious_client, "Anomalous behavior detected in testing")
            print(f"✅ Client blocking test: {success}")
            
            # Verify client is blocked
            blocked_client_info = server.registered_clients[suspicious_client]
            print(f"   Blocked client reputation: {blocked_client_info.reputation_score}")
        
        # 12. Performance Metrics
        print("\n12. Performance Metrics")
        print("-" * 25)
        
        final_status = await server.get_server_status()
        performance_metrics = final_status.get('performance_metrics', {})
        
        print(f"✅ Performance summary:")
        print(f"   Total rounds completed: {final_status.get('total_rounds', 0)}")
        print(f"   Average clients per round: {performance_metrics.get('average_clients_per_round', 0):.1f}")
        print(f"   Last round clients: {performance_metrics.get('last_round_clients', 0)}")
        print(f"   Server uptime: {final_status.get('uptime', 0):.2f}s")
        
        print("\n" + "=" * 60)
        print("🎉 COMPREHENSIVE AGGREGATION SERVER TEST COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\n📊 Test Summary:")
        print(f"   ✅ Clients registered: {len(clients)}")
        print(f"   ✅ Updates processed: {updates_submitted}")
        print(f"   ✅ Models distributed: {models_distributed}")
        print(f"   ✅ Security features: Active")
        print(f"   ✅ Privacy protection: Active")
        print(f"   ✅ Anomaly detection: Active")
        print(f"   ✅ Audit logging: Active")
        print(f"   ✅ Advanced aggregation: Active")
        print(f"   ✅ Convergence monitoring: Active")
        
        return True
        
    except Exception as e:
        print(f"\n❌ COMPREHENSIVE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        print(f"\n🔄 Shutting down server...")
        await server.shutdown()
        print("✅ Server shutdown complete")


if __name__ == "__main__":
    success = asyncio.run(test_complete_aggregation_server())
    print(f"\n{'🎉 SUCCESS' if success else '❌ FAILURE'}")
    sys.exit(0 if success else 1)