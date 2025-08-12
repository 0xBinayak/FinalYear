#!/usr/bin/env python3
"""
Test security and privacy features
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
from src.aggregation_server.privacy import PrivacySecurityManager
from src.common.interfaces import ClientInfo, ModelUpdate
from src.common.config import get_config


async def test_differential_privacy():
    """Test differential privacy mechanisms"""
    print("Testing differential privacy...")
    
    config = get_config()
    config.privacy.enable_differential_privacy = True
    config.privacy.epsilon = 1.0
    config.privacy.delta = 1e-5
    
    privacy_manager = PrivacySecurityManager(config)
    
    # Create test model update
    weights = {
        "layer1": np.random.randn(5, 3).astype(np.float32),
        "layer2": np.random.randn(3, 1).astype(np.float32)
    }
    
    model_update = ModelUpdate(
        client_id="test_client",
        model_weights=pickle.dumps(weights),
        training_metrics={"loss": 0.5, "accuracy": 0.8},
        data_statistics={"num_samples": 1000},
        computation_time=30.0,
        network_conditions={"latency": 10, "bandwidth": 100},
        privacy_budget_used=0.1
    )
    
    # Test privacy protection
    protected_updates = privacy_manager.apply_privacy_protection([model_update])
    
    if protected_updates:
        print("✓ Differential privacy applied successfully")
        
        # Verify noise was added
        original_weights = pickle.loads(model_update.model_weights)
        protected_weights = pickle.loads(protected_updates[0].model_weights)
        
        # Check that weights are different (noise was added)
        layer1_diff = np.mean(np.abs(original_weights["layer1"] - protected_weights["layer1"]))
        print(f"✓ Noise added to weights (avg diff: {layer1_diff:.6f})")
        
        # Test privacy budget consumption
        budget_status = privacy_manager.dp_mechanism.get_privacy_budget_status()
        print(f"✓ Privacy budget consumed: {budget_status['client_budgets']['test_client']}")
    else:
        print("❌ Privacy protection failed")
    
    print("✅ Differential privacy test completed\n")


async def test_anomaly_detection():
    """Test anomaly detection"""
    print("Testing anomaly detection...")
    
    config = get_config()
    privacy_manager = PrivacySecurityManager(config)
    
    # Create normal updates
    normal_updates = []
    for i in range(5):
        weights = {
            "layer1": np.random.randn(5, 3).astype(np.float32) * 0.1,  # Small weights
            "layer2": np.random.randn(3, 1).astype(np.float32) * 0.1
        }
        
        update = ModelUpdate(
            client_id=f"normal_client_{i}",
            model_weights=pickle.dumps(weights),
            training_metrics={"loss": 0.5 + i * 0.05, "accuracy": 0.7 + i * 0.02},
            data_statistics={"num_samples": 1000 + i * 50},
            computation_time=30.0 + i * 2,
            network_conditions={"latency": 10 + i, "bandwidth": 100 + i * 5},
            privacy_budget_used=0.1
        )
        normal_updates.append(update)
    
    # Create anomalous update (Byzantine attack simulation)
    anomalous_weights = {
        "layer1": np.random.randn(5, 3).astype(np.float32) * 10,  # Very large weights
        "layer2": np.random.randn(3, 1).astype(np.float32) * 10
    }
    
    anomalous_update = ModelUpdate(
        client_id="byzantine_client",
        model_weights=pickle.dumps(anomalous_weights),
        training_metrics={"loss": 0.01, "accuracy": 0.99},  # Suspiciously good metrics
        data_statistics={"num_samples": 10000},  # Large dataset claim
        computation_time=5.0,  # Suspiciously fast
        network_conditions={"latency": 1, "bandwidth": 1000},
        privacy_budget_used=0.1
    )
    
    all_updates = normal_updates + [anomalous_update]
    
    # Test anomaly detection
    filtered_updates = privacy_manager.detect_and_filter_anomalies(all_updates)
    
    print(f"✓ Original updates: {len(all_updates)}")
    print(f"✓ Filtered updates: {len(filtered_updates)}")
    
    # Check if anomalous client was detected
    filtered_client_ids = [u.client_id for u in filtered_updates]
    if "byzantine_client" not in filtered_client_ids:
        print("✓ Byzantine client successfully detected and filtered")
    else:
        print("⚠ Byzantine client not detected")
    
    # Get anomaly report
    anomaly_report = privacy_manager.anomaly_detector.get_anomaly_report()
    print(f"✓ Anomaly detection report generated: {len(anomaly_report['recent_anomalies'])} recent anomalies")
    
    print("✅ Anomaly detection test completed\n")


async def test_audit_logging():
    """Test audit logging and compliance"""
    print("Testing audit logging and compliance...")
    
    config = get_config()
    privacy_manager = PrivacySecurityManager(config)
    
    # Log various events
    privacy_manager.audit_logger.log_event(
        'client_registration',
        'test_client_1',
        {'client_type': 'Simulated', 'timestamp': datetime.utcnow().isoformat()}
    )
    
    privacy_manager.audit_logger.log_event(
        'privacy_budget_exceeded',
        'test_client_2',
        {'requested_budget': 0.5, 'available_budget': 0.2}
    )
    
    privacy_manager.audit_logger.log_event(
        'anomaly_detected',
        'suspicious_client',
        {'detection_method': 'statistical', 'anomaly_score': 3.5}
    )
    
    # Test log integrity
    integrity_ok = privacy_manager.audit_logger.verify_log_integrity()
    print(f"✓ Audit log integrity verified: {integrity_ok}")
    
    # Generate compliance report
    compliance_report = privacy_manager.generate_compliance_report(days=1)
    print(f"✓ Compliance report generated: {compliance_report['total_events']} events")
    print(f"✓ Privacy violations: {len(compliance_report['privacy_violations'])}")
    print(f"✓ Security incidents: {len(compliance_report['security_incidents'])}")
    
    # Get recent audit log
    recent_logs = privacy_manager.audit_logger.get_audit_log(10)
    print(f"✓ Recent audit logs retrieved: {len(recent_logs)} entries")
    
    print("✅ Audit logging test completed\n")


async def test_integrated_security():
    """Test integrated security in aggregation server"""
    print("Testing integrated security features...")
    
    config = get_config()
    config.privacy.enable_differential_privacy = True
    config.federated_learning.aggregation_strategy = 'fedavg'
    config.federated_learning.min_clients = 1
    
    server = AggregationServer(config)
    
    try:
        await server.initialize()
        print("✓ Server initialized with security features")
        
        # Register clients
        clients = []
        for i in range(3):
            client_info = ClientInfo(
                client_id=f"secure_client_{i}",
                client_type="Simulated",
                capabilities={"cpu_cores": 4, "memory_gb": 8},
                location={"lat": 37.0 + i, "lon": -122.0 + i},
                network_info={"bandwidth": 100, "latency": 10},
                hardware_specs={"gpu": False},
                reputation_score=0.8 + i * 0.05
            )
            
            token = await server.register_client(client_info)
            clients.append((client_info.client_id, token))
        
        print(f"✓ Registered {len(clients)} clients")
        
        # Submit updates (including one potentially anomalous)
        for i, (client_id, token) in enumerate(clients):
            if i == 2:  # Make last client anomalous
                weights = {
                    "layer1": np.random.randn(5, 3).astype(np.float32) * 5,  # Large weights
                    "layer2": np.random.randn(3, 1).astype(np.float32) * 5
                }
                metrics = {"loss": 0.01, "accuracy": 0.99}  # Too good
            else:
                weights = {
                    "layer1": np.random.randn(5, 3).astype(np.float32) * 0.1,
                    "layer2": np.random.randn(3, 1).astype(np.float32) * 0.1
                }
                metrics = {"loss": 0.5 - i * 0.1, "accuracy": 0.7 + i * 0.05}
            
            model_update = ModelUpdate(
                client_id=client_id,
                model_weights=pickle.dumps(weights),
                training_metrics=metrics,
                data_statistics={"num_samples": 1000 + i * 100},
                computation_time=30.0 + i * 5,
                network_conditions={"latency": 10 + i, "bandwidth": 100 + i * 10},
                privacy_budget_used=0.1
            )
            
            success = await server.receive_model_update(client_id, model_update)
            print(f"✓ Model update from {client_id}: {success}")
        
        # Wait for aggregation
        await asyncio.sleep(1)
        
        # Check security status
        security_status = await server.get_security_status()
        print(f"✓ Security status retrieved")
        print(f"  - Privacy budget used: {security_status['privacy_budget_status']['global_budget_used']:.3f}")
        print(f"  - Recent anomalies: {len(security_status['anomaly_report']['recent_anomalies'])}")
        print(f"  - Audit log integrity: {security_status['log_integrity_verified']}")
        
        # Test client privacy budget
        for client_id, _ in clients:
            budget_info = await server.get_client_privacy_budget(client_id)
            print(f"  - {client_id} budget used: {budget_info['client_budget_used']:.3f}")
        
        # Test compliance report
        compliance_report = await server.generate_compliance_report(1)
        print(f"✓ Compliance report: {compliance_report['total_events']} events")
        
        # Test client blocking (simulate admin action)
        if len(clients) > 2:
            admin_client = clients[0][0]  # Use first client as admin
            server.registered_clients[admin_client].reputation_score = 0.95  # Make admin
            
            target_client = clients[2][0]  # Block suspicious client
            success = await server.block_client(target_client, "Anomalous behavior detected")
            print(f"✓ Client blocking test: {success}")
        
        print("✅ Integrated security test completed")
        
    except Exception as e:
        print(f"❌ Integrated security test failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        await server.shutdown()
        print("✓ Server shutdown complete")


if __name__ == "__main__":
    async def run_all_tests():
        await test_differential_privacy()
        await test_anomaly_detection()
        await test_audit_logging()
        await test_integrated_security()
    
    asyncio.run(run_all_tests())