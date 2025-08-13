#!/usr/bin/env python3
"""
Security and privacy demonstration for the Advanced Federated Pipeline.
Shows differential privacy, anomaly detection, audit logging, and security features.
"""

import asyncio
import sys
import os
import pickle
import numpy as np
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))

from src.aggregation_server.server import AggregationServer
from src.aggregation_server.privacy import PrivacySecurityManager
from src.common.interfaces import ClientInfo, ModelUpdate
from src.common.config import get_config


def print_banner():
    """Print demonstration banner."""
    print("""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║              SECURITY & PRIVACY DEMONSTRATION                                ║
    ║                                                                              ║
    ║  Advanced Privacy-Preserving Federated Learning                             ║
    ║  • Differential Privacy                                                      ║
    ║  • Anomaly Detection                                                         ║
    ║  • Byzantine Fault Tolerance                                                 ║
    ║  • Audit Logging                                                             ║
    ║  • Compliance Reporting                                                      ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """)


async def demonstrate_differential_privacy():
    """Demonstrate differential privacy mechanisms."""
    print("\n🔒 Differential Privacy Demonstration")
    print("="*50)
    
    config = get_config()
    config.privacy.enable_differential_privacy = True
    config.privacy.epsilon = 1.0
    config.privacy.delta = 1e-5
    
    privacy_manager = PrivacySecurityManager(config)
    
    print(f"Privacy Parameters:")
    print(f"  • Epsilon (ε): {config.privacy.epsilon}")
    print(f"  • Delta (δ): {config.privacy.delta}")
    print(f"  • Mechanism: Gaussian noise")
    
    # Create test model updates
    test_clients = ["client_A", "client_B", "client_C"]
    model_updates = []
    
    for client_id in test_clients:
        weights = {
            "layer1": np.random.randn(5, 3).astype(np.float32),
            "layer2": np.random.randn(3, 1).astype(np.float32)
        }
        
        model_update = ModelUpdate(
            client_id=client_id,
            model_weights=pickle.dumps(weights),
            training_metrics={"loss": np.random.uniform(0.3, 0.7), "accuracy": np.random.uniform(0.7, 0.9)},
            data_statistics={"num_samples": np.random.randint(800, 1200)},
            computation_time=np.random.uniform(25, 45),
            network_conditions={"latency": np.random.uniform(5, 20), "bandwidth": np.random.uniform(50, 200)},
            privacy_budget_used=0.1
        )
        model_updates.append(model_update)
    
    print(f"\nApplying differential privacy to {len(model_updates)} model updates...")
    
    # Apply privacy protection
    protected_updates = privacy_manager.apply_privacy_protection(model_updates)
    
    if protected_updates:
        print("✅ Differential privacy applied successfully")
        
        # Analyze privacy protection
        for i, (original, protected) in enumerate(zip(model_updates, protected_updates)):
            original_weights = pickle.loads(original.model_weights)
            protected_weights = pickle.loads(protected.model_weights)
            
            # Calculate noise magnitude
            layer1_noise = np.mean(np.abs(original_weights["layer1"] - protected_weights["layer1"]))
            layer2_noise = np.mean(np.abs(original_weights["layer2"] - protected_weights["layer2"]))
            
            print(f"  Client {test_clients[i]}:")
            print(f"    • Layer 1 noise magnitude: {layer1_noise:.6f}")
            print(f"    • Layer 2 noise magnitude: {layer2_noise:.6f}")
        
        # Check privacy budget consumption
        budget_status = privacy_manager.dp_mechanism.get_privacy_budget_status()
        print(f"\nPrivacy Budget Status:")
        for client_id in test_clients:
            budget_used = budget_status['client_budgets'].get(client_id, 0)
            print(f"  • {client_id}: {budget_used:.3f} / {config.privacy.epsilon:.1f}")
        
        print(f"  • Global budget used: {budget_status['global_budget_used']:.3f}")
        
    else:
        print("❌ Privacy protection failed")
    
    return len(protected_updates) == len(model_updates)


async def demonstrate_anomaly_detection():
    """Demonstrate anomaly detection and Byzantine fault tolerance."""
    print("\n🛡️ Anomaly Detection Demonstration")
    print("="*50)
    
    config = get_config()
    privacy_manager = PrivacySecurityManager(config)
    
    print("Creating normal and anomalous model updates...")
    
    # Create normal updates
    normal_updates = []
    for i in range(5):
        weights = {
            "layer1": np.random.randn(5, 3).astype(np.float32) * 0.1,  # Normal magnitude
            "layer2": np.random.randn(3, 1).astype(np.float32) * 0.1
        }
        
        update = ModelUpdate(
            client_id=f"honest_client_{i}",
            model_weights=pickle.dumps(weights),
            training_metrics={
                "loss": 0.4 + np.random.normal(0, 0.05),  # Normal loss
                "accuracy": 0.8 + np.random.normal(0, 0.02)  # Normal accuracy
            },
            data_statistics={"num_samples": 1000 + np.random.randint(-100, 100)},
            computation_time=30.0 + np.random.normal(0, 5),
            network_conditions={"latency": 10 + np.random.randint(-3, 3), "bandwidth": 100 + np.random.randint(-20, 20)},
            privacy_budget_used=0.1
        )
        normal_updates.append(update)
    
    # Create different types of anomalous updates
    anomalous_updates = []
    
    # Type 1: Model poisoning attack (large weights)
    poisoning_weights = {
        "layer1": np.random.randn(5, 3).astype(np.float32) * 10,  # Very large weights
        "layer2": np.random.randn(3, 1).astype(np.float32) * 10
    }
    
    poisoning_update = ModelUpdate(
        client_id="poisoning_attacker",
        model_weights=pickle.dumps(poisoning_weights),
        training_metrics={"loss": 0.45, "accuracy": 0.82},  # Seems normal
        data_statistics={"num_samples": 1000},
        computation_time=30.0,
        network_conditions={"latency": 10, "bandwidth": 100},
        privacy_budget_used=0.1
    )
    anomalous_updates.append(("Model Poisoning", poisoning_update))
    
    # Type 2: Data poisoning attack (suspicious metrics)
    data_poisoning_weights = {
        "layer1": np.random.randn(5, 3).astype(np.float32) * 0.1,
        "layer2": np.random.randn(3, 1).astype(np.float32) * 0.1
    }
    
    data_poisoning_update = ModelUpdate(
        client_id="data_poisoning_attacker",
        model_weights=pickle.dumps(data_poisoning_weights),
        training_metrics={"loss": 0.01, "accuracy": 0.99},  # Suspiciously good
        data_statistics={"num_samples": 10000},  # Claims large dataset
        computation_time=5.0,  # Suspiciously fast
        network_conditions={"latency": 1, "bandwidth": 1000},
        privacy_budget_used=0.1
    )
    anomalous_updates.append(("Data Poisoning", data_poisoning_update))
    
    # Type 3: Gradient inversion attack (unusual patterns)
    inversion_weights = {
        "layer1": np.ones((5, 3), dtype=np.float32) * 0.5,  # Uniform weights
        "layer2": np.zeros((3, 1), dtype=np.float32)  # Zero weights
    }
    
    inversion_update = ModelUpdate(
        client_id="gradient_inversion_attacker",
        model_weights=pickle.dumps(inversion_weights),
        training_metrics={"loss": 0.5, "accuracy": 0.75},
        data_statistics={"num_samples": 500},  # Small dataset
        computation_time=60.0,  # Slow computation
        network_conditions={"latency": 50, "bandwidth": 10},
        privacy_budget_used=0.1
    )
    anomalous_updates.append(("Gradient Inversion", inversion_update))
    
    # Combine all updates
    all_updates = normal_updates + [update for _, update in anomalous_updates]
    
    print(f"Testing anomaly detection on {len(all_updates)} updates:")
    print(f"  • Normal updates: {len(normal_updates)}")
    print(f"  • Anomalous updates: {len(anomalous_updates)}")
    
    # Test anomaly detection
    filtered_updates = privacy_manager.detect_and_filter_anomalies(all_updates)
    
    print(f"\nAnomaly Detection Results:")
    print(f"  • Original updates: {len(all_updates)}")
    print(f"  • Filtered updates: {len(filtered_updates)}")
    print(f"  • Detected anomalies: {len(all_updates) - len(filtered_updates)}")
    
    # Check which anomalies were detected
    filtered_client_ids = {u.client_id for u in filtered_updates}
    detected_attacks = []
    
    for attack_type, update in anomalous_updates:
        if update.client_id not in filtered_client_ids:
            detected_attacks.append(attack_type)
            print(f"  ✅ Detected: {attack_type} ({update.client_id})")
        else:
            print(f"  ❌ Missed: {attack_type} ({update.client_id})")
    
    # Get detailed anomaly report
    anomaly_report = privacy_manager.anomaly_detector.get_anomaly_report()
    print(f"\nDetailed Anomaly Report:")
    print(f"  • Recent anomalies: {len(anomaly_report['recent_anomalies'])}")
    print(f"  • Detection methods used: {', '.join(anomaly_report['detection_methods'])}")
    
    for anomaly in anomaly_report['recent_anomalies'][:3]:  # Show first 3
        print(f"    - Client: {anomaly['client_id']}")
        print(f"      Type: {anomaly['anomaly_type']}")
        print(f"      Score: {anomaly['anomaly_score']:.2f}")
        print(f"      Reason: {anomaly['reason']}")
    
    detection_rate = len(detected_attacks) / len(anomalous_updates)
    print(f"\nDetection Rate: {detection_rate:.1%}")
    
    return detection_rate > 0.5  # Expect at least 50% detection rate


async def demonstrate_audit_logging():
    """Demonstrate audit logging and compliance features."""
    print("\n📋 Audit Logging & Compliance Demonstration")
    print("="*50)
    
    config = get_config()
    privacy_manager = PrivacySecurityManager(config)
    
    print("Generating audit events...")
    
    # Simulate various system events
    events = [
        ('client_registration', 'mobile_client_001', {'client_type': 'Mobile', 'location': 'New York'}),
        ('client_registration', 'sdr_client_001', {'client_type': 'SDR', 'location': 'San Francisco'}),
        ('model_update_received', 'mobile_client_001', {'model_size': 2048, 'accuracy': 0.85}),
        ('model_update_received', 'sdr_client_001', {'model_size': 1024, 'accuracy': 0.78}),
        ('privacy_budget_consumed', 'mobile_client_001', {'budget_used': 0.15, 'remaining': 0.85}),
        ('anomaly_detected', 'suspicious_client_999', {'anomaly_type': 'model_poisoning', 'score': 3.2}),
        ('client_blocked', 'suspicious_client_999', {'reason': 'Repeated anomalous behavior'}),
        ('aggregation_completed', 'system', {'round': 42, 'participants': 8, 'accuracy': 0.89}),
        ('privacy_budget_exceeded', 'greedy_client_123', {'requested': 0.5, 'available': 0.2}),
        ('compliance_violation', 'non_compliant_client', {'violation_type': 'data_retention', 'severity': 'medium'})
    ]
    
    # Log all events
    for event_type, client_id, details in events:
        details['timestamp'] = datetime.utcnow().isoformat()
        privacy_manager.audit_logger.log_event(event_type, client_id, details)
    
    print(f"✅ Logged {len(events)} audit events")
    
    # Test log integrity
    integrity_ok = privacy_manager.audit_logger.verify_log_integrity()
    print(f"✅ Audit log integrity verified: {integrity_ok}")
    
    # Generate compliance report
    compliance_report = privacy_manager.generate_compliance_report(days=1)
    
    print(f"\nCompliance Report Summary:")
    print(f"  • Total events: {compliance_report['total_events']}")
    print(f"  • Privacy violations: {len(compliance_report['privacy_violations'])}")
    print(f"  • Security incidents: {len(compliance_report['security_incidents'])}")
    print(f"  • Client registrations: {compliance_report['client_registrations']}")
    print(f"  • Model updates processed: {compliance_report['model_updates_processed']}")
    
    # Show privacy violations
    if compliance_report['privacy_violations']:
        print(f"\nPrivacy Violations:")
        for violation in compliance_report['privacy_violations'][:3]:
            print(f"  • Client: {violation['client_id']}")
            print(f"    Type: {violation['violation_type']}")
            print(f"    Severity: {violation['severity']}")
    
    # Show security incidents
    if compliance_report['security_incidents']:
        print(f"\nSecurity Incidents:")
        for incident in compliance_report['security_incidents'][:3]:
            print(f"  • Client: {incident['client_id']}")
            print(f"    Type: {incident['incident_type']}")
            print(f"    Action taken: {incident.get('action_taken', 'None')}")
    
    # Get recent audit logs
    recent_logs = privacy_manager.audit_logger.get_audit_log(5)
    print(f"\nRecent Audit Log Entries:")
    for log_entry in recent_logs:
        print(f"  • {log_entry['timestamp']}: {log_entry['event_type']} - {log_entry['client_id']}")
    
    return len(recent_logs) > 0


async def demonstrate_integrated_security():
    """Demonstrate integrated security in the aggregation server."""
    print("\n🔐 Integrated Security Demonstration")
    print("="*50)
    
    config = get_config()
    config.privacy.enable_differential_privacy = True
    config.federated_learning.aggregation_strategy = 'fedavg'
    config.federated_learning.min_clients = 1
    
    server = AggregationServer(config)
    
    try:
        await server.initialize()
        print("✅ Server initialized with security features enabled")
        
        # Register clients with different security profiles
        clients = []
        client_profiles = [
            {"id": "trusted_client", "reputation": 0.95, "type": "Enterprise"},
            {"id": "normal_client", "reputation": 0.80, "type": "Individual"},
            {"id": "suspicious_client", "reputation": 0.60, "type": "Unknown"}
        ]
        
        for profile in client_profiles:
            client_info = ClientInfo(
                client_id=profile["id"],
                client_type=profile["type"],
                capabilities={"cpu_cores": 4, "memory_gb": 8},
                location={"lat": 37.0, "lon": -122.0},
                network_info={"bandwidth": 100, "latency": 10},
                hardware_specs={"gpu": False},
                reputation_score=profile["reputation"]
            )
            
            token = await server.register_client(client_info)
            clients.append((client_info.client_id, token, profile["reputation"]))
            print(f"✅ Registered {profile['type']} client: {profile['id']} (reputation: {profile['reputation']:.2f})")
        
        # Submit model updates with varying security characteristics
        for i, (client_id, token, reputation) in enumerate(clients):
            if reputation < 0.7:  # Make suspicious client anomalous
                weights = {
                    "layer1": np.random.randn(5, 3).astype(np.float32) * 5,  # Large weights
                    "layer2": np.random.randn(3, 1).astype(np.float32) * 5
                }
                metrics = {"loss": 0.01, "accuracy": 0.99}  # Too good to be true
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
            print(f"✅ Model update from {client_id}: {'Accepted' if success else 'Rejected'}")
        
        # Wait for processing
        await asyncio.sleep(2)
        
        # Check comprehensive security status
        security_status = await server.get_security_status()
        
        print(f"\nSecurity Status Report:")
        print(f"  • Privacy budget used: {security_status['privacy_budget_status']['global_budget_used']:.3f}")
        print(f"  • Recent anomalies: {len(security_status['anomaly_report']['recent_anomalies'])}")
        print(f"  • Log integrity verified: {security_status['log_integrity_verified']}")
        print(f"  • Active security measures: {', '.join(security_status['active_security_measures'])}")
        
        # Check individual client privacy budgets
        print(f"\nClient Privacy Budget Status:")
        for client_id, _, reputation in clients:
            budget_info = await server.get_client_privacy_budget(client_id)
            print(f"  • {client_id}: {budget_info['client_budget_used']:.3f} / {budget_info['client_budget_limit']:.1f}")
            print(f"    Reputation: {reputation:.2f}, Status: {budget_info['status']}")
        
        # Generate comprehensive compliance report
        compliance_report = await server.generate_compliance_report(1)
        
        print(f"\nCompliance Report:")
        print(f"  • Total events: {compliance_report['total_events']}")
        print(f"  • Privacy compliance: {'✅ PASS' if compliance_report['privacy_compliant'] else '❌ FAIL'}")
        print(f"  • Security compliance: {'✅ PASS' if compliance_report['security_compliant'] else '❌ FAIL'}")
        print(f"  • Audit trail complete: {'✅ YES' if compliance_report['audit_trail_complete'] else '❌ NO'}")
        
        # Test administrative security actions
        print(f"\nTesting Administrative Security Actions:")
        
        # Block suspicious client
        suspicious_client = "suspicious_client"
        block_success = await server.block_client(suspicious_client, "Anomalous behavior detected")
        print(f"  • Client blocking: {'✅ SUCCESS' if block_success else '❌ FAILED'}")
        
        # Reset privacy budget (admin action)
        reset_success = await server.reset_client_privacy_budget(suspicious_client)
        print(f"  • Privacy budget reset: {'✅ SUCCESS' if reset_success else '❌ FAILED'}")
        
        # Generate security incident report
        incident_report = await server.generate_security_incident_report()
        print(f"  • Security incidents: {len(incident_report['incidents'])}")
        
        print(f"\n✅ Integrated security demonstration completed successfully")
        return True
        
    except Exception as e:
        print(f"❌ Integrated security test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        await server.shutdown()
        print("✅ Server shutdown complete")


async def run_security_demo():
    """Run the complete security and privacy demonstration."""
    print_banner()
    
    demonstrations = [
        ("Differential Privacy", demonstrate_differential_privacy),
        ("Anomaly Detection", demonstrate_anomaly_detection),
        ("Audit Logging & Compliance", demonstrate_audit_logging),
        ("Integrated Security", demonstrate_integrated_security)
    ]
    
    passed = 0
    total = len(demonstrations)
    
    for demo_name, demo_func in demonstrations:
        try:
            print(f"\n{'='*60}")
            success = await demo_func()
            if success:
                passed += 1
                print(f"✅ {demo_name} demonstration completed successfully")
            else:
                print(f"❌ {demo_name} demonstration failed")
        except Exception as e:
            print(f"❌ {demo_name} demonstration failed with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*60}")
    print(f"SECURITY & PRIVACY DEMONSTRATION RESULTS")
    print(f"{'='*60}")
    print(f"Completed: {passed}/{total}")
    print(f"Success rate: {passed/total*100:.1f}%")
    
    if passed == total:
        print("🎉 All security demonstrations completed successfully!")
        print("🔒 System is ready for secure federated learning deployment")
        return True
    else:
        print("⚠️  Some security demonstrations failed")
        print("🔧 Review security configuration before deployment")
        return False


def main():
    """Main entry point."""
    try:
        success = asyncio.run(run_security_demo())
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\n⏹️  Demonstration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()