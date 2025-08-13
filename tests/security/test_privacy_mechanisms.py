"""
Automated privacy validation tests for federated learning system.
"""
import pytest
import numpy as np
import pickle
from unittest.mock import Mock, patch
from datetime import datetime

from src.aggregation_server.privacy import DifferentialPrivacyManager, PrivacyBudgetTracker
from src.aggregation_server.server import AggregationServer
from src.common.interfaces import ClientInfo, ModelUpdate


@pytest.mark.privacy
@pytest.mark.security
class TestDifferentialPrivacy:
    """Test differential privacy mechanisms."""
    
    def test_noise_addition(self, privacy_test_data):
        """Test noise addition for differential privacy."""
        dp_manager = DifferentialPrivacyManager(epsilon=1.0, delta=1e-5)
        
        # Test Gaussian noise addition
        original_data = privacy_test_data["sensitive_data"]
        noisy_data = dp_manager.add_gaussian_noise(original_data, sensitivity=1.0)
        
        assert noisy_data.shape == original_data.shape
        assert not np.array_equal(noisy_data, original_data)
        
        # Verify noise magnitude is reasonable
        noise = noisy_data - original_data
        noise_std = np.std(noise)
        expected_std = dp_manager.gaussian_noise_scale(sensitivity=1.0)
        assert abs(noise_std - expected_std) < 0.1 * expected_std
    
    def test_laplace_mechanism(self, privacy_test_data):
        """Test Laplace mechanism for differential privacy."""
        dp_manager = DifferentialPrivacyManager(epsilon=1.0, delta=1e-5)
        
        # Test scalar query
        true_count = 1000
        private_count = dp_manager.laplace_mechanism(true_count, sensitivity=1.0)
        
        assert isinstance(private_count, (int, float))
        # Should be reasonably close to true value
        assert abs(private_count - true_count) < 100
    
    def test_privacy_budget_tracking(self):
        """Test privacy budget tracking and enforcement."""
        budget_tracker = PrivacyBudgetTracker(total_epsilon=5.0, total_delta=1e-4)
        
        # Test budget allocation
        client_id = "test_client"
        allocated = budget_tracker.allocate_budget(client_id, epsilon=1.0, delta=1e-5)
        assert allocated
        
        # Test budget consumption
        consumed = budget_tracker.consume_budget(client_id, epsilon=0.5, delta=5e-6)
        assert consumed
        
        # Check remaining budget
        remaining = budget_tracker.get_remaining_budget(client_id)
        assert remaining["epsilon"] == 0.5
        assert remaining["delta"] == 5e-6
        
        # Test budget exhaustion
        exhausted = budget_tracker.consume_budget(client_id, epsilon=1.0, delta=1e-5)
        assert not exhausted  # Should fail due to insufficient budget
    
    def test_model_weight_privatization(self):
        """Test privatization of model weights."""
        dp_manager = DifferentialPrivacyManager(epsilon=1.0, delta=1e-5)
        
        # Create sample model weights
        weights = {
            "layer1": np.random.randn(10, 5).astype(np.float32),
            "layer2": np.random.randn(5, 1).astype(np.float32)
        }
        
        # Privatize weights
        private_weights = dp_manager.privatize_model_weights(weights, sensitivity=0.1)
        
        assert set(private_weights.keys()) == set(weights.keys())
        for layer in weights:
            assert private_weights[layer].shape == weights[layer].shape
            assert not np.array_equal(private_weights[layer], weights[layer])
    
    def test_gradient_clipping(self):
        """Test gradient clipping for privacy."""
        dp_manager = DifferentialPrivacyManager(epsilon=1.0, delta=1e-5)
        
        # Create gradients with varying magnitudes
        gradients = {
            "layer1": np.random.randn(10, 5).astype(np.float32) * 10,  # Large gradients
            "layer2": np.random.randn(5, 1).astype(np.float32) * 0.1   # Small gradients
        }
        
        # Clip gradients
        clipped_gradients = dp_manager.clip_gradients(gradients, max_norm=1.0)
        
        # Verify clipping
        for layer in gradients:
            layer_norm = np.linalg.norm(clipped_gradients[layer])
            assert layer_norm <= 1.0 + 1e-6  # Allow small numerical error
    
    def test_privacy_accounting(self):
        """Test privacy accounting across multiple queries."""
        dp_manager = DifferentialPrivacyManager(epsilon=2.0, delta=1e-4)
        
        # Simulate multiple queries
        queries = [
            {"epsilon": 0.5, "delta": 1e-5, "mechanism": "gaussian"},
            {"epsilon": 0.3, "delta": 2e-5, "mechanism": "laplace"},
            {"epsilon": 0.7, "delta": 3e-5, "mechanism": "gaussian"}
        ]
        
        total_epsilon = 0
        total_delta = 0
        
        for query in queries:
            # Record query
            dp_manager.record_query(query)
            total_epsilon += query["epsilon"]
            total_delta += query["delta"]
        
        # Check accounting
        accounting = dp_manager.get_privacy_accounting()
        assert accounting["total_epsilon"] == total_epsilon
        assert accounting["total_delta"] == total_delta
        assert accounting["num_queries"] == len(queries)
    
    async def test_end_to_end_privacy(self, test_config, privacy_test_data):
        """Test end-to-end privacy in federated learning."""
        # Enable differential privacy
        test_config.privacy.differential_privacy_enabled = True
        test_config.privacy.epsilon = 1.0
        test_config.privacy.delta = 1e-5
        
        server = AggregationServer(test_config)
        await server.initialize()
        
        try:
            # Register client
            client_info = ClientInfo(
                client_id="privacy_test_client",
                client_type="Simulated",
                capabilities={"cpu_cores": 4, "memory_gb": 8},
                location={"lat": 37.0, "lon": -122.0},
                network_info={"bandwidth": 100, "latency": 10},
                hardware_specs={},
                reputation_score=1.0
            )
            
            await server.register_client(client_info)
            
            # Submit model update
            weights = {
                "layer1": np.random.randn(20, 10).astype(np.float32),
                "layer2": np.random.randn(10, 5).astype(np.float32)
            }
            
            model_update = ModelUpdate(
                client_id=client_info.client_id,
                model_weights=pickle.dumps(weights),
                training_metrics={"loss": 0.5, "accuracy": 0.8},
                data_statistics={"num_samples": 1000},
                computation_time=30.0,
                network_conditions={"latency": 10, "bandwidth": 100},
                privacy_budget_used=0.5
            )
            
            success = await server.receive_model_update(client_info.client_id, model_update)
            assert success
            
            # Wait for aggregation
            await asyncio.sleep(1.0)
            
            # Retrieve global model
            global_model = await server.get_global_model(client_info.client_id)
            assert global_model is not None
            
            # Verify privacy was applied
            global_weights = pickle.loads(global_model["weights"])
            assert not np.array_equal(global_weights["layer1"], weights["layer1"])
            assert not np.array_equal(global_weights["layer2"], weights["layer2"])
            
            # Check privacy budget consumption
            privacy_status = await server.get_privacy_status(client_info.client_id)
            assert privacy_status["budget_used"] > 0
            assert privacy_status["budget_remaining"] < test_config.privacy.epsilon
            
        finally:
            await server.shutdown()


@pytest.mark.privacy
@pytest.mark.security
class TestPrivacyAttackDefense:
    """Test defense against privacy attacks."""
    
    def test_membership_inference_defense(self):
        """Test defense against membership inference attacks."""
        dp_manager = DifferentialPrivacyManager(epsilon=1.0, delta=1e-5)
        
        # Simulate training data
        member_data = np.random.randn(100, 10)
        non_member_data = np.random.randn(100, 10)
        
        # Simulate model outputs with and without privacy
        def simulate_model_output(data, private=False):
            # Simple simulation of model confidence scores
            scores = np.random.rand(len(data))
            if private:
                # Add noise to reduce distinguishability
                noise = dp_manager.add_gaussian_noise(scores, sensitivity=0.1)
                scores = np.clip(scores + noise, 0, 1)
            return scores
        
        # Test without privacy
        member_scores = simulate_model_output(member_data, private=False)
        non_member_scores = simulate_model_output(non_member_data, private=False)
        
        # Test with privacy
        private_member_scores = simulate_model_output(member_data, private=True)
        private_non_member_scores = simulate_model_output(non_member_data, private=True)
        
        # Privacy should reduce distinguishability
        non_private_diff = abs(np.mean(member_scores) - np.mean(non_member_scores))
        private_diff = abs(np.mean(private_member_scores) - np.mean(private_non_member_scores))
        
        assert private_diff < non_private_diff
    
    def test_model_inversion_defense(self):
        """Test defense against model inversion attacks."""
        dp_manager = DifferentialPrivacyManager(epsilon=1.0, delta=1e-5)
        
        # Simulate sensitive features
        sensitive_features = np.random.randn(1000, 50)
        
        # Simulate model gradients
        gradients = np.random.randn(50, 10)
        
        # Apply privacy protection
        protected_gradients = dp_manager.privatize_gradients(gradients, sensitivity=1.0)
        
        # Verify gradients are modified
        assert not np.array_equal(gradients, protected_gradients)
        
        # Simulate inversion attempt
        def attempt_inversion(grads):
            # Simplified inversion simulation
            return np.random.randn(50)  # Random reconstruction
        
        original_reconstruction = attempt_inversion(gradients)
        private_reconstruction = attempt_inversion(protected_gradients)
        
        # Both should be equally poor reconstructions due to privacy
        assert np.linalg.norm(original_reconstruction) > 0
        assert np.linalg.norm(private_reconstruction) > 0
    
    def test_property_inference_defense(self):
        """Test defense against property inference attacks."""
        dp_manager = DifferentialPrivacyManager(epsilon=1.0, delta=1e-5)
        
        # Simulate dataset with sensitive property
        dataset_with_property = np.random.randn(1000, 20)
        dataset_without_property = np.random.randn(1000, 20) + 0.5  # Slight shift
        
        # Simulate model statistics
        def compute_statistics(data, private=False):
            stats = {
                "mean": np.mean(data, axis=0),
                "std": np.std(data, axis=0),
                "correlation": np.corrcoef(data.T)
            }
            
            if private:
                # Add noise to statistics
                stats["mean"] = dp_manager.add_gaussian_noise(stats["mean"], sensitivity=0.1)
                stats["std"] = dp_manager.add_gaussian_noise(stats["std"], sensitivity=0.1)
            
            return stats
        
        # Compute statistics with and without privacy
        stats_with = compute_statistics(dataset_with_property, private=True)
        stats_without = compute_statistics(dataset_without_property, private=True)
        
        # Privacy should make it harder to distinguish datasets
        mean_diff = np.linalg.norm(stats_with["mean"] - stats_without["mean"])
        assert mean_diff > 0  # Some difference should remain
        
        # But the difference should be bounded by privacy guarantees
        expected_noise_scale = dp_manager.gaussian_noise_scale(sensitivity=0.1)
        assert mean_diff < 5 * expected_noise_scale  # Reasonable bound


@pytest.mark.security
class TestSecurityMechanisms:
    """Test security mechanisms and attack detection."""
    
    def test_byzantine_client_detection(self):
        """Test detection of Byzantine (malicious) clients."""
        from src.aggregation_server.aggregation import ByzantineDetector
        
        detector = ByzantineDetector()
        
        # Create normal model updates
        normal_updates = []
        for i in range(8):
            weights = {
                "layer1": np.random.randn(5, 3).astype(np.float32),
                "layer2": np.random.randn(3, 1).astype(np.float32)
            }
            normal_updates.append({
                "client_id": f"normal_client_{i}",
                "weights": weights,
                "metrics": {"loss": 0.5 + np.random.randn() * 0.1}
            })
        
        # Create Byzantine updates (extreme values)
        byzantine_updates = []
        for i in range(2):
            weights = {
                "layer1": np.random.randn(5, 3).astype(np.float32) * 100,  # Extreme values
                "layer2": np.random.randn(3, 1).astype(np.float32) * 100
            }
            byzantine_updates.append({
                "client_id": f"byzantine_client_{i}",
                "weights": weights,
                "metrics": {"loss": 10.0}  # Suspiciously high loss
            })
        
        all_updates = normal_updates + byzantine_updates
        
        # Test detection
        suspected_byzantine = detector.detect_byzantine_clients(all_updates)
        
        assert len(suspected_byzantine) > 0
        # Should detect at least one Byzantine client
        byzantine_client_ids = [u["client_id"] for u in byzantine_updates]
        detected_byzantine = [client_id for client_id in suspected_byzantine 
                            if client_id in byzantine_client_ids]
        assert len(detected_byzantine) > 0
    
    def test_gradient_anomaly_detection(self):
        """Test detection of anomalous gradients."""
        from src.aggregation_server.aggregation import GradientAnomalyDetector
        
        detector = GradientAnomalyDetector()
        
        # Create normal gradients
        normal_gradients = []
        for i in range(10):
            gradients = {
                "layer1": np.random.randn(10, 5).astype(np.float32) * 0.1,
                "layer2": np.random.randn(5, 1).astype(np.float32) * 0.1
            }
            normal_gradients.append(gradients)
        
        # Create anomalous gradients
        anomalous_gradients = {
            "layer1": np.random.randn(10, 5).astype(np.float32) * 10,  # Much larger
            "layer2": np.random.randn(5, 1).astype(np.float32) * 10
        }
        
        # Test detection
        is_anomalous = detector.is_anomalous(anomalous_gradients, normal_gradients)
        assert is_anomalous
        
        # Test normal gradient
        is_normal = detector.is_anomalous(normal_gradients[0], normal_gradients[1:])
        assert not is_normal
    
    def test_model_poisoning_detection(self):
        """Test detection of model poisoning attacks."""
        from src.aggregation_server.aggregation import ModelPoisoningDetector
        
        detector = ModelPoisoningDetector()
        
        # Simulate clean model updates
        clean_updates = []
        for i in range(5):
            weights = {
                "layer1": np.random.randn(8, 4).astype(np.float32) * 0.5,
                "layer2": np.random.randn(4, 2).astype(np.float32) * 0.5
            }
            clean_updates.append({
                "client_id": f"clean_client_{i}",
                "weights": weights,
                "training_metrics": {"loss": 0.4 + np.random.randn() * 0.05}
            })
        
        # Simulate poisoned model update
        poisoned_weights = {
            "layer1": np.random.randn(8, 4).astype(np.float32) * 2.0,  # Different scale
            "layer2": np.random.randn(4, 2).astype(np.float32) * 2.0
        }
        poisoned_update = {
            "client_id": "poisoned_client",
            "weights": poisoned_weights,
            "training_metrics": {"loss": 0.2}  # Suspiciously low loss
        }
        
        # Test detection
        is_poisoned = detector.detect_poisoning(poisoned_update, clean_updates)
        assert is_poisoned
        
        # Test clean update
        is_clean = detector.detect_poisoning(clean_updates[0], clean_updates[1:])
        assert not is_clean
    
    def test_secure_aggregation_simulation(self):
        """Test secure multi-party computation simulation."""
        from src.aggregation_server.privacy import SecureAggregationManager
        
        secure_agg = SecureAggregationManager()
        
        # Simulate client secret shares
        num_clients = 5
        model_size = (10, 5)
        
        client_updates = []
        for i in range(num_clients):
            update = np.random.randn(*model_size).astype(np.float32)
            client_updates.append(update)
        
        # Create secret shares
        shares = []
        for update in client_updates:
            client_shares = secure_agg.create_secret_shares(update, num_clients)
            shares.append(client_shares)
        
        # Simulate secure aggregation
        aggregated_result = secure_agg.secure_aggregate(shares)
        
        # Verify result matches direct aggregation
        direct_aggregate = np.mean(client_updates, axis=0)
        
        # Should be close (allowing for numerical precision)
        assert np.allclose(aggregated_result, direct_aggregate, rtol=1e-5)
    
    def test_audit_logging(self):
        """Test security audit logging."""
        from src.aggregation_server.auth import SecurityAuditor
        
        auditor = SecurityAuditor()
        
        # Test various security events
        events = [
            {"type": "client_registration", "client_id": "test_client", "timestamp": datetime.now()},
            {"type": "model_update", "client_id": "test_client", "size": 1024, "timestamp": datetime.now()},
            {"type": "anomaly_detected", "client_id": "suspicious_client", "reason": "extreme_gradients", "timestamp": datetime.now()},
            {"type": "privacy_budget_exhausted", "client_id": "test_client", "timestamp": datetime.now()}
        ]
        
        # Log events
        for event in events:
            auditor.log_security_event(event)
        
        # Retrieve audit log
        audit_log = auditor.get_audit_log()
        assert len(audit_log) == len(events)
        
        # Test filtering
        anomaly_events = auditor.get_audit_log(event_type="anomaly_detected")
        assert len(anomaly_events) == 1
        assert anomaly_events[0]["reason"] == "extreme_gradients"
        
        # Test compliance report generation
        compliance_report = auditor.generate_compliance_report()
        assert "total_events" in compliance_report
        assert "event_types" in compliance_report
        assert compliance_report["total_events"] == len(events)