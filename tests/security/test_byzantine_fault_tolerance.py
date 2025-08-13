"""
Comprehensive tests for Byzantine fault tolerance in federated learning.
"""
import pytest
import asyncio
import numpy as np
import pickle
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta
import random

from src.aggregation_server.server import AggregationServer
from src.aggregation_server.aggregation import ByzantineDetector, RobustAggregator
from src.common.interfaces import ClientInfo, ModelUpdate


@pytest.mark.security
@pytest.mark.byzantine
class TestByzantineDetection:
    """Test Byzantine client detection mechanisms."""
    
    def test_statistical_outlier_detection(self):
        """Test detection of Byzantine clients using statistical methods."""
        detector = ByzantineDetector()
        
        # Create normal model updates
        normal_updates = []
        for i in range(10):
            weights = {
                "layer1": np.random.randn(20, 10).astype(np.float32) * 0.1,  # Small weights
                "layer2": np.random.randn(10, 5).astype(np.float32) * 0.1
            }
            
            update = {
                "client_id": f"honest_client_{i}",
                "weights": weights,
                "training_metrics": {
                    "loss": 0.5 + np.random.randn() * 0.05,  # Normal loss around 0.5
                    "accuracy": 0.8 + np.random.randn() * 0.02  # Normal accuracy around 0.8
                },
                "data_statistics": {"num_samples": 1000 + np.random.randint(-100, 100)}
            }
            normal_updates.append(update)
        
        # Create Byzantine updates with extreme values
        byzantine_updates = []
        for i in range(3):
            weights = {
                "layer1": np.random.randn(20, 10).astype(np.float32) * 10,  # Extreme weights
                "layer2": np.random.randn(10, 5).astype(np.float32) * 10
            }
            
            update = {
                "client_id": f"byzantine_client_{i}",
                "weights": weights,
                "training_metrics": {
                    "loss": 0.01,  # Suspiciously low loss
                    "accuracy": 0.99  # Suspiciously high accuracy
                },
                "data_statistics": {"num_samples": 10000}  # Suspiciously large dataset
            }
            byzantine_updates.append(update)
        
        all_updates = normal_updates + byzantine_updates
        
        # Test detection
        suspected_byzantine = detector.detect_statistical_outliers(all_updates)
        
        # Should detect most Byzantine clients
        byzantine_client_ids = [u["client_id"] for u in byzantine_updates]
        detected_byzantine = [client_id for client_id in suspected_byzantine 
                            if client_id in byzantine_client_ids]
        
        assert len(detected_byzantine) >= 2  # Should detect at least 2 out of 3
        
        # Should not flag too many honest clients
        honest_client_ids = [u["client_id"] for u in normal_updates]
        false_positives = [client_id for client_id in suspected_byzantine 
                          if client_id in honest_client_ids]
        
        assert len(false_positives) <= 1  # At most 1 false positive
    
    def test_gradient_norm_analysis(self):
        """Test Byzantine detection based on gradient norm analysis."""
        detector = ByzantineDetector()
        
        # Normal gradients with reasonable norms
        normal_gradients = []
        for i in range(15):
            gradients = {
                "layer1": np.random.randn(30, 15).astype(np.float32) * 0.01,
                "layer2": np.random.randn(15, 8).astype(np.float32) * 0.01
            }
            normal_gradients.append({
                "client_id": f"normal_client_{i}",
                "gradients": gradients
            })
        
        # Byzantine gradients with extreme norms
        byzantine_gradients = []
        for i in range(4):
            gradients = {
                "layer1": np.random.randn(30, 15).astype(np.float32) * 5,  # Very large gradients
                "layer2": np.random.randn(15, 8).astype(np.float32) * 5
            }
            byzantine_gradients.append({
                "client_id": f"byzantine_client_{i}",
                "gradients": gradients
            })
        
        all_gradients = normal_gradients + byzantine_gradients
        
        # Analyze gradient norms
        suspected_clients = detector.analyze_gradient_norms(all_gradients, threshold=3.0)
        
        # Should detect Byzantine clients with extreme gradients
        byzantine_ids = [g["client_id"] for g in byzantine_gradients]
        detected_byzantine = [client_id for client_id in suspected_clients 
                            if client_id in byzantine_ids]
        
        assert len(detected_byzantine) >= 3  # Should detect most Byzantine clients
    
    def test_cosine_similarity_detection(self):
        """Test Byzantine detection using cosine similarity analysis."""
        detector = ByzantineDetector()
        
        # Create updates with similar directions (honest clients)
        base_direction = np.random.randn(100)
        base_direction = base_direction / np.linalg.norm(base_direction)
        
        honest_updates = []
        for i in range(12):
            # Add small random perturbations to base direction
            direction = base_direction + np.random.randn(100) * 0.1
            direction = direction / np.linalg.norm(direction)
            
            weights = {"layer1": direction.reshape(10, 10).astype(np.float32)}
            
            honest_updates.append({
                "client_id": f"honest_client_{i}",
                "weights": weights
            })
        
        # Create Byzantine updates with opposite/random directions
        byzantine_updates = []
        for i in range(3):
            if i == 0:
                # Opposite direction
                direction = -base_direction + np.random.randn(100) * 0.1
            else:
                # Random direction
                direction = np.random.randn(100)
            
            direction = direction / np.linalg.norm(direction)
            weights = {"layer1": direction.reshape(10, 10).astype(np.float32)}
            
            byzantine_updates.append({
                "client_id": f"byzantine_client_{i}",
                "weights": weights
            })
        
        all_updates = honest_updates + byzantine_updates
        
        # Test cosine similarity detection
        suspected_clients = detector.detect_cosine_similarity_outliers(all_updates, threshold=0.3)
        
        # Should detect Byzantine clients with dissimilar directions
        byzantine_ids = [u["client_id"] for u in byzantine_updates]
        detected_byzantine = [client_id for client_id in suspected_clients 
                            if client_id in byzantine_ids]
        
        assert len(detected_byzantine) >= 2  # Should detect most Byzantine clients
    
    def test_temporal_consistency_analysis(self):
        """Test Byzantine detection based on temporal consistency."""
        detector = ByzantineDetector()
        
        # Simulate multiple rounds of updates
        rounds_data = []
        
        for round_num in range(5):
            round_updates = []
            
            # Honest clients with consistent behavior
            for i in range(8):
                # Gradual improvement in metrics
                loss = 1.0 - round_num * 0.1 + np.random.randn() * 0.02
                accuracy = 0.6 + round_num * 0.05 + np.random.randn() * 0.01
                
                weights = {
                    "layer1": np.random.randn(15, 8).astype(np.float32) * 0.1,
                    "layer2": np.random.randn(8, 4).astype(np.float32) * 0.1
                }
                
                update = {
                    "client_id": f"consistent_client_{i}",
                    "weights": weights,
                    "training_metrics": {"loss": loss, "accuracy": accuracy},
                    "round": round_num
                }
                round_updates.append(update)
            
            # Byzantine clients with inconsistent behavior
            for i in range(2):
                # Erratic metrics
                if round_num % 2 == 0:
                    loss = 0.1  # Suspiciously good
                    accuracy = 0.95
                else:
                    loss = 2.0  # Suspiciously bad
                    accuracy = 0.3
                
                weights = {
                    "layer1": np.random.randn(15, 8).astype(np.float32) * (5 if round_num % 2 == 0 else 0.01),
                    "layer2": np.random.randn(8, 4).astype(np.float32) * (5 if round_num % 2 == 0 else 0.01)
                }
                
                update = {
                    "client_id": f"inconsistent_client_{i}",
                    "weights": weights,
                    "training_metrics": {"loss": loss, "accuracy": accuracy},
                    "round": round_num
                }
                round_updates.append(update)
            
            rounds_data.append(round_updates)
        
        # Analyze temporal consistency
        suspected_clients = detector.analyze_temporal_consistency(rounds_data)
        
        # Should detect inconsistent Byzantine clients
        byzantine_ids = ["inconsistent_client_0", "inconsistent_client_1"]
        detected_byzantine = [client_id for client_id in suspected_clients 
                            if client_id in byzantine_ids]
        
        assert len(detected_byzantine) >= 1  # Should detect at least one inconsistent client
    
    def test_reputation_based_detection(self):
        """Test Byzantine detection using reputation scores."""
        detector = ByzantineDetector()
        
        # Initialize client reputations
        client_reputations = {}
        
        # Honest clients start with good reputation
        for i in range(10):
            client_reputations[f"honest_client_{i}"] = 0.9
        
        # Byzantine clients start with average reputation
        for i in range(3):
            client_reputations[f"byzantine_client_{i}"] = 0.8
        
        # Simulate multiple rounds where Byzantine clients behave badly
        for round_num in range(10):
            round_updates = []
            
            # Honest clients submit normal updates
            for i in range(10):
                weights = {
                    "layer1": np.random.randn(12, 6).astype(np.float32) * 0.1,
                    "layer2": np.random.randn(6, 3).astype(np.float32) * 0.1
                }
                
                update = {
                    "client_id": f"honest_client_{i}",
                    "weights": weights,
                    "training_metrics": {"loss": 0.5, "accuracy": 0.8},
                    "is_malicious": False
                }
                round_updates.append(update)
            
            # Byzantine clients submit malicious updates
            for i in range(3):
                weights = {
                    "layer1": np.random.randn(12, 6).astype(np.float32) * 3,  # Extreme weights
                    "layer2": np.random.randn(6, 3).astype(np.float32) * 3
                }
                
                update = {
                    "client_id": f"byzantine_client_{i}",
                    "weights": weights,
                    "training_metrics": {"loss": 0.01, "accuracy": 0.99},  # Suspicious metrics
                    "is_malicious": True
                }
                round_updates.append(update)
            
            # Update reputations based on behavior
            client_reputations = detector.update_reputations(round_updates, client_reputations)
        
        # Test reputation-based detection
        low_reputation_clients = detector.detect_low_reputation_clients(
            client_reputations, threshold=0.5
        )
        
        # Byzantine clients should have low reputation
        byzantine_ids = [f"byzantine_client_{i}" for i in range(3)]
        detected_byzantine = [client_id for client_id in low_reputation_clients 
                            if client_id in byzantine_ids]
        
        assert len(detected_byzantine) >= 2  # Should detect most Byzantine clients
        
        # Honest clients should maintain good reputation
        honest_ids = [f"honest_client_{i}" for i in range(10)]
        false_positives = [client_id for client_id in low_reputation_clients 
                          if client_id in honest_ids]
        
        assert len(false_positives) <= 1  # Minimal false positives


@pytest.mark.security
@pytest.mark.byzantine
class TestRobustAggregation:
    """Test robust aggregation algorithms against Byzantine attacks."""
    
    def test_krum_aggregation(self):
        """Test Krum aggregation algorithm."""
        aggregator = RobustAggregator(algorithm="krum")
        
        # Create honest updates (clustered around similar values)
        honest_updates = []
        base_weights = {
            "layer1": np.ones((10, 5), dtype=np.float32),
            "layer2": np.ones((5, 2), dtype=np.float32) * 0.5
        }
        
        for i in range(10):
            # Add small random noise to base weights
            weights = {}
            for layer, base_weight in base_weights.items():
                weights[layer] = base_weight + np.random.randn(*base_weight.shape).astype(np.float32) * 0.1
            
            honest_updates.append({
                "client_id": f"honest_client_{i}",
                "weights": weights,
                "num_samples": 1000
            })
        
        # Create Byzantine updates (extreme values)
        byzantine_updates = []
        for i in range(4):  # 4 Byzantine out of 14 total (< 1/3)
            weights = {
                "layer1": np.ones((10, 5), dtype=np.float32) * 10,  # Extreme values
                "layer2": np.ones((5, 2), dtype=np.float32) * -10
            }
            
            byzantine_updates.append({
                "client_id": f"byzantine_client_{i}",
                "weights": weights,
                "num_samples": 1000
            })
        
        all_updates = honest_updates + byzantine_updates
        
        # Test Krum aggregation
        aggregated_weights = aggregator.aggregate(all_updates)
        
        # Aggregated weights should be close to honest clients' weights
        assert aggregated_weights is not None
        
        # Check that aggregated weights are reasonable (not extreme)
        layer1_mean = np.mean(aggregated_weights["layer1"])
        layer2_mean = np.mean(aggregated_weights["layer2"])
        
        assert 0.5 <= layer1_mean <= 1.5  # Should be close to 1.0
        assert 0.2 <= layer2_mean <= 0.8  # Should be close to 0.5
        
        # Should not be influenced by Byzantine extreme values
        assert np.abs(layer1_mean - 10) > 5  # Far from Byzantine values
        assert np.abs(layer2_mean - (-10)) > 5
    
    def test_trimmed_mean_aggregation(self):
        """Test Trimmed Mean aggregation algorithm."""
        aggregator = RobustAggregator(algorithm="trimmed_mean", trim_ratio=0.2)
        
        # Create updates with some extreme values
        updates = []
        
        # Normal updates
        for i in range(8):
            weights = {
                "layer1": np.ones((8, 4), dtype=np.float32) * (2.0 + np.random.randn() * 0.1),
                "layer2": np.ones((4, 2), dtype=np.float32) * (1.0 + np.random.randn() * 0.1)
            }
            
            updates.append({
                "client_id": f"normal_client_{i}",
                "weights": weights,
                "num_samples": 1000
            })
        
        # Extreme updates (will be trimmed)
        for i in range(2):
            weights = {
                "layer1": np.ones((8, 4), dtype=np.float32) * 20,  # Very high
                "layer2": np.ones((4, 2), dtype=np.float32) * (-10)  # Very low
            }
            
            updates.append({
                "client_id": f"extreme_client_{i}",
                "weights": weights,
                "num_samples": 1000
            })
        
        # Test trimmed mean aggregation
        aggregated_weights = aggregator.aggregate(updates)
        
        # Should trim extreme values and average the rest
        layer1_mean = np.mean(aggregated_weights["layer1"])
        layer2_mean = np.mean(aggregated_weights["layer2"])
        
        # Should be close to normal values, not extreme ones
        assert 1.5 <= layer1_mean <= 2.5  # Close to 2.0
        assert 0.5 <= layer2_mean <= 1.5  # Close to 1.0
    
    def test_median_aggregation(self):
        """Test Median aggregation algorithm."""
        aggregator = RobustAggregator(algorithm="median")
        
        # Create updates with various values
        updates = []
        values = [0.5, 1.0, 1.5, 2.0, 2.5, 100.0, -50.0]  # Include outliers
        
        for i, value in enumerate(values):
            weights = {
                "layer1": np.ones((6, 3), dtype=np.float32) * value,
                "layer2": np.ones((3, 2), dtype=np.float32) * (value * 0.5)
            }
            
            updates.append({
                "client_id": f"client_{i}",
                "weights": weights,
                "num_samples": 1000
            })
        
        # Test median aggregation
        aggregated_weights = aggregator.aggregate(updates)
        
        # Median should be robust to outliers
        layer1_median = np.median(aggregated_weights["layer1"])
        layer2_median = np.median(aggregated_weights["layer2"])
        
        # Should be close to median of normal values (around 2.0)
        assert 1.8 <= layer1_median <= 2.2
        assert 0.9 <= layer2_median <= 1.1
    
    def test_coordinate_wise_median(self):
        """Test coordinate-wise median aggregation."""
        aggregator = RobustAggregator(algorithm="coordinate_median")
        
        # Create updates where different coordinates have different outliers
        updates = []
        
        for i in range(7):
            layer1 = np.ones((4, 3), dtype=np.float32)
            layer2 = np.ones((3, 2), dtype=np.float32)
            
            # Add outliers to specific coordinates
            if i == 0:  # Outlier in first coordinate
                layer1[0, 0] = 100.0
            elif i == 1:  # Outlier in different coordinate
                layer1[1, 1] = -50.0
            elif i == 2:  # Outlier in layer2
                layer2[0, 0] = 200.0
            
            weights = {"layer1": layer1, "layer2": layer2}
            
            updates.append({
                "client_id": f"client_{i}",
                "weights": weights,
                "num_samples": 1000
            })
        
        # Test coordinate-wise median
        aggregated_weights = aggregator.aggregate(updates)
        
        # Each coordinate should be the median, robust to outliers
        assert aggregated_weights is not None
        
        # Most coordinates should be close to 1.0
        layer1_values = aggregated_weights["layer1"].flatten()
        normal_coords = [v for v in layer1_values if 0.5 <= v <= 1.5]
        assert len(normal_coords) >= len(layer1_values) * 0.8  # Most coordinates normal
    
    def test_byzantine_resilient_aggregation_performance(self):
        """Test performance of Byzantine-resilient aggregation under attack."""
        # Test different algorithms under various attack scenarios
        algorithms = ["krum", "trimmed_mean", "median", "coordinate_median"]
        attack_scenarios = [
            {"name": "model_poisoning", "byzantine_ratio": 0.3},
            {"name": "sign_flipping", "byzantine_ratio": 0.25},
            {"name": "gaussian_attack", "byzantine_ratio": 0.2}
        ]
        
        results = {}
        
        for algorithm in algorithms:
            results[algorithm] = {}
            aggregator = RobustAggregator(algorithm=algorithm)
            
            for scenario in attack_scenarios:
                # Create honest updates
                honest_updates = []
                num_honest = int(20 * (1 - scenario["byzantine_ratio"]))
                
                for i in range(num_honest):
                    weights = {
                        "layer1": np.random.randn(15, 8).astype(np.float32) * 0.1,
                        "layer2": np.random.randn(8, 4).astype(np.float32) * 0.1
                    }
                    
                    honest_updates.append({
                        "client_id": f"honest_{i}",
                        "weights": weights,
                        "num_samples": 1000
                    })
                
                # Create Byzantine updates based on attack type
                byzantine_updates = []
                num_byzantine = 20 - num_honest
                
                for i in range(num_byzantine):
                    if scenario["name"] == "model_poisoning":
                        # Extreme weight values
                        weights = {
                            "layer1": np.random.randn(15, 8).astype(np.float32) * 5,
                            "layer2": np.random.randn(8, 4).astype(np.float32) * 5
                        }
                    elif scenario["name"] == "sign_flipping":
                        # Flip signs of honest updates
                        honest_weights = honest_updates[i % len(honest_updates)]["weights"]
                        weights = {
                            "layer1": -honest_weights["layer1"] * 2,
                            "layer2": -honest_weights["layer2"] * 2
                        }
                    elif scenario["name"] == "gaussian_attack":
                        # Gaussian noise attack
                        weights = {
                            "layer1": np.random.randn(15, 8).astype(np.float32) * 2,
                            "layer2": np.random.randn(8, 4).astype(np.float32) * 2
                        }
                    
                    byzantine_updates.append({
                        "client_id": f"byzantine_{i}",
                        "weights": weights,
                        "num_samples": 1000
                    })
                
                all_updates = honest_updates + byzantine_updates
                
                # Test aggregation
                try:
                    aggregated_weights = aggregator.aggregate(all_updates)
                    success = aggregated_weights is not None
                    
                    if success:
                        # Measure how close aggregated weights are to honest average
                        honest_avg = {}
                        for layer in ["layer1", "layer2"]:
                            honest_weights_layer = [u["weights"][layer] for u in honest_updates]
                            honest_avg[layer] = np.mean(honest_weights_layer, axis=0)
                        
                        # Calculate distance from honest average
                        distance = 0
                        for layer in ["layer1", "layer2"]:
                            distance += np.linalg.norm(aggregated_weights[layer] - honest_avg[layer])
                        
                        robustness_score = 1.0 / (1.0 + distance)  # Higher is better
                    else:
                        robustness_score = 0.0
                
                except Exception:
                    success = False
                    robustness_score = 0.0
                
                results[algorithm][scenario["name"]] = {
                    "success": success,
                    "robustness_score": robustness_score,
                    "byzantine_ratio": scenario["byzantine_ratio"]
                }
        
        # Verify that robust algorithms perform well
        for algorithm in algorithms:
            for scenario_name, result in results[algorithm].items():
                assert result["success"], f"{algorithm} failed on {scenario_name}"
                assert result["robustness_score"] > 0.1, f"{algorithm} not robust enough on {scenario_name}"
        
        return results


@pytest.mark.security
@pytest.mark.byzantine
class TestByzantineAttackScenarios:
    """Test specific Byzantine attack scenarios."""
    
    async def test_model_poisoning_attack(self, test_config):
        """Test defense against model poisoning attacks."""
        # Enable Byzantine-robust aggregation
        test_config.federated_learning.aggregation_strategy = "krum"
        test_config.security.byzantine_detection_enabled = True
        
        server = AggregationServer(test_config)
        await server.initialize()
        
        try:
            # Register honest clients
            honest_clients = []
            for i in range(12):
                client_info = ClientInfo(
                    client_id=f"honest_client_{i}",
                    client_type="Simulated",
                    capabilities={"cpu_cores": 4, "memory_gb": 8},
                    location={"lat": 37.0, "lon": -122.0},
                    network_info={"bandwidth": 100, "latency": 10},
                    hardware_specs={},
                    reputation_score=0.95
                )
                await server.register_client(client_info)
                honest_clients.append(client_info)
            
            # Register Byzantine clients
            byzantine_clients = []
            for i in range(4):  # 25% Byzantine clients
                client_info = ClientInfo(
                    client_id=f"byzantine_client_{i}",
                    client_type="Simulated",
                    capabilities={"cpu_cores": 4, "memory_gb": 8},
                    location={"lat": 37.0, "lon": -122.0},
                    network_info={"bandwidth": 100, "latency": 10},
                    hardware_specs={},
                    reputation_score=0.9
                )
                await server.register_client(client_info)
                byzantine_clients.append(client_info)
            
            # Execute federated learning round with model poisoning
            update_tasks = []
            
            # Honest clients submit normal updates
            for client_info in honest_clients:
                weights = {
                    "layer1": np.random.randn(20, 10).astype(np.float32) * 0.1,
                    "layer2": np.random.randn(10, 5).astype(np.float32) * 0.1
                }
                
                update = ModelUpdate(
                    client_id=client_info.client_id,
                    model_weights=pickle.dumps(weights),
                    training_metrics={"loss": 0.5, "accuracy": 0.8},
                    data_statistics={"num_samples": 1000},
                    computation_time=30.0,
                    network_conditions={"latency": 10, "bandwidth": 100},
                    privacy_budget_used=0.1
                )
                
                task = server.receive_model_update(client_info.client_id, update)
                update_tasks.append(task)
            
            # Byzantine clients submit poisoned updates
            for client_info in byzantine_clients:
                # Model poisoning: extreme weight values
                weights = {
                    "layer1": np.random.randn(20, 10).astype(np.float32) * 10,  # 100x larger
                    "layer2": np.random.randn(10, 5).astype(np.float32) * 10
                }
                
                update = ModelUpdate(
                    client_id=client_info.client_id,
                    model_weights=pickle.dumps(weights),
                    training_metrics={"loss": 0.01, "accuracy": 0.99},  # Suspicious metrics
                    data_statistics={"num_samples": 1000},
                    computation_time=5.0,  # Suspiciously fast
                    network_conditions={"latency": 10, "bandwidth": 100},
                    privacy_budget_used=0.1
                )
                
                task = server.receive_model_update(client_info.client_id, update)
                update_tasks.append(task)
            
            # Wait for all updates
            results = await asyncio.gather(*update_tasks, return_exceptions=True)
            
            # Wait for aggregation and Byzantine detection
            await asyncio.sleep(3.0)
            
            # Verify system handled the attack
            successful_updates = sum(1 for result in results if result is True)
            assert successful_updates >= len(honest_clients)  # At least honest clients succeeded
            
            # Check if Byzantine clients were detected
            security_status = await server.get_security_status()
            detected_byzantine = security_status.get("detected_byzantine_clients", [])
            
            # Should detect some Byzantine clients
            byzantine_ids = [c.client_id for c in byzantine_clients]
            actual_detected = [client_id for client_id in detected_byzantine 
                             if client_id in byzantine_ids]
            
            detection_rate = len(actual_detected) / len(byzantine_clients)
            assert detection_rate >= 0.5  # Should detect at least 50% of Byzantine clients
            
            # Verify global model is reasonable (not poisoned)
            global_model = await server.get_global_model(honest_clients[0].client_id)
            assert global_model is not None
            
            # Model weights should not be extreme (indicating successful defense)
            model_weights = pickle.loads(global_model["weights"])
            layer1_max = np.max(np.abs(model_weights["layer1"]))
            layer2_max = np.max(np.abs(model_weights["layer2"]))
            
            # Should be much smaller than Byzantine values (which were ~10)
            assert layer1_max < 2.0, f"Model may be poisoned, max weight: {layer1_max}"
            assert layer2_max < 2.0, f"Model may be poisoned, max weight: {layer2_max}"
            
        finally:
            await server.shutdown()
    
    async def test_sybil_attack_defense(self, test_config):
        """Test defense against Sybil attacks (multiple fake identities)."""
        server = AggregationServer(test_config)
        await server.initialize()
        
        try:
            # Register legitimate clients
            legitimate_clients = []
            for i in range(8):
                client_info = ClientInfo(
                    client_id=f"legitimate_client_{i}",
                    client_type="Simulated",
                    capabilities={"cpu_cores": 4, "memory_gb": 8},
                    location={"lat": 37.0 + i * 0.1, "lon": -122.0 + i * 0.1},  # Different locations
                    network_info={"bandwidth": 100, "latency": 10 + i * 5},  # Varied network
                    hardware_specs={"device_id": f"device_{i}"},
                    reputation_score=0.9
                )
                await server.register_client(client_info)
                legitimate_clients.append(client_info)
            
            # Register Sybil clients (same attacker, multiple identities)
            sybil_clients = []
            for i in range(6):
                client_info = ClientInfo(
                    client_id=f"sybil_client_{i}",
                    client_type="Simulated",
                    capabilities={"cpu_cores": 4, "memory_gb": 8},
                    location={"lat": 37.5, "lon": -122.5},  # Same location (suspicious)
                    network_info={"bandwidth": 100, "latency": 15},  # Same network characteristics
                    hardware_specs={"device_id": "sybil_device"},  # Same device (very suspicious)
                    reputation_score=0.8
                )
                await server.register_client(client_info)
                sybil_clients.append(client_info)
            
            # Submit updates
            update_tasks = []
            
            # Legitimate clients submit diverse updates
            for client_info in legitimate_clients:
                weights = {
                    "layer1": np.random.randn(15, 8).astype(np.float32) * 0.1,
                    "layer2": np.random.randn(8, 4).astype(np.float32) * 0.1
                }
                
                update = ModelUpdate(
                    client_id=client_info.client_id,
                    model_weights=pickle.dumps(weights),
                    training_metrics={"loss": 0.5 + np.random.randn() * 0.05, "accuracy": 0.8 + np.random.randn() * 0.02},
                    data_statistics={"num_samples": 1000 + np.random.randint(-100, 100)},
                    computation_time=30.0 + np.random.randn() * 5,
                    network_conditions=client_info.network_info,
                    privacy_budget_used=0.1
                )
                
                task = server.receive_model_update(client_info.client_id, update)
                update_tasks.append(task)
            
            # Sybil clients submit coordinated updates (suspicious similarity)
            sybil_base_weights = {
                "layer1": np.random.randn(15, 8).astype(np.float32) * 2,  # Coordinated attack
                "layer2": np.random.randn(8, 4).astype(np.float32) * 2
            }
            
            for client_info in sybil_clients:
                # Add small variations to avoid exact duplicates
                weights = {}
                for layer, base_weight in sybil_base_weights.items():
                    weights[layer] = base_weight + np.random.randn(*base_weight.shape).astype(np.float32) * 0.01
                
                update = ModelUpdate(
                    client_id=client_info.client_id,
                    model_weights=pickle.dumps(weights),
                    training_metrics={"loss": 0.3, "accuracy": 0.9},  # Suspiciously good and similar
                    data_statistics={"num_samples": 1500},  # Same dataset size
                    computation_time=25.0,  # Same computation time
                    network_conditions=client_info.network_info,
                    privacy_budget_used=0.1
                )
                
                task = server.receive_model_update(client_info.client_id, update)
                update_tasks.append(task)
            
            # Wait for updates and Sybil detection
            results = await asyncio.gather(*update_tasks, return_exceptions=True)
            await asyncio.sleep(2.0)
            
            # Check Sybil detection
            security_status = await server.get_security_status()
            detected_sybils = security_status.get("detected_sybil_clients", [])
            
            # Should detect Sybil clients based on similarity and network characteristics
            sybil_ids = [c.client_id for c in sybil_clients]
            actual_detected_sybils = [client_id for client_id in detected_sybils 
                                    if client_id in sybil_ids]
            
            sybil_detection_rate = len(actual_detected_sybils) / len(sybil_clients)
            assert sybil_detection_rate >= 0.5  # Should detect at least 50% of Sybil clients
            
            # Legitimate clients should not be flagged as Sybils
            legitimate_ids = [c.client_id for c in legitimate_clients]
            false_sybil_detections = [client_id for client_id in detected_sybils 
                                    if client_id in legitimate_ids]
            
            false_positive_rate = len(false_sybil_detections) / len(legitimate_clients)
            assert false_positive_rate <= 0.2  # Less than 20% false positive rate
            
        finally:
            await server.shutdown()
    
    async def test_backdoor_attack_defense(self, test_config):
        """Test defense against backdoor attacks."""
        server = AggregationServer(test_config)
        await server.initialize()
        
        try:
            # Register honest clients
            honest_clients = []
            for i in range(10):
                client_info = ClientInfo(
                    client_id=f"honest_client_{i}",
                    client_type="Simulated",
                    capabilities={"cpu_cores": 4, "memory_gb": 8},
                    location={"lat": 37.0, "lon": -122.0},
                    network_info={"bandwidth": 100, "latency": 10},
                    hardware_specs={},
                    reputation_score=0.9
                )
                await server.register_client(client_info)
                honest_clients.append(client_info)
            
            # Register backdoor attacker
            backdoor_clients = []
            for i in range(2):
                client_info = ClientInfo(
                    client_id=f"backdoor_client_{i}",
                    client_type="Simulated",
                    capabilities={"cpu_cores": 4, "memory_gb": 8},
                    location={"lat": 37.0, "lon": -122.0},
                    network_info={"bandwidth": 100, "latency": 10},
                    hardware_specs={},
                    reputation_score=0.9
                )
                await server.register_client(client_info)
                backdoor_clients.append(client_info)
            
            # Execute multiple rounds to establish backdoor
            for round_num in range(3):
                update_tasks = []
                
                # Honest clients submit normal updates
                for client_info in honest_clients:
                    weights = {
                        "layer1": np.random.randn(12, 6).astype(np.float32) * 0.1,
                        "layer2": np.random.randn(6, 3).astype(np.float32) * 0.1
                    }
                    
                    update = ModelUpdate(
                        client_id=client_info.client_id,
                        model_weights=pickle.dumps(weights),
                        training_metrics={"loss": 0.5 - round_num * 0.02, "accuracy": 0.75 + round_num * 0.01},
                        data_statistics={"num_samples": 1000},
                        computation_time=30.0,
                        network_conditions={"latency": 10, "bandwidth": 100},
                        privacy_budget_used=0.1
                    )
                    
                    task = server.receive_model_update(client_info.client_id, update)
                    update_tasks.append(task)
                
                # Backdoor clients submit subtly poisoned updates
                for client_info in backdoor_clients:
                    # Backdoor: slightly modify specific neurons to create backdoor
                    weights = {
                        "layer1": np.random.randn(12, 6).astype(np.float32) * 0.1,
                        "layer2": np.random.randn(6, 3).astype(np.float32) * 0.1
                    }
                    
                    # Subtle backdoor modification (harder to detect)
                    weights["layer1"][0, 0] += 0.5  # Backdoor trigger
                    weights["layer2"][0, 0] += 0.3
                    
                    update = ModelUpdate(
                        client_id=client_info.client_id,
                        model_weights=pickle.dumps(weights),
                        training_metrics={"loss": 0.48 - round_num * 0.02, "accuracy": 0.76 + round_num * 0.01},  # Slightly better to avoid suspicion
                        data_statistics={"num_samples": 1000},
                        computation_time=32.0,  # Slightly longer (backdoor training)
                        network_conditions={"latency": 10, "bandwidth": 100},
                        privacy_budget_used=0.1
                    )
                    
                    task = server.receive_model_update(client_info.client_id, update)
                    update_tasks.append(task)
                
                # Wait for round completion
                await asyncio.gather(*update_tasks, return_exceptions=True)
                await asyncio.sleep(1.0)
            
            # Check for backdoor detection
            security_status = await server.get_security_status()
            
            # Advanced backdoor detection might flag suspicious patterns
            detected_backdoors = security_status.get("detected_backdoor_clients", [])
            anomaly_scores = security_status.get("client_anomaly_scores", {})
            
            # Backdoor clients might have higher anomaly scores
            backdoor_ids = [c.client_id for c in backdoor_clients]
            
            if anomaly_scores:
                backdoor_scores = [anomaly_scores.get(client_id, 0) for client_id in backdoor_ids]
                honest_scores = [anomaly_scores.get(c.client_id, 0) for c in honest_clients]
                
                avg_backdoor_score = np.mean(backdoor_scores) if backdoor_scores else 0
                avg_honest_score = np.mean(honest_scores) if honest_scores else 0
                
                # Backdoor clients should have higher anomaly scores
                if avg_backdoor_score > 0 and avg_honest_score > 0:
                    assert avg_backdoor_score > avg_honest_score, "Backdoor clients should have higher anomaly scores"
            
            # Verify system remains functional despite backdoor attempts
            global_model = await server.get_global_model(honest_clients[0].client_id)
            assert global_model is not None
            
        finally:
            await server.shutdown()


@pytest.mark.security
@pytest.mark.byzantine
class TestByzantineToleranceMetrics:
    """Test metrics and monitoring for Byzantine fault tolerance."""
    
    def test_byzantine_tolerance_threshold(self):
        """Test Byzantine tolerance threshold calculations."""
        from src.aggregation_server.aggregation import ByzantineToleranceCalculator
        
        calculator = ByzantineToleranceCalculator()
        
        # Test different total client counts
        test_cases = [
            {"total_clients": 10, "expected_max_byzantine": 3},  # 10/3 = 3.33, so 3
            {"total_clients": 15, "expected_max_byzantine": 4},  # 15/3 = 5, so 4
            {"total_clients": 21, "expected_max_byzantine": 6},  # 21/3 = 7, so 6
            {"total_clients": 30, "expected_max_byzantine": 9}   # 30/3 = 10, so 9
        ]
        
        for case in test_cases:
            max_byzantine = calculator.calculate_max_byzantine_clients(case["total_clients"])
            assert max_byzantine == case["expected_max_byzantine"], \
                f"For {case['total_clients']} clients, expected {case['expected_max_byzantine']} max Byzantine, got {max_byzantine}"
    
    def test_security_metrics_collection(self):
        """Test collection of security metrics."""
        from src.monitoring.security_metrics import SecurityMetricsCollector
        
        collector = SecurityMetricsCollector()
        
        # Simulate detection events
        detection_events = [
            {"client_id": "client_1", "attack_type": "model_poisoning", "confidence": 0.9},
            {"client_id": "client_2", "attack_type": "sybil", "confidence": 0.8},
            {"client_id": "client_3", "attack_type": "model_poisoning", "confidence": 0.95},
            {"client_id": "client_4", "attack_type": "backdoor", "confidence": 0.7}
        ]
        
        for event in detection_events:
            collector.record_detection_event(event)
        
        # Test metrics calculation
        metrics = collector.get_security_metrics()
        
        assert metrics["total_detections"] == 4
        assert metrics["attack_types"]["model_poisoning"] == 2
        assert metrics["attack_types"]["sybil"] == 1
        assert metrics["attack_types"]["backdoor"] == 1
        
        # Test detection rate calculation
        collector.record_round_completion(total_clients=10, participating_clients=8)
        
        detection_rate = collector.calculate_detection_rate()
        assert 0 <= detection_rate <= 1  # Should be a valid rate
    
    def test_reputation_system_metrics(self):
        """Test reputation system metrics."""
        from src.aggregation_server.reputation import ReputationSystem
        
        reputation_system = ReputationSystem()
        
        # Initialize client reputations
        clients = [f"client_{i}" for i in range(10)]
        for client_id in clients:
            reputation_system.initialize_client(client_id)
        
        # Simulate good and bad behavior
        for round_num in range(5):
            # Most clients behave well
            for i in range(8):
                reputation_system.update_reputation(f"client_{i}", behavior_score=0.9)
            
            # Some clients behave poorly
            for i in range(8, 10):
                reputation_system.update_reputation(f"client_{i}", behavior_score=0.3)
        
        # Test reputation metrics
        metrics = reputation_system.get_reputation_metrics()
        
        assert "average_reputation" in metrics
        assert "reputation_distribution" in metrics
        assert "low_reputation_clients" in metrics
        
        # Good clients should have high reputation
        good_client_reputation = reputation_system.get_reputation("client_0")
        bad_client_reputation = reputation_system.get_reputation("client_8")
        
        assert good_client_reputation > bad_client_reputation
        assert good_client_reputation > 0.8
        assert bad_client_reputation < 0.6
    
    def test_attack_success_rate_tracking(self):
        """Test tracking of attack success rates."""
        from src.monitoring.attack_tracker import AttackTracker
        
        tracker = AttackTracker()
        
        # Simulate various attack attempts and their outcomes
        attack_scenarios = [
            {"attack_type": "model_poisoning", "detected": True, "successful": False},
            {"attack_type": "model_poisoning", "detected": False, "successful": True},
            {"attack_type": "sybil", "detected": True, "successful": False},
            {"attack_type": "sybil", "detected": True, "successful": False},
            {"attack_type": "backdoor", "detected": False, "successful": True},
            {"attack_type": "model_poisoning", "detected": True, "successful": False}
        ]
        
        for scenario in attack_scenarios:
            tracker.record_attack_attempt(
                attack_type=scenario["attack_type"],
                detected=scenario["detected"],
                successful=scenario["successful"]
            )
        
        # Test attack statistics
        stats = tracker.get_attack_statistics()
        
        # Model poisoning: 3 attempts, 2 detected, 1 successful
        mp_stats = stats["model_poisoning"]
        assert mp_stats["total_attempts"] == 3
        assert mp_stats["detection_rate"] == 2/3
        assert mp_stats["success_rate"] == 1/3
        
        # Sybil: 2 attempts, 2 detected, 0 successful
        sybil_stats = stats["sybil"]
        assert sybil_stats["total_attempts"] == 2
        assert sybil_stats["detection_rate"] == 1.0
        assert sybil_stats["success_rate"] == 0.0
        
        # Overall detection effectiveness
        overall_stats = tracker.get_overall_statistics()
        assert overall_stats["overall_detection_rate"] == 4/6  # 4 detected out of 6 total
        assert overall_stats["overall_success_rate"] == 2/6    # 2 successful out of 6 total