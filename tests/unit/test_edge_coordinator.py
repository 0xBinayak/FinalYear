"""
Comprehensive unit tests for edge coordinator components.
"""
import pytest
import asyncio
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from src.edge_coordinator.coordinator import EdgeCoordinator
from src.edge_coordinator.resource_manager import ResourceManager
from src.edge_coordinator.network_partition import NetworkPartitionHandler
from src.edge_coordinator.offline_operation import OfflineOperationManager
from src.edge_coordinator.data_quality import DataQualityValidator
from src.common.interfaces import ClientInfo, ModelUpdate


@pytest.mark.unit
class TestEdgeCoordinator:
    """Test cases for EdgeCoordinator class."""
    
    async def test_coordinator_initialization(self, test_config):
        """Test edge coordinator initialization."""
        coordinator = EdgeCoordinator(test_config, region="us-west-1")
        await coordinator.initialize()
        
        assert coordinator.region == "us-west-1"
        assert coordinator.is_running
        assert coordinator.local_clients == {}
        
        await coordinator.shutdown()
    
    async def test_local_client_management(self, test_config, sample_client_info):
        """Test local client registration and management."""
        coordinator = EdgeCoordinator(test_config, region="us-west-1")
        await coordinator.initialize()
        
        # Test client registration
        success = await coordinator.register_local_client(sample_client_info)
        assert success
        assert sample_client_info.client_id in coordinator.local_clients
        
        # Test client info retrieval
        client_info = await coordinator.get_local_client_info(sample_client_info.client_id)
        assert client_info.client_id == sample_client_info.client_id
        
        # Test client deregistration
        success = await coordinator.deregister_local_client(sample_client_info.client_id)
        assert success
        assert sample_client_info.client_id not in coordinator.local_clients
        
        await coordinator.shutdown()
    
    async def test_local_aggregation(self, test_config):
        """Test local model aggregation."""
        coordinator = EdgeCoordinator(test_config, region="us-west-1")
        await coordinator.initialize()
        
        # Create sample model updates
        updates = []
        for i in range(3):
            weights = {
                "layer1": np.random.randn(5, 3).astype(np.float32),
                "layer2": np.random.randn(3, 1).astype(np.float32)
            }
            
            update = ModelUpdate(
                client_id=f"local_client_{i}",
                model_weights=weights,
                training_metrics={"loss": 0.5 - i * 0.1, "accuracy": 0.7 + i * 0.05},
                data_statistics={"num_samples": 1000 + i * 100},
                computation_time=30.0,
                network_conditions={"latency": 10, "bandwidth": 100},
                privacy_budget_used=0.1
            )
            updates.append(update)
        
        # Test local aggregation
        local_aggregate = await coordinator.aggregate_local_models(updates)
        
        assert local_aggregate is not None
        assert "aggregated_weights" in local_aggregate
        assert "participating_clients" in local_aggregate
        assert local_aggregate["participating_clients"] == 3
        
        await coordinator.shutdown()
    
    async def test_global_server_synchronization(self, test_config):
        """Test synchronization with global aggregation server."""
        coordinator = EdgeCoordinator(test_config, region="us-west-1")
        await coordinator.initialize()
        
        # Mock global server
        mock_global_server = AsyncMock()
        mock_global_server.receive_edge_update.return_value = True
        mock_global_server.get_global_model.return_value = {
            "version": 5,
            "weights": {"layer1": np.random.randn(5, 3), "layer2": np.random.randn(3, 1)}
        }
        
        # Test synchronization
        success = await coordinator.sync_with_global_server(mock_global_server)
        assert success
        
        # Verify global model was retrieved
        global_model = await coordinator.get_current_global_model()
        assert global_model is not None
        assert global_model["version"] == 5
        
        await coordinator.shutdown()
    
    async def test_bandwidth_optimization(self, test_config):
        """Test bandwidth optimization strategies."""
        coordinator = EdgeCoordinator(test_config, region="us-west-1")
        await coordinator.initialize()
        
        # Create large model update
        large_weights = {
            "layer1": np.random.randn(1000, 500).astype(np.float32),
            "layer2": np.random.randn(500, 100).astype(np.float32)
        }
        
        # Test compression
        compressed_update = await coordinator.compress_model_update(large_weights)
        assert compressed_update is not None
        assert len(compressed_update) < len(str(large_weights))
        
        # Test differential updates
        previous_weights = {
            "layer1": np.random.randn(1000, 500).astype(np.float32),
            "layer2": np.random.randn(500, 100).astype(np.float32)
        }
        
        differential_update = await coordinator.create_differential_update(
            large_weights, previous_weights
        )
        assert differential_update is not None
        
        await coordinator.shutdown()


@pytest.mark.unit
class TestResourceManager:
    """Test cases for ResourceManager class."""
    
    def test_resource_assessment(self):
        """Test resource assessment functionality."""
        manager = ResourceManager()
        
        # Mock system resources
        with patch('psutil.cpu_percent') as mock_cpu, \
             patch('psutil.virtual_memory') as mock_memory, \
             patch('psutil.disk_usage') as mock_disk:
            
            mock_cpu.return_value = 75.0
            mock_memory.return_value.percent = 60.0
            mock_disk.return_value.percent = 45.0
            
            resources = manager.assess_current_resources()
            
            assert resources["cpu_usage"] == 75.0
            assert resources["memory_usage"] == 60.0
            assert resources["disk_usage"] == 45.0
    
    def test_client_resource_allocation(self):
        """Test resource allocation for clients."""
        manager = ResourceManager()
        
        # Define available resources
        available_resources = {
            "cpu_cores": 8,
            "memory_gb": 16,
            "gpu_memory_gb": 4
        }
        
        # Define client requirements
        client_requirements = [
            {"cpu_cores": 2, "memory_gb": 4, "gpu_memory_gb": 1},
            {"cpu_cores": 3, "memory_gb": 6, "gpu_memory_gb": 2},
            {"cpu_cores": 4, "memory_gb": 8, "gpu_memory_gb": 2}
        ]
        
        # Test allocation
        allocation = manager.allocate_resources(available_resources, client_requirements)
        
        assert allocation is not None
        assert len(allocation) <= len(client_requirements)
        
        # Verify resource constraints are respected
        total_allocated = {"cpu_cores": 0, "memory_gb": 0, "gpu_memory_gb": 0}
        for alloc in allocation:
            for resource, amount in alloc.items():
                total_allocated[resource] += amount
        
        assert total_allocated["cpu_cores"] <= available_resources["cpu_cores"]
        assert total_allocated["memory_gb"] <= available_resources["memory_gb"]
    
    def test_load_balancing(self):
        """Test load balancing across clients."""
        manager = ResourceManager()
        
        # Define client loads
        client_loads = {
            "client_1": {"cpu": 80, "memory": 70, "network": 50},
            "client_2": {"cpu": 30, "memory": 40, "network": 20},
            "client_3": {"cpu": 90, "memory": 85, "network": 75}
        }
        
        # Test load balancing recommendations
        recommendations = manager.balance_load(client_loads)
        
        assert recommendations is not None
        assert "redistribute_tasks" in recommendations
        assert "throttle_clients" in recommendations
        
        # High-load clients should be throttled
        assert "client_3" in recommendations["throttle_clients"]
    
    def test_adaptive_scheduling(self):
        """Test adaptive training scheduling."""
        manager = ResourceManager()
        
        # Define client profiles
        client_profiles = {
            "mobile_client": {
                "battery_level": 30,
                "network_type": "cellular",
                "processing_power": "low"
            },
            "edge_server": {
                "power_source": "mains",
                "network_type": "ethernet",
                "processing_power": "high"
            }
        }
        
        # Test scheduling
        schedule = manager.create_adaptive_schedule(client_profiles)
        
        assert schedule is not None
        assert "mobile_client" in schedule
        assert "edge_server" in schedule
        
        # Mobile client should have longer intervals
        mobile_interval = schedule["mobile_client"]["training_interval"]
        server_interval = schedule["edge_server"]["training_interval"]
        assert mobile_interval > server_interval


@pytest.mark.unit
class TestNetworkPartitionHandler:
    """Test cases for NetworkPartitionHandler class."""
    
    def test_partition_detection(self):
        """Test network partition detection."""
        handler = NetworkPartitionHandler()
        
        # Mock network connectivity checks
        with patch('src.edge_coordinator.network_partition.ping_host') as mock_ping:
            # Simulate partition - can't reach global server
            mock_ping.return_value = False
            
            is_partitioned = handler.detect_partition("global.server.com")
            assert is_partitioned
            
            # Simulate recovery
            mock_ping.return_value = True
            is_partitioned = handler.detect_partition("global.server.com")
            assert not is_partitioned
    
    def test_partition_handling(self):
        """Test partition handling strategies."""
        handler = NetworkPartitionHandler()
        
        # Test entering partition mode
        handler.enter_partition_mode()
        assert handler.is_partitioned
        assert handler.partition_start_time is not None
        
        # Test partition mode operations
        operations = handler.get_partition_operations()
        assert operations["enable_local_aggregation"] is True
        assert operations["cache_updates"] is True
        assert operations["reduce_communication"] is True
        
        # Test exiting partition mode
        handler.exit_partition_mode()
        assert not handler.is_partitioned
    
    def test_partition_recovery(self):
        """Test recovery from network partition."""
        handler = NetworkPartitionHandler()
        
        # Simulate partition with cached updates
        handler.enter_partition_mode()
        
        # Add cached updates
        cached_updates = [
            {"client_id": "client_1", "update": "data_1", "timestamp": datetime.now()},
            {"client_id": "client_2", "update": "data_2", "timestamp": datetime.now()}
        ]
        
        for update in cached_updates:
            handler.cache_update(update)
        
        # Test recovery
        recovery_data = handler.prepare_recovery_data()
        assert recovery_data is not None
        assert len(recovery_data["cached_updates"]) == 2
        assert "partition_duration" in recovery_data
        
        # Test synchronization after recovery
        sync_plan = handler.create_sync_plan(recovery_data)
        assert sync_plan is not None
        assert "update_order" in sync_plan
        assert "conflict_resolution" in sync_plan


@pytest.mark.unit
class TestOfflineOperationManager:
    """Test cases for OfflineOperationManager class."""
    
    def test_offline_mode_activation(self):
        """Test offline mode activation."""
        manager = OfflineOperationManager()
        
        # Test entering offline mode
        manager.enter_offline_mode()
        assert manager.is_offline
        assert manager.offline_start_time is not None
        
        # Test offline capabilities
        capabilities = manager.get_offline_capabilities()
        assert capabilities["local_training"] is True
        assert capabilities["local_aggregation"] is True
        assert capabilities["data_caching"] is True
    
    def test_offline_data_management(self, temp_dir):
        """Test offline data management."""
        manager = OfflineOperationManager(cache_dir=temp_dir)
        manager.enter_offline_mode()
        
        # Test data caching
        test_data = {"model_weights": np.random.randn(10, 5), "metadata": {"version": 1}}
        cache_id = manager.cache_data("model_update", test_data)
        assert cache_id is not None
        
        # Test data retrieval
        retrieved_data = manager.get_cached_data(cache_id)
        assert retrieved_data is not None
        assert "model_weights" in retrieved_data
        
        # Test cache management
        cache_info = manager.get_cache_info()
        assert cache_info["total_items"] == 1
        assert cache_info["total_size"] > 0
    
    def test_offline_training_coordination(self):
        """Test offline training coordination."""
        manager = OfflineOperationManager()
        manager.enter_offline_mode()
        
        # Define local clients
        local_clients = ["client_1", "client_2", "client_3"]
        
        # Test training round coordination
        training_round = manager.coordinate_offline_training(local_clients)
        assert training_round is not None
        assert "round_id" in training_round
        assert "participating_clients" in training_round
        assert len(training_round["participating_clients"]) == 3
    
    def test_eventual_consistency(self):
        """Test eventual consistency mechanisms."""
        manager = OfflineOperationManager()
        
        # Simulate offline operations
        manager.enter_offline_mode()
        
        offline_operations = [
            {"type": "model_update", "client": "client_1", "data": "update_1"},
            {"type": "aggregation", "participants": ["client_1", "client_2"]},
            {"type": "model_update", "client": "client_2", "data": "update_2"}
        ]
        
        for op in offline_operations:
            manager.record_offline_operation(op)
        
        # Test consistency resolution
        manager.exit_offline_mode()
        consistency_plan = manager.create_consistency_plan()
        
        assert consistency_plan is not None
        assert "operation_order" in consistency_plan
        assert "conflict_resolution" in consistency_plan
        assert len(consistency_plan["operation_order"]) == 3


@pytest.mark.unit
class TestDataQualityValidator:
    """Test cases for DataQualityValidator class."""
    
    def test_signal_quality_validation(self, sample_signal_data):
        """Test signal quality validation."""
        validator = DataQualityValidator()
        
        # Test valid signal
        quality_score = validator.validate_signal_quality(sample_signal_data)
        assert 0.0 <= quality_score <= 1.0
        
        # Test signal with issues
        noisy_signal = sample_signal_data
        noisy_signal.snr = -5.0  # Very low SNR
        
        quality_score = validator.validate_signal_quality(noisy_signal)
        assert quality_score < 0.5  # Should be low quality
    
    def test_data_completeness_check(self):
        """Test data completeness validation."""
        validator = DataQualityValidator()
        
        # Test complete data
        complete_data = {
            "features": np.random.randn(100, 10),
            "labels": np.random.randint(0, 5, 100),
            "metadata": {"source": "test", "timestamp": datetime.now()}
        }
        
        is_complete = validator.check_data_completeness(complete_data)
        assert is_complete
        
        # Test incomplete data
        incomplete_data = {
            "features": np.random.randn(100, 10),
            # Missing labels
            "metadata": {"source": "test"}
        }
        
        is_complete = validator.check_data_completeness(incomplete_data)
        assert not is_complete
    
    def test_statistical_validation(self):
        """Test statistical data validation."""
        validator = DataQualityValidator()
        
        # Generate test data with known properties
        normal_data = np.random.normal(0, 1, 1000)
        uniform_data = np.random.uniform(-1, 1, 1000)
        
        # Test distribution validation
        normal_stats = validator.validate_distribution(normal_data, expected_type="normal")
        assert normal_stats["is_valid"]
        assert normal_stats["p_value"] > 0.05
        
        uniform_stats = validator.validate_distribution(uniform_data, expected_type="uniform")
        assert uniform_stats["is_valid"]
    
    def test_anomaly_detection(self):
        """Test anomaly detection in data."""
        validator = DataQualityValidator()
        
        # Generate data with anomalies
        normal_data = np.random.normal(0, 1, 1000)
        anomalous_data = np.concatenate([normal_data, [10, -10, 15]])  # Add outliers
        
        # Test anomaly detection
        anomalies = validator.detect_anomalies(anomalous_data)
        assert len(anomalies) > 0
        assert 10 in anomalies or -10 in anomalies or 15 in anomalies
    
    def test_data_quality_scoring(self):
        """Test comprehensive data quality scoring."""
        validator = DataQualityValidator()
        
        # High quality data
        high_quality_data = {
            "signal_data": sample_signal_data,
            "features": np.random.randn(1000, 20),
            "labels": np.random.randint(0, 5, 1000),
            "snr": 20.0,
            "completeness": 1.0
        }
        
        quality_score = validator.compute_overall_quality_score(high_quality_data)
        assert quality_score > 0.8
        
        # Low quality data
        low_quality_data = {
            "signal_data": sample_signal_data,
            "features": np.random.randn(100, 5),  # Less data
            "labels": np.random.randint(0, 5, 100),
            "snr": -5.0,  # Low SNR
            "completeness": 0.6  # Incomplete
        }
        
        quality_score = validator.compute_overall_quality_score(low_quality_data)
        assert quality_score < 0.5