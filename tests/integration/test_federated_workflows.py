"""
Integration tests for complete federated learning workflows.
"""
import pytest
import asyncio
import numpy as np
import pickle
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from src.aggregation_server.server import AggregationServer
from src.edge_coordinator.coordinator import EdgeCoordinator
from src.sdr_client.sdr_client import SDRClient
from src.mobile_client.mobile_client import MobileClient
from src.common.interfaces import ClientInfo, ModelUpdate


@pytest.mark.integration
class TestEndToEndFederatedLearning:
    """Test complete end-to-end federated learning workflows."""
    
    async def test_basic_federated_round(self, test_config):
        """Test a basic federated learning round with multiple clients."""
        # Initialize aggregation server
        server = AggregationServer(test_config)
        await server.initialize()
        
        try:
            # Create multiple clients
            clients = []
            for i in range(5):
                client_info = ClientInfo(
                    client_id=f"integration_client_{i}",
                    client_type="Simulated",
                    capabilities={"cpu_cores": 4 + i, "memory_gb": 8 + i * 2},
                    location={"lat": 37.0 + i, "lon": -122.0 + i},
                    network_info={"bandwidth": 50 + i * 20, "latency": 20 - i * 2},
                    hardware_specs={"gpu": i % 2 == 0},
                    reputation_score=0.8 + i * 0.04
                )
                
                # Register client
                token = await server.register_client(client_info)
                clients.append((client_info, token))
            
            # Submit model updates from all clients
            for i, (client_info, token) in enumerate(clients):
                weights = {
                    "layer1": np.random.randn(20, 10).astype(np.float32) + i * 0.1,
                    "layer2": np.random.randn(10, 5).astype(np.float32) + i * 0.1,
                    "layer3": np.random.randn(5, 1).astype(np.float32) + i * 0.1
                }
                
                model_update = ModelUpdate(
                    client_id=client_info.client_id,
                    model_weights=pickle.dumps(weights),
                    training_metrics={
                        "loss": 1.0 - i * 0.1,
                        "accuracy": 0.6 + i * 0.05,
                        "f1_score": 0.65 + i * 0.04
                    },
                    data_statistics={"num_samples": 800 + i * 200},
                    computation_time=25.0 + i * 5,
                    network_conditions={
                        "latency": 20 - i * 2,
                        "bandwidth": 50 + i * 20,
                        "packet_loss": 0.01 + i * 0.005
                    },
                    privacy_budget_used=0.1
                )
                
                success = await server.receive_model_update(client_info.client_id, model_update)
                assert success, f"Failed to submit update from {client_info.client_id}"
            
            # Wait for aggregation to complete
            await asyncio.sleep(2.0)
            
            # Verify global model is available
            for client_info, _ in clients:
                global_model = await server.get_global_model(client_info.client_id)
                assert global_model is not None
                assert "version" in global_model
                assert "weights" in global_model
                assert global_model["version"] > 0
            
            # Verify server status
            status = await server.get_server_status()
            assert status["status"] == "running"
            assert status["total_clients"] == 5
            assert status["current_round"] > 0
            
            # Verify convergence metrics
            convergence_history = await server.get_convergence_history()
            assert len(convergence_history) > 0
            
            latest_metrics = convergence_history[-1]
            assert "average_loss" in latest_metrics
            assert "average_accuracy" in latest_metrics
            assert latest_metrics["participating_clients"] == 5
            
        finally:
            await server.shutdown()
    
    async def test_hierarchical_federated_learning(self, test_config):
        """Test hierarchical federated learning with edge coordinators."""
        # Initialize global aggregation server
        global_server = AggregationServer(test_config)
        await global_server.initialize()
        
        # Initialize edge coordinators
        edge_coordinators = []
        for region in ["us-west", "us-east", "eu-central"]:
            coordinator = EdgeCoordinator(test_config, region=region)
            await coordinator.initialize()
            edge_coordinators.append(coordinator)
        
        try:
            # Register edge coordinators with global server
            for coordinator in edge_coordinators:
                edge_client_info = ClientInfo(
                    client_id=f"edge_{coordinator.region}",
                    client_type="EdgeCoordinator",
                    capabilities={"cpu_cores": 16, "memory_gb": 64},
                    location={"lat": 40.0, "lon": -100.0},
                    network_info={"bandwidth": 1000, "latency": 5},
                    hardware_specs={"gpu": True},
                    reputation_score=1.0
                )
                
                await global_server.register_client(edge_client_info)
            
            # Create local clients for each edge coordinator
            all_local_clients = []
            for i, coordinator in enumerate(edge_coordinators):
                local_clients = []
                for j in range(3):  # 3 clients per edge
                    client_info = ClientInfo(
                        client_id=f"local_{coordinator.region}_{j}",
                        client_type="SDR",
                        capabilities={"cpu_cores": 4, "memory_gb": 8},
                        location={"lat": 40.0 + i, "lon": -100.0 + j},
                        network_info={"bandwidth": 100, "latency": 15},
                        hardware_specs={"sdr_type": "rtlsdr"},
                        reputation_score=0.9
                    )
                    
                    await coordinator.register_local_client(client_info)
                    local_clients.append(client_info)
                
                all_local_clients.extend(local_clients)
            
            # Simulate local training and aggregation
            for coordinator in edge_coordinators:
                local_updates = []
                for client_info in coordinator.local_clients.values():
                    weights = {
                        "layer1": np.random.randn(15, 8).astype(np.float32),
                        "layer2": np.random.randn(8, 3).astype(np.float32)
                    }
                    
                    update = ModelUpdate(
                        client_id=client_info.client_id,
                        model_weights=weights,
                        training_metrics={"loss": 0.4, "accuracy": 0.8},
                        data_statistics={"num_samples": 1000},
                        computation_time=30.0,
                        network_conditions={"latency": 15, "bandwidth": 100},
                        privacy_budget_used=0.1
                    )
                    local_updates.append(update)
                
                # Perform local aggregation
                local_aggregate = await coordinator.aggregate_local_models(local_updates)
                assert local_aggregate is not None
                assert local_aggregate["participating_clients"] == 3
                
                # Submit aggregated update to global server
                edge_update = ModelUpdate(
                    client_id=f"edge_{coordinator.region}",
                    model_weights=pickle.dumps(local_aggregate["aggregated_weights"]),
                    training_metrics=local_aggregate["aggregated_metrics"],
                    data_statistics={"num_samples": local_aggregate["total_samples"]},
                    computation_time=local_aggregate["computation_time"],
                    network_conditions={"latency": 5, "bandwidth": 1000},
                    privacy_budget_used=0.3
                )
                
                success = await global_server.receive_model_update(
                    f"edge_{coordinator.region}", edge_update
                )
                assert success
            
            # Wait for global aggregation
            await asyncio.sleep(2.0)
            
            # Verify global model distribution
            for coordinator in edge_coordinators:
                global_model = await global_server.get_global_model(f"edge_{coordinator.region}")
                assert global_model is not None
                
                # Distribute to local clients
                for client_info in coordinator.local_clients.values():
                    await coordinator.distribute_global_model(client_info.client_id, global_model)
            
            # Verify final status
            status = await global_server.get_server_status()
            assert status["total_clients"] == 3  # 3 edge coordinators
            assert status["current_round"] > 0
            
        finally:
            await global_server.shutdown()
            for coordinator in edge_coordinators:
                await coordinator.shutdown()
    
    async def test_mixed_client_types(self, test_config):
        """Test federated learning with mixed client types (SDR, Mobile, Simulated)."""
        server = AggregationServer(test_config)
        await server.initialize()
        
        try:
            # Create mixed client types
            clients = []
            
            # SDR clients
            for i in range(2):
                with patch('src.sdr_client.hardware_abstraction.SDRHardwareManager'):
                    sdr_client = SDRClient(test_config)
                    await sdr_client.initialize()
                    
                    client_info = ClientInfo(
                        client_id=f"sdr_client_{i}",
                        client_type="SDR",
                        capabilities={"cpu_cores": 4, "memory_gb": 8, "sdr_type": "rtlsdr"},
                        location={"lat": 37.0 + i, "lon": -122.0},
                        network_info={"bandwidth": 50, "latency": 20},
                        hardware_specs={"sdr_type": "rtlsdr", "frequency_range": "24MHz-1766MHz"},
                        reputation_score=0.9
                    )
                    
                    token = await server.register_client(client_info)
                    clients.append(("SDR", sdr_client, client_info, token))
            
            # Mobile clients
            for i in range(2):
                mobile_client = MobileClient(test_config)
                await mobile_client.initialize()
                
                client_info = ClientInfo(
                    client_id=f"mobile_client_{i}",
                    client_type="Mobile",
                    capabilities={"cpu_cores": 6, "memory_gb": 4, "gpu_available": True},
                    location={"lat": 37.5 + i, "lon": -122.5},
                    network_info={"bandwidth": 30, "latency": 50, "connection_type": "cellular"},
                    hardware_specs={"device_model": "smartphone", "os": "android"},
                    reputation_score=0.85
                )
                
                token = await server.register_client(client_info)
                clients.append(("Mobile", mobile_client, client_info, token))
            
            # Simulated clients
            for i in range(3):
                client_info = ClientInfo(
                    client_id=f"sim_client_{i}",
                    client_type="Simulated",
                    capabilities={"cpu_cores": 8, "memory_gb": 16},
                    location={"lat": 38.0 + i, "lon": -121.0},
                    network_info={"bandwidth": 100, "latency": 10},
                    hardware_specs={"virtualized": True},
                    reputation_score=1.0
                )
                
                token = await server.register_client(client_info)
                clients.append(("Simulated", None, client_info, token))
            
            # Submit updates from all client types
            for client_type, client_obj, client_info, token in clients:
                if client_type == "SDR":
                    # Mock SDR-specific training
                    with patch.object(client_obj, 'train_local_model') as mock_train:
                        mock_weights = {
                            "conv1": np.random.randn(32, 1, 3, 3).astype(np.float32),
                            "conv2": np.random.randn(64, 32, 3, 3).astype(np.float32),
                            "fc1": np.random.randn(128, 64).astype(np.float32),
                            "fc2": np.random.randn(10, 128).astype(np.float32)
                        }
                        
                        mock_update = ModelUpdate(
                            client_id=client_info.client_id,
                            model_weights=pickle.dumps(mock_weights),
                            training_metrics={"loss": 0.3, "accuracy": 0.85, "snr": 15.0},
                            data_statistics={"num_samples": 2000, "signal_types": 5},
                            computation_time=45.0,
                            network_conditions={"latency": 20, "bandwidth": 50},
                            privacy_budget_used=0.15
                        )
                        mock_train.return_value = mock_update
                        
                        model_update = await client_obj.train_local_model({})
                
                elif client_type == "Mobile":
                    # Mock mobile-specific training
                    with patch.object(client_obj, 'train_with_resource_constraints') as mock_train:
                        mock_weights = {
                            "layer1": np.random.randn(64, 32).astype(np.float32),
                            "layer2": np.random.randn(32, 16).astype(np.float32),
                            "layer3": np.random.randn(16, 5).astype(np.float32)
                        }
                        
                        mock_update = ModelUpdate(
                            client_id=client_info.client_id,
                            model_weights=pickle.dumps(mock_weights),
                            training_metrics={"loss": 0.4, "accuracy": 0.78, "battery_used": 5.0},
                            data_statistics={"num_samples": 1500, "synthetic_ratio": 0.6},
                            computation_time=35.0,
                            network_conditions={"latency": 50, "bandwidth": 30},
                            privacy_budget_used=0.12
                        )
                        mock_train.return_value = mock_update
                        
                        model_update = await client_obj.train_with_resource_constraints({}, {})
                
                else:  # Simulated
                    weights = {
                        "layer1": np.random.randn(50, 25).astype(np.float32),
                        "layer2": np.random.randn(25, 10).astype(np.float32),
                        "layer3": np.random.randn(10, 1).astype(np.float32)
                    }
                    
                    model_update = ModelUpdate(
                        client_id=client_info.client_id,
                        model_weights=pickle.dumps(weights),
                        training_metrics={"loss": 0.35, "accuracy": 0.82},
                        data_statistics={"num_samples": 1800},
                        computation_time=25.0,
                        network_conditions={"latency": 10, "bandwidth": 100},
                        privacy_budget_used=0.1
                    )
                
                # Submit update to server
                success = await server.receive_model_update(client_info.client_id, model_update)
                assert success, f"Failed to submit update from {client_type} client {client_info.client_id}"
            
            # Wait for aggregation
            await asyncio.sleep(3.0)
            
            # Verify aggregation handled different client types
            status = await server.get_server_status()
            assert status["total_clients"] == 7
            assert status["current_round"] > 0
            
            # Verify all clients can retrieve global model
            for _, _, client_info, _ in clients:
                global_model = await server.get_global_model(client_info.client_id)
                assert global_model is not None
                assert "weights" in global_model
            
            # Verify convergence metrics account for different client types
            convergence_history = await server.get_convergence_history()
            assert len(convergence_history) > 0
            
            latest_metrics = convergence_history[-1]
            assert latest_metrics["participating_clients"] == 7
            assert "client_type_distribution" in latest_metrics
            
        finally:
            await server.shutdown()
            # Cleanup client objects
            for client_type, client_obj, _, _ in clients:
                if client_obj:
                    await client_obj.shutdown()


@pytest.mark.integration
class TestNetworkResilienceWorkflows:
    """Test federated learning workflows under network challenges."""
    
    async def test_network_partition_recovery(self, test_config):
        """Test federated learning recovery from network partitions."""
        # Initialize components
        global_server = AggregationServer(test_config)
        await global_server.initialize()
        
        edge_coordinator = EdgeCoordinator(test_config, region="test-region")
        await edge_coordinator.initialize()
        
        try:
            # Register edge coordinator
            edge_info = ClientInfo(
                client_id="edge_test",
                client_type="EdgeCoordinator",
                capabilities={"cpu_cores": 8, "memory_gb": 32},
                location={"lat": 40.0, "lon": -100.0},
                network_info={"bandwidth": 500, "latency": 10},
                hardware_specs={"edge_device": True},
                reputation_score=1.0
            )
            
            await global_server.register_client(edge_info)
            
            # Register local clients with edge coordinator
            local_clients = []
            for i in range(3):
                client_info = ClientInfo(
                    client_id=f"local_client_{i}",
                    client_type="SDR",
                    capabilities={"cpu_cores": 4, "memory_gb": 8},
                    location={"lat": 40.0 + i, "lon": -100.0},
                    network_info={"bandwidth": 50, "latency": 20},
                    hardware_specs={"sdr_type": "rtlsdr"},
                    reputation_score=0.9
                )
                
                await edge_coordinator.register_local_client(client_info)
                local_clients.append(client_info)
            
            # Simulate normal operation
            local_updates = []
            for client_info in local_clients:
                weights = {
                    "layer1": np.random.randn(10, 5).astype(np.float32),
                    "layer2": np.random.randn(5, 1).astype(np.float32)
                }
                
                update = ModelUpdate(
                    client_id=client_info.client_id,
                    model_weights=weights,
                    training_metrics={"loss": 0.4, "accuracy": 0.8},
                    data_statistics={"num_samples": 1000},
                    computation_time=30.0,
                    network_conditions={"latency": 20, "bandwidth": 50},
                    privacy_budget_used=0.1
                )
                local_updates.append(update)
            
            # Perform local aggregation
            local_aggregate = await edge_coordinator.aggregate_local_models(local_updates)
            assert local_aggregate is not None
            
            # Simulate network partition
            edge_coordinator.network_partition_handler.enter_partition_mode()
            assert edge_coordinator.network_partition_handler.is_partitioned
            
            # Continue local operations during partition
            partition_updates = []
            for i, client_info in enumerate(local_clients):
                weights = {
                    "layer1": np.random.randn(10, 5).astype(np.float32) + i * 0.1,
                    "layer2": np.random.randn(5, 1).astype(np.float32) + i * 0.1
                }
                
                update = ModelUpdate(
                    client_id=client_info.client_id,
                    model_weights=weights,
                    training_metrics={"loss": 0.35 - i * 0.05, "accuracy": 0.82 + i * 0.02},
                    data_statistics={"num_samples": 1000},
                    computation_time=30.0,
                    network_conditions={"latency": 20, "bandwidth": 50},
                    privacy_budget_used=0.1
                )
                partition_updates.append(update)
                
                # Cache update during partition
                edge_coordinator.network_partition_handler.cache_update({
                    "client_id": client_info.client_id,
                    "update": update,
                    "timestamp": datetime.now()
                })
            
            # Perform local aggregation during partition
            partition_aggregate = await edge_coordinator.aggregate_local_models(partition_updates)
            assert partition_aggregate is not None
            
            # Simulate partition recovery
            edge_coordinator.network_partition_handler.exit_partition_mode()
            assert not edge_coordinator.network_partition_handler.is_partitioned
            
            # Prepare recovery data
            recovery_data = edge_coordinator.network_partition_handler.prepare_recovery_data()
            assert recovery_data is not None
            assert len(recovery_data["cached_updates"]) == 3
            assert "partition_duration" in recovery_data
            
            # Sync with global server after recovery
            edge_update = ModelUpdate(
                client_id="edge_test",
                model_weights=pickle.dumps(partition_aggregate["aggregated_weights"]),
                training_metrics=partition_aggregate["aggregated_metrics"],
                data_statistics={"num_samples": partition_aggregate["total_samples"]},
                computation_time=partition_aggregate["computation_time"],
                network_conditions={"latency": 10, "bandwidth": 500},
                privacy_budget_used=0.3
            )
            
            success = await global_server.receive_model_update("edge_test", edge_update)
            assert success
            
            # Verify recovery completed successfully
            status = await global_server.get_server_status()
            assert status["total_clients"] == 1
            assert status["current_round"] > 0
            
        finally:
            await global_server.shutdown()
            await edge_coordinator.shutdown()
    
    async def test_intermittent_connectivity(self, test_config):
        """Test handling of intermittent client connectivity."""
        server = AggregationServer(test_config)
        await server.initialize()
        
        try:
            # Register clients with different connectivity patterns
            stable_clients = []
            unstable_clients = []
            
            # Stable clients
            for i in range(3):
                client_info = ClientInfo(
                    client_id=f"stable_client_{i}",
                    client_type="Simulated",
                    capabilities={"cpu_cores": 4, "memory_gb": 8},
                    location={"lat": 37.0 + i, "lon": -122.0},
                    network_info={"bandwidth": 100, "latency": 10, "reliability": 0.99},
                    hardware_specs={"connection": "ethernet"},
                    reputation_score=1.0
                )
                
                token = await server.register_client(client_info)
                stable_clients.append((client_info, token))
            
            # Unstable clients (mobile with poor connectivity)
            for i in range(2):
                client_info = ClientInfo(
                    client_id=f"unstable_client_{i}",
                    client_type="Mobile",
                    capabilities={"cpu_cores": 2, "memory_gb": 4},
                    location={"lat": 37.5 + i, "lon": -122.5},
                    network_info={"bandwidth": 20, "latency": 100, "reliability": 0.7},
                    hardware_specs={"connection": "cellular"},
                    reputation_score=0.8
                )
                
                token = await server.register_client(client_info)
                unstable_clients.append((client_info, token))
            
            # Simulate multiple rounds with intermittent connectivity
            for round_num in range(3):
                print(f"Starting round {round_num + 1}")
                
                # Stable clients always participate
                for client_info, token in stable_clients:
                    weights = {
                        "layer1": np.random.randn(15, 8).astype(np.float32),
                        "layer2": np.random.randn(8, 3).astype(np.float32)
                    }
                    
                    model_update = ModelUpdate(
                        client_id=client_info.client_id,
                        model_weights=pickle.dumps(weights),
                        training_metrics={"loss": 0.4 - round_num * 0.05, "accuracy": 0.75 + round_num * 0.03},
                        data_statistics={"num_samples": 1000},
                        computation_time=25.0,
                        network_conditions={"latency": 10, "bandwidth": 100},
                        privacy_budget_used=0.1
                    )
                    
                    success = await server.receive_model_update(client_info.client_id, model_update)
                    assert success
                
                # Unstable clients participate randomly
                for i, (client_info, token) in enumerate(unstable_clients):
                    # Simulate 70% participation rate
                    if np.random.random() < 0.7:
                        weights = {
                            "layer1": np.random.randn(15, 8).astype(np.float32),
                            "layer2": np.random.randn(8, 3).astype(np.float32)
                        }
                        
                        model_update = ModelUpdate(
                            client_id=client_info.client_id,
                            model_weights=pickle.dumps(weights),
                            training_metrics={"loss": 0.5 - round_num * 0.04, "accuracy": 0.7 + round_num * 0.025},
                            data_statistics={"num_samples": 800},
                            computation_time=40.0,  # Slower due to resource constraints
                            network_conditions={"latency": 100, "bandwidth": 20},
                            privacy_budget_used=0.12
                        )
                        
                        # Simulate potential connection failures
                        if np.random.random() < 0.3:  # 30% chance of failure
                            print(f"Connection failed for {client_info.client_id} in round {round_num + 1}")
                            continue
                        
                        success = await server.receive_model_update(client_info.client_id, model_update)
                        if success:
                            print(f"Update received from {client_info.client_id} in round {round_num + 1}")
                
                # Wait for aggregation
                await asyncio.sleep(1.5)
                
                # Verify round completion
                status = await server.get_server_status()
                assert status["current_round"] == round_num + 1
            
            # Verify system handled intermittent connectivity gracefully
            convergence_history = await server.get_convergence_history()
            assert len(convergence_history) == 3
            
            # Check that stable clients maintained consistent participation
            for round_metrics in convergence_history:
                assert round_metrics["participating_clients"] >= 3  # At least stable clients
                assert round_metrics["participating_clients"] <= 5  # At most all clients
            
        finally:
            await server.shutdown()


@pytest.mark.integration
@pytest.mark.slow
class TestLongRunningWorkflows:
    """Test long-running federated learning workflows."""
    
    async def test_multi_round_convergence(self, test_config):
        """Test convergence over multiple federated learning rounds."""
        # Reduce aggregation interval for faster testing
        test_config.federated_learning.aggregation_interval = 0.5
        
        server = AggregationServer(test_config)
        await server.initialize()
        
        try:
            # Register clients with different data quality
            clients = []
            for i in range(4):
                client_info = ClientInfo(
                    client_id=f"convergence_client_{i}",
                    client_type="Simulated",
                    capabilities={"cpu_cores": 4, "memory_gb": 8},
                    location={"lat": 37.0 + i, "lon": -122.0},
                    network_info={"bandwidth": 100, "latency": 10},
                    hardware_specs={"high_quality": i < 2},  # First 2 clients have high quality data
                    reputation_score=0.9 + i * 0.025
                )
                
                token = await server.register_client(client_info)
                clients.append((client_info, token))
            
            # Run multiple rounds
            num_rounds = 10
            initial_loss = 1.0
            
            for round_num in range(num_rounds):
                # Simulate learning progress - loss decreases over rounds
                current_loss = initial_loss * (0.9 ** round_num)
                current_accuracy = 0.5 + (0.4 * (1 - 0.9 ** round_num))
                
                for i, (client_info, token) in enumerate(clients):
                    # High quality clients have better metrics
                    quality_factor = 1.2 if client_info.hardware_specs.get("high_quality") else 1.0
                    
                    weights = {
                        "layer1": np.random.randn(12, 6).astype(np.float32) * (1.0 / (round_num + 1)),
                        "layer2": np.random.randn(6, 3).astype(np.float32) * (1.0 / (round_num + 1)),
                        "layer3": np.random.randn(3, 1).astype(np.float32) * (1.0 / (round_num + 1))
                    }
                    
                    model_update = ModelUpdate(
                        client_id=client_info.client_id,
                        model_weights=pickle.dumps(weights),
                        training_metrics={
                            "loss": current_loss / quality_factor,
                            "accuracy": current_accuracy * quality_factor,
                            "convergence_rate": 0.1 * quality_factor
                        },
                        data_statistics={"num_samples": 1000 * int(quality_factor)},
                        computation_time=20.0 + round_num * 2,  # Slightly increasing computation time
                        network_conditions={"latency": 10, "bandwidth": 100},
                        privacy_budget_used=0.1
                    )
                    
                    success = await server.receive_model_update(client_info.client_id, model_update)
                    assert success
                
                # Wait for aggregation
                await asyncio.sleep(1.0)
                
                # Check intermediate progress
                if round_num % 3 == 0:
                    convergence_history = await server.get_convergence_history()
                    if len(convergence_history) > 0:
                        latest_metrics = convergence_history[-1]
                        print(f"Round {round_num}: Loss={latest_metrics.get('average_loss', 'N/A'):.4f}, "
                              f"Accuracy={latest_metrics.get('average_accuracy', 'N/A'):.4f}")
            
            # Verify convergence
            convergence_history = await server.get_convergence_history()
            assert len(convergence_history) >= num_rounds
            
            # Check that loss decreased and accuracy increased
            first_round = convergence_history[0]
            last_round = convergence_history[-1]
            
            if "average_loss" in first_round and "average_loss" in last_round:
                assert last_round["average_loss"] < first_round["average_loss"]
            
            if "average_accuracy" in first_round and "average_accuracy" in last_round:
                assert last_round["average_accuracy"] > first_round["average_accuracy"]
            
            # Verify final model quality
            for client_info, _ in clients:
                global_model = await server.get_global_model(client_info.client_id)
                assert global_model is not None
                assert global_model["version"] == num_rounds
            
        finally:
            await server.shutdown()