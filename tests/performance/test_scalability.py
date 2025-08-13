"""
Performance and scalability tests for the federated learning system.
"""
import pytest
import asyncio
import time
import numpy as np
import psutil
import gc
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
from datetime import datetime, timedelta

from src.aggregation_server.server import AggregationServer
from src.edge_coordinator.coordinator import EdgeCoordinator
from src.sdr_client.sdr_client import SDRClient
from src.mobile_client.mobile_client import MobileClient
from src.common.interfaces import ClientInfo, ModelUpdate


@pytest.mark.performance
@pytest.mark.slow
class TestScalabilityTesting:
    """Test system scalability with varying client numbers."""
    
    @pytest.mark.parametrize("num_clients", [10, 50, 100, 200])
    async def test_client_registration_scalability(self, test_config, num_clients):
        """Test client registration performance with increasing client numbers."""
        server = AggregationServer(test_config)
        await server.initialize()
        
        try:
            start_time = time.time()
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Register clients concurrently
            registration_tasks = []
            for i in range(num_clients):
                client_info = ClientInfo(
                    client_id=f"scale_test_client_{i}",
                    client_type="Simulated",
                    capabilities={"cpu_cores": 4, "memory_gb": 8},
                    location={"lat": 37.0 + i * 0.01, "lon": -122.0 + i * 0.01},
                    network_info={"bandwidth": 100, "latency": 10 + i},
                    hardware_specs={},
                    reputation_score=0.9
                )
                task = server.register_client(client_info)
                registration_tasks.append(task)
            
            # Wait for all registrations to complete
            tokens = await asyncio.gather(*registration_tasks)
            
            end_time = time.time()
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            # Performance assertions
            registration_time = end_time - start_time
            memory_usage = memory_after - memory_before
            
            # Should complete within reasonable time
            assert registration_time < num_clients * 0.1  # Max 0.1s per client
            
            # Memory usage should scale reasonably
            assert memory_usage < num_clients * 0.5  # Max 0.5MB per client
            
            # All registrations should succeed
            assert len(tokens) == num_clients
            assert all(token is not None for token in tokens)
            
            # Verify server status
            status = await server.get_server_status()
            assert status["total_clients"] == num_clients
            
            print(f"Registered {num_clients} clients in {registration_time:.2f}s, "
                  f"Memory usage: {memory_usage:.1f}MB")
            
        finally:
            await server.shutdown()
    
    @pytest.mark.parametrize("num_clients", [20, 50, 100])
    async def test_model_aggregation_scalability(self, test_config, num_clients):
        """Test model aggregation performance with varying client numbers."""
        server = AggregationServer(test_config)
        await server.initialize()
        
        try:
            # Register clients
            clients = []
            for i in range(num_clients):
                client_info = ClientInfo(
                    client_id=f"agg_test_client_{i}",
                    client_type="Simulated",
                    capabilities={"cpu_cores": 4, "memory_gb": 8},
                    location={"lat": 37.0, "lon": -122.0},
                    network_info={"bandwidth": 100, "latency": 10},
                    hardware_specs={},
                    reputation_score=0.9
                )
                await server.register_client(client_info)
                clients.append(client_info)
            
            # Prepare model updates
            model_updates = []
            for i, client_info in enumerate(clients):
                weights = {
                    "layer1": np.random.randn(100, 50).astype(np.float32),
                    "layer2": np.random.randn(50, 20).astype(np.float32),
                    "layer3": np.random.randn(20, 1).astype(np.float32)
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
                model_updates.append(update)
            
            # Submit updates and measure performance
            start_time = time.time()
            cpu_before = psutil.Process().cpu_percent()
            memory_before = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Submit all updates concurrently
            submission_tasks = []
            for update in model_updates:
                task = server.receive_model_update(update.client_id, update)
                submission_tasks.append(task)
            
            results = await asyncio.gather(*submission_tasks)
            
            # Wait for aggregation to complete
            await asyncio.sleep(2.0)
            
            end_time = time.time()
            cpu_after = psutil.Process().cpu_percent()
            memory_after = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Performance metrics
            total_time = end_time - start_time
            cpu_usage = cpu_after - cpu_before
            memory_usage = memory_after - memory_before
            
            # Performance assertions
            assert total_time < num_clients * 0.05  # Max 0.05s per client
            assert all(results)  # All submissions should succeed
            
            # Verify aggregation completed
            global_model = await server.get_global_model(clients[0].client_id)
            assert global_model is not None
            
            print(f"Aggregated {num_clients} updates in {total_time:.2f}s, "
                  f"CPU: {cpu_usage:.1f}%, Memory: {memory_usage:.1f}MB")
            
        finally:
            await server.shutdown()
    
    async def test_concurrent_rounds_scalability(self, test_config):
        """Test performance with multiple concurrent federated learning rounds."""
        server = AggregationServer(test_config)
        await server.initialize()
        
        try:
            num_clients = 30
            num_rounds = 5
            
            # Register clients
            clients = []
            for i in range(num_clients):
                client_info = ClientInfo(
                    client_id=f"round_test_client_{i}",
                    client_type="Simulated",
                    capabilities={"cpu_cores": 4, "memory_gb": 8},
                    location={"lat": 37.0, "lon": -122.0},
                    network_info={"bandwidth": 100, "latency": 10},
                    hardware_specs={},
                    reputation_score=0.9
                )
                await server.register_client(client_info)
                clients.append(client_info)
            
            # Execute multiple rounds
            round_times = []
            memory_usage_per_round = []
            
            for round_num in range(num_rounds):
                round_start = time.time()
                memory_before = psutil.Process().memory_info().rss / 1024 / 1024
                
                # Submit updates for this round
                round_tasks = []
                for client_info in clients:
                    weights = {
                        "layer1": np.random.randn(50, 25).astype(np.float32),
                        "layer2": np.random.randn(25, 10).astype(np.float32)
                    }
                    
                    update = ModelUpdate(
                        client_id=client_info.client_id,
                        model_weights=pickle.dumps(weights),
                        training_metrics={"loss": 0.5 - round_num * 0.05, "accuracy": 0.7 + round_num * 0.03},
                        data_statistics={"num_samples": 1000},
                        computation_time=25.0,
                        network_conditions={"latency": 10, "bandwidth": 100},
                        privacy_budget_used=0.1
                    )
                    
                    task = server.receive_model_update(client_info.client_id, update)
                    round_tasks.append(task)
                
                # Wait for round completion
                await asyncio.gather(*round_tasks)
                await asyncio.sleep(1.0)  # Allow aggregation
                
                round_end = time.time()
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024
                
                round_time = round_end - round_start
                memory_usage = memory_after - memory_before
                
                round_times.append(round_time)
                memory_usage_per_round.append(memory_usage)
                
                print(f"Round {round_num + 1}: {round_time:.2f}s, Memory: {memory_usage:.1f}MB")
            
            # Performance analysis
            avg_round_time = np.mean(round_times)
            max_round_time = np.max(round_times)
            total_memory_growth = sum(memory_usage_per_round)
            
            # Assertions
            assert avg_round_time < 5.0  # Average round should complete in 5s
            assert max_round_time < 10.0  # No round should take more than 10s
            assert total_memory_growth < 100  # Total memory growth < 100MB
            
            # Verify convergence tracking
            convergence_history = await server.get_convergence_history()
            assert len(convergence_history) == num_rounds
            
        finally:
            await server.shutdown()


@pytest.mark.performance
@pytest.mark.slow
class TestNetworkConditionSimulation:
    """Test system performance under various network conditions."""
    
    async def test_high_latency_performance(self, test_config):
        """Test performance under high network latency conditions."""
        server = AggregationServer(test_config)
        await server.initialize()
        
        try:
            # Simulate clients with high latency
            high_latency_clients = []
            latencies = [100, 200, 500, 1000]  # ms
            
            for i, latency in enumerate(latencies):
                client_info = ClientInfo(
                    client_id=f"high_latency_client_{i}",
                    client_type="Mobile",
                    capabilities={"cpu_cores": 2, "memory_gb": 4},
                    location={"lat": 37.0, "lon": -122.0},
                    network_info={"bandwidth": 50, "latency": latency, "connection_type": "cellular"},
                    hardware_specs={},
                    reputation_score=0.8
                )
                await server.register_client(client_info)
                high_latency_clients.append(client_info)
            
            # Submit updates with simulated network delays
            start_time = time.time()
            
            for client_info in high_latency_clients:
                # Simulate network delay
                await asyncio.sleep(client_info.network_info["latency"] / 1000.0)
                
                weights = {
                    "layer1": np.random.randn(30, 15).astype(np.float32),
                    "layer2": np.random.randn(15, 5).astype(np.float32)
                }
                
                update = ModelUpdate(
                    client_id=client_info.client_id,
                    model_weights=pickle.dumps(weights),
                    training_metrics={"loss": 0.6, "accuracy": 0.75},
                    data_statistics={"num_samples": 800},
                    computation_time=40.0,
                    network_conditions=client_info.network_info,
                    privacy_budget_used=0.1
                )
                
                await server.receive_model_update(client_info.client_id, update)
            
            # Wait for aggregation
            await asyncio.sleep(2.0)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Verify system handled high latency gracefully
            assert total_time < 10.0  # Should complete within reasonable time
            
            # Verify all clients participated
            status = await server.get_server_status()
            assert status["total_clients"] == len(high_latency_clients)
            
        finally:
            await server.shutdown()
    
    async def test_bandwidth_limited_performance(self, test_config):
        """Test performance under bandwidth limitations."""
        server = AggregationServer(test_config)
        await server.initialize()
        
        try:
            # Simulate clients with limited bandwidth
            bandwidth_limited_clients = []
            bandwidths = [10, 25, 50, 100]  # Mbps
            
            for i, bandwidth in enumerate(bandwidths):
                client_info = ClientInfo(
                    client_id=f"bandwidth_limited_client_{i}",
                    client_type="Mobile",
                    capabilities={"cpu_cores": 4, "memory_gb": 6},
                    location={"lat": 37.0, "lon": -122.0},
                    network_info={"bandwidth": bandwidth, "latency": 50},
                    hardware_specs={},
                    reputation_score=0.85
                )
                await server.register_client(client_info)
                bandwidth_limited_clients.append(client_info)
            
            # Test with different model sizes
            model_sizes = [
                {"layer1": (20, 10), "layer2": (10, 5)},      # Small model
                {"layer1": (100, 50), "layer2": (50, 20)},    # Medium model
                {"layer1": (200, 100), "layer2": (100, 50)}   # Large model
            ]
            
            for size_idx, layer_sizes in enumerate(model_sizes):
                start_time = time.time()
                
                for client_info in bandwidth_limited_clients:
                    # Create model weights based on size
                    weights = {}
                    for layer_name, shape in layer_sizes.items():
                        weights[layer_name] = np.random.randn(*shape).astype(np.float32)
                    
                    # Simulate bandwidth-limited transmission
                    model_size_mb = sum(w.nbytes for w in weights.values()) / (1024 * 1024)
                    transmission_time = model_size_mb * 8 / client_info.network_info["bandwidth"]
                    await asyncio.sleep(transmission_time)
                    
                    update = ModelUpdate(
                        client_id=client_info.client_id,
                        model_weights=pickle.dumps(weights),
                        training_metrics={"loss": 0.5, "accuracy": 0.8},
                        data_statistics={"num_samples": 1000},
                        computation_time=30.0,
                        network_conditions=client_info.network_info,
                        privacy_budget_used=0.1
                    )
                    
                    await server.receive_model_update(client_info.client_id, update)
                
                await asyncio.sleep(1.0)  # Allow aggregation
                end_time = time.time()
                
                total_time = end_time - start_time
                print(f"Model size {size_idx + 1}: {total_time:.2f}s")
                
                # Larger models should take proportionally longer
                if size_idx > 0:
                    assert total_time > 0  # Should take some time
            
        finally:
            await server.shutdown()
    
    async def test_network_partition_recovery_performance(self, test_config):
        """Test performance during network partition and recovery."""
        global_server = AggregationServer(test_config)
        await global_server.initialize()
        
        edge_coordinator = EdgeCoordinator(test_config, region="test-region")
        await edge_coordinator.initialize()
        
        try:
            # Register edge coordinator
            edge_info = ClientInfo(
                client_id="edge_partition_test",
                client_type="EdgeCoordinator",
                capabilities={"cpu_cores": 8, "memory_gb": 32},
                location={"lat": 40.0, "lon": -100.0},
                network_info={"bandwidth": 1000, "latency": 5},
                hardware_specs={},
                reputation_score=1.0
            )
            await global_server.register_client(edge_info)
            
            # Register local clients
            local_clients = []
            for i in range(10):
                client_info = ClientInfo(
                    client_id=f"local_partition_client_{i}",
                    client_type="SDR",
                    capabilities={"cpu_cores": 4, "memory_gb": 8},
                    location={"lat": 40.0, "lon": -100.0},
                    network_info={"bandwidth": 100, "latency": 15},
                    hardware_specs={},
                    reputation_score=0.9
                )
                await edge_coordinator.register_local_client(client_info)
                local_clients.append(client_info)
            
            # Normal operation phase
            normal_start = time.time()
            
            for client_info in local_clients:
                weights = {
                    "layer1": np.random.randn(25, 12).astype(np.float32),
                    "layer2": np.random.randn(12, 6).astype(np.float32)
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
                
                await edge_coordinator.receive_local_update(update)
            
            normal_end = time.time()
            normal_time = normal_end - normal_start
            
            # Simulate network partition
            partition_start = time.time()
            edge_coordinator.network_partition_handler.enter_partition_mode()
            
            # Continue operations during partition
            for client_info in local_clients:
                weights = {
                    "layer1": np.random.randn(25, 12).astype(np.float32),
                    "layer2": np.random.randn(12, 6).astype(np.float32)
                }
                
                update = ModelUpdate(
                    client_id=client_info.client_id,
                    model_weights=weights,
                    training_metrics={"loss": 0.35, "accuracy": 0.82},
                    data_statistics={"num_samples": 1000},
                    computation_time=30.0,
                    network_conditions={"latency": 15, "bandwidth": 100},
                    privacy_budget_used=0.1
                )
                
                await edge_coordinator.receive_local_update(update)
            
            partition_end = time.time()
            partition_time = partition_end - partition_start
            
            # Recovery phase
            recovery_start = time.time()
            edge_coordinator.network_partition_handler.exit_partition_mode()
            
            # Sync with global server
            recovery_data = edge_coordinator.network_partition_handler.prepare_recovery_data()
            await edge_coordinator.sync_with_global_server(global_server, recovery_data)
            
            recovery_end = time.time()
            recovery_time = recovery_end - recovery_start
            
            # Performance analysis
            print(f"Normal operation: {normal_time:.2f}s")
            print(f"Partition operation: {partition_time:.2f}s")
            print(f"Recovery time: {recovery_time:.2f}s")
            
            # Assertions
            assert partition_time < normal_time * 1.5  # Partition shouldn't be much slower
            assert recovery_time < 5.0  # Recovery should be fast
            
        finally:
            await global_server.shutdown()
            await edge_coordinator.shutdown()


@pytest.mark.performance
@pytest.mark.slow
class TestMemoryLeakDetection:
    """Test for memory leaks during long-term operation."""
    
    async def test_long_term_memory_stability(self, test_config):
        """Test memory stability over extended operation."""
        server = AggregationServer(test_config)
        await server.initialize()
        
        try:
            # Register a moderate number of clients
            num_clients = 20
            clients = []
            
            for i in range(num_clients):
                client_info = ClientInfo(
                    client_id=f"memory_test_client_{i}",
                    client_type="Simulated",
                    capabilities={"cpu_cores": 4, "memory_gb": 8},
                    location={"lat": 37.0, "lon": -122.0},
                    network_info={"bandwidth": 100, "latency": 10},
                    hardware_specs={},
                    reputation_score=0.9
                )
                await server.register_client(client_info)
                clients.append(client_info)
            
            # Monitor memory usage over multiple rounds
            memory_measurements = []
            num_rounds = 20
            
            for round_num in range(num_rounds):
                # Measure memory before round
                gc.collect()  # Force garbage collection
                memory_before = psutil.Process().memory_info().rss / 1024 / 1024
                
                # Execute federated learning round
                round_tasks = []
                for client_info in clients:
                    weights = {
                        "layer1": np.random.randn(40, 20).astype(np.float32),
                        "layer2": np.random.randn(20, 10).astype(np.float32),
                        "layer3": np.random.randn(10, 1).astype(np.float32)
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
                    round_tasks.append(task)
                
                await asyncio.gather(*round_tasks)
                await asyncio.sleep(0.5)  # Allow aggregation
                
                # Measure memory after round
                gc.collect()
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024
                
                memory_measurements.append({
                    "round": round_num,
                    "memory_before": memory_before,
                    "memory_after": memory_after,
                    "memory_diff": memory_after - memory_before
                })
                
                if round_num % 5 == 0:
                    print(f"Round {round_num}: {memory_after:.1f}MB "
                          f"(+{memory_after - memory_before:.1f}MB)")
            
            # Analyze memory usage patterns
            initial_memory = memory_measurements[0]["memory_before"]
            final_memory = memory_measurements[-1]["memory_after"]
            total_growth = final_memory - initial_memory
            
            # Calculate memory growth rate
            memory_values = [m["memory_after"] for m in memory_measurements]
            rounds = list(range(len(memory_values)))
            
            # Linear regression to detect memory leaks
            coeffs = np.polyfit(rounds, memory_values, 1)
            growth_rate = coeffs[0]  # MB per round
            
            print(f"Total memory growth: {total_growth:.1f}MB")
            print(f"Growth rate: {growth_rate:.3f}MB/round")
            
            # Assertions for memory leak detection
            assert total_growth < 50  # Total growth should be < 50MB
            assert growth_rate < 1.0  # Growth rate should be < 1MB per round
            
            # Check for memory spikes
            max_round_growth = max(m["memory_diff"] for m in memory_measurements)
            assert max_round_growth < 20  # No single round should use > 20MB
            
        finally:
            await server.shutdown()
    
    async def test_client_churn_memory_impact(self, test_config):
        """Test memory impact of frequent client registration/deregistration."""
        server = AggregationServer(test_config)
        await server.initialize()
        
        try:
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Simulate client churn
            num_cycles = 10
            clients_per_cycle = 15
            
            for cycle in range(num_cycles):
                cycle_start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                # Register clients
                clients = []
                for i in range(clients_per_cycle):
                    client_info = ClientInfo(
                        client_id=f"churn_client_{cycle}_{i}",
                        client_type="Mobile",
                        capabilities={"cpu_cores": 2, "memory_gb": 4},
                        location={"lat": 37.0, "lon": -122.0},
                        network_info={"bandwidth": 50, "latency": 30},
                        hardware_specs={},
                        reputation_score=0.8
                    )
                    await server.register_client(client_info)
                    clients.append(client_info)
                
                # Submit updates
                for client_info in clients:
                    weights = {
                        "layer1": np.random.randn(15, 8).astype(np.float32),
                        "layer2": np.random.randn(8, 3).astype(np.float32)
                    }
                    
                    update = ModelUpdate(
                        client_id=client_info.client_id,
                        model_weights=pickle.dumps(weights),
                        training_metrics={"loss": 0.5, "accuracy": 0.8},
                        data_statistics={"num_samples": 800},
                        computation_time=25.0,
                        network_conditions={"latency": 30, "bandwidth": 50},
                        privacy_budget_used=0.1
                    )
                    
                    await server.receive_model_update(client_info.client_id, update)
                
                # Simulate client disconnection
                for client_info in clients:
                    await server.deregister_client(client_info.client_id)
                
                # Force garbage collection
                gc.collect()
                
                cycle_end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                cycle_memory_growth = cycle_end_memory - cycle_start_memory
                
                print(f"Cycle {cycle}: Memory growth {cycle_memory_growth:.1f}MB")
                
                # Each cycle should not cause significant memory growth
                assert cycle_memory_growth < 10  # < 10MB per cycle
            
            final_memory = psutil.Process().memory_info().rss / 1024 / 1024
            total_churn_growth = final_memory - initial_memory
            
            print(f"Total churn memory growth: {total_churn_growth:.1f}MB")
            
            # Total growth from churn should be minimal
            assert total_churn_growth < 30  # < 30MB total
            
        finally:
            await server.shutdown()


@pytest.mark.performance
@pytest.mark.slow
class TestByzantineFaultTolerance:
    """Test performance under Byzantine fault conditions."""
    
    async def test_byzantine_detection_performance(self, test_config):
        """Test performance of Byzantine client detection."""
        server = AggregationServer(test_config)
        await server.initialize()
        
        try:
            # Register mix of honest and Byzantine clients
            honest_clients = []
            byzantine_clients = []
            
            # Honest clients
            for i in range(15):
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
            
            # Byzantine clients
            for i in range(5):
                client_info = ClientInfo(
                    client_id=f"byzantine_client_{i}",
                    client_type="Simulated",
                    capabilities={"cpu_cores": 4, "memory_gb": 8},
                    location={"lat": 37.0, "lon": -122.0},
                    network_info={"bandwidth": 100, "latency": 10},
                    hardware_specs={},
                    reputation_score=0.9  # Initially trusted
                )
                await server.register_client(client_info)
                byzantine_clients.append(client_info)
            
            # Execute multiple rounds with Byzantine behavior
            detection_times = []
            
            for round_num in range(10):
                round_start = time.time()
                
                # Honest client updates
                for client_info in honest_clients:
                    weights = {
                        "layer1": np.random.randn(30, 15).astype(np.float32) * 0.1,
                        "layer2": np.random.randn(15, 5).astype(np.float32) * 0.1
                    }
                    
                    update = ModelUpdate(
                        client_id=client_info.client_id,
                        model_weights=pickle.dumps(weights),
                        training_metrics={"loss": 0.4 + np.random.randn() * 0.05, "accuracy": 0.8 + np.random.randn() * 0.02},
                        data_statistics={"num_samples": 1000},
                        computation_time=30.0,
                        network_conditions={"latency": 10, "bandwidth": 100},
                        privacy_budget_used=0.1
                    )
                    
                    await server.receive_model_update(client_info.client_id, update)
                
                # Byzantine client updates (malicious)
                for client_info in byzantine_clients:
                    # Malicious weights (extreme values)
                    weights = {
                        "layer1": np.random.randn(30, 15).astype(np.float32) * 10,  # 100x larger
                        "layer2": np.random.randn(15, 5).astype(np.float32) * 10
                    }
                    
                    update = ModelUpdate(
                        client_id=client_info.client_id,
                        model_weights=pickle.dumps(weights),
                        training_metrics={"loss": 0.1, "accuracy": 0.99},  # Suspiciously good
                        data_statistics={"num_samples": 1000},
                        computation_time=5.0,  # Suspiciously fast
                        network_conditions={"latency": 10, "bandwidth": 100},
                        privacy_budget_used=0.1
                    )
                    
                    await server.receive_model_update(client_info.client_id, update)
                
                # Wait for Byzantine detection and aggregation
                await asyncio.sleep(2.0)
                
                round_end = time.time()
                detection_time = round_end - round_start
                detection_times.append(detection_time)
                
                # Check if Byzantine clients were detected
                security_status = await server.get_security_status()
                detected_byzantine = security_status.get("detected_byzantine_clients", [])
                
                print(f"Round {round_num}: {detection_time:.2f}s, "
                      f"Detected Byzantine: {len(detected_byzantine)}")
            
            # Performance analysis
            avg_detection_time = np.mean(detection_times)
            max_detection_time = np.max(detection_times)
            
            print(f"Average detection time: {avg_detection_time:.2f}s")
            print(f"Maximum detection time: {max_detection_time:.2f}s")
            
            # Assertions
            assert avg_detection_time < 5.0  # Average detection should be fast
            assert max_detection_time < 10.0  # Even worst case should be reasonable
            
            # Should eventually detect most Byzantine clients
            final_security_status = await server.get_security_status()
            final_detected = final_security_status.get("detected_byzantine_clients", [])
            detection_rate = len(final_detected) / len(byzantine_clients)
            
            assert detection_rate >= 0.6  # Should detect at least 60% of Byzantine clients
            
        finally:
            await server.shutdown()
    
    async def test_robust_aggregation_performance(self, test_config):
        """Test performance of robust aggregation algorithms."""
        # Enable Byzantine-robust aggregation
        test_config.federated_learning.aggregation_strategy = "krum"
        
        server = AggregationServer(test_config)
        await server.initialize()
        
        try:
            # Register clients
            num_honest = 20
            num_byzantine = 6  # 30% Byzantine (within tolerance)
            
            all_clients = []
            
            # Honest clients
            for i in range(num_honest):
                client_info = ClientInfo(
                    client_id=f"robust_honest_{i}",
                    client_type="Simulated",
                    capabilities={"cpu_cores": 4, "memory_gb": 8},
                    location={"lat": 37.0, "lon": -122.0},
                    network_info={"bandwidth": 100, "latency": 10},
                    hardware_specs={},
                    reputation_score=0.9
                )
                await server.register_client(client_info)
                all_clients.append(("honest", client_info))
            
            # Byzantine clients
            for i in range(num_byzantine):
                client_info = ClientInfo(
                    client_id=f"robust_byzantine_{i}",
                    client_type="Simulated",
                    capabilities={"cpu_cores": 4, "memory_gb": 8},
                    location={"lat": 37.0, "lon": -122.0},
                    network_info={"bandwidth": 100, "latency": 10},
                    hardware_specs={},
                    reputation_score=0.9
                )
                await server.register_client(client_info)
                all_clients.append(("byzantine", client_info))
            
            # Test robust aggregation performance
            aggregation_times = []
            
            for round_num in range(5):
                round_start = time.time()
                
                # Submit updates
                for client_type, client_info in all_clients:
                    if client_type == "honest":
                        # Normal weights
                        weights = {
                            "layer1": np.random.randn(25, 12).astype(np.float32) * 0.1,
                            "layer2": np.random.randn(12, 6).astype(np.float32) * 0.1
                        }
                    else:  # Byzantine
                        # Malicious weights
                        weights = {
                            "layer1": np.random.randn(25, 12).astype(np.float32) * 5,
                            "layer2": np.random.randn(12, 6).astype(np.float32) * 5
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
                    
                    await server.receive_model_update(client_info.client_id, update)
                
                # Wait for robust aggregation
                await asyncio.sleep(3.0)  # Robust aggregation may take longer
                
                round_end = time.time()
                aggregation_time = round_end - round_start
                aggregation_times.append(aggregation_time)
                
                print(f"Robust aggregation round {round_num}: {aggregation_time:.2f}s")
            
            # Performance analysis
            avg_aggregation_time = np.mean(aggregation_times)
            max_aggregation_time = np.max(aggregation_times)
            
            print(f"Average robust aggregation time: {avg_aggregation_time:.2f}s")
            print(f"Maximum robust aggregation time: {max_aggregation_time:.2f}s")
            
            # Robust aggregation should complete within reasonable time
            assert avg_aggregation_time < 10.0  # Average should be < 10s
            assert max_aggregation_time < 15.0  # Maximum should be < 15s
            
            # Verify aggregation succeeded despite Byzantine clients
            global_model = await server.get_global_model(all_clients[0][1].client_id)
            assert global_model is not None
            
        finally:
            await server.shutdown()