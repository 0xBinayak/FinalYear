"""
Stress testing for the federated learning system under extreme conditions.
"""
import pytest
import asyncio
import time
import numpy as np
import psutil
import gc
import threading
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor, as_completed
import pickle
from datetime import datetime, timedelta
import random

from src.aggregation_server.server import AggregationServer
from src.edge_coordinator.coordinator import EdgeCoordinator
from src.sdr_client.sdr_client import SDRClient
from src.mobile_client.mobile_client import MobileClient
from src.common.interfaces import ClientInfo, ModelUpdate


@pytest.mark.stress
@pytest.mark.slow
class TestExtremeLoadTesting:
    """Test system behavior under extreme load conditions."""
    
    async def test_maximum_client_capacity(self, test_config):
        """Test system with maximum number of concurrent clients."""
        # Increase limits for stress testing
        test_config.federated_learning.max_clients = 500
        
        server = AggregationServer(test_config)
        await server.initialize()
        
        try:
            max_clients = 500
            batch_size = 50  # Register in batches to avoid overwhelming
            
            all_clients = []
            registration_times = []
            
            # Register clients in batches
            for batch_start in range(0, max_clients, batch_size):
                batch_end = min(batch_start + batch_size, max_clients)
                batch_start_time = time.time()
                
                batch_tasks = []
                for i in range(batch_start, batch_end):
                    client_info = ClientInfo(
                        client_id=f"stress_client_{i}",
                        client_type="Simulated",
                        capabilities={"cpu_cores": 2, "memory_gb": 4},
                        location={"lat": 37.0 + i * 0.001, "lon": -122.0 + i * 0.001},
                        network_info={"bandwidth": 50 + random.randint(-20, 20), "latency": 10 + random.randint(0, 50)},
                        hardware_specs={},
                        reputation_score=0.8 + random.random() * 0.2
                    )
                    
                    task = server.register_client(client_info)
                    batch_tasks.append((task, client_info))
                
                # Wait for batch completion
                batch_results = await asyncio.gather(*[task for task, _ in batch_tasks], return_exceptions=True)
                
                batch_end_time = time.time()
                batch_time = batch_end_time - batch_start_time
                registration_times.append(batch_time)
                
                # Count successful registrations
                successful_registrations = sum(1 for result in batch_results if not isinstance(result, Exception))
                
                print(f"Batch {batch_start//batch_size + 1}: {successful_registrations}/{batch_size} clients "
                      f"registered in {batch_time:.2f}s")
                
                # Add successfully registered clients
                for (task, client_info), result in zip(batch_tasks, batch_results):
                    if not isinstance(result, Exception):
                        all_clients.append(client_info)
                
                # Brief pause between batches
                await asyncio.sleep(0.5)
            
            print(f"Total registered clients: {len(all_clients)}")
            
            # Verify server status
            status = await server.get_server_status()
            assert status["total_clients"] == len(all_clients)
            
            # Test system responsiveness with maximum clients
            response_start = time.time()
            health_status = await server.get_health_status()
            response_time = time.time() - response_start
            
            assert response_time < 5.0  # Should respond within 5 seconds
            assert health_status["status"] in ["healthy", "degraded"]  # May be degraded under stress
            
            # Test partial model update submission (not all clients at once)
            subset_size = min(100, len(all_clients))
            selected_clients = random.sample(all_clients, subset_size)
            
            update_start = time.time()
            update_tasks = []
            
            for client_info in selected_clients:
                weights = {
                    "layer1": np.random.randn(20, 10).astype(np.float32),
                    "layer2": np.random.randn(10, 5).astype(np.float32)
                }
                
                update = ModelUpdate(
                    client_id=client_info.client_id,
                    model_weights=pickle.dumps(weights),
                    training_metrics={"loss": 0.5, "accuracy": 0.8},
                    data_statistics={"num_samples": 1000},
                    computation_time=30.0,
                    network_conditions=client_info.network_info,
                    privacy_budget_used=0.1
                )
                
                task = server.receive_model_update(client_info.client_id, update)
                update_tasks.append(task)
            
            update_results = await asyncio.gather(*update_tasks, return_exceptions=True)
            update_end = time.time()
            update_time = update_end - update_start
            
            successful_updates = sum(1 for result in update_results if result is True)
            print(f"Submitted {successful_updates}/{subset_size} updates in {update_time:.2f}s")
            
            # Should handle at least 80% of updates successfully
            success_rate = successful_updates / subset_size
            assert success_rate >= 0.8
            
        finally:
            await server.shutdown()
    
    async def test_rapid_client_churn(self, test_config):
        """Test system under rapid client registration/deregistration."""
        server = AggregationServer(test_config)
        await server.initialize()
        
        try:
            churn_duration = 60  # 60 seconds of churn
            clients_per_second = 10
            client_lifetime = 5  # Average client lifetime in seconds
            
            start_time = time.time()
            active_clients = {}
            total_registrations = 0
            total_deregistrations = 0
            client_counter = 0
            
            while time.time() - start_time < churn_duration:
                current_time = time.time()
                
                # Register new clients
                for _ in range(clients_per_second):
                    client_id = f"churn_client_{client_counter}"
                    client_counter += 1
                    
                    client_info = ClientInfo(
                        client_id=client_id,
                        client_type="Mobile",
                        capabilities={"cpu_cores": 2, "memory_gb": 3},
                        location={"lat": 37.0, "lon": -122.0},
                        network_info={"bandwidth": 25, "latency": 100},
                        hardware_specs={},
                        reputation_score=0.7
                    )
                    
                    try:
                        await server.register_client(client_info)
                        active_clients[client_id] = {
                            "info": client_info,
                            "registration_time": current_time
                        }
                        total_registrations += 1
                    except Exception as e:
                        print(f"Registration failed for {client_id}: {e}")
                
                # Deregister old clients
                clients_to_remove = []
                for client_id, client_data in active_clients.items():
                    if current_time - client_data["registration_time"] > client_lifetime:
                        clients_to_remove.append(client_id)
                
                for client_id in clients_to_remove:
                    try:
                        await server.deregister_client(client_id)
                        del active_clients[client_id]
                        total_deregistrations += 1
                    except Exception as e:
                        print(f"Deregistration failed for {client_id}: {e}")
                
                # Brief pause
                await asyncio.sleep(1.0)
                
                # Periodic status check
                if int(current_time - start_time) % 10 == 0:
                    status = await server.get_server_status()
                    print(f"Time {int(current_time - start_time)}s: "
                          f"{len(active_clients)} active clients, "
                          f"{total_registrations} total registrations, "
                          f"{total_deregistrations} total deregistrations")
            
            # Final cleanup
            for client_id in list(active_clients.keys()):
                try:
                    await server.deregister_client(client_id)
                    total_deregistrations += 1
                except:
                    pass
            
            print(f"Churn test completed: {total_registrations} registrations, "
                  f"{total_deregistrations} deregistrations")
            
            # System should remain stable
            final_status = await server.get_server_status()
            assert final_status["status"] in ["healthy", "degraded"]
            
            # Should have processed significant churn
            assert total_registrations > 300  # At least 5 minutes * 60s * 10 clients/s
            assert total_deregistrations > 200  # Most clients should have been deregistered
            
        finally:
            await server.shutdown()
    
    async def test_large_model_stress(self, test_config):
        """Test system with extremely large model updates."""
        server = AggregationServer(test_config)
        await server.initialize()
        
        try:
            # Register clients
            num_clients = 20
            clients = []
            
            for i in range(num_clients):
                client_info = ClientInfo(
                    client_id=f"large_model_client_{i}",
                    client_type="Simulated",
                    capabilities={"cpu_cores": 8, "memory_gb": 16},
                    location={"lat": 37.0, "lon": -122.0},
                    network_info={"bandwidth": 1000, "latency": 5},  # High bandwidth for large models
                    hardware_specs={},
                    reputation_score=0.9
                )
                await server.register_client(client_info)
                clients.append(client_info)
            
            # Test with increasingly large models
            model_sizes = [
                {"layer1": (100, 50), "layer2": (50, 20)},        # ~40KB
                {"layer1": (500, 250), "layer2": (250, 100)},     # ~1MB
                {"layer1": (1000, 500), "layer2": (500, 200)},    # ~4MB
                {"layer1": (2000, 1000), "layer2": (1000, 400)}   # ~16MB
            ]
            
            for size_idx, layer_sizes in enumerate(model_sizes):
                print(f"Testing model size {size_idx + 1}")
                
                # Calculate approximate model size
                total_params = sum(np.prod(shape) for shape in layer_sizes.values())
                model_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
                print(f"Approximate model size: {model_size_mb:.1f}MB")
                
                start_time = time.time()
                memory_before = psutil.Process().memory_info().rss / 1024 / 1024
                
                # Submit large model updates
                update_tasks = []
                for client_info in clients:
                    weights = {}
                    for layer_name, shape in layer_sizes.items():
                        weights[layer_name] = np.random.randn(*shape).astype(np.float32)
                    
                    update = ModelUpdate(
                        client_id=client_info.client_id,
                        model_weights=pickle.dumps(weights),
                        training_metrics={"loss": 0.5, "accuracy": 0.8},
                        data_statistics={"num_samples": 1000},
                        computation_time=60.0,  # Longer computation for large models
                        network_conditions=client_info.network_info,
                        privacy_budget_used=0.1
                    )
                    
                    task = server.receive_model_update(client_info.client_id, update)
                    update_tasks.append(task)
                
                # Wait for all updates
                results = await asyncio.gather(*update_tasks, return_exceptions=True)
                
                # Wait for aggregation
                await asyncio.sleep(5.0)
                
                end_time = time.time()
                memory_after = psutil.Process().memory_info().rss / 1024 / 1024
                
                processing_time = end_time - start_time
                memory_usage = memory_after - memory_before
                
                print(f"Size {size_idx + 1}: {processing_time:.2f}s, Memory: {memory_usage:.1f}MB")
                
                # Verify results
                successful_updates = sum(1 for result in results if result is True)
                success_rate = successful_updates / len(clients)
                
                assert success_rate >= 0.8  # At least 80% success rate
                
                # Verify aggregation completed
                global_model = await server.get_global_model(clients[0].client_id)
                assert global_model is not None
                
                # Memory usage should be reasonable
                assert memory_usage < model_size_mb * num_clients * 2  # At most 2x expected
                
                # Force garbage collection between tests
                gc.collect()
                await asyncio.sleep(1.0)
            
        finally:
            await server.shutdown()
    
    async def test_concurrent_aggregation_rounds(self, test_config):
        """Test system with multiple concurrent aggregation rounds."""
        server = AggregationServer(test_config)
        await server.initialize()
        
        try:
            # Register clients
            num_clients = 30
            clients = []
            
            for i in range(num_clients):
                client_info = ClientInfo(
                    client_id=f"concurrent_client_{i}",
                    client_type="Simulated",
                    capabilities={"cpu_cores": 4, "memory_gb": 8},
                    location={"lat": 37.0, "lon": -122.0},
                    network_info={"bandwidth": 100, "latency": 10},
                    hardware_specs={},
                    reputation_score=0.9
                )
                await server.register_client(client_info)
                clients.append(client_info)
            
            # Launch multiple concurrent rounds
            num_concurrent_rounds = 5
            round_tasks = []
            
            async def execute_round(round_id):
                """Execute a single federated learning round."""
                round_start = time.time()
                
                # Submit updates for this round
                update_tasks = []
                for client_info in clients:
                    weights = {
                        "layer1": np.random.randn(30, 15).astype(np.float32),
                        "layer2": np.random.randn(15, 8).astype(np.float32),
                        "layer3": np.random.randn(8, 1).astype(np.float32)
                    }
                    
                    update = ModelUpdate(
                        client_id=client_info.client_id,
                        model_weights=pickle.dumps(weights),
                        training_metrics={"loss": 0.5 - round_id * 0.02, "accuracy": 0.75 + round_id * 0.01},
                        data_statistics={"num_samples": 1000},
                        computation_time=30.0,
                        network_conditions=client_info.network_info,
                        privacy_budget_used=0.1
                    )
                    
                    task = server.receive_model_update(client_info.client_id, update)
                    update_tasks.append(task)
                
                # Wait for all updates in this round
                results = await asyncio.gather(*update_tasks, return_exceptions=True)
                
                # Wait for aggregation
                await asyncio.sleep(2.0)
                
                round_end = time.time()
                round_time = round_end - round_start
                
                successful_updates = sum(1 for result in results if result is True)
                success_rate = successful_updates / len(clients)
                
                return {
                    "round_id": round_id,
                    "round_time": round_time,
                    "success_rate": success_rate,
                    "successful_updates": successful_updates
                }
            
            # Launch all rounds concurrently
            start_time = time.time()
            
            for round_id in range(num_concurrent_rounds):
                task = execute_round(round_id)
                round_tasks.append(task)
            
            # Wait for all rounds to complete
            round_results = await asyncio.gather(*round_tasks, return_exceptions=True)
            
            end_time = time.time()
            total_time = end_time - start_time
            
            print(f"Completed {num_concurrent_rounds} concurrent rounds in {total_time:.2f}s")
            
            # Analyze results
            successful_rounds = [r for r in round_results if not isinstance(r, Exception)]
            
            assert len(successful_rounds) >= num_concurrent_rounds * 0.8  # At least 80% success
            
            for result in successful_rounds:
                print(f"Round {result['round_id']}: {result['round_time']:.2f}s, "
                      f"Success rate: {result['success_rate']:.2f}")
                
                assert result['success_rate'] >= 0.7  # At least 70% updates successful
            
            # System should remain responsive
            health_status = await server.get_health_status()
            assert health_status["status"] in ["healthy", "degraded"]
            
        finally:
            await server.shutdown()


@pytest.mark.stress
@pytest.mark.slow
class TestResourceExhaustionTesting:
    """Test system behavior under resource exhaustion conditions."""
    
    async def test_memory_pressure_handling(self, test_config):
        """Test system behavior under memory pressure."""
        server = AggregationServer(test_config)
        await server.initialize()
        
        try:
            # Register clients
            num_clients = 50
            clients = []
            
            for i in range(num_clients):
                client_info = ClientInfo(
                    client_id=f"memory_pressure_client_{i}",
                    client_type="Simulated",
                    capabilities={"cpu_cores": 4, "memory_gb": 8},
                    location={"lat": 37.0, "lon": -122.0},
                    network_info={"bandwidth": 100, "latency": 10},
                    hardware_specs={},
                    reputation_score=0.9
                )
                await server.register_client(client_info)
                clients.append(client_info)
            
            # Create memory pressure by submitting many large updates
            memory_pressure_rounds = 10
            large_model_size = (1000, 500)  # Large model
            
            initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
            max_memory_usage = 0
            
            for round_num in range(memory_pressure_rounds):
                round_start_memory = psutil.Process().memory_info().rss / 1024 / 1024
                
                # Submit large updates
                update_tasks = []
                for client_info in clients:
                    weights = {
                        "large_layer": np.random.randn(*large_model_size).astype(np.float32)
                    }
                    
                    update = ModelUpdate(
                        client_id=client_info.client_id,
                        model_weights=pickle.dumps(weights),
                        training_metrics={"loss": 0.5, "accuracy": 0.8},
                        data_statistics={"num_samples": 1000},
                        computation_time=30.0,
                        network_conditions=client_info.network_info,
                        privacy_budget_used=0.1
                    )
                    
                    task = server.receive_model_update(client_info.client_id, update)
                    update_tasks.append(task)
                
                # Submit updates in batches to create memory pressure
                batch_size = 10
                for i in range(0, len(update_tasks), batch_size):
                    batch = update_tasks[i:i + batch_size]
                    await asyncio.gather(*batch, return_exceptions=True)
                    
                    # Check memory usage
                    current_memory = psutil.Process().memory_info().rss / 1024 / 1024
                    max_memory_usage = max(max_memory_usage, current_memory)
                    
                    # Brief pause between batches
                    await asyncio.sleep(0.1)
                
                # Wait for aggregation
                await asyncio.sleep(1.0)
                
                round_end_memory = psutil.Process().memory_info().rss / 1024 / 1024
                round_memory_usage = round_end_memory - round_start_memory
                
                print(f"Memory pressure round {round_num}: {round_memory_usage:.1f}MB increase, "
                      f"Total: {round_end_memory:.1f}MB")
                
                # Force garbage collection
                gc.collect()
                
                # System should remain responsive even under memory pressure
                try:
                    health_status = await asyncio.wait_for(server.get_health_status(), timeout=10.0)
                    assert health_status is not None
                except asyncio.TimeoutError:
                    pytest.fail("System became unresponsive under memory pressure")
            
            total_memory_growth = max_memory_usage - initial_memory
            print(f"Maximum memory usage: {max_memory_usage:.1f}MB "
                  f"(+{total_memory_growth:.1f}MB from initial)")
            
            # System should handle memory pressure gracefully
            # (May use significant memory but shouldn't crash)
            assert total_memory_growth < 2000  # Less than 2GB growth
            
        finally:
            await server.shutdown()
    
    async def test_cpu_intensive_operations(self, test_config):
        """Test system under CPU-intensive conditions."""
        server = AggregationServer(test_config)
        await server.initialize()
        
        try:
            # Register clients
            num_clients = 25
            clients = []
            
            for i in range(num_clients):
                client_info = ClientInfo(
                    client_id=f"cpu_intensive_client_{i}",
                    client_type="Simulated",
                    capabilities={"cpu_cores": 4, "memory_gb": 8},
                    location={"lat": 37.0, "lon": -122.0},
                    network_info={"bandwidth": 100, "latency": 10},
                    hardware_specs={},
                    reputation_score=0.9
                )
                await server.register_client(client_info)
                clients.append(client_info)
            
            # Create CPU-intensive workload with complex models
            cpu_intensive_rounds = 5
            complex_model_shapes = {
                "conv1": (64, 3, 7, 7),      # Convolutional layer
                "conv2": (128, 64, 5, 5),    # Convolutional layer
                "conv3": (256, 128, 3, 3),   # Convolutional layer
                "fc1": (1024, 256 * 4 * 4),  # Fully connected
                "fc2": (512, 1024),          # Fully connected
                "fc3": (10, 512)             # Output layer
            }
            
            cpu_usage_measurements = []
            
            for round_num in range(cpu_intensive_rounds):
                # Monitor CPU usage
                cpu_before = psutil.Process().cpu_percent()
                round_start = time.time()
                
                # Submit complex model updates
                update_tasks = []
                for client_info in clients:
                    weights = {}
                    for layer_name, shape in complex_model_shapes.items():
                        weights[layer_name] = np.random.randn(*shape).astype(np.float32)
                    
                    update = ModelUpdate(
                        client_id=client_info.client_id,
                        model_weights=pickle.dumps(weights),
                        training_metrics={"loss": 0.5, "accuracy": 0.8},
                        data_statistics={"num_samples": 1000},
                        computation_time=60.0,  # Simulate intensive computation
                        network_conditions=client_info.network_info,
                        privacy_budget_used=0.1
                    )
                    
                    task = server.receive_model_update(client_info.client_id, update)
                    update_tasks.append(task)
                
                # Process updates
                results = await asyncio.gather(*update_tasks, return_exceptions=True)
                
                # Wait for intensive aggregation
                await asyncio.sleep(3.0)
                
                round_end = time.time()
                cpu_after = psutil.Process().cpu_percent()
                
                round_time = round_end - round_start
                cpu_usage = cpu_after - cpu_before
                cpu_usage_measurements.append(cpu_usage)
                
                successful_updates = sum(1 for result in results if result is True)
                success_rate = successful_updates / len(clients)
                
                print(f"CPU intensive round {round_num}: {round_time:.2f}s, "
                      f"CPU usage: {cpu_usage:.1f}%, Success rate: {success_rate:.2f}")
                
                # System should handle CPU-intensive operations
                assert success_rate >= 0.7  # At least 70% success
                assert round_time < 30.0  # Should complete within reasonable time
            
            # Analyze CPU usage
            avg_cpu_usage = np.mean(cpu_usage_measurements)
            max_cpu_usage = np.max(cpu_usage_measurements)
            
            print(f"Average CPU usage: {avg_cpu_usage:.1f}%")
            print(f"Maximum CPU usage: {max_cpu_usage:.1f}%")
            
            # System should remain responsive despite high CPU usage
            final_health = await server.get_health_status()
            assert final_health["status"] in ["healthy", "degraded"]
            
        finally:
            await server.shutdown()
    
    async def test_disk_io_stress(self, test_config):
        """Test system under disk I/O stress."""
        server = AggregationServer(test_config)
        await server.initialize()
        
        try:
            # Register clients
            num_clients = 30
            clients = []
            
            for i in range(num_clients):
                client_info = ClientInfo(
                    client_id=f"disk_io_client_{i}",
                    client_type="Simulated",
                    capabilities={"cpu_cores": 4, "memory_gb": 8},
                    location={"lat": 37.0, "lon": -122.0},
                    network_info={"bandwidth": 100, "latency": 10},
                    hardware_specs={},
                    reputation_score=0.9
                )
                await server.register_client(client_info)
                clients.append(client_info)
            
            # Create disk I/O stress with frequent model storage/retrieval
            io_stress_rounds = 8
            
            for round_num in range(io_stress_rounds):
                round_start = time.time()
                
                # Submit updates that will trigger storage operations
                update_tasks = []
                for client_info in clients:
                    weights = {
                        "layer1": np.random.randn(200, 100).astype(np.float32),
                        "layer2": np.random.randn(100, 50).astype(np.float32),
                        "layer3": np.random.randn(50, 20).astype(np.float32)
                    }
                    
                    update = ModelUpdate(
                        client_id=client_info.client_id,
                        model_weights=pickle.dumps(weights),
                        training_metrics={"loss": 0.5, "accuracy": 0.8},
                        data_statistics={"num_samples": 1000},
                        computation_time=30.0,
                        network_conditions=client_info.network_info,
                        privacy_budget_used=0.1
                    )
                    
                    task = server.receive_model_update(client_info.client_id, update)
                    update_tasks.append(task)
                
                # Process updates (triggers storage I/O)
                results = await asyncio.gather(*update_tasks, return_exceptions=True)
                
                # Wait for aggregation and storage
                await asyncio.sleep(2.0)
                
                # Trigger additional I/O by retrieving models
                retrieval_tasks = []
                for client_info in clients[:10]:  # Retrieve for subset of clients
                    task = server.get_global_model(client_info.client_id)
                    retrieval_tasks.append(task)
                
                retrieved_models = await asyncio.gather(*retrieval_tasks, return_exceptions=True)
                
                round_end = time.time()
                round_time = round_end - round_start
                
                successful_updates = sum(1 for result in results if result is True)
                successful_retrievals = sum(1 for model in retrieved_models 
                                          if not isinstance(model, Exception) and model is not None)
                
                print(f"Disk I/O round {round_num}: {round_time:.2f}s, "
                      f"Updates: {successful_updates}/{len(clients)}, "
                      f"Retrievals: {successful_retrievals}/10")
                
                # System should handle I/O stress
                assert successful_updates >= len(clients) * 0.8  # 80% success rate
                assert successful_retrievals >= 8  # 80% retrieval success
                assert round_time < 15.0  # Reasonable completion time
            
            # Verify system stability after I/O stress
            final_status = await server.get_server_status()
            assert final_status["status"] in ["healthy", "degraded"]
            
        finally:
            await server.shutdown()


@pytest.mark.stress
@pytest.mark.slow
class TestFailureRecoveryTesting:
    """Test system recovery from various failure scenarios."""
    
    async def test_cascading_failure_recovery(self, test_config):
        """Test recovery from cascading failures."""
        # Set up hierarchical system
        global_server = AggregationServer(test_config)
        await global_server.initialize()
        
        edge_coordinators = []
        for i in range(3):
            coordinator = EdgeCoordinator(test_config, region=f"region_{i}")
            await coordinator.initialize()
            edge_coordinators.append(coordinator)
        
        try:
            # Register edge coordinators
            for i, coordinator in enumerate(edge_coordinators):
                edge_info = ClientInfo(
                    client_id=f"edge_cascade_{i}",
                    client_type="EdgeCoordinator",
                    capabilities={"cpu_cores": 8, "memory_gb": 32},
                    location={"lat": 40.0 + i, "lon": -100.0},
                    network_info={"bandwidth": 1000, "latency": 5},
                    hardware_specs={},
                    reputation_score=1.0
                )
                await global_server.register_client(edge_info)
            
            # Register local clients for each edge
            all_local_clients = []
            for coordinator in edge_coordinators:
                local_clients = []
                for j in range(5):
                    client_info = ClientInfo(
                        client_id=f"local_{coordinator.region}_{j}",
                        client_type="SDR",
                        capabilities={"cpu_cores": 4, "memory_gb": 8},
                        location={"lat": 40.0, "lon": -100.0},
                        network_info={"bandwidth": 100, "latency": 15},
                        hardware_specs={},
                        reputation_score=0.9
                    )
                    await coordinator.register_local_client(client_info)
                    local_clients.append(client_info)
                all_local_clients.extend(local_clients)
            
            # Normal operation
            print("Starting normal operation...")
            await self._execute_federated_round(global_server, edge_coordinators)
            
            # Simulate cascading failures
            print("Simulating cascading failures...")
            
            # Failure 1: One edge coordinator fails
            failed_coordinator = edge_coordinators[0]
            await failed_coordinator.simulate_failure()
            
            # Continue operation with remaining coordinators
            remaining_coordinators = edge_coordinators[1:]
            await self._execute_federated_round(global_server, remaining_coordinators)
            
            # Failure 2: Network partition affects another coordinator
            partitioned_coordinator = edge_coordinators[1]
            partitioned_coordinator.network_partition_handler.enter_partition_mode()
            
            # Continue with only one coordinator
            working_coordinators = [edge_coordinators[2]]
            await self._execute_federated_round(global_server, working_coordinators)
            
            # Recovery 1: Restore failed coordinator
            print("Starting recovery...")
            await failed_coordinator.recover_from_failure()
            working_coordinators.append(failed_coordinator)
            
            # Recovery 2: Resolve network partition
            partitioned_coordinator.network_partition_handler.exit_partition_mode()
            working_coordinators.append(partitioned_coordinator)
            
            # Verify full recovery
            await self._execute_federated_round(global_server, working_coordinators)
            
            # System should be fully operational
            final_status = await global_server.get_server_status()
            assert final_status["status"] in ["healthy", "degraded"]
            assert final_status["total_clients"] == len(edge_coordinators)
            
            print("Cascading failure recovery test completed successfully")
            
        finally:
            await global_server.shutdown()
            for coordinator in edge_coordinators:
                await coordinator.shutdown()
    
    async def _execute_federated_round(self, global_server, edge_coordinators):
        """Helper method to execute a federated learning round."""
        for coordinator in edge_coordinators:
            if coordinator.is_operational:
                # Simulate local aggregation
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
                
                if local_updates:
                    local_aggregate = await coordinator.aggregate_local_models(local_updates)
                    
                    # Submit to global server
                    edge_update = ModelUpdate(
                        client_id=f"edge_{coordinator.region}",
                        model_weights=pickle.dumps(local_aggregate["aggregated_weights"]),
                        training_metrics=local_aggregate["aggregated_metrics"],
                        data_statistics={"num_samples": local_aggregate["total_samples"]},
                        computation_time=local_aggregate["computation_time"],
                        network_conditions={"latency": 5, "bandwidth": 1000},
                        privacy_budget_used=0.3
                    )
                    
                    await global_server.receive_model_update(
                        f"edge_{coordinator.region}", edge_update
                    )
        
        # Wait for global aggregation
        await asyncio.sleep(1.0)
    
    async def test_data_corruption_recovery(self, test_config):
        """Test recovery from data corruption scenarios."""
        server = AggregationServer(test_config)
        await server.initialize()
        
        try:
            # Register clients
            num_clients = 20
            clients = []
            
            for i in range(num_clients):
                client_info = ClientInfo(
                    client_id=f"corruption_test_client_{i}",
                    client_type="Simulated",
                    capabilities={"cpu_cores": 4, "memory_gb": 8},
                    location={"lat": 37.0, "lon": -122.0},
                    network_info={"bandwidth": 100, "latency": 10},
                    hardware_specs={},
                    reputation_score=0.9
                )
                await server.register_client(client_info)
                clients.append(client_info)
            
            # Normal operation
            print("Executing normal round...")
            await self._execute_normal_round(server, clients)
            
            # Simulate data corruption scenarios
            corruption_scenarios = [
                "corrupted_model_weights",
                "invalid_metrics",
                "malformed_update",
                "network_transmission_error"
            ]
            
            for scenario in corruption_scenarios:
                print(f"Testing {scenario} recovery...")
                
                corrupted_updates = []
                normal_updates = []
                
                for i, client_info in enumerate(clients):
                    if i < 5:  # First 5 clients send corrupted data
                        update = self._create_corrupted_update(client_info, scenario)
                        corrupted_updates.append(update)
                    else:  # Remaining clients send normal data
                        update = self._create_normal_update(client_info)
                        normal_updates.append(update)
                
                # Submit all updates
                all_tasks = []
                for update in corrupted_updates + normal_updates:
                    task = server.receive_model_update(update.client_id, update)
                    all_tasks.append(task)
                
                results = await asyncio.gather(*all_tasks, return_exceptions=True)
                
                # Wait for processing
                await asyncio.sleep(2.0)
                
                # Analyze results
                successful_submissions = sum(1 for result in results if result is True)
                
                print(f"Scenario {scenario}: {successful_submissions}/{len(clients)} successful submissions")
                
                # System should handle corruption gracefully
                # Normal updates should succeed, corrupted ones may fail
                assert successful_submissions >= len(normal_updates)  # At least normal updates succeed
                
                # System should remain operational
                health_status = await server.get_health_status()
                assert health_status["status"] in ["healthy", "degraded"]
            
            # Final recovery verification
            print("Verifying final recovery...")
            await self._execute_normal_round(server, clients)
            
            final_status = await server.get_server_status()
            assert final_status["status"] in ["healthy", "degraded"]
            
        finally:
            await server.shutdown()
    
    def _create_normal_update(self, client_info):
        """Create a normal model update."""
        weights = {
            "layer1": np.random.randn(20, 10).astype(np.float32),
            "layer2": np.random.randn(10, 5).astype(np.float32)
        }
        
        return ModelUpdate(
            client_id=client_info.client_id,
            model_weights=pickle.dumps(weights),
            training_metrics={"loss": 0.5, "accuracy": 0.8},
            data_statistics={"num_samples": 1000},
            computation_time=30.0,
            network_conditions=client_info.network_info,
            privacy_budget_used=0.1
        )
    
    def _create_corrupted_update(self, client_info, corruption_type):
        """Create a corrupted model update based on corruption type."""
        if corruption_type == "corrupted_model_weights":
            # Corrupt model weights with NaN/Inf values
            weights = {
                "layer1": np.full((20, 10), np.nan, dtype=np.float32),
                "layer2": np.full((10, 5), np.inf, dtype=np.float32)
            }
            model_weights = pickle.dumps(weights)
            
        elif corruption_type == "invalid_metrics":
            weights = {
                "layer1": np.random.randn(20, 10).astype(np.float32),
                "layer2": np.random.randn(10, 5).astype(np.float32)
            }
            model_weights = pickle.dumps(weights)
            
        elif corruption_type == "malformed_update":
            # Completely invalid data
            model_weights = b"corrupted_data_not_pickle"
            
        elif corruption_type == "network_transmission_error":
            # Truncated data
            weights = {
                "layer1": np.random.randn(20, 10).astype(np.float32),
                "layer2": np.random.randn(10, 5).astype(np.float32)
            }
            full_data = pickle.dumps(weights)
            model_weights = full_data[:len(full_data)//2]  # Truncate
        
        # Create update with potentially invalid metrics for some scenarios
        if corruption_type == "invalid_metrics":
            training_metrics = {"loss": float('nan'), "accuracy": -1.5}  # Invalid values
        else:
            training_metrics = {"loss": 0.5, "accuracy": 0.8}
        
        return ModelUpdate(
            client_id=client_info.client_id,
            model_weights=model_weights,
            training_metrics=training_metrics,
            data_statistics={"num_samples": 1000},
            computation_time=30.0,
            network_conditions=client_info.network_info,
            privacy_budget_used=0.1
        )
    
    async def _execute_normal_round(self, server, clients):
        """Execute a normal federated learning round."""
        update_tasks = []
        for client_info in clients:
            update = self._create_normal_update(client_info)
            task = server.receive_model_update(client_info.client_id, update)
            update_tasks.append(task)
        
        results = await asyncio.gather(*update_tasks, return_exceptions=True)
        await asyncio.sleep(1.0)  # Allow aggregation
        
        successful_updates = sum(1 for result in results if result is True)
        success_rate = successful_updates / len(clients)
        
        assert success_rate >= 0.9  # 90% success rate for normal operation
        return success_rate