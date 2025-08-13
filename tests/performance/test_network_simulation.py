"""
Network condition simulation and testing for federated learning system.
"""
import pytest
import asyncio
import time
import numpy as np
import random
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime, timedelta

from src.aggregation_server.server import AggregationServer
from src.edge_coordinator.coordinator import EdgeCoordinator
from src.common.interfaces import ClientInfo, ModelUpdate


class NetworkSimulator:
    """Simulate various network conditions for testing."""
    
    def __init__(self):
        self.conditions = {
            "latency": 10,      # ms
            "bandwidth": 100,   # Mbps
            "packet_loss": 0.01, # 1%
            "jitter": 5,        # ms
            "connection_drops": 0.001  # 0.1%
        }
        self.active = False
    
    def set_conditions(self, **kwargs):
        """Set network conditions."""
        self.conditions.update(kwargs)
    
    async def simulate_latency(self, base_latency=None):
        """Simulate network latency with jitter."""
        if not self.active:
            return
        
        latency = base_latency or self.conditions["latency"]
        jitter = self.conditions["jitter"]
        
        # Add random jitter
        actual_latency = latency + random.uniform(-jitter, jitter)
        actual_latency = max(0, actual_latency)  # Ensure non-negative
        
        await asyncio.sleep(actual_latency / 1000.0)  # Convert ms to seconds
    
    async def simulate_packet_loss(self):
        """Simulate packet loss by randomly failing operations."""
        if not self.active:
            return False
        
        return random.random() < self.conditions["packet_loss"]
    
    async def simulate_bandwidth_limit(self, data_size_mb):
        """Simulate bandwidth limitations."""
        if not self.active:
            return
        
        bandwidth_mbps = self.conditions["bandwidth"]
        transmission_time = (data_size_mb * 8) / bandwidth_mbps  # Convert to seconds
        
        await asyncio.sleep(transmission_time)
    
    async def simulate_connection_drop(self):
        """Simulate connection drops."""
        if not self.active:
            return False
        
        return random.random() < self.conditions["connection_drops"]
    
    def start_simulation(self):
        """Start network simulation."""
        self.active = True
    
    def stop_simulation(self):
        """Stop network simulation."""
        self.active = False


@pytest.mark.performance
@pytest.mark.slow
class TestNetworkConditionSimulation:
    """Test system performance under various network conditions."""
    
    async def test_variable_latency_performance(self, test_config):
        """Test performance under variable network latency."""
        server = AggregationServer(test_config)
        await server.initialize()
        
        network_sim = NetworkSimulator()
        
        try:
            # Register clients
            num_clients = 20
            clients = []
            
            for i in range(num_clients):
                client_info = ClientInfo(
                    client_id=f"latency_test_client_{i}",
                    client_type="Mobile",
                    capabilities={"cpu_cores": 2, "memory_gb": 4},
                    location={"lat": 37.0 + i * 0.1, "lon": -122.0 + i * 0.1},
                    network_info={"bandwidth": 50, "latency": 50, "connection_type": "cellular"},
                    hardware_specs={},
                    reputation_score=0.8
                )
                await server.register_client(client_info)
                clients.append(client_info)
            
            # Test different latency scenarios
            latency_scenarios = [
                {"name": "Low Latency", "latency": 10, "jitter": 2},
                {"name": "Medium Latency", "latency": 100, "jitter": 20},
                {"name": "High Latency", "latency": 500, "jitter": 100},
                {"name": "Satellite", "latency": 800, "jitter": 200}
            ]
            
            results = []
            
            for scenario in latency_scenarios:
                print(f"Testing {scenario['name']} scenario...")
                
                # Configure network simulation
                network_sim.set_conditions(
                    latency=scenario["latency"],
                    jitter=scenario["jitter"]
                )
                network_sim.start_simulation()
                
                start_time = time.time()
                
                # Submit model updates with simulated latency
                update_tasks = []
                for client_info in clients:
                    # Simulate network latency before sending
                    await network_sim.simulate_latency()
                    
                    weights = {
                        "layer1": np.random.randn(30, 15).astype(np.float32),
                        "layer2": np.random.randn(15, 8).astype(np.float32)
                    }
                    
                    update = ModelUpdate(
                        client_id=client_info.client_id,
                        model_weights=pickle.dumps(weights),
                        training_metrics={"loss": 0.5, "accuracy": 0.8},
                        data_statistics={"num_samples": 1000},
                        computation_time=30.0,
                        network_conditions={
                            "latency": scenario["latency"],
                            "bandwidth": 50,
                            "jitter": scenario["jitter"]
                        },
                        privacy_budget_used=0.1
                    )
                    
                    task = server.receive_model_update(client_info.client_id, update)
                    update_tasks.append(task)
                
                # Wait for all updates with timeout
                try:
                    results_batch = await asyncio.wait_for(
                        asyncio.gather(*update_tasks, return_exceptions=True),
                        timeout=60.0
                    )
                except asyncio.TimeoutError:
                    results_batch = [False] * len(update_tasks)
                
                # Wait for aggregation
                await asyncio.sleep(2.0)
                
                end_time = time.time()
                total_time = end_time - start_time
                
                successful_updates = sum(1 for result in results_batch if result is True)
                success_rate = successful_updates / len(clients)
                
                network_sim.stop_simulation()
                
                result = {
                    "scenario": scenario["name"],
                    "latency": scenario["latency"],
                    "total_time": total_time,
                    "success_rate": success_rate,
                    "successful_updates": successful_updates
                }
                results.append(result)
                
                print(f"{scenario['name']}: {total_time:.2f}s, "
                      f"Success rate: {success_rate:.2%}")
                
                # Performance assertions based on latency
                if scenario["latency"] < 100:
                    assert success_rate >= 0.9  # High success rate for low latency
                    assert total_time < 30.0    # Should complete quickly
                elif scenario["latency"] < 500:
                    assert success_rate >= 0.8  # Good success rate for medium latency
                    assert total_time < 60.0    # Reasonable completion time
                else:
                    assert success_rate >= 0.7  # Acceptable success rate for high latency
                    assert total_time < 120.0   # Extended completion time acceptable
            
            # Verify system remains responsive after all scenarios
            health_status = await server.get_health_status()
            assert health_status["status"] in ["healthy", "degraded"]
            
        finally:
            await server.shutdown()
    
    async def test_bandwidth_throttling_performance(self, test_config):
        """Test performance under bandwidth limitations."""
        server = AggregationServer(test_config)
        await server.initialize()
        
        network_sim = NetworkSimulator()
        
        try:
            # Register clients
            num_clients = 15
            clients = []
            
            for i in range(num_clients):
                client_info = ClientInfo(
                    client_id=f"bandwidth_test_client_{i}",
                    client_type="Mobile",
                    capabilities={"cpu_cores": 4, "memory_gb": 6},
                    location={"lat": 37.0, "lon": -122.0},
                    network_info={"bandwidth": 25, "latency": 50},
                    hardware_specs={},
                    reputation_score=0.85
                )
                await server.register_client(client_info)
                clients.append(client_info)
            
            # Test different bandwidth scenarios
            bandwidth_scenarios = [
                {"name": "High Bandwidth", "bandwidth": 1000, "expected_time": 10},
                {"name": "Medium Bandwidth", "bandwidth": 100, "expected_time": 20},
                {"name": "Low Bandwidth", "bandwidth": 10, "expected_time": 60},
                {"name": "Very Low Bandwidth", "bandwidth": 1, "expected_time": 180}
            ]
            
            # Different model sizes to test
            model_sizes = [
                {"name": "Small", "shapes": {"layer1": (20, 10), "layer2": (10, 5)}},
                {"name": "Medium", "shapes": {"layer1": (100, 50), "layer2": (50, 20)}},
                {"name": "Large", "shapes": {"layer1": (200, 100), "layer2": (100, 50)}}
            ]
            
            for model_size in model_sizes:
                print(f"\nTesting {model_size['name']} model size...")
                
                # Calculate approximate model size
                total_params = sum(np.prod(shape) for shape in model_size["shapes"].values())
                model_size_mb = total_params * 4 / (1024 * 1024)  # 4 bytes per float32
                
                for bandwidth_scenario in bandwidth_scenarios:
                    print(f"  {bandwidth_scenario['name']} ({bandwidth_scenario['bandwidth']} Mbps)...")
                    
                    # Configure network simulation
                    network_sim.set_conditions(bandwidth=bandwidth_scenario["bandwidth"])
                    network_sim.start_simulation()
                    
                    start_time = time.time()
                    
                    # Submit updates with bandwidth simulation
                    update_tasks = []
                    for client_info in clients:
                        # Create model weights
                        weights = {}
                        for layer_name, shape in model_size["shapes"].items():
                            weights[layer_name] = np.random.randn(*shape).astype(np.float32)
                        
                        # Simulate bandwidth-limited transmission
                        await network_sim.simulate_bandwidth_limit(model_size_mb)
                        
                        update = ModelUpdate(
                            client_id=client_info.client_id,
                            model_weights=pickle.dumps(weights),
                            training_metrics={"loss": 0.5, "accuracy": 0.8},
                            data_statistics={"num_samples": 1000},
                            computation_time=30.0,
                            network_conditions={
                                "bandwidth": bandwidth_scenario["bandwidth"],
                                "latency": 50
                            },
                            privacy_budget_used=0.1
                        )
                        
                        task = server.receive_model_update(client_info.client_id, update)
                        update_tasks.append(task)
                    
                    # Wait for updates with appropriate timeout
                    timeout = max(120, bandwidth_scenario["expected_time"] * 2)
                    try:
                        results_batch = await asyncio.wait_for(
                            asyncio.gather(*update_tasks, return_exceptions=True),
                            timeout=timeout
                        )
                    except asyncio.TimeoutError:
                        results_batch = [False] * len(update_tasks)
                    
                    await asyncio.sleep(1.0)  # Allow aggregation
                    
                    end_time = time.time()
                    total_time = end_time - start_time
                    
                    successful_updates = sum(1 for result in results_batch if result is True)
                    success_rate = successful_updates / len(clients)
                    
                    network_sim.stop_simulation()
                    
                    print(f"    Time: {total_time:.1f}s, Success: {success_rate:.2%}")
                    
                    # Performance assertions
                    if bandwidth_scenario["bandwidth"] >= 100:
                        assert success_rate >= 0.9  # High success rate for good bandwidth
                    elif bandwidth_scenario["bandwidth"] >= 10:
                        assert success_rate >= 0.8  # Good success rate for medium bandwidth
                    else:
                        assert success_rate >= 0.6  # Acceptable success rate for low bandwidth
                    
                    # Time should scale with bandwidth (inversely)
                    if bandwidth_scenario["bandwidth"] >= 100:
                        assert total_time < 60.0
                    elif bandwidth_scenario["bandwidth"] >= 10:
                        assert total_time < 120.0
                    else:
                        assert total_time < 300.0  # 5 minutes max for very low bandwidth
        
        finally:
            await server.shutdown()
    
    async def test_packet_loss_resilience(self, test_config):
        """Test system resilience to packet loss."""
        server = AggregationServer(test_config)
        await server.initialize()
        
        network_sim = NetworkSimulator()
        
        try:
            # Register clients
            num_clients = 25
            clients = []
            
            for i in range(num_clients):
                client_info = ClientInfo(
                    client_id=f"packet_loss_client_{i}",
                    client_type="Mobile",
                    capabilities={"cpu_cores": 4, "memory_gb": 8},
                    location={"lat": 37.0, "lon": -122.0},
                    network_info={"bandwidth": 50, "latency": 100},
                    hardware_specs={},
                    reputation_score=0.8
                )
                await server.register_client(client_info)
                clients.append(client_info)
            
            # Test different packet loss rates
            packet_loss_scenarios = [
                {"name": "No Loss", "packet_loss": 0.0, "min_success_rate": 0.95},
                {"name": "Low Loss", "packet_loss": 0.01, "min_success_rate": 0.9},
                {"name": "Medium Loss", "packet_loss": 0.05, "min_success_rate": 0.8},
                {"name": "High Loss", "packet_loss": 0.1, "min_success_rate": 0.7},
                {"name": "Very High Loss", "packet_loss": 0.2, "min_success_rate": 0.5}
            ]
            
            for scenario in packet_loss_scenarios:
                print(f"Testing {scenario['name']} scenario ({scenario['packet_loss']:.1%} loss)...")
                
                # Configure network simulation
                network_sim.set_conditions(packet_loss=scenario["packet_loss"])
                network_sim.start_simulation()
                
                # Run multiple rounds to get statistical significance
                round_results = []
                
                for round_num in range(3):
                    start_time = time.time()
                    
                    # Submit updates with packet loss simulation
                    successful_submissions = 0
                    
                    for client_info in clients:
                        # Simulate packet loss
                        if await network_sim.simulate_packet_loss():
                            continue  # Skip this update due to packet loss
                        
                        weights = {
                            "layer1": np.random.randn(25, 12).astype(np.float32),
                            "layer2": np.random.randn(12, 6).astype(np.float32)
                        }
                        
                        update = ModelUpdate(
                            client_id=client_info.client_id,
                            model_weights=pickle.dumps(weights),
                            training_metrics={"loss": 0.5, "accuracy": 0.8},
                            data_statistics={"num_samples": 1000},
                            computation_time=30.0,
                            network_conditions={
                                "packet_loss": scenario["packet_loss"],
                                "bandwidth": 50,
                                "latency": 100
                            },
                            privacy_budget_used=0.1
                        )
                        
                        try:
                            success = await server.receive_model_update(client_info.client_id, update)
                            if success:
                                successful_submissions += 1
                        except Exception:
                            pass  # Network error due to packet loss
                    
                    await asyncio.sleep(1.0)  # Allow processing
                    
                    end_time = time.time()
                    round_time = end_time - start_time
                    
                    success_rate = successful_submissions / len(clients)
                    round_results.append({
                        "round": round_num,
                        "success_rate": success_rate,
                        "successful_submissions": successful_submissions,
                        "round_time": round_time
                    })
                
                network_sim.stop_simulation()
                
                # Analyze results across rounds
                avg_success_rate = np.mean([r["success_rate"] for r in round_results])
                avg_round_time = np.mean([r["round_time"] for r in round_results])
                
                print(f"  Average success rate: {avg_success_rate:.2%}")
                print(f"  Average round time: {avg_round_time:.1f}s")
                
                # Verify system handles packet loss gracefully
                assert avg_success_rate >= scenario["min_success_rate"]
                
                # System should remain responsive despite packet loss
                assert avg_round_time < 30.0
                
                # Verify system health
                health_status = await server.get_health_status()
                assert health_status["status"] in ["healthy", "degraded"]
        
        finally:
            await server.shutdown()
    
    async def test_connection_instability(self, test_config):
        """Test system behavior with unstable connections."""
        server = AggregationServer(test_config)
        await server.initialize()
        
        network_sim = NetworkSimulator()
        
        try:
            # Register clients
            num_clients = 20
            clients = []
            
            for i in range(num_clients):
                client_info = ClientInfo(
                    client_id=f"unstable_client_{i}",
                    client_type="Mobile",
                    capabilities={"cpu_cores": 2, "memory_gb": 4},
                    location={"lat": 37.0, "lon": -122.0},
                    network_info={"bandwidth": 25, "latency": 150, "connection_type": "cellular"},
                    hardware_specs={},
                    reputation_score=0.7
                )
                await server.register_client(client_info)
                clients.append(client_info)
            
            # Simulate unstable network conditions
            network_sim.set_conditions(
                latency=150,
                jitter=75,
                bandwidth=25,
                packet_loss=0.03,
                connection_drops=0.05  # 5% chance of connection drop
            )
            network_sim.start_simulation()
            
            # Run federated learning rounds with connection instability
            num_rounds = 5
            round_results = []
            
            for round_num in range(num_rounds):
                print(f"Round {round_num + 1} with unstable connections...")
                
                start_time = time.time()
                active_clients = []
                
                # Simulate connection drops
                for client_info in clients:
                    if not await network_sim.simulate_connection_drop():
                        active_clients.append(client_info)
                
                print(f"  Active clients: {len(active_clients)}/{len(clients)}")
                
                # Submit updates from active clients
                update_tasks = []
                for client_info in active_clients:
                    # Simulate network conditions
                    await network_sim.simulate_latency()
                    
                    # Skip if packet loss occurs
                    if await network_sim.simulate_packet_loss():
                        continue
                    
                    weights = {
                        "layer1": np.random.randn(20, 10).astype(np.float32),
                        "layer2": np.random.randn(10, 5).astype(np.float32)
                    }
                    
                    update = ModelUpdate(
                        client_id=client_info.client_id,
                        model_weights=pickle.dumps(weights),
                        training_metrics={"loss": 0.5 - round_num * 0.02, "accuracy": 0.75 + round_num * 0.01},
                        data_statistics={"num_samples": 800},
                        computation_time=35.0,
                        network_conditions={
                            "latency": 150,
                            "bandwidth": 25,
                            "packet_loss": 0.03,
                            "connection_unstable": True
                        },
                        privacy_budget_used=0.1
                    )
                    
                    task = server.receive_model_update(client_info.client_id, update)
                    update_tasks.append(task)
                
                # Wait for updates with timeout
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*update_tasks, return_exceptions=True),
                        timeout=60.0
                    )
                except asyncio.TimeoutError:
                    results = [False] * len(update_tasks)
                
                await asyncio.sleep(2.0)  # Allow aggregation
                
                end_time = time.time()
                round_time = end_time - start_time
                
                successful_updates = sum(1 for result in results if result is True)
                participation_rate = len(active_clients) / len(clients)
                success_rate = successful_updates / len(active_clients) if active_clients else 0
                
                round_result = {
                    "round": round_num,
                    "participation_rate": participation_rate,
                    "success_rate": success_rate,
                    "round_time": round_time,
                    "active_clients": len(active_clients),
                    "successful_updates": successful_updates
                }
                round_results.append(round_result)
                
                print(f"  Participation: {participation_rate:.2%}, "
                      f"Success: {success_rate:.2%}, "
                      f"Time: {round_time:.1f}s")
                
                # System should handle instability gracefully
                assert participation_rate >= 0.5  # At least 50% clients should connect
                if active_clients:
                    assert success_rate >= 0.6  # At least 60% of active clients should succeed
                assert round_time < 90.0  # Should complete within reasonable time
            
            network_sim.stop_simulation()
            
            # Analyze overall performance
            avg_participation = np.mean([r["participation_rate"] for r in round_results])
            avg_success = np.mean([r["success_rate"] for r in round_results if r["active_clients"] > 0])
            avg_round_time = np.mean([r["round_time"] for r in round_results])
            
            print(f"\nOverall Results:")
            print(f"  Average participation rate: {avg_participation:.2%}")
            print(f"  Average success rate: {avg_success:.2%}")
            print(f"  Average round time: {avg_round_time:.1f}s")
            
            # System should maintain reasonable performance despite instability
            assert avg_participation >= 0.6  # 60% average participation
            assert avg_success >= 0.7       # 70% success rate for active clients
            assert avg_round_time < 60.0    # Average round time under 1 minute
            
            # Verify system remains operational
            final_status = await server.get_server_status()
            assert final_status["status"] in ["healthy", "degraded"]
            
        finally:
            await server.shutdown()


@pytest.mark.performance
class TestNetworkAdaptation:
    """Test system adaptation to changing network conditions."""
    
    async def test_adaptive_timeout_adjustment(self, test_config):
        """Test adaptive timeout adjustment based on network conditions."""
        server = AggregationServer(test_config)
        await server.initialize()
        
        try:
            # Register clients with different network characteristics
            client_types = [
                {"type": "fiber", "latency": 5, "bandwidth": 1000, "reliability": 0.99},
                {"type": "cable", "latency": 20, "bandwidth": 100, "reliability": 0.95},
                {"type": "cellular", "latency": 100, "bandwidth": 25, "reliability": 0.85},
                {"type": "satellite", "latency": 600, "bandwidth": 5, "reliability": 0.9}
            ]
            
            clients = []
            for i, client_type in enumerate(client_types * 5):  # 20 clients total
                client_info = ClientInfo(
                    client_id=f"adaptive_client_{i}",
                    client_type="Simulated",
                    capabilities={"cpu_cores": 4, "memory_gb": 8},
                    location={"lat": 37.0, "lon": -122.0},
                    network_info={
                        "bandwidth": client_type["bandwidth"],
                        "latency": client_type["latency"],
                        "connection_type": client_type["type"],
                        "reliability": client_type["reliability"]
                    },
                    hardware_specs={},
                    reputation_score=0.9
                )
                await server.register_client(client_info)
                clients.append(client_info)
            
            # Test adaptive behavior over multiple rounds
            for round_num in range(3):
                print(f"Adaptive round {round_num + 1}...")
                
                start_time = time.time()
                
                # Submit updates with varying network delays
                update_tasks = []
                for client_info in clients:
                    # Simulate network delay based on client type
                    network_delay = client_info.network_info["latency"] / 1000.0
                    await asyncio.sleep(network_delay)
                    
                    weights = {
                        "layer1": np.random.randn(30, 15).astype(np.float32),
                        "layer2": np.random.randn(15, 8).astype(np.float32)
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
                
                # Wait for updates with adaptive timeout
                adaptive_timeout = 30 + round_num * 10  # Increase timeout over rounds
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*update_tasks, return_exceptions=True),
                        timeout=adaptive_timeout
                    )
                except asyncio.TimeoutError:
                    results = [False] * len(update_tasks)
                
                await asyncio.sleep(2.0)
                
                end_time = time.time()
                round_time = end_time - start_time
                
                successful_updates = sum(1 for result in results if result is True)
                success_rate = successful_updates / len(clients)
                
                print(f"  Round {round_num + 1}: {success_rate:.2%} success, {round_time:.1f}s")
                
                # Success rate should improve with adaptive timeouts
                if round_num == 0:
                    assert success_rate >= 0.7  # Initial round may have some timeouts
                else:
                    assert success_rate >= 0.8  # Later rounds should adapt better
            
            # Verify system learned to adapt
            final_status = await server.get_server_status()
            assert final_status["status"] in ["healthy", "degraded"]
            
        finally:
            await server.shutdown()
    
    async def test_quality_of_service_prioritization(self, test_config):
        """Test QoS prioritization based on network conditions."""
        server = AggregationServer(test_config)
        await server.initialize()
        
        try:
            # Register clients with different QoS levels
            qos_levels = [
                {"level": "premium", "priority": 1, "bandwidth": 1000, "latency": 5},
                {"level": "standard", "priority": 2, "bandwidth": 100, "latency": 20},
                {"level": "basic", "priority": 3, "bandwidth": 25, "latency": 100}
            ]
            
            clients_by_qos = {}
            for qos in qos_levels:
                clients_by_qos[qos["level"]] = []
                for i in range(5):  # 5 clients per QoS level
                    client_info = ClientInfo(
                        client_id=f"{qos['level']}_client_{i}",
                        client_type="Simulated",
                        capabilities={"cpu_cores": 4, "memory_gb": 8},
                        location={"lat": 37.0, "lon": -122.0},
                        network_info={
                            "bandwidth": qos["bandwidth"],
                            "latency": qos["latency"],
                            "qos_level": qos["level"],
                            "priority": qos["priority"]
                        },
                        hardware_specs={},
                        reputation_score=0.9
                    )
                    await server.register_client(client_info)
                    clients_by_qos[qos["level"]].append(client_info)
            
            # Test QoS prioritization under load
            print("Testing QoS prioritization...")
            
            start_time = time.time()
            
            # Submit updates from all QoS levels simultaneously
            all_update_tasks = []
            submission_times = {}
            
            for qos_level, clients in clients_by_qos.items():
                submission_times[qos_level] = []
                
                for client_info in clients:
                    submit_start = time.time()
                    
                    weights = {
                        "layer1": np.random.randn(40, 20).astype(np.float32),
                        "layer2": np.random.randn(20, 10).astype(np.float32)
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
                    all_update_tasks.append((task, qos_level, submit_start))
            
            # Wait for all updates
            results = []
            for task, qos_level, submit_start in all_update_tasks:
                try:
                    result = await task
                    process_time = time.time() - submit_start
                    results.append({
                        "qos_level": qos_level,
                        "success": result,
                        "process_time": process_time
                    })
                except Exception:
                    results.append({
                        "qos_level": qos_level,
                        "success": False,
                        "process_time": float('inf')
                    })
            
            await asyncio.sleep(2.0)  # Allow aggregation
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Analyze QoS performance
            qos_performance = {}
            for qos_level in clients_by_qos.keys():
                qos_results = [r for r in results if r["qos_level"] == qos_level]
                
                success_rate = sum(1 for r in qos_results if r["success"]) / len(qos_results)
                avg_process_time = np.mean([r["process_time"] for r in qos_results if r["process_time"] != float('inf')])
                
                qos_performance[qos_level] = {
                    "success_rate": success_rate,
                    "avg_process_time": avg_process_time
                }
                
                print(f"  {qos_level.capitalize()}: {success_rate:.2%} success, {avg_process_time:.2f}s avg")
            
            # Verify QoS prioritization
            # Premium should have better performance than standard, standard better than basic
            assert qos_performance["premium"]["success_rate"] >= qos_performance["standard"]["success_rate"]
            assert qos_performance["standard"]["success_rate"] >= qos_performance["basic"]["success_rate"]
            
            # Premium should have faster processing
            assert qos_performance["premium"]["avg_process_time"] <= qos_performance["standard"]["avg_process_time"]
            assert qos_performance["standard"]["avg_process_time"] <= qos_performance["basic"]["avg_process_time"]
            
            print(f"Total QoS test time: {total_time:.1f}s")
            
        finally:
            await server.shutdown()