"""
Comprehensive benchmarking and profiling for federated learning system.
"""
import pytest
import asyncio
import time
import numpy as np
import psutil
import gc
import cProfile
import pstats
import io
from unittest.mock import Mock, patch
from concurrent.futures import ThreadPoolExecutor
import pickle
from datetime import datetime, timedelta
import json
import os

from src.aggregation_server.server import AggregationServer
from src.edge_coordinator.coordinator import EdgeCoordinator
from src.sdr_client.sdr_client import SDRClient
from src.mobile_client.mobile_client import MobileClient
from src.common.interfaces import ClientInfo, ModelUpdate


class PerformanceProfiler:
    """Performance profiling utility for benchmarking."""
    
    def __init__(self):
        self.profiler = None
        self.start_time = None
        self.measurements = []
    
    def start_profiling(self):
        """Start performance profiling."""
        self.profiler = cProfile.Profile()
        self.profiler.enable()
        self.start_time = time.time()
    
    def stop_profiling(self):
        """Stop performance profiling and return results."""
        if self.profiler:
            self.profiler.disable()
            
            # Get profiling results
            s = io.StringIO()
            ps = pstats.Stats(self.profiler, stream=s)
            ps.sort_stats('cumulative')
            ps.print_stats(20)  # Top 20 functions
            
            profile_output = s.getvalue()
            
            return {
                "duration": time.time() - self.start_time,
                "profile_output": profile_output,
                "stats": ps
            }
        return None
    
    def measure_resource_usage(self, label=""):
        """Measure current resource usage."""
        process = psutil.Process()
        
        measurement = {
            "timestamp": time.time(),
            "label": label,
            "cpu_percent": process.cpu_percent(),
            "memory_mb": process.memory_info().rss / 1024 / 1024,
            "memory_percent": process.memory_percent(),
            "num_threads": process.num_threads(),
            "num_fds": process.num_fds() if hasattr(process, 'num_fds') else 0
        }
        
        self.measurements.append(measurement)
        return measurement
    
    def get_resource_summary(self):
        """Get summary of resource usage measurements."""
        if not self.measurements:
            return {}
        
        cpu_values = [m["cpu_percent"] for m in self.measurements]
        memory_values = [m["memory_mb"] for m in self.measurements]
        
        return {
            "cpu": {
                "min": min(cpu_values),
                "max": max(cpu_values),
                "avg": sum(cpu_values) / len(cpu_values),
                "std": np.std(cpu_values)
            },
            "memory": {
                "min": min(memory_values),
                "max": max(memory_values),
                "avg": sum(memory_values) / len(memory_values),
                "std": np.std(memory_values)
            },
            "measurements_count": len(self.measurements)
        }


class BenchmarkSuite:
    """Comprehensive benchmark suite for federated learning system."""
    
    def __init__(self):
        self.results = {}
        self.profiler = PerformanceProfiler()
    
    async def run_client_registration_benchmark(self, server, num_clients_list):
        """Benchmark client registration performance."""
        results = {}
        
        for num_clients in num_clients_list:
            print(f"Benchmarking {num_clients} client registrations...")
            
            # Prepare clients
            clients = []
            for i in range(num_clients):
                client_info = ClientInfo(
                    client_id=f"benchmark_client_{i}",
                    client_type="Simulated",
                    capabilities={"cpu_cores": 4, "memory_gb": 8},
                    location={"lat": 37.0 + i * 0.001, "lon": -122.0 + i * 0.001},
                    network_info={"bandwidth": 100, "latency": 10},
                    hardware_specs={},
                    reputation_score=0.9
                )
                clients.append(client_info)
            
            # Benchmark registration
            self.profiler.start_profiling()
            self.profiler.measure_resource_usage("start")
            
            start_time = time.time()
            
            # Register clients concurrently
            registration_tasks = [server.register_client(client) for client in clients]
            tokens = await asyncio.gather(*registration_tasks)
            
            end_time = time.time()
            
            self.profiler.measure_resource_usage("end")
            profile_results = self.profiler.stop_profiling()
            
            # Calculate metrics
            total_time = end_time - start_time
            registrations_per_second = num_clients / total_time
            avg_time_per_registration = total_time / num_clients
            
            # Verify all registrations succeeded
            successful_registrations = sum(1 for token in tokens if token is not None)
            success_rate = successful_registrations / num_clients
            
            results[num_clients] = {
                "total_time": total_time,
                "registrations_per_second": registrations_per_second,
                "avg_time_per_registration": avg_time_per_registration,
                "success_rate": success_rate,
                "resource_usage": self.profiler.get_resource_summary(),
                "profile_duration": profile_results["duration"] if profile_results else 0
            }
            
            print(f"  {registrations_per_second:.1f} registrations/sec, "
                  f"{avg_time_per_registration*1000:.1f}ms avg, "
                  f"{success_rate:.2%} success")
            
            # Cleanup for next iteration
            for client in clients:
                try:
                    await server.deregister_client(client.client_id)
                except:
                    pass
            
            gc.collect()
            await asyncio.sleep(1.0)
        
        return results
    
    async def run_model_aggregation_benchmark(self, server, clients, model_sizes):
        """Benchmark model aggregation performance."""
        results = {}
        
        for size_name, layer_shapes in model_sizes.items():
            print(f"Benchmarking {size_name} model aggregation...")
            
            # Calculate model size
            total_params = sum(np.prod(shape) for shape in layer_shapes.values())
            model_size_mb = total_params * 4 / (1024 * 1024)
            
            # Prepare model updates
            updates = []
            for client_info in clients:
                weights = {}
                for layer_name, shape in layer_shapes.items():
                    weights[layer_name] = np.random.randn(*shape).astype(np.float32)
                
                update = ModelUpdate(
                    client_id=client_info.client_id,
                    model_weights=pickle.dumps(weights),
                    training_metrics={"loss": 0.5, "accuracy": 0.8},
                    data_statistics={"num_samples": 1000},
                    computation_time=30.0,
                    network_conditions={"latency": 10, "bandwidth": 100},
                    privacy_budget_used=0.1
                )
                updates.append(update)
            
            # Benchmark aggregation
            self.profiler.start_profiling()
            self.profiler.measure_resource_usage("start")
            
            start_time = time.time()
            
            # Submit all updates
            submission_tasks = [
                server.receive_model_update(update.client_id, update) 
                for update in updates
            ]
            submission_results = await asyncio.gather(*submission_tasks)
            
            # Wait for aggregation
            await asyncio.sleep(3.0)
            
            end_time = time.time()
            
            self.profiler.measure_resource_usage("end")
            profile_results = self.profiler.stop_profiling()
            
            # Calculate metrics
            total_time = end_time - start_time
            updates_per_second = len(clients) / total_time
            mb_per_second = (model_size_mb * len(clients)) / total_time
            
            successful_submissions = sum(1 for result in submission_results if result)
            success_rate = successful_submissions / len(clients)
            
            # Verify aggregation completed
            try:
                global_model = await server.get_global_model(clients[0].client_id)
                aggregation_successful = global_model is not None
            except:
                aggregation_successful = False
            
            results[size_name] = {
                "model_size_mb": model_size_mb,
                "total_params": total_params,
                "num_clients": len(clients),
                "total_time": total_time,
                "updates_per_second": updates_per_second,
                "mb_per_second": mb_per_second,
                "success_rate": success_rate,
                "aggregation_successful": aggregation_successful,
                "resource_usage": self.profiler.get_resource_summary(),
                "profile_duration": profile_results["duration"] if profile_results else 0
            }
            
            print(f"  {updates_per_second:.1f} updates/sec, "
                  f"{mb_per_second:.1f} MB/sec, "
                  f"{success_rate:.2%} success, "
                  f"Aggregation: {'✓' if aggregation_successful else '✗'}")
            
            gc.collect()
            await asyncio.sleep(2.0)
        
        return results
    
    async def run_concurrent_rounds_benchmark(self, server, clients, num_rounds):
        """Benchmark concurrent federated learning rounds."""
        print(f"Benchmarking {num_rounds} concurrent rounds...")
        
        async def execute_round(round_id):
            """Execute a single federated learning round."""
            round_start = time.time()
            
            # Prepare updates for this round
            update_tasks = []
            for client_info in clients:
                weights = {
                    "layer1": np.random.randn(25, 12).astype(np.float32),
                    "layer2": np.random.randn(12, 6).astype(np.float32)
                }
                
                update = ModelUpdate(
                    client_id=client_info.client_id,
                    model_weights=pickle.dumps(weights),
                    training_metrics={"loss": 0.5 - round_id * 0.01, "accuracy": 0.75 + round_id * 0.005},
                    data_statistics={"num_samples": 1000},
                    computation_time=30.0,
                    network_conditions={"latency": 10, "bandwidth": 100},
                    privacy_budget_used=0.1
                )
                
                task = server.receive_model_update(client_info.client_id, update)
                update_tasks.append(task)
            
            # Wait for all updates in this round
            results = await asyncio.gather(*update_tasks, return_exceptions=True)
            
            # Wait for aggregation
            await asyncio.sleep(1.0)
            
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
        
        # Benchmark concurrent rounds
        self.profiler.start_profiling()
        self.profiler.measure_resource_usage("start")
        
        start_time = time.time()
        
        # Execute all rounds concurrently
        round_tasks = [execute_round(i) for i in range(num_rounds)]
        round_results = await asyncio.gather(*round_tasks, return_exceptions=True)
        
        end_time = time.time()
        
        self.profiler.measure_resource_usage("end")
        profile_results = self.profiler.stop_profiling()
        
        # Analyze results
        successful_rounds = [r for r in round_results if not isinstance(r, Exception)]
        
        total_time = end_time - start_time
        rounds_per_second = len(successful_rounds) / total_time
        avg_round_time = np.mean([r["round_time"] for r in successful_rounds])
        avg_success_rate = np.mean([r["success_rate"] for r in successful_rounds])
        
        results = {
            "num_rounds": num_rounds,
            "successful_rounds": len(successful_rounds),
            "total_time": total_time,
            "rounds_per_second": rounds_per_second,
            "avg_round_time": avg_round_time,
            "avg_success_rate": avg_success_rate,
            "resource_usage": self.profiler.get_resource_summary(),
            "profile_duration": profile_results["duration"] if profile_results else 0,
            "round_details": successful_rounds
        }
        
        print(f"  {rounds_per_second:.2f} rounds/sec, "
              f"{avg_round_time:.1f}s avg round time, "
              f"{avg_success_rate:.2%} avg success")
        
        return results
    
    def save_results(self, filename):
        """Save benchmark results to file."""
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        print(f"Benchmark results saved to {filename}")


@pytest.mark.performance
@pytest.mark.slow
class TestComprehensiveBenchmarking:
    """Comprehensive benchmarking test suite."""
    
    async def test_full_system_benchmark(self, test_config):
        """Run comprehensive system benchmarks."""
        server = AggregationServer(test_config)
        await server.initialize()
        
        benchmark_suite = BenchmarkSuite()
        
        try:
            print("Starting comprehensive system benchmarks...")
            
            # 1. Client Registration Benchmark
            print("\n1. Client Registration Benchmark")
            client_counts = [10, 25, 50, 100]
            registration_results = await benchmark_suite.run_client_registration_benchmark(
                server, client_counts
            )
            benchmark_suite.results["client_registration"] = registration_results
            
            # Register a standard set of clients for other benchmarks
            num_standard_clients = 30
            standard_clients = []
            for i in range(num_standard_clients):
                client_info = ClientInfo(
                    client_id=f"standard_client_{i}",
                    client_type="Simulated",
                    capabilities={"cpu_cores": 4, "memory_gb": 8},
                    location={"lat": 37.0, "lon": -122.0},
                    network_info={"bandwidth": 100, "latency": 10},
                    hardware_specs={},
                    reputation_score=0.9
                )
                await server.register_client(client_info)
                standard_clients.append(client_info)
            
            # 2. Model Aggregation Benchmark
            print("\n2. Model Aggregation Benchmark")
            model_sizes = {
                "small": {"layer1": (20, 10), "layer2": (10, 5)},
                "medium": {"layer1": (100, 50), "layer2": (50, 20)},
                "large": {"layer1": (500, 250), "layer2": (250, 100)},
                "xlarge": {"layer1": (1000, 500), "layer2": (500, 200)}
            }
            aggregation_results = await benchmark_suite.run_model_aggregation_benchmark(
                server, standard_clients, model_sizes
            )
            benchmark_suite.results["model_aggregation"] = aggregation_results
            
            # 3. Concurrent Rounds Benchmark
            print("\n3. Concurrent Rounds Benchmark")
            concurrent_rounds_results = await benchmark_suite.run_concurrent_rounds_benchmark(
                server, standard_clients[:15], 5  # Use subset for concurrent test
            )
            benchmark_suite.results["concurrent_rounds"] = concurrent_rounds_results
            
            # 4. Memory Usage Benchmark
            print("\n4. Memory Usage Benchmark")
            memory_results = await self._run_memory_benchmark(server, standard_clients)
            benchmark_suite.results["memory_usage"] = memory_results
            
            # 5. Throughput Benchmark
            print("\n5. Throughput Benchmark")
            throughput_results = await self._run_throughput_benchmark(server, standard_clients)
            benchmark_suite.results["throughput"] = throughput_results
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"benchmark_results_{timestamp}.json"
            benchmark_suite.save_results(results_file)
            
            # Performance assertions
            self._validate_benchmark_results(benchmark_suite.results)
            
            print("\n✅ Comprehensive benchmarking completed successfully!")
            
        finally:
            await server.shutdown()
    
    async def _run_memory_benchmark(self, server, clients):
        """Benchmark memory usage patterns."""
        print("  Running memory usage benchmark...")
        
        initial_memory = psutil.Process().memory_info().rss / 1024 / 1024
        memory_measurements = []
        
        # Baseline measurement
        memory_measurements.append({
            "phase": "baseline",
            "memory_mb": initial_memory,
            "clients_registered": len(clients)
        })
        
        # Memory usage during multiple rounds
        for round_num in range(5):
            round_start_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            # Submit updates
            update_tasks = []
            for client_info in clients:
                weights = {
                    "layer1": np.random.randn(50, 25).astype(np.float32),
                    "layer2": np.random.randn(25, 12).astype(np.float32)
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
            
            await asyncio.gather(*update_tasks)
            await asyncio.sleep(1.0)
            
            round_end_memory = psutil.Process().memory_info().rss / 1024 / 1024
            
            memory_measurements.append({
                "phase": f"round_{round_num}",
                "memory_mb": round_end_memory,
                "memory_increase": round_end_memory - round_start_memory,
                "total_increase": round_end_memory - initial_memory
            })
            
            # Force garbage collection
            gc.collect()
        
        # Calculate memory statistics
        memory_increases = [m.get("memory_increase", 0) for m in memory_measurements[1:]]
        total_increase = memory_measurements[-1]["total_increase"]
        
        results = {
            "initial_memory_mb": initial_memory,
            "final_memory_mb": memory_measurements[-1]["memory_mb"],
            "total_increase_mb": total_increase,
            "avg_round_increase_mb": np.mean(memory_increases),
            "max_round_increase_mb": max(memory_increases),
            "memory_per_client_mb": total_increase / len(clients),
            "measurements": memory_measurements
        }
        
        print(f"    Total memory increase: {total_increase:.1f}MB")
        print(f"    Memory per client: {results['memory_per_client_mb']:.2f}MB")
        
        return results
    
    async def _run_throughput_benchmark(self, server, clients):
        """Benchmark system throughput."""
        print("  Running throughput benchmark...")
        
        # Test different batch sizes
        batch_sizes = [5, 10, 15, 20]
        throughput_results = {}
        
        for batch_size in batch_sizes:
            print(f"    Testing batch size: {batch_size}")
            
            # Use subset of clients for this batch size
            batch_clients = clients[:batch_size]
            
            # Measure throughput over multiple iterations
            iteration_results = []
            
            for iteration in range(3):
                start_time = time.time()
                
                # Submit batch of updates
                update_tasks = []
                for client_info in batch_clients:
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
                        network_conditions={"latency": 10, "bandwidth": 100},
                        privacy_budget_used=0.1
                    )
                    
                    task = server.receive_model_update(client_info.client_id, update)
                    update_tasks.append(task)
                
                # Wait for completion
                results = await asyncio.gather(*update_tasks, return_exceptions=True)
                await asyncio.sleep(0.5)  # Brief aggregation time
                
                end_time = time.time()
                
                # Calculate metrics
                duration = end_time - start_time
                successful_updates = sum(1 for result in results if result is True)
                throughput = successful_updates / duration
                
                iteration_results.append({
                    "iteration": iteration,
                    "duration": duration,
                    "successful_updates": successful_updates,
                    "throughput": throughput
                })
            
            # Calculate average throughput for this batch size
            avg_throughput = np.mean([r["throughput"] for r in iteration_results])
            max_throughput = max([r["throughput"] for r in iteration_results])
            avg_duration = np.mean([r["duration"] for r in iteration_results])
            
            throughput_results[batch_size] = {
                "avg_throughput": avg_throughput,
                "max_throughput": max_throughput,
                "avg_duration": avg_duration,
                "iterations": iteration_results
            }
            
            print(f"      Average throughput: {avg_throughput:.1f} updates/sec")
        
        # Find optimal batch size
        optimal_batch_size = max(throughput_results.keys(), 
                               key=lambda k: throughput_results[k]["avg_throughput"])
        
        results = {
            "batch_results": throughput_results,
            "optimal_batch_size": optimal_batch_size,
            "max_throughput": throughput_results[optimal_batch_size]["max_throughput"],
            "optimal_avg_throughput": throughput_results[optimal_batch_size]["avg_throughput"]
        }
        
        print(f"    Optimal batch size: {optimal_batch_size}")
        print(f"    Maximum throughput: {results['max_throughput']:.1f} updates/sec")
        
        return results
    
    def _validate_benchmark_results(self, results):
        """Validate benchmark results against performance expectations."""
        print("\nValidating benchmark results...")
        
        # Client registration performance
        if "client_registration" in results:
            reg_results = results["client_registration"]
            
            # Should handle at least 10 registrations per second
            for num_clients, metrics in reg_results.items():
                assert metrics["registrations_per_second"] >= 5.0, \
                    f"Registration rate too low: {metrics['registrations_per_second']:.1f}/sec"
                assert metrics["success_rate"] >= 0.95, \
                    f"Registration success rate too low: {metrics['success_rate']:.2%}"
        
        # Model aggregation performance
        if "model_aggregation" in results:
            agg_results = results["model_aggregation"]
            
            for size_name, metrics in agg_results.items():
                assert metrics["success_rate"] >= 0.9, \
                    f"Aggregation success rate too low for {size_name}: {metrics['success_rate']:.2%}"
                assert metrics["aggregation_successful"], \
                    f"Aggregation failed for {size_name} model"
                
                # Performance should scale reasonably with model size
                if size_name == "small":
                    assert metrics["updates_per_second"] >= 10.0
                elif size_name == "medium":
                    assert metrics["updates_per_second"] >= 5.0
                elif size_name == "large":
                    assert metrics["updates_per_second"] >= 2.0
        
        # Memory usage validation
        if "memory_usage" in results:
            mem_results = results["memory_usage"]
            
            # Memory per client should be reasonable
            assert mem_results["memory_per_client_mb"] <= 2.0, \
                f"Memory usage per client too high: {mem_results['memory_per_client_mb']:.2f}MB"
            
            # Total memory increase should be bounded
            assert mem_results["total_increase_mb"] <= 100.0, \
                f"Total memory increase too high: {mem_results['total_increase_mb']:.1f}MB"
        
        # Throughput validation
        if "throughput" in results:
            throughput_results = results["throughput"]
            
            # Should achieve reasonable maximum throughput
            assert throughput_results["max_throughput"] >= 15.0, \
                f"Maximum throughput too low: {throughput_results['max_throughput']:.1f} updates/sec"
        
        print("✅ All benchmark validations passed!")


@pytest.mark.performance
class TestPerformanceRegression:
    """Test for performance regressions against historical baselines."""
    
    def load_baseline_results(self, baseline_file="baseline_performance.json"):
        """Load baseline performance results."""
        try:
            with open(baseline_file, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return None
    
    def save_baseline_results(self, results, baseline_file="baseline_performance.json"):
        """Save current results as new baseline."""
        with open(baseline_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
    
    async def test_performance_regression(self, test_config):
        """Test for performance regressions."""
        # Load baseline results
        baseline = self.load_baseline_results()
        
        if baseline is None:
            pytest.skip("No baseline performance data available")
        
        # Run current performance tests
        server = AggregationServer(test_config)
        await server.initialize()
        
        try:
            # Quick performance test
            num_clients = 20
            clients = []
            
            for i in range(num_clients):
                client_info = ClientInfo(
                    client_id=f"regression_client_{i}",
                    client_type="Simulated",
                    capabilities={"cpu_cores": 4, "memory_gb": 8},
                    location={"lat": 37.0, "lon": -122.0},
                    network_info={"bandwidth": 100, "latency": 10},
                    hardware_specs={},
                    reputation_score=0.9
                )
                await server.register_client(client_info)
                clients.append(client_info)
            
            # Measure current performance
            start_time = time.time()
            
            update_tasks = []
            for client_info in clients:
                weights = {
                    "layer1": np.random.randn(50, 25).astype(np.float32),
                    "layer2": np.random.randn(25, 12).astype(np.float32)
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
            
            results = await asyncio.gather(*update_tasks)
            await asyncio.sleep(1.0)
            
            end_time = time.time()
            
            # Calculate current metrics
            current_metrics = {
                "total_time": end_time - start_time,
                "updates_per_second": num_clients / (end_time - start_time),
                "success_rate": sum(1 for r in results if r) / num_clients
            }
            
            # Compare with baseline
            baseline_metrics = baseline.get("regression_test", {})
            
            if baseline_metrics:
                # Check for regressions (allow 10% degradation)
                regression_threshold = 0.9
                
                if "updates_per_second" in baseline_metrics:
                    baseline_throughput = baseline_metrics["updates_per_second"]
                    current_throughput = current_metrics["updates_per_second"]
                    
                    assert current_throughput >= baseline_throughput * regression_threshold, \
                        f"Throughput regression detected: {current_throughput:.1f} < {baseline_throughput * regression_threshold:.1f}"
                
                if "success_rate" in baseline_metrics:
                    baseline_success = baseline_metrics["success_rate"]
                    current_success = current_metrics["success_rate"]
                    
                    assert current_success >= baseline_success * regression_threshold, \
                        f"Success rate regression detected: {current_success:.2%} < {baseline_success * regression_threshold:.2%}"
                
                print("✅ No performance regressions detected")
            else:
                print("⚠️  No baseline metrics found, saving current results as baseline")
                self.save_baseline_results({"regression_test": current_metrics})
        
        finally:
            await server.shutdown()