"""
Comprehensive unit tests for the aggregation server component.
"""
import pytest
import asyncio
import pickle
import numpy as np
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from src.aggregation_server.server import AggregationServer
from src.aggregation_server.aggregation import FederatedAggregator
from src.aggregation_server.auth import AuthenticationManager
from src.aggregation_server.storage import ModelStorage
from src.common.interfaces import ClientInfo, ModelUpdate


@pytest.mark.unit
class TestAggregationServer:
    """Test cases for AggregationServer class."""
    
    async def test_server_initialization(self, test_config):
        """Test server initialization and configuration."""
        server = AggregationServer(test_config)
        
        # Test initialization
        await server.initialize()
        assert server.is_running
        assert server.config == test_config
        
        # Cleanup
        await server.shutdown()
        assert not server.is_running
    
    async def test_client_registration(self, aggregation_server, sample_client_info):
        """Test client registration process."""
        # Test successful registration
        token = await aggregation_server.register_client(sample_client_info)
        assert token is not None
        assert len(token) > 0
        
        # Test duplicate registration
        token2 = await aggregation_server.register_client(sample_client_info)
        assert token2 == token  # Should return same token
        
        # Test client info retrieval
        client_info = await aggregation_server.get_client_info(sample_client_info.client_id)
        assert client_info.client_id == sample_client_info.client_id
        assert client_info.client_type == sample_client_info.client_type
    
    async def test_model_update_processing(self, aggregation_server, sample_client_info, sample_model_update):
        """Test model update processing."""
        # Register client first
        await aggregation_server.register_client(sample_client_info)
        
        # Test model update submission
        success = await aggregation_server.receive_model_update(
            sample_client_info.client_id, 
            sample_model_update
        )
        assert success
        
        # Test invalid client
        invalid_update = sample_model_update
        invalid_update.client_id = "invalid_client"
        success = await aggregation_server.receive_model_update(
            "invalid_client", 
            invalid_update
        )
        assert not success
    
    async def test_global_model_distribution(self, aggregation_server, sample_client_info, sample_model_update):
        """Test global model distribution."""
        # Register client and submit update
        await aggregation_server.register_client(sample_client_info)
        await aggregation_server.receive_model_update(
            sample_client_info.client_id, 
            sample_model_update
        )
        
        # Wait for aggregation
        await asyncio.sleep(0.1)
        
        # Test global model retrieval
        global_model = await aggregation_server.get_global_model(sample_client_info.client_id)
        assert global_model is not None
        assert "version" in global_model
        assert "weights" in global_model
    
    async def test_server_status(self, aggregation_server, sample_client_info):
        """Test server status reporting."""
        # Test initial status
        status = await aggregation_server.get_server_status()
        assert status["status"] == "running"
        assert status["total_clients"] == 0
        
        # Register client and check status
        await aggregation_server.register_client(sample_client_info)
        status = await aggregation_server.get_server_status()
        assert status["total_clients"] == 1
        assert status["active_clients"] >= 0
    
    async def test_health_check(self, aggregation_server):
        """Test health check functionality."""
        health = await aggregation_server.get_health_status()
        assert health["status"] == "healthy"
        assert "uptime" in health
        assert "memory_usage" in health
        assert "cpu_usage" in health
    
    async def test_training_configuration(self, aggregation_server, sample_client_info):
        """Test training configuration management."""
        await aggregation_server.register_client(sample_client_info)
        
        config = await aggregation_server.get_training_configuration(sample_client_info.client_id)
        assert "aggregation_strategy" in config
        assert "learning_rate" in config
        assert "batch_size" in config
        assert "epochs" in config
    
    async def test_client_metrics_reporting(self, aggregation_server, sample_client_info):
        """Test client metrics reporting."""
        await aggregation_server.register_client(sample_client_info)
        
        metrics = {
            "cpu_usage": 75.5,
            "memory_usage": 60.2,
            "network_latency": 15.0,
            "battery_level": 85.0
        }
        
        await aggregation_server.report_client_metrics(sample_client_info.client_id, metrics)
        
        # Verify metrics were stored
        client_metrics = await aggregation_server.get_client_metrics(sample_client_info.client_id)
        assert client_metrics is not None
        assert client_metrics["cpu_usage"] == 75.5


@pytest.mark.unit
class TestFederatedAggregator:
    """Test cases for FederatedAggregator class."""
    
    def test_fedavg_aggregation(self):
        """Test FedAvg aggregation algorithm."""
        aggregator = FederatedAggregator("fedavg")
        
        # Create sample model updates
        updates = []
        for i in range(3):
            weights = {
                "layer1": np.random.randn(5, 3).astype(np.float32),
                "layer2": np.random.randn(3, 1).astype(np.float32)
            }
            updates.append({
                "weights": weights,
                "num_samples": 100 + i * 50,
                "client_id": f"client_{i}"
            })
        
        # Test aggregation
        aggregated = aggregator.aggregate(updates)
        assert "layer1" in aggregated
        assert "layer2" in aggregated
        assert aggregated["layer1"].shape == (5, 3)
        assert aggregated["layer2"].shape == (3, 1)
    
    def test_krum_aggregation(self):
        """Test Krum aggregation algorithm."""
        aggregator = FederatedAggregator("krum")
        
        # Create sample updates with one Byzantine
        updates = []
        for i in range(5):
            weights = {
                "layer1": np.random.randn(3, 2).astype(np.float32) * (10 if i == 4 else 1),
                "layer2": np.random.randn(2, 1).astype(np.float32) * (10 if i == 4 else 1)
            }
            updates.append({
                "weights": weights,
                "num_samples": 100,
                "client_id": f"client_{i}"
            })
        
        # Test aggregation (should exclude Byzantine client)
        aggregated = aggregator.aggregate(updates)
        assert aggregated is not None
        assert "layer1" in aggregated
        assert "layer2" in aggregated
    
    def test_trimmed_mean_aggregation(self):
        """Test Trimmed Mean aggregation algorithm."""
        aggregator = FederatedAggregator("trimmed_mean")
        
        # Create sample updates
        updates = []
        for i in range(7):
            weights = {
                "layer1": np.ones((2, 2), dtype=np.float32) * (i + 1),
                "layer2": np.ones((2, 1), dtype=np.float32) * (i + 1)
            }
            updates.append({
                "weights": weights,
                "num_samples": 100,
                "client_id": f"client_{i}"
            })
        
        # Test aggregation
        aggregated = aggregator.aggregate(updates)
        assert aggregated is not None
        # Should trim extreme values and average the rest
        assert np.allclose(aggregated["layer1"], np.ones((2, 2)) * 4.0, atol=1.0)
    
    def test_weighted_aggregation(self):
        """Test weighted aggregation algorithm."""
        aggregator = FederatedAggregator("weighted")
        
        # Create updates with different sample sizes
        updates = []
        sample_sizes = [100, 200, 300]
        for i, size in enumerate(sample_sizes):
            weights = {
                "layer1": np.ones((2, 2), dtype=np.float32) * (i + 1),
                "layer2": np.ones((2, 1), dtype=np.float32) * (i + 1)
            }
            updates.append({
                "weights": weights,
                "num_samples": size,
                "client_id": f"client_{i}",
                "data_quality": 0.8 + i * 0.1
            })
        
        # Test aggregation
        aggregated = aggregator.aggregate(updates)
        assert aggregated is not None
        # Larger datasets should have more influence
        assert np.mean(aggregated["layer1"]) > 1.5


@pytest.mark.unit
class TestAuthenticationManager:
    """Test cases for AuthenticationManager class."""
    
    def test_token_generation(self):
        """Test authentication token generation."""
        auth_manager = AuthenticationManager()
        
        client_info = ClientInfo(
            client_id="test_client",
            client_type="Simulated",
            capabilities={},
            location={},
            network_info={},
            hardware_specs={},
            reputation_score=1.0
        )
        
        token = auth_manager.generate_token(client_info)
        assert token is not None
        assert len(token) > 0
        assert isinstance(token, str)
    
    def test_token_validation(self):
        """Test authentication token validation."""
        auth_manager = AuthenticationManager()
        
        client_info = ClientInfo(
            client_id="test_client",
            client_type="Simulated",
            capabilities={},
            location={},
            network_info={},
            hardware_specs={},
            reputation_score=1.0
        )
        
        # Generate and validate token
        token = auth_manager.generate_token(client_info)
        is_valid = auth_manager.validate_token(token, client_info.client_id)
        assert is_valid
        
        # Test invalid token
        is_valid = auth_manager.validate_token("invalid_token", client_info.client_id)
        assert not is_valid
    
    def test_token_expiration(self):
        """Test token expiration handling."""
        auth_manager = AuthenticationManager(token_expiry_hours=0.001)  # Very short expiry
        
        client_info = ClientInfo(
            client_id="test_client",
            client_type="Simulated",
            capabilities={},
            location={},
            network_info={},
            hardware_specs={},
            reputation_score=1.0
        )
        
        token = auth_manager.generate_token(client_info)
        
        # Token should be valid initially
        assert auth_manager.validate_token(token, client_info.client_id)
        
        # Wait for expiration (in real implementation, would use time.sleep)
        # For testing, we'll manually expire the token
        auth_manager._tokens[token]["expires_at"] = datetime.now() - timedelta(hours=1)
        
        # Token should now be invalid
        assert not auth_manager.validate_token(token, client_info.client_id)


@pytest.mark.unit
class TestModelStorage:
    """Test cases for ModelStorage class."""
    
    def test_model_storage_and_retrieval(self, temp_dir):
        """Test model storage and retrieval."""
        storage = ModelStorage(temp_dir)
        
        # Create sample model
        model_data = {
            "weights": {
                "layer1": np.random.randn(5, 3).astype(np.float32),
                "layer2": np.random.randn(3, 1).astype(np.float32)
            },
            "metadata": {
                "version": 1,
                "timestamp": datetime.now().isoformat(),
                "accuracy": 0.85
            }
        }
        
        # Store model
        model_id = storage.store_model(model_data)
        assert model_id is not None
        
        # Retrieve model
        retrieved_model = storage.get_model(model_id)
        assert retrieved_model is not None
        assert "weights" in retrieved_model
        assert "metadata" in retrieved_model
        assert retrieved_model["metadata"]["version"] == 1
    
    def test_model_versioning(self, temp_dir):
        """Test model versioning functionality."""
        storage = ModelStorage(temp_dir)
        
        # Store multiple versions
        versions = []
        for i in range(3):
            model_data = {
                "weights": {"layer1": np.random.randn(2, 2).astype(np.float32)},
                "metadata": {"version": i + 1, "accuracy": 0.7 + i * 0.05}
            }
            model_id = storage.store_model(model_data)
            versions.append(model_id)
        
        # Test version retrieval
        latest_model = storage.get_latest_model()
        assert latest_model is not None
        assert latest_model["metadata"]["version"] == 3
        
        # Test specific version retrieval
        first_model = storage.get_model(versions[0])
        assert first_model["metadata"]["version"] == 1
    
    def test_model_cleanup(self, temp_dir):
        """Test model cleanup functionality."""
        storage = ModelStorage(temp_dir, max_versions=2)
        
        # Store more models than max_versions
        for i in range(5):
            model_data = {
                "weights": {"layer1": np.random.randn(2, 2).astype(np.float32)},
                "metadata": {"version": i + 1}
            }
            storage.store_model(model_data)
        
        # Should only keep the latest 2 versions
        all_models = storage.list_models()
        assert len(all_models) <= 2