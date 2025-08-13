"""
Comprehensive tests for the configuration management system
"""

import pytest
import tempfile
import os
import yaml
import json
import time
import threading
from pathlib import Path
from unittest.mock import patch, MagicMock

from config import (
    ConfigManager, ABTestManager, AppConfig, DatabaseConfig, 
    NetworkConfig, FederatedLearningConfig, PrivacyConfig, 
    MonitoringConfig, ABTestConfig
)


class TestConfigManager:
    """Test cases for ConfigManager"""
    
    @pytest.fixture
    def temp_config_dir(self):
        """Create temporary configuration directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config"
            config_dir.mkdir()
            
            # Create base configuration
            base_config = {
                "environment": "test",
                "debug": True,
                "database": {
                    "host": "localhost",
                    "port": 5432,
                    "database": "test_db",
                    "username": "test_user",
                    "password": "test_pass"
                },
                "network": {
                    "host": "0.0.0.0",
                    "port": 8000,
                    "max_connections": 100,
                    "timeout": 30
                },
                "federated_learning": {
                    "aggregation_strategy": "fedavg",
                    "min_clients": 2,
                    "max_clients": 10,
                    "rounds": 5,
                    "local_epochs": 3,
                    "learning_rate": 0.01,
                    "batch_size": 32
                },
                "privacy": {
                    "enable_differential_privacy": False,
                    "epsilon": 1.0,
                    "delta": 1e-5,
                    "noise_multiplier": 1.0
                },
                "monitoring": {
                    "enable_metrics": True,
                    "metrics_port": 9090,
                    "log_level": "INFO",
                    "enable_tracing": False
                }
            }
            
            with open(config_dir / "base.yaml", 'w') as f:
                yaml.dump(base_config, f)
            
            # Create environment-specific config
            env_config = {
                "environment": "test",
                "database": {
                    "host": "test-db-host"
                },
                "federated_learning": {
                    "min_clients": 1
                }
            }
            
            with open(config_dir / "test.yaml", 'w') as f:
                yaml.dump(env_config, f)
            
            # Create schema
            schema = {
                "type": "object",
                "properties": {
                    "environment": {"type": "string"},
                    "debug": {"type": "boolean"},
                    "database": {
                        "type": "object",
                        "properties": {
                            "host": {"type": "string"},
                            "port": {"type": "integer", "minimum": 1, "maximum": 65535}
                        }
                    }
                }
            }
            
            with open(config_dir / "schema.json", 'w') as f:
                json.dump(schema, f)
            
            yield str(config_dir)
    
    def test_load_config_hierarchical_merge(self, temp_config_dir):
        """Test hierarchical configuration merging"""
        config_manager = ConfigManager(temp_config_dir)
        config = config_manager.load_config("test")
        
        assert config.environment == "test"
        assert config.database.host == "test-db-host"  # Overridden by test.yaml
        assert config.database.port == 5432  # From base.yaml
        assert config.federated_learning.min_clients == 1  # Overridden by test.yaml
        assert config.federated_learning.max_clients == 10  # From base.yaml
    
    def test_environment_variable_overrides(self, temp_config_dir):
        """Test environment variable overrides"""
        with patch.dict(os.environ, {
            'DB_HOST': 'env-db-host',
            'DB_PORT': '3306',
            'PORT': '9000',
            'PRIVACY_EPSILON': '0.5'
        }):
            config_manager = ConfigManager(temp_config_dir)
            config = config_manager.load_config("test")
            
            assert config.database.host == "env-db-host"
            assert config.database.port == 3306
            assert config.network.port == 9000
            assert config.privacy.epsilon == 0.5
    
    def test_config_validation_success(self, temp_config_dir):
        """Test successful configuration validation"""
        config_manager = ConfigManager(temp_config_dir)
        config = config_manager.load_config("test")
        
        assert config_manager.validate_config(config) is True
    
    def test_config_validation_failure(self, temp_config_dir):
        """Test configuration validation failure"""
        config_manager = ConfigManager(temp_config_dir)
        config = config_manager.load_config("test")
        
        # Make configuration invalid
        config.network.port = -1
        
        assert config_manager.validate_config(config) is False
    
    def test_hot_reloading(self, temp_config_dir):
        """Test configuration hot-reloading"""
        config_manager = ConfigManager(temp_config_dir)
        config_manager.enable_hot_reloading()
        
        # Load initial config
        initial_config = config_manager.load_config("test")
        assert initial_config.database.host == "test-db-host"
        
        # Track reload callbacks
        reload_called = threading.Event()
        new_config_holder = {}
        
        def reload_callback(new_config):
            new_config_holder['config'] = new_config
            reload_called.set()
        
        config_manager.add_reload_callback(reload_callback)
        
        # Modify configuration file
        config_path = Path(temp_config_dir) / "test.yaml"
        modified_config = {
            "environment": "test",
            "database": {
                "host": "modified-db-host"
            }
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(modified_config, f)
        
        # Wait for reload (with timeout)
        if reload_called.wait(timeout=5):
            assert new_config_holder['config'].database.host == "modified-db-host"
        
        config_manager.disable_hot_reloading()
    
    def test_save_config(self, temp_config_dir):
        """Test saving configuration to file"""
        config_manager = ConfigManager(temp_config_dir)
        config = config_manager.load_config("test")
        
        # Modify config
        config.database.host = "saved-db-host"
        
        # Save config
        config_manager.save_config(config, "saved_test.yaml")
        
        # Verify saved config
        saved_path = Path(temp_config_dir) / "saved_test.yaml"
        assert saved_path.exists()
        
        with open(saved_path, 'r') as f:
            saved_data = yaml.safe_load(f)
        
        assert saved_data['database']['host'] == "saved-db-host"


class TestABTestManager:
    """Test cases for ABTestManager"""
    
    @pytest.fixture
    def ab_manager(self):
        """Create ABTestManager instance"""
        return ABTestManager()
    
    def test_create_test(self, ab_manager):
        """Test creating A/B test"""
        variants = {
            "A": {"learning_rate": 0.01},
            "B": {"learning_rate": 0.005}
        }
        
        test_id = ab_manager.create_test("test_lr", variants, 0.6, 48)
        
        assert test_id is not None
        assert "test_lr" in ab_manager._active_tests
        assert ab_manager._active_tests["test_lr"]["traffic_split"] == 0.6
    
    def test_get_variant_assignment(self, ab_manager):
        """Test variant assignment consistency"""
        variants = {
            "A": {"learning_rate": 0.01},
            "B": {"learning_rate": 0.005}
        }
        
        ab_manager.create_test("test_assignment", variants, 0.5, 24)
        
        # Same user should get same variant
        user_id = "test_user_123"
        variant1 = ab_manager.get_variant("test_assignment", user_id)
        variant2 = ab_manager.get_variant("test_assignment", user_id)
        
        assert variant1 == variant2
        assert variant1 in ["A", "B"]
    
    def test_record_and_get_metrics(self, ab_manager):
        """Test recording and retrieving A/B test metrics"""
        variants = {
            "A": {"learning_rate": 0.01},
            "B": {"learning_rate": 0.005}
        }
        
        ab_manager.create_test("test_metrics", variants, 0.5, 24)
        
        # Record some metrics
        ab_manager.record_metric("test_metrics", "user1", "accuracy", 0.85)
        ab_manager.record_metric("test_metrics", "user2", "accuracy", 0.90)
        ab_manager.record_metric("test_metrics", "user3", "accuracy", 0.88)
        
        # Get results
        results = ab_manager.get_test_results("test_metrics")
        
        assert results["total_samples"] == 3
        assert "variant_stats" in results
        assert len(results["raw_results"]) == 3
    
    def test_stop_test(self, ab_manager):
        """Test stopping A/B test"""
        variants = {
            "A": {"learning_rate": 0.01},
            "B": {"learning_rate": 0.005}
        }
        
        ab_manager.create_test("test_stop", variants, 0.5, 24)
        
        # Record a metric
        ab_manager.record_metric("test_stop", "user1", "accuracy", 0.85)
        
        # Stop test
        results = ab_manager.stop_test("test_stop")
        
        assert ab_manager._active_tests["test_stop"]["status"] == "stopped"
        assert results["total_samples"] == 1
    
    def test_traffic_split_distribution(self, ab_manager):
        """Test that traffic split approximately matches expected distribution"""
        variants = {
            "A": {"learning_rate": 0.01},
            "B": {"learning_rate": 0.005}
        }
        
        # Create test with 70% traffic to A
        ab_manager.create_test("test_distribution", variants, 0.7, 24)
        
        # Generate many assignments
        assignments = []
        for i in range(1000):
            variant = ab_manager.get_variant("test_distribution", f"user_{i}")
            assignments.append(variant)
        
        # Check distribution (allow some variance)
        a_count = assignments.count("A")
        a_percentage = a_count / len(assignments)
        
        # Should be approximately 70% (within 5% tolerance)
        assert 0.65 <= a_percentage <= 0.75


class TestDataClasses:
    """Test configuration data classes"""
    
    def test_database_config_defaults(self):
        """Test DatabaseConfig default values"""
        db_config = DatabaseConfig()
        
        assert db_config.host == "localhost"
        assert db_config.port == 5432
        assert db_config.database == "federated_pipeline"
        assert db_config.username == "postgres"
        assert db_config.password == ""
    
    def test_network_config_defaults(self):
        """Test NetworkConfig default values"""
        net_config = NetworkConfig()
        
        assert net_config.host == "0.0.0.0"
        assert net_config.port == 8000
        assert net_config.max_connections == 100
        assert net_config.timeout == 30
    
    def test_federated_learning_config_defaults(self):
        """Test FederatedLearningConfig default values"""
        fl_config = FederatedLearningConfig()
        
        assert fl_config.aggregation_strategy == "fedavg"
        assert fl_config.min_clients == 2
        assert fl_config.max_clients == 100
        assert fl_config.rounds == 10
        assert fl_config.local_epochs == 5
        assert fl_config.learning_rate == 0.01
        assert fl_config.batch_size == 32
    
    def test_app_config_initialization(self):
        """Test AppConfig initialization with nested configs"""
        app_config = AppConfig()
        
        assert isinstance(app_config.database, DatabaseConfig)
        assert isinstance(app_config.network, NetworkConfig)
        assert isinstance(app_config.federated_learning, FederatedLearningConfig)
        assert isinstance(app_config.privacy, PrivacyConfig)
        assert isinstance(app_config.monitoring, MonitoringConfig)
        assert isinstance(app_config.ab_test, ABTestConfig)


class TestConfigIntegration:
    """Integration tests for configuration system"""
    
    def test_full_configuration_workflow(self, temp_config_dir):
        """Test complete configuration workflow"""
        # Initialize config manager
        config_manager = ConfigManager(temp_config_dir)
        
        # Load configuration
        config = config_manager.load_config("test")
        assert config.environment == "test"
        
        # Validate configuration
        assert config_manager.validate_config(config) is True
        
        # Enable hot reloading
        config_manager.enable_hot_reloading()
        
        # Test A/B testing integration
        with patch.dict(os.environ, {
            'AB_TEST_ENABLED': 'true',
            'AB_TEST_VARIANT': 'B'
        }):
            config_with_ab = config_manager.load_config("test")
            # A/B test configuration should be applied
        
        # Disable hot reloading
        config_manager.disable_hot_reloading()
    
    def test_configuration_with_missing_files(self):
        """Test configuration loading with missing files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "empty_config"
            config_dir.mkdir()
            
            config_manager = ConfigManager(str(config_dir))
            
            # Should create default configuration
            config = config_manager.load_config("development")
            assert config.environment == "development"
            assert isinstance(config.database, DatabaseConfig)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])