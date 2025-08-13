"""
Advanced configuration management system with hierarchical configuration,
validation, hot-reloading, and A/B testing capabilities
"""
import os
import yaml
import json
import threading
import time
import hashlib
from typing import Dict, Any, Optional, List, Callable, Union
from pathlib import Path
from dataclasses import dataclass, asdict, field
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from jsonschema import validate, ValidationError
import logging

logger = logging.getLogger(__name__)


@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 5432
    database: str = "federated_pipeline"
    username: str = "postgres"
    password: str = ""


@dataclass
class NetworkConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    max_connections: int = 100
    timeout: int = 30


@dataclass
class FederatedLearningConfig:
    aggregation_strategy: str = "fedavg"
    min_clients: int = 2
    max_clients: int = 100
    rounds: int = 10
    local_epochs: int = 5
    learning_rate: float = 0.01
    batch_size: int = 32


@dataclass
class PrivacyConfig:
    enable_differential_privacy: bool = False
    epsilon: float = 1.0
    delta: float = 1e-5
    noise_multiplier: float = 1.0


@dataclass
class MonitoringConfig:
    enable_metrics: bool = True
    metrics_port: int = 9090
    log_level: str = "INFO"
    enable_tracing: bool = False


@dataclass
class ABTestConfig:
    """A/B testing configuration for hyperparameter optimization"""
    enabled: bool = False
    test_name: str = ""
    variant: str = "A"  # A or B
    traffic_split: float = 0.5  # Percentage for variant A
    parameters: Dict[str, Any] = field(default_factory=dict)
    start_time: Optional[str] = None
    end_time: Optional[str] = None


@dataclass
class AppConfig:
    environment: str = "development"
    debug: bool = True
    database: DatabaseConfig = None
    network: NetworkConfig = None
    federated_learning: FederatedLearningConfig = None
    privacy: PrivacyConfig = None
    monitoring: MonitoringConfig = None
    ab_test: ABTestConfig = None
    config_version: str = "1.0.0"
    last_updated: Optional[str] = None
    
    def __post_init__(self):
        if self.database is None:
            self.database = DatabaseConfig()
        if self.network is None:
            self.network = NetworkConfig()
        if self.federated_learning is None:
            self.federated_learning = FederatedLearningConfig()
        if self.privacy is None:
            self.privacy = PrivacyConfig()
        if self.monitoring is None:
            self.monitoring = MonitoringConfig()
        if self.ab_test is None:
            self.ab_test = ABTestConfig()


class ConfigFileHandler(FileSystemEventHandler):
    """File system event handler for configuration hot-reloading"""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.last_modified = {}
    
    def on_modified(self, event):
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        if file_path.suffix in ['.yaml', '.yml', '.json']:
            # Debounce rapid file changes
            current_time = time.time()
            if file_path in self.last_modified:
                if current_time - self.last_modified[file_path] < 1.0:
                    return
            
            self.last_modified[file_path] = current_time
            logger.info(f"Configuration file changed: {file_path}")
            self.config_manager._reload_from_file_change(str(file_path))


class ConfigManager:
    """Advanced configuration manager with hierarchical configuration,
    validation, hot-reloading, and A/B testing capabilities"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self._config: Optional[AppConfig] = None
        self._config_lock = threading.RLock()
        self._reload_callbacks: List[Callable[[AppConfig], None]] = []
        self._observer: Optional[Observer] = None
        self._file_hashes: Dict[str, str] = {}
        self._ab_test_manager = ABTestManager()
        self._schema = self._load_config_schema()
    
    def load_config(self, environment: str = None) -> AppConfig:
        """Load configuration for specified environment with hierarchical merging"""
        with self._config_lock:
            if environment is None:
                environment = os.getenv("ENVIRONMENT", "development")
            
            # Load configurations in hierarchical order
            configs = []
            
            # 1. Load base configuration
            base_config = self._load_config_file("base.yaml")
            if base_config:
                configs.append(base_config)
            
            # 2. Load environment-specific configuration
            env_config = self._load_config_file(f"{environment}.yaml")
            if env_config:
                configs.append(env_config)
            
            # 3. Load local overrides (if exists)
            local_config = self._load_config_file("local.yaml")
            if local_config:
                configs.append(local_config)
            
            # 4. Load user-specific overrides (if exists)
            user_config = self._load_config_file(f"user-{os.getenv('USER', 'default')}.yaml")
            if user_config:
                configs.append(user_config)
            
            # Merge all configurations hierarchically
            merged_config = {}
            for config in configs:
                merged_config = self._merge_configs(merged_config, config)
            
            # Override with environment variables
            merged_config = self._apply_env_overrides(merged_config)
            
            # Apply A/B testing parameters
            merged_config = self._apply_ab_testing(merged_config)
            
            # Create AppConfig instance
            config_obj = self._dict_to_config(merged_config)
            
            # Validate configuration
            if not self.validate_config(config_obj):
                raise ValueError("Configuration validation failed")
            
            # Update file hashes for change detection
            self._update_file_hashes()
            
            self._config = config_obj
            return self._config
    
    def get_config(self) -> AppConfig:
        """Get current configuration"""
        if self._config is None:
            return self.load_config()
        return self._config
    
    def reload_config(self) -> AppConfig:
        """Reload configuration from files"""
        self._config = None
        return self.load_config()
    
    def save_config(self, config: AppConfig, filename: str = None):
        """Save configuration to file"""
        if filename is None:
            filename = f"{config.environment}.yaml"
        
        config_path = self.config_dir / filename
        config_dict = asdict(config)
        
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    def validate_config(self, config: AppConfig) -> bool:
        """Comprehensive configuration validation using JSON schema"""
        try:
            # Convert config to dict for schema validation
            config_dict = asdict(config)
            
            # Validate against schema
            if self._schema:
                validate(instance=config_dict, schema=self._schema)
            
            # Additional business logic validation
            assert config.network.port > 0 and config.network.port < 65536
            assert config.federated_learning.min_clients > 0
            assert config.federated_learning.max_clients >= config.federated_learning.min_clients
            assert config.federated_learning.learning_rate > 0
            assert 0 < config.privacy.epsilon <= 10
            assert config.privacy.delta > 0
            assert config.federated_learning.rounds > 0
            assert config.federated_learning.local_epochs > 0
            
            # A/B testing validation
            if config.ab_test.enabled:
                assert config.ab_test.variant in ["A", "B"]
                assert 0 <= config.ab_test.traffic_split <= 1
                assert config.ab_test.test_name
            
            return True
        except (AssertionError, AttributeError, ValidationError) as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def enable_hot_reloading(self):
        """Enable hot-reloading of configuration files"""
        if self._observer is not None:
            return  # Already enabled
        
        self._observer = Observer()
        event_handler = ConfigFileHandler(self)
        self._observer.schedule(event_handler, str(self.config_dir), recursive=False)
        self._observer.start()
        logger.info("Configuration hot-reloading enabled")
    
    def disable_hot_reloading(self):
        """Disable hot-reloading of configuration files"""
        if self._observer is not None:
            self._observer.stop()
            self._observer.join()
            self._observer = None
            logger.info("Configuration hot-reloading disabled")
    
    def add_reload_callback(self, callback: Callable[[AppConfig], None]):
        """Add callback to be called when configuration is reloaded"""
        self._reload_callbacks.append(callback)
    
    def remove_reload_callback(self, callback: Callable[[AppConfig], None]):
        """Remove reload callback"""
        if callback in self._reload_callbacks:
            self._reload_callbacks.remove(callback)
    
    def _reload_from_file_change(self, file_path: str):
        """Handle configuration reload from file system change"""
        try:
            old_config = self._config
            new_config = self.load_config()
            
            # Notify callbacks of configuration change
            for callback in self._reload_callbacks:
                try:
                    callback(new_config)
                except Exception as e:
                    logger.error(f"Error in reload callback: {e}")
            
            logger.info("Configuration reloaded successfully")
        except Exception as e:
            logger.error(f"Failed to reload configuration: {e}")
    
    def _update_file_hashes(self):
        """Update file hashes for change detection"""
        for config_file in self.config_dir.glob("*.yaml"):
            if config_file.is_file():
                with open(config_file, 'rb') as f:
                    content = f.read()
                    self._file_hashes[str(config_file)] = hashlib.md5(content).hexdigest()
    
    def _load_config_schema(self) -> Optional[Dict[str, Any]]:
        """Load JSON schema for configuration validation"""
        schema_path = self.config_dir / "schema.json"
        if schema_path.exists():
            try:
                with open(schema_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load config schema: {e}")
        return None
    
    def _load_config_file(self, filename: str) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        config_path = self.config_dir / filename
        
        if not config_path.exists():
            return {}
        
        with open(config_path, 'r') as f:
            if filename.endswith('.yaml') or filename.endswith('.yml'):
                return yaml.safe_load(f) or {}
            elif filename.endswith('.json'):
                return json.load(f)
        
        return {}
    
    def _merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge configuration dictionaries"""
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_env_overrides(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply environment variable overrides"""
        # Database overrides
        if os.getenv("DB_HOST"):
            config.setdefault("database", {})["host"] = os.getenv("DB_HOST")
        if os.getenv("DB_PORT"):
            config.setdefault("database", {})["port"] = int(os.getenv("DB_PORT"))
        if os.getenv("DB_PASSWORD"):
            config.setdefault("database", {})["password"] = os.getenv("DB_PASSWORD")
        
        # Network overrides
        if os.getenv("PORT"):
            config.setdefault("network", {})["port"] = int(os.getenv("PORT"))
        if os.getenv("HOST"):
            config.setdefault("network", {})["host"] = os.getenv("HOST")
        
        # Privacy overrides
        if os.getenv("PRIVACY_EPSILON"):
            config.setdefault("privacy", {})["epsilon"] = float(os.getenv("PRIVACY_EPSILON"))
        
        # A/B testing overrides
        if os.getenv("AB_TEST_ENABLED"):
            config.setdefault("ab_test", {})["enabled"] = os.getenv("AB_TEST_ENABLED").lower() == "true"
        if os.getenv("AB_TEST_VARIANT"):
            config.setdefault("ab_test", {})["variant"] = os.getenv("AB_TEST_VARIANT")
        
        return config
    
    def _apply_ab_testing(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply A/B testing parameters"""
        ab_config = config.get("ab_test", {})
        
        if ab_config.get("enabled", False):
            variant = self._ab_test_manager.get_variant(
                test_name=ab_config.get("test_name", "default"),
                traffic_split=ab_config.get("traffic_split", 0.5)
            )
            
            ab_config["variant"] = variant
            
            # Apply variant-specific parameters
            variant_params = ab_config.get("parameters", {}).get(variant, {})
            if variant_params:
                config = self._merge_configs(config, variant_params)
                logger.info(f"Applied A/B test variant {variant} parameters: {variant_params}")
        
        return config
    
    def _dict_to_config(self, config_dict: Dict[str, Any]) -> AppConfig:
        """Convert dictionary to AppConfig instance"""
        # Extract nested configurations
        db_config = DatabaseConfig(**config_dict.get("database", {}))
        network_config = NetworkConfig(**config_dict.get("network", {}))
        fl_config = FederatedLearningConfig(**config_dict.get("federated_learning", {}))
        privacy_config = PrivacyConfig(**config_dict.get("privacy", {}))
        monitoring_config = MonitoringConfig(**config_dict.get("monitoring", {}))
        
        return AppConfig(
            environment=config_dict.get("environment", "development"),
            debug=config_dict.get("debug", True),
            database=db_config,
            network=network_config,
            federated_learning=fl_config,
            privacy=privacy_config,
            monitoring=monitoring_config
        )


# Global configuration manager instance
config_manager = ConfigManager()


def get_config() -> AppConfig:
    """Get application configuration"""
    return config_manager.get_config()


def reload_config() -> AppConfig:
    """Reload application configuration"""
    return config_manager.reload_config()