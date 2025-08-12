"""
Configuration management system with environment-specific settings
"""
import os
import yaml
import json
from typing import Dict, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict


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
class AppConfig:
    environment: str = "development"
    debug: bool = True
    database: DatabaseConfig = None
    network: NetworkConfig = None
    federated_learning: FederatedLearningConfig = None
    privacy: PrivacyConfig = None
    monitoring: MonitoringConfig = None
    
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


class ConfigManager:
    """Configuration manager with environment-specific settings"""
    
    def __init__(self, config_dir: str = "config"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self._config: Optional[AppConfig] = None
        self._watchers = []
    
    def load_config(self, environment: str = None) -> AppConfig:
        """Load configuration for specified environment"""
        if environment is None:
            environment = os.getenv("ENVIRONMENT", "development")
        
        # Load base configuration
        base_config = self._load_config_file("base.yaml")
        
        # Load environment-specific configuration
        env_config = self._load_config_file(f"{environment}.yaml")
        
        # Merge configurations
        merged_config = self._merge_configs(base_config, env_config)
        
        # Override with environment variables
        merged_config = self._apply_env_overrides(merged_config)
        
        # Create AppConfig instance
        self._config = self._dict_to_config(merged_config)
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
        """Validate configuration"""
        try:
            # Basic validation
            assert config.network.port > 0
            assert config.federated_learning.min_clients > 0
            assert config.federated_learning.learning_rate > 0
            assert 0 < config.privacy.epsilon <= 10
            return True
        except (AssertionError, AttributeError):
            return False
    
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