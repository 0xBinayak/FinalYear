"""
Cross-platform mobile client with real data support for federated learning
"""
import asyncio
import json
import logging
import platform
import psutil
import time
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import requests
from dataclasses import dataclass, asdict

from ..common.interfaces import BaseClient, ClientInfo, ModelUpdate, TrainingConfig
from ..common.signal_models import EnhancedSignalSample, ModulationType, HardwareType, HardwareInfo
from ..common.federated_data_structures import (
    EnhancedModelUpdate, NetworkConditions, ComputeResources, 
    PrivacyBudget, CompressionType, PrivacyMechanism
)
from ..common.data_loaders import RadioMLLoader, HDF5Loader, BinaryIQLoader
from ..aggregation_server.auth import generate_client_token, verify_client_token
from .mobile_optimizations import (
    BackgroundTrainingManager, BackgroundTrainingConfig,
    NetworkHandoffManager, AdaptiveModelComplexityManager,
    IncentiveReputationManager
)


@dataclass
class MobileDeviceCapabilities:
    """Mobile device capabilities and constraints"""
    platform: str  # android, ios, windows, linux, macos
    cpu_cores: int
    cpu_frequency_ghz: float
    total_memory_gb: float
    available_memory_gb: float
    battery_level: Optional[float] = None  # 0.0 to 1.0
    is_charging: bool = False
    thermal_state: str = "normal"  # normal, warm, hot, critical
    network_type: str = "wifi"  # wifi, cellular, ethernet
    is_metered_connection: bool = False
    gpu_available: bool = False
    gpu_memory_gb: float = 0.0
    storage_available_gb: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MobileTrainingConfig:
    """Mobile-specific training configuration"""
    max_training_time_minutes: int = 30
    max_battery_usage_percent: float = 10.0
    min_battery_level: float = 0.3
    background_training_enabled: bool = True
    adaptive_batch_size: bool = True
    thermal_throttling_enabled: bool = True
    network_aware_scheduling: bool = True
    cache_datasets_locally: bool = True
    max_cache_size_gb: float = 2.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BatteryManager:
    """Battery and power management for mobile devices"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.battery_history = []
        self.power_monitoring_active = False
        self._monitor_thread = None
    
    def get_battery_info(self) -> Dict[str, Any]:
        """Get current battery information"""
        try:
            battery = psutil.sensors_battery()
            if battery:
                return {
                    'level': battery.percent / 100.0,
                    'is_charging': battery.power_plugged,
                    'time_left_hours': battery.secsleft / 3600 if battery.secsleft != psutil.POWER_TIME_UNLIMITED else None
                }
            else:
                # Fallback for systems without battery
                return {
                    'level': 1.0,
                    'is_charging': True,
                    'time_left_hours': None
                }
        except Exception as e:
            self.logger.warning(f"Could not get battery info: {e}")
            return {
                'level': 1.0,
                'is_charging': True,
                'time_left_hours': None
            }
    
    def can_start_training(self, config: MobileTrainingConfig) -> Tuple[bool, str]:
        """Check if training can start based on battery constraints"""
        battery_info = self.get_battery_info()
        
        # Check minimum battery level
        if battery_info['level'] < config.min_battery_level:
            return False, f"Battery level too low: {battery_info['level']:.1%}"
        
        # If not charging, check if we have enough battery for training
        if not battery_info['is_charging']:
            estimated_usage = config.max_battery_usage_percent / 100.0
            if battery_info['level'] - estimated_usage < 0.1:  # Keep 10% reserve
                return False, "Insufficient battery for training session"
        
        return True, "Battery level sufficient"
    
    def start_monitoring(self, callback=None):
        """Start battery monitoring"""
        if self.power_monitoring_active:
            return
        
        self.power_monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._monitor_battery, args=(callback,))
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop battery monitoring"""
        self.power_monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)
    
    def _monitor_battery(self, callback=None):
        """Monitor battery levels"""
        while self.power_monitoring_active:
            try:
                battery_info = self.get_battery_info()
                self.battery_history.append({
                    'timestamp': datetime.now(),
                    'level': battery_info['level'],
                    'is_charging': battery_info['is_charging']
                })
                
                # Keep only last hour of history
                cutoff_time = datetime.now() - timedelta(hours=1)
                self.battery_history = [
                    entry for entry in self.battery_history 
                    if entry['timestamp'] > cutoff_time
                ]
                
                if callback:
                    callback(battery_info)
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Battery monitoring error: {e}")
                time.sleep(60)  # Wait longer on error


class NetworkManager:
    """Network condition monitoring and adaptation"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.network_history = []
    
    def get_network_conditions(self) -> NetworkConditions:
        """Get current network conditions"""
        try:
            # Get network interface statistics
            net_io = psutil.net_io_counters()
            
            # Simple bandwidth estimation (this is very basic)
            # In a real implementation, you'd use more sophisticated methods
            bandwidth_mbps = 10.0  # Default assumption
            
            # Estimate latency (placeholder - would need actual ping tests)
            latency_ms = 50.0
            
            # Check if on mobile network
            is_mobile = self._is_mobile_network()
            is_metered = self._is_metered_connection()
            
            conditions = NetworkConditions(
                bandwidth_mbps=bandwidth_mbps,
                latency_ms=latency_ms,
                packet_loss_rate=0.01,  # 1% default
                jitter_ms=10.0,
                connection_stability=0.9,
                is_mobile=is_mobile,
                is_metered=is_metered
            )
            
            # Store in history
            self.network_history.append({
                'timestamp': datetime.now(),
                'conditions': conditions
            })
            
            # Keep only last hour
            cutoff_time = datetime.now() - timedelta(hours=1)
            self.network_history = [
                entry for entry in self.network_history 
                if entry['timestamp'] > cutoff_time
            ]
            
            return conditions
            
        except Exception as e:
            self.logger.error(f"Network monitoring error: {e}")
            return NetworkConditions(10.0, 100.0, 0.05, 20.0, 0.8, True, True)
    
    def _is_mobile_network(self) -> bool:
        """Check if currently on mobile network"""
        # Platform-specific detection would go here
        # For now, return False (assume WiFi)
        return False
    
    def _is_metered_connection(self) -> bool:
        """Check if connection is metered"""
        # Platform-specific detection would go here
        return False
    
    def estimate_transfer_time(self, data_size_bytes: int) -> float:
        """Estimate time to transfer data"""
        conditions = self.get_network_conditions()
        return conditions.get_transmission_time_estimate(data_size_bytes)
    
    def should_defer_training(self) -> Tuple[bool, str]:
        """Check if training should be deferred due to network conditions"""
        conditions = self.get_network_conditions()
        
        # Defer if on metered connection with low bandwidth
        if conditions.is_metered and conditions.bandwidth_mbps < 5.0:
            return True, "On metered connection with low bandwidth"
        
        # Defer if connection is unstable
        if conditions.connection_stability < 0.5:
            return True, "Network connection unstable"
        
        # Defer if latency is too high
        if conditions.latency_ms > 1000:
            return True, "Network latency too high"
        
        return False, "Network conditions acceptable"


class DatasetCache:
    """Local dataset caching for mobile devices"""
    
    def __init__(self, cache_dir: Path, max_size_gb: float = 2.0):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.max_size_gb = max_size_gb
        self.logger = logging.getLogger(__name__)
        
        # Cache metadata
        self.metadata_file = self.cache_dir / "cache_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict[str, Any]:
        """Load cache metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Error loading cache metadata: {e}")
        
        return {
            'datasets': {},
            'total_size_bytes': 0,
            'last_cleanup': datetime.now().isoformat()
        }
    
    def _save_metadata(self):
        """Save cache metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving cache metadata: {e}")
    
    def cache_dataset(self, dataset_name: str, dataset_url: str, 
                     samples: List[EnhancedSignalSample]) -> bool:
        """Cache dataset locally"""
        try:
            dataset_dir = self.cache_dir / dataset_name
            dataset_dir.mkdir(exist_ok=True)
            
            # Save samples
            total_size = 0
            for i, sample in enumerate(samples):
                sample_file = dataset_dir / f"sample_{i:06d}.pkl"
                sample_data = sample.serialize(format='pickle', compress=True)
                
                with open(sample_file, 'wb') as f:
                    f.write(sample_data)
                
                total_size += len(sample_data)
            
            # Update metadata
            self.metadata['datasets'][dataset_name] = {
                'url': dataset_url,
                'sample_count': len(samples),
                'size_bytes': total_size,
                'cached_at': datetime.now().isoformat(),
                'last_accessed': datetime.now().isoformat()
            }
            
            self.metadata['total_size_bytes'] += total_size
            self._save_metadata()
            
            # Cleanup if needed
            self._cleanup_cache()
            
            self.logger.info(f"Cached dataset {dataset_name} with {len(samples)} samples")
            return True
            
        except Exception as e:
            self.logger.error(f"Error caching dataset {dataset_name}: {e}")
            return False
    
    def load_cached_dataset(self, dataset_name: str) -> Optional[List[EnhancedSignalSample]]:
        """Load cached dataset"""
        if dataset_name not in self.metadata['datasets']:
            return None
        
        try:
            dataset_dir = self.cache_dir / dataset_name
            if not dataset_dir.exists():
                return None
            
            samples = []
            sample_files = sorted(dataset_dir.glob("sample_*.pkl"))
            
            for sample_file in sample_files:
                with open(sample_file, 'rb') as f:
                    sample_data = f.read()
                
                sample = EnhancedSignalSample.deserialize(sample_data, format='pickle', compressed=True)
                samples.append(sample)
            
            # Update last accessed
            self.metadata['datasets'][dataset_name]['last_accessed'] = datetime.now().isoformat()
            self._save_metadata()
            
            self.logger.info(f"Loaded {len(samples)} samples from cached dataset {dataset_name}")
            return samples
            
        except Exception as e:
            self.logger.error(f"Error loading cached dataset {dataset_name}: {e}")
            return None
    
    def is_dataset_cached(self, dataset_name: str) -> bool:
        """Check if dataset is cached"""
        return dataset_name in self.metadata['datasets']
    
    def _cleanup_cache(self):
        """Cleanup cache if it exceeds size limit"""
        max_size_bytes = self.max_size_gb * 1024 * 1024 * 1024
        
        if self.metadata['total_size_bytes'] <= max_size_bytes:
            return
        
        # Sort datasets by last accessed (LRU)
        datasets = list(self.metadata['datasets'].items())
        datasets.sort(key=lambda x: x[1]['last_accessed'])
        
        # Remove oldest datasets until under limit
        for dataset_name, dataset_info in datasets:
            if self.metadata['total_size_bytes'] <= max_size_bytes:
                break
            
            self._remove_dataset(dataset_name)
    
    def _remove_dataset(self, dataset_name: str):
        """Remove dataset from cache"""
        try:
            dataset_dir = self.cache_dir / dataset_name
            if dataset_dir.exists():
                import shutil
                shutil.rmtree(dataset_dir)
            
            if dataset_name in self.metadata['datasets']:
                size_bytes = self.metadata['datasets'][dataset_name]['size_bytes']
                self.metadata['total_size_bytes'] -= size_bytes
                del self.metadata['datasets'][dataset_name]
                self._save_metadata()
            
            self.logger.info(f"Removed cached dataset {dataset_name}")
            
        except Exception as e:
            self.logger.error(f"Error removing cached dataset {dataset_name}: {e}")


class MobileClient(BaseClient):
    """Cross-platform mobile client for federated learning"""
    
    def __init__(self, client_id: str, server_url: str, cache_dir: Optional[Path] = None):
        self.client_id = client_id
        self.server_url = server_url.rstrip('/')
        self.logger = logging.getLogger(__name__)
        
        # Initialize managers
        self.battery_manager = BatteryManager()
        self.network_manager = NetworkManager()
        
        # Initialize cache
        if cache_dir is None:
            cache_dir = Path.home() / ".federated_mobile_cache"
        self.dataset_cache = DatasetCache(cache_dir)
        
        # Device capabilities
        self.capabilities = self._detect_device_capabilities()
        
        # Configuration
        self.training_config = MobileTrainingConfig()
        
        # State
        self.is_registered = False
        self.auth_token = None
        self.current_model = None
        self.training_active = False
        self.background_training_enabled = True
        
        # Data loaders
        self.data_loaders = {
            'radioml': RadioMLLoader(),
            'hdf5': HDF5Loader(),
            'binary': BinaryIQLoader()
        }
        
        # Initialize optimization managers
        self.background_training_manager = BackgroundTrainingManager(
            BackgroundTrainingConfig(enabled=True)
        )
        self.network_handoff_manager = NetworkHandoffManager()
        self.adaptive_complexity_manager = AdaptiveModelComplexityManager()
        self.incentive_manager = IncentiveReputationManager(self.client_id)
        
        # Set up callbacks
        self.background_training_manager.set_training_callback(self._background_training_callback)
        self.network_handoff_manager.set_handoff_callback(self._network_handoff_callback)
        
        # Start monitoring
        self.battery_manager.start_monitoring(self._on_battery_change)
        self.background_training_manager.start_system_monitoring()
        self.network_handoff_manager.start_monitoring()
    
    def _detect_device_capabilities(self) -> MobileDeviceCapabilities:
        """Detect mobile device capabilities"""
        try:
            # Get system information
            cpu_count = psutil.cpu_count(logical=True)
            cpu_freq = psutil.cpu_freq()
            memory = psutil.virtual_memory()
            
            # Get battery info
            battery_info = self.battery_manager.get_battery_info()
            
            # Get storage info
            disk_usage = psutil.disk_usage('/')
            
            capabilities = MobileDeviceCapabilities(
                platform=platform.system().lower(),
                cpu_cores=cpu_count,
                cpu_frequency_ghz=cpu_freq.current / 1000.0 if cpu_freq else 2.0,
                total_memory_gb=memory.total / (1024**3),
                available_memory_gb=memory.available / (1024**3),
                battery_level=battery_info['level'],
                is_charging=battery_info['is_charging'],
                thermal_state="normal",
                network_type="wifi",
                is_metered_connection=False,
                gpu_available=False,  # Would need GPU detection
                gpu_memory_gb=0.0,
                storage_available_gb=disk_usage.free / (1024**3)
            )
            
            self.logger.info(f"Detected device capabilities: {capabilities.platform}, "
                           f"{capabilities.cpu_cores} cores, "
                           f"{capabilities.total_memory_gb:.1f}GB RAM")
            
            return capabilities
            
        except Exception as e:
            self.logger.error(f"Error detecting device capabilities: {e}")
            # Return default capabilities
            return MobileDeviceCapabilities(
                platform="unknown",
                cpu_cores=4,
                cpu_frequency_ghz=2.0,
                total_memory_gb=4.0,
                available_memory_gb=2.0
            )
    
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the mobile client"""
        try:
            # Update configuration
            if 'training_config' in config:
                training_config_dict = config['training_config']
                for key, value in training_config_dict.items():
                    if hasattr(self.training_config, key):
                        setattr(self.training_config, key, value)
            
            # Update cache settings
            if 'max_cache_size_gb' in config:
                self.dataset_cache.max_size_gb = config['max_cache_size_gb']
            
            # Register with server
            if not self.is_registered:
                success = self._register_with_server()
                if not success:
                    return False
            
            self.logger.info("Mobile client initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing mobile client: {e}")
            return False
    
    def _register_with_server(self) -> bool:
        """Register with aggregation server"""
        try:
            client_info = self.get_client_info()
            
            response = requests.post(
                f"{self.server_url}/register",
                json=asdict(client_info),
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                self.auth_token = result.get('token')
                self.is_registered = True
                
                self.logger.info(f"Successfully registered with server")
                return True
            else:
                self.logger.error(f"Registration failed: {response.status_code} - {response.text}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error registering with server: {e}")
            return False
    
    def get_client_info(self) -> ClientInfo:
        """Get client information"""
        # Update capabilities
        self.capabilities = self._detect_device_capabilities()
        
        return ClientInfo(
            client_id=self.client_id,
            client_type="Mobile",
            capabilities=self.capabilities.to_dict(),
            location=None,  # Would need location services
            network_info=self.network_manager.get_network_conditions().to_dict(),
            hardware_specs={
                'platform': self.capabilities.platform,
                'cpu_cores': self.capabilities.cpu_cores,
                'memory_gb': self.capabilities.total_memory_gb,
                'gpu_available': self.capabilities.gpu_available
            },
            reputation_score=1.0
        )
    
    def train_local_model(self, training_data: Any) -> ModelUpdate:
        """Train local model with mobile-specific optimizations"""
        if self.training_active:
            raise RuntimeError("Training already in progress")
        
        try:
            self.training_active = True
            
            # Check if training can start
            can_train, reason = self._can_start_training()
            if not can_train:
                raise RuntimeError(f"Cannot start training: {reason}")
            
            self.logger.info("Starting mobile federated learning training")
            
            # Prepare training data
            processed_data = self._prepare_training_data(training_data)
            
            # Adaptive configuration based on device capabilities
            training_params = self._get_adaptive_training_params()
            
            # Simulate training (in real implementation, this would use actual ML framework)
            training_result = self._simulate_training(processed_data, training_params)
            
            # Create enhanced model update
            model_update = self._create_model_update(training_result, training_params)
            
            self.logger.info(f"Training completed in {training_result['training_time']:.1f}s")
            
            return model_update.to_legacy_format()
            
        except Exception as e:
            self.logger.error(f"Training error: {e}")
            raise
        finally:
            self.training_active = False
    
    def _can_start_training(self) -> Tuple[bool, str]:
        """Check if training can start based on device constraints"""
        # Battery check
        can_train_battery, battery_reason = self.battery_manager.can_start_training(self.training_config)
        if not can_train_battery:
            return False, battery_reason
        
        # Network check
        should_defer, network_reason = self.network_manager.should_defer_training()
        if should_defer:
            return False, network_reason
        
        # Memory check
        if self.capabilities.available_memory_gb < 1.0:
            return False, "Insufficient available memory"
        
        # Thermal check
        if self.capabilities.thermal_state in ["hot", "critical"]:
            return False, f"Device thermal state: {self.capabilities.thermal_state}"
        
        return True, "All constraints satisfied"
    
    def _prepare_training_data(self, training_data: Any) -> Dict[str, Any]:
        """Prepare training data for mobile training"""
        if isinstance(training_data, str):
            # Dataset name - try to load from cache or download
            return self._load_dataset(training_data)
        elif isinstance(training_data, list):
            # List of signal samples
            return self._process_signal_samples(training_data)
        else:
            # Assume it's already processed data
            return training_data
    
    def _load_dataset(self, dataset_name: str) -> Dict[str, Any]:
        """Load dataset from cache or download"""
        # Try cache first
        cached_samples = self.dataset_cache.load_cached_dataset(dataset_name)
        if cached_samples:
            self.logger.info(f"Loaded {len(cached_samples)} samples from cache")
            return self._process_signal_samples(cached_samples)
        
        # Download and cache dataset
        self.logger.info(f"Downloading dataset {dataset_name}")
        samples = self._download_dataset(dataset_name)
        
        if samples:
            # Cache for future use
            self.dataset_cache.cache_dataset(dataset_name, f"dataset://{dataset_name}", samples)
            return self._process_signal_samples(samples)
        
        raise RuntimeError(f"Could not load dataset: {dataset_name}")
    
    def _download_dataset(self, dataset_name: str) -> Optional[List[EnhancedSignalSample]]:
        """Download dataset from server or public source"""
        try:
            # In a real implementation, this would download from actual sources
            # For now, generate synthetic data
            self.logger.info(f"Generating synthetic dataset: {dataset_name}")
            
            samples = []
            modulations = [ModulationType.QPSK, ModulationType.QAM16, ModulationType.BPSK]
            
            for i in range(100):  # Small dataset for mobile
                # Generate synthetic IQ data
                samples_per_symbol = 8
                num_symbols = 128
                iq_data = self._generate_synthetic_signal(
                    modulations[i % len(modulations)], 
                    num_symbols, 
                    samples_per_symbol
                )
                
                sample = EnhancedSignalSample(
                    iq_data=iq_data,
                    timestamp=datetime.now(),
                    duration=len(iq_data) / 200000,  # 200kHz sample rate
                    rf_params=self._get_default_rf_params(),
                    modulation_type=modulations[i % len(modulations)],
                    device_id=self.client_id,
                    environment="synthetic"
                )
                
                samples.append(sample)
            
            return samples
            
        except Exception as e:
            self.logger.error(f"Error downloading dataset {dataset_name}: {e}")
            return None
    
    def _generate_synthetic_signal(self, modulation: ModulationType, 
                                 num_symbols: int, samples_per_symbol: int) -> np.ndarray:
        """Generate synthetic signal for testing"""
        # Simple signal generation for demonstration
        t = np.arange(num_symbols * samples_per_symbol)
        
        if modulation == ModulationType.QPSK:
            # QPSK signal
            symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], num_symbols)
            signal = np.repeat(symbols, samples_per_symbol)
        elif modulation == ModulationType.QAM16:
            # 16-QAM signal
            constellation = [1+1j, 1+3j, 3+1j, 3+3j, 1-1j, 1-3j, 3-1j, 3-3j,
                           -1+1j, -1+3j, -3+1j, -3+3j, -1-1j, -1-3j, -3-1j, -3-3j]
            symbols = np.random.choice(constellation, num_symbols)
            signal = np.repeat(symbols, samples_per_symbol)
        else:  # BPSK
            symbols = np.random.choice([1+0j, -1+0j], num_symbols)
            signal = np.repeat(symbols, samples_per_symbol)
        
        # Add noise
        noise_power = 0.1
        noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
        
        return signal + noise
    
    def _get_default_rf_params(self):
        """Get default RF parameters"""
        from ..common.signal_models import RFParameters
        return RFParameters(
            center_frequency=915e6,
            sample_rate=200e3,
            gain=20.0,
            bandwidth=100e3
        )
    
    def _process_signal_samples(self, samples: List[EnhancedSignalSample]) -> Dict[str, Any]:
        """Process signal samples for training"""
        processed_data = {
            'iq_samples': [],
            'labels': [],
            'modulation_types': [],
            'snr_values': [],
            'sample_count': len(samples)
        }
        
        for sample in samples:
            # Extract features
            features = sample.extract_features()
            
            # Use IQ real/imag representation
            processed_data['iq_samples'].append(features['iq_real_imag'])
            processed_data['labels'].append(sample.modulation_type.value)
            processed_data['modulation_types'].append(sample.modulation_type.value)
            processed_data['snr_values'].append(sample.quality_metrics.snr_db)
        
        return processed_data
    
    def _get_adaptive_training_params(self) -> Dict[str, Any]:
        """Get adaptive training parameters based on device capabilities"""
        params = {
            'batch_size': 32,
            'learning_rate': 0.001,
            'epochs': 5,
            'model_complexity': 'medium'
        }
        
        # Adapt based on memory
        if self.capabilities.available_memory_gb < 2.0:
            params['batch_size'] = 16
            params['model_complexity'] = 'low'
        elif self.capabilities.available_memory_gb > 4.0:
            params['batch_size'] = 64
            params['model_complexity'] = 'high'
        
        # Adapt based on CPU
        if self.capabilities.cpu_cores < 4:
            params['epochs'] = 3
        elif self.capabilities.cpu_cores > 6:
            params['epochs'] = 8
        
        # Adapt based on battery
        if self.capabilities.battery_level and self.capabilities.battery_level < 0.5:
            params['epochs'] = max(1, params['epochs'] // 2)
            params['batch_size'] = min(16, params['batch_size'])
        
        # Adapt based on thermal state
        if self.capabilities.thermal_state == "warm":
            params['epochs'] = max(1, params['epochs'] - 1)
        elif self.capabilities.thermal_state in ["hot", "critical"]:
            params['epochs'] = 1
            params['batch_size'] = 8
        
        self.logger.info(f"Adaptive training params: {params}")
        return params
    
    def _simulate_training(self, data: Dict[str, Any], params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate model training (replace with actual ML training)"""
        start_time = time.time()
        
        # Simulate training time based on data size and parameters
        base_time = len(data['iq_samples']) * params['epochs'] * 0.001
        training_time = base_time * (1.0 + np.random.random() * 0.2)  # Add some variance
        
        # Simulate training by sleeping (in real implementation, this would be actual training)
        time.sleep(min(training_time, 5.0))  # Cap at 5 seconds for demo
        
        actual_training_time = time.time() - start_time
        
        # Simulate training metrics
        training_loss = 2.0 * np.exp(-params['epochs'] * 0.3) + 0.1 * np.random.random()
        training_accuracy = 0.6 + 0.3 * (1 - np.exp(-params['epochs'] * 0.2)) + 0.05 * np.random.random()
        
        # Generate fake model weights
        model_size = 1024 * 1024  # 1MB model
        model_weights = np.random.randn(model_size // 4).astype(np.float32).tobytes()
        
        return {
            'model_weights': model_weights,
            'training_time': actual_training_time,
            'training_loss': training_loss,
            'training_accuracy': training_accuracy,
            'samples_used': len(data['iq_samples']),
            'epochs_completed': params['epochs'],
            'batch_size': params['batch_size']
        }
    
    def _create_model_update(self, training_result: Dict[str, Any], 
                           training_params: Dict[str, Any]) -> EnhancedModelUpdate:
        """Create enhanced model update with mobile-specific metadata"""
        # Get current system state
        network_conditions = self.network_manager.get_network_conditions()
        
        # Create compute resources info
        compute_resources = ComputeResources(
            cpu_cores=self.capabilities.cpu_cores,
            cpu_frequency_ghz=self.capabilities.cpu_frequency_ghz,
            memory_gb=self.capabilities.total_memory_gb,
            available_memory_gb=self.capabilities.available_memory_gb,
            gpu_available=self.capabilities.gpu_available,
            gpu_memory_gb=self.capabilities.gpu_memory_gb,
            battery_level=self.capabilities.battery_level,
            thermal_state=self.capabilities.thermal_state,
            power_source="battery" if not self.capabilities.is_charging else "plugged"
        )
        
        # Create privacy budget (basic implementation)
        privacy_budget = PrivacyBudget(epsilon=1.0, delta=1e-5)
        
        # Create enhanced model update
        model_update = EnhancedModelUpdate(
            client_id=self.client_id,
            model_weights=training_result['model_weights'],
            model_size_bytes=len(training_result['model_weights']),
            training_rounds=1,
            local_epochs=training_result['epochs_completed'],
            batch_size=training_result['batch_size'],
            learning_rate=training_params['learning_rate'],
            samples_used=training_result['samples_used'],
            training_time_seconds=training_result['training_time'],
            training_loss=training_result['training_loss'],
            training_accuracy=training_result['training_accuracy'],
            network_conditions=network_conditions,
            compute_resources=compute_resources,
            privacy_budget_used=privacy_budget,
            compression_type=CompressionType.NONE,
            model_version="1.0",
            client_version="1.0",
            custom_metadata={
                'platform': self.capabilities.platform,
                'device_type': 'mobile',
                'training_mode': 'adaptive',
                'thermal_state': self.capabilities.thermal_state,
                'battery_level': self.capabilities.battery_level
            }
        )
        
        return model_update
    
    def receive_global_model(self, model_weights: bytes) -> bool:
        """Receive and apply global model update"""
        try:
            self.logger.info(f"Received global model update ({len(model_weights)} bytes)")
            
            # Store the model
            self.current_model = model_weights
            
            # In a real implementation, you would load this into your ML framework
            # For now, just store it
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error receiving global model: {e}")
            return False
    
    def _on_battery_change(self, battery_info: Dict[str, Any]):
        """Handle battery level changes"""
        self.capabilities.battery_level = battery_info['level']
        self.capabilities.is_charging = battery_info['is_charging']
        
        # If battery is critically low and training is active, consider stopping
        if (battery_info['level'] < 0.15 and not battery_info['is_charging'] 
            and self.training_active):
            self.logger.warning("Battery critically low during training")
            # In a real implementation, you might pause or stop training
    
    def enable_background_training(self, enabled: bool = True):
        """Enable or disable background training"""
        self.background_training_enabled = enabled
        self.training_config.background_training_enabled = enabled
        self.logger.info(f"Background training {'enabled' if enabled else 'disabled'}")
    
    def get_training_status(self) -> Dict[str, Any]:
        """Get current training status"""
        base_status = {
            'training_active': self.training_active,
            'background_training_enabled': self.background_training_enabled,
            'is_registered': self.is_registered,
            'battery_level': self.capabilities.battery_level,
            'is_charging': self.capabilities.is_charging,
            'thermal_state': self.capabilities.thermal_state,
            'available_memory_gb': self.capabilities.available_memory_gb,
            'network_conditions': self.network_manager.get_network_conditions().to_dict(),
            'cached_datasets': list(self.dataset_cache.metadata['datasets'].keys())
        }
        
        # Add optimization status
        optimization_status = self.get_optimization_status()
        base_status.update({
            'optimizations': optimization_status,
            'reputation_score': self.get_reputation_score()
        })
        
        return base_status
    
    def _background_training_callback(self, training_params: Dict[str, Any]) -> Dict[str, Any]:
        """Callback for background training execution"""
        try:
            self.logger.info("Executing background training")
            
            # Use cached dataset for background training
            dataset_name = "background_training_dataset"
            if not self.dataset_cache.is_dataset_cached(dataset_name):
                # Generate small synthetic dataset for background training
                samples = self._download_dataset(dataset_name)
                if samples:
                    self.dataset_cache.cache_dataset(dataset_name, f"synthetic://{dataset_name}", samples[:20])  # Small dataset
            
            # Load training data
            training_data = self.dataset_cache.load_cached_dataset(dataset_name)
            if not training_data:
                return {'success': False, 'error': 'No training data available'}
            
            # Adapt training parameters for background mode
            adaptive_params = self._get_adaptive_training_params()
            adaptive_params['epochs'] = min(3, adaptive_params['epochs'])  # Reduce epochs for background
            adaptive_params['batch_size'] = min(16, adaptive_params['batch_size'])  # Smaller batches
            
            # Simulate background training
            training_result = self._simulate_training(
                self._process_signal_samples(training_data), 
                adaptive_params
            )
            
            # Record participation for incentives
            participation_result = {
                'success': True,
                'samples_used': len(training_data),
                'training_time_minutes': training_result['training_time'] / 60,
                'model_quality_score': 0.7 + 0.2 * np.random.random(),
                'efficiency_score': 0.8 + 0.1 * np.random.random(),
                'data_quality_score': 0.75 + 0.15 * np.random.random(),
                'background_mode': True
            }
            
            self.incentive_manager.record_training_participation(participation_result)
            
            return {
                'success': True,
                'training_result': training_result,
                'participation_record': participation_result
            }
            
        except Exception as e:
            self.logger.error(f"Background training error: {e}")
            
            # Record failed participation
            self.incentive_manager.record_training_participation({
                'success': False,
                'error': str(e),
                'background_mode': True
            })
            
            return {'success': False, 'error': str(e)}
    
    def _network_handoff_callback(self, handoff_event: Dict[str, Any]):
        """Callback for network handoff events"""
        self.logger.info(f"Network handoff event: {handoff_event['from_network']['type']} -> {handoff_event['to_network']['type']}")
        
        # Pause training if handoff is in progress and training is active
        if self.training_active and not handoff_event['success']:
            self.logger.warning("Pausing training due to network handoff issues")
            # In a real implementation, you might pause the training here
    
    def enable_background_training(self, enabled: bool = True):
        """Enable or disable background training"""
        self.background_training_enabled = enabled
        self.training_config.background_training_enabled = enabled
        self.background_training_manager.config.enabled = enabled
        self.logger.info(f"Background training {'enabled' if enabled else 'disabled'}")
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get status of all optimization features"""
        return {
            'background_training': self.background_training_manager.get_training_statistics(),
            'network_handoff': self.network_handoff_manager.get_network_status(),
            'adaptive_complexity': {
                'current_profile': self.adaptive_complexity_manager.current_profile.profile_name if self.adaptive_complexity_manager.current_profile else None,
                'adaptation_history_count': len(self.adaptive_complexity_manager.adaptation_history)
            },
            'incentives': self.incentive_manager.get_reputation_breakdown(),
            'rewards': self.incentive_manager.get_reward_summary()
        }
    
    def get_reputation_score(self) -> float:
        """Get current reputation score"""
        return self.incentive_manager.metrics.reputation_score
    
    def force_model_complexity_adaptation(self) -> Dict[str, Any]:
        """Force adaptation of model complexity based on current conditions"""
        network_conditions = self.network_manager.get_network_conditions()
        
        compute_resources = ComputeResources(
            cpu_cores=self.capabilities.cpu_cores,
            cpu_frequency_ghz=self.capabilities.cpu_frequency_ghz,
            memory_gb=self.capabilities.total_memory_gb,
            available_memory_gb=self.capabilities.available_memory_gb,
            gpu_available=self.capabilities.gpu_available,
            gpu_memory_gb=self.capabilities.gpu_memory_gb,
            battery_level=self.capabilities.battery_level,
            thermal_state=self.capabilities.thermal_state,
            power_source="battery" if not self.capabilities.is_charging else "plugged"
        )
        
        constraints = {
            'max_training_time_minutes': self.training_config.max_training_time_minutes
        }
        
        selected_profile = self.adaptive_complexity_manager.select_optimal_profile(
            compute_resources, network_conditions, constraints
        )
        
        return {
            'selected_profile': selected_profile.profile_name,
            'profile_details': selected_profile.to_dict(),
            'adaptation_reason': 'manual_trigger'
        }
    
    def cleanup(self):
        """Cleanup resources"""
        self.logger.info("Cleaning up mobile client")
        self.battery_manager.stop_monitoring()
        self.background_training_manager.stop_system_monitoring()
        self.network_handoff_manager.stop_monitoring()
        self.training_active = False
    
    def __del__(self):
        """Destructor"""
        try:
            self.cleanup()
        except:
            pass