"""
Main SDR Client Interface

Provides the main interface for SDR client functionality, integrating
hardware abstraction, signal collection, and error handling.
"""
import logging
import threading
import time
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
import numpy as np

from ..common.interfaces import BaseClient, ClientInfo, ModelUpdate, TrainingConfig, SignalSample
from .device_manager import SDRDeviceManager, SDRHealthMonitor
from .hardware_abstraction import SDRConfig, SDRType, SignalBuffer
from .signal_collector import SignalCollector, CollectionConfig, StreamingCollector
from .error_handling import SDRErrorHandler, SDRError, ErrorType, ErrorSeverity, ThermalManager
from .signal_processing import (
    FeatureExtractor, ChannelSimulator, AdaptiveSignalProcessor,
    ModulationClassifier, WidebandProcessor, ChannelModel, ChannelType
)
from .federated_learning import (
    FederatedLearningClient, TrainingConfig, CompressionMethod, PrivacyMethod
)

logger = logging.getLogger(__name__)


class SDRClient(BaseClient):
    """Main SDR client implementation"""
    
    def __init__(self, client_id: str, config: Dict[str, Any]):
        self.client_id = client_id
        self.config = config
        self.device_manager = SDRDeviceManager()
        self.error_handler = SDRErrorHandler()
        self.thermal_manager = ThermalManager()
        self.health_monitor = None
        
        # Signal collection components
        self.signal_collector = SignalCollector(self.device_manager)
        self.streaming_collector = StreamingCollector(self.device_manager)
        
        # Signal processing components
        self.feature_extractor = FeatureExtractor(config.get('sample_rate', 1e6))
        self.channel_simulator = ChannelSimulator()
        self.adaptive_processor = AdaptiveSignalProcessor(config.get('sample_rate', 1e6))
        self.modulation_classifier = ModulationClassifier()
        self.wideband_processor = WidebandProcessor(config.get('sample_rate', 20e6))
        
        # Federated learning components
        self.federated_client = None
        if config.get('enable_federated_learning', False):
            self._setup_federated_learning(config)
        
        # Client state
        self.is_initialized = False
        self.is_collecting = False
        self.current_device_id = None
        self.current_sdr_config = None
        
        # Signal processing
        self.collected_samples = []
        self.processing_lock = threading.Lock()
        
        # Setup error handling
        self._setup_error_handling()
        
    def initialize(self, config: Dict[str, Any]) -> bool:
        """Initialize the SDR client"""
        try:
            logger.info(f"Initializing SDR client {self.client_id}")
            
            # Update configuration
            self.config.update(config)
            
            # Detect available devices
            detected_devices = self.device_manager.detect_all_devices()
            
            if not detected_devices:
                logger.error("No SDR devices detected")
                # Fall back to simulated device for testing
                detected_devices = {SDRType.SIMULATED: ["simulated_0"]}
                logger.info("Using simulated SDR device for testing")
            
            # Select device based on preference or availability
            device_preference = self.config.get('preferred_device_type', 'rtl_sdr')
            selected_device = self._select_device(detected_devices, device_preference)
            
            if not selected_device:
                logger.error("No suitable SDR device found")
                return False
            
            self.current_device_id = selected_device
            
            # Create SDR configuration
            sdr_requirements = self.config.get('sdr_requirements', {})
            self.current_sdr_config = self.device_manager.get_optimal_config(
                selected_device, sdr_requirements
            )
            
            if not self.current_sdr_config:
                logger.error("Failed to create SDR configuration")
                return False
            
            # Initialize device
            if not self.device_manager.initialize_device(selected_device, self.current_sdr_config):
                logger.error(f"Failed to initialize device {selected_device}")
                return False
            
            # Start health monitoring
            self.health_monitor = SDRHealthMonitor(self.device_manager)
            self.health_monitor.start_monitoring()
            
            self.is_initialized = True
            logger.info(f"SDR client {self.client_id} initialized successfully with device {selected_device}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error initializing SDR client: {e}")
            self._handle_error(ErrorType.CONFIGURATION_ERROR, ErrorSeverity.CRITICAL, str(e))
            return False
    
    def train_local_model(self, training_data: Any) -> ModelUpdate:
        """Train local model using collected signal data"""
        try:
            logger.info(f"Starting local model training for client {self.client_id}")
            
            # Start signal collection if not already collecting
            if not self.is_collecting:
                self._start_signal_collection()
            
            # Wait for sufficient data collection
            collection_duration = self.config.get('collection_duration', 30.0)  # seconds
            time.sleep(collection_duration)
            
            # Stop collection and get samples
            self._stop_signal_collection()
            
            with self.processing_lock:
                if not self.collected_samples:
                    raise ValueError("No signal samples collected for training")
                
                # Convert collected samples to training format
                training_samples = self._prepare_training_data(self.collected_samples)
                
                # Simulate model training (in practice, this would use PyTorch)
                model_weights = self._simulate_model_training(training_samples)
                
                # Calculate training metrics
                training_metrics = self._calculate_training_metrics(training_samples)
                
                # Create model update
                model_update = ModelUpdate(
                    client_id=self.client_id,
                    model_weights=model_weights,
                    training_metrics=training_metrics,
                    data_statistics=self._get_data_statistics(),
                    computation_time=collection_duration,
                    network_conditions=self._get_network_conditions(),
                    privacy_budget_used=0.0  # TODO: Implement differential privacy
                )
                
                logger.info(f"Local model training completed for client {self.client_id}")
                return model_update
                
        except Exception as e:
            logger.error(f"Error in local model training: {e}")
            self._handle_error(ErrorType.SIGNAL_QUALITY, ErrorSeverity.HIGH, str(e))
            raise
    
    def receive_global_model(self, model_weights: bytes) -> bool:
        """Receive and apply global model update"""
        try:
            logger.info(f"Receiving global model update for client {self.client_id}")
            
            # In practice, this would update the local model with global weights
            # For now, we'll just log the reception
            logger.info(f"Applied global model update ({len(model_weights)} bytes)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error receiving global model: {e}")
            return False
    
    def get_client_info(self) -> ClientInfo:
        """Get client information"""
        device_info = None
        if self.current_device_id:
            device_info = self.device_manager.get_device_info(self.current_device_id)
        
        return ClientInfo(
            client_id=self.client_id,
            client_type="SDR",
            capabilities=device_info['capabilities'] if device_info else {},
            location=self.config.get('location'),
            network_info=self._get_network_conditions(),
            hardware_specs=self._get_hardware_specs(),
            reputation_score=1.0  # TODO: Implement reputation system
        )
    
    def cleanup(self) -> bool:
        """Cleanup client resources"""
        try:
            logger.info(f"Cleaning up SDR client {self.client_id}")
            
            # Stop signal collection
            self._stop_signal_collection()
            
            # Stop health monitoring
            if self.health_monitor:
                self.health_monitor.stop_monitoring()
            
            # Cleanup devices
            self.device_manager.cleanup_all_devices()
            
            self.is_initialized = False
            logger.info(f"SDR client {self.client_id} cleaned up successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Error cleaning up SDR client: {e}")
            return False
    
    def get_signal_quality_metrics(self) -> Dict[str, Any]:
        """Get current signal quality metrics"""
        if not self.current_device_id:
            return {}
        
        try:
            # Get recent samples for analysis
            buffer = self.device_manager.read_samples(self.current_device_id, 1024)
            
            if buffer is None:
                return {'status': 'no_signal'}
            
            # Calculate signal metrics
            samples = buffer.iq_samples
            signal_power = np.mean(np.abs(samples) ** 2)
            noise_power = np.var(np.abs(samples))
            snr = 10 * np.log10(signal_power / max(noise_power, 1e-10))
            
            # Calculate other metrics
            peak_power = np.max(np.abs(samples) ** 2)
            papr = 10 * np.log10(peak_power / max(signal_power, 1e-10))
            
            return {
                'snr_db': snr,
                'signal_power_db': 10 * np.log10(signal_power),
                'papr_db': papr,
                'sample_count': len(samples),
                'frequency': buffer.frequency,
                'sample_rate': buffer.sample_rate,
                'timestamp': buffer.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error calculating signal quality metrics: {e}")
            return {'status': 'error', 'message': str(e)}
    
    def _select_device(self, detected_devices: Dict[SDRType, List[str]], 
                      preference: str) -> Optional[str]:
        """Select best available device"""
        # Priority order based on preference
        priority_order = {
            'rtl_sdr': [SDRType.RTL_SDR, SDRType.HACKRF, SDRType.SIMULATED],
            'hackrf': [SDRType.HACKRF, SDRType.RTL_SDR, SDRType.SIMULATED],
            'usrp': [SDRType.USRP, SDRType.HACKRF, SDRType.RTL_SDR, SDRType.SIMULATED],
            'simulated': [SDRType.SIMULATED]
        }
        
        order = priority_order.get(preference, priority_order['rtl_sdr'])
        
        for sdr_type in order:
            if sdr_type in detected_devices and detected_devices[sdr_type]:
                return detected_devices[sdr_type][0]  # Return first available device
        
        return None
    
    def _start_signal_collection(self) -> bool:
        """Start signal collection"""
        if self.is_collecting or not self.current_device_id:
            return False
        
        try:
            collection_config = CollectionConfig(
                device_id=self.current_device_id,
                sdr_config=self.current_sdr_config,
                buffer_size=self.config.get('buffer_size', 8192),
                processing_callback=self._process_signal_buffer
            )
            
            success = self.signal_collector.start_collection(collection_config)
            if success:
                self.is_collecting = True
                logger.info("Signal collection started")
            
            return success
            
        except Exception as e:
            logger.error(f"Error starting signal collection: {e}")
            return False
    
    def _stop_signal_collection(self) -> bool:
        """Stop signal collection"""
        if not self.is_collecting:
            return True
        
        try:
            success = self.signal_collector.stop_collection()
            if success:
                self.is_collecting = False
                logger.info("Signal collection stopped")
            
            return success
            
        except Exception as e:
            logger.error(f"Error stopping signal collection: {e}")
            return False
    
    def _process_signal_buffer(self, buffer: SignalBuffer):
        """Process incoming signal buffer with advanced signal processing"""
        try:
            with self.processing_lock:
                # Apply adaptive signal processing
                features, processing_metadata = self.adaptive_processor.process_signal_adaptive(buffer)
                
                # Classify modulation
                classification_result = self.modulation_classifier.classify_modulation(
                    buffer.iq_samples, buffer.frequency
                )
                
                # Apply channel simulation if configured
                processed_samples = buffer.iq_samples
                if self.config.get('apply_channel_simulation', False):
                    channel_model = self._get_channel_model()
                    processed_samples = self.channel_simulator.apply_channel_model(
                        buffer.iq_samples, channel_model
                    )
                
                # Convert to SignalSample format with enhanced metadata
                signal_sample = SignalSample(
                    timestamp=buffer.timestamp,
                    frequency=buffer.frequency,
                    sample_rate=buffer.sample_rate,
                    iq_data=processed_samples,
                    modulation_type=classification_result.predicted_modulation.value,
                    snr=processing_metadata.get('snr_estimate', self._estimate_snr(buffer.iq_samples)),
                    location=self.config.get('location'),
                    device_id=self.current_device_id,
                    metadata={
                        **buffer.metadata,
                        'features': features,
                        'classification': classification_result,
                        'processing': processing_metadata
                    }
                )
                
                self.collected_samples.append(signal_sample)
                
                # Limit buffer size
                max_samples = self.config.get('max_collected_samples', 1000)
                if len(self.collected_samples) > max_samples:
                    self.collected_samples = self.collected_samples[-max_samples:]
                    
        except Exception as e:
            logger.error(f"Error processing signal buffer: {e}")
            # Fallback to basic processing
            signal_sample = SignalSample(
                timestamp=buffer.timestamp,
                frequency=buffer.frequency,
                sample_rate=buffer.sample_rate,
                iq_data=buffer.iq_samples,
                modulation_type="unknown",
                snr=self._estimate_snr(buffer.iq_samples),
                location=self.config.get('location'),
                device_id=self.current_device_id,
                metadata=buffer.metadata
            )
            self.collected_samples.append(signal_sample)
    
    def _estimate_snr(self, samples: np.ndarray) -> float:
        """Estimate SNR of signal samples"""
        try:
            signal_power = np.mean(np.abs(samples) ** 2)
            noise_power = np.var(np.abs(samples))
            return 10 * np.log10(signal_power / max(noise_power, 1e-10))
        except:
            return 0.0
    
    def _prepare_training_data(self, samples: List[SignalSample]) -> np.ndarray:
        """Prepare signal samples for training"""
        # Extract features from IQ samples
        features = []
        
        for sample in samples:
            # Simple feature extraction (in practice, this would be more sophisticated)
            iq_data = sample.iq_data
            
            # Power spectral density features
            fft = np.fft.fft(iq_data)
            psd = np.abs(fft) ** 2
            
            # Statistical features
            mean_power = np.mean(np.abs(iq_data) ** 2)
            std_power = np.std(np.abs(iq_data) ** 2)
            
            # Combine features
            feature_vector = np.concatenate([
                psd[:64],  # First 64 PSD bins
                [mean_power, std_power, sample.snr]
            ])
            
            features.append(feature_vector)
        
        return np.array(features)
    
    def _simulate_model_training(self, training_data: np.ndarray) -> bytes:
        """Simulate model training and return model weights"""
        # In practice, this would use PyTorch to train a neural network
        # For now, we'll simulate by creating random weights
        
        input_size = training_data.shape[1] if len(training_data.shape) > 1 else 64
        hidden_size = 128
        output_size = 10  # Number of modulation classes
        
        # Simulate model weights
        weights = {
            'layer1': np.random.randn(input_size, hidden_size).astype(np.float32),
            'bias1': np.random.randn(hidden_size).astype(np.float32),
            'layer2': np.random.randn(hidden_size, output_size).astype(np.float32),
            'bias2': np.random.randn(output_size).astype(np.float32)
        }
        
        # Serialize weights (in practice, would use PyTorch's serialization)
        import pickle
        return pickle.dumps(weights)
    
    def _calculate_training_metrics(self, training_data: np.ndarray) -> Dict[str, float]:
        """Calculate training metrics"""
        return {
            'samples_used': len(training_data),
            'training_loss': np.random.uniform(0.1, 0.5),  # Simulated
            'training_accuracy': np.random.uniform(0.7, 0.95),  # Simulated
            'convergence_time': np.random.uniform(10, 30),  # Simulated
        }
    
    def _get_data_statistics(self) -> Dict[str, Any]:
        """Get statistics about collected data"""
        with self.processing_lock:
            if not self.collected_samples:
                return {}
            
            snr_values = [sample.snr for sample in self.collected_samples]
            frequencies = [sample.frequency for sample in self.collected_samples]
            
            return {
                'sample_count': len(self.collected_samples),
                'avg_snr': np.mean(snr_values),
                'min_snr': np.min(snr_values),
                'max_snr': np.max(snr_values),
                'frequency_range': [np.min(frequencies), np.max(frequencies)],
                'collection_duration': (
                    self.collected_samples[-1].timestamp - self.collected_samples[0].timestamp
                ).total_seconds() if len(self.collected_samples) > 1 else 0
            }
    
    def _get_network_conditions(self) -> Dict[str, Any]:
        """Get current network conditions"""
        # In practice, this would measure actual network conditions
        return {
            'bandwidth_mbps': 10.0,  # Simulated
            'latency_ms': 50.0,  # Simulated
            'packet_loss': 0.01,  # Simulated
            'connection_type': 'wifi'
        }
    
    def _get_hardware_specs(self) -> Dict[str, Any]:
        """Get hardware specifications"""
        device_info = None
        if self.current_device_id:
            device_info = self.device_manager.get_device_info(self.current_device_id)
        
        return {
            'device_type': device_info['device_type'] if device_info else 'unknown',
            'device_id': self.current_device_id,
            'capabilities': device_info['capabilities'] if device_info else {},
            'thermal_status': self.thermal_manager.get_thermal_status(self.current_device_id or 'unknown')
        }
    
    def _setup_error_handling(self):
        """Setup error handling callbacks"""
        def error_callback(error: SDRError):
            logger.error(f"SDR Error: {error.message} (Type: {error.error_type}, Severity: {error.severity})")
            
            # Take action based on error severity
            if error.severity == ErrorSeverity.CRITICAL:
                # Stop collection and try to recover
                self._stop_signal_collection()
                
        self.error_handler.register_error_callback(error_callback)
    
    def _handle_error(self, error_type: ErrorType, severity: ErrorSeverity, message: str):
        """Handle an error"""
        error = SDRError(
            error_type=error_type,
            severity=severity,
            message=message,
            timestamp=datetime.now(),
            device_id=self.current_device_id or 'unknown',
            context={
                'client_id': self.client_id,
                'is_collecting': self.is_collecting,
                'config': self.config
            }
        )
        
        self.error_handler.handle_error(error)
    
    def _get_channel_model(self) -> ChannelModel:
        """Get channel model configuration"""
        channel_config = self.config.get('channel_model', {})
        
        return ChannelModel(
            channel_type=ChannelType(channel_config.get('type', 'awgn')),
            snr_db=channel_config.get('snr_db', 20.0),
            fading_rate=channel_config.get('fading_rate', 0.0),
            rician_k_factor=channel_config.get('rician_k_factor', 0.0),
            multipath_delays=channel_config.get('multipath_delays', []),
            multipath_gains=channel_config.get('multipath_gains', []),
            doppler_shift=channel_config.get('doppler_shift', 0.0)
        )
    
    def process_wideband_signal(self, iq_samples: np.ndarray, 
                               center_frequency: float) -> Dict[str, Any]:
        """Process wideband signal for multiple simultaneous transmissions"""
        try:
            return self.wideband_processor.process_wideband_signal(
                iq_samples, center_frequency
            )
        except Exception as e:
            logger.error(f"Error processing wideband signal: {e}")
            return {'error': str(e)}
    
    def extract_signal_features(self, iq_samples: np.ndarray, 
                               frequency: float = 0.0) -> Any:
        """Extract comprehensive signal features"""
        try:
            return self.feature_extractor.extract_features(iq_samples, frequency)
        except Exception as e:
            logger.error(f"Error extracting signal features: {e}")
            return None
    
    def classify_signal_modulation(self, iq_samples: np.ndarray, 
                                  frequency: float = 0.0) -> Any:
        """Classify signal modulation type"""
        try:
            return self.modulation_classifier.classify_modulation(iq_samples, frequency)
        except Exception as e:
            logger.error(f"Error classifying modulation: {e}")
            return None
    
    def _setup_federated_learning(self, config: Dict[str, Any]):
        """Setup federated learning client"""
        try:
            # Create federated learning configuration
            fl_config = config.get('federated_learning', {})
            
            training_config = TrainingConfig(
                learning_rate=fl_config.get('learning_rate', 0.001),
                batch_size=fl_config.get('batch_size', 32),
                local_epochs=fl_config.get('local_epochs', 5),
                compression_method=CompressionMethod(fl_config.get('compression_method', 'quantization')),
                privacy_method=PrivacyMethod(fl_config.get('privacy_method', 'differential_privacy')),
                privacy_epsilon=fl_config.get('privacy_epsilon', 1.0),
                quantization_bits=fl_config.get('quantization_bits', 8)
            )
            
            # Create federated learning client
            server_url = fl_config.get('server_url', 'http://localhost:8000')
            self.federated_client = FederatedLearningClient(
                client_id=self.client_id,
                server_url=server_url,
                config=training_config
            )
            
            logger.info(f"Federated learning client initialized for {self.client_id}")
            
        except Exception as e:
            logger.error(f"Error setting up federated learning: {e}")
            self.federated_client = None
    
    def train_federated_model(self) -> bool:
        """Train federated learning model using collected signal data"""
        if not self.federated_client:
            logger.warning("Federated learning not enabled")
            return False
        
        try:
            with self.processing_lock:
                if not self.collected_samples:
                    logger.warning("No signal samples available for training")
                    return False
                
                # Extract features and labels from collected samples
                features = []
                labels = []
                
                for sample in self.collected_samples:
                    if 'features' in sample.metadata:
                        features.append(sample.metadata['features'])
                        
                        # Convert modulation type to label
                        mod_type = sample.modulation_type
                        if mod_type == "bpsk":
                            labels.append(0)
                        elif mod_type == "qpsk":
                            labels.append(1)
                        elif mod_type == "8psk":
                            labels.append(2)
                        elif mod_type == "16qam":
                            labels.append(3)
                        elif mod_type == "ofdm":
                            labels.append(4)
                        else:
                            labels.append(5)  # Unknown
                
                if not features:
                    logger.warning("No features available for training")
                    return False
                
                # Participate in federated learning round
                success = self.federated_client.participate_in_round(features, labels)
                
                if success:
                    logger.info(f"Federated learning round completed successfully")
                    return True
                else:
                    logger.error("Federated learning round failed")
                    return False
                    
        except Exception as e:
            logger.error(f"Error in federated model training: {e}")
            return False
    
    def evaluate_federated_model(self) -> Dict[str, float]:
        """Evaluate federated learning model performance"""
        if not self.federated_client:
            logger.warning("Federated learning not enabled")
            return {}
        
        try:
            with self.processing_lock:
                if not self.collected_samples:
                    logger.warning("No signal samples available for evaluation")
                    return {}
                
                # Extract features and labels
                features = []
                labels = []
                
                for sample in self.collected_samples:
                    if 'features' in sample.metadata:
                        features.append(sample.metadata['features'])
                        
                        # Convert modulation type to label
                        mod_type = sample.modulation_type
                        if mod_type == "bpsk":
                            labels.append(0)
                        elif mod_type == "qpsk":
                            labels.append(1)
                        elif mod_type == "8psk":
                            labels.append(2)
                        elif mod_type == "16qam":
                            labels.append(3)
                        elif mod_type == "ofdm":
                            labels.append(4)
                        else:
                            labels.append(5)  # Unknown
                
                if not features:
                    return {}
                
                # Evaluate model
                metrics = self.federated_client.evaluate_model(features, labels)
                logger.info(f"Model evaluation: {metrics}")
                
                return metrics
                
        except Exception as e:
            logger.error(f"Error evaluating federated model: {e}")
            return {}
    
    def get_federated_learning_status(self) -> Dict[str, Any]:
        """Get federated learning status"""
        if not self.federated_client:
            return {'enabled': False}
        
        try:
            status = self.federated_client.get_training_status()
            status['enabled'] = True
            return status
        except Exception as e:
            logger.error(f"Error getting federated learning status: {e}")
            return {'enabled': True, 'error': str(e)}