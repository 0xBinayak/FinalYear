"""
SDR Device Manager

Provides unified interface for SDR device detection, initialization, and management.
Handles hardware-specific optimizations and error handling.
"""
from typing import Dict, List, Optional, Type, Any
import logging
import threading
import time
from contextlib import contextmanager

from .hardware_abstraction import (
    BaseSDRHardware, SDRType, SDRConfig, SignalBuffer,
    RTLSDRHardware, HackRFHardware, USRPHardware, SimulatedSDRHardware
)

logger = logging.getLogger(__name__)


class SDRDeviceManager:
    """Manages SDR devices and provides unified interface"""
    
    def __init__(self):
        self.hardware_classes: Dict[SDRType, Type[BaseSDRHardware]] = {
            SDRType.RTL_SDR: RTLSDRHardware,
            SDRType.HACKRF: HackRFHardware,
            SDRType.USRP: USRPHardware,
            SDRType.SIMULATED: SimulatedSDRHardware
        }
        
        self.active_devices: Dict[str, BaseSDRHardware] = {}
        self.device_lock = threading.Lock()
        
    def detect_all_devices(self) -> Dict[SDRType, List[str]]:
        """Detect all available SDR devices"""
        detected_devices = {}
        
        for sdr_type, hardware_class in self.hardware_classes.items():
            try:
                hardware = hardware_class()
                devices = hardware.detect_devices()
                if devices:
                    detected_devices[sdr_type] = devices
                    logger.info(f"Detected {len(devices)} {sdr_type.value} devices: {devices}")
                else:
                    logger.debug(f"No {sdr_type.value} devices detected")
                    
            except Exception as e:
                logger.error(f"Error detecting {sdr_type.value} devices: {e}")
                
        return detected_devices
    
    def get_device_info(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a specific device"""
        sdr_type = self._parse_device_type(device_id)
        if not sdr_type:
            return None
            
        try:
            hardware_class = self.hardware_classes[sdr_type]
            hardware = hardware_class(device_id)
            capabilities = hardware.get_capabilities()
            
            return {
                'device_id': device_id,
                'device_type': sdr_type.value,
                'capabilities': {
                    'frequency_range': capabilities.frequency_range,
                    'sample_rate_range': capabilities.sample_rate_range,
                    'rx_channels': capabilities.rx_channels,
                    'tx_channels': capabilities.tx_channels,
                    'gain_range': capabilities.gain_range,
                    'bandwidth_range': capabilities.bandwidth_range,
                    'supports_full_duplex': capabilities.supports_full_duplex,
                    'max_buffer_size': capabilities.max_buffer_size
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting device info for {device_id}: {e}")
            return None
    
    def initialize_device(self, device_id: str, config: SDRConfig) -> bool:
        """Initialize a specific SDR device"""
        with self.device_lock:
            # Check if device is already initialized
            if device_id in self.active_devices:
                logger.warning(f"Device {device_id} already initialized")
                return True
            
            sdr_type = self._parse_device_type(device_id)
            if not sdr_type:
                logger.error(f"Unknown device type for {device_id}")
                return False
            
            try:
                # Create hardware instance
                hardware_class = self.hardware_classes[sdr_type]
                hardware = hardware_class(device_id)
                
                # Initialize with configuration
                if hardware.initialize(config):
                    self.active_devices[device_id] = hardware
                    logger.info(f"Successfully initialized device {device_id}")
                    return True
                else:
                    logger.error(f"Failed to initialize device {device_id}")
                    return False
                    
            except Exception as e:
                logger.error(f"Error initializing device {device_id}: {e}")
                return False
    
    def cleanup_device(self, device_id: str) -> bool:
        """Cleanup and release a specific device"""
        with self.device_lock:
            if device_id not in self.active_devices:
                logger.warning(f"Device {device_id} not active")
                return True
            
            try:
                hardware = self.active_devices[device_id]
                success = hardware.cleanup()
                
                if success:
                    del self.active_devices[device_id]
                    logger.info(f"Successfully cleaned up device {device_id}")
                else:
                    logger.error(f"Failed to cleanup device {device_id}")
                
                return success
                
            except Exception as e:
                logger.error(f"Error cleaning up device {device_id}: {e}")
                return False
    
    def cleanup_all_devices(self) -> bool:
        """Cleanup all active devices"""
        success = True
        device_ids = list(self.active_devices.keys())
        
        for device_id in device_ids:
            if not self.cleanup_device(device_id):
                success = False
                
        return success
    
    def get_active_devices(self) -> List[str]:
        """Get list of active device IDs"""
        with self.device_lock:
            return list(self.active_devices.keys())
    
    def is_device_active(self, device_id: str) -> bool:
        """Check if device is active"""
        with self.device_lock:
            return device_id in self.active_devices
    
    def start_streaming(self, device_id: str) -> bool:
        """Start streaming on a device"""
        with self.device_lock:
            if device_id not in self.active_devices:
                logger.error(f"Device {device_id} not initialized")
                return False
            
            try:
                hardware = self.active_devices[device_id]
                return hardware.start_streaming()
            except Exception as e:
                logger.error(f"Error starting streaming on {device_id}: {e}")
                return False
    
    def stop_streaming(self, device_id: str) -> bool:
        """Stop streaming on a device"""
        with self.device_lock:
            if device_id not in self.active_devices:
                logger.error(f"Device {device_id} not initialized")
                return False
            
            try:
                hardware = self.active_devices[device_id]
                return hardware.stop_streaming()
            except Exception as e:
                logger.error(f"Error stopping streaming on {device_id}: {e}")
                return False
    
    def read_samples(self, device_id: str, num_samples: int) -> Optional[SignalBuffer]:
        """Read samples from a device"""
        with self.device_lock:
            if device_id not in self.active_devices:
                logger.error(f"Device {device_id} not initialized")
                return None
            
            try:
                hardware = self.active_devices[device_id]
                return hardware.read_samples(num_samples)
            except Exception as e:
                logger.error(f"Error reading samples from {device_id}: {e}")
                return None
    
    def set_frequency(self, device_id: str, frequency: float) -> bool:
        """Set frequency on a device"""
        with self.device_lock:
            if device_id not in self.active_devices:
                logger.error(f"Device {device_id} not initialized")
                return False
            
            try:
                hardware = self.active_devices[device_id]
                return hardware.set_frequency(frequency)
            except Exception as e:
                logger.error(f"Error setting frequency on {device_id}: {e}")
                return False
    
    def set_sample_rate(self, device_id: str, sample_rate: float) -> bool:
        """Set sample rate on a device"""
        with self.device_lock:
            if device_id not in self.active_devices:
                logger.error(f"Device {device_id} not initialized")
                return False
            
            try:
                hardware = self.active_devices[device_id]
                return hardware.set_sample_rate(sample_rate)
            except Exception as e:
                logger.error(f"Error setting sample rate on {device_id}: {e}")
                return False
    
    def set_gain(self, device_id: str, gain: float) -> bool:
        """Set gain on a device"""
        with self.device_lock:
            if device_id not in self.active_devices:
                logger.error(f"Device {device_id} not initialized")
                return False
            
            try:
                hardware = self.active_devices[device_id]
                return hardware.set_gain(gain)
            except Exception as e:
                logger.error(f"Error setting gain on {device_id}: {e}")
                return False
    
    @contextmanager
    def device_context(self, device_id: str, config: SDRConfig):
        """Context manager for device operations"""
        try:
            # Initialize device
            if not self.initialize_device(device_id, config):
                raise RuntimeError(f"Failed to initialize device {device_id}")
            
            yield device_id
            
        finally:
            # Cleanup device
            self.cleanup_device(device_id)
    
    def _parse_device_type(self, device_id: str) -> Optional[SDRType]:
        """Parse device type from device ID"""
        if device_id.startswith("rtlsdr_"):
            return SDRType.RTL_SDR
        elif device_id.startswith("hackrf_"):
            return SDRType.HACKRF
        elif device_id.startswith("usrp_"):
            return SDRType.USRP
        elif device_id.startswith("simulated_"):
            return SDRType.SIMULATED
        else:
            return None
    
    def get_optimal_config(self, device_id: str, requirements: Dict[str, Any]) -> Optional[SDRConfig]:
        """Get optimal configuration for device based on requirements"""
        device_info = self.get_device_info(device_id)
        if not device_info:
            return None
        
        caps = device_info['capabilities']
        
        # Extract requirements with defaults
        req_freq = requirements.get('frequency', 100e6)
        req_sample_rate = requirements.get('sample_rate', 2e6)
        req_gain = requirements.get('gain', 20)
        req_bandwidth = requirements.get('bandwidth', req_sample_rate)
        
        # Clamp values to device capabilities
        frequency = max(caps['frequency_range'][0], 
                       min(caps['frequency_range'][1], req_freq))
        
        sample_rate = max(caps['sample_rate_range'][0],
                         min(caps['sample_rate_range'][1], req_sample_rate))
        
        gain = max(caps['gain_range'][0],
                  min(caps['gain_range'][1], req_gain))
        
        bandwidth = max(caps['bandwidth_range'][0],
                       min(caps['bandwidth_range'][1], req_bandwidth))
        
        buffer_size = min(caps['max_buffer_size'], 
                         requirements.get('buffer_size', caps['max_buffer_size'] // 4))
        
        return SDRConfig(
            frequency=frequency,
            sample_rate=sample_rate,
            gain=gain,
            bandwidth=bandwidth,
            buffer_size=buffer_size,
            antenna=requirements.get('antenna', 'RX2'),
            channel=requirements.get('channel', 0)
        )


class SDRHealthMonitor:
    """Monitor SDR device health and performance"""
    
    def __init__(self, device_manager: SDRDeviceManager):
        self.device_manager = device_manager
        self.monitoring = False
        self.monitor_thread = None
        self.health_stats = {}
        
    def start_monitoring(self, interval: float = 5.0):
        """Start health monitoring"""
        if self.monitoring:
            return
            
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitor_loop,
            args=(interval,),
            daemon=True
        )
        self.monitor_thread.start()
        logger.info("Started SDR health monitoring")
    
    def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("Stopped SDR health monitoring")
    
    def get_health_stats(self, device_id: str) -> Optional[Dict[str, Any]]:
        """Get health statistics for a device"""
        return self.health_stats.get(device_id)
    
    def _monitor_loop(self, interval: float):
        """Main monitoring loop"""
        while self.monitoring:
            try:
                active_devices = self.device_manager.get_active_devices()
                
                for device_id in active_devices:
                    self._check_device_health(device_id)
                
                time.sleep(interval)
                
            except Exception as e:
                logger.error(f"Error in health monitoring loop: {e}")
                time.sleep(interval)
    
    def _check_device_health(self, device_id: str):
        """Check health of a specific device"""
        try:
            # Try to read a small number of samples to test device
            start_time = time.time()
            buffer = self.device_manager.read_samples(device_id, 1024)
            read_time = time.time() - start_time
            
            if buffer is not None:
                # Device is healthy
                stats = {
                    'status': 'healthy',
                    'last_check': time.time(),
                    'read_time': read_time,
                    'sample_count': len(buffer.iq_samples),
                    'frequency': buffer.frequency,
                    'sample_rate': buffer.sample_rate
                }
            else:
                # Device has issues
                stats = {
                    'status': 'unhealthy',
                    'last_check': time.time(),
                    'error': 'Failed to read samples'
                }
            
            self.health_stats[device_id] = stats
            
        except Exception as e:
            self.health_stats[device_id] = {
                'status': 'error',
                'last_check': time.time(),
                'error': str(e)
            }