"""
SDR Hardware Abstraction Layer

This module provides a unified interface for different SDR hardware types
including RTL-SDR, HackRF, and USRP devices.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import numpy as np
import logging
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class SDRType(Enum):
    """Supported SDR hardware types"""
    RTL_SDR = "rtl_sdr"
    HACKRF = "hackrf"
    USRP = "usrp"
    SIMULATED = "simulated"


@dataclass
class SDRCapabilities:
    """SDR device capabilities"""
    frequency_range: Tuple[float, float]  # Hz
    sample_rate_range: Tuple[float, float]  # Hz
    rx_channels: int
    tx_channels: int
    gain_range: Tuple[float, float]  # dB
    bandwidth_range: Tuple[float, float]  # Hz
    supports_full_duplex: bool
    max_buffer_size: int


@dataclass
class SDRConfig:
    """SDR configuration parameters"""
    frequency: float  # Hz
    sample_rate: float  # Hz
    gain: float  # dB
    bandwidth: float  # Hz
    buffer_size: int
    antenna: str = "RX2"
    channel: int = 0


@dataclass
class SignalBuffer:
    """Signal buffer containing IQ samples"""
    iq_samples: np.ndarray
    timestamp: datetime
    frequency: float
    sample_rate: float
    gain: float
    metadata: Dict[str, Any]


class BaseSDRHardware(ABC):
    """Abstract base class for SDR hardware implementations"""
    
    def __init__(self, device_id: Optional[str] = None):
        self.device_id = device_id
        self.is_initialized = False
        self.is_streaming = False
        self.config: Optional[SDRConfig] = None
        self.capabilities: Optional[SDRCapabilities] = None
        
    @abstractmethod
    def detect_devices(self) -> List[str]:
        """Detect available SDR devices"""
        pass
    
    @abstractmethod
    def initialize(self, config: SDRConfig) -> bool:
        """Initialize SDR hardware with configuration"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> SDRCapabilities:
        """Get device capabilities"""
        pass
    
    @abstractmethod
    def start_streaming(self) -> bool:
        """Start signal streaming"""
        pass
    
    @abstractmethod
    def stop_streaming(self) -> bool:
        """Stop signal streaming"""
        pass
    
    @abstractmethod
    def read_samples(self, num_samples: int) -> Optional[SignalBuffer]:
        """Read IQ samples from device"""
        pass
    
    @abstractmethod
    def set_frequency(self, frequency: float) -> bool:
        """Set center frequency"""
        pass
    
    @abstractmethod
    def set_sample_rate(self, sample_rate: float) -> bool:
        """Set sample rate"""
        pass
    
    @abstractmethod
    def set_gain(self, gain: float) -> bool:
        """Set RF gain"""
        pass
    
    @abstractmethod
    def cleanup(self) -> bool:
        """Cleanup and release resources"""
        pass
    
    def validate_config(self, config: SDRConfig) -> bool:
        """Validate configuration against device capabilities"""
        if not self.capabilities:
            return False
            
        caps = self.capabilities
        
        # Check frequency range
        if not (caps.frequency_range[0] <= config.frequency <= caps.frequency_range[1]):
            logger.error(f"Frequency {config.frequency} outside range {caps.frequency_range}")
            return False
            
        # Check sample rate range
        if not (caps.sample_rate_range[0] <= config.sample_rate <= caps.sample_rate_range[1]):
            logger.error(f"Sample rate {config.sample_rate} outside range {caps.sample_rate_range}")
            return False
            
        # Check gain range
        if not (caps.gain_range[0] <= config.gain <= caps.gain_range[1]):
            logger.error(f"Gain {config.gain} outside range {caps.gain_range}")
            return False
            
        return True


class RTLSDRHardware(BaseSDRHardware):
    """RTL-SDR hardware implementation"""
    
    def __init__(self, device_id: Optional[str] = None):
        super().__init__(device_id)
        self.sdr = None
        
    def detect_devices(self) -> List[str]:
        """Detect RTL-SDR devices"""
        try:
            # Try to import rtlsdr library
            from rtlsdr import RtlSdr
            
            # Get device count
            device_count = RtlSdr.get_device_count()
            devices = []
            
            for i in range(device_count):
                try:
                    device_name = RtlSdr.get_device_name(i)
                    devices.append(f"rtlsdr_{i}_{device_name}")
                except Exception as e:
                    logger.warning(f"Could not get name for RTL-SDR device {i}: {e}")
                    devices.append(f"rtlsdr_{i}")
                    
            return devices
            
        except ImportError:
            logger.warning("RTL-SDR library not available")
            return []
        except Exception as e:
            logger.error(f"Error detecting RTL-SDR devices: {e}")
            return []
    
    def initialize(self, config: SDRConfig) -> bool:
        """Initialize RTL-SDR device"""
        try:
            from rtlsdr import RtlSdr
            
            # Parse device index from device_id
            device_index = 0
            if self.device_id and "rtlsdr_" in self.device_id:
                try:
                    device_index = int(self.device_id.split("_")[1])
                except (IndexError, ValueError):
                    logger.warning(f"Could not parse device index from {self.device_id}, using 0")
            
            # Create RTL-SDR instance
            self.sdr = RtlSdr(device_index=device_index)
            
            # Get capabilities first
            self.capabilities = self.get_capabilities()
            
            # Validate configuration
            if not self.validate_config(config):
                return False
            
            # Configure device
            self.sdr.sample_rate = config.sample_rate
            self.sdr.center_freq = config.frequency
            self.sdr.gain = config.gain
            
            self.config = config
            self.is_initialized = True
            
            logger.info(f"RTL-SDR initialized: freq={config.frequency/1e6:.1f}MHz, "
                       f"sr={config.sample_rate/1e6:.1f}MHz, gain={config.gain}dB")
            
            return True
            
        except ImportError:
            logger.error("RTL-SDR library not available. Install with: pip install pyrtlsdr")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize RTL-SDR: {e}")
            return False
    
    def get_capabilities(self) -> SDRCapabilities:
        """Get RTL-SDR capabilities"""
        return SDRCapabilities(
            frequency_range=(24e6, 1766e6),  # 24MHz to 1.766GHz
            sample_rate_range=(225e3, 3.2e6),  # 225kHz to 3.2MHz
            rx_channels=1,
            tx_channels=0,  # RTL-SDR is RX only
            gain_range=(0, 49.6),  # 0 to 49.6 dB
            bandwidth_range=(225e3, 3.2e6),
            supports_full_duplex=False,
            max_buffer_size=262144  # 256k samples
        )
    
    def start_streaming(self) -> bool:
        """Start streaming (RTL-SDR doesn't need explicit streaming start)"""
        if not self.is_initialized:
            logger.error("Device not initialized")
            return False
        
        self.is_streaming = True
        return True
    
    def stop_streaming(self) -> bool:
        """Stop streaming"""
        self.is_streaming = False
        return True
    
    def read_samples(self, num_samples: int) -> Optional[SignalBuffer]:
        """Read IQ samples from RTL-SDR"""
        if not self.is_initialized or not self.sdr:
            logger.error("Device not initialized")
            return None
            
        try:
            # Read samples
            samples = self.sdr.read_samples(num_samples)
            
            # Create signal buffer
            buffer = SignalBuffer(
                iq_samples=samples,
                timestamp=datetime.now(),
                frequency=self.config.frequency,
                sample_rate=self.config.sample_rate,
                gain=self.config.gain,
                metadata={
                    'device_type': 'rtl_sdr',
                    'device_id': self.device_id,
                    'num_samples': len(samples)
                }
            )
            
            return buffer
            
        except Exception as e:
            logger.error(f"Error reading samples: {e}")
            return None
    
    def set_frequency(self, frequency: float) -> bool:
        """Set center frequency"""
        if not self.is_initialized or not self.sdr:
            return False
            
        try:
            self.sdr.center_freq = frequency
            if self.config:
                self.config.frequency = frequency
            return True
        except Exception as e:
            logger.error(f"Error setting frequency: {e}")
            return False
    
    def set_sample_rate(self, sample_rate: float) -> bool:
        """Set sample rate"""
        if not self.is_initialized or not self.sdr:
            return False
            
        try:
            self.sdr.sample_rate = sample_rate
            if self.config:
                self.config.sample_rate = sample_rate
            return True
        except Exception as e:
            logger.error(f"Error setting sample rate: {e}")
            return False
    
    def set_gain(self, gain: float) -> bool:
        """Set RF gain"""
        if not self.is_initialized or not self.sdr:
            return False
            
        try:
            self.sdr.gain = gain
            if self.config:
                self.config.gain = gain
            return True
        except Exception as e:
            logger.error(f"Error setting gain: {e}")
            return False
    
    def cleanup(self) -> bool:
        """Cleanup RTL-SDR resources"""
        try:
            if self.sdr:
                self.sdr.close()
                self.sdr = None
            
            self.is_initialized = False
            self.is_streaming = False
            return True
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return False


class HackRFHardware(BaseSDRHardware):
    """HackRF hardware implementation"""
    
    def __init__(self, device_id: Optional[str] = None):
        super().__init__(device_id)
        self.sdr = None
        
    def detect_devices(self) -> List[str]:
        """Detect HackRF devices"""
        try:
            # Try to import hackrf library
            import hackrf
            
            # Get device list
            devices = hackrf.device_list()
            device_names = []
            
            for i, device in enumerate(devices):
                serial = device.get('serial_number', f'unknown_{i}')
                device_names.append(f"hackrf_{i}_{serial}")
                
            return device_names
            
        except ImportError:
            logger.warning("HackRF library not available")
            return []
        except Exception as e:
            logger.error(f"Error detecting HackRF devices: {e}")
            return []
    
    def initialize(self, config: SDRConfig) -> bool:
        """Initialize HackRF device"""
        try:
            import hackrf
            
            # Create HackRF instance
            self.sdr = hackrf.HackRF()
            
            # Get capabilities
            self.capabilities = self.get_capabilities()
            
            # Validate configuration
            if not self.validate_config(config):
                return False
            
            # Configure device
            self.sdr.sample_rate = config.sample_rate
            self.sdr.center_freq = config.frequency
            self.sdr.lna_gain = min(config.gain, 40)  # LNA gain max 40dB
            self.sdr.vga_gain = min(config.gain - 40, 62) if config.gain > 40 else 0
            
            self.config = config
            self.is_initialized = True
            
            logger.info(f"HackRF initialized: freq={config.frequency/1e6:.1f}MHz, "
                       f"sr={config.sample_rate/1e6:.1f}MHz, gain={config.gain}dB")
            
            return True
            
        except ImportError:
            logger.error("HackRF library not available. Install with: pip install hackrf")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize HackRF: {e}")
            return False
    
    def get_capabilities(self) -> SDRCapabilities:
        """Get HackRF capabilities"""
        return SDRCapabilities(
            frequency_range=(1e6, 6e9),  # 1MHz to 6GHz
            sample_rate_range=(2e6, 20e6),  # 2MHz to 20MHz
            rx_channels=1,
            tx_channels=1,
            gain_range=(0, 102),  # 0 to 102 dB (LNA + VGA)
            bandwidth_range=(1.75e6, 28e6),
            supports_full_duplex=False,  # Half duplex
            max_buffer_size=262144
        )
    
    def start_streaming(self) -> bool:
        """Start streaming"""
        if not self.is_initialized or not self.sdr:
            return False
            
        try:
            self.sdr.start_rx_mode()
            self.is_streaming = True
            return True
        except Exception as e:
            logger.error(f"Error starting streaming: {e}")
            return False
    
    def stop_streaming(self) -> bool:
        """Stop streaming"""
        if not self.sdr:
            return True
            
        try:
            self.sdr.stop_rx_mode()
            self.is_streaming = False
            return True
        except Exception as e:
            logger.error(f"Error stopping streaming: {e}")
            return False
    
    def read_samples(self, num_samples: int) -> Optional[SignalBuffer]:
        """Read IQ samples from HackRF"""
        if not self.is_initialized or not self.sdr:
            return None
            
        try:
            # Read samples
            samples = self.sdr.read_samples(num_samples)
            
            # Create signal buffer
            buffer = SignalBuffer(
                iq_samples=samples,
                timestamp=datetime.now(),
                frequency=self.config.frequency,
                sample_rate=self.config.sample_rate,
                gain=self.config.gain,
                metadata={
                    'device_type': 'hackrf',
                    'device_id': self.device_id,
                    'num_samples': len(samples)
                }
            )
            
            return buffer
            
        except Exception as e:
            logger.error(f"Error reading samples: {e}")
            return None
    
    def set_frequency(self, frequency: float) -> bool:
        """Set center frequency"""
        if not self.is_initialized or not self.sdr:
            return False
            
        try:
            self.sdr.center_freq = frequency
            if self.config:
                self.config.frequency = frequency
            return True
        except Exception as e:
            logger.error(f"Error setting frequency: {e}")
            return False
    
    def set_sample_rate(self, sample_rate: float) -> bool:
        """Set sample rate"""
        if not self.is_initialized or not self.sdr:
            return False
            
        try:
            self.sdr.sample_rate = sample_rate
            if self.config:
                self.config.sample_rate = sample_rate
            return True
        except Exception as e:
            logger.error(f"Error setting sample rate: {e}")
            return False
    
    def set_gain(self, gain: float) -> bool:
        """Set RF gain"""
        if not self.is_initialized or not self.sdr:
            return False
            
        try:
            # Split gain between LNA and VGA
            lna_gain = min(gain, 40)
            vga_gain = min(gain - 40, 62) if gain > 40 else 0
            
            self.sdr.lna_gain = lna_gain
            self.sdr.vga_gain = vga_gain
            
            if self.config:
                self.config.gain = gain
            return True
        except Exception as e:
            logger.error(f"Error setting gain: {e}")
            return False
    
    def cleanup(self) -> bool:
        """Cleanup HackRF resources"""
        try:
            if self.sdr:
                self.stop_streaming()
                self.sdr.close()
                self.sdr = None
            
            self.is_initialized = False
            self.is_streaming = False
            return True
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return False


class USRPHardware(BaseSDRHardware):
    """USRP hardware implementation"""
    
    def __init__(self, device_id: Optional[str] = None):
        super().__init__(device_id)
        self.usrp = None
        
    def detect_devices(self) -> List[str]:
        """Detect USRP devices"""
        try:
            # Try to import UHD library
            import uhd
            
            # Find USRP devices
            devices = uhd.find_devices()
            device_names = []
            
            for i, device in enumerate(devices):
                serial = device.get('serial', f'unknown_{i}')
                product = device.get('product', 'USRP')
                device_names.append(f"usrp_{i}_{product}_{serial}")
                
            return device_names
            
        except ImportError:
            logger.warning("UHD library not available")
            return []
        except Exception as e:
            logger.error(f"Error detecting USRP devices: {e}")
            return []
    
    def initialize(self, config: SDRConfig) -> bool:
        """Initialize USRP device"""
        try:
            import uhd
            
            # Parse device args from device_id
            device_args = ""
            if self.device_id and "usrp_" in self.device_id:
                parts = self.device_id.split("_")
                if len(parts) >= 4:
                    serial = parts[3]
                    device_args = f"serial={serial}"
            
            # Create USRP instance
            self.usrp = uhd.usrp.MultiUSRP(device_args)
            
            # Get capabilities
            self.capabilities = self.get_capabilities()
            
            # Validate configuration
            if not self.validate_config(config):
                return False
            
            # Configure device
            self.usrp.set_rx_rate(config.sample_rate)
            self.usrp.set_rx_freq(config.frequency)
            self.usrp.set_rx_gain(config.gain)
            self.usrp.set_rx_bandwidth(config.bandwidth)
            self.usrp.set_rx_antenna(config.antenna)
            
            # Allow time for configuration to settle
            time.sleep(0.1)
            
            self.config = config
            self.is_initialized = True
            
            logger.info(f"USRP initialized: freq={config.frequency/1e6:.1f}MHz, "
                       f"sr={config.sample_rate/1e6:.1f}MHz, gain={config.gain}dB")
            
            return True
            
        except ImportError:
            logger.error("UHD library not available. Install UHD and Python bindings")
            return False
        except Exception as e:
            logger.error(f"Failed to initialize USRP: {e}")
            return False
    
    def get_capabilities(self) -> SDRCapabilities:
        """Get USRP capabilities"""
        if self.usrp:
            try:
                # Get actual device capabilities
                freq_range = self.usrp.get_rx_freq_range()
                rate_range = self.usrp.get_rx_rates()
                gain_range = self.usrp.get_rx_gain_range()
                
                return SDRCapabilities(
                    frequency_range=(freq_range.start(), freq_range.stop()),
                    sample_rate_range=(min(rate_range), max(rate_range)),
                    rx_channels=self.usrp.get_rx_num_channels(),
                    tx_channels=self.usrp.get_tx_num_channels(),
                    gain_range=(gain_range.start(), gain_range.stop()),
                    bandwidth_range=(200e3, 56e6),  # Typical USRP range
                    supports_full_duplex=True,
                    max_buffer_size=1048576  # 1M samples
                )
            except Exception as e:
                logger.warning(f"Could not get actual USRP capabilities: {e}")
        
        # Return default USRP capabilities
        return SDRCapabilities(
            frequency_range=(10e6, 6e9),  # 10MHz to 6GHz (typical)
            sample_rate_range=(195e3, 61.44e6),  # 195kHz to 61.44MHz
            rx_channels=2,
            tx_channels=2,
            gain_range=(0, 76),  # 0 to 76 dB (typical)
            bandwidth_range=(200e3, 56e6),
            supports_full_duplex=True,
            max_buffer_size=1048576
        )
    
    def start_streaming(self) -> bool:
        """Start streaming"""
        if not self.is_initialized or not self.usrp:
            return False
            
        try:
            # Create stream args
            import uhd
            
            stream_args = uhd.usrp.StreamArgs("fc32", "sc16")
            stream_args.channels = [0]  # Use first channel
            
            # Create RX streamer
            self.rx_streamer = self.usrp.get_rx_stream(stream_args)
            
            # Start streaming
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.start_cont)
            stream_cmd.stream_now = True
            self.rx_streamer.issue_stream_cmd(stream_cmd)
            
            self.is_streaming = True
            return True
            
        except Exception as e:
            logger.error(f"Error starting USRP streaming: {e}")
            return False
    
    def stop_streaming(self) -> bool:
        """Stop streaming"""
        if not self.usrp or not hasattr(self, 'rx_streamer'):
            return True
            
        try:
            import uhd
            
            # Stop streaming
            stream_cmd = uhd.types.StreamCMD(uhd.types.StreamMode.stop_cont)
            self.rx_streamer.issue_stream_cmd(stream_cmd)
            
            self.is_streaming = False
            return True
            
        except Exception as e:
            logger.error(f"Error stopping USRP streaming: {e}")
            return False
    
    def read_samples(self, num_samples: int) -> Optional[SignalBuffer]:
        """Read IQ samples from USRP"""
        if not self.is_initialized or not self.usrp or not hasattr(self, 'rx_streamer'):
            return None
            
        try:
            import uhd
            
            # Allocate buffer
            recv_buffer = np.zeros(num_samples, dtype=np.complex64)
            metadata = uhd.types.RXMetadata()
            
            # Receive samples
            samples_received = self.rx_streamer.recv(recv_buffer, metadata)
            
            if samples_received == 0:
                return None
            
            # Trim buffer to actual samples received
            if samples_received < num_samples:
                recv_buffer = recv_buffer[:samples_received]
            
            # Create signal buffer
            buffer = SignalBuffer(
                iq_samples=recv_buffer,
                timestamp=datetime.now(),
                frequency=self.config.frequency,
                sample_rate=self.config.sample_rate,
                gain=self.config.gain,
                metadata={
                    'device_type': 'usrp',
                    'device_id': self.device_id,
                    'num_samples': len(recv_buffer),
                    'has_timespec': metadata.has_time_spec,
                    'error_code': metadata.error_code.name if hasattr(metadata.error_code, 'name') else str(metadata.error_code)
                }
            )
            
            return buffer
            
        except Exception as e:
            logger.error(f"Error reading USRP samples: {e}")
            return None
    
    def set_frequency(self, frequency: float) -> bool:
        """Set center frequency"""
        if not self.is_initialized or not self.usrp:
            return False
            
        try:
            self.usrp.set_rx_freq(frequency)
            if self.config:
                self.config.frequency = frequency
            return True
        except Exception as e:
            logger.error(f"Error setting USRP frequency: {e}")
            return False
    
    def set_sample_rate(self, sample_rate: float) -> bool:
        """Set sample rate"""
        if not self.is_initialized or not self.usrp:
            return False
            
        try:
            self.usrp.set_rx_rate(sample_rate)
            if self.config:
                self.config.sample_rate = sample_rate
            return True
        except Exception as e:
            logger.error(f"Error setting USRP sample rate: {e}")
            return False
    
    def set_gain(self, gain: float) -> bool:
        """Set RF gain"""
        if not self.is_initialized or not self.usrp:
            return False
            
        try:
            self.usrp.set_rx_gain(gain)
            if self.config:
                self.config.gain = gain
            return True
        except Exception as e:
            logger.error(f"Error setting USRP gain: {e}")
            return False
    
    def cleanup(self) -> bool:
        """Cleanup USRP resources"""
        try:
            if hasattr(self, 'rx_streamer'):
                self.stop_streaming()
                delattr(self, 'rx_streamer')
            
            if self.usrp:
                self.usrp = None
            
            self.is_initialized = False
            self.is_streaming = False
            return True
            
        except Exception as e:
            logger.error(f"Error during USRP cleanup: {e}")
            return False


class SimulatedSDRHardware(BaseSDRHardware):
    """Simulated SDR hardware for testing"""
    
    def __init__(self, device_id: Optional[str] = None):
        super().__init__(device_id or "simulated_0")
        
    def detect_devices(self) -> List[str]:
        """Return simulated devices"""
        return ["simulated_0", "simulated_1"]
    
    def initialize(self, config: SDRConfig) -> bool:
        """Initialize simulated device"""
        self.capabilities = self.get_capabilities()
        
        if not self.validate_config(config):
            return False
            
        self.config = config
        self.is_initialized = True
        
        logger.info(f"Simulated SDR initialized: freq={config.frequency/1e6:.1f}MHz, "
                   f"sr={config.sample_rate/1e6:.1f}MHz, gain={config.gain}dB")
        
        return True
    
    def get_capabilities(self) -> SDRCapabilities:
        """Get simulated capabilities"""
        return SDRCapabilities(
            frequency_range=(1e6, 6e9),
            sample_rate_range=(1e6, 50e6),
            rx_channels=2,
            tx_channels=2,
            gain_range=(0, 100),
            bandwidth_range=(1e6, 50e6),
            supports_full_duplex=True,
            max_buffer_size=1048576  # 1M samples
        )
    
    def start_streaming(self) -> bool:
        """Start simulated streaming"""
        self.is_streaming = True
        return True
    
    def stop_streaming(self) -> bool:
        """Stop simulated streaming"""
        self.is_streaming = False
        return True
    
    def read_samples(self, num_samples: int) -> Optional[SignalBuffer]:
        """Generate simulated IQ samples"""
        if not self.is_initialized:
            return None
            
        # Generate complex noise with some signal components
        t = np.arange(num_samples) / self.config.sample_rate
        
        # Add some simulated signals
        signal = 0.1 * np.exp(1j * 2 * np.pi * 1e6 * t)  # 1MHz tone
        noise = 0.01 * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
        
        samples = signal + noise
        
        buffer = SignalBuffer(
            iq_samples=samples,
            timestamp=datetime.now(),
            frequency=self.config.frequency,
            sample_rate=self.config.sample_rate,
            gain=self.config.gain,
            metadata={
                'device_type': 'simulated',
                'device_id': self.device_id,
                'num_samples': len(samples),
                'simulated': True
            }
        )
        
        return buffer
    
    def set_frequency(self, frequency: float) -> bool:
        """Set simulated frequency"""
        if self.config:
            self.config.frequency = frequency
        return True
    
    def set_sample_rate(self, sample_rate: float) -> bool:
        """Set simulated sample rate"""
        if self.config:
            self.config.sample_rate = sample_rate
        return True
    
    def set_gain(self, gain: float) -> bool:
        """Set simulated gain"""
        if self.config:
            self.config.gain = gain
        return True
    
    def cleanup(self) -> bool:
        """Cleanup simulated resources"""
        self.is_initialized = False
        self.is_streaming = False
        return True