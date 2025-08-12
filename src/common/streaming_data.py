"""
Support for streaming real-time data from SDR hardware
"""
import numpy as np
import threading
import queue
import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import dataclass
from datetime import datetime
import logging

from .interfaces import SignalSample


@dataclass
class StreamingConfig:
    """Configuration for streaming data"""
    sample_rate: float = 2.048e6  # 2.048 MHz
    center_frequency: float = 915e6  # 915 MHz
    gain: float = 20.0  # dB
    buffer_size: int = 1024 * 16  # Samples per buffer
    num_buffers: int = 32  # Number of buffers
    device_type: str = "rtlsdr"  # rtlsdr, hackrf, usrp
    device_index: int = 0


class StreamingDataInterface:
    """Interface for streaming real-time SDR data"""
    
    def __init__(self, config: StreamingConfig):
        self.config = config
        self.is_streaming = False
        self.data_queue = queue.Queue(maxsize=config.num_buffers)
        self.streaming_thread = None
        self.callbacks = []
        self.logger = logging.getLogger(__name__)
        
        # Statistics
        self.samples_received = 0
        self.buffers_dropped = 0
        self.start_time = None
        
        # Initialize SDR device
        self.sdr_device = None
        self._initialize_sdr()
    
    def _initialize_sdr(self):
        """Initialize SDR device based on configuration"""
        try:
            if self.config.device_type == "rtlsdr":
                self._initialize_rtlsdr()
            elif self.config.device_type == "hackrf":
                self._initialize_hackrf()
            elif self.config.device_type == "usrp":
                self._initialize_usrp()
            else:
                raise ValueError(f"Unsupported device type: {self.config.device_type}")
        except ImportError as e:
            self.logger.warning(f"SDR library not available: {e}")
            self.logger.info("Falling back to simulated data")
            self.sdr_device = None
        except Exception as e:
            self.logger.error(f"Failed to initialize SDR device: {e}")
            self.sdr_device = None
    
    def _initialize_rtlsdr(self):
        """Initialize RTL-SDR device"""
        try:
            from rtlsdr import RtlSdr
            
            self.sdr_device = RtlSdr(device_index=self.config.device_index)
            self.sdr_device.sample_rate = self.config.sample_rate
            self.sdr_device.center_freq = self.config.center_frequency
            self.sdr_device.gain = self.config.gain
            
            self.logger.info(f"Initialized RTL-SDR device {self.config.device_index}")
            
        except ImportError:
            raise ImportError("pyrtlsdr library not installed. Install with: pip install pyrtlsdr")
    
    def _initialize_hackrf(self):
        """Initialize HackRF device"""
        try:
            import pyhackrf
            
            # HackRF initialization would go here
            # This is a placeholder as pyhackrf API varies
            self.logger.info("HackRF initialization placeholder")
            
        except ImportError:
            raise ImportError("pyhackrf library not installed")
    
    def _initialize_usrp(self):
        """Initialize USRP device"""
        try:
            import uhd
            
            # USRP initialization would go here
            # This is a placeholder as UHD Python API setup is complex
            self.logger.info("USRP initialization placeholder")
            
        except ImportError:
            raise ImportError("UHD Python API not installed")
    
    def start_streaming(self):
        """Start streaming data from SDR device"""
        if self.is_streaming:
            self.logger.warning("Already streaming")
            return
        
        self.is_streaming = True
        self.start_time = time.time()
        self.samples_received = 0
        self.buffers_dropped = 0
        
        # Start streaming thread
        self.streaming_thread = threading.Thread(target=self._streaming_worker)
        self.streaming_thread.daemon = True
        self.streaming_thread.start()
        
        self.logger.info("Started streaming data")
    
    def stop_streaming(self):
        """Stop streaming data"""
        if not self.is_streaming:
            return
        
        self.is_streaming = False
        
        if self.streaming_thread:
            self.streaming_thread.join(timeout=5.0)
        
        # Clear the queue
        while not self.data_queue.empty():
            try:
                self.data_queue.get_nowait()
            except queue.Empty:
                break
        
        self.logger.info("Stopped streaming data")
    
    def _streaming_worker(self):
        """Worker thread for streaming data"""
        while self.is_streaming:
            try:
                # Get data from SDR device
                if self.sdr_device:
                    iq_data = self._read_from_sdr()
                else:
                    # Simulate data if no SDR device
                    iq_data = self._generate_simulated_data()
                
                # Create signal sample
                sample = SignalSample(
                    timestamp=datetime.now(),
                    frequency=self.config.center_frequency,
                    sample_rate=self.config.sample_rate,
                    iq_data=iq_data,
                    modulation_type='unknown',
                    snr=self._estimate_snr(iq_data),
                    location=None,
                    device_id=f"{self.config.device_type}_{self.config.device_index}",
                    metadata={
                        'streaming': True,
                        'buffer_size': len(iq_data),
                        'gain': self.config.gain
                    }
                )
                
                # Add to queue
                try:
                    self.data_queue.put(sample, timeout=0.1)
                    self.samples_received += len(iq_data)
                except queue.Full:
                    self.buffers_dropped += 1
                    self.logger.warning("Buffer dropped due to full queue")
                
                # Call registered callbacks
                for callback in self.callbacks:
                    try:
                        callback(sample)
                    except Exception as e:
                        self.logger.error(f"Callback error: {e}")
                
            except Exception as e:
                self.logger.error(f"Streaming error: {e}")
                time.sleep(0.1)  # Brief pause before retry
    
    def _read_from_sdr(self) -> np.ndarray:
        """Read data from SDR device"""
        if self.config.device_type == "rtlsdr":
            return self._read_from_rtlsdr()
        elif self.config.device_type == "hackrf":
            return self._read_from_hackrf()
        elif self.config.device_type == "usrp":
            return self._read_from_usrp()
        else:
            return self._generate_simulated_data()
    
    def _read_from_rtlsdr(self) -> np.ndarray:
        """Read data from RTL-SDR device"""
        try:
            samples = self.sdr_device.read_samples(self.config.buffer_size)
            return samples
        except Exception as e:
            self.logger.error(f"RTL-SDR read error: {e}")
            return self._generate_simulated_data()
    
    def _read_from_hackrf(self) -> np.ndarray:
        """Read data from HackRF device"""
        # Placeholder for HackRF implementation
        return self._generate_simulated_data()
    
    def _read_from_usrp(self) -> np.ndarray:
        """Read data from USRP device"""
        # Placeholder for USRP implementation
        return self._generate_simulated_data()
    
    def _generate_simulated_data(self) -> np.ndarray:
        """Generate simulated IQ data for testing"""
        t = np.arange(self.config.buffer_size) / self.config.sample_rate
        
        # Generate multiple signals
        signal = np.zeros(self.config.buffer_size, dtype=complex)
        
        # Add some QPSK-like signal
        symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], self.config.buffer_size//4)
        upsampled = np.repeat(symbols, 4)
        signal += upsampled[:self.config.buffer_size] * 0.5
        
        # Add some FM-like signal
        message = np.sin(2 * np.pi * 1000 * t)  # 1 kHz message
        fm_signal = np.exp(1j * 2 * np.pi * 10000 * np.cumsum(message) / self.config.sample_rate)
        signal += fm_signal * 0.3
        
        # Add noise
        noise_power = 0.1
        noise = (np.random.normal(0, np.sqrt(noise_power/2), self.config.buffer_size) + 
                1j * np.random.normal(0, np.sqrt(noise_power/2), self.config.buffer_size))
        
        signal += noise
        
        # Simulate some time delay
        time.sleep(self.config.buffer_size / self.config.sample_rate * 0.1)
        
        return signal
    
    def _estimate_snr(self, iq_data: np.ndarray) -> float:
        """Estimate SNR from IQ data"""
        # Simple SNR estimation
        power = np.mean(np.abs(iq_data) ** 2)
        
        # Estimate noise from high-frequency components
        fft_data = np.fft.fft(iq_data)
        high_freq_power = np.mean(np.abs(fft_data[len(fft_data)//4:3*len(fft_data)//4])**2)
        
        if high_freq_power > 0:
            snr_linear = power / high_freq_power
            return 10 * np.log10(snr_linear)
        return 0.0
    
    def get_next_sample(self, timeout: float = 1.0) -> Optional[SignalSample]:
        """Get next sample from streaming queue"""
        try:
            return self.data_queue.get(timeout=timeout)
        except queue.Empty:
            return None
    
    def register_callback(self, callback: Callable[[SignalSample], None]):
        """Register callback for real-time data processing"""
        self.callbacks.append(callback)
    
    def unregister_callback(self, callback: Callable[[SignalSample], None]):
        """Unregister callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get streaming statistics"""
        if self.start_time:
            elapsed_time = time.time() - self.start_time
            sample_rate_actual = self.samples_received / elapsed_time if elapsed_time > 0 else 0
        else:
            elapsed_time = 0
            sample_rate_actual = 0
        
        return {
            'is_streaming': self.is_streaming,
            'samples_received': self.samples_received,
            'buffers_dropped': self.buffers_dropped,
            'elapsed_time': elapsed_time,
            'sample_rate_actual': sample_rate_actual,
            'sample_rate_configured': self.config.sample_rate,
            'queue_size': self.data_queue.qsize(),
            'queue_max_size': self.data_queue.maxsize
        }
    
    def update_config(self, **kwargs):
        """Update streaming configuration"""
        was_streaming = self.is_streaming
        
        if was_streaming:
            self.stop_streaming()
        
        # Update configuration
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
        
        # Reinitialize SDR with new config
        if self.sdr_device:
            try:
                if hasattr(self.sdr_device, 'sample_rate'):
                    self.sdr_device.sample_rate = self.config.sample_rate
                if hasattr(self.sdr_device, 'center_freq'):
                    self.sdr_device.center_freq = self.config.center_frequency
                if hasattr(self.sdr_device, 'gain'):
                    self.sdr_device.gain = self.config.gain
            except Exception as e:
                self.logger.error(f"Failed to update SDR config: {e}")
        
        if was_streaming:
            self.start_streaming()
    
    def __enter__(self):
        """Context manager entry"""
        self.start_streaming()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop_streaming()
        
        # Close SDR device
        if self.sdr_device:
            try:
                if hasattr(self.sdr_device, 'close'):
                    self.sdr_device.close()
            except Exception as e:
                self.logger.error(f"Error closing SDR device: {e}")


class StreamingDataBuffer:
    """Buffer for managing streaming data with windowing"""
    
    def __init__(self, window_size: int = 1024, overlap: float = 0.5):
        self.window_size = window_size
        self.overlap = overlap
        self.step_size = int(window_size * (1 - overlap))
        self.buffer = np.array([], dtype=complex)
        self.sample_count = 0
        
    def add_data(self, iq_data: np.ndarray):
        """Add new IQ data to buffer"""
        self.buffer = np.concatenate([self.buffer, iq_data])
        self.sample_count += len(iq_data)
    
    def get_windows(self) -> List[np.ndarray]:
        """Get windowed data segments"""
        windows = []
        
        while len(self.buffer) >= self.window_size:
            window = self.buffer[:self.window_size].copy()
            windows.append(window)
            
            # Advance buffer by step size
            self.buffer = self.buffer[self.step_size:]
        
        return windows
    
    def clear(self):
        """Clear buffer"""
        self.buffer = np.array([], dtype=complex)
        self.sample_count = 0
    
    def get_buffer_info(self) -> Dict[str, Any]:
        """Get buffer information"""
        return {
            'buffer_length': len(self.buffer),
            'window_size': self.window_size,
            'overlap': self.overlap,
            'step_size': self.step_size,
            'total_samples': self.sample_count,
            'available_windows': len(self.buffer) // self.step_size
        }


class RealTimeProcessor:
    """Real-time signal processor for streaming data"""
    
    def __init__(self, streaming_interface: StreamingDataInterface):
        self.streaming_interface = streaming_interface
        self.buffer = StreamingDataBuffer()
        self.processing_callbacks = []
        self.is_processing = False
        self.processing_thread = None
        
        # Register with streaming interface
        self.streaming_interface.register_callback(self._on_new_data)
    
    def _on_new_data(self, sample: SignalSample):
        """Handle new streaming data"""
        self.buffer.add_data(sample.iq_data)
    
    def start_processing(self):
        """Start real-time processing"""
        if self.is_processing:
            return
        
        self.is_processing = True
        self.processing_thread = threading.Thread(target=self._processing_worker)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def stop_processing(self):
        """Stop real-time processing"""
        self.is_processing = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2.0)
    
    def _processing_worker(self):
        """Worker thread for real-time processing"""
        while self.is_processing:
            try:
                windows = self.buffer.get_windows()
                
                for window in windows:
                    # Process each window
                    for callback in self.processing_callbacks:
                        try:
                            callback(window)
                        except Exception as e:
                            logging.error(f"Processing callback error: {e}")
                
                if not windows:
                    time.sleep(0.01)  # Brief pause if no data
                    
            except Exception as e:
                logging.error(f"Processing worker error: {e}")
                time.sleep(0.1)
    
    def register_processing_callback(self, callback: Callable[[np.ndarray], None]):
        """Register callback for processing windowed data"""
        self.processing_callbacks.append(callback)
    
    def unregister_processing_callback(self, callback: Callable[[np.ndarray], None]):
        """Unregister processing callback"""
        if callback in self.processing_callbacks:
            self.processing_callbacks.remove(callback)