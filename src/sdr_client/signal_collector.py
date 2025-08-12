"""
Signal Collection and Real-time Processing

Provides high-level interface for signal collection with real-time processing capabilities.
Handles buffering, streaming, and basic signal processing operations.
"""
import numpy as np
import threading
import time
import queue
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging

from .device_manager import SDRDeviceManager
from .hardware_abstraction import SDRConfig, SignalBuffer

logger = logging.getLogger(__name__)


@dataclass
class CollectionConfig:
    """Configuration for signal collection"""
    device_id: str
    sdr_config: SDRConfig
    collection_duration: Optional[float] = None  # seconds, None for continuous
    buffer_size: int = 8192
    overlap_samples: int = 0
    processing_callback: Optional[Callable[[SignalBuffer], None]] = None
    save_to_file: bool = False
    output_file: Optional[str] = None


@dataclass
class CollectionStats:
    """Statistics for signal collection session"""
    start_time: datetime
    end_time: Optional[datetime]
    total_samples: int
    total_buffers: int
    dropped_buffers: int
    average_snr: float
    frequency_drift: float
    processing_errors: int


class SignalCollector:
    """High-level signal collection interface"""
    
    def __init__(self, device_manager: SDRDeviceManager):
        self.device_manager = device_manager
        self.collecting = False
        self.collection_thread = None
        self.buffer_queue = queue.Queue(maxsize=100)
        self.processing_thread = None
        self.stats = None
        
    def start_collection(self, config: CollectionConfig) -> bool:
        """Start signal collection"""
        if self.collecting:
            logger.warning("Collection already in progress")
            return False
        
        try:
            # Initialize device if not already active
            if not self.device_manager.is_device_active(config.device_id):
                if not self.device_manager.initialize_device(config.device_id, config.sdr_config):
                    logger.error(f"Failed to initialize device {config.device_id}")
                    return False
            
            # Start streaming
            if not self.device_manager.start_streaming(config.device_id):
                logger.error(f"Failed to start streaming on {config.device_id}")
                return False
            
            # Initialize statistics
            self.stats = CollectionStats(
                start_time=datetime.now(),
                end_time=None,
                total_samples=0,
                total_buffers=0,
                dropped_buffers=0,
                average_snr=0.0,
                frequency_drift=0.0,
                processing_errors=0
            )
            
            # Start collection thread
            self.collecting = True
            self.collection_thread = threading.Thread(
                target=self._collection_loop,
                args=(config,),
                daemon=True
            )
            self.collection_thread.start()
            
            # Start processing thread if callback provided
            if config.processing_callback:
                self.processing_thread = threading.Thread(
                    target=self._processing_loop,
                    args=(config.processing_callback,),
                    daemon=True
                )
                self.processing_thread.start()
            
            logger.info(f"Started signal collection on {config.device_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting collection: {e}")
            self.collecting = False
            return False
    
    def stop_collection(self) -> bool:
        """Stop signal collection"""
        if not self.collecting:
            return True
        
        try:
            # Stop collection
            self.collecting = False
            
            # Wait for threads to finish
            if self.collection_thread:
                self.collection_thread.join(timeout=2.0)
            
            if self.processing_thread:
                self.processing_thread.join(timeout=2.0)
            
            # Update statistics
            if self.stats:
                self.stats.end_time = datetime.now()
            
            logger.info("Stopped signal collection")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping collection: {e}")
            return False
    
    def get_stats(self) -> Optional[CollectionStats]:
        """Get collection statistics"""
        return self.stats
    
    def _collection_loop(self, config: CollectionConfig):
        """Main collection loop"""
        start_time = time.time()
        last_buffer_time = start_time
        
        try:
            while self.collecting:
                # Check duration limit
                if config.collection_duration:
                    if time.time() - start_time >= config.collection_duration:
                        logger.info("Collection duration reached")
                        break
                
                # Read samples
                buffer = self.device_manager.read_samples(
                    config.device_id, 
                    config.buffer_size
                )
                
                if buffer is None:
                    logger.warning("Failed to read samples")
                    time.sleep(0.001)  # Small delay to prevent busy loop
                    continue
                
                # Update statistics
                if self.stats:
                    self.stats.total_samples += len(buffer.iq_samples)
                    self.stats.total_buffers += 1
                    
                    # Calculate SNR estimate
                    signal_power = np.mean(np.abs(buffer.iq_samples) ** 2)
                    if signal_power > 0:
                        snr_estimate = 10 * np.log10(signal_power)
                        self.stats.average_snr = (
                            (self.stats.average_snr * (self.stats.total_buffers - 1) + snr_estimate) /
                            self.stats.total_buffers
                        )
                
                # Add to processing queue
                try:
                    self.buffer_queue.put_nowait(buffer)
                except queue.Full:
                    # Queue is full, drop oldest buffer
                    try:
                        self.buffer_queue.get_nowait()
                        self.buffer_queue.put_nowait(buffer)
                        if self.stats:
                            self.stats.dropped_buffers += 1
                    except queue.Empty:
                        pass
                
                # Control collection rate
                current_time = time.time()
                expected_interval = config.buffer_size / config.sdr_config.sample_rate
                actual_interval = current_time - last_buffer_time
                
                if actual_interval < expected_interval:
                    time.sleep(expected_interval - actual_interval)
                
                last_buffer_time = current_time
                
        except Exception as e:
            logger.error(f"Error in collection loop: {e}")
        finally:
            # Stop streaming
            self.device_manager.stop_streaming(config.device_id)
    
    def _processing_loop(self, callback: Callable[[SignalBuffer], None]):
        """Processing loop for real-time signal processing"""
        try:
            while self.collecting or not self.buffer_queue.empty():
                try:
                    # Get buffer with timeout
                    buffer = self.buffer_queue.get(timeout=1.0)
                    
                    # Process buffer
                    callback(buffer)
                    
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"Error in processing callback: {e}")
                    if self.stats:
                        self.stats.processing_errors += 1
                        
        except Exception as e:
            logger.error(f"Error in processing loop: {e}")


class StreamingCollector:
    """Streaming signal collector with advanced buffering"""
    
    def __init__(self, device_manager: SDRDeviceManager):
        self.device_manager = device_manager
        self.streaming = False
        self.stream_thread = None
        self.circular_buffer = None
        self.buffer_lock = threading.Lock()
        self.write_index = 0
        self.read_index = 0
        
    def start_streaming(self, config: CollectionConfig, buffer_duration: float = 10.0) -> bool:
        """Start streaming with circular buffer"""
        if self.streaming:
            return False
        
        try:
            # Calculate buffer size
            total_samples = int(config.sdr_config.sample_rate * buffer_duration)
            self.circular_buffer = np.zeros(total_samples, dtype=np.complex64)
            
            # Initialize device
            if not self.device_manager.is_device_active(config.device_id):
                if not self.device_manager.initialize_device(config.device_id, config.sdr_config):
                    return False
            
            if not self.device_manager.start_streaming(config.device_id):
                return False
            
            # Start streaming thread
            self.streaming = True
            self.write_index = 0
            self.read_index = 0
            
            self.stream_thread = threading.Thread(
                target=self._streaming_loop,
                args=(config,),
                daemon=True
            )
            self.stream_thread.start()
            
            logger.info(f"Started streaming collection with {buffer_duration}s buffer")
            return True
            
        except Exception as e:
            logger.error(f"Error starting streaming: {e}")
            return False
    
    def stop_streaming(self) -> bool:
        """Stop streaming"""
        if not self.streaming:
            return True
        
        self.streaming = False
        
        if self.stream_thread:
            self.stream_thread.join(timeout=2.0)
        
        return True
    
    def get_recent_samples(self, num_samples: int) -> Optional[np.ndarray]:
        """Get most recent samples from circular buffer"""
        if self.circular_buffer is None:
            return None
        
        with self.buffer_lock:
            buffer_size = len(self.circular_buffer)
            
            if num_samples > buffer_size:
                num_samples = buffer_size
            
            # Calculate read position
            start_idx = (self.write_index - num_samples) % buffer_size
            
            if start_idx + num_samples <= buffer_size:
                # No wrap-around
                return self.circular_buffer[start_idx:start_idx + num_samples].copy()
            else:
                # Handle wrap-around
                first_part = self.circular_buffer[start_idx:].copy()
                second_part = self.circular_buffer[:num_samples - len(first_part)].copy()
                return np.concatenate([first_part, second_part])
    
    def get_samples_at_time(self, timestamp: datetime, duration: float) -> Optional[np.ndarray]:
        """Get samples at specific timestamp (if still in buffer)"""
        # This is a simplified implementation
        # In practice, you'd need to track timestamps for each sample
        num_samples = int(duration * self.device_manager.active_devices[list(self.device_manager.active_devices.keys())[0]].config.sample_rate)
        return self.get_recent_samples(num_samples)
    
    def _streaming_loop(self, config: CollectionConfig):
        """Streaming loop that fills circular buffer"""
        try:
            while self.streaming:
                buffer = self.device_manager.read_samples(
                    config.device_id,
                    config.buffer_size
                )
                
                if buffer is None:
                    time.sleep(0.001)
                    continue
                
                # Write to circular buffer
                with self.buffer_lock:
                    samples = buffer.iq_samples
                    buffer_size = len(self.circular_buffer)
                    
                    for sample in samples:
                        self.circular_buffer[self.write_index] = sample
                        self.write_index = (self.write_index + 1) % buffer_size
                        
        except Exception as e:
            logger.error(f"Error in streaming loop: {e}")
        finally:
            self.device_manager.stop_streaming(config.device_id)


class AdaptiveCollector:
    """Adaptive signal collector that adjusts parameters based on conditions"""
    
    def __init__(self, device_manager: SDRDeviceManager):
        self.device_manager = device_manager
        self.collecting = False
        self.adaptive_thread = None
        self.current_config = None
        
    def start_adaptive_collection(self, base_config: CollectionConfig) -> bool:
        """Start adaptive collection"""
        if self.collecting:
            return False
        
        self.current_config = base_config
        self.collecting = True
        
        self.adaptive_thread = threading.Thread(
            target=self._adaptive_loop,
            daemon=True
        )
        self.adaptive_thread.start()
        
        return True
    
    def stop_adaptive_collection(self) -> bool:
        """Stop adaptive collection"""
        self.collecting = False
        
        if self.adaptive_thread:
            self.adaptive_thread.join(timeout=2.0)
        
        return True
    
    def _adaptive_loop(self):
        """Adaptive collection loop"""
        collector = SignalCollector(self.device_manager)
        adaptation_interval = 5.0  # seconds
        last_adaptation = time.time()
        
        try:
            # Start initial collection
            collector.start_collection(self.current_config)
            
            while self.collecting:
                time.sleep(1.0)
                
                # Check if it's time to adapt
                if time.time() - last_adaptation >= adaptation_interval:
                    self._adapt_parameters(collector)
                    last_adaptation = time.time()
                    
        except Exception as e:
            logger.error(f"Error in adaptive loop: {e}")
        finally:
            collector.stop_collection()
    
    def _adapt_parameters(self, collector: SignalCollector):
        """Adapt collection parameters based on current conditions"""
        try:
            stats = collector.get_stats()
            if not stats:
                return
            
            # Adapt based on SNR
            if stats.average_snr < -10:  # Low SNR
                # Increase gain if possible
                current_gain = self.current_config.sdr_config.gain
                new_gain = min(current_gain + 5, 50)  # Max gain limit
                
                if new_gain != current_gain:
                    self.device_manager.set_gain(
                        self.current_config.device_id,
                        new_gain
                    )
                    self.current_config.sdr_config.gain = new_gain
                    logger.info(f"Adapted gain to {new_gain} dB due to low SNR")
            
            elif stats.average_snr > 20:  # High SNR
                # Decrease gain to prevent saturation
                current_gain = self.current_config.sdr_config.gain
                new_gain = max(current_gain - 5, 0)  # Min gain limit
                
                if new_gain != current_gain:
                    self.device_manager.set_gain(
                        self.current_config.device_id,
                        new_gain
                    )
                    self.current_config.sdr_config.gain = new_gain
                    logger.info(f"Adapted gain to {new_gain} dB due to high SNR")
            
            # Adapt based on dropped buffers
            if stats.dropped_buffers > stats.total_buffers * 0.1:  # >10% dropped
                # Increase buffer size
                current_size = self.current_config.buffer_size
                new_size = min(current_size * 2, 32768)  # Max buffer size
                
                if new_size != current_size:
                    self.current_config.buffer_size = new_size
                    logger.info(f"Adapted buffer size to {new_size} due to dropped buffers")
                    
        except Exception as e:
            logger.error(f"Error adapting parameters: {e}")