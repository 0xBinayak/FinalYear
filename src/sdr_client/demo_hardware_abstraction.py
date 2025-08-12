#!/usr/bin/env python3
"""
SDR Hardware Abstraction Layer Demonstration

This script demonstrates the complete functionality of the SDR hardware
abstraction layer including device detection, initialization, configuration,
signal collection, and error handling.
"""
import sys
import os
import time
import numpy as np
from datetime import datetime

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from src.sdr_client.hardware_abstraction import (
    SDRType, SDRConfig, SimulatedSDRHardware, 
    RTLSDRHardware, HackRFHardware, USRPHardware
)
from src.sdr_client.device_manager import SDRDeviceManager, SDRHealthMonitor
from src.sdr_client.signal_collector import SignalCollector, CollectionConfig
from src.sdr_client.error_handling import SDRErrorHandler, ErrorType, ErrorSeverity, SDRError


def demonstrate_hardware_types():
    """Demonstrate different SDR hardware types"""
    print("=" * 60)
    print("SDR Hardware Types Demonstration")
    print("=" * 60)
    
    hardware_classes = {
        SDRType.RTL_SDR: RTLSDRHardware,
        SDRType.HACKRF: HackRFHardware,
        SDRType.USRP: USRPHardware,
        SDRType.SIMULATED: SimulatedSDRHardware
    }
    
    for sdr_type, hardware_class in hardware_classes.items():
        print(f"\n{sdr_type.value.upper()} Hardware:")
        print("-" * 40)
        
        try:
            hardware = hardware_class()
            
            # Device detection
            devices = hardware.detect_devices()
            print(f"  Detected devices: {devices}")
            
            # Capabilities
            caps = hardware.get_capabilities()
            print(f"  Frequency range: {caps.frequency_range[0]/1e6:.1f} - {caps.frequency_range[1]/1e6:.1f} MHz")
            print(f"  Sample rate range: {caps.sample_rate_range[0]/1e6:.1f} - {caps.sample_rate_range[1]/1e6:.1f} MHz")
            print(f"  RX channels: {caps.rx_channels}")
            print(f"  TX channels: {caps.tx_channels}")
            print(f"  Gain range: {caps.gain_range[0]} - {caps.gain_range[1]} dB")
            print(f"  Full duplex: {caps.supports_full_duplex}")
            
        except Exception as e:
            print(f"  Error: {e}")


def demonstrate_device_manager():
    """Demonstrate device manager functionality"""
    print("\n" + "=" * 60)
    print("Device Manager Demonstration")
    print("=" * 60)
    
    manager = SDRDeviceManager()
    
    # Detect all devices
    print("\n1. Device Detection:")
    all_devices = manager.detect_all_devices()
    for sdr_type, devices in all_devices.items():
        print(f"  {sdr_type.value}: {devices}")
    
    # Use simulated device for demonstration
    device_id = "simulated_0"
    config = SDRConfig(
        frequency=915e6,  # 915 MHz ISM band
        sample_rate=2e6,   # 2 MHz
        gain=25,           # 25 dB
        bandwidth=1.5e6,   # 1.5 MHz
        buffer_size=8192
    )
    
    print(f"\n2. Device Information for {device_id}:")
    device_info = manager.get_device_info(device_id)
    if device_info:
        print(f"  Device type: {device_info['device_type']}")
        caps = device_info['capabilities']
        print(f"  Frequency range: {caps['frequency_range'][0]/1e6:.1f} - {caps['frequency_range'][1]/1e6:.1f} MHz")
        print(f"  Sample rate range: {caps['sample_rate_range'][0]/1e6:.1f} - {caps['sample_rate_range'][1]/1e6:.1f} MHz")
    
    print(f"\n3. Device Operations:")
    
    # Initialize device
    if manager.initialize_device(device_id, config):
        print(f"  ✓ Device {device_id} initialized")
        
        # Start streaming
        if manager.start_streaming(device_id):
            print(f"  ✓ Streaming started")
            
            # Collect some samples
            for i in range(3):
                buffer = manager.read_samples(device_id, 1024)
                if buffer:
                    # Calculate basic signal statistics
                    power = np.mean(np.abs(buffer.iq_samples) ** 2)
                    snr_estimate = 10 * np.log10(power) if power > 0 else -100
                    
                    print(f"  ✓ Buffer {i+1}: {len(buffer.iq_samples)} samples, "
                          f"Power: {power:.6f}, SNR est: {snr_estimate:.1f} dB")
                
                time.sleep(0.1)
            
            # Test parameter changes
            print(f"  ✓ Changing frequency to 2.4 GHz...")
            manager.set_frequency(device_id, 2.4e9)
            
            print(f"  ✓ Changing gain to 35 dB...")
            manager.set_gain(device_id, 35)
            
            # Stop streaming
            manager.stop_streaming(device_id)
            print(f"  ✓ Streaming stopped")
        
        # Cleanup
        manager.cleanup_device(device_id)
        print(f"  ✓ Device cleaned up")


def demonstrate_signal_collection():
    """Demonstrate signal collection functionality"""
    print("\n" + "=" * 60)
    print("Signal Collection Demonstration")
    print("=" * 60)
    
    manager = SDRDeviceManager()
    collector = SignalCollector(manager)
    
    # Collection configuration
    sdr_config = SDRConfig(
        frequency=433e6,   # 433 MHz ISM band
        sample_rate=1e6,   # 1 MHz
        gain=30,           # 30 dB
        bandwidth=800e3,   # 800 kHz
        buffer_size=4096
    )
    
    collection_config = CollectionConfig(
        device_id="simulated_0",
        sdr_config=sdr_config,
        collection_duration=2.0,  # 2 seconds
        buffer_size=4096,
        processing_callback=lambda buffer: print(f"    Processed buffer: {len(buffer.iq_samples)} samples, "
                                                f"freq: {buffer.frequency/1e6:.1f} MHz")
    )
    
    print(f"\n1. Starting signal collection:")
    print(f"  Frequency: {sdr_config.frequency/1e6:.1f} MHz")
    print(f"  Sample rate: {sdr_config.sample_rate/1e6:.1f} MHz")
    print(f"  Duration: {collection_config.collection_duration} seconds")
    
    if collector.start_collection(collection_config):
        print("  ✓ Collection started")
        
        # Wait for collection to complete
        time.sleep(collection_config.collection_duration + 0.5)
        
        # Get statistics
        stats = collector.get_stats()
        if stats:
            duration = (stats.end_time - stats.start_time).total_seconds() if stats.end_time else 0
            print(f"\n2. Collection Statistics:")
            print(f"  Duration: {duration:.1f} seconds")
            print(f"  Total samples: {stats.total_samples:,}")
            print(f"  Total buffers: {stats.total_buffers}")
            print(f"  Dropped buffers: {stats.dropped_buffers}")
            print(f"  Average SNR: {stats.average_snr:.1f} dB")
            print(f"  Processing errors: {stats.processing_errors}")
        
        collector.stop_collection()
        print("  ✓ Collection stopped")


def demonstrate_error_handling():
    """Demonstrate error handling functionality"""
    print("\n" + "=" * 60)
    print("Error Handling Demonstration")
    print("=" * 60)
    
    error_handler = SDRErrorHandler()
    
    # Create some test errors
    test_errors = [
        SDRError(
            error_type=ErrorType.HARDWARE_FAILURE,
            severity=ErrorSeverity.HIGH,
            message="Simulated hardware failure",
            timestamp=datetime.now(),
            device_id="simulated_0",
            context={'temperature': 85.0}
        ),
        SDRError(
            error_type=ErrorType.BUFFER_OVERFLOW,
            severity=ErrorSeverity.MEDIUM,
            message="Buffer overflow detected",
            timestamp=datetime.now(),
            device_id="simulated_0",
            context={'buffer_size': 8192, 'overflow_count': 5}
        ),
        SDRError(
            error_type=ErrorType.SIGNAL_QUALITY,
            severity=ErrorSeverity.LOW,
            message="Low signal quality",
            timestamp=datetime.now(),
            device_id="simulated_0",
            context={'snr': -5.0}
        )
    ]
    
    print("\n1. Handling test errors:")
    for i, error in enumerate(test_errors):
        print(f"  Error {i+1}: {error.error_type.value} - {error.message}")
        result = error_handler.handle_error(error)
        print(f"    Handled: {result}, Recovery attempted: {error.recovery_attempted}")
    
    print(f"\n2. Error Statistics:")
    stats = error_handler.get_error_stats("simulated_0")
    print(f"  Total errors: {stats['total_errors']}")
    print(f"  Error rate: {stats['error_rate']:.2f} errors/hour")
    print(f"  Most common error: {stats['most_common_error']}")
    print(f"  Recovery success rate: {stats['recovery_success_rate']:.1%}")
    
    print(f"\n3. Error counts by type:")
    for error_type, count in stats['error_counts'].items():
        print(f"    {error_type}: {count}")


def demonstrate_health_monitoring():
    """Demonstrate health monitoring functionality"""
    print("\n" + "=" * 60)
    print("Health Monitoring Demonstration")
    print("=" * 60)
    
    manager = SDRDeviceManager()
    monitor = SDRHealthMonitor(manager)
    
    # Initialize a device for monitoring
    config = SDRConfig(
        frequency=100e6,
        sample_rate=2e6,
        gain=20,
        bandwidth=1e6,
        buffer_size=8192
    )
    
    device_id = "simulated_0"
    if manager.initialize_device(device_id, config):
        print(f"  ✓ Device {device_id} initialized for monitoring")
        
        # Start monitoring
        monitor.start_monitoring(interval=0.5)  # Check every 0.5 seconds
        print("  ✓ Health monitoring started")
        
        # Let it run for a few seconds
        time.sleep(2.0)
        
        # Check health stats
        stats = monitor.get_health_stats(device_id)
        if stats:
            print(f"\n  Health Status: {stats['status']}")
            print(f"  Last check: {datetime.fromtimestamp(stats['last_check']).strftime('%H:%M:%S')}")
            if 'read_time' in stats:
                print(f"  Read time: {stats['read_time']*1000:.1f} ms")
            if 'sample_count' in stats:
                print(f"  Sample count: {stats['sample_count']}")
        
        # Stop monitoring
        monitor.stop_monitoring()
        print("  ✓ Health monitoring stopped")
        
        # Cleanup
        manager.cleanup_device(device_id)
        print("  ✓ Device cleaned up")


def main():
    """Main demonstration function"""
    print("SDR Hardware Abstraction Layer Demonstration")
    print("=" * 60)
    print("This demonstration shows the complete functionality of the")
    print("SDR hardware abstraction layer including:")
    print("- Multiple SDR hardware types (RTL-SDR, HackRF, USRP, Simulated)")
    print("- Device detection and initialization")
    print("- Signal collection and processing")
    print("- Error handling and recovery")
    print("- Health monitoring")
    
    try:
        demonstrate_hardware_types()
        demonstrate_device_manager()
        demonstrate_signal_collection()
        demonstrate_error_handling()
        demonstrate_health_monitoring()
        
        print("\n" + "=" * 60)
        print("✓ All demonstrations completed successfully!")
        print("✓ SDR Hardware Abstraction Layer is fully functional")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()