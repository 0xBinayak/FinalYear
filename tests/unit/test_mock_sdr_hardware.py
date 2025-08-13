"""
Comprehensive tests for mock SDR hardware for development testing.
"""
import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import asyncio
from datetime import datetime

from src.sdr_client.hardware_abstraction import MockSDRDevice, SDRHardwareManager
from src.sdr_client.signal_processing import SignalProcessor
from src.common.interfaces import SignalSample


@pytest.mark.unit
@pytest.mark.mock_sdr
class TestMockSDRDevice:
    """Test cases for MockSDRDevice class."""
    
    def test_mock_device_initialization(self):
        """Test mock SDR device initialization."""
        mock_device = MockSDRDevice(device_type="rtlsdr", device_index=0)
        
        assert mock_device.device_type == "rtlsdr"
        assert mock_device.device_index == 0
        assert mock_device.is_connected is False
        assert mock_device.sample_rate == 2.048e6  # Default RTL-SDR rate
        assert mock_device.center_freq == 100e6    # Default frequency
    
    def test_device_connection(self):
        """Test mock device connection and disconnection."""
        mock_device = MockSDRDevice("hackrf", 0)
        
        # Test connection
        success = mock_device.connect()
        assert success
        assert mock_device.is_connected
        
        # Test double connection (should be idempotent)
        success = mock_device.connect()
        assert success
        assert mock_device.is_connected
        
        # Test disconnection
        mock_device.disconnect()
        assert not mock_device.is_connected
    
    def test_parameter_configuration(self):
        """Test SDR parameter configuration."""
        mock_device = MockSDRDevice("usrp", 0)
        mock_device.connect()
        
        # Test sample rate configuration
        mock_device.set_sample_rate(10e6)
        assert mock_device.sample_rate == 10e6
        
        # Test center frequency configuration
        mock_device.set_center_freq(915e6)
        assert mock_device.center_freq == 915e6
        
        # Test gain configuration
        mock_device.set_gain(20)
        assert mock_device.gain == 20
        
        # Test invalid parameters
        with pytest.raises(ValueError):
            mock_device.set_sample_rate(-1000)  # Negative sample rate
        
        with pytest.raises(ValueError):
            mock_device.set_center_freq(0)  # Zero frequency
    
    def test_signal_generation(self):
        """Test mock signal generation."""
        mock_device = MockSDRDevice("rtlsdr", 0)
        mock_device.connect()
        
        # Configure device
        mock_device.set_sample_rate(2e6)
        mock_device.set_center_freq(915e6)
        
        # Generate signal samples
        num_samples = 1024
        samples = mock_device.read_samples(num_samples)
        
        assert samples is not None
        assert len(samples) == num_samples
        assert samples.dtype == np.complex64
        
        # Verify signal characteristics
        assert np.all(np.isfinite(samples))  # No NaN or inf values
        assert np.std(samples) > 0  # Some variation in signal
    
    def test_modulation_simulation(self):
        """Test different modulation type simulation."""
        mock_device = MockSDRDevice("hackrf", 0)
        mock_device.connect()
        
        modulation_types = ["BPSK", "QPSK", "8PSK", "QAM16", "QAM64", "FSK", "ASK"]
        
        for mod_type in modulation_types:
            mock_device.set_modulation_type(mod_type)
            samples = mock_device.read_samples(512)
            
            assert samples is not None
            assert len(samples) == 512
            
            # Verify modulation-specific characteristics
            if mod_type in ["BPSK", "QPSK", "8PSK"]:
                # PSK signals should have constant amplitude
                amplitudes = np.abs(samples)
                amplitude_variation = np.std(amplitudes) / np.mean(amplitudes)
                assert amplitude_variation < 0.2  # Low amplitude variation
            
            elif mod_type in ["QAM16", "QAM64"]:
                # QAM signals should have multiple amplitude levels
                amplitudes = np.abs(samples)
                unique_levels = len(np.unique(np.round(amplitudes, 1)))
                assert unique_levels > 2  # Multiple amplitude levels
    
    def test_noise_and_channel_effects(self):
        """Test noise and channel effect simulation."""
        mock_device = MockSDRDevice("usrp", 0)
        mock_device.connect()
        
        # Test different SNR levels
        snr_levels = [0, 10, 20, 30]
        
        for snr in snr_levels:
            mock_device.set_snr(snr)
            samples = mock_device.read_samples(1000)
            
            # Higher SNR should result in cleaner signals
            signal_power = np.mean(np.abs(samples) ** 2)
            assert signal_power > 0
            
            # Verify SNR is approximately correct
            if snr > 0:
                estimated_snr = mock_device.estimate_snr(samples)
                assert abs(estimated_snr - snr) < 5  # Within 5 dB tolerance
    
    def test_frequency_offset_simulation(self):
        """Test frequency offset simulation."""
        mock_device = MockSDRDevice("rtlsdr", 0)
        mock_device.connect()
        
        # Set frequency offset
        offset_hz = 1000  # 1 kHz offset
        mock_device.set_frequency_offset(offset_hz)
        
        samples = mock_device.read_samples(2048)
        
        # Verify frequency offset is present
        fft = np.fft.fft(samples)
        freqs = np.fft.fftfreq(len(samples), 1/mock_device.sample_rate)
        
        # Find peak frequency
        peak_idx = np.argmax(np.abs(fft))
        peak_freq = freqs[peak_idx]
        
        # Should be close to the offset frequency
        assert abs(peak_freq - offset_hz) < 100  # Within 100 Hz
    
    def test_multipath_fading_simulation(self):
        """Test multipath fading simulation."""
        mock_device = MockSDRDevice("hackrf", 0)
        mock_device.connect()
        
        # Enable multipath fading
        mock_device.enable_multipath_fading(
            delays=[0, 5, 10],  # Delay taps in samples
            gains=[1.0, 0.5, 0.3],  # Relative gains
            doppler_freq=10  # 10 Hz Doppler
        )
        
        samples = mock_device.read_samples(1024)
        
        # Multipath should cause amplitude variations
        amplitudes = np.abs(samples)
        amplitude_variation = np.std(amplitudes) / np.mean(amplitudes)
        assert amplitude_variation > 0.1  # Significant amplitude variation
        
        # Should have frequency-selective fading characteristics
        fft = np.fft.fft(samples)
        power_spectrum = np.abs(fft) ** 2
        spectral_variation = np.std(power_spectrum) / np.mean(power_spectrum)
        assert spectral_variation > 0.2  # Frequency-selective characteristics
    
    def test_hardware_impairments(self):
        """Test hardware impairment simulation."""
        mock_device = MockSDRDevice("rtlsdr", 0)
        mock_device.connect()
        
        # Enable various impairments
        mock_device.set_iq_imbalance(amplitude_imbalance=0.1, phase_imbalance=5.0)
        mock_device.set_dc_offset(i_offset=0.05, q_offset=0.03)
        mock_device.set_phase_noise(phase_noise_std=0.1)
        
        samples = mock_device.read_samples(1000)
        
        # Verify impairments are present
        i_samples = np.real(samples)
        q_samples = np.imag(samples)
        
        # DC offset should shift mean
        i_mean = np.mean(i_samples)
        q_mean = np.mean(q_samples)
        assert abs(i_mean - 0.05) < 0.02
        assert abs(q_mean - 0.03) < 0.02
        
        # IQ imbalance should affect amplitude ratio
        i_std = np.std(i_samples)
        q_std = np.std(q_samples)
        imbalance_ratio = abs(i_std / q_std - 1.1)  # Expected 10% imbalance
        assert imbalance_ratio < 0.05
    
    def test_streaming_mode(self):
        """Test streaming mode operation."""
        mock_device = MockSDRDevice("usrp", 0)
        mock_device.connect()
        
        # Start streaming
        mock_device.start_streaming(buffer_size=1024)
        assert mock_device.is_streaming
        
        # Read multiple buffers
        buffers = []
        for _ in range(5):
            buffer = mock_device.read_stream_buffer()
            assert buffer is not None
            assert len(buffer) == 1024
            buffers.append(buffer)
        
        # Stop streaming
        mock_device.stop_streaming()
        assert not mock_device.is_streaming
        
        # Verify buffers are different (not just repeated)
        for i in range(1, len(buffers)):
            assert not np.array_equal(buffers[0], buffers[i])
    
    async def test_async_operations(self):
        """Test asynchronous operations."""
        mock_device = MockSDRDevice("hackrf", 0)
        mock_device.connect()
        
        # Test async sample reading
        samples = await mock_device.read_samples_async(512)
        assert samples is not None
        assert len(samples) == 512
        
        # Test async streaming
        await mock_device.start_streaming_async(buffer_size=256)
        
        stream_buffers = []
        for _ in range(3):
            buffer = await mock_device.read_stream_buffer_async()
            stream_buffers.append(buffer)
        
        await mock_device.stop_streaming_async()
        
        # Verify async streaming worked
        assert len(stream_buffers) == 3
        assert all(len(buf) == 256 for buf in stream_buffers)


@pytest.mark.unit
@pytest.mark.mock_sdr
class TestSDRHardwareManager:
    """Test cases for SDRHardwareManager with mock devices."""
    
    def test_mock_device_detection(self):
        """Test mock device detection and enumeration."""
        manager = SDRHardwareManager(use_mock_devices=True)
        
        # Should detect mock devices
        devices = manager.detect_devices()
        assert len(devices) > 0
        
        # Should have different device types
        device_types = [dev["type"] for dev in devices]
        assert "rtlsdr" in device_types
        assert "hackrf" in device_types
        assert "usrp" in device_types
    
    def test_device_factory(self):
        """Test device factory for creating mock devices."""
        manager = SDRHardwareManager(use_mock_devices=True)
        
        # Create different types of mock devices
        rtl_device = manager.create_device("rtlsdr", 0)
        assert isinstance(rtl_device, MockSDRDevice)
        assert rtl_device.device_type == "rtlsdr"
        
        hackrf_device = manager.create_device("hackrf", 0)
        assert isinstance(hackrf_device, MockSDRDevice)
        assert hackrf_device.device_type == "hackrf"
        
        usrp_device = manager.create_device("usrp", 0)
        assert isinstance(usrp_device, MockSDRDevice)
        assert usrp_device.device_type == "usrp"
    
    def test_device_configuration_profiles(self):
        """Test device configuration profiles."""
        manager = SDRHardwareManager(use_mock_devices=True)
        
        # Test predefined configuration profiles
        profiles = {
            "gsm_900": {
                "center_freq": 900e6,
                "sample_rate": 2e6,
                "gain": 20,
                "modulation": "GMSK"
            },
            "wifi_2_4": {
                "center_freq": 2.4e9,
                "sample_rate": 20e6,
                "gain": 15,
                "modulation": "OFDM"
            },
            "lte_band_7": {
                "center_freq": 2.6e9,
                "sample_rate": 20e6,
                "gain": 25,
                "modulation": "OFDM"
            }
        }
        
        device = manager.create_device("usrp", 0)
        device.connect()
        
        for profile_name, config in profiles.items():
            manager.apply_configuration_profile(device, profile_name, config)
            
            assert device.center_freq == config["center_freq"]
            assert device.sample_rate == config["sample_rate"]
            assert device.gain == config["gain"]
    
    def test_device_health_monitoring(self):
        """Test device health monitoring."""
        manager = SDRHardwareManager(use_mock_devices=True)
        device = manager.create_device("rtlsdr", 0)
        device.connect()
        
        # Get device health status
        health_status = manager.get_device_health(device)
        
        assert "temperature" in health_status
        assert "power_consumption" in health_status
        assert "signal_quality" in health_status
        assert "error_rate" in health_status
        
        # All health metrics should be within reasonable ranges
        assert 20 <= health_status["temperature"] <= 80  # Celsius
        assert 0 <= health_status["power_consumption"] <= 10  # Watts
        assert 0 <= health_status["signal_quality"] <= 1.0
        assert 0 <= health_status["error_rate"] <= 0.1
    
    def test_concurrent_device_access(self):
        """Test concurrent access to multiple mock devices."""
        manager = SDRHardwareManager(use_mock_devices=True)
        
        # Create multiple devices
        devices = []
        for i in range(3):
            device = manager.create_device("rtlsdr", i)
            device.connect()
            devices.append(device)
        
        # Configure devices with different parameters
        for i, device in enumerate(devices):
            device.set_center_freq(900e6 + i * 100e6)
            device.set_sample_rate(2e6)
            device.set_gain(10 + i * 5)
        
        # Read samples from all devices simultaneously
        all_samples = []
        for device in devices:
            samples = device.read_samples(512)
            all_samples.append(samples)
        
        # Verify all devices produced samples
        assert len(all_samples) == 3
        assert all(len(samples) == 512 for samples in all_samples)
        
        # Samples should be different due to different configurations
        for i in range(1, len(all_samples)):
            assert not np.array_equal(all_samples[0], all_samples[i])
    
    def test_error_simulation(self):
        """Test error condition simulation."""
        manager = SDRHardwareManager(use_mock_devices=True)
        device = manager.create_device("hackrf", 0)
        
        # Test connection failure simulation
        device.simulate_connection_failure(True)
        success = device.connect()
        assert not success
        assert not device.is_connected
        
        # Disable failure simulation
        device.simulate_connection_failure(False)
        success = device.connect()
        assert success
        assert device.is_connected
        
        # Test sample read errors
        device.simulate_read_errors(error_probability=0.5)
        
        # Some reads should fail
        failed_reads = 0
        total_reads = 10
        
        for _ in range(total_reads):
            try:
                samples = device.read_samples(256)
                if samples is None:
                    failed_reads += 1
            except Exception:
                failed_reads += 1
        
        # Should have some failures but not all
        assert 0 < failed_reads < total_reads
    
    def test_calibration_simulation(self):
        """Test device calibration simulation."""
        manager = SDRHardwareManager(use_mock_devices=True)
        device = manager.create_device("usrp", 0)
        device.connect()
        
        # Perform calibration
        calibration_result = manager.calibrate_device(device)
        
        assert calibration_result["success"] is True
        assert "frequency_correction" in calibration_result
        assert "gain_correction" in calibration_result
        assert "phase_correction" in calibration_result
        
        # Calibration should improve signal quality
        pre_cal_quality = device.get_signal_quality()
        manager.apply_calibration(device, calibration_result)
        post_cal_quality = device.get_signal_quality()
        
        assert post_cal_quality >= pre_cal_quality


@pytest.mark.unit
@pytest.mark.mock_sdr
class TestSignalProcessorWithMockHardware:
    """Test signal processor with mock SDR hardware."""
    
    def test_real_time_processing_simulation(self):
        """Test real-time signal processing with mock hardware."""
        processor = SignalProcessor()
        mock_device = MockSDRDevice("rtlsdr", 0)
        mock_device.connect()
        
        # Configure for real-time processing
        mock_device.set_sample_rate(2e6)
        mock_device.set_center_freq(915e6)
        mock_device.set_modulation_type("QPSK")
        
        # Start streaming
        mock_device.start_streaming(buffer_size=1024)
        
        # Process multiple buffers
        processed_results = []
        for _ in range(5):
            buffer = mock_device.read_stream_buffer()
            
            # Process buffer
            features = processor.extract_features(buffer, mock_device.sample_rate)
            classification = processor.classify_modulation(buffer)
            
            processed_results.append({
                "features": features,
                "classification": classification,
                "timestamp": datetime.now()
            })
        
        mock_device.stop_streaming()
        
        # Verify processing results
        assert len(processed_results) == 5
        for result in processed_results:
            assert "spectral_features" in result["features"]
            assert "time_domain_features" in result["features"]
            assert "modulation_type" in result["classification"]
            assert result["classification"]["modulation_type"] == "QPSK"
    
    def test_adaptive_processing(self):
        """Test adaptive processing based on signal conditions."""
        processor = SignalProcessor()
        mock_device = MockSDRDevice("hackrf", 0)
        mock_device.connect()
        
        # Test processing under different SNR conditions
        snr_levels = [5, 15, 25]
        
        for snr in snr_levels:
            mock_device.set_snr(snr)
            samples = mock_device.read_samples(2048)
            
            # Adaptive processing should adjust based on SNR
            processed_samples = processor.adaptive_process(samples, estimated_snr=snr)
            
            # Lower SNR should result in more aggressive processing
            if snr < 10:
                # Should apply more filtering/smoothing
                assert len(processed_samples) <= len(samples)
            else:
                # Should preserve more of the original signal
                assert len(processed_samples) == len(samples)
    
    def test_multi_signal_processing(self):
        """Test processing of multiple simultaneous signals."""
        processor = SignalProcessor()
        mock_device = MockSDRDevice("usrp", 0)
        mock_device.connect()
        
        # Configure for wideband reception
        mock_device.set_sample_rate(20e6)  # 20 MHz bandwidth
        mock_device.set_center_freq(2.4e9)  # 2.4 GHz center
        
        # Enable multiple signal simulation
        mock_device.enable_multi_signal_mode([
            {"freq_offset": -5e6, "modulation": "QPSK", "power": 0.8},
            {"freq_offset": 0, "modulation": "QAM16", "power": 1.0},
            {"freq_offset": 3e6, "modulation": "FSK", "power": 0.6}
        ])
        
        samples = mock_device.read_samples(4096)
        
        # Detect and separate signals
        detected_signals = processor.detect_multiple_signals(
            samples, 
            sample_rate=mock_device.sample_rate
        )
        
        assert len(detected_signals) >= 2  # Should detect at least 2 signals
        
        # Verify signal characteristics
        for signal in detected_signals:
            assert "center_frequency" in signal
            assert "bandwidth" in signal
            assert "modulation_type" in signal
            assert "power_level" in signal
    
    def test_interference_mitigation(self):
        """Test interference mitigation techniques."""
        processor = SignalProcessor()
        mock_device = MockSDRDevice("rtlsdr", 0)
        mock_device.connect()
        
        # Add interference
        mock_device.add_interference(
            interference_type="narrowband",
            frequency_offset=100e3,  # 100 kHz offset
            power_ratio=0.5  # 50% of signal power
        )
        
        samples = mock_device.read_samples(2048)
        
        # Apply interference mitigation
        clean_samples = processor.mitigate_interference(
            samples,
            interference_type="narrowband",
            sample_rate=mock_device.sample_rate
        )
        
        # Verify interference reduction
        original_spectrum = np.abs(np.fft.fft(samples))
        clean_spectrum = np.abs(np.fft.fft(clean_samples))
        
        # Should have reduced power at interference frequency
        freqs = np.fft.fftfreq(len(samples), 1/mock_device.sample_rate)
        interference_bin = np.argmin(np.abs(freqs - 100e3))
        
        interference_reduction = (
            original_spectrum[interference_bin] / clean_spectrum[interference_bin]
        )
        assert interference_reduction > 1.5  # At least 1.5x reduction
    
    async def test_continuous_monitoring(self):
        """Test continuous signal monitoring."""
        processor = SignalProcessor()
        mock_device = MockSDRDevice("usrp", 0)
        mock_device.connect()
        
        # Start continuous monitoring
        monitoring_results = []
        
        async def monitor_callback(signal_data):
            analysis = processor.analyze_signal_quality(signal_data)
            monitoring_results.append(analysis)
        
        # Simulate monitoring for a short period
        await mock_device.start_continuous_monitoring(
            callback=monitor_callback,
            monitoring_duration=1.0,  # 1 second
            analysis_interval=0.2     # Every 200ms
        )
        
        # Should have collected multiple analysis results
        assert len(monitoring_results) >= 4  # At least 4 analyses in 1 second
        
        # Verify analysis results
        for result in monitoring_results:
            assert "snr_estimate" in result
            assert "signal_power" in result
            assert "noise_floor" in result
            assert "spectral_occupancy" in result