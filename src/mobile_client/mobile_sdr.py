"""
Mobile SDR integration for mobile devices with SDR dongles and built-in radio capabilities
"""
import logging
import threading
import time
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple, Callable
import numpy as np
from dataclasses import dataclass
from enum import Enum

from ..common.signal_models import (
    EnhancedSignalSample, ModulationType, HardwareType, 
    HardwareInfo, RFParameters, SignalQualityMetrics, GPSCoordinate
)


class MobileSDRType(Enum):
    """Mobile SDR types"""
    RTL_SDR_MOBILE = "RTL-SDR Mobile"
    HACKRF_MOBILE = "HackRF Mobile"
    AIRSPY_MOBILE = "Airspy Mobile"
    BUILT_IN_RADIO = "Built-in Radio"
    SIMULATED = "Simulated"


@dataclass
class MobileSDRCapabilities:
    """Mobile SDR device capabilities"""
    sdr_type: MobileSDRType
    frequency_range: Tuple[float, float]  # Hz
    sample_rate_range: Tuple[float, float]  # Hz
    gain_range: Tuple[float, float]  # dB
    supports_tx: bool = False
    supports_full_duplex: bool = False
    power_consumption_mw: float = 500.0  # Estimated power consumption
    usb_powered: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'sdr_type': self.sdr_type.value,
            'frequency_range': list(self.frequency_range),
            'sample_rate_range': list(self.sample_rate_range),
            'gain_range': list(self.gain_range),
            'supports_tx': self.supports_tx,
            'supports_full_duplex': self.supports_full_duplex,
            'power_consumption_mw': self.power_consumption_mw,
            'usb_powered': self.usb_powered
        }


class MobileSDRManager:
    """Manager for mobile SDR devices"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.available_sdrs = {}
        self.active_sdr = None
        self.collection_active = False
        self._collection_thread = None
        self._data_callback = None
        
        # Detect available SDR devices
        self._detect_sdr_devices()
    
    def _detect_sdr_devices(self):
        """Detect available SDR devices"""
        self.logger.info("Detecting mobile SDR devices...")
        
        # In a real implementation, this would use actual SDR libraries
        # For now, simulate detection
        
        # Simulate RTL-SDR detection
        if self._check_rtl_sdr():
            self.available_sdrs['rtl_sdr'] = MobileSDRCapabilities(
                sdr_type=MobileSDRType.RTL_SDR_MOBILE,
                frequency_range=(24e6, 1766e6),
                sample_rate_range=(225e3, 3.2e6),
                gain_range=(0.0, 50.0),
                supports_tx=False,
                power_consumption_mw=300.0
            )
            self.logger.info("Detected RTL-SDR mobile device")
        
        # Simulate HackRF detection
        if self._check_hackrf():
            self.available_sdrs['hackrf'] = MobileSDRCapabilities(
                sdr_type=MobileSDRType.HACKRF_MOBILE,
                frequency_range=(1e6, 6e9),
                sample_rate_range=(2e6, 20e6),
                gain_range=(0.0, 62.0),
                supports_tx=True,
                supports_full_duplex=True,
                power_consumption_mw=800.0
            )
            self.logger.info("Detected HackRF mobile device")
        
        # Check for built-in radio capabilities (WiFi, Bluetooth, Cellular)
        if self._check_builtin_radio():
            self.available_sdrs['builtin'] = MobileSDRCapabilities(
                sdr_type=MobileSDRType.BUILT_IN_RADIO,
                frequency_range=(2.4e9, 5.8e9),  # WiFi bands
                sample_rate_range=(1e6, 20e6),
                gain_range=(-10.0, 30.0),
                supports_tx=True,
                power_consumption_mw=200.0,
                usb_powered=False
            )
            self.logger.info("Detected built-in radio capabilities")
        
        # Always have simulated SDR available
        self.available_sdrs['simulated'] = MobileSDRCapabilities(
            sdr_type=MobileSDRType.SIMULATED,
            frequency_range=(1e6, 6e9),
            sample_rate_range=(1e3, 50e6),
            gain_range=(-20.0, 60.0),
            supports_tx=True,
            supports_full_duplex=True,
            power_consumption_mw=0.0,
            usb_powered=False
        )
        
        self.logger.info(f"Detected {len(self.available_sdrs)} SDR devices")
    
    def _check_rtl_sdr(self) -> bool:
        """Check if RTL-SDR is available"""
        try:
            # In real implementation, would check for RTL-SDR library and hardware
            # For now, simulate based on platform
            import platform
            return platform.system().lower() in ['linux', 'windows', 'darwin']
        except:
            return False
    
    def _check_hackrf(self) -> bool:
        """Check if HackRF is available"""
        try:
            # In real implementation, would check for HackRF library and hardware
            return False  # Simulate not available for demo
        except:
            return False
    
    def _check_builtin_radio(self) -> bool:
        """Check for built-in radio capabilities"""
        try:
            # Most mobile devices have WiFi/Bluetooth
            return True
        except:
            return False
    
    def get_available_sdrs(self) -> Dict[str, MobileSDRCapabilities]:
        """Get available SDR devices"""
        return self.available_sdrs.copy()
    
    def select_sdr(self, sdr_id: str) -> bool:
        """Select SDR device for use"""
        if sdr_id not in self.available_sdrs:
            self.logger.error(f"SDR device not available: {sdr_id}")
            return False
        
        self.active_sdr = sdr_id
        self.logger.info(f"Selected SDR device: {sdr_id}")
        return True
    
    def configure_sdr(self, rf_params: RFParameters) -> bool:
        """Configure active SDR device"""
        if not self.active_sdr:
            self.logger.error("No SDR device selected")
            return False
        
        sdr_caps = self.available_sdrs[self.active_sdr]
        
        # Validate parameters against capabilities
        if not self._validate_rf_params(rf_params, sdr_caps):
            return False
        
        self.logger.info(f"Configured SDR: {rf_params.center_frequency/1e6:.1f} MHz, "
                        f"{rf_params.sample_rate/1e6:.1f} Msps, {rf_params.gain:.1f} dB")
        return True
    
    def _validate_rf_params(self, rf_params: RFParameters, 
                          sdr_caps: MobileSDRCapabilities) -> bool:
        """Validate RF parameters against SDR capabilities"""
        # Check frequency range
        if not (sdr_caps.frequency_range[0] <= rf_params.center_frequency <= sdr_caps.frequency_range[1]):
            self.logger.error(f"Frequency {rf_params.center_frequency/1e6:.1f} MHz out of range "
                            f"[{sdr_caps.frequency_range[0]/1e6:.1f}, {sdr_caps.frequency_range[1]/1e6:.1f}] MHz")
            return False
        
        # Check sample rate range
        if not (sdr_caps.sample_rate_range[0] <= rf_params.sample_rate <= sdr_caps.sample_rate_range[1]):
            self.logger.error(f"Sample rate {rf_params.sample_rate/1e6:.1f} Msps out of range "
                            f"[{sdr_caps.sample_rate_range[0]/1e6:.1f}, {sdr_caps.sample_rate_range[1]/1e6:.1f}] Msps")
            return False
        
        # Check gain range
        if not (sdr_caps.gain_range[0] <= rf_params.gain <= sdr_caps.gain_range[1]):
            self.logger.error(f"Gain {rf_params.gain:.1f} dB out of range "
                            f"[{sdr_caps.gain_range[0]:.1f}, {sdr_caps.gain_range[1]:.1f}] dB")
            return False
        
        return True
    
    def start_collection(self, rf_params: RFParameters, duration_seconds: float,
                        callback: Optional[Callable[[EnhancedSignalSample], None]] = None) -> bool:
        """Start signal collection"""
        if self.collection_active:
            self.logger.warning("Collection already active")
            return False
        
        if not self.active_sdr:
            self.logger.error("No SDR device selected")
            return False
        
        if not self.configure_sdr(rf_params):
            return False
        
        self._data_callback = callback
        self.collection_active = True
        
        # Start collection thread
        self._collection_thread = threading.Thread(
            target=self._collection_worker,
            args=(rf_params, duration_seconds)
        )
        self._collection_thread.daemon = True
        self._collection_thread.start()
        
        self.logger.info(f"Started signal collection for {duration_seconds}s")
        return True
    
    def stop_collection(self):
        """Stop signal collection"""
        if not self.collection_active:
            return
        
        self.collection_active = False
        
        if self._collection_thread:
            self._collection_thread.join(timeout=5.0)
        
        self.logger.info("Stopped signal collection")
    
    def _collection_worker(self, rf_params: RFParameters, duration_seconds: float):
        """Worker thread for signal collection"""
        try:
            start_time = time.time()
            samples_collected = 0
            
            while self.collection_active and (time.time() - start_time) < duration_seconds:
                # Generate or collect signal sample
                sample = self._collect_sample(rf_params)
                
                if sample and self._data_callback:
                    self._data_callback(sample)
                
                samples_collected += 1
                
                # Control collection rate
                time.sleep(0.1)  # 10 Hz collection rate
            
            self.logger.info(f"Collection completed: {samples_collected} samples in {time.time() - start_time:.1f}s")
            
        except Exception as e:
            self.logger.error(f"Collection error: {e}")
        finally:
            self.collection_active = False
    
    def _collect_sample(self, rf_params: RFParameters) -> Optional[EnhancedSignalSample]:
        """Collect a single signal sample"""
        try:
            if self.active_sdr == 'simulated':
                return self._generate_simulated_sample(rf_params)
            elif self.active_sdr == 'builtin':
                return self._collect_builtin_sample(rf_params)
            else:
                return self._collect_sdr_sample(rf_params)
                
        except Exception as e:
            self.logger.error(f"Error collecting sample: {e}")
            return None
    
    def _generate_simulated_sample(self, rf_params: RFParameters) -> EnhancedSignalSample:
        """Generate simulated signal sample"""
        # Generate synthetic signal
        duration = 0.001  # 1ms sample
        num_samples = int(rf_params.sample_rate * duration)
        
        # Random modulation type
        modulations = [ModulationType.QPSK, ModulationType.QAM16, ModulationType.BPSK]
        modulation = np.random.choice(modulations)
        
        # Generate signal based on modulation
        if modulation == ModulationType.QPSK:
            symbols = np.random.choice([1+1j, 1-1j, -1+1j, -1-1j], num_samples//8)
            signal = np.repeat(symbols, 8)
        elif modulation == ModulationType.QAM16:
            constellation = [1+1j, 1+3j, 3+1j, 3+3j, 1-1j, 1-3j, 3-1j, 3-3j,
                           -1+1j, -1+3j, -3+1j, -3+3j, -1-1j, -1-3j, -3-1j, -3-3j]
            symbols = np.random.choice(constellation, num_samples//8)
            signal = np.repeat(symbols, 8)
        else:  # BPSK
            symbols = np.random.choice([1+0j, -1+0j], num_samples//8)
            signal = np.repeat(symbols, 8)
        
        # Ensure correct length
        if len(signal) > num_samples:
            signal = signal[:num_samples]
        elif len(signal) < num_samples:
            signal = np.pad(signal, (0, num_samples - len(signal)), 'constant')
        
        # Add noise
        snr_db = 10 + 20 * np.random.random()  # 10-30 dB SNR
        noise_power = 10**(-snr_db/10)
        noise = np.sqrt(noise_power/2) * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
        iq_data = signal + noise
        
        # Create quality metrics
        quality_metrics = SignalQualityMetrics(snr_db=snr_db)
        quality_metrics.calculate_derived_metrics(iq_data)
        
        # Create hardware info
        hardware_info = HardwareInfo(
            hardware_type=HardwareType.SIMULATED,
            serial_number="SIM001",
            firmware_version="1.0",
            frequency_range=(1e6, 6e9),
            max_sample_rate=50e6
        )
        
        return EnhancedSignalSample(
            iq_data=iq_data,
            timestamp=datetime.now(),
            duration=duration,
            rf_params=rf_params,
            modulation_type=modulation,
            quality_metrics=quality_metrics,
            hardware_info=hardware_info,
            device_id=f"mobile_sdr_{self.active_sdr}",
            environment="mobile_simulated"
        )
    
    def _collect_builtin_sample(self, rf_params: RFParameters) -> EnhancedSignalSample:
        """Collect sample from built-in radio"""
        # In a real implementation, this would interface with WiFi/Bluetooth chips
        # For now, generate WiFi-like signal
        
        duration = 0.001  # 1ms
        num_samples = int(rf_params.sample_rate * duration)
        
        # Generate OFDM-like signal (WiFi)
        subcarriers = 64
        ofdm_symbols = num_samples // subcarriers
        
        # Generate random OFDM data
        data = np.random.randn(ofdm_symbols, subcarriers) + 1j * np.random.randn(ofdm_symbols, subcarriers)
        
        # Simple IFFT to create time domain signal
        time_signal = []
        for symbol in data:
            time_symbol = np.fft.ifft(symbol)
            time_signal.extend(time_symbol)
        
        iq_data = np.array(time_signal[:num_samples])
        
        # Add realistic WiFi noise and interference
        snr_db = 15 + 10 * np.random.random()  # 15-25 dB typical for WiFi
        noise_power = 10**(-snr_db/10)
        noise = np.sqrt(noise_power/2) * (np.random.randn(len(iq_data)) + 1j * np.random.randn(len(iq_data)))
        iq_data = iq_data + noise
        
        quality_metrics = SignalQualityMetrics(snr_db=snr_db)
        quality_metrics.calculate_derived_metrics(iq_data)
        
        hardware_info = HardwareInfo(
            hardware_type=HardwareType.UNKNOWN,
            serial_number="BUILTIN001",
            firmware_version="mobile_radio_1.0",
            frequency_range=(2.4e9, 5.8e9),
            max_sample_rate=20e6
        )
        
        return EnhancedSignalSample(
            iq_data=iq_data,
            timestamp=datetime.now(),
            duration=duration,
            rf_params=rf_params,
            modulation_type=ModulationType.OFDM,
            quality_metrics=quality_metrics,
            hardware_info=hardware_info,
            device_id=f"mobile_builtin_{self.active_sdr}",
            environment="mobile_builtin"
        )
    
    def _collect_sdr_sample(self, rf_params: RFParameters) -> EnhancedSignalSample:
        """Collect sample from external SDR device"""
        # In a real implementation, this would interface with actual SDR hardware
        # For now, simulate RTL-SDR or HackRF collection
        
        duration = 0.001
        num_samples = int(rf_params.sample_rate * duration)
        
        # Simulate real-world signal collection with more realistic characteristics
        # Add frequency-dependent characteristics
        if rf_params.center_frequency < 100e6:
            # HF/VHF - more atmospheric noise
            base_snr = 5
            noise_variation = 10
        elif rf_params.center_frequency < 1e9:
            # UHF - moderate noise
            base_snr = 10
            noise_variation = 8
        else:
            # Microwave - lower noise floor
            base_snr = 15
            noise_variation = 5
        
        snr_db = base_snr + noise_variation * np.random.random()
        
        # Generate more realistic signal with fading
        signal_power = 1.0
        fading = 0.8 + 0.4 * np.random.random()  # Rayleigh-like fading
        
        # Create signal with multiple components (realistic spectrum)
        iq_data = np.zeros(num_samples, dtype=complex)
        
        # Add multiple signal components
        for i in range(3):
            freq_offset = (np.random.random() - 0.5) * rf_params.sample_rate * 0.3
            phase = 2 * np.pi * np.random.random()
            amplitude = fading * signal_power * (0.5 + 0.5 * np.random.random())
            
            t = np.arange(num_samples) / rf_params.sample_rate
            component = amplitude * np.exp(1j * (2 * np.pi * freq_offset * t + phase))
            iq_data += component
        
        # Add noise
        noise_power = signal_power * 10**(-snr_db/10)
        noise = np.sqrt(noise_power/2) * (np.random.randn(num_samples) + 1j * np.random.randn(num_samples))
        iq_data += noise
        
        quality_metrics = SignalQualityMetrics(snr_db=snr_db)
        quality_metrics.calculate_derived_metrics(iq_data)
        
        # Hardware info based on SDR type
        sdr_caps = self.available_sdrs[self.active_sdr]
        if sdr_caps.sdr_type == MobileSDRType.RTL_SDR_MOBILE:
            hardware_type = HardwareType.RTL_SDR
        elif sdr_caps.sdr_type == MobileSDRType.HACKRF_MOBILE:
            hardware_type = HardwareType.HACKRF
        else:
            hardware_type = HardwareType.UNKNOWN
        
        hardware_info = HardwareInfo(
            hardware_type=hardware_type,
            serial_number=f"MOBILE_{self.active_sdr.upper()}001",
            firmware_version="mobile_1.0",
            frequency_range=sdr_caps.frequency_range,
            max_sample_rate=sdr_caps.sample_rate_range[1]
        )
        
        return EnhancedSignalSample(
            iq_data=iq_data,
            timestamp=datetime.now(),
            duration=duration,
            rf_params=rf_params,
            modulation_type=ModulationType.UNKNOWN,  # Would need classification
            quality_metrics=quality_metrics,
            hardware_info=hardware_info,
            device_id=f"mobile_sdr_{self.active_sdr}",
            environment="mobile_sdr"
        )
    
    def get_power_consumption(self) -> float:
        """Get estimated power consumption in mW"""
        if not self.active_sdr:
            return 0.0
        
        sdr_caps = self.available_sdrs[self.active_sdr]
        base_power = sdr_caps.power_consumption_mw
        
        # Increase power consumption if collecting
        if self.collection_active:
            return base_power * 1.5
        else:
            return base_power * 0.1  # Idle power
    
    def cleanup(self):
        """Cleanup SDR resources"""
        self.stop_collection()
        self.active_sdr = None
        self.logger.info("Mobile SDR manager cleaned up")