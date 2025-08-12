"""
Signal augmentation with realistic channel effects (fading, noise, interference)
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from scipy import signal
import random

from .interfaces import SignalSample


@dataclass
class ChannelConfig:
    """Configuration for channel effects"""
    # Fading parameters
    enable_fading: bool = True
    fading_type: str = 'rayleigh'  # rayleigh, rician, nakagami
    k_factor: float = 1.0  # For Rician fading
    m_parameter: float = 1.0  # For Nakagami fading
    doppler_frequency: float = 10.0  # Hz
    
    # Noise parameters
    enable_awgn: bool = True
    snr_range: Tuple[float, float] = (-10.0, 30.0)  # dB
    
    # Interference parameters
    enable_interference: bool = True
    interference_types: List[str] = None  # ['narrowband', 'wideband', 'impulsive']
    interference_probability: float = 0.3
    
    # Multipath parameters
    enable_multipath: bool = True
    max_paths: int = 3
    delay_spread: float = 1e-6  # seconds
    
    # Frequency offset and drift
    enable_frequency_offset: bool = True
    frequency_offset_range: Tuple[float, float] = (-1000.0, 1000.0)  # Hz
    
    # Phase noise
    enable_phase_noise: bool = True
    phase_noise_power: float = 0.01
    
    # IQ imbalance
    enable_iq_imbalance: bool = True
    amplitude_imbalance_db: float = 0.5
    phase_imbalance_deg: float = 2.0


class ChannelSimulator:
    """Simulate realistic RF channel effects"""
    
    def __init__(self, config: ChannelConfig = None):
        self.config = config or ChannelConfig()
        if self.config.interference_types is None:
            self.config.interference_types = ['narrowband', 'wideband', 'impulsive']
    
    def apply_channel_effects(self, samples: List[SignalSample]) -> List[SignalSample]:
        """Apply realistic channel effects to signal samples"""
        augmented_samples = []
        
        for sample in samples:
            # Create multiple versions with different channel conditions
            base_sample = self._apply_single_channel_realization(sample)
            augmented_samples.append(base_sample)
            
            # Add additional realizations for diversity
            for _ in range(2):  # Create 2 additional realizations
                augmented_sample = self._apply_single_channel_realization(sample)
                augmented_samples.append(augmented_sample)
        
        return augmented_samples
    
    def _apply_single_channel_realization(self, sample: SignalSample) -> SignalSample:
        """Apply a single channel realization to a sample"""
        iq_data = sample.iq_data.copy()
        channel_metadata = {}
        
        # Apply channel effects in realistic order
        if self.config.enable_multipath:
            iq_data, multipath_info = self._apply_multipath(iq_data, sample.sample_rate)
            channel_metadata['multipath'] = multipath_info
        
        if self.config.enable_fading:
            iq_data, fading_info = self._apply_fading(iq_data)
            channel_metadata['fading'] = fading_info
        
        if self.config.enable_frequency_offset:
            iq_data, freq_offset = self._apply_frequency_offset(iq_data, sample.sample_rate)
            channel_metadata['frequency_offset_hz'] = freq_offset
        
        if self.config.enable_phase_noise:
            iq_data = self._apply_phase_noise(iq_data)
            channel_metadata['phase_noise_applied'] = True
        
        if self.config.enable_iq_imbalance:
            iq_data = self._apply_iq_imbalance(iq_data)
            channel_metadata['iq_imbalance_applied'] = True
        
        if self.config.enable_interference:
            iq_data, interference_info = self._apply_interference(iq_data, sample.sample_rate)
            channel_metadata['interference'] = interference_info
        
        if self.config.enable_awgn:
            iq_data, actual_snr = self._apply_awgn(iq_data)
            channel_metadata['actual_snr_db'] = actual_snr
        else:
            actual_snr = sample.snr
        
        # Create augmented sample
        augmented_sample = SignalSample(
            timestamp=sample.timestamp,
            frequency=sample.frequency,
            sample_rate=sample.sample_rate,
            iq_data=iq_data,
            modulation_type=sample.modulation_type,
            snr=actual_snr,
            location=sample.location,
            device_id=sample.device_id,
            metadata={
                **sample.metadata,
                'channel_effects': channel_metadata,
                'augmented': True
            }
        )
        
        return augmented_sample
    
    def _apply_fading(self, iq_data: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply fading channel effects"""
        fading_info = {'type': self.config.fading_type}
        
        if self.config.fading_type == 'rayleigh':
            # Rayleigh fading (no line-of-sight)
            h_real = np.random.normal(0, 1/np.sqrt(2), len(iq_data))
            h_imag = np.random.normal(0, 1/np.sqrt(2), len(iq_data))
            h = h_real + 1j * h_imag
            fading_info['average_power'] = 1.0
            
        elif self.config.fading_type == 'rician':
            # Rician fading (with line-of-sight component)
            k = self.config.k_factor
            # Line-of-sight component
            h_los = np.sqrt(k / (k + 1))
            # Scattered component
            h_nlos_real = np.random.normal(0, 1/np.sqrt(2*(k + 1)), len(iq_data))
            h_nlos_imag = np.random.normal(0, 1/np.sqrt(2*(k + 1)), len(iq_data))
            h = h_los + h_nlos_real + 1j * h_nlos_imag
            fading_info['k_factor'] = k
            
        elif self.config.fading_type == 'nakagami':
            # Nakagami-m fading
            m = self.config.m_parameter
            # Generate Nakagami fading using Gamma distribution
            amplitude = np.sqrt(np.random.gamma(m, 1/m, len(iq_data)))
            phase = np.random.uniform(0, 2*np.pi, len(iq_data))
            h = amplitude * np.exp(1j * phase)
            fading_info['m_parameter'] = m
            
        else:
            h = np.ones(len(iq_data), dtype=complex)
        
        # Apply Doppler effect if specified
        if self.config.doppler_frequency > 0:
            t = np.arange(len(iq_data)) / len(iq_data)  # Normalized time
            doppler_phase = 2 * np.pi * self.config.doppler_frequency * t
            h *= np.exp(1j * doppler_phase)
            fading_info['doppler_frequency_hz'] = self.config.doppler_frequency
        
        faded_signal = iq_data * h
        return faded_signal, fading_info
    
    def _apply_multipath(self, iq_data: np.ndarray, sample_rate: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply multipath propagation effects"""
        num_paths = random.randint(1, self.config.max_paths)
        delays_samples = []
        path_gains = []
        
        # Generate path delays and gains
        for i in range(num_paths):
            if i == 0:
                # Direct path
                delay_samples = 0
                gain = 1.0
            else:
                # Reflected paths
                delay_time = random.uniform(0, self.config.delay_spread)
                delay_samples = int(delay_time * sample_rate)
                # Path loss with distance (simplified)
                gain = random.uniform(0.1, 0.8) * np.exp(-i * 0.5)
            
            delays_samples.append(delay_samples)
            path_gains.append(gain)
        
        # Apply multipath
        multipath_signal = np.zeros_like(iq_data)
        
        for delay, gain in zip(delays_samples, path_gains):
            if delay < len(iq_data):
                # Add delayed and attenuated version
                delayed_signal = np.zeros_like(iq_data)
                delayed_signal[delay:] = iq_data[:-delay] if delay > 0 else iq_data
                multipath_signal += gain * delayed_signal
        
        multipath_info = {
            'num_paths': num_paths,
            'delays_samples': delays_samples,
            'path_gains': path_gains,
            'delay_spread_s': self.config.delay_spread
        }
        
        return multipath_signal, multipath_info
    
    def _apply_frequency_offset(self, iq_data: np.ndarray, sample_rate: float) -> Tuple[np.ndarray, float]:
        """Apply frequency offset and drift"""
        freq_offset = random.uniform(*self.config.frequency_offset_range)
        
        # Create time vector
        t = np.arange(len(iq_data)) / sample_rate
        
        # Apply frequency offset
        offset_signal = iq_data * np.exp(1j * 2 * np.pi * freq_offset * t)
        
        return offset_signal, freq_offset
    
    def _apply_phase_noise(self, iq_data: np.ndarray) -> np.ndarray:
        """Apply phase noise (oscillator imperfections)"""
        # Generate colored phase noise
        phase_noise = np.random.normal(0, np.sqrt(self.config.phase_noise_power), len(iq_data))
        
        # Apply 1/f characteristic (simplified)
        if len(phase_noise) > 1:
            # Simple first-order filter to create colored noise
            alpha = 0.9
            for i in range(1, len(phase_noise)):
                phase_noise[i] = alpha * phase_noise[i-1] + (1-alpha) * phase_noise[i]
        
        # Apply phase noise
        noisy_signal = iq_data * np.exp(1j * phase_noise)
        
        return noisy_signal
    
    def _apply_iq_imbalance(self, iq_data: np.ndarray) -> np.ndarray:
        """Apply IQ imbalance (amplitude and phase mismatch)"""
        # Amplitude imbalance
        amp_imbalance_linear = 10 ** (self.config.amplitude_imbalance_db / 20)
        
        # Phase imbalance
        phase_imbalance_rad = np.deg2rad(self.config.phase_imbalance_deg)
        
        # Apply imbalance
        i_component = iq_data.real * amp_imbalance_linear
        q_component = iq_data.imag * np.cos(phase_imbalance_rad) + iq_data.real * np.sin(phase_imbalance_rad)
        
        imbalanced_signal = i_component + 1j * q_component
        
        return imbalanced_signal
    
    def _apply_interference(self, iq_data: np.ndarray, sample_rate: float) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply various types of interference"""
        interference_info = {'applied': False}
        
        if random.random() > self.config.interference_probability:
            return iq_data, interference_info
        
        interference_type = random.choice(self.config.interference_types)
        interference_info['type'] = interference_type
        interference_info['applied'] = True
        
        if interference_type == 'narrowband':
            # Narrowband interference (e.g., carrier leak, spurious signals)
            interference_freq = random.uniform(-sample_rate/4, sample_rate/4)
            interference_power = random.uniform(0.1, 0.5)
            t = np.arange(len(iq_data)) / sample_rate
            
            interference = np.sqrt(interference_power) * np.exp(1j * 2 * np.pi * interference_freq * t)
            interference_info['frequency_hz'] = interference_freq
            interference_info['power'] = interference_power
            
        elif interference_type == 'wideband':
            # Wideband interference (e.g., adjacent channel)
            interference_power = random.uniform(0.05, 0.3)
            interference = np.sqrt(interference_power) * (
                np.random.normal(0, 1, len(iq_data)) + 1j * np.random.normal(0, 1, len(iq_data))
            )
            interference_info['power'] = interference_power
            
        elif interference_type == 'impulsive':
            # Impulsive interference (e.g., ignition noise, switching transients)
            num_impulses = random.randint(1, 5)
            interference = np.zeros_like(iq_data)
            
            for _ in range(num_impulses):
                impulse_start = random.randint(0, len(iq_data) - 10)
                impulse_length = random.randint(5, 20)
                impulse_end = min(impulse_start + impulse_length, len(iq_data))
                
                impulse_amplitude = random.uniform(2.0, 10.0)
                interference[impulse_start:impulse_end] = impulse_amplitude * (
                    np.random.normal(0, 1, impulse_end - impulse_start) + 
                    1j * np.random.normal(0, 1, impulse_end - impulse_start)
                )
            
            interference_info['num_impulses'] = num_impulses
        
        else:
            interference = np.zeros_like(iq_data)
        
        interfered_signal = iq_data + interference
        return interfered_signal, interference_info
    
    def _apply_awgn(self, iq_data: np.ndarray) -> Tuple[np.ndarray, float]:
        """Apply Additive White Gaussian Noise"""
        # Calculate signal power
        signal_power = np.mean(np.abs(iq_data) ** 2)
        
        # Random SNR from specified range
        target_snr_db = random.uniform(*self.config.snr_range)
        
        # Calculate noise power
        snr_linear = 10 ** (target_snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # Generate AWGN
        noise_real = np.random.normal(0, np.sqrt(noise_power/2), len(iq_data))
        noise_imag = np.random.normal(0, np.sqrt(noise_power/2), len(iq_data))
        noise = noise_real + 1j * noise_imag
        
        # Add noise to signal
        noisy_signal = iq_data + noise
        
        # Calculate actual SNR
        actual_signal_power = np.mean(np.abs(iq_data) ** 2)
        actual_noise_power = np.mean(np.abs(noise) ** 2)
        actual_snr_db = 10 * np.log10(actual_signal_power / actual_noise_power)
        
        return noisy_signal, actual_snr_db


class EnvironmentalEffects:
    """Simulate environmental effects on RF signals"""
    
    @staticmethod
    def apply_weather_effects(samples: List[SignalSample], weather_condition: str) -> List[SignalSample]:
        """Apply weather-based signal degradation"""
        affected_samples = []
        
        for sample in samples:
            iq_data = sample.iq_data.copy()
            weather_metadata = {'condition': weather_condition}
            
            if weather_condition == 'rain':
                # Rain attenuation (frequency dependent)
                if sample.frequency > 1e9:  # Above 1 GHz
                    attenuation_db = random.uniform(0.5, 3.0)  # dB
                    attenuation_linear = 10 ** (-attenuation_db / 20)
                    iq_data *= attenuation_linear
                    weather_metadata['attenuation_db'] = attenuation_db
            
            elif weather_condition == 'fog':
                # Fog scattering (mainly at higher frequencies)
                if sample.frequency > 10e9:  # Above 10 GHz
                    attenuation_db = random.uniform(0.1, 1.0)
                    attenuation_linear = 10 ** (-attenuation_db / 20)
                    iq_data *= attenuation_linear
                    weather_metadata['attenuation_db'] = attenuation_db
            
            elif weather_condition == 'atmospheric_ducting':
                # Atmospheric ducting can cause signal enhancement or fading
                enhancement_db = random.uniform(-5.0, 10.0)
                enhancement_linear = 10 ** (enhancement_db / 20)
                iq_data *= enhancement_linear
                weather_metadata['enhancement_db'] = enhancement_db
            
            # Create weather-affected sample
            affected_sample = SignalSample(
                timestamp=sample.timestamp,
                frequency=sample.frequency,
                sample_rate=sample.sample_rate,
                iq_data=iq_data,
                modulation_type=sample.modulation_type,
                snr=sample.snr,
                location=sample.location,
                device_id=sample.device_id,
                metadata={
                    **sample.metadata,
                    'weather_effects': weather_metadata
                }
            )
            affected_samples.append(affected_sample)
        
        return affected_samples
    
    @staticmethod
    def apply_urban_effects(samples: List[SignalSample], environment_type: str) -> List[SignalSample]:
        """Apply urban environment effects"""
        affected_samples = []
        
        for sample in samples:
            iq_data = sample.iq_data.copy()
            urban_metadata = {'environment': environment_type}
            
            if environment_type == 'dense_urban':
                # High multipath, shadowing
                # Additional path loss
                path_loss_db = random.uniform(5.0, 15.0)
                path_loss_linear = 10 ** (-path_loss_db / 20)
                iq_data *= path_loss_linear
                urban_metadata['path_loss_db'] = path_loss_db
                
            elif environment_type == 'suburban':
                # Moderate multipath
                path_loss_db = random.uniform(2.0, 8.0)
                path_loss_linear = 10 ** (-path_loss_db / 20)
                iq_data *= path_loss_linear
                urban_metadata['path_loss_db'] = path_loss_db
                
            elif environment_type == 'indoor':
                # Wall penetration loss
                penetration_loss_db = random.uniform(10.0, 30.0)
                penetration_loss_linear = 10 ** (-penetration_loss_db / 20)
                iq_data *= penetration_loss_linear
                urban_metadata['penetration_loss_db'] = penetration_loss_db
            
            # Create environment-affected sample
            affected_sample = SignalSample(
                timestamp=sample.timestamp,
                frequency=sample.frequency,
                sample_rate=sample.sample_rate,
                iq_data=iq_data,
                modulation_type=sample.modulation_type,
                snr=sample.snr,
                location=sample.location,
                device_id=sample.device_id,
                metadata={
                    **sample.metadata,
                    'urban_effects': urban_metadata
                }
            )
            affected_samples.append(affected_sample)
        
        return affected_samples