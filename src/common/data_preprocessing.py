"""
Data preprocessing pipeline for real IQ samples
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from dataclasses import dataclass

from .interfaces import SignalSample


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing"""
    normalize_power: bool = True
    apply_filtering: bool = True
    filter_type: str = 'lowpass'  # lowpass, highpass, bandpass
    filter_cutoff: float = 0.4  # Normalized frequency
    resample_rate: Optional[float] = None
    add_noise: bool = False
    noise_power: float = 0.1
    apply_fading: bool = False
    fading_type: str = 'rayleigh'  # rayleigh, rician
    segment_length: int = 1024
    overlap_ratio: float = 0.5


class SignalPreprocessor:
    """Signal preprocessing pipeline for IQ samples"""
    
    def __init__(self, config: PreprocessingConfig = None):
        self.config = config or PreprocessingConfig()
    
    def preprocess_samples(self, samples: List[SignalSample]) -> List[SignalSample]:
        """Preprocess a list of signal samples"""
        processed_samples = []
        
        for sample in samples:
            processed_sample = self.preprocess_single_sample(sample)
            processed_samples.append(processed_sample)
        
        return processed_samples
    
    def preprocess_single_sample(self, sample: SignalSample) -> SignalSample:
        """Preprocess a single signal sample"""
        iq_data = sample.iq_data.copy()
        
        # Apply preprocessing steps
        if self.config.normalize_power:
            iq_data = self._normalize_power(iq_data)
        
        if self.config.apply_filtering:
            iq_data = self._apply_filter(iq_data, sample.sample_rate)
        
        if self.config.resample_rate and self.config.resample_rate != sample.sample_rate:
            iq_data, new_sample_rate = self._resample(iq_data, sample.sample_rate, self.config.resample_rate)
        else:
            new_sample_rate = sample.sample_rate
        
        if self.config.add_noise:
            iq_data = self._add_noise(iq_data)
        
        if self.config.apply_fading:
            iq_data = self._apply_fading(iq_data)
        
        # Create new sample with processed data
        processed_sample = SignalSample(
            timestamp=sample.timestamp,
            frequency=sample.frequency,
            sample_rate=new_sample_rate,
            iq_data=iq_data,
            modulation_type=sample.modulation_type,
            snr=self._estimate_snr(iq_data),
            location=sample.location,
            device_id=sample.device_id,
            metadata={**sample.metadata, 'preprocessed': True}
        )
        
        return processed_sample
    
    def segment_samples(self, samples: List[SignalSample]) -> List[SignalSample]:
        """Segment long samples into smaller chunks"""
        segmented_samples = []
        
        for sample in samples:
            segments = self._segment_iq_data(
                sample.iq_data, 
                self.config.segment_length, 
                self.config.overlap_ratio
            )
            
            for i, segment in enumerate(segments):
                segmented_sample = SignalSample(
                    timestamp=sample.timestamp,
                    frequency=sample.frequency,
                    sample_rate=sample.sample_rate,
                    iq_data=segment,
                    modulation_type=sample.modulation_type,
                    snr=sample.snr,
                    location=sample.location,
                    device_id=sample.device_id,
                    metadata={**sample.metadata, 'segment_index': i, 'total_segments': len(segments)}
                )
                segmented_samples.append(segmented_sample)
        
        return segmented_samples
    
    def extract_features(self, samples: List[SignalSample]) -> Dict[str, np.ndarray]:
        """Extract features from signal samples"""
        features = {
            'iq_samples': [],
            'spectrograms': [],
            'constellation_features': [],
            'statistical_features': [],
            'cyclic_features': []
        }
        
        for sample in samples:
            # Raw IQ samples
            features['iq_samples'].append(self._format_iq_for_ml(sample.iq_data))
            
            # Spectrogram features
            spectrogram = self._compute_spectrogram(sample.iq_data, sample.sample_rate)
            features['spectrograms'].append(spectrogram)
            
            # Constellation diagram features
            constellation_features = self._extract_constellation_features(sample.iq_data)
            features['constellation_features'].append(constellation_features)
            
            # Statistical features
            statistical_features = self._extract_statistical_features(sample.iq_data)
            features['statistical_features'].append(statistical_features)
            
            # Cyclic features
            cyclic_features = self._extract_cyclic_features(sample.iq_data)
            features['cyclic_features'].append(cyclic_features)
        
        # Convert lists to numpy arrays
        for key in features:
            features[key] = np.array(features[key])
        
        return features
    
    def _normalize_power(self, iq_data: np.ndarray) -> np.ndarray:
        """Normalize signal power to unit power"""
        power = np.mean(np.abs(iq_data) ** 2)
        return iq_data / np.sqrt(power) if power > 0 else iq_data
    
    def _apply_filter(self, iq_data: np.ndarray, sample_rate: float) -> np.ndarray:
        """Apply digital filter to IQ data"""
        nyquist = sample_rate / 2
        
        if self.config.filter_type == 'lowpass':
            b, a = signal.butter(4, self.config.filter_cutoff, btype='low')
        elif self.config.filter_type == 'highpass':
            b, a = signal.butter(4, self.config.filter_cutoff, btype='high')
        elif self.config.filter_type == 'bandpass':
            low = self.config.filter_cutoff - 0.1
            high = self.config.filter_cutoff + 0.1
            b, a = signal.butter(4, [low, high], btype='band')
        else:
            return iq_data
        
        # Apply filter to real and imaginary parts separately
        filtered_real = signal.filtfilt(b, a, iq_data.real)
        filtered_imag = signal.filtfilt(b, a, iq_data.imag)
        
        return filtered_real + 1j * filtered_imag
    
    def _resample(self, iq_data: np.ndarray, original_rate: float, target_rate: float) -> Tuple[np.ndarray, float]:
        """Resample IQ data to target sample rate"""
        if original_rate == target_rate:
            return iq_data, original_rate
        
        # Calculate resampling ratio
        ratio = target_rate / original_rate
        new_length = int(len(iq_data) * ratio)
        
        # Resample using scipy
        resampled_data = signal.resample(iq_data, new_length)
        
        return resampled_data, target_rate
    
    def _add_noise(self, iq_data: np.ndarray) -> np.ndarray:
        """Add AWGN to signal"""
        noise_real = np.random.normal(0, np.sqrt(self.config.noise_power/2), len(iq_data))
        noise_imag = np.random.normal(0, np.sqrt(self.config.noise_power/2), len(iq_data))
        noise = noise_real + 1j * noise_imag
        
        return iq_data + noise
    
    def _apply_fading(self, iq_data: np.ndarray) -> np.ndarray:
        """Apply fading channel effects"""
        if self.config.fading_type == 'rayleigh':
            # Rayleigh fading
            h_real = np.random.normal(0, 1/np.sqrt(2), len(iq_data))
            h_imag = np.random.normal(0, 1/np.sqrt(2), len(iq_data))
            h = h_real + 1j * h_imag
        elif self.config.fading_type == 'rician':
            # Rician fading with K-factor = 1
            k_factor = 1
            h_los = np.sqrt(k_factor / (k_factor + 1))
            h_nlos_real = np.random.normal(0, 1/np.sqrt(2*(k_factor + 1)), len(iq_data))
            h_nlos_imag = np.random.normal(0, 1/np.sqrt(2*(k_factor + 1)), len(iq_data))
            h = h_los + h_nlos_real + 1j * h_nlos_imag
        else:
            return iq_data
        
        return iq_data * h
    
    def _segment_iq_data(self, iq_data: np.ndarray, segment_length: int, overlap_ratio: float) -> List[np.ndarray]:
        """Segment IQ data into overlapping windows"""
        if len(iq_data) <= segment_length:
            return [iq_data]
        
        step_size = int(segment_length * (1 - overlap_ratio))
        segments = []
        
        for start in range(0, len(iq_data) - segment_length + 1, step_size):
            segment = iq_data[start:start + segment_length]
            segments.append(segment)
        
        return segments
    
    def _format_iq_for_ml(self, iq_data: np.ndarray) -> np.ndarray:
        """Format IQ data for machine learning (real/imag channels)"""
        return np.stack([iq_data.real, iq_data.imag], axis=0)
    
    def _compute_spectrogram(self, iq_data: np.ndarray, sample_rate: float) -> np.ndarray:
        """Compute spectrogram of IQ data"""
        f, t, Sxx = signal.spectrogram(iq_data, fs=sample_rate, nperseg=256, noverlap=128)
        
        # Convert to dB and normalize
        Sxx_db = 10 * np.log10(np.abs(Sxx) + 1e-10)
        Sxx_normalized = (Sxx_db - np.min(Sxx_db)) / (np.max(Sxx_db) - np.min(Sxx_db) + 1e-10)
        
        return Sxx_normalized
    
    def _extract_constellation_features(self, iq_data: np.ndarray) -> np.ndarray:
        """Extract constellation diagram features"""
        # Basic constellation features
        features = []
        
        # Amplitude and phase
        amplitude = np.abs(iq_data)
        phase = np.angle(iq_data)
        
        # Statistical moments
        features.extend([
            np.mean(amplitude), np.std(amplitude), np.var(amplitude),
            np.mean(phase), np.std(phase), np.var(phase),
            np.mean(iq_data.real), np.std(iq_data.real),
            np.mean(iq_data.imag), np.std(iq_data.imag)
        ])
        
        # Higher order moments
        features.extend([
            np.mean(amplitude**4) / (np.mean(amplitude**2)**2),  # Kurtosis-like
            np.mean(np.abs(iq_data)**4) / (np.mean(np.abs(iq_data)**2)**2)
        ])
        
        return np.array(features)
    
    def _extract_statistical_features(self, iq_data: np.ndarray) -> np.ndarray:
        """Extract statistical features from IQ data"""
        features = []
        
        # Instantaneous amplitude and phase
        amplitude = np.abs(iq_data)
        phase = np.angle(iq_data)
        
        # Amplitude statistics
        features.extend([
            np.mean(amplitude), np.std(amplitude), np.var(amplitude),
            np.min(amplitude), np.max(amplitude),
            np.percentile(amplitude, 25), np.percentile(amplitude, 75)
        ])
        
        # Phase statistics
        phase_unwrapped = np.unwrap(phase)
        features.extend([
            np.mean(phase_unwrapped), np.std(phase_unwrapped),
            np.mean(np.diff(phase_unwrapped))  # Instantaneous frequency
        ])
        
        # Zero crossing rate
        zero_crossings = np.sum(np.diff(np.sign(iq_data.real)) != 0)
        features.append(zero_crossings / len(iq_data))
        
        return np.array(features)
    
    def _extract_cyclic_features(self, iq_data: np.ndarray) -> np.ndarray:
        """Extract cyclic features (simplified cyclostationary analysis)"""
        # Autocorrelation features
        autocorr = np.correlate(iq_data, iq_data, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # Take first few lags
        max_lags = min(50, len(autocorr))
        autocorr_features = np.abs(autocorr[:max_lags])
        
        # Normalize
        if len(autocorr_features) > 0:
            autocorr_features = autocorr_features / np.max(autocorr_features)
        
        # Pad if necessary
        if len(autocorr_features) < 50:
            autocorr_features = np.pad(autocorr_features, (0, 50 - len(autocorr_features)))
        
        return autocorr_features
    
    def _estimate_snr(self, iq_data: np.ndarray) -> float:
        """Estimate SNR from preprocessed IQ data"""
        # Simple SNR estimation based on signal power variation
        power = np.abs(iq_data) ** 2
        signal_power = np.mean(power)
        
        # Estimate noise power from high-frequency components
        fft_data = fft(iq_data)
        high_freq_power = np.mean(np.abs(fft_data[len(fft_data)//4:3*len(fft_data)//4])**2)
        
        if high_freq_power > 0:
            snr_linear = signal_power / high_freq_power
            return 10 * np.log10(snr_linear)
        return 0.0


class DataAugmentor:
    """Data augmentation for signal samples"""
    
    def __init__(self):
        self.augmentation_methods = [
            'add_noise', 'frequency_shift', 'time_shift', 'amplitude_scale',
            'phase_rotation', 'multipath', 'doppler_shift'
        ]
    
    def augment_samples(self, samples: List[SignalSample], augmentation_factor: int = 2) -> List[SignalSample]:
        """Augment signal samples"""
        augmented_samples = samples.copy()
        
        for _ in range(augmentation_factor - 1):
            for sample in samples:
                augmented_sample = self._apply_random_augmentation(sample)
                augmented_samples.append(augmented_sample)
        
        return augmented_samples
    
    def _apply_random_augmentation(self, sample: SignalSample) -> SignalSample:
        """Apply random augmentation to a sample"""
        method = np.random.choice(self.augmentation_methods)
        iq_data = sample.iq_data.copy()
        
        if method == 'add_noise':
            noise_power = np.random.uniform(0.01, 0.1)
            noise = (np.random.normal(0, np.sqrt(noise_power/2), len(iq_data)) + 
                    1j * np.random.normal(0, np.sqrt(noise_power/2), len(iq_data)))
            iq_data += noise
        
        elif method == 'frequency_shift':
            shift_hz = np.random.uniform(-sample.sample_rate/10, sample.sample_rate/10)
            t = np.arange(len(iq_data)) / sample.sample_rate
            iq_data *= np.exp(1j * 2 * np.pi * shift_hz * t)
        
        elif method == 'time_shift':
            shift_samples = np.random.randint(-len(iq_data)//10, len(iq_data)//10)
            iq_data = np.roll(iq_data, shift_samples)
        
        elif method == 'amplitude_scale':
            scale_factor = np.random.uniform(0.5, 2.0)
            iq_data *= scale_factor
        
        elif method == 'phase_rotation':
            phase_shift = np.random.uniform(0, 2*np.pi)
            iq_data *= np.exp(1j * phase_shift)
        
        # Create augmented sample
        augmented_sample = SignalSample(
            timestamp=sample.timestamp,
            frequency=sample.frequency,
            sample_rate=sample.sample_rate,
            iq_data=iq_data,
            modulation_type=sample.modulation_type,
            snr=sample.snr,
            location=sample.location,
            device_id=sample.device_id,
            metadata={**sample.metadata, 'augmented': True, 'augmentation_method': method}
        )
        
        return augmented_sample