"""
Real-World Signal Processing Pipeline

Implements advanced signal processing capabilities for real-world SDR signals including:
- Feature extraction from IQ samples (constellation diagrams, spectrograms, cyclic features)
- Realistic channel modeling (Rayleigh/Rician fading, AWGN, multipath)
- Adaptive signal processing based on SNR and interference conditions
- Modulation classification for real-world signals with impairments
- Support for wideband signals and multiple simultaneous transmissions
"""
import numpy as np
import scipy.signal
import scipy.fft
import scipy.stats
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import logging
from datetime import datetime
import threading
import queue

from .hardware_abstraction import SignalBuffer

logger = logging.getLogger(__name__)


class ModulationType(Enum):
    """Supported modulation types"""
    BPSK = "bpsk"
    QPSK = "qpsk"
    PSK8 = "8psk"
    QAM16 = "16qam"
    QAM64 = "64qam"
    QAM256 = "256qam"
    FSK2 = "2fsk"
    FSK4 = "4fsk"
    GMSK = "gmsk"
    OFDM = "ofdm"
    AM = "am"
    FM = "fm"
    UNKNOWN = "unknown"


class ChannelType(Enum):
    """Channel model types"""
    AWGN = "awgn"
    RAYLEIGH = "rayleigh"
    RICIAN = "rician"
    MULTIPATH = "multipath"


@dataclass
class FeatureVector:
    """Signal feature vector"""
    timestamp: datetime
    frequency: float
    sample_rate: float
    
    # Spectral features
    power_spectral_density: np.ndarray
    spectral_centroid: float
    spectral_bandwidth: float
    spectral_rolloff: float
    
    # Time domain features
    rms_power: float
    peak_power: float
    papr: float  # Peak-to-Average Power Ratio
    zero_crossing_rate: float
    
    # Statistical features
    mean_amplitude: float
    std_amplitude: float
    skewness: float
    kurtosis: float
    
    # Constellation features
    constellation_points: np.ndarray
    evm: float  # Error Vector Magnitude
    
    # Cyclic features
    cyclic_spectrum: np.ndarray
    cyclic_frequencies: np.ndarray
    
    # Higher-order statistics
    cumulants: np.ndarray
    
    # Modulation-specific features
    instantaneous_frequency: np.ndarray
    instantaneous_phase: np.ndarray
    
    metadata: Dict[str, Any]


@dataclass
class ChannelModel:
    """Channel model parameters"""
    channel_type: ChannelType
    snr_db: float
    fading_rate: float = 0.0  # Hz
    rician_k_factor: float = 0.0  # dB
    multipath_delays: List[float] = None  # seconds
    multipath_gains: List[float] = None  # linear
    doppler_shift: float = 0.0  # Hz


@dataclass
class ClassificationResult:
    """Modulation classification result"""
    predicted_modulation: ModulationType
    confidence: float
    probabilities: Dict[ModulationType, float]
    features_used: List[str]
    processing_time: float
    metadata: Dict[str, Any]


class FeatureExtractor:
    """Advanced feature extraction from IQ samples"""
    
    def __init__(self, sample_rate: float = 1e6):
        self.sample_rate = sample_rate
        self.window_size = 1024
        self.overlap = 0.5
        
    def extract_features(self, iq_samples: np.ndarray, 
                        frequency: float = 0.0) -> FeatureVector:
        """Extract comprehensive features from IQ samples"""
        try:
            # Ensure we have enough samples
            if len(iq_samples) < self.window_size:
                iq_samples = np.pad(iq_samples, (0, self.window_size - len(iq_samples)), 'constant')
            
            # Extract different feature types
            spectral_features = self._extract_spectral_features(iq_samples)
            time_features = self._extract_time_domain_features(iq_samples)
            statistical_features = self._extract_statistical_features(iq_samples)
            constellation_features = self._extract_constellation_features(iq_samples)
            cyclic_features = self._extract_cyclic_features(iq_samples)
            higher_order_features = self._extract_higher_order_features(iq_samples)
            modulation_features = self._extract_modulation_features(iq_samples)
            
            # Combine all features
            feature_vector = FeatureVector(
                timestamp=datetime.now(),
                frequency=frequency,
                sample_rate=self.sample_rate,
                **spectral_features,
                **time_features,
                **statistical_features,
                **constellation_features,
                **cyclic_features,
                **higher_order_features,
                **modulation_features,
                metadata={
                    'num_samples': len(iq_samples),
                    'window_size': self.window_size,
                    'overlap': self.overlap
                }
            )
            
            return feature_vector
            
        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            raise
    
    def _extract_spectral_features(self, iq_samples: np.ndarray) -> Dict[str, Any]:
        """Extract spectral domain features"""
        # Compute power spectral density
        freqs, psd = scipy.signal.welch(iq_samples, fs=self.sample_rate, 
                                       nperseg=min(len(iq_samples), 1024))
        
        # Normalize PSD
        psd = psd / np.sum(psd)
        
        # Spectral centroid
        spectral_centroid = np.sum(freqs * psd) / np.sum(psd)
        
        # Spectral bandwidth
        spectral_bandwidth = np.sqrt(np.sum(((freqs - spectral_centroid) ** 2) * psd) / np.sum(psd))
        
        # Spectral rolloff (95% of energy)
        cumsum_psd = np.cumsum(psd)
        rolloff_idx = np.where(cumsum_psd >= 0.95 * cumsum_psd[-1])[0]
        spectral_rolloff = freqs[rolloff_idx[0]] if len(rolloff_idx) > 0 else freqs[-1]
        
        return {
            'power_spectral_density': psd,
            'spectral_centroid': spectral_centroid,
            'spectral_bandwidth': spectral_bandwidth,
            'spectral_rolloff': spectral_rolloff
        }
    
    def _extract_time_domain_features(self, iq_samples: np.ndarray) -> Dict[str, Any]:
        """Extract time domain features"""
        # Power calculations
        instantaneous_power = np.abs(iq_samples) ** 2
        rms_power = np.sqrt(np.mean(instantaneous_power))
        peak_power = np.max(instantaneous_power)
        papr = 10 * np.log10(peak_power / (rms_power ** 2 + 1e-12))
        
        # Zero crossing rate
        real_part = np.real(iq_samples)
        zero_crossings = np.where(np.diff(np.signbit(real_part)))[0]
        zero_crossing_rate = len(zero_crossings) / len(iq_samples)
        
        return {
            'rms_power': rms_power,
            'peak_power': peak_power,
            'papr': papr,
            'zero_crossing_rate': zero_crossing_rate
        }
    
    def _extract_statistical_features(self, iq_samples: np.ndarray) -> Dict[str, Any]:
        """Extract statistical features"""
        amplitudes = np.abs(iq_samples)
        
        mean_amplitude = np.mean(amplitudes)
        std_amplitude = np.std(amplitudes)
        
        # Higher order moments
        skewness = scipy.stats.skew(amplitudes) if hasattr(scipy, 'stats') else 0.0
        kurtosis = scipy.stats.kurtosis(amplitudes) if hasattr(scipy, 'stats') else 0.0
        
        return {
            'mean_amplitude': mean_amplitude,
            'std_amplitude': std_amplitude,
            'skewness': skewness,
            'kurtosis': kurtosis
        }
    
    def _extract_constellation_features(self, iq_samples: np.ndarray) -> Dict[str, Any]:
        """Extract constellation diagram features"""
        # Normalize samples
        normalized_samples = iq_samples / (np.std(iq_samples) + 1e-12)
        
        # Downsample for constellation analysis
        downsample_factor = max(1, len(iq_samples) // 1000)
        constellation_points = normalized_samples[::downsample_factor]
        
        # Calculate Error Vector Magnitude (simplified)
        # This would normally require knowledge of the ideal constellation
        # For now, we'll estimate based on clustering
        evm = self._estimate_evm(constellation_points)
        
        return {
            'constellation_points': constellation_points,
            'evm': evm
        }
    
    def _extract_cyclic_features(self, iq_samples: np.ndarray) -> Dict[str, Any]:
        """Extract cyclic spectrum features"""
        # Simplified cyclic spectrum calculation
        # In practice, this would use more sophisticated algorithms
        
        # Calculate autocorrelation
        autocorr = np.correlate(iq_samples, iq_samples, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        # FFT of autocorrelation gives power spectral density
        cyclic_spectrum = np.abs(np.fft.fft(autocorr[:min(len(autocorr), 512)]))
        cyclic_frequencies = np.fft.fftfreq(len(cyclic_spectrum), 1/self.sample_rate)
        
        return {
            'cyclic_spectrum': cyclic_spectrum,
            'cyclic_frequencies': cyclic_frequencies
        }
    
    def _extract_higher_order_features(self, iq_samples: np.ndarray) -> Dict[str, Any]:
        """Extract higher-order statistics (cumulants)"""
        # Calculate 2nd and 4th order cumulants
        # These are useful for modulation classification
        
        # 2nd order cumulant (variance)
        c20 = np.var(iq_samples)
        c21 = np.mean(iq_samples * np.conj(iq_samples))
        
        # 4th order cumulants (simplified)
        c40 = np.mean(iq_samples**4) - 3 * (np.mean(iq_samples**2))**2
        c41 = np.mean(np.abs(iq_samples)**2 * iq_samples**2) - np.abs(np.mean(iq_samples**2))**2
        c42 = np.mean(np.abs(iq_samples)**4) - np.abs(np.mean(iq_samples**2))**2 - 2 * (np.mean(np.abs(iq_samples)**2))**2
        
        cumulants = np.array([c20, c21, c40, c41, c42], dtype=complex)
        
        return {
            'cumulants': cumulants
        }
    
    def _extract_modulation_features(self, iq_samples: np.ndarray) -> Dict[str, Any]:
        """Extract modulation-specific features"""
        # Instantaneous frequency and phase
        analytic_signal = scipy.signal.hilbert(np.real(iq_samples))
        instantaneous_phase = np.unwrap(np.angle(analytic_signal))
        instantaneous_frequency = np.diff(instantaneous_phase) * self.sample_rate / (2 * np.pi)
        
        # Pad to match original length
        instantaneous_frequency = np.pad(instantaneous_frequency, (0, 1), 'edge')
        
        return {
            'instantaneous_frequency': instantaneous_frequency,
            'instantaneous_phase': instantaneous_phase
        }
    
    def _estimate_evm(self, constellation_points: np.ndarray) -> float:
        """Estimate Error Vector Magnitude"""
        # Simplified EVM calculation
        # In practice, this would require knowledge of the ideal constellation
        
        # Use k-means-like approach to find constellation centers
        try:
            from sklearn.cluster import KMeans
            
            # Try different numbers of clusters
            best_evm = float('inf')
            for n_clusters in [2, 4, 8, 16]:
                if len(constellation_points) < n_clusters:
                    continue
                    
                # Prepare data for clustering
                data = np.column_stack([np.real(constellation_points), 
                                      np.imag(constellation_points)])
                
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                labels = kmeans.fit_predict(data)
                centers = kmeans.cluster_centers_
                
                # Calculate EVM
                evm = 0.0
                for i, point in enumerate(constellation_points):
                    center_idx = labels[i]
                    center = centers[center_idx][0] + 1j * centers[center_idx][1]
                    error = np.abs(point - center)
                    evm += error ** 2
                
                evm = np.sqrt(evm / len(constellation_points))
                
                if evm < best_evm:
                    best_evm = evm
            
            return best_evm
            
        except ImportError:
            # Fallback: use standard deviation as rough EVM estimate
            return np.std(np.abs(constellation_points))


class ChannelSimulator:
    """Realistic channel modeling and simulation"""
    
    def __init__(self):
        self.rng = np.random.RandomState(42)
        
    def apply_channel_model(self, iq_samples: np.ndarray, 
                          channel_model: ChannelModel) -> np.ndarray:
        """Apply realistic channel effects to IQ samples"""
        try:
            # Start with original samples
            output_samples = iq_samples.copy()
            
            # Apply channel-specific effects
            if channel_model.channel_type == ChannelType.AWGN:
                output_samples = self._apply_awgn(output_samples, channel_model.snr_db)
                
            elif channel_model.channel_type == ChannelType.RAYLEIGH:
                output_samples = self._apply_rayleigh_fading(output_samples, channel_model)
                output_samples = self._apply_awgn(output_samples, channel_model.snr_db)
                
            elif channel_model.channel_type == ChannelType.RICIAN:
                output_samples = self._apply_rician_fading(output_samples, channel_model)
                output_samples = self._apply_awgn(output_samples, channel_model.snr_db)
                
            elif channel_model.channel_type == ChannelType.MULTIPATH:
                output_samples = self._apply_multipath(output_samples, channel_model)
                output_samples = self._apply_awgn(output_samples, channel_model.snr_db)
            
            # Apply Doppler shift if specified
            if channel_model.doppler_shift != 0.0:
                output_samples = self._apply_doppler_shift(output_samples, channel_model.doppler_shift)
            
            return output_samples
            
        except Exception as e:
            logger.error(f"Error applying channel model: {e}")
            return iq_samples
    
    def _apply_awgn(self, samples: np.ndarray, snr_db: float) -> np.ndarray:
        """Apply Additive White Gaussian Noise"""
        # Calculate signal power
        signal_power = np.mean(np.abs(samples) ** 2)
        
        # Calculate noise power from SNR
        snr_linear = 10 ** (snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # Generate complex Gaussian noise
        noise_std = np.sqrt(noise_power / 2)  # Divide by 2 for complex noise
        noise = noise_std * (self.rng.randn(len(samples)) + 1j * self.rng.randn(len(samples)))
        
        return samples + noise
    
    def _apply_rayleigh_fading(self, samples: np.ndarray, 
                             channel_model: ChannelModel) -> np.ndarray:
        """Apply Rayleigh fading"""
        # Generate Rayleigh fading coefficients
        num_samples = len(samples)
        
        # Generate complex Gaussian random variables
        h_real = self.rng.randn(num_samples)
        h_imag = self.rng.randn(num_samples)
        h = (h_real + 1j * h_imag) / np.sqrt(2)
        
        # Apply fading rate (simplified model)
        if channel_model.fading_rate > 0:
            # Low-pass filter to control fading rate
            cutoff = channel_model.fading_rate / (2 * np.pi)
            b, a = scipy.signal.butter(2, cutoff, btype='low', fs=1.0)
            h = scipy.signal.filtfilt(b, a, h)
        
        return samples * h
    
    def _apply_rician_fading(self, samples: np.ndarray, 
                           channel_model: ChannelModel) -> np.ndarray:
        """Apply Rician fading"""
        # Rician fading = LOS component + Rayleigh component
        k_linear = 10 ** (channel_model.rician_k_factor / 10)
        
        # LOS component (constant)
        los_component = np.sqrt(k_linear / (k_linear + 1))
        
        # Rayleigh component
        rayleigh_component = np.sqrt(1 / (k_linear + 1))
        h_real = self.rng.randn(len(samples))
        h_imag = self.rng.randn(len(samples))
        rayleigh_h = (h_real + 1j * h_imag) / np.sqrt(2)
        
        # Combine components
        h = los_component + rayleigh_component * rayleigh_h
        
        return samples * h
    
    def _apply_multipath(self, samples: np.ndarray, 
                        channel_model: ChannelModel) -> np.ndarray:
        """Apply multipath channel"""
        if not channel_model.multipath_delays or not channel_model.multipath_gains:
            return samples
        
        # Convert delays to sample delays
        sample_rate = 1e6  # Assume 1 MHz sample rate
        sample_delays = [int(delay * sample_rate) for delay in channel_model.multipath_delays]
        
        # Apply multipath
        output = np.zeros_like(samples)
        
        for delay, gain in zip(sample_delays, channel_model.multipath_gains):
            if delay < len(samples):
                # Add delayed and scaled version
                delayed_samples = np.zeros_like(samples)
                delayed_samples[delay:] = samples[:-delay] if delay > 0 else samples
                output += gain * delayed_samples
        
        return output
    
    def _apply_doppler_shift(self, samples: np.ndarray, doppler_hz: float) -> np.ndarray:
        """Apply Doppler frequency shift"""
        sample_rate = 1e6  # Assume 1 MHz sample rate
        t = np.arange(len(samples)) / sample_rate
        
        # Apply frequency shift
        shift_factor = np.exp(1j * 2 * np.pi * doppler_hz * t)
        
        return samples * shift_factor


class AdaptiveSignalProcessor:
    """Adaptive signal processing based on real-time conditions"""
    
    def __init__(self, sample_rate: float = 1e6):
        self.sample_rate = sample_rate
        self.feature_extractor = FeatureExtractor(sample_rate)
        self.channel_simulator = ChannelSimulator()
        
        # Adaptive parameters
        self.snr_threshold_low = -10.0  # dB
        self.snr_threshold_high = 20.0  # dB
        self.interference_threshold = 0.1
        
        # Processing history
        self.processing_history = []
        self.max_history = 100
        
    def process_signal_adaptive(self, signal_buffer: SignalBuffer) -> Tuple[FeatureVector, Dict[str, Any]]:
        """Process signal with adaptive parameters based on conditions"""
        try:
            # Extract initial features to assess signal quality
            features = self.feature_extractor.extract_features(
                signal_buffer.iq_samples, 
                signal_buffer.frequency
            )
            
            # Estimate SNR and interference
            snr_estimate = self._estimate_snr(signal_buffer.iq_samples)
            interference_level = self._estimate_interference(signal_buffer.iq_samples)
            
            # Adapt processing parameters
            processing_params = self._adapt_processing_parameters(
                snr_estimate, interference_level
            )
            
            # Apply adaptive processing
            processed_samples = self._apply_adaptive_processing(
                signal_buffer.iq_samples, processing_params
            )
            
            # Re-extract features from processed signal
            if not np.array_equal(processed_samples, signal_buffer.iq_samples):
                processed_buffer = SignalBuffer(
                    iq_samples=processed_samples,
                    timestamp=signal_buffer.timestamp,
                    frequency=signal_buffer.frequency,
                    sample_rate=signal_buffer.sample_rate,
                    gain=signal_buffer.gain,
                    metadata=signal_buffer.metadata
                )
                features = self.feature_extractor.extract_features(
                    processed_buffer.iq_samples,
                    processed_buffer.frequency
                )
            
            # Update processing history
            self._update_processing_history(snr_estimate, interference_level, processing_params)
            
            # Create processing metadata
            processing_metadata = {
                'snr_estimate': snr_estimate,
                'interference_level': interference_level,
                'processing_params': processing_params,
                'samples_processed': len(processed_samples),
                'processing_applied': not np.array_equal(processed_samples, signal_buffer.iq_samples)
            }
            
            return features, processing_metadata
            
        except Exception as e:
            logger.error(f"Error in adaptive signal processing: {e}")
            # Fallback to basic processing
            features = self.feature_extractor.extract_features(
                signal_buffer.iq_samples,
                signal_buffer.frequency
            )
            return features, {'error': str(e)}
    
    def _estimate_snr(self, iq_samples: np.ndarray) -> float:
        """Estimate signal-to-noise ratio"""
        # Simple SNR estimation based on signal statistics
        signal_power = np.mean(np.abs(iq_samples) ** 2)
        
        # Estimate noise power from high-frequency components
        fft_samples = np.fft.fft(iq_samples)
        # Assume noise dominates in the outer 20% of the spectrum
        noise_bins = int(0.2 * len(fft_samples))
        noise_power = np.mean(np.abs(fft_samples[-noise_bins:]) ** 2)
        
        if noise_power > 0:
            snr_linear = signal_power / noise_power
            return 10 * np.log10(snr_linear)
        else:
            return 50.0  # Very high SNR
    
    def _estimate_interference(self, iq_samples: np.ndarray) -> float:
        """Estimate interference level"""
        # Estimate interference based on spectral characteristics
        freqs, psd = scipy.signal.welch(iq_samples, fs=self.sample_rate)
        
        # Look for peaks that might indicate interference
        peak_indices, _ = scipy.signal.find_peaks(psd, height=np.mean(psd) * 2)
        
        # Interference level based on number and strength of peaks
        interference_level = len(peak_indices) / len(psd)
        
        return min(interference_level, 1.0)
    
    def _adapt_processing_parameters(self, snr_db: float, 
                                   interference_level: float) -> Dict[str, Any]:
        """Adapt processing parameters based on signal conditions"""
        params = {
            'apply_filtering': False,
            'filter_type': 'none',
            'filter_params': {},
            'apply_equalization': False,
            'equalization_params': {},
            'apply_noise_reduction': False,
            'noise_reduction_params': {}
        }
        
        # Low SNR conditions
        if snr_db < self.snr_threshold_low:
            params['apply_noise_reduction'] = True
            params['noise_reduction_params'] = {
                'method': 'wiener',
                'noise_variance': 10 ** (-snr_db / 10)
            }
        
        # High interference conditions
        if interference_level > self.interference_threshold:
            params['apply_filtering'] = True
            params['filter_type'] = 'notch'
            params['filter_params'] = {
                'quality_factor': 30,
                'adaptive': True
            }
        
        # High SNR conditions - can use more aggressive processing
        if snr_db > self.snr_threshold_high:
            params['apply_equalization'] = True
            params['equalization_params'] = {
                'method': 'adaptive',
                'num_taps': 11
            }
        
        return params
    
    def _apply_adaptive_processing(self, iq_samples: np.ndarray, 
                                 params: Dict[str, Any]) -> np.ndarray:
        """Apply adaptive processing based on parameters"""
        processed_samples = iq_samples.copy()
        
        # Apply noise reduction
        if params['apply_noise_reduction']:
            processed_samples = self._apply_noise_reduction(
                processed_samples, params['noise_reduction_params']
            )
        
        # Apply filtering
        if params['apply_filtering']:
            processed_samples = self._apply_adaptive_filtering(
                processed_samples, params['filter_type'], params['filter_params']
            )
        
        # Apply equalization
        if params['apply_equalization']:
            processed_samples = self._apply_equalization(
                processed_samples, params['equalization_params']
            )
        
        return processed_samples
    
    def _apply_noise_reduction(self, samples: np.ndarray, 
                             params: Dict[str, Any]) -> np.ndarray:
        """Apply noise reduction"""
        if params['method'] == 'wiener':
            # Simple Wiener filtering
            noise_var = params['noise_variance']
            signal_var = np.var(samples)
            
            # Wiener filter coefficient
            wiener_coeff = signal_var / (signal_var + noise_var)
            
            return samples * wiener_coeff
        
        return samples
    
    def _apply_adaptive_filtering(self, samples: np.ndarray, 
                                filter_type: str, params: Dict[str, Any]) -> np.ndarray:
        """Apply adaptive filtering"""
        if filter_type == 'notch':
            # Apply notch filtering to remove interference
            # This is a simplified implementation
            
            # Find interference frequencies
            freqs, psd = scipy.signal.welch(samples, fs=self.sample_rate)
            peak_indices, _ = scipy.signal.find_peaks(psd, height=np.mean(psd) * 2)
            
            filtered_samples = samples.copy()
            
            for peak_idx in peak_indices:
                # Create notch filter for this frequency
                notch_freq = freqs[peak_idx]
                quality_factor = params.get('quality_factor', 30)
                
                # Design notch filter
                b, a = scipy.signal.iirnotch(notch_freq, quality_factor, fs=self.sample_rate)
                
                # Apply filter
                filtered_samples = scipy.signal.filtfilt(b, a, filtered_samples)
            
            return filtered_samples
        
        return samples
    
    def _apply_equalization(self, samples: np.ndarray, 
                          params: Dict[str, Any]) -> np.ndarray:
        """Apply channel equalization"""
        # Simplified equalization - in practice would use more sophisticated algorithms
        if params['method'] == 'adaptive':
            # Simple adaptive equalization using LMS-like approach
            num_taps = params.get('num_taps', 11)
            
            # Create simple FIR equalizer
            equalizer_taps = np.zeros(num_taps, dtype=complex)
            equalizer_taps[num_taps // 2] = 1.0  # Initialize as pass-through
            
            # Apply equalization (simplified)
            equalized = scipy.signal.lfilter(equalizer_taps, 1, samples)
            
            return equalized
        
        return samples
    
    def _update_processing_history(self, snr: float, interference: float, 
                                 params: Dict[str, Any]):
        """Update processing history for adaptation"""
        history_entry = {
            'timestamp': datetime.now(),
            'snr': snr,
            'interference': interference,
            'params': params.copy()
        }
        
        self.processing_history.append(history_entry)
        
        # Limit history size
        if len(self.processing_history) > self.max_history:
            self.processing_history = self.processing_history[-self.max_history:]


class ModulationClassifier:
    """Real-world modulation classification with impairments"""
    
    def __init__(self):
        self.feature_extractor = FeatureExtractor()
        
        # Classification thresholds (these would be learned from training data)
        self.classification_thresholds = {
            'cumulant_c20': {'bpsk': 1.0, 'qpsk': 1.0, '8psk': 1.0},
            'cumulant_c21': {'bpsk': 1.0, 'qpsk': 0.0, '8psk': 0.0},
            'cumulant_c40': {'bpsk': -2.0, 'qpsk': -1.0, '8psk': 0.0},
            'papr_threshold': {'ofdm': 8.0, 'single_carrier': 4.0}
        }
    
    def classify_modulation(self, iq_samples: np.ndarray, 
                          frequency: float = 0.0) -> ClassificationResult:
        """Classify modulation type of real-world signals with impairments"""
        try:
            start_time = datetime.now()
            
            # Extract features
            features = self.feature_extractor.extract_features(iq_samples, frequency)
            
            # Apply classification rules
            probabilities = self._calculate_modulation_probabilities(features)
            
            # Find most likely modulation
            predicted_modulation = max(probabilities.keys(), key=lambda k: probabilities[k])
            confidence = probabilities[predicted_modulation]
            
            # Determine which features were most important
            features_used = self._get_important_features(features, predicted_modulation)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            result = ClassificationResult(
                predicted_modulation=predicted_modulation,
                confidence=confidence,
                probabilities=probabilities,
                features_used=features_used,
                processing_time=processing_time,
                metadata={
                    'num_samples': len(iq_samples),
                    'frequency': frequency,
                    'snr_estimate': self._estimate_snr_from_features(features)
                }
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error in modulation classification: {e}")
            return ClassificationResult(
                predicted_modulation=ModulationType.UNKNOWN,
                confidence=0.0,
                probabilities={ModulationType.UNKNOWN: 1.0},
                features_used=[],
                processing_time=0.0,
                metadata={'error': str(e)}
            )
    
    def _calculate_modulation_probabilities(self, features: FeatureVector) -> Dict[ModulationType, float]:
        """Calculate probabilities for each modulation type"""
        probabilities = {}
        
        # Extract relevant cumulants
        c20 = np.abs(features.cumulants[0]) if len(features.cumulants) > 0 else 0
        c21 = np.abs(features.cumulants[1]) if len(features.cumulants) > 1 else 0
        c40 = np.real(features.cumulants[2]) if len(features.cumulants) > 2 else 0
        
        # BPSK classification
        bpsk_score = 0.0
        if np.abs(c21 - 1.0) < 0.3:  # c21 ≈ 1 for BPSK
            bpsk_score += 0.4
        if c40 < -1.5:  # c40 ≈ -2 for BPSK
            bpsk_score += 0.3
        if features.papr < 3.0:  # Low PAPR for BPSK
            bpsk_score += 0.3
        probabilities[ModulationType.BPSK] = bpsk_score
        
        # QPSK classification
        qpsk_score = 0.0
        if np.abs(c21) < 0.3:  # c21 ≈ 0 for QPSK
            qpsk_score += 0.4
        if -1.5 < c40 < -0.5:  # c40 ≈ -1 for QPSK
            qpsk_score += 0.3
        if 2.0 < features.papr < 5.0:  # Moderate PAPR for QPSK
            qpsk_score += 0.3
        probabilities[ModulationType.QPSK] = qpsk_score
        
        # 8PSK classification
        psk8_score = 0.0
        if np.abs(c21) < 0.3:  # c21 ≈ 0 for 8PSK
            psk8_score += 0.3
        if c40 > -0.5:  # c40 ≈ 0 for 8PSK
            psk8_score += 0.3
        if 3.0 < features.papr < 6.0:  # Higher PAPR for 8PSK
            psk8_score += 0.4
        probabilities[ModulationType.PSK8] = psk8_score
        
        # QAM classification (simplified)
        qam16_score = 0.0
        if features.evm > 0.1:  # Higher EVM for QAM
            qam16_score += 0.2
        if 4.0 < features.papr < 8.0:  # Higher PAPR for QAM
            qam16_score += 0.3
        if features.spectral_bandwidth > features.sample_rate * 0.3:  # Wider bandwidth
            qam16_score += 0.2
        probabilities[ModulationType.QAM16] = qam16_score
        
        # OFDM classification
        ofdm_score = 0.0
        if features.papr > 8.0:  # High PAPR characteristic of OFDM
            ofdm_score += 0.5
        if features.spectral_bandwidth > features.sample_rate * 0.4:  # Wide bandwidth
            ofdm_score += 0.3
        # Check for multiple peaks in spectrum (subcarriers)
        psd_peaks, _ = scipy.signal.find_peaks(features.power_spectral_density)
        if len(psd_peaks) > 10:  # Multiple subcarriers
            ofdm_score += 0.2
        probabilities[ModulationType.OFDM] = ofdm_score
        
        # FM classification
        fm_score = 0.0
        if np.std(features.instantaneous_frequency) > features.sample_rate * 0.01:  # Frequency variation
            fm_score += 0.4
        if features.papr < 4.0:  # Constant envelope
            fm_score += 0.3
        probabilities[ModulationType.FM] = fm_score
        
        # AM classification
        am_score = 0.0
        if np.std(np.abs(features.constellation_points)) > 0.3:  # Amplitude variation
            am_score += 0.4
        if features.papr > 3.0:  # Variable envelope
            am_score += 0.3
        probabilities[ModulationType.AM] = am_score
        
        # Normalize probabilities
        total_score = sum(probabilities.values())
        if total_score > 0:
            probabilities = {k: v / total_score for k, v in probabilities.items()}
        else:
            # If no clear classification, assign equal probabilities
            num_types = len(probabilities)
            probabilities = {k: 1.0 / num_types for k in probabilities.keys()}
        
        # Add unknown category
        max_prob = max(probabilities.values())
        if max_prob < 0.5:  # Low confidence
            probabilities[ModulationType.UNKNOWN] = 1.0 - max_prob
        else:
            probabilities[ModulationType.UNKNOWN] = 0.1
        
        return probabilities
    
    def _get_important_features(self, features: FeatureVector, 
                              modulation: ModulationType) -> List[str]:
        """Get list of features that were important for classification"""
        important_features = []
        
        if modulation in [ModulationType.BPSK, ModulationType.QPSK, ModulationType.PSK8]:
            important_features.extend(['cumulants', 'papr', 'constellation_points'])
        
        if modulation in [ModulationType.QAM16, ModulationType.QAM64]:
            important_features.extend(['evm', 'papr', 'constellation_points'])
        
        if modulation == ModulationType.OFDM:
            important_features.extend(['papr', 'power_spectral_density', 'spectral_bandwidth'])
        
        if modulation == ModulationType.FM:
            important_features.extend(['instantaneous_frequency', 'papr'])
        
        if modulation == ModulationType.AM:
            important_features.extend(['constellation_points', 'papr'])
        
        return important_features
    
    def _estimate_snr_from_features(self, features: FeatureVector) -> float:
        """Estimate SNR from extracted features"""
        # Simple SNR estimation based on constellation tightness
        if len(features.constellation_points) > 0:
            constellation_std = np.std(np.abs(features.constellation_points))
            # Convert to approximate SNR (this is a rough estimate)
            snr_estimate = -20 * np.log10(constellation_std + 1e-12)
            return np.clip(snr_estimate, -20, 50)
        
        return 0.0


class WidebandProcessor:
    """Support for wideband signals and multiple simultaneous transmissions"""
    
    def __init__(self, sample_rate: float = 20e6):  # 20 MHz for wideband
        self.sample_rate = sample_rate
        self.feature_extractor = FeatureExtractor(sample_rate)
        self.modulation_classifier = ModulationClassifier()
        
        # Channelization parameters
        self.channel_bandwidth = 1e6  # 1 MHz channels
        self.num_channels = int(sample_rate / self.channel_bandwidth)
        
    def process_wideband_signal(self, iq_samples: np.ndarray, 
                              center_frequency: float) -> Dict[str, Any]:
        """Process wideband signal and detect multiple transmissions"""
        try:
            # Channelize the wideband signal
            channels = self._channelize_signal(iq_samples)
            
            # Process each channel
            channel_results = {}
            active_channels = []
            
            for channel_idx, channel_samples in channels.items():
                # Calculate channel center frequency
                channel_freq = (center_frequency - self.sample_rate/2 + 
                              (channel_idx + 0.5) * self.channel_bandwidth)
                
                # Detect if channel is active
                if self._is_channel_active(channel_samples):
                    active_channels.append(channel_idx)
                    
                    # Extract features for active channel
                    features = self.feature_extractor.extract_features(
                        channel_samples, channel_freq
                    )
                    
                    # Classify modulation
                    classification = self.modulation_classifier.classify_modulation(
                        channel_samples, channel_freq
                    )
                    
                    channel_results[channel_idx] = {
                        'frequency': channel_freq,
                        'active': True,
                        'features': features,
                        'classification': classification,
                        'power_db': 10 * np.log10(np.mean(np.abs(channel_samples)**2) + 1e-12)
                    }
                else:
                    channel_results[channel_idx] = {
                        'frequency': channel_freq,
                        'active': False,
                        'power_db': -80.0  # Very low power
                    }
            
            # Analyze simultaneous transmissions
            simultaneous_analysis = self._analyze_simultaneous_transmissions(
                channel_results, active_channels
            )
            
            return {
                'total_channels': self.num_channels,
                'active_channels': active_channels,
                'channel_results': channel_results,
                'simultaneous_analysis': simultaneous_analysis,
                'processing_metadata': {
                    'sample_rate': self.sample_rate,
                    'channel_bandwidth': self.channel_bandwidth,
                    'center_frequency': center_frequency,
                    'num_samples': len(iq_samples)
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing wideband signal: {e}")
            return {'error': str(e)}
    
    def _channelize_signal(self, iq_samples: np.ndarray) -> Dict[int, np.ndarray]:
        """Split wideband signal into individual channels"""
        channels = {}
        
        # Use FFT-based channelization
        fft_size = 1024
        overlap = 0.5
        hop_size = int(fft_size * (1 - overlap))
        
        # Calculate decimation factor for each channel
        decimation_factor = int(self.sample_rate / self.channel_bandwidth)
        
        for channel_idx in range(self.num_channels):
            # Calculate frequency shift for this channel
            freq_shift = (channel_idx - self.num_channels/2 + 0.5) * self.channel_bandwidth
            
            # Apply frequency shift
            t = np.arange(len(iq_samples)) / self.sample_rate
            shift_factor = np.exp(-1j * 2 * np.pi * freq_shift * t)
            shifted_samples = iq_samples * shift_factor
            
            # Low-pass filter and decimate
            # Design anti-aliasing filter
            nyquist = self.sample_rate / 2
            cutoff = self.channel_bandwidth / 2
            b, a = scipy.signal.butter(6, cutoff / nyquist, btype='low')
            
            # Apply filter
            filtered_samples = scipy.signal.filtfilt(b, a, shifted_samples)
            
            # Decimate
            decimated_samples = filtered_samples[::decimation_factor]
            
            channels[channel_idx] = decimated_samples
        
        return channels
    
    def _is_channel_active(self, channel_samples: np.ndarray, 
                          threshold_db: float = -60.0) -> bool:
        """Determine if a channel contains active signal"""
        # Calculate average power
        power = np.mean(np.abs(channel_samples) ** 2)
        power_db = 10 * np.log10(power + 1e-12)
        
        return power_db > threshold_db
    
    def _analyze_simultaneous_transmissions(self, channel_results: Dict[int, Any], 
                                          active_channels: List[int]) -> Dict[str, Any]:
        """Analyze characteristics of simultaneous transmissions"""
        analysis = {
            'num_simultaneous': len(active_channels),
            'frequency_separation': [],
            'power_differences': [],
            'modulation_diversity': [],
            'interference_analysis': {}
        }
        
        if len(active_channels) < 2:
            return analysis
        
        # Calculate frequency separations
        active_freqs = [channel_results[ch]['frequency'] for ch in active_channels]
        for i in range(len(active_freqs)):
            for j in range(i+1, len(active_freqs)):
                separation = abs(active_freqs[i] - active_freqs[j])
                analysis['frequency_separation'].append(separation)
        
        # Calculate power differences
        active_powers = [channel_results[ch]['power_db'] for ch in active_channels]
        for i in range(len(active_powers)):
            for j in range(i+1, len(active_powers)):
                power_diff = abs(active_powers[i] - active_powers[j])
                analysis['power_differences'].append(power_diff)
        
        # Analyze modulation diversity
        modulations = []
        for ch in active_channels:
            if 'classification' in channel_results[ch]:
                mod_type = channel_results[ch]['classification'].predicted_modulation
                modulations.append(mod_type)
        
        analysis['modulation_diversity'] = list(set(modulations))
        
        # Simple interference analysis
        min_separation = min(analysis['frequency_separation']) if analysis['frequency_separation'] else float('inf')
        max_power_diff = max(analysis['power_differences']) if analysis['power_differences'] else 0
        
        analysis['interference_analysis'] = {
            'potential_adjacent_channel_interference': min_separation < self.channel_bandwidth * 1.5,
            'potential_near_far_problem': max_power_diff > 20.0,  # >20dB difference
            'spectral_efficiency': len(active_channels) / self.num_channels
        }
        
        return analysis