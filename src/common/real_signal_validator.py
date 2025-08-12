"""
Real-world signal validation with RF-specific quality metrics
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Optional, Union
from dataclasses import dataclass, asdict
from scipy import signal, stats
from scipy.fft import fft, fftfreq, fftshift
import matplotlib.pyplot as plt
from datetime import datetime

from .interfaces import SignalSample
from .signal_models import EnhancedSignalSample, SignalQualityMetrics, ModulationType


@dataclass
class RFQualityThresholds:
    """RF-specific quality thresholds for real-world signals"""
    # SNR thresholds (more realistic for RF)
    min_snr_db: float = -30.0
    max_snr_db: float = 60.0
    
    # EVM thresholds by modulation type
    evm_thresholds: Dict[str, float] = None
    
    # PAPR thresholds
    max_papr_db: float = 15.0
    min_papr_db: float = 0.5
    
    # IQ imbalance thresholds
    max_amplitude_imbalance_db: float = 3.0
    max_phase_imbalance_deg: float = 10.0
    
    # Frequency domain thresholds
    max_spurious_power_db: float = -40.0
    min_occupied_bandwidth_ratio: float = 0.05
    max_occupied_bandwidth_ratio: float = 0.95
    
    # Phase noise thresholds
    max_phase_noise_power: float = 0.1
    
    # DC offset thresholds
    max_dc_offset_ratio: float = 0.1
    
    # Spectral characteristics
    max_spectral_flatness: float = 0.95
    min_spectral_flatness: float = 0.01
    
    def __post_init__(self):
        if self.evm_thresholds is None:
            self.evm_thresholds = {
                'BPSK': 15.0,    # %
                'QPSK': 12.0,
                '8PSK': 10.0,
                'QAM16': 8.0,
                'QAM64': 5.0,
                'QAM256': 3.0,
                'AM-DSB': 20.0,
                'AM-SSB': 15.0,
                'FM': 25.0,
                'WBFM': 30.0,
                'FSK': 15.0,
                'GFSK': 12.0,
                'CPFSK': 10.0,
                'PAM4': 10.0,
                'OFDM': 8.0
            }


class RealSignalValidator:
    """Comprehensive validator for real-world RF signals"""
    
    def __init__(self, thresholds: RFQualityThresholds = None):
        self.thresholds = thresholds or RFQualityThresholds()
    
    def validate_enhanced_sample(self, sample: EnhancedSignalSample) -> Dict[str, Any]:
        """Validate enhanced signal sample with RF-specific metrics"""
        validation_results = {
            'overall_valid': True,
            'quality_score': 0.0,
            'rf_metrics': {},
            'issues': [],
            'warnings': [],
            'recommendations': []
        }
        
        # Basic validation
        basic_score = self._validate_basic_properties(sample, validation_results)
        
        # RF-specific validation
        rf_score = self._validate_rf_characteristics(sample, validation_results)
        
        # Modulation-specific validation
        mod_score = self._validate_modulation_specific(sample, validation_results)
        
        # Hardware-specific validation
        hw_score = self._validate_hardware_characteristics(sample, validation_results)
        
        # Environmental validation
        env_score = self._validate_environmental_factors(sample, validation_results)
        
        # Calculate overall score
        scores = [basic_score, rf_score, mod_score, hw_score, env_score]
        validation_results['quality_score'] = np.mean([s for s in scores if s is not None])
        
        # Determine overall validity
        validation_results['overall_valid'] = (
            validation_results['quality_score'] > 0.3 and
            len([issue for issue in validation_results['issues'] if 'critical' in issue.lower()]) == 0
        )
        
        return validation_results
    
    def _validate_basic_properties(self, sample: EnhancedSignalSample, results: Dict[str, Any]) -> float:
        """Validate basic signal properties"""
        score = 1.0
        
        # Check IQ data validity
        if np.any(np.isnan(sample.iq_data)) or np.any(np.isinf(sample.iq_data)):
            results['issues'].append("CRITICAL: Invalid IQ data (NaN or Inf values)")
            return 0.0
        
        if len(sample.iq_data) == 0:
            results['issues'].append("CRITICAL: Empty IQ data")
            return 0.0
        
        # Check sample length
        if len(sample.iq_data) < 64:
            results['issues'].append("WARNING: Very short signal sample")
            results['recommendations'].append("Increase capture duration for better analysis")
            score *= 0.7
        
        # Check dynamic range
        amplitude = np.abs(sample.iq_data)
        max_amp = np.max(amplitude)
        min_amp = np.min(amplitude[amplitude > 0]) if np.any(amplitude > 0) else 0
        
        if max_amp == 0:
            results['issues'].append("CRITICAL: Zero amplitude signal")
            return 0.0
        
        if min_amp > 0:
            dynamic_range_db = 20 * np.log10(max_amp / min_amp)
            if dynamic_range_db > 80:
                results['warnings'].append(f"Very high dynamic range: {dynamic_range_db:.1f} dB")
            elif dynamic_range_db < 20:
                results['warnings'].append(f"Low dynamic range: {dynamic_range_db:.1f} dB")
                score *= 0.8
        
        return score
    
    def _validate_rf_characteristics(self, sample: EnhancedSignalSample, results: Dict[str, Any]) -> float:
        """Validate RF-specific characteristics"""
        score = 1.0
        rf_metrics = {}
        
        # SNR validation
        snr = sample.quality_metrics.snr_db
        if snr < self.thresholds.min_snr_db:
            results['issues'].append(f"Low SNR: {snr:.1f} dB")
            results['recommendations'].append("Improve signal strength or reduce noise")
            score *= 0.5
        elif snr > self.thresholds.max_snr_db:
            results['warnings'].append(f"Unusually high SNR: {snr:.1f} dB")
            results['recommendations'].append("Check for signal saturation")
        
        rf_metrics['snr_db'] = snr
        
        # PAPR validation
        papr = sample.quality_metrics.papr_db
        if papr is not None:
            if papr > self.thresholds.max_papr_db:
                results['warnings'].append(f"High PAPR: {papr:.1f} dB")
                results['recommendations'].append("Check for signal clipping or distortion")
                score *= 0.8
            elif papr < self.thresholds.min_papr_db:
                results['warnings'].append(f"Very low PAPR: {papr:.1f} dB (possible CW signal)")
            
            rf_metrics['papr_db'] = papr
        
        # DC offset validation
        dc_offset = np.abs(np.mean(sample.iq_data))
        signal_rms = np.sqrt(np.mean(np.abs(sample.iq_data) ** 2))
        dc_ratio = dc_offset / (signal_rms + 1e-10)
        
        if dc_ratio > self.thresholds.max_dc_offset_ratio:
            results['issues'].append(f"Significant DC offset: {dc_ratio:.3f}")
            results['recommendations'].append("Apply DC offset correction")
            score *= 0.7
        
        rf_metrics['dc_offset_ratio'] = dc_ratio
        
        # IQ imbalance validation
        iq_imbalance = self._calculate_iq_imbalance(sample.iq_data)
        if iq_imbalance['amplitude_imbalance_db'] > self.thresholds.max_amplitude_imbalance_db:
            results['warnings'].append(f"IQ amplitude imbalance: {iq_imbalance['amplitude_imbalance_db']:.2f} dB")
            results['recommendations'].append("Apply IQ imbalance correction")
            score *= 0.9
        
        if iq_imbalance['phase_imbalance_deg'] > self.thresholds.max_phase_imbalance_deg:
            results['warnings'].append(f"IQ phase imbalance: {iq_imbalance['phase_imbalance_deg']:.1f} degrees")
            results['recommendations'].append("Apply IQ imbalance correction")
            score *= 0.9
        
        rf_metrics['iq_imbalance'] = iq_imbalance
        
        # Spectral analysis
        spectral_metrics = self._analyze_spectrum(sample.iq_data, sample.rf_params.sample_rate)
        
        # Occupied bandwidth validation
        occupied_bw_ratio = spectral_metrics['occupied_bandwidth'] / sample.rf_params.sample_rate
        if occupied_bw_ratio < self.thresholds.min_occupied_bandwidth_ratio:
            results['warnings'].append(f"Very narrow occupied bandwidth: {occupied_bw_ratio:.3f}")
        elif occupied_bw_ratio > self.thresholds.max_occupied_bandwidth_ratio:
            results['warnings'].append(f"Very wide occupied bandwidth: {occupied_bw_ratio:.3f}")
            results['recommendations'].append("Check for aliasing or interference")
            score *= 0.8
        
        # Spurious signals validation
        if spectral_metrics['max_spurious_power_db'] > self.thresholds.max_spurious_power_db:
            results['warnings'].append(f"High spurious signals: {spectral_metrics['max_spurious_power_db']:.1f} dB")
            results['recommendations'].append("Check for interference or hardware issues")
            score *= 0.8
        
        rf_metrics.update(spectral_metrics)
        results['rf_metrics'] = rf_metrics
        
        return score
    
    def _validate_modulation_specific(self, sample: EnhancedSignalSample, results: Dict[str, Any]) -> float:
        """Validate modulation-specific characteristics"""
        score = 1.0
        mod_type = sample.modulation_type.value
        
        # EVM validation (if available)
        evm = sample.quality_metrics.evm_percent
        if evm is not None and mod_type in self.thresholds.evm_thresholds:
            threshold = self.thresholds.evm_thresholds[mod_type]
            if evm > threshold:
                results['issues'].append(f"High EVM for {mod_type}: {evm:.1f}% (threshold: {threshold}%)")
                results['recommendations'].append("Improve signal quality or check modulation accuracy")
                score *= 0.6
        
        # Modulation-specific checks
        if mod_type in ['BPSK', 'QPSK', '8PSK']:
            # PSK-specific validation
            score *= self._validate_psk_signal(sample, results)
        elif mod_type in ['QAM16', 'QAM64', 'QAM256']:
            # QAM-specific validation
            score *= self._validate_qam_signal(sample, results)
        elif mod_type in ['AM-DSB', 'AM-SSB']:
            # AM-specific validation
            score *= self._validate_am_signal(sample, results)
        elif mod_type in ['FM', 'WBFM']:
            # FM-specific validation
            score *= self._validate_fm_signal(sample, results)
        elif mod_type in ['FSK', 'GFSK', 'CPFSK']:
            # FSK-specific validation
            score *= self._validate_fsk_signal(sample, results)
        
        return score
    
    def _validate_hardware_characteristics(self, sample: EnhancedSignalSample, results: Dict[str, Any]) -> Optional[float]:
        """Validate hardware-specific characteristics"""
        if sample.hardware_info is None:
            return None
        
        score = 1.0
        hw_info = sample.hardware_info
        
        # Check frequency range
        if not (hw_info.frequency_range[0] <= sample.rf_params.center_frequency <= hw_info.frequency_range[1]):
            results['warnings'].append("Signal frequency outside hardware range")
            score *= 0.8
        
        # Check sample rate
        if sample.rf_params.sample_rate > hw_info.max_sample_rate:
            results['warnings'].append("Sample rate exceeds hardware maximum")
            score *= 0.7
        
        # Check gain range
        if not (hw_info.gain_range[0] <= sample.rf_params.gain <= hw_info.gain_range[1]):
            results['warnings'].append("Gain setting outside hardware range")
            score *= 0.9
        
        # Hardware-specific checks
        hw_type = hw_info.hardware_type.value
        if hw_type == 'RTL-SDR':
            # RTL-SDR specific checks
            if sample.rf_params.center_frequency < 24e6:
                results['warnings'].append("Frequency below RTL-SDR reliable range")
                score *= 0.8
        elif hw_type in ['USRP-B200', 'USRP-B210']:
            # USRP B-series specific checks
            if sample.rf_params.sample_rate > 56e6:
                results['warnings'].append("Sample rate near USRP B-series limit")
        
        return score
    
    def _validate_environmental_factors(self, sample: EnhancedSignalSample, results: Dict[str, Any]) -> float:
        """Validate environmental factors"""
        score = 1.0
        
        # Weather condition impact
        if sample.weather_condition == 'rain' and sample.rf_params.center_frequency > 10e9:
            results['warnings'].append("Rain attenuation possible at this frequency")
            score *= 0.9
        
        # Environment-specific checks
        if sample.environment == 'indoor':
            if sample.rf_params.center_frequency > 2.4e9:
                results['warnings'].append("Indoor propagation may affect higher frequencies")
        elif sample.environment == 'urban':
            results['warnings'].append("Urban environment may introduce multipath effects")
        
        # Interference level impact
        if sample.interference_level == 'high':
            results['warnings'].append("High interference environment detected")
            results['recommendations'].append("Consider frequency planning or filtering")
            score *= 0.8
        
        return score
    
    def _calculate_iq_imbalance(self, iq_data: np.ndarray) -> Dict[str, float]:
        """Calculate IQ imbalance metrics"""
        i_component = iq_data.real
        q_component = iq_data.imag
        
        # Amplitude imbalance
        i_power = np.mean(i_component ** 2)
        q_power = np.mean(q_component ** 2)
        amplitude_imbalance_db = 10 * np.log10(i_power / (q_power + 1e-10))
        
        # Phase imbalance (correlation-based estimate)
        correlation = np.corrcoef(i_component, q_component)[0, 1]
        phase_imbalance_deg = np.degrees(np.arccos(np.abs(correlation)))
        
        return {
            'amplitude_imbalance_db': abs(amplitude_imbalance_db),
            'phase_imbalance_deg': phase_imbalance_deg,
            'i_power': i_power,
            'q_power': q_power,
            'correlation': correlation
        }
    
    def _analyze_spectrum(self, iq_data: np.ndarray, sample_rate: float) -> Dict[str, float]:
        """Analyze signal spectrum"""
        # Compute FFT
        fft_data = fftshift(fft(iq_data))
        freqs = fftshift(fftfreq(len(iq_data), 1/sample_rate))
        power_spectrum = np.abs(fft_data) ** 2
        
        # Normalize power spectrum
        power_spectrum_db = 10 * np.log10(power_spectrum / np.max(power_spectrum) + 1e-10)
        
        # Find occupied bandwidth (99% power)
        total_power = np.sum(power_spectrum)
        cumulative_power = np.cumsum(power_spectrum)
        
        # Find 0.5% and 99.5% power points
        low_idx = np.argmax(cumulative_power >= 0.005 * total_power)
        high_idx = np.argmax(cumulative_power >= 0.995 * total_power)
        
        occupied_bandwidth = freqs[high_idx] - freqs[low_idx]
        
        # Find spurious signals (peaks outside main signal)
        # Simple approach: find peaks more than 20 dB below main peak
        main_peak_idx = np.argmax(power_spectrum)
        main_peak_power = power_spectrum_db[main_peak_idx]
        
        # Look for spurious signals outside occupied bandwidth
        spurious_mask = np.ones_like(power_spectrum_db, dtype=bool)
        spurious_mask[low_idx:high_idx] = False
        
        spurious_powers = power_spectrum_db[spurious_mask]
        max_spurious_power_db = np.max(spurious_powers) - main_peak_power if len(spurious_powers) > 0 else -100
        
        # Spectral flatness
        geometric_mean = np.exp(np.mean(np.log(power_spectrum + 1e-10)))
        arithmetic_mean = np.mean(power_spectrum)
        spectral_flatness = geometric_mean / arithmetic_mean
        
        return {
            'occupied_bandwidth': abs(occupied_bandwidth),
            'max_spurious_power_db': max_spurious_power_db,
            'spectral_flatness': spectral_flatness,
            'peak_frequency': freqs[main_peak_idx],
            'bandwidth_efficiency': abs(occupied_bandwidth) / sample_rate
        }
    
    def _validate_psk_signal(self, sample: EnhancedSignalSample, results: Dict[str, Any]) -> float:
        """Validate PSK-specific characteristics"""
        score = 1.0
        
        # Check constellation clustering
        constellation_quality = self._analyze_constellation(sample.iq_data, sample.modulation_type.value)
        
        if constellation_quality['cluster_separation'] < 0.5:
            results['warnings'].append("Poor PSK constellation clustering")
            results['recommendations'].append("Improve SNR or check phase noise")
            score *= 0.7
        
        return score
    
    def _validate_qam_signal(self, sample: EnhancedSignalSample, results: Dict[str, Any]) -> float:
        """Validate QAM-specific characteristics"""
        score = 1.0
        
        # QAM signals should have specific amplitude levels
        amplitude = np.abs(sample.iq_data)
        amplitude_levels = len(np.unique(np.round(amplitude, 1)))
        
        expected_levels = {'QAM16': 3, 'QAM64': 4, 'QAM256': 8}
        mod_type = sample.modulation_type.value
        
        if mod_type in expected_levels:
            if amplitude_levels < expected_levels[mod_type] * 0.7:
                results['warnings'].append(f"Fewer amplitude levels than expected for {mod_type}")
                score *= 0.8
        
        return score
    
    def _validate_am_signal(self, sample: EnhancedSignalSample, results: Dict[str, Any]) -> float:
        """Validate AM-specific characteristics"""
        score = 1.0
        
        # AM signals should have amplitude modulation
        amplitude = np.abs(sample.iq_data)
        modulation_depth = (np.max(amplitude) - np.min(amplitude)) / (np.max(amplitude) + np.min(amplitude))
        
        if modulation_depth < 0.1:
            results['warnings'].append("Low AM modulation depth")
            score *= 0.8
        elif modulation_depth > 0.95:
            results['warnings'].append("Very high AM modulation depth (possible overmodulation)")
            score *= 0.9
        
        return score
    
    def _validate_fm_signal(self, sample: EnhancedSignalSample, results: Dict[str, Any]) -> float:
        """Validate FM-specific characteristics"""
        score = 1.0
        
        # FM signals should have constant amplitude
        amplitude = np.abs(sample.iq_data)
        amplitude_variation = np.std(amplitude) / np.mean(amplitude)
        
        if amplitude_variation > 0.2:
            results['warnings'].append("High amplitude variation in FM signal")
            results['recommendations'].append("Check for amplitude limiting or interference")
            score *= 0.8
        
        return score
    
    def _validate_fsk_signal(self, sample: EnhancedSignalSample, results: Dict[str, Any]) -> float:
        """Validate FSK-specific characteristics"""
        score = 1.0
        
        # FSK signals should show frequency shifts
        instantaneous_freq = np.diff(np.unwrap(np.angle(sample.iq_data)))
        freq_levels = len(np.unique(np.round(instantaneous_freq, 3)))
        
        if freq_levels < 2:
            results['warnings'].append("FSK signal shows limited frequency variation")
            score *= 0.8
        
        return score
    
    def _analyze_constellation(self, iq_data: np.ndarray, modulation_type: str) -> Dict[str, float]:
        """Analyze constellation diagram characteristics"""
        # Simple constellation analysis
        # For more sophisticated analysis, would need symbol timing recovery
        
        # Downsample to approximate symbol rate (rough estimate)
        decimation_factor = max(1, len(iq_data) // 1000)
        symbols = iq_data[::decimation_factor]
        
        # Calculate cluster separation (simplified)
        if modulation_type == 'BPSK':
            # BPSK should cluster around +1 and -1 on real axis
            real_parts = symbols.real
            cluster_separation = np.std(real_parts)  # Higher std indicates better separation
        elif modulation_type == 'QPSK':
            # QPSK should cluster around 4 points
            angles = np.angle(symbols)
            # Quantize to nearest QPSK phase
            quantized_angles = np.round(angles / (np.pi/2)) * (np.pi/2)
            phase_error = np.std(angles - quantized_angles)
            cluster_separation = 1.0 / (phase_error + 0.1)  # Inverse relationship
        else:
            # Generic approach
            cluster_separation = np.std(np.abs(symbols))
        
        return {
            'cluster_separation': cluster_separation,
            'num_symbols_analyzed': len(symbols)
        }