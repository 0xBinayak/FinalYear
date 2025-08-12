"""
Data quality validation and metadata extraction tools
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path

from .interfaces import SignalSample


@dataclass
class QualityMetrics:
    """Data quality metrics"""
    overall_score: float  # 0.0 to 1.0
    snr_quality: float
    amplitude_quality: float
    phase_quality: float
    spectral_quality: float
    temporal_quality: float
    issues: List[str]
    recommendations: List[str]


@dataclass
class DatasetMetadata:
    """Comprehensive dataset metadata"""
    total_samples: int
    unique_modulations: List[str]
    snr_range: Tuple[float, float]
    frequency_range: Tuple[float, float]
    sample_rate_range: Tuple[float, float]
    duration_seconds: float
    file_size_mb: float
    quality_score: float
    geographic_coverage: Dict[str, Any]
    temporal_coverage: Dict[str, Any]
    device_diversity: Dict[str, int]
    signal_characteristics: Dict[str, Any]


class DataQualityValidator:
    """Comprehensive data quality validation"""
    
    def __init__(self):
        self.quality_thresholds = {
            'min_snr': -20.0,
            'max_snr': 50.0,
            'min_samples_per_signal': 100,
            'max_amplitude_ratio': 100.0,
            'min_spectral_occupancy': 0.1,
            'max_dc_offset': 0.1,
            'max_phase_discontinuity': np.pi/2
        }
    
    def validate_sample(self, sample: SignalSample) -> QualityMetrics:
        """Validate a single signal sample"""
        issues = []
        recommendations = []
        
        # SNR quality check
        snr_quality = self._assess_snr_quality(sample.snr, issues, recommendations)
        
        # Amplitude quality check
        amplitude_quality = self._assess_amplitude_quality(sample.iq_data, issues, recommendations)
        
        # Phase quality check
        phase_quality = self._assess_phase_quality(sample.iq_data, issues, recommendations)
        
        # Spectral quality check
        spectral_quality = self._assess_spectral_quality(sample.iq_data, sample.sample_rate, issues, recommendations)
        
        # Temporal quality check
        temporal_quality = self._assess_temporal_quality(sample.iq_data, issues, recommendations)
        
        # Calculate overall score
        quality_scores = [snr_quality, amplitude_quality, phase_quality, spectral_quality, temporal_quality]
        overall_score = np.mean(quality_scores)
        
        return QualityMetrics(
            overall_score=overall_score,
            snr_quality=snr_quality,
            amplitude_quality=amplitude_quality,
            phase_quality=phase_quality,
            spectral_quality=spectral_quality,
            temporal_quality=temporal_quality,
            issues=issues,
            recommendations=recommendations
        )
    
    def validate_dataset(self, samples: List[SignalSample]) -> Dict[str, Any]:
        """Validate entire dataset"""
        if not samples:
            return {
                'valid': False,
                'error': 'Empty dataset',
                'quality_score': 0.0
            }
        
        individual_metrics = [self.validate_sample(sample) for sample in samples]
        
        # Aggregate quality metrics
        overall_scores = [m.overall_score for m in individual_metrics]
        all_issues = [issue for m in individual_metrics for issue in m.issues]
        all_recommendations = [rec for m in individual_metrics for rec in m.recommendations]
        
        # Dataset-level checks
        dataset_issues = []
        dataset_recommendations = []
        
        # Check modulation diversity
        modulations = set(sample.modulation_type for sample in samples)
        if len(modulations) < 3:
            dataset_issues.append("Low modulation diversity")
            dataset_recommendations.append("Include more modulation types")
        
        # Check SNR diversity
        snr_values = [sample.snr for sample in samples]
        snr_range = max(snr_values) - min(snr_values)
        if snr_range < 10:
            dataset_issues.append("Limited SNR range")
            dataset_recommendations.append("Include samples with wider SNR range")
        
        # Check sample rate consistency
        sample_rates = set(sample.sample_rate for sample in samples)
        if len(sample_rates) > 5:
            dataset_issues.append("Too many different sample rates")
            dataset_recommendations.append("Standardize sample rates")
        
        # Check temporal distribution
        timestamps = [sample.timestamp for sample in samples if sample.timestamp]
        if len(set(t.date() for t in timestamps)) < 2:
            dataset_issues.append("Limited temporal diversity")
            dataset_recommendations.append("Collect data over multiple days")
        
        # Calculate dataset quality score
        dataset_quality = np.mean(overall_scores)
        
        # Apply penalties for dataset-level issues
        penalty = len(dataset_issues) * 0.05
        dataset_quality = max(0.0, dataset_quality - penalty)
        
        return {
            'valid': dataset_quality > 0.3,
            'quality_score': dataset_quality,
            'total_samples': len(samples),
            'individual_scores': overall_scores,
            'sample_issues': len(all_issues),
            'dataset_issues': dataset_issues,
            'recommendations': list(set(all_recommendations + dataset_recommendations)),
            'modulation_diversity': len(modulations),
            'snr_range': snr_range,
            'quality_distribution': self._calculate_quality_distribution(overall_scores)
        }
    
    def _assess_snr_quality(self, snr: float, issues: List[str], recommendations: List[str]) -> float:
        """Assess SNR quality"""
        if snr < self.quality_thresholds['min_snr']:
            issues.append(f"SNR too low: {snr:.1f} dB")
            recommendations.append("Increase signal power or reduce noise")
            return 0.1
        elif snr > self.quality_thresholds['max_snr']:
            issues.append(f"SNR unusually high: {snr:.1f} dB")
            recommendations.append("Check for signal saturation")
            return 0.7
        elif snr < 0:
            return 0.3 + (snr + 20) / 20 * 0.4  # Scale from -20 to 0 dB
        else:
            return min(1.0, 0.7 + snr / 30 * 0.3)  # Scale from 0 to 30 dB
    
    def _assess_amplitude_quality(self, iq_data: np.ndarray, issues: List[str], recommendations: List[str]) -> float:
        """Assess amplitude quality"""
        if len(iq_data) == 0:
            issues.append("Empty IQ data")
            return 0.0
        
        amplitude = np.abs(iq_data)
        
        # Check for invalid values
        if np.any(np.isnan(amplitude)) or np.any(np.isinf(amplitude)):
            issues.append("Invalid amplitude values (NaN or Inf)")
            recommendations.append("Check data acquisition and processing")
            return 0.0
        
        # Check amplitude range
        max_amp = np.max(amplitude)
        min_amp = np.min(amplitude)
        
        if max_amp == 0:
            issues.append("Zero amplitude signal")
            return 0.0
        
        amplitude_ratio = max_amp / (min_amp + 1e-10)
        if amplitude_ratio > self.quality_thresholds['max_amplitude_ratio']:
            issues.append(f"High amplitude variation: {amplitude_ratio:.1f}")
            recommendations.append("Check for signal clipping or interference")
            return 0.3
        
        # Check for DC offset
        dc_offset = np.abs(np.mean(iq_data))
        normalized_dc = dc_offset / (np.std(np.abs(iq_data)) + 1e-10)
        
        if normalized_dc > self.quality_thresholds['max_dc_offset']:
            issues.append(f"Significant DC offset: {normalized_dc:.3f}")
            recommendations.append("Apply DC offset correction")
            return max(0.5, 1.0 - normalized_dc)
        
        # Check amplitude distribution
        amplitude_std = np.std(amplitude)
        amplitude_mean = np.mean(amplitude)
        cv = amplitude_std / (amplitude_mean + 1e-10)
        
        if cv < 0.1:
            issues.append("Low amplitude variation (possible CW signal)")
            return 0.6
        elif cv > 2.0:
            issues.append("Very high amplitude variation")
            recommendations.append("Check for interference or signal quality")
            return 0.4
        
        return 1.0
    
    def _assess_phase_quality(self, iq_data: np.ndarray, issues: List[str], recommendations: List[str]) -> float:
        """Assess phase quality"""
        if len(iq_data) < 2:
            return 0.0
        
        phase = np.angle(iq_data)
        phase_unwrapped = np.unwrap(phase)
        
        # Check for phase discontinuities
        phase_diff = np.diff(phase_unwrapped)
        max_phase_jump = np.max(np.abs(phase_diff))
        
        if max_phase_jump > self.quality_thresholds['max_phase_discontinuity']:
            issues.append(f"Large phase discontinuity: {max_phase_jump:.2f} rad")
            recommendations.append("Check for phase noise or signal interruption")
            return 0.3
        
        # Check phase noise
        phase_noise = np.std(phase_diff)
        if phase_noise > 1.0:
            issues.append(f"High phase noise: {phase_noise:.2f} rad")
            recommendations.append("Improve signal quality or filtering")
            return 0.5
        
        # Check for phase wrapping issues
        phase_range = np.max(phase) - np.min(phase)
        if phase_range < np.pi/4:
            issues.append("Limited phase variation")
            return 0.7
        
        return 1.0
    
    def _assess_spectral_quality(self, iq_data: np.ndarray, sample_rate: float, 
                               issues: List[str], recommendations: List[str]) -> float:
        """Assess spectral quality"""
        if len(iq_data) < 64:
            issues.append("Insufficient samples for spectral analysis")
            return 0.2
        
        # Compute power spectral density
        fft_data = np.fft.fft(iq_data)
        psd = np.abs(fft_data) ** 2
        psd_normalized = psd / np.sum(psd)
        
        # Check spectral occupancy
        significant_bins = np.sum(psd_normalized > 0.01 * np.max(psd_normalized))
        spectral_occupancy = significant_bins / len(psd_normalized)
        
        if spectral_occupancy < self.quality_thresholds['min_spectral_occupancy']:
            issues.append(f"Low spectral occupancy: {spectral_occupancy:.2f}")
            recommendations.append("Check signal bandwidth and filtering")
            return 0.4
        
        # Check for spectral peaks (potential interference)
        peak_threshold = 10 * np.mean(psd_normalized)
        peaks = np.sum(psd_normalized > peak_threshold)
        
        if peaks > len(psd_normalized) * 0.1:
            issues.append("Multiple spectral peaks detected")
            recommendations.append("Check for interference or multipath")
            return 0.6
        
        # Check spectral flatness (for noise-like signals)
        spectral_flatness = np.exp(np.mean(np.log(psd_normalized + 1e-10))) / np.mean(psd_normalized)
        
        if spectral_flatness > 0.9:
            issues.append("Signal appears noise-like")
            return 0.3
        
        return 1.0
    
    def _assess_temporal_quality(self, iq_data: np.ndarray, issues: List[str], recommendations: List[str]) -> float:
        """Assess temporal quality"""
        if len(iq_data) < self.quality_thresholds['min_samples_per_signal']:
            issues.append(f"Insufficient samples: {len(iq_data)}")
            recommendations.append("Increase capture duration")
            return 0.2
        
        # Check for signal interruptions (zeros or very low amplitude)
        amplitude = np.abs(iq_data)
        threshold = 0.01 * np.max(amplitude)
        interruptions = np.sum(amplitude < threshold)
        
        if interruptions > len(iq_data) * 0.1:
            issues.append(f"Signal interruptions detected: {interruptions} samples")
            recommendations.append("Check signal continuity")
            return 0.4
        
        # Check for temporal stationarity (simplified)
        segment_size = len(iq_data) // 4
        if segment_size > 10:
            segments = [iq_data[i:i+segment_size] for i in range(0, len(iq_data), segment_size)][:4]
            segment_powers = [np.mean(np.abs(seg)**2) for seg in segments]
            
            power_variation = np.std(segment_powers) / (np.mean(segment_powers) + 1e-10)
            if power_variation > 0.5:
                issues.append(f"High temporal power variation: {power_variation:.2f}")
                recommendations.append("Check for signal stability")
                return 0.6
        
        return 1.0
    
    def _calculate_quality_distribution(self, scores: List[float]) -> Dict[str, int]:
        """Calculate quality score distribution"""
        distribution = {
            'excellent': 0,  # > 0.8
            'good': 0,       # 0.6 - 0.8
            'fair': 0,       # 0.4 - 0.6
            'poor': 0        # < 0.4
        }
        
        for score in scores:
            if score > 0.8:
                distribution['excellent'] += 1
            elif score > 0.6:
                distribution['good'] += 1
            elif score > 0.4:
                distribution['fair'] += 1
            else:
                distribution['poor'] += 1
        
        return distribution


class MetadataExtractor:
    """Extract comprehensive metadata from signal datasets"""
    
    def extract_dataset_metadata(self, samples: List[SignalSample], 
                                dataset_path: Optional[str] = None) -> DatasetMetadata:
        """Extract comprehensive metadata from dataset"""
        if not samples:
            raise ValueError("Empty dataset")
        
        # Basic statistics
        total_samples = len(samples)
        modulations = list(set(sample.modulation_type for sample in samples))
        
        # SNR analysis
        snr_values = [sample.snr for sample in samples]
        snr_range = (min(snr_values), max(snr_values))
        
        # Frequency analysis
        frequencies = [sample.frequency for sample in samples]
        frequency_range = (min(frequencies), max(frequencies))
        
        # Sample rate analysis
        sample_rates = [sample.sample_rate for sample in samples]
        sample_rate_range = (min(sample_rates), max(sample_rates))
        
        # Duration calculation
        total_duration = sum(len(sample.iq_data) / sample.sample_rate for sample in samples)
        
        # File size estimation
        total_samples_count = sum(len(sample.iq_data) for sample in samples)
        file_size_mb = total_samples_count * 8 / (1024 * 1024)  # Complex64 = 8 bytes
        
        # Quality assessment
        validator = DataQualityValidator()
        quality_results = validator.validate_dataset(samples)
        quality_score = quality_results['quality_score']
        
        # Geographic coverage
        geographic_coverage = self._analyze_geographic_coverage(samples)
        
        # Temporal coverage
        temporal_coverage = self._analyze_temporal_coverage(samples)
        
        # Device diversity
        device_diversity = self._analyze_device_diversity(samples)
        
        # Signal characteristics
        signal_characteristics = self._analyze_signal_characteristics(samples)
        
        return DatasetMetadata(
            total_samples=total_samples,
            unique_modulations=modulations,
            snr_range=snr_range,
            frequency_range=frequency_range,
            sample_rate_range=sample_rate_range,
            duration_seconds=total_duration,
            file_size_mb=file_size_mb,
            quality_score=quality_score,
            geographic_coverage=geographic_coverage,
            temporal_coverage=temporal_coverage,
            device_diversity=device_diversity,
            signal_characteristics=signal_characteristics
        )
    
    def _analyze_geographic_coverage(self, samples: List[SignalSample]) -> Dict[str, Any]:
        """Analyze geographic coverage of samples"""
        locations = [sample.location for sample in samples if sample.location]
        
        if not locations:
            return {'coverage': 'unknown', 'locations': 0}
        
        latitudes = [loc['latitude'] for loc in locations]
        longitudes = [loc['longitude'] for loc in locations]
        
        lat_range = max(latitudes) - min(latitudes)
        lon_range = max(longitudes) - min(longitudes)
        
        # Estimate coverage area (rough approximation)
        area_km2 = lat_range * 111 * lon_range * 111 * np.cos(np.radians(np.mean(latitudes)))
        
        return {
            'locations': len(locations),
            'lat_range': lat_range,
            'lon_range': lon_range,
            'estimated_area_km2': area_km2,
            'center_lat': np.mean(latitudes),
            'center_lon': np.mean(longitudes)
        }
    
    def _analyze_temporal_coverage(self, samples: List[SignalSample]) -> Dict[str, Any]:
        """Analyze temporal coverage of samples"""
        timestamps = [sample.timestamp for sample in samples if sample.timestamp]
        
        if not timestamps:
            return {'coverage': 'unknown', 'samples_with_timestamps': 0}
        
        timestamps.sort()
        time_span = timestamps[-1] - timestamps[0]
        
        # Analyze time distribution
        hours = [t.hour for t in timestamps]
        days = [t.weekday() for t in timestamps]
        
        return {
            'samples_with_timestamps': len(timestamps),
            'time_span_days': time_span.days,
            'time_span_hours': time_span.total_seconds() / 3600,
            'earliest': timestamps[0].isoformat(),
            'latest': timestamps[-1].isoformat(),
            'hour_distribution': {h: hours.count(h) for h in range(24)},
            'day_distribution': {d: days.count(d) for d in range(7)}
        }
    
    def _analyze_device_diversity(self, samples: List[SignalSample]) -> Dict[str, int]:
        """Analyze device diversity in dataset"""
        devices = [sample.device_id for sample in samples]
        device_counts = {}
        
        for device in devices:
            device_counts[device] = device_counts.get(device, 0) + 1
        
        return device_counts
    
    def _analyze_signal_characteristics(self, samples: List[SignalSample]) -> Dict[str, Any]:
        """Analyze signal characteristics"""
        characteristics = {
            'modulation_distribution': {},
            'snr_distribution': {},
            'frequency_distribution': {},
            'sample_rate_distribution': {},
            'amplitude_statistics': {},
            'phase_statistics': {}
        }
        
        # Modulation distribution
        for sample in samples:
            mod = sample.modulation_type
            characteristics['modulation_distribution'][mod] = \
                characteristics['modulation_distribution'].get(mod, 0) + 1
        
        # SNR distribution (binned)
        for sample in samples:
            snr_bin = f"{int(sample.snr//5)*5}-{int(sample.snr//5)*5+5}dB"
            characteristics['snr_distribution'][snr_bin] = \
                characteristics['snr_distribution'].get(snr_bin, 0) + 1
        
        # Frequency distribution (binned)
        for sample in samples:
            freq_mhz = sample.frequency / 1e6
            freq_bin = f"{int(freq_mhz//100)*100}-{int(freq_mhz//100)*100+100}MHz"
            characteristics['frequency_distribution'][freq_bin] = \
                characteristics['frequency_distribution'].get(freq_bin, 0) + 1
        
        # Sample rate distribution
        for sample in samples:
            sr_msps = sample.sample_rate / 1e6
            sr_bin = f"{sr_msps:.1f}Msps"
            characteristics['sample_rate_distribution'][sr_bin] = \
                characteristics['sample_rate_distribution'].get(sr_bin, 0) + 1
        
        # Amplitude and phase statistics
        all_amplitudes = []
        all_phases = []
        
        for sample in samples[:100]:  # Limit for performance
            amplitude = np.abs(sample.iq_data)
            phase = np.angle(sample.iq_data)
            
            all_amplitudes.extend(amplitude.tolist())
            all_phases.extend(phase.tolist())
        
        if all_amplitudes:
            characteristics['amplitude_statistics'] = {
                'mean': float(np.mean(all_amplitudes)),
                'std': float(np.std(all_amplitudes)),
                'min': float(np.min(all_amplitudes)),
                'max': float(np.max(all_amplitudes))
            }
        
        if all_phases:
            characteristics['phase_statistics'] = {
                'mean': float(np.mean(all_phases)),
                'std': float(np.std(all_phases)),
                'range': float(np.max(all_phases) - np.min(all_phases))
            }
        
        return characteristics
    
    def save_metadata(self, metadata: DatasetMetadata, filepath: str):
        """Save metadata to JSON file"""
        metadata_dict = {
            'total_samples': metadata.total_samples,
            'unique_modulations': metadata.unique_modulations,
            'snr_range': metadata.snr_range,
            'frequency_range': metadata.frequency_range,
            'sample_rate_range': metadata.sample_rate_range,
            'duration_seconds': metadata.duration_seconds,
            'file_size_mb': metadata.file_size_mb,
            'quality_score': metadata.quality_score,
            'geographic_coverage': metadata.geographic_coverage,
            'temporal_coverage': metadata.temporal_coverage,
            'device_diversity': metadata.device_diversity,
            'signal_characteristics': metadata.signal_characteristics,
            'extraction_timestamp': datetime.now().isoformat()
        }
        
        with open(filepath, 'w') as f:
            json.dump(metadata_dict, f, indent=2, default=str)
    
    def load_metadata(self, filepath: str) -> DatasetMetadata:
        """Load metadata from JSON file"""
        with open(filepath, 'r') as f:
            metadata_dict = json.load(f)
        
        return DatasetMetadata(
            total_samples=metadata_dict['total_samples'],
            unique_modulations=metadata_dict['unique_modulations'],
            snr_range=tuple(metadata_dict['snr_range']),
            frequency_range=tuple(metadata_dict['frequency_range']),
            sample_rate_range=tuple(metadata_dict['sample_rate_range']),
            duration_seconds=metadata_dict['duration_seconds'],
            file_size_mb=metadata_dict['file_size_mb'],
            quality_score=metadata_dict['quality_score'],
            geographic_coverage=metadata_dict['geographic_coverage'],
            temporal_coverage=metadata_dict['temporal_coverage'],
            device_diversity=metadata_dict['device_diversity'],
            signal_characteristics=metadata_dict['signal_characteristics']
        )