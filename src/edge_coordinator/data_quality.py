"""
Data Quality Validation and Preprocessing for Edge Coordinator
"""
import asyncio
import logging
import numpy as np
import json
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import statistics

from ..common.signal_models import EnhancedSignalSample, SignalQualityMetrics
from ..common.federated_data_structures import EnhancedModelUpdate


class DataQualityIssue(Enum):
    """Types of data quality issues"""
    LOW_SNR = "low_snr"
    HIGH_NOISE = "high_noise"
    CORRUPTED_SAMPLES = "corrupted_samples"
    INSUFFICIENT_DIVERSITY = "insufficient_diversity"
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"
    FREQUENCY_DRIFT = "frequency_drift"
    AMPLITUDE_ANOMALY = "amplitude_anomaly"
    MISSING_METADATA = "missing_metadata"
    DUPLICATE_SAMPLES = "duplicate_samples"
    OUTLIER_DETECTION = "outlier_detection"


class ValidationSeverity(Enum):
    """Severity levels for validation issues"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class DataQualityReport:
    """Data quality validation report"""
    client_id: str
    dataset_id: str
    validation_timestamp: datetime
    
    # Overall quality metrics
    overall_score: float  # 0.0 to 1.0
    sample_count: int
    valid_samples: int
    
    # Quality issues found
    issues: List[Dict[str, Any]] = field(default_factory=list)
    
    # Signal quality statistics
    snr_statistics: Dict[str, float] = field(default_factory=dict)
    frequency_statistics: Dict[str, float] = field(default_factory=dict)
    modulation_distribution: Dict[str, int] = field(default_factory=dict)
    
    # Preprocessing recommendations
    preprocessing_recommendations: List[str] = field(default_factory=list)
    
    # Validation metadata
    validation_duration_seconds: float = 0.0
    validator_version: str = "1.0"
    
    def add_issue(self, issue_type: DataQualityIssue, severity: ValidationSeverity, 
                  description: str, affected_samples: int = 0, metadata: Dict[str, Any] = None):
        """Add a data quality issue to the report"""
        issue = {
            'type': issue_type.value,
            'severity': severity.value,
            'description': description,
            'affected_samples': affected_samples,
            'metadata': metadata or {},
            'detected_at': datetime.now().isoformat()
        }
        self.issues.append(issue)
    
    def get_issues_by_severity(self, severity: ValidationSeverity) -> List[Dict[str, Any]]:
        """Get issues by severity level"""
        return [issue for issue in self.issues if issue['severity'] == severity.value]
    
    def has_critical_issues(self) -> bool:
        """Check if report has critical issues"""
        return len(self.get_issues_by_severity(ValidationSeverity.CRITICAL)) > 0
    
    def calculate_overall_score(self):
        """Calculate overall quality score based on issues"""
        if self.sample_count == 0:
            self.overall_score = 0.0
            return
        
        # Start with perfect score
        score = 1.0
        
        # Deduct points for issues
        for issue in self.issues:
            severity = issue['severity']
            affected_ratio = issue['affected_samples'] / self.sample_count
            
            if severity == ValidationSeverity.CRITICAL.value:
                score -= 0.5 * affected_ratio
            elif severity == ValidationSeverity.ERROR.value:
                score -= 0.3 * affected_ratio
            elif severity == ValidationSeverity.WARNING.value:
                score -= 0.1 * affected_ratio
            elif severity == ValidationSeverity.INFO.value:
                score -= 0.05 * affected_ratio
        
        # Bonus for high valid sample ratio
        valid_ratio = self.valid_samples / self.sample_count
        score *= valid_ratio
        
        self.overall_score = max(0.0, min(1.0, score))


class DataQualityValidator:
    """
    Data Quality Validator for Edge Coordinator
    
    Validates and preprocesses signal data from edge clients to ensure
    high-quality training data for federated learning.
    """
    
    def __init__(self, coordinator_id: str, config: Dict[str, Any]):
        self.coordinator_id = coordinator_id
        self.config = config
        self.logger = logging.getLogger(f"DataQualityValidator-{coordinator_id}")
        
        # Validation thresholds
        self.min_snr_db = config.get('min_snr_db', -10.0)
        self.max_noise_floor_db = config.get('max_noise_floor_db', -80.0)
        self.min_samples_per_class = config.get('min_samples_per_class', 10)
        self.max_frequency_drift_hz = config.get('max_frequency_drift_hz', 1000.0)
        self.outlier_threshold_std = config.get('outlier_threshold_std', 3.0)
        
        # Preprocessing options
        self.auto_preprocessing = config.get('auto_preprocessing', True)
        self.noise_reduction_enabled = config.get('noise_reduction_enabled', True)
        self.normalization_enabled = config.get('normalization_enabled', True)
        self.outlier_removal_enabled = config.get('outlier_removal_enabled', True)
        
        # Quality tracking
        self.client_quality_history: Dict[str, List[DataQualityReport]] = {}
        self.global_quality_stats: Dict[str, Any] = {}
        
        # Background tasks
        self.running = False
        self.background_tasks: List[asyncio.Task] = []
    
    async def start(self):
        """Start data quality validator"""
        self.running = True
        
        self.background_tasks = [
            asyncio.create_task(self._quality_monitoring_loop()),
            asyncio.create_task(self._statistics_update_loop())
        ]
        
        self.logger.info("Data quality validator started")
    
    async def stop(self):
        """Stop data quality validator"""
        self.running = False
        
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.logger.info("Data quality validator stopped")
    
    async def validate_signal_data(self, client_id: str, signal_samples: List[EnhancedSignalSample]) -> DataQualityReport:
        """Validate signal data quality"""
        start_time = datetime.now()
        
        report = DataQualityReport(
            client_id=client_id,
            dataset_id=f"dataset-{client_id}-{int(start_time.timestamp())}",
            validation_timestamp=start_time,
            sample_count=len(signal_samples),
            valid_samples=0
        )
        
        if not signal_samples:
            report.add_issue(
                DataQualityIssue.INSUFFICIENT_DIVERSITY,
                ValidationSeverity.CRITICAL,
                "No signal samples provided"
            )
            report.calculate_overall_score()
            return report
        
        # Validate individual samples
        valid_samples = []
        for i, sample in enumerate(signal_samples):
            sample_issues = await self._validate_single_sample(sample, i)
            
            if not any(issue['severity'] in ['error', 'critical'] for issue in sample_issues):
                valid_samples.append(sample)
            
            # Add sample issues to report
            for issue in sample_issues:
                report.add_issue(
                    DataQualityIssue(issue['type']),
                    ValidationSeverity(issue['severity']),
                    issue['description'],
                    1,  # Single sample affected
                    issue.get('metadata', {})
                )
        
        report.valid_samples = len(valid_samples)
        
        # Validate dataset-level properties
        await self._validate_dataset_properties(valid_samples, report)
        
        # Calculate statistics
        self._calculate_quality_statistics(valid_samples, report)
        
        # Generate preprocessing recommendations
        self._generate_preprocessing_recommendations(report)
        
        # Calculate overall score
        report.calculate_overall_score()
        
        # Record validation duration
        report.validation_duration_seconds = (datetime.now() - start_time).total_seconds()
        
        # Store report in history
        if client_id not in self.client_quality_history:
            self.client_quality_history[client_id] = []
        
        self.client_quality_history[client_id].append(report)
        
        # Keep only recent history
        max_history = 50
        if len(self.client_quality_history[client_id]) > max_history:
            self.client_quality_history[client_id] = self.client_quality_history[client_id][-max_history:]
        
        self.logger.info(f"Validated {len(signal_samples)} samples from {client_id}, quality score: {report.overall_score:.2f}")
        
        return report
    
    async def _validate_single_sample(self, sample: EnhancedSignalSample, sample_index: int) -> List[Dict[str, Any]]:
        """Validate a single signal sample"""
        issues = []
        
        # Check SNR
        if sample.quality_metrics.snr_db < self.min_snr_db:
            issues.append({
                'type': DataQualityIssue.LOW_SNR.value,
                'severity': ValidationSeverity.WARNING.value,
                'description': f"Low SNR: {sample.quality_metrics.snr_db:.1f} dB (min: {self.min_snr_db} dB)",
                'metadata': {'snr_db': sample.quality_metrics.snr_db, 'sample_index': sample_index}
            })
        
        # Check noise floor
        if sample.quality_metrics.noise_floor_db > self.max_noise_floor_db:
            issues.append({
                'type': DataQualityIssue.HIGH_NOISE.value,
                'severity': ValidationSeverity.WARNING.value,
                'description': f"High noise floor: {sample.quality_metrics.noise_floor_db:.1f} dB",
                'metadata': {'noise_floor_db': sample.quality_metrics.noise_floor_db, 'sample_index': sample_index}
            })
        
        # Check for corrupted IQ data
        if sample.iq_data is not None:
            if np.any(np.isnan(sample.iq_data)) or np.any(np.isinf(sample.iq_data)):
                issues.append({
                    'type': DataQualityIssue.CORRUPTED_SAMPLES.value,
                    'severity': ValidationSeverity.ERROR.value,
                    'description': "IQ data contains NaN or infinite values",
                    'metadata': {'sample_index': sample_index}
                })
            
            # Check for amplitude anomalies
            iq_magnitude = np.abs(sample.iq_data)
            if np.max(iq_magnitude) > 10.0 or np.min(iq_magnitude) < 1e-6:
                issues.append({
                    'type': DataQualityIssue.AMPLITUDE_ANOMALY.value,
                    'severity': ValidationSeverity.WARNING.value,
                    'description': f"Unusual amplitude range: {np.min(iq_magnitude):.2e} to {np.max(iq_magnitude):.2e}",
                    'metadata': {'min_amplitude': float(np.min(iq_magnitude)), 'max_amplitude': float(np.max(iq_magnitude)), 'sample_index': sample_index}
                })
        
        # Check for missing metadata
        required_fields = ['timestamp', 'center_frequency', 'sample_rate']
        for field in required_fields:
            if not hasattr(sample.rf_params, field) or getattr(sample.rf_params, field) is None:
                issues.append({
                    'type': DataQualityIssue.MISSING_METADATA.value,
                    'severity': ValidationSeverity.ERROR.value,
                    'description': f"Missing required field: {field}",
                    'metadata': {'missing_field': field, 'sample_index': sample_index}
                })
        
        return issues
    
    async def _validate_dataset_properties(self, samples: List[EnhancedSignalSample], report: DataQualityReport):
        """Validate dataset-level properties"""
        if not samples:
            return
        
        # Check modulation diversity
        modulation_counts = {}
        for sample in samples:
            mod_type = sample.modulation_type.value
            modulation_counts[mod_type] = modulation_counts.get(mod_type, 0) + 1
        
        report.modulation_distribution = modulation_counts
        
        # Check if we have minimum samples per class
        insufficient_classes = [
            mod_type for mod_type, count in modulation_counts.items()
            if count < self.min_samples_per_class
        ]
        
        if insufficient_classes:
            report.add_issue(
                DataQualityIssue.INSUFFICIENT_DIVERSITY,
                ValidationSeverity.WARNING,
                f"Insufficient samples for modulations: {insufficient_classes}",
                sum(modulation_counts[mod] for mod in insufficient_classes),
                {'insufficient_classes': insufficient_classes}
            )
        
        # Check temporal consistency
        timestamps = [sample.timestamp for sample in samples if sample.timestamp]
        if len(timestamps) > 1:
            timestamps.sort()
            time_gaps = [(timestamps[i+1] - timestamps[i]).total_seconds() for i in range(len(timestamps)-1)]
            
            # Check for large time gaps (more than 1 hour)
            large_gaps = [gap for gap in time_gaps if gap > 3600]
            if large_gaps:
                report.add_issue(
                    DataQualityIssue.TEMPORAL_INCONSISTENCY,
                    ValidationSeverity.INFO,
                    f"Found {len(large_gaps)} large time gaps in data",
                    0,
                    {'max_gap_seconds': max(large_gaps), 'large_gaps_count': len(large_gaps)}
                )
        
        # Check frequency consistency
        frequencies = [sample.rf_params.center_frequency for sample in samples]
        if frequencies:
            freq_std = statistics.stdev(frequencies) if len(frequencies) > 1 else 0
            if freq_std > self.max_frequency_drift_hz:
                report.add_issue(
                    DataQualityIssue.FREQUENCY_DRIFT,
                    ValidationSeverity.WARNING,
                    f"High frequency drift: {freq_std:.1f} Hz std dev",
                    len(samples),
                    {'frequency_std_hz': freq_std}
                )
        
        # Check for duplicate samples
        sample_hashes = set()
        duplicate_count = 0
        
        for sample in samples:
            if sample.iq_data is not None:
                # Create hash of IQ data
                sample_hash = hashlib.md5(sample.iq_data.tobytes()).hexdigest()
                if sample_hash in sample_hashes:
                    duplicate_count += 1
                else:
                    sample_hashes.add(sample_hash)
        
        if duplicate_count > 0:
            report.add_issue(
                DataQualityIssue.DUPLICATE_SAMPLES,
                ValidationSeverity.WARNING,
                f"Found {duplicate_count} duplicate samples",
                duplicate_count,
                {'duplicate_count': duplicate_count}
            )
        
        # Outlier detection
        if len(samples) > 10:  # Need sufficient samples for outlier detection
            snr_values = [sample.quality_metrics.snr_db for sample in samples]
            snr_mean = statistics.mean(snr_values)
            snr_std = statistics.stdev(snr_values)
            
            outliers = [
                snr for snr in snr_values
                if abs(snr - snr_mean) > self.outlier_threshold_std * snr_std
            ]
            
            if outliers:
                report.add_issue(
                    DataQualityIssue.OUTLIER_DETECTION,
                    ValidationSeverity.INFO,
                    f"Found {len(outliers)} SNR outliers",
                    len(outliers),
                    {'outlier_count': len(outliers), 'outlier_threshold_std': self.outlier_threshold_std}
                )
    
    def _calculate_quality_statistics(self, samples: List[EnhancedSignalSample], report: DataQualityReport):
        """Calculate quality statistics for the dataset"""
        if not samples:
            return
        
        # SNR statistics
        snr_values = [sample.quality_metrics.snr_db for sample in samples]
        report.snr_statistics = {
            'mean': statistics.mean(snr_values),
            'median': statistics.median(snr_values),
            'std': statistics.stdev(snr_values) if len(snr_values) > 1 else 0,
            'min': min(snr_values),
            'max': max(snr_values),
            'count': len(snr_values)
        }
        
        # Frequency statistics
        frequencies = [sample.rf_params.center_frequency for sample in samples]
        report.frequency_statistics = {
            'mean': statistics.mean(frequencies),
            'median': statistics.median(frequencies),
            'std': statistics.stdev(frequencies) if len(frequencies) > 1 else 0,
            'min': min(frequencies),
            'max': max(frequencies),
            'count': len(frequencies)
        }
    
    def _generate_preprocessing_recommendations(self, report: DataQualityReport):
        """Generate preprocessing recommendations based on quality issues"""
        recommendations = []
        
        # Check for low SNR issues
        low_snr_issues = [issue for issue in report.issues if issue['type'] == DataQualityIssue.LOW_SNR.value]
        if low_snr_issues:
            recommendations.append("Apply noise reduction filtering to improve SNR")
        
        # Check for amplitude anomalies
        amplitude_issues = [issue for issue in report.issues if issue['type'] == DataQualityIssue.AMPLITUDE_ANOMALY.value]
        if amplitude_issues:
            recommendations.append("Apply amplitude normalization to standardize signal levels")
        
        # Check for frequency drift
        freq_drift_issues = [issue for issue in report.issues if issue['type'] == DataQualityIssue.FREQUENCY_DRIFT.value]
        if freq_drift_issues:
            recommendations.append("Apply frequency correction to compensate for drift")
        
        # Check for insufficient diversity
        diversity_issues = [issue for issue in report.issues if issue['type'] == DataQualityIssue.INSUFFICIENT_DIVERSITY.value]
        if diversity_issues:
            recommendations.append("Collect more samples for underrepresented modulation types")
        
        # Check for outliers
        outlier_issues = [issue for issue in report.issues if issue['type'] == DataQualityIssue.OUTLIER_DETECTION.value]
        if outlier_issues:
            recommendations.append("Consider removing or investigating outlier samples")
        
        # Check for duplicates
        duplicate_issues = [issue for issue in report.issues if issue['type'] == DataQualityIssue.DUPLICATE_SAMPLES.value]
        if duplicate_issues:
            recommendations.append("Remove duplicate samples to avoid bias")
        
        report.preprocessing_recommendations = recommendations
    
    async def preprocess_signal_data(self, samples: List[EnhancedSignalSample], 
                                   quality_report: DataQualityReport) -> List[EnhancedSignalSample]:
        """Preprocess signal data based on quality report"""
        if not self.auto_preprocessing:
            return samples
        
        processed_samples = samples.copy()
        
        # Remove corrupted samples
        processed_samples = [
            sample for sample in processed_samples
            if sample.iq_data is not None and 
            not np.any(np.isnan(sample.iq_data)) and 
            not np.any(np.isinf(sample.iq_data))
        ]
        
        # Remove duplicates if enabled
        if self.outlier_removal_enabled:
            processed_samples = self._remove_duplicate_samples(processed_samples)
        
        # Apply noise reduction if enabled
        if self.noise_reduction_enabled:
            processed_samples = await self._apply_noise_reduction(processed_samples)
        
        # Apply normalization if enabled
        if self.normalization_enabled:
            processed_samples = self._apply_normalization(processed_samples)
        
        # Remove outliers if enabled
        if self.outlier_removal_enabled:
            processed_samples = self._remove_outlier_samples(processed_samples)
        
        self.logger.info(f"Preprocessed {len(samples)} samples to {len(processed_samples)} samples")
        
        return processed_samples
    
    def _remove_duplicate_samples(self, samples: List[EnhancedSignalSample]) -> List[EnhancedSignalSample]:
        """Remove duplicate samples based on IQ data hash"""
        seen_hashes = set()
        unique_samples = []
        
        for sample in samples:
            if sample.iq_data is not None:
                sample_hash = hashlib.md5(sample.iq_data.tobytes()).hexdigest()
                if sample_hash not in seen_hashes:
                    seen_hashes.add(sample_hash)
                    unique_samples.append(sample)
            else:
                unique_samples.append(sample)
        
        return unique_samples
    
    async def _apply_noise_reduction(self, samples: List[EnhancedSignalSample]) -> List[EnhancedSignalSample]:
        """Apply noise reduction to signal samples"""
        processed_samples = []
        
        for sample in samples:
            if sample.iq_data is not None:
                # Simple noise reduction using moving average
                # In practice, you'd use more sophisticated techniques
                window_size = min(5, len(sample.iq_data) // 10)
                if window_size > 1:
                    # Apply moving average filter
                    filtered_iq = np.convolve(sample.iq_data, np.ones(window_size)/window_size, mode='same')
                    
                    # Create new sample with filtered data
                    processed_sample = EnhancedSignalSample(
                        timestamp=sample.timestamp,
                        iq_data=filtered_iq,
                        modulation_type=sample.modulation_type,
                        rf_params=sample.rf_params,
                        quality_metrics=sample.quality_metrics,
                        duration=sample.duration,
                        metadata=sample.metadata
                    )
                    processed_samples.append(processed_sample)
                else:
                    processed_samples.append(sample)
            else:
                processed_samples.append(sample)
        
        return processed_samples
    
    def _apply_normalization(self, samples: List[EnhancedSignalSample]) -> List[EnhancedSignalSample]:
        """Apply amplitude normalization to signal samples"""
        processed_samples = []
        
        for sample in samples:
            if sample.iq_data is not None:
                # Normalize to unit power
                power = np.mean(np.abs(sample.iq_data) ** 2)
                if power > 0:
                    normalized_iq = sample.iq_data / np.sqrt(power)
                    
                    # Create new sample with normalized data
                    processed_sample = EnhancedSignalSample(
                        timestamp=sample.timestamp,
                        iq_data=normalized_iq,
                        modulation_type=sample.modulation_type,
                        rf_params=sample.rf_params,
                        quality_metrics=sample.quality_metrics,
                        duration=sample.duration,
                        metadata=sample.metadata
                    )
                    processed_samples.append(processed_sample)
                else:
                    processed_samples.append(sample)
            else:
                processed_samples.append(sample)
        
        return processed_samples
    
    def _remove_outlier_samples(self, samples: List[EnhancedSignalSample]) -> List[EnhancedSignalSample]:
        """Remove outlier samples based on SNR"""
        if len(samples) < 10:  # Need sufficient samples
            return samples
        
        snr_values = [sample.quality_metrics.snr_db for sample in samples]
        snr_mean = statistics.mean(snr_values)
        snr_std = statistics.stdev(snr_values)
        
        filtered_samples = []
        for sample in samples:
            snr_z_score = abs(sample.quality_metrics.snr_db - snr_mean) / snr_std
            if snr_z_score <= self.outlier_threshold_std:
                filtered_samples.append(sample)
        
        return filtered_samples
    
    def get_client_quality_summary(self, client_id: str) -> Dict[str, Any]:
        """Get quality summary for a specific client"""
        if client_id not in self.client_quality_history:
            return {'error': 'No quality history for client'}
        
        history = self.client_quality_history[client_id]
        if not history:
            return {'error': 'Empty quality history'}
        
        # Calculate average quality score
        quality_scores = [report.overall_score for report in history]
        avg_quality = statistics.mean(quality_scores)
        
        # Count issues by type
        issue_counts = {}
        for report in history:
            for issue in report.issues:
                issue_type = issue['type']
                issue_counts[issue_type] = issue_counts.get(issue_type, 0) + 1
        
        # Get recent trends
        recent_reports = history[-10:]  # Last 10 reports
        recent_quality_scores = [report.overall_score for report in recent_reports]
        
        quality_trend = "stable"
        if len(recent_quality_scores) >= 3:
            if recent_quality_scores[-1] > recent_quality_scores[0] + 0.1:
                quality_trend = "improving"
            elif recent_quality_scores[-1] < recent_quality_scores[0] - 0.1:
                quality_trend = "declining"
        
        return {
            'client_id': client_id,
            'total_reports': len(history),
            'average_quality_score': avg_quality,
            'latest_quality_score': history[-1].overall_score,
            'quality_trend': quality_trend,
            'common_issues': dict(sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]),
            'last_validation': history[-1].validation_timestamp.isoformat()
        }
    
    async def _quality_monitoring_loop(self):
        """Background task for quality monitoring"""
        while self.running:
            try:
                # Update global quality statistics
                await self._update_global_quality_stats()
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Quality monitoring error: {e}")
                await asyncio.sleep(300)
    
    async def _update_global_quality_stats(self):
        """Update global quality statistics"""
        all_scores = []
        all_issues = {}
        
        for client_id, history in self.client_quality_history.items():
            for report in history:
                all_scores.append(report.overall_score)
                
                for issue in report.issues:
                    issue_type = issue['type']
                    all_issues[issue_type] = all_issues.get(issue_type, 0) + 1
        
        if all_scores:
            self.global_quality_stats = {
                'average_quality_score': statistics.mean(all_scores),
                'quality_score_std': statistics.stdev(all_scores) if len(all_scores) > 1 else 0,
                'total_clients': len(self.client_quality_history),
                'total_reports': len(all_scores),
                'common_issues': dict(sorted(all_issues.items(), key=lambda x: x[1], reverse=True)[:10]),
                'last_updated': datetime.now().isoformat()
            }
    
    async def _statistics_update_loop(self):
        """Background task for statistics updates"""
        while self.running:
            try:
                # Clean up old quality reports
                cutoff_date = datetime.now() - timedelta(days=7)  # Keep 7 days of history
                
                for client_id in list(self.client_quality_history.keys()):
                    history = self.client_quality_history[client_id]
                    filtered_history = [
                        report for report in history
                        if report.validation_timestamp > cutoff_date
                    ]
                    
                    if filtered_history:
                        self.client_quality_history[client_id] = filtered_history
                    else:
                        del self.client_quality_history[client_id]
                
                await asyncio.sleep(3600)  # Clean up every hour
                
            except Exception as e:
                self.logger.error(f"Statistics update error: {e}")
                await asyncio.sleep(3600)
    
    def get_validation_status(self) -> Dict[str, Any]:
        """Get comprehensive validation status"""
        return {
            'validator_id': self.coordinator_id,
            'clients_monitored': len(self.client_quality_history),
            'total_reports': sum(len(history) for history in self.client_quality_history.values()),
            'global_quality_stats': self.global_quality_stats,
            'validation_thresholds': {
                'min_snr_db': self.min_snr_db,
                'max_noise_floor_db': self.max_noise_floor_db,
                'min_samples_per_class': self.min_samples_per_class,
                'max_frequency_drift_hz': self.max_frequency_drift_hz,
                'outlier_threshold_std': self.outlier_threshold_std
            },
            'preprocessing_config': {
                'auto_preprocessing': self.auto_preprocessing,
                'noise_reduction_enabled': self.noise_reduction_enabled,
                'normalization_enabled': self.normalization_enabled,
                'outlier_removal_enabled': self.outlier_removal_enabled
            }
        }