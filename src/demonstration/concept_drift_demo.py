"""
Demonstration of concept drift handling with time-varying signal conditions.
Simulates realistic RF environment changes and model adaptation strategies.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
import random
from scipy import stats

from src.common.interfaces import SignalSample
from src.sdr_client.signal_processing import SignalProcessor
from .comparison_engine import SignalClassificationModel


@dataclass
class DriftEvent:
    """Represents a concept drift event."""
    timestamp: datetime
    drift_type: str  # "gradual", "sudden", "recurring"
    affected_modulations: List[str]
    severity: float  # 0.0 to 1.0
    description: str
    environmental_cause: str


@dataclass
class DriftDetectionResult:
    """Result from drift detection algorithm."""
    drift_detected: bool
    drift_score: float
    confidence: float
    affected_classes: List[str]
    detection_method: str
    timestamp: datetime


class ConceptDriftSimulator:
    """Simulates various types of concept drift in RF environments."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Define drift scenarios
        self.drift_scenarios = {
            "weather_change": {
                "description": "Atmospheric conditions affecting propagation",
                "affected_modulations": ["AM", "FM", "BPSK"],
                "severity_range": (0.2, 0.6),
                "duration_hours": (2, 12)
            },
            "interference_increase": {
                "description": "New interference sources in environment",
                "affected_modulations": ["QAM16", "QAM64", "8PSK"],
                "severity_range": (0.3, 0.8),
                "duration_hours": (1, 24)
            },
            "hardware_aging": {
                "description": "Gradual degradation of RF hardware",
                "affected_modulations": ["all"],
                "severity_range": (0.1, 0.4),
                "duration_hours": (24, 168)  # Days to weeks
            },
            "frequency_reallocation": {
                "description": "Regulatory changes in frequency allocation",
                "affected_modulations": ["cellular", "wifi"],
                "severity_range": (0.5, 1.0),
                "duration_hours": (0.1, 1)  # Sudden change
            },
            "seasonal_variation": {
                "description": "Seasonal atmospheric propagation changes",
                "affected_modulations": ["all"],
                "severity_range": (0.2, 0.5),
                "duration_hours": (720, 2160)  # Months
            }
        }
    
    def generate_drift_timeline(self, duration_hours: int = 168, 
                              num_events: int = 5) -> List[DriftEvent]:
        """Generate a timeline of concept drift events."""
        
        events = []
        start_time = datetime.now()
        
        for i in range(num_events):
            # Select random drift scenario
            scenario_name = random.choice(list(self.drift_scenarios.keys()))
            scenario = self.drift_scenarios[scenario_name]
            
            # Generate event timing
            event_time = start_time + timedelta(
                hours=random.uniform(0, duration_hours)
            )
            
            # Determine drift type based on scenario
            if scenario_name in ["weather_change", "interference_increase"]:
                drift_type = "sudden"
            elif scenario_name == "hardware_aging":
                drift_type = "gradual"
            elif scenario_name == "seasonal_variation":
                drift_type = "recurring"
            else:
                drift_type = random.choice(["gradual", "sudden"])
            
            # Generate severity
            severity = random.uniform(*scenario["severity_range"])
            
            # Select affected modulations
            if scenario["affected_modulations"] == ["all"]:
                all_mods = ["BPSK", "QPSK", "8PSK", "QAM16", "QAM64", "AM", "FM"]
                affected_mods = random.sample(all_mods, random.randint(2, len(all_mods)))
            else:
                affected_mods = scenario["affected_modulations"]
            
            event = DriftEvent(
                timestamp=event_time,
                drift_type=drift_type,
                affected_modulations=affected_mods,
                severity=severity,
                description=scenario["description"],
                environmental_cause=scenario_name
            )
            
            events.append(event)
        
        # Sort events by timestamp
        events.sort(key=lambda x: x.timestamp)
        
        self.logger.info(f"Generated {len(events)} drift events over {duration_hours} hours")
        
        return events
    
    def apply_drift_to_samples(self, samples: List[SignalSample], 
                             drift_event: DriftEvent) -> List[SignalSample]:
        """Apply concept drift effects to signal samples."""
        
        modified_samples = []
        
        for sample in samples:
            if sample.modulation_type in drift_event.affected_modulations:
                # Apply drift effects based on type and severity
                modified_sample = self._apply_drift_effects(sample, drift_event)
                modified_samples.append(modified_sample)
            else:
                modified_samples.append(sample)
        
        return modified_samples
    
    def _apply_drift_effects(self, sample: SignalSample, 
                           drift_event: DriftEvent) -> SignalSample:
        """Apply specific drift effects to a signal sample."""
        
        modified_sample = SignalSample(
            timestamp=sample.timestamp,
            frequency=sample.frequency,
            sample_rate=sample.sample_rate,
            iq_data=sample.iq_data.copy(),
            modulation_type=sample.modulation_type,
            snr=sample.snr,
            location=sample.location,
            device_id=sample.device_id,
            metadata=sample.metadata.copy()
        )
        
        # Apply effects based on environmental cause
        if drift_event.environmental_cause == "weather_change":
            modified_sample.iq_data = self._apply_atmospheric_effects(
                modified_sample.iq_data, drift_event.severity
            )
            modified_sample.snr -= drift_event.severity * 5  # SNR degradation
            
        elif drift_event.environmental_cause == "interference_increase":
            modified_sample.iq_data = self._add_interference_drift(
                modified_sample.iq_data, drift_event.severity
            )
            modified_sample.snr -= drift_event.severity * 8
            
        elif drift_event.environmental_cause == "hardware_aging":
            modified_sample.iq_data = self._apply_hardware_degradation(
                modified_sample.iq_data, drift_event.severity
            )
            modified_sample.snr -= drift_event.severity * 3
            
        elif drift_event.environmental_cause == "frequency_reallocation":
            modified_sample.iq_data = self._apply_frequency_shift(
                modified_sample.iq_data, drift_event.severity
            )
            
        elif drift_event.environmental_cause == "seasonal_variation":
            modified_sample.iq_data = self._apply_seasonal_effects(
                modified_sample.iq_data, drift_event.severity
            )
            modified_sample.snr -= drift_event.severity * 2
        
        # Add drift metadata
        modified_sample.metadata.update({
            "drift_applied": True,
            "drift_type": drift_event.drift_type,
            "drift_severity": drift_event.severity,
            "drift_cause": drift_event.environmental_cause,
            "drift_timestamp": drift_event.timestamp.isoformat()
        })
        
        return modified_sample
    
    def _apply_atmospheric_effects(self, iq_data: np.ndarray, severity: float) -> np.ndarray:
        """Apply atmospheric propagation effects."""
        
        # Simulate multipath fading due to atmospheric conditions
        num_paths = int(2 + severity * 3)
        result = np.zeros_like(iq_data, dtype=complex)
        
        for path in range(num_paths):
            delay = random.randint(0, int(severity * 10))
            attenuation = random.uniform(0.5, 1.0) if path == 0 else random.uniform(0.1, 0.4)
            phase_shift = random.uniform(0, 2 * np.pi)
            
            delayed_signal = np.roll(iq_data, delay) * attenuation * np.exp(1j * phase_shift)
            result += delayed_signal
        
        # Add atmospheric noise
        noise_power = severity * 0.1
        noise = (np.random.normal(0, np.sqrt(noise_power/2), len(iq_data)) + 
                1j * np.random.normal(0, np.sqrt(noise_power/2), len(iq_data)))
        
        return (result / num_paths) + noise
    
    def _add_interference_drift(self, iq_data: np.ndarray, severity: float) -> np.ndarray:
        """Add time-varying interference."""
        
        # Generate interference signal
        interference_power = severity * 0.3
        
        # Simulate multiple interference sources
        interference = np.zeros_like(iq_data, dtype=complex)
        
        # Broadband interference
        broadband = (np.random.normal(0, np.sqrt(interference_power/4), len(iq_data)) + 
                    1j * np.random.normal(0, np.sqrt(interference_power/4), len(iq_data)))
        interference += broadband
        
        # Narrowband interference
        for _ in range(int(severity * 3)):
            freq = random.uniform(-0.4, 0.4)
            t = np.arange(len(iq_data))
            narrowband = np.exp(1j * 2 * np.pi * freq * t) * np.sqrt(interference_power/6)
            interference += narrowband
        
        return iq_data + interference
    
    def _apply_hardware_degradation(self, iq_data: np.ndarray, severity: float) -> np.ndarray:
        """Apply hardware aging effects."""
        
        # Phase noise increase
        phase_noise_std = severity * 0.1
        phase_noise = np.random.normal(0, phase_noise_std, len(iq_data))
        degraded_signal = iq_data * np.exp(1j * phase_noise)
        
        # Amplitude distortion
        amplitude_distortion = 1 + severity * 0.1 * np.random.normal(0, 0.1, len(iq_data))
        degraded_signal *= amplitude_distortion
        
        # DC offset
        dc_offset = severity * 0.05 * (1 + 1j)
        degraded_signal += dc_offset
        
        return degraded_signal
    
    def _apply_frequency_shift(self, iq_data: np.ndarray, severity: float) -> np.ndarray:
        """Apply frequency offset due to regulatory changes."""
        
        # Frequency offset
        freq_offset = severity * 0.1  # Normalized frequency
        t = np.arange(len(iq_data))
        
        return iq_data * np.exp(1j * 2 * np.pi * freq_offset * t)
    
    def _apply_seasonal_effects(self, iq_data: np.ndarray, severity: float) -> np.ndarray:
        """Apply seasonal propagation variations."""
        
        # Slow fading variations
        fade_rate = 0.01  # Very slow variations
        t = np.arange(len(iq_data))
        fade_envelope = 1 + severity * 0.2 * np.sin(2 * np.pi * fade_rate * t)
        
        return iq_data * fade_envelope


class DriftDetector:
    """Detects concept drift in signal classification performance."""
    
    def __init__(self, window_size: int = 100, threshold: float = 0.5):
        self.window_size = window_size
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)
        
        # Historical data for drift detection
        self.accuracy_history = []
        self.prediction_history = []
        self.confidence_history = []
    
    def detect_drift(self, current_accuracy: float, current_predictions: List[int],
                    current_confidences: List[float]) -> DriftDetectionResult:
        """Detect concept drift using multiple methods."""
        
        # Update history
        self.accuracy_history.append(current_accuracy)
        self.prediction_history.extend(current_predictions)
        self.confidence_history.extend(current_confidences)
        
        # Keep only recent history
        if len(self.accuracy_history) > self.window_size:
            self.accuracy_history = self.accuracy_history[-self.window_size:]
        if len(self.prediction_history) > self.window_size * 10:
            self.prediction_history = self.prediction_history[-self.window_size * 10:]
        if len(self.confidence_history) > self.window_size * 10:
            self.confidence_history = self.confidence_history[-self.window_size * 10:]
        
        # Apply multiple drift detection methods
        drift_scores = []
        
        # Method 1: Accuracy-based detection
        if len(self.accuracy_history) >= 20:
            accuracy_drift = self._detect_accuracy_drift()
            drift_scores.append(accuracy_drift)
        
        # Method 2: Distribution-based detection (Kolmogorov-Smirnov test)
        if len(self.confidence_history) >= 50:
            distribution_drift = self._detect_distribution_drift()
            drift_scores.append(distribution_drift)
        
        # Method 3: Prediction stability
        if len(self.prediction_history) >= 50:
            stability_drift = self._detect_stability_drift()
            drift_scores.append(stability_drift)
        
        # Combine drift scores
        if drift_scores:
            combined_score = np.mean(drift_scores)
            drift_detected = combined_score > self.threshold
            confidence = min(1.0, combined_score * 2)
        else:
            combined_score = 0.0
            drift_detected = False
            confidence = 0.0
        
        result = DriftDetectionResult(
            drift_detected=drift_detected,
            drift_score=combined_score,
            confidence=confidence,
            affected_classes=[],  # Would need more sophisticated analysis
            detection_method="combined",
            timestamp=datetime.now()
        )
        
        if drift_detected:
            self.logger.warning(f"Concept drift detected! Score: {combined_score:.3f}")
        
        return result
    
    def _detect_accuracy_drift(self) -> float:
        """Detect drift based on accuracy degradation."""
        
        if len(self.accuracy_history) < 20:
            return 0.0
        
        # Compare recent accuracy with historical baseline
        recent_window = 10
        baseline_window = 10
        
        recent_accuracy = np.mean(self.accuracy_history[-recent_window:])
        baseline_accuracy = np.mean(self.accuracy_history[-recent_window-baseline_window:-recent_window])
        
        # Calculate relative degradation
        if baseline_accuracy > 0:
            degradation = (baseline_accuracy - recent_accuracy) / baseline_accuracy
            return max(0.0, degradation * 2)  # Scale to 0-1 range
        
        return 0.0
    
    def _detect_distribution_drift(self) -> float:
        """Detect drift using statistical tests on confidence distributions."""
        
        if len(self.confidence_history) < 50:
            return 0.0
        
        # Split into two windows
        mid_point = len(self.confidence_history) // 2
        window1 = self.confidence_history[:mid_point]
        window2 = self.confidence_history[mid_point:]
        
        # Kolmogorov-Smirnov test
        try:
            statistic, p_value = stats.ks_2samp(window1, window2)
            
            # Convert p-value to drift score (lower p-value = higher drift)
            drift_score = 1.0 - p_value
            return min(1.0, drift_score)
            
        except Exception as e:
            self.logger.warning(f"Error in distribution drift detection: {e}")
            return 0.0
    
    def _detect_stability_drift(self) -> float:
        """Detect drift based on prediction stability."""
        
        if len(self.prediction_history) < 50:
            return 0.0
        
        # Calculate prediction entropy in sliding windows
        window_size = 25
        entropies = []
        
        for i in range(len(self.prediction_history) - window_size + 1):
            window = self.prediction_history[i:i + window_size]
            entropy = self._calculate_entropy(window)
            entropies.append(entropy)
        
        if len(entropies) < 2:
            return 0.0
        
        # Compare recent entropy with baseline
        recent_entropy = np.mean(entropies[-5:])
        baseline_entropy = np.mean(entropies[:5])
        
        # Higher entropy indicates more instability
        if baseline_entropy > 0:
            entropy_increase = (recent_entropy - baseline_entropy) / baseline_entropy
            return max(0.0, min(1.0, entropy_increase))
        
        return 0.0
    
    def _calculate_entropy(self, predictions: List[int]) -> float:
        """Calculate entropy of prediction distribution."""
        
        if not predictions:
            return 0.0
        
        # Count occurrences
        unique, counts = np.unique(predictions, return_counts=True)
        probabilities = counts / len(predictions)
        
        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        
        return entropy


class ConceptDriftDemonstration:
    """Main demonstration class for concept drift handling."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.simulator = ConceptDriftSimulator()
        self.detector = DriftDetector()
        self.signal_processor = SignalProcessor()
    
    def run_drift_demonstration(self, samples: List[SignalSample], 
                              duration_hours: int = 24) -> Dict[str, Any]:
        """Run comprehensive concept drift demonstration."""
        
        self.logger.info(f"Starting concept drift demonstration ({duration_hours} hours)")
        
        # Generate drift timeline
        drift_events = self.simulator.generate_drift_timeline(
            duration_hours=duration_hours, num_events=5
        )
        
        # Initialize model
        model = self._initialize_model(samples)
        
        # Simulation results
        results = {
            "drift_events": [],
            "detection_results": [],
            "adaptation_results": [],
            "performance_timeline": [],
            "model_versions": []
        }
        
        # Simulate time progression
        current_time = datetime.now()
        time_step = timedelta(hours=1)
        
        for hour in range(duration_hours):
            current_time += time_step
            
            # Check for drift events at this time
            active_drifts = [
                event for event in drift_events 
                if abs((event.timestamp - current_time).total_seconds()) < 1800  # Within 30 minutes
            ]
            
            # Apply drift effects to samples
            current_samples = samples.copy()
            for drift_event in active_drifts:
                current_samples = self.simulator.apply_drift_to_samples(
                    current_samples, drift_event
                )
                
                results["drift_events"].append({
                    "timestamp": drift_event.timestamp.isoformat(),
                    "type": drift_event.drift_type,
                    "severity": drift_event.severity,
                    "description": drift_event.description,
                    "affected_modulations": drift_event.affected_modulations
                })
            
            # Evaluate model performance
            accuracy, predictions, confidences = self._evaluate_model_performance(
                model, current_samples[:100]  # Sample subset for efficiency
            )
            
            # Detect drift
            detection_result = self.detector.detect_drift(
                accuracy, predictions, confidences
            )
            
            results["detection_results"].append({
                "timestamp": current_time.isoformat(),
                "drift_detected": detection_result.drift_detected,
                "drift_score": detection_result.drift_score,
                "confidence": detection_result.confidence
            })
            
            # Adapt model if drift detected
            if detection_result.drift_detected:
                self.logger.info(f"Adapting model at hour {hour} due to detected drift")
                
                adapted_model, adaptation_metrics = self._adapt_model(
                    model, current_samples[:200]
                )
                
                results["adaptation_results"].append({
                    "timestamp": current_time.isoformat(),
                    "adaptation_method": "incremental_learning",
                    "accuracy_before": accuracy,
                    "accuracy_after": adaptation_metrics["accuracy_after"],
                    "adaptation_time": adaptation_metrics["adaptation_time"]
                })
                
                model = adapted_model
                results["model_versions"].append({
                    "timestamp": current_time.isoformat(),
                    "version": len(results["model_versions"]) + 1,
                    "trigger": "drift_detection"
                })
            
            # Record performance
            results["performance_timeline"].append({
                "timestamp": current_time.isoformat(),
                "hour": hour,
                "accuracy": accuracy,
                "mean_confidence": np.mean(confidences),
                "active_drifts": len(active_drifts)
            })
            
            if hour % 6 == 0:  # Log every 6 hours
                self.logger.info(f"Hour {hour}: Accuracy = {accuracy:.3f}, "
                               f"Drift Score = {detection_result.drift_score:.3f}")
        
        # Generate summary
        results["summary"] = self._generate_drift_summary(results)
        
        self.logger.info("Concept drift demonstration completed")
        
        return results
    
    def _initialize_model(self, samples: List[SignalSample]) -> nn.Module:
        """Initialize and train baseline model."""
        
        # Extract features
        features = []
        labels = []
        label_map = {}
        
        for sample in samples[:1000]:  # Use subset for initialization
            try:
                feature_vector = self.signal_processor.extract_features(sample.iq_data)
                features.append(feature_vector)
                
                if sample.modulation_type not in label_map:
                    label_map[sample.modulation_type] = len(label_map)
                labels.append(label_map[sample.modulation_type])
                
            except Exception as e:
                continue
        
        if not features:
            raise ValueError("No valid features extracted")
        
        # Create and train model
        X = torch.FloatTensor(features)
        y = torch.LongTensor(labels)
        
        model = SignalClassificationModel(
            input_size=X.shape[1],
            num_classes=len(label_map)
        )
        
        # Quick training
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        model.train()
        for epoch in range(20):
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        
        self.logger.info("Baseline model initialized")
        
        return model
    
    def _evaluate_model_performance(self, model: nn.Module, 
                                  samples: List[SignalSample]) -> Tuple[float, List[int], List[float]]:
        """Evaluate model performance on current samples."""
        
        features = []
        true_labels = []
        label_map = {"BPSK": 0, "QPSK": 1, "8PSK": 2, "QAM16": 3, "QAM64": 4, "AM": 5, "FM": 6}
        
        for sample in samples:
            try:
                feature_vector = self.signal_processor.extract_features(sample.iq_data)
                features.append(feature_vector)
                true_labels.append(label_map.get(sample.modulation_type, 0))
            except Exception:
                continue
        
        if not features:
            return 0.0, [], []
        
        X = torch.FloatTensor(features)
        y_true = torch.LongTensor(true_labels)
        
        model.eval()
        with torch.no_grad():
            outputs = model(X)
            probabilities = torch.softmax(outputs, dim=1)
            _, predictions = torch.max(outputs, 1)
            
            # Calculate accuracy
            accuracy = (predictions == y_true).float().mean().item()
            
            # Get confidence scores (max probability)
            confidences = torch.max(probabilities, dim=1)[0].tolist()
            
            predictions_list = predictions.tolist()
        
        return accuracy, predictions_list, confidences
    
    def _adapt_model(self, model: nn.Module, 
                    samples: List[SignalSample]) -> Tuple[nn.Module, Dict[str, Any]]:
        """Adapt model to handle concept drift."""
        
        start_time = time.time()
        
        # Extract features from new samples
        features = []
        labels = []
        label_map = {"BPSK": 0, "QPSK": 1, "8PSK": 2, "QAM16": 3, "QAM64": 4, "AM": 5, "FM": 6}
        
        for sample in samples:
            try:
                feature_vector = self.signal_processor.extract_features(sample.iq_data)
                features.append(feature_vector)
                labels.append(label_map.get(sample.modulation_type, 0))
            except Exception:
                continue
        
        if not features:
            return model, {"accuracy_after": 0.0, "adaptation_time": 0.0}
        
        X = torch.FloatTensor(features)
        y = torch.LongTensor(labels)
        
        # Incremental learning with lower learning rate
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        
        model.train()
        for epoch in range(10):  # Fewer epochs for adaptation
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()
        
        # Evaluate adapted model
        accuracy_after, _, _ = self._evaluate_model_performance(model, samples)
        adaptation_time = time.time() - start_time
        
        metrics = {
            "accuracy_after": accuracy_after,
            "adaptation_time": adaptation_time,
            "samples_used": len(features)
        }
        
        return model, metrics
    
    def _generate_drift_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of drift demonstration results."""
        
        # Count drift events by type
        drift_events = results["drift_events"]
        drift_types = {}
        for event in drift_events:
            drift_type = event["type"]
            drift_types[drift_type] = drift_types.get(drift_type, 0) + 1
        
        # Calculate detection accuracy
        detection_results = results["detection_results"]
        total_detections = sum(1 for r in detection_results if r["drift_detected"])
        
        # Calculate performance statistics
        performance_timeline = results["performance_timeline"]
        accuracies = [p["accuracy"] for p in performance_timeline]
        
        summary = {
            "total_drift_events": len(drift_events),
            "drift_types_distribution": drift_types,
            "total_detections": total_detections,
            "detection_rate": total_detections / max(len(drift_events), 1),
            "model_adaptations": len(results["adaptation_results"]),
            "performance_statistics": {
                "mean_accuracy": np.mean(accuracies),
                "min_accuracy": np.min(accuracies),
                "max_accuracy": np.max(accuracies),
                "accuracy_std": np.std(accuracies)
            },
            "adaptation_effectiveness": self._calculate_adaptation_effectiveness(results)
        }
        
        return summary
    
    def _calculate_adaptation_effectiveness(self, results: Dict[str, Any]) -> Dict[str, float]:
        """Calculate effectiveness of model adaptations."""
        
        adaptations = results["adaptation_results"]
        
        if not adaptations:
            return {"mean_improvement": 0.0, "successful_adaptations": 0.0}
        
        improvements = []
        successful = 0
        
        for adaptation in adaptations:
            improvement = adaptation["accuracy_after"] - adaptation["accuracy_before"]
            improvements.append(improvement)
            if improvement > 0:
                successful += 1
        
        return {
            "mean_improvement": np.mean(improvements),
            "successful_adaptations": successful / len(adaptations),
            "max_improvement": np.max(improvements),
            "min_improvement": np.min(improvements)
        }