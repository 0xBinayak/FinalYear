"""
Comprehensive tests for the real-world demonstration system.
"""

import pytest
import asyncio
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import json

from .real_world_demo import RealWorldDemonstration, DemoConfig, run_demonstration
from .dataset_integration import DatasetIntegrator, LocationProfile, MultiLocationScenario
from .visualization import SignalVisualization, VisualizationConfig
from .comparison_engine import CentralizedComparison, TrainingConfig, ExperimentResults
from .concept_drift_demo import ConceptDriftDemonstration, ConceptDriftSimulator, DriftDetector
from src.common.interfaces import SignalSample


class TestDatasetIntegration:
    """Test dataset integration functionality."""
    
    def test_location_profile_creation(self):
        """Test creation of location profiles."""
        integrator = DatasetIntegrator()
        
        assert len(integrator.location_profiles) > 0
        
        # Check urban profile
        urban = integrator.location_profiles["urban_downtown"]
        assert urban.environment_type == "urban"
        assert urban.noise_floor_db < -90
        assert "cellular" in urban.interference_sources
    
    def test_realistic_scenario_creation(self):
        """Test creation of realistic multi-location scenario."""
        integrator = DatasetIntegrator()
        scenario = integrator.create_realistic_scenario()
        
        assert isinstance(scenario, MultiLocationScenario)
        assert len(scenario.locations) > 0
        assert sum(scenario.client_distribution.values()) > 0
        assert scenario.temporal_variation is True
    
    def test_location_characteristics_application(self):
        """Test application of location-specific characteristics."""
        integrator = DatasetIntegrator()
        
        # Create test sample
        sample = SignalSample(
            timestamp=datetime.now(),
            frequency=915e6,
            sample_rate=200e3,
            iq_data=np.random.randn(1024) + 1j * np.random.randn(1024),
            modulation_type="QPSK",
            snr=10.0,
            location=None,
            device_id="test_device",
            metadata={}
        )
        
        location = integrator.location_profiles["urban_downtown"]
        modified_samples = integrator.apply_location_characteristics([sample], location)
        
        assert len(modified_samples) == 1
        modified_sample = modified_samples[0]
        
        assert modified_sample.location["latitude"] == location.latitude
        assert modified_sample.location["longitude"] == location.longitude
        assert "location_profile" in modified_sample.metadata
    
    def test_temporal_variations(self):
        """Test creation of temporal variations."""
        integrator = DatasetIntegrator()
        
        # Create test samples
        samples = []
        for i in range(10):
            sample = SignalSample(
                timestamp=None,
                frequency=915e6,
                sample_rate=200e3,
                iq_data=np.random.randn(1024) + 1j * np.random.randn(1024),
                modulation_type="QPSK",
                snr=10.0,
                location=None,
                device_id=f"test_device_{i}",
                metadata={}
            )
            samples.append(sample)
        
        varied_samples = integrator.create_temporal_variations(samples, time_span_hours=24)
        
        assert len(varied_samples) == len(samples)
        
        # Check that timestamps are assigned
        for sample in varied_samples:
            assert sample.timestamp is not None
            assert "hour_of_day" in sample.metadata


class TestVisualization:
    """Test visualization functionality."""
    
    def test_visualization_config(self):
        """Test visualization configuration."""
        config = VisualizationConfig(
            update_interval_ms=200,
            max_history_points=500,
            enable_animation=False
        )
        
        assert config.update_interval_ms == 200
        assert config.max_history_points == 500
        assert config.enable_animation is False
    
    def test_signal_visualization_creation(self):
        """Test creation of signal visualization."""
        viz = SignalVisualization()
        assert viz.config is not None
        assert hasattr(viz, 'signal_history')
        assert hasattr(viz, 'classification_history')
    
    @patch('plotly.graph_objects.Figure')
    def test_constellation_plot_creation(self, mock_figure):
        """Test constellation plot creation."""
        viz = SignalVisualization()
        
        # Create test samples
        samples = []
        for mod_type in ["BPSK", "QPSK"]:
            for i in range(5):
                sample = SignalSample(
                    timestamp=datetime.now(),
                    frequency=915e6,
                    sample_rate=200e3,
                    iq_data=np.random.randn(128) + 1j * np.random.randn(128),
                    modulation_type=mod_type,
                    snr=10.0,
                    location=None,
                    device_id=f"test_device_{i}",
                    metadata={}
                )
                samples.append(sample)
        
        # This should not raise an exception
        fig = viz.create_signal_constellation_plot(samples)
        assert fig is not None


class TestComparisonEngine:
    """Test comparison engine functionality."""
    
    def test_training_config(self):
        """Test training configuration."""
        config = TrainingConfig(
            batch_size=16,
            learning_rate=0.01,
            num_epochs=10
        )
        
        assert config.batch_size == 16
        assert config.learning_rate == 0.01
        assert config.num_epochs == 10
    
    def test_comparison_engine_creation(self):
        """Test comparison engine creation."""
        engine = CentralizedComparison()
        assert engine.config is not None
        assert hasattr(engine, 'signal_processor')
    
    def test_dataset_preparation(self):
        """Test dataset preparation."""
        engine = CentralizedComparison()
        
        # Create test samples
        samples = []
        for mod_type in ["BPSK", "QPSK", "8PSK"]:
            for i in range(10):
                sample = SignalSample(
                    timestamp=datetime.now(),
                    frequency=915e6,
                    sample_rate=200e3,
                    iq_data=np.random.randn(128) + 1j * np.random.randn(128),
                    modulation_type=mod_type,
                    snr=10.0,
                    location=None,
                    device_id=f"test_device_{i}",
                    metadata={}
                )
                samples.append(sample)
        
        with patch.object(engine.signal_processor, 'extract_features', 
                         return_value=np.random.randn(64)):
            X, y, class_names = engine.prepare_dataset(samples)
            
            assert X.shape[0] == len(samples)
            assert len(y) == len(samples)
            assert len(class_names) == 3  # BPSK, QPSK, 8PSK


class TestConceptDriftDemo:
    """Test concept drift demonstration functionality."""
    
    def test_drift_simulator_creation(self):
        """Test drift simulator creation."""
        simulator = ConceptDriftSimulator()
        assert len(simulator.drift_scenarios) > 0
        assert "weather_change" in simulator.drift_scenarios
        assert "interference_increase" in simulator.drift_scenarios
    
    def test_drift_timeline_generation(self):
        """Test drift timeline generation."""
        simulator = ConceptDriftSimulator()
        events = simulator.generate_drift_timeline(duration_hours=24, num_events=3)
        
        assert len(events) == 3
        
        # Check that events are sorted by timestamp
        for i in range(1, len(events)):
            assert events[i].timestamp >= events[i-1].timestamp
    
    def test_drift_detector_creation(self):
        """Test drift detector creation."""
        detector = DriftDetector(window_size=50, threshold=0.6)
        
        assert detector.window_size == 50
        assert detector.threshold == 0.6
        assert len(detector.accuracy_history) == 0
    
    def test_drift_detection(self):
        """Test drift detection functionality."""
        detector = DriftDetector(window_size=10, threshold=0.5)
        
        # Simulate stable performance
        for i in range(15):
            result = detector.detect_drift(
                current_accuracy=0.9,
                current_predictions=[0, 1, 0, 1, 0],
                current_confidences=[0.9, 0.8, 0.9, 0.8, 0.9]
            )
        
        # Should not detect drift with stable performance
        assert result.drift_detected is False
        
        # Simulate performance degradation
        for i in range(10):
            result = detector.detect_drift(
                current_accuracy=0.5,  # Significant drop
                current_predictions=[0, 0, 0, 0, 0],  # Less diverse
                current_confidences=[0.6, 0.5, 0.6, 0.5, 0.6]  # Lower confidence
            )
        
        # Should detect drift with degraded performance
        assert result.drift_detected is True


class TestRealWorldDemo:
    """Test main demonstration orchestrator."""
    
    def test_demo_config(self):
        """Test demonstration configuration."""
        config = DemoConfig(
            duration_hours=12,
            num_clients=5,
            enable_visualization=False,
            save_results=False
        )
        
        assert config.duration_hours == 12
        assert config.num_clients == 5
        assert config.enable_visualization is False
        assert config.save_results is False
    
    def test_demo_creation(self):
        """Test demonstration creation."""
        config = DemoConfig(save_results=False, enable_visualization=False)
        demo = RealWorldDemonstration(config)
        
        assert demo.config == config
        assert hasattr(demo, 'dataset_integrator')
        assert hasattr(demo, 'comparison_engine')
        assert hasattr(demo, 'concept_drift_demo')
    
    @pytest.mark.asyncio
    async def test_dataset_integration_step(self):
        """Test dataset integration step."""
        config = DemoConfig(save_results=False, enable_visualization=False)
        demo = RealWorldDemonstration(config)
        
        # Mock the dataset integration
        mock_datasets = {
            "test_dataset": [
                SignalSample(
                    timestamp=datetime.now(),
                    frequency=915e6,
                    sample_rate=200e3,
                    iq_data=np.random.randn(128) + 1j * np.random.randn(128),
                    modulation_type="QPSK",
                    snr=10.0,
                    location=None,
                    device_id="test_device",
                    metadata={}
                )
            ]
        }
        
        with patch.object(demo.dataset_integrator, 'integrate_radioml_datasets', 
                         return_value=mock_datasets):
            datasets = await demo._integrate_datasets()
            
            assert "test_dataset" in datasets
            assert len(datasets["test_dataset"]) > 0
    
    def test_config_serialization(self):
        """Test configuration serialization."""
        config = DemoConfig(duration_hours=12, num_clients=5)
        demo = RealWorldDemonstration(config)
        
        serialized = demo._serialize_config()
        
        assert serialized["duration_hours"] == 12
        assert serialized["num_clients"] == 5
        assert isinstance(serialized, dict)
    
    def test_scenario_serialization(self):
        """Test scenario serialization."""
        demo = RealWorldDemonstration(DemoConfig(save_results=False))
        
        # Create test scenario
        location = LocationProfile(
            name="Test Location",
            latitude=40.0,
            longitude=-74.0,
            environment_type="urban",
            noise_floor_db=-95,
            interference_sources=["wifi", "cellular"],
            propagation_model="urban_canyon"
        )
        
        scenario = MultiLocationScenario(
            scenario_name="test_scenario",
            locations=[location],
            client_distribution={"Test Location": 5},
            data_distribution_strategy="iid",
            temporal_variation=True,
            concept_drift_enabled=False
        )
        
        serialized = demo._serialize_scenario(scenario)
        
        assert serialized["scenario_name"] == "test_scenario"
        assert len(serialized["locations"]) == 1
        assert serialized["locations"][0]["name"] == "Test Location"
    
    @pytest.mark.asyncio
    async def test_save_results(self):
        """Test results saving functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config = DemoConfig(save_results=True, output_dir=temp_dir)
            demo = RealWorldDemonstration(config)
            
            test_results = {
                "summary": {"test": "data"},
                "performance_metrics": {"accuracy": 0.9}
            }
            
            await demo._save_results(test_results)
            
            # Check that files were created
            files = os.listdir(temp_dir)
            json_files = [f for f in files if f.endswith('.json')]
            
            assert len(json_files) >= 1  # At least main results file
            
            # Check that main results file contains expected data
            results_files = [f for f in json_files if f.startswith('demo_results_')]
            assert len(results_files) == 1
            
            with open(os.path.join(temp_dir, results_files[0]), 'r') as f:
                saved_data = json.load(f)
                assert "summary" in saved_data
                assert "performance_metrics" in saved_data


class TestIntegration:
    """Integration tests for the complete demonstration system."""
    
    @pytest.mark.asyncio
    async def test_minimal_demonstration(self):
        """Test minimal demonstration run."""
        config = DemoConfig(
            duration_hours=1,
            num_clients=2,
            enable_visualization=False,
            enable_concept_drift=False,
            enable_comparison=False,
            save_results=False
        )
        
        # Mock external dependencies
        with patch('src.demonstration.dataset_integration.DatasetIntegrator.integrate_radioml_datasets') as mock_integrate:
            mock_integrate.return_value = {
                "test_dataset": [
                    SignalSample(
                        timestamp=datetime.now(),
                        frequency=915e6,
                        sample_rate=200e3,
                        iq_data=np.random.randn(128) + 1j * np.random.randn(128),
                        modulation_type="QPSK",
                        snr=10.0,
                        location=None,
                        device_id="test_device",
                        metadata={}
                    )
                ]
            }
            
            demo = RealWorldDemonstration(config)
            results = await demo.run_complete_demonstration()
            
            assert "scenario" in results
            assert "datasets" in results
            assert "summary" in results
            assert results.get("error") is None
    
    @pytest.mark.asyncio
    async def test_run_demonstration_function(self):
        """Test the convenience run_demonstration function."""
        config = DemoConfig(
            duration_hours=1,
            num_clients=2,
            enable_visualization=False,
            enable_concept_drift=False,
            enable_comparison=False,
            save_results=False
        )
        
        with patch('src.demonstration.dataset_integration.DatasetIntegrator.integrate_radioml_datasets') as mock_integrate:
            mock_integrate.return_value = {"test": []}
            
            results = await run_demonstration(config)
            
            assert isinstance(results, dict)
            assert "config" in results
            assert "summary" in results


if __name__ == "__main__":
    pytest.main([__file__])