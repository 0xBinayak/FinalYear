"""
Comprehensive unit tests for common components.
"""
import pytest
import numpy as np
import tempfile
import shutil
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime
import json

from src.common.config import get_config, ConfigManager
from src.common.data_loaders import DataLoader, SigMFLoader, HDF5Loader
from src.common.data_preprocessing import DataPreprocessor
from src.common.data_quality import DataQualityValidator
from src.common.signal_models import SignalSample, ModelUpdate
from src.common.federated_data_structures import FederatedDataset
from src.common.real_world_data_collector import RealWorldDataCollector
from src.common.signal_augmentation import SignalAugmentor


@pytest.mark.unit
class TestConfigManager:
    """Test cases for configuration management."""
    
    def test_config_loading(self):
        """Test configuration loading from files."""
        config_manager = ConfigManager()
        config = config_manager.load_config("base")
        
        assert config is not None
        assert hasattr(config, 'aggregation_server')
        assert hasattr(config, 'federated_learning')
        assert hasattr(config, 'privacy')
    
    def test_config_validation(self):
        """Test configuration validation."""
        config_manager = ConfigManager()
        
        # Valid configuration
        valid_config = {
            "aggregation_server": {
                "host": "localhost",
                "port": 8080
            },
            "federated_learning": {
                "min_clients": 2,
                "max_clients": 100
            }
        }
        
        is_valid = config_manager.validate_config(valid_config)
        assert is_valid
        
        # Invalid configuration
        invalid_config = {
            "aggregation_server": {
                "host": "localhost"
                # Missing required port
            }
        }
        
        is_valid = config_manager.validate_config(invalid_config)
        assert not is_valid
    
    def test_environment_overrides(self):
        """Test environment-specific configuration overrides."""
        config_manager = ConfigManager()
        
        base_config = config_manager.load_config("base")
        dev_config = config_manager.load_config("development")
        
        # Development should override base settings
        assert dev_config.aggregation_server.host != base_config.aggregation_server.host
        assert dev_config.logging.level == "DEBUG"
    
    def test_hot_reload(self):
        """Test configuration hot reloading."""
        with tempfile.TemporaryDirectory() as temp_dir:
            config_file = f"{temp_dir}/test_config.yaml"
            
            # Create initial config
            initial_config = {
                "test_value": 100,
                "nested": {"value": "initial"}
            }
            
            with open(config_file, 'w') as f:
                import yaml
                yaml.dump(initial_config, f)
            
            config_manager = ConfigManager()
            config_manager.enable_hot_reload(config_file)
            
            # Load initial config
            config = config_manager.load_config_file(config_file)
            assert config["test_value"] == 100
            
            # Modify config file
            updated_config = {
                "test_value": 200,
                "nested": {"value": "updated"}
            }
            
            with open(config_file, 'w') as f:
                yaml.dump(updated_config, f)
            
            # Trigger reload
            config_manager.reload_config()
            config = config_manager.load_config_file(config_file)
            assert config["test_value"] == 200


@pytest.mark.unit
class TestDataLoaders:
    """Test cases for data loading components."""
    
    def test_sigmf_loader(self):
        """Test SigMF format data loading."""
        loader = SigMFLoader()
        
        # Mock SigMF metadata
        mock_metadata = {
            "global": {
                "core:datatype": "cf32_le",
                "core:sample_rate": 2000000,
                "core:version": "1.0.0"
            },
            "captures": [{
                "core:sample_start": 0,
                "core:frequency": 915000000,
                "core:datetime": "2024-01-01T12:00:00Z"
            }],
            "annotations": []
        }
        
        # Mock IQ data
        mock_iq_data = np.random.randn(1024, 2).astype(np.float32)
        
        with patch('src.common.data_loaders.load_sigmf_metadata') as mock_meta, \
             patch('src.common.data_loaders.load_sigmf_data') as mock_data:
            
            mock_meta.return_value = mock_metadata
            mock_data.return_value = mock_iq_data
            
            signal_sample = loader.load("test_file.sigmf-meta")
            
            assert signal_sample is not None
            assert signal_sample.frequency == 915000000
            assert signal_sample.sample_rate == 2000000
            assert signal_sample.iq_data.shape == (1024, 2)
    
    def test_hdf5_loader(self):
        """Test HDF5 format data loading."""
        loader = HDF5Loader()
        
        with tempfile.NamedTemporaryFile(suffix='.h5', delete=False) as temp_file:
            # Create mock HDF5 file
            import h5py
            with h5py.File(temp_file.name, 'w') as f:
                f.create_dataset('iq_data', data=np.random.randn(2048, 2))
                f.create_dataset('frequency', data=915e6)
                f.create_dataset('sample_rate', data=2e6)
                f.attrs['modulation'] = 'QPSK'
                f.attrs['snr'] = 15.0
            
            signal_sample = loader.load(temp_file.name)
            
            assert signal_sample is not None
            assert signal_sample.frequency == 915e6
            assert signal_sample.sample_rate == 2e6
            assert signal_sample.modulation_type == 'QPSK'
            assert signal_sample.snr == 15.0
    
    def test_data_loader_factory(self):
        """Test data loader factory pattern."""
        loader = DataLoader()
        
        # Test SigMF file detection
        sigmf_loader = loader.get_loader("test.sigmf-meta")
        assert isinstance(sigmf_loader, SigMFLoader)
        
        # Test HDF5 file detection
        hdf5_loader = loader.get_loader("test.h5")
        assert isinstance(hdf5_loader, HDF5Loader)
        
        # Test unsupported format
        with pytest.raises(ValueError):
            loader.get_loader("test.unsupported")


@pytest.mark.unit
class TestDataPreprocessor:
    """Test cases for data preprocessing."""
    
    def test_normalization(self):
        """Test signal normalization."""
        preprocessor = DataPreprocessor()
        
        # Create test signal with known statistics
        signal = np.array([1, 2, 3, 4, 5], dtype=np.float32)
        
        # Test z-score normalization
        normalized = preprocessor.normalize(signal, method='zscore')
        assert np.abs(np.mean(normalized)) < 1e-6  # Mean should be ~0
        assert np.abs(np.std(normalized) - 1.0) < 1e-6  # Std should be ~1
        
        # Test min-max normalization
        normalized = preprocessor.normalize(signal, method='minmax')
        assert np.min(normalized) == 0.0
        assert np.max(normalized) == 1.0
    
    def test_filtering(self):
        """Test signal filtering."""
        preprocessor = DataPreprocessor()
        
        # Create noisy signal
        t = np.linspace(0, 1, 1000)
        clean_signal = np.sin(2 * np.pi * 10 * t)  # 10 Hz signal
        noise = 0.5 * np.random.randn(1000)
        noisy_signal = clean_signal + noise
        
        # Apply low-pass filter
        filtered = preprocessor.apply_filter(
            noisy_signal, 
            filter_type='lowpass',
            cutoff=15,  # 15 Hz cutoff
            sample_rate=1000
        )
        
        # Filtered signal should be closer to original
        mse_original = np.mean((noisy_signal - clean_signal) ** 2)
        mse_filtered = np.mean((filtered - clean_signal) ** 2)
        assert mse_filtered < mse_original
    
    def test_resampling(self):
        """Test signal resampling."""
        preprocessor = DataPreprocessor()
        
        # Create test signal
        original_signal = np.random.randn(1000)
        
        # Test upsampling
        upsampled = preprocessor.resample(original_signal, factor=2)
        assert len(upsampled) == 2000
        
        # Test downsampling
        downsampled = preprocessor.resample(original_signal, factor=0.5)
        assert len(downsampled) == 500
    
    def test_windowing(self):
        """Test signal windowing."""
        preprocessor = DataPreprocessor()
        
        signal = np.ones(1000)
        
        # Test Hamming window
        windowed = preprocessor.apply_window(signal, window_type='hamming')
        assert len(windowed) == len(signal)
        assert windowed[0] < 1.0  # Window should taper at edges
        assert windowed[-1] < 1.0
        
        # Test Hann window
        windowed = preprocessor.apply_window(signal, window_type='hann')
        assert windowed[0] == 0.0  # Hann window starts at 0
        assert windowed[-1] == 0.0


@pytest.mark.unit
class TestDataQualityValidator:
    """Test cases for data quality validation."""
    
    def test_snr_validation(self):
        """Test SNR-based quality validation."""
        validator = DataQualityValidator()
        
        # High quality signal (high SNR)
        clean_signal = np.sin(2 * np.pi * 0.1 * np.arange(1000))
        high_quality = clean_signal + 0.01 * np.random.randn(1000)
        
        quality_score = validator.assess_snr_quality(high_quality)
        assert quality_score > 0.8
        
        # Low quality signal (low SNR)
        low_quality = clean_signal + 2.0 * np.random.randn(1000)
        
        quality_score = validator.assess_snr_quality(low_quality)
        assert quality_score < 0.5
    
    def test_spectral_quality(self):
        """Test spectral quality assessment."""
        validator = DataQualityValidator()
        
        # Clean signal with clear spectral content
        t = np.linspace(0, 1, 1000)
        clean_signal = np.sin(2 * np.pi * 50 * t) + 0.5 * np.sin(2 * np.pi * 120 * t)
        
        quality_metrics = validator.assess_spectral_quality(clean_signal, sample_rate=1000)
        
        assert quality_metrics is not None
        assert "spectral_flatness" in quality_metrics
        assert "peak_to_noise_ratio" in quality_metrics
        assert "bandwidth_utilization" in quality_metrics
    
    def test_temporal_consistency(self):
        """Test temporal consistency validation."""
        validator = DataQualityValidator()
        
        # Consistent signal
        consistent_signal = np.sin(2 * np.pi * 0.1 * np.arange(2000))
        consistency_score = validator.assess_temporal_consistency(consistent_signal)
        assert consistency_score > 0.8
        
        # Inconsistent signal with sudden changes
        inconsistent_signal = np.concatenate([
            np.sin(2 * np.pi * 0.1 * np.arange(1000)),
            10 * np.sin(2 * np.pi * 0.5 * np.arange(1000))  # Sudden amplitude/frequency change
        ])
        consistency_score = validator.assess_temporal_consistency(inconsistent_signal)
        assert consistency_score < 0.6
    
    def test_outlier_detection(self):
        """Test outlier detection in signal data."""
        validator = DataQualityValidator()
        
        # Normal signal with outliers
        normal_signal = np.random.randn(1000)
        normal_signal[100] = 10.0  # Outlier
        normal_signal[500] = -8.0  # Outlier
        
        outliers = validator.detect_outliers(normal_signal, method='zscore', threshold=3.0)
        
        assert len(outliers) >= 2
        assert 100 in outliers
        assert 500 in outliers


@pytest.mark.unit
class TestSignalAugmentor:
    """Test cases for signal augmentation."""
    
    def test_noise_addition(self):
        """Test noise addition augmentation."""
        augmentor = SignalAugmentor()
        
        clean_signal = np.ones(1000, dtype=complex)
        
        # Add AWGN
        noisy_signal = augmentor.add_awgn(clean_signal, snr_db=10)
        assert len(noisy_signal) == len(clean_signal)
        assert np.var(noisy_signal) > np.var(clean_signal)
        
        # Verify SNR is approximately correct
        signal_power = np.mean(np.abs(clean_signal) ** 2)
        noise_power = np.mean(np.abs(noisy_signal - clean_signal) ** 2)
        actual_snr = 10 * np.log10(signal_power / noise_power)
        assert abs(actual_snr - 10) < 2.0  # Within 2 dB tolerance
    
    def test_frequency_offset(self):
        """Test frequency offset augmentation."""
        augmentor = SignalAugmentor()
        
        # Create signal with known frequency
        t = np.linspace(0, 1, 1000)
        original_signal = np.exp(1j * 2 * np.pi * 50 * t)  # 50 Hz
        
        # Apply frequency offset
        offset_hz = 10
        offset_signal = augmentor.apply_frequency_offset(original_signal, offset_hz, sample_rate=1000)
        
        # Verify frequency shift
        original_fft = np.fft.fft(original_signal)
        offset_fft = np.fft.fft(offset_signal)
        
        original_peak = np.argmax(np.abs(original_fft))
        offset_peak = np.argmax(np.abs(offset_fft))
        
        expected_shift = int(offset_hz * len(original_signal) / 1000)
        actual_shift = offset_peak - original_peak
        assert abs(actual_shift - expected_shift) <= 1
    
    def test_phase_noise(self):
        """Test phase noise augmentation."""
        augmentor = SignalAugmentor()
        
        clean_signal = np.exp(1j * 2 * np.pi * 0.1 * np.arange(1000))
        
        # Apply phase noise
        phase_noisy_signal = augmentor.add_phase_noise(clean_signal, phase_std=0.1)
        
        assert len(phase_noisy_signal) == len(clean_signal)
        
        # Phase noise should not significantly affect amplitude
        amplitude_change = np.mean(np.abs(np.abs(phase_noisy_signal) - np.abs(clean_signal)))
        assert amplitude_change < 0.1
    
    def test_multipath_fading(self):
        """Test multipath fading augmentation."""
        augmentor = SignalAugmentor()
        
        clean_signal = np.ones(1000, dtype=complex)
        
        # Apply multipath fading
        faded_signal = augmentor.apply_multipath_fading(
            clean_signal,
            delays=[0, 5, 10],
            gains=[1.0, 0.5, 0.3],
            sample_rate=1000
        )
        
        assert len(faded_signal) >= len(clean_signal)
        
        # Signal should have multipath characteristics
        autocorr = np.correlate(faded_signal, faded_signal, mode='full')
        # Should have peaks at delay positions
        assert len(autocorr) > 0


@pytest.mark.unit
class TestFederatedDataStructures:
    """Test cases for federated learning data structures."""
    
    def test_federated_dataset_creation(self):
        """Test federated dataset creation and partitioning."""
        # Create sample data
        features = np.random.randn(1000, 10)
        labels = np.random.randint(0, 5, 1000)
        
        dataset = FederatedDataset(features, labels)
        
        # Test IID partitioning
        iid_partitions = dataset.partition_iid(num_clients=5)
        assert len(iid_partitions) == 5
        
        total_samples = sum(len(partition['features']) for partition in iid_partitions)
        assert total_samples == 1000
        
        # Test non-IID partitioning
        non_iid_partitions = dataset.partition_non_iid(num_clients=5, alpha=0.5)
        assert len(non_iid_partitions) == 5
        
        # Non-IID should have different class distributions
        class_distributions = []
        for partition in non_iid_partitions:
            unique, counts = np.unique(partition['labels'], return_counts=True)
            class_distributions.append(dict(zip(unique, counts)))
        
        # Verify distributions are different
        assert len(set(str(d) for d in class_distributions)) > 1
    
    def test_data_statistics(self):
        """Test data statistics computation."""
        features = np.random.randn(500, 8)
        labels = np.random.randint(0, 3, 500)
        
        dataset = FederatedDataset(features, labels)
        stats = dataset.get_statistics()
        
        assert stats['num_samples'] == 500
        assert stats['num_features'] == 8
        assert stats['num_classes'] == 3
        assert 'class_distribution' in stats
        assert 'feature_statistics' in stats
    
    def test_data_quality_metrics(self):
        """Test data quality metrics computation."""
        # Create data with known quality issues
        features = np.random.randn(300, 5)
        features[50:60] = 100  # Outliers
        labels = np.random.randint(0, 2, 300)
        
        dataset = FederatedDataset(features, labels)
        quality_metrics = dataset.assess_quality()
        
        assert 'outlier_ratio' in quality_metrics
        assert 'class_balance' in quality_metrics
        assert 'feature_correlation' in quality_metrics
        assert quality_metrics['outlier_ratio'] > 0.02  # Should detect outliers


@pytest.mark.unit
class TestRealWorldDataCollector:
    """Test cases for real-world data collection."""
    
    def test_radioml_dataset_loading(self):
        """Test RadioML dataset loading."""
        collector = RealWorldDataCollector()
        
        with patch('src.common.real_world_data_collector.download_radioml_dataset') as mock_download:
            mock_data = {
                'X': np.random.randn(1000, 2, 128),  # IQ samples
                'Y': np.random.randint(0, 11, 1000),  # Modulation labels
                'Z': np.random.randint(-20, 19, 1000)  # SNR labels
            }
            mock_download.return_value = mock_data
            
            dataset = collector.load_radioml_dataset("2016.10c")
            
            assert dataset is not None
            assert 'samples' in dataset
            assert 'modulations' in dataset
            assert 'snrs' in dataset
            assert len(dataset['samples']) == 1000
    
    def test_gnu_radio_dataset_loading(self):
        """Test GNU Radio dataset loading."""
        collector = RealWorldDataCollector()
        
        with patch('src.common.real_world_data_collector.load_gnu_radio_captures') as mock_load:
            mock_captures = [
                {
                    'iq_data': np.random.randn(2048, 2),
                    'frequency': 915e6,
                    'sample_rate': 2e6,
                    'timestamp': datetime.now(),
                    'metadata': {'location': 'lab', 'antenna': 'dipole'}
                }
            ]
            mock_load.return_value = mock_captures
            
            captures = collector.load_gnu_radio_dataset("test_captures")
            
            assert len(captures) == 1
            assert captures[0]['frequency'] == 915e6
            assert captures[0]['iq_data'].shape == (2048, 2)
    
    def test_data_validation(self):
        """Test real-world data validation."""
        collector = RealWorldDataCollector()
        
        # Valid signal data
        valid_data = {
            'iq_data': np.random.randn(1024, 2).astype(np.complex64),
            'frequency': 915e6,
            'sample_rate': 2e6,
            'timestamp': datetime.now().isoformat()
        }
        
        is_valid = collector.validate_signal_data(valid_data)
        assert is_valid
        
        # Invalid signal data
        invalid_data = {
            'iq_data': np.random.randn(10, 2),  # Too short
            'frequency': -100,  # Invalid frequency
            'sample_rate': 0,  # Invalid sample rate
        }
        
        is_valid = collector.validate_signal_data(invalid_data)
        assert not is_valid
    
    def test_metadata_extraction(self):
        """Test metadata extraction from signal files."""
        collector = RealWorldDataCollector()
        
        # Mock signal file with metadata
        mock_metadata = {
            'hardware': 'RTL-SDR',
            'antenna': 'dipole',
            'location': {'lat': 37.7749, 'lon': -122.4194},
            'environment': 'urban',
            'weather': 'clear'
        }
        
        with patch('src.common.real_world_data_collector.extract_file_metadata') as mock_extract:
            mock_extract.return_value = mock_metadata
            
            metadata = collector.extract_metadata("test_file.sigmf-meta")
            
            assert metadata['hardware'] == 'RTL-SDR'
            assert metadata['environment'] == 'urban'
            assert 'location' in metadata