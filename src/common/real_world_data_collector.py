"""
Real-world signal data collection system with integration to public datasets
"""
import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import logging
from dataclasses import dataclass, asdict

from .interfaces import SignalSample
from .data_loaders import SigMFLoader, RadioMLLoader, BinaryIQLoader
from .dataset_downloader import DatasetDownloader
from .signal_augmentation import ChannelSimulator, EnvironmentalEffects, ChannelConfig


@dataclass
class CollectionConfig:
    """Configuration for data collection"""
    # Dataset sources
    enable_radioml: bool = True
    enable_gnu_radio: bool = True
    enable_sigmf: bool = True
    enable_binary_iq: bool = True
    
    # Augmentation settings
    enable_channel_effects: bool = True
    augmentation_factor: int = 3  # Number of augmented versions per sample
    
    # Environmental effects
    enable_weather_effects: bool = True
    weather_conditions: List[str] = None
    enable_urban_effects: bool = True
    urban_environments: List[str] = None
    
    # Quality control
    min_snr_db: float = -20.0
    max_snr_db: float = 40.0
    min_sample_length: int = 512
    max_sample_length: int = 8192
    
    # Output settings
    output_format: str = 'hdf5'  # hdf5, sigmf, numpy
    save_metadata: bool = True
    compress_data: bool = True


class RealWorldDataCollector:
    """Collect and process real-world signal data from multiple sources"""
    
    def __init__(self, config: CollectionConfig = None, data_dir: str = "data"):
        self.config = config or CollectionConfig()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Initialize default lists
        if self.config.weather_conditions is None:
            self.config.weather_conditions = ['clear', 'rain', 'fog', 'atmospheric_ducting']
        if self.config.urban_environments is None:
            self.config.urban_environments = ['rural', 'suburban', 'dense_urban', 'indoor']
        
        # Initialize components
        self.dataset_downloader = DatasetDownloader(str(self.data_dir))
        self.channel_simulator = ChannelSimulator()
        
        # Initialize loaders
        self.loaders = {
            'sigmf': SigMFLoader(),
            'radioml': RadioMLLoader(),
            'binary_iq': BinaryIQLoader()
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def collect_all_datasets(self, force_redownload: bool = False) -> Dict[str, List[SignalSample]]:
        """Collect data from all configured sources"""
        collected_data = {}
        
        if self.config.enable_radioml:
            self.logger.info("Collecting RadioML datasets...")
            radioml_data = self._collect_radioml_data(force_redownload)
            collected_data['radioml'] = radioml_data
        
        if self.config.enable_gnu_radio:
            self.logger.info("Collecting GNU Radio datasets...")
            gnu_radio_data = self._collect_gnu_radio_data(force_redownload)
            collected_data['gnu_radio'] = gnu_radio_data
        
        if self.config.enable_sigmf:
            self.logger.info("Collecting SigMF datasets...")
            sigmf_data = self._collect_sigmf_data()
            collected_data['sigmf'] = sigmf_data
        
        if self.config.enable_binary_iq:
            self.logger.info("Collecting Binary IQ datasets...")
            binary_data = self._collect_binary_iq_data()
            collected_data['binary_iq'] = binary_data
        
        return collected_data
    
    def _collect_radioml_data(self, force_redownload: bool = False) -> List[SignalSample]:
        """Collect RadioML datasets"""
        samples = []
        
        try:
            # RadioML 2016.10
            if not self.dataset_downloader._is_dataset_downloaded('radioml_2016_10') or force_redownload:
                self.dataset_downloader.download_dataset('radioml_2016_10', force_redownload)
            
            radioml_2016_samples = self.dataset_downloader.load_radioml_2016_10()
            samples.extend(radioml_2016_samples)
            self.logger.info(f"Loaded {len(radioml_2016_samples)} samples from RadioML 2016.10")
            
        except Exception as e:
            self.logger.warning(f"Failed to load RadioML 2016.10: {e}")
        
        try:
            # RadioML 2018.01
            if not self.dataset_downloader._is_dataset_downloaded('radioml_2018_01') or force_redownload:
                self.dataset_downloader.download_dataset('radioml_2018_01', force_redownload)
            
            radioml_2018_samples = self.dataset_downloader.load_radioml_2018_01()
            samples.extend(radioml_2018_samples)
            self.logger.info(f"Loaded {len(radioml_2018_samples)} samples from RadioML 2018.01")
            
        except Exception as e:
            self.logger.warning(f"Failed to load RadioML 2018.01: {e}")
        
        return samples
    
    def _collect_gnu_radio_data(self, force_redownload: bool = False) -> List[SignalSample]:
        """Collect GNU Radio test data"""
        samples = []
        
        try:
            if not self.dataset_downloader._is_dataset_downloaded('gnu_radio_captures') or force_redownload:
                self.dataset_downloader.download_dataset('gnu_radio_captures', force_redownload)
            
            # Look for GNU Radio data files
            gnu_radio_dir = self.data_dir / 'gnu_radio_captures' / 'test_data'
            if gnu_radio_dir.exists():
                for file_path in gnu_radio_dir.rglob('*.dat'):
                    try:
                        binary_loader = BinaryIQLoader()
                        file_samples = binary_loader.load_signal_data(str(file_path))
                        samples.extend(file_samples)
                    except Exception as e:
                        self.logger.warning(f"Failed to load {file_path}: {e}")
            
            self.logger.info(f"Loaded {len(samples)} samples from GNU Radio datasets")
            
        except Exception as e:
            self.logger.warning(f"Failed to collect GNU Radio data: {e}")
        
        return samples
    
    def _collect_sigmf_data(self) -> List[SignalSample]:
        """Collect SigMF format data"""
        samples = []
        
        # Look for SigMF files in data directory
        sigmf_files = list(self.data_dir.rglob('*.sigmf-meta'))
        
        for meta_file in sigmf_files:
            try:
                sigmf_loader = SigMFLoader()
                file_samples = sigmf_loader.load_signal_data(str(meta_file.with_suffix('')))
                samples.extend(file_samples)
                self.logger.info(f"Loaded {len(file_samples)} samples from {meta_file}")
            except Exception as e:
                self.logger.warning(f"Failed to load SigMF file {meta_file}: {e}")
        
        return samples
    
    def _collect_binary_iq_data(self) -> List[SignalSample]:
        """Collect binary IQ files"""
        samples = []
        
        # Look for binary IQ files
        binary_extensions = ['.iq', '.dat', '.bin', '.raw']
        
        for ext in binary_extensions:
            binary_files = list(self.data_dir.rglob(f'*{ext}'))
            
            for binary_file in binary_files:
                # Skip if it's part of a SigMF dataset
                if binary_file.with_suffix('.sigmf-meta').exists():
                    continue
                
                try:
                    binary_loader = BinaryIQLoader()
                    file_samples = binary_loader.load_signal_data(str(binary_file))
                    samples.extend(file_samples)
                    self.logger.info(f"Loaded {len(file_samples)} samples from {binary_file}")
                except Exception as e:
                    self.logger.warning(f"Failed to load binary file {binary_file}: {e}")
        
        return samples
    
    def apply_realistic_augmentation(self, samples: List[SignalSample]) -> List[SignalSample]:
        """Apply realistic channel effects and environmental conditions"""
        if not self.config.enable_channel_effects:
            return samples
        
        self.logger.info(f"Applying realistic augmentation to {len(samples)} samples...")
        
        augmented_samples = []
        
        for sample in samples:
            # Apply quality filters
            if not self._passes_quality_check(sample):
                continue
            
            # Add original sample
            augmented_samples.append(sample)
            
            # Generate augmented versions
            for _ in range(self.config.augmentation_factor - 1):
                # Apply channel effects
                channel_sample = self.channel_simulator._apply_single_channel_realization(sample)
                
                # Apply environmental effects
                if self.config.enable_weather_effects:
                    weather_condition = np.random.choice(self.config.weather_conditions)
                    channel_sample = EnvironmentalEffects.apply_weather_effects(
                        [channel_sample], weather_condition
                    )[0]
                
                if self.config.enable_urban_effects:
                    urban_environment = np.random.choice(self.config.urban_environments)
                    channel_sample = EnvironmentalEffects.apply_urban_effects(
                        [channel_sample], urban_environment
                    )[0]
                
                augmented_samples.append(channel_sample)
        
        self.logger.info(f"Generated {len(augmented_samples)} augmented samples")
        return augmented_samples
    
    def _passes_quality_check(self, sample: SignalSample) -> bool:
        """Check if sample meets quality criteria"""
        # SNR check
        if sample.snr < self.config.min_snr_db or sample.snr > self.config.max_snr_db:
            return False
        
        # Length check
        if len(sample.iq_data) < self.config.min_sample_length or len(sample.iq_data) > self.config.max_sample_length:
            return False
        
        # Data validity check
        if np.any(np.isnan(sample.iq_data)) or np.any(np.isinf(sample.iq_data)):
            return False
        
        # Dynamic range check
        max_amplitude = np.max(np.abs(sample.iq_data))
        if max_amplitude == 0 or max_amplitude > 100:
            return False
        
        return True
    
    def save_collected_data(self, data: Dict[str, List[SignalSample]], output_dir: str = None) -> Dict[str, str]:
        """Save collected data in specified format"""
        if output_dir is None:
            output_dir = self.data_dir / "collected_data"
        else:
            output_dir = Path(output_dir)
        
        output_dir.mkdir(exist_ok=True)
        saved_files = {}
        
        for dataset_name, samples in data.items():
            if not samples:
                continue
            
            self.logger.info(f"Saving {len(samples)} samples from {dataset_name}...")
            
            if self.config.output_format == 'hdf5':
                output_file = output_dir / f"{dataset_name}_collected.h5"
                self._save_as_hdf5(samples, output_file)
                saved_files[dataset_name] = str(output_file)
                
            elif self.config.output_format == 'sigmf':
                output_file = output_dir / f"{dataset_name}_collected"
                self._save_as_sigmf(samples, output_file)
                saved_files[dataset_name] = str(output_file)
                
            elif self.config.output_format == 'numpy':
                output_file = output_dir / f"{dataset_name}_collected.npz"
                self._save_as_numpy(samples, output_file)
                saved_files[dataset_name] = str(output_file)
        
        # Save collection metadata
        metadata_file = output_dir / "collection_metadata.json"
        self._save_collection_metadata(data, metadata_file)
        
        return saved_files
    
    def _save_as_hdf5(self, samples: List[SignalSample], output_file: Path):
        """Save samples in HDF5 format"""
        import h5py
        
        with h5py.File(output_file, 'w') as f:
            # Create datasets
            iq_data_list = []
            metadata_list = []
            
            for i, sample in enumerate(samples):
                # Store IQ data
                iq_group = f.create_group(f'sample_{i}')
                iq_group.create_dataset('iq_data', data=sample.iq_data, compression='gzip' if self.config.compress_data else None)
                
                # Store metadata
                iq_group.attrs['timestamp'] = sample.timestamp.isoformat() if sample.timestamp else ''
                iq_group.attrs['frequency'] = sample.frequency
                iq_group.attrs['sample_rate'] = sample.sample_rate
                iq_group.attrs['modulation_type'] = sample.modulation_type
                iq_group.attrs['snr'] = sample.snr
                iq_group.attrs['device_id'] = sample.device_id
                
                if sample.location:
                    iq_group.attrs['latitude'] = sample.location.get('latitude', 0.0)
                    iq_group.attrs['longitude'] = sample.location.get('longitude', 0.0)
                    iq_group.attrs['altitude'] = sample.location.get('altitude', 0.0)
                
                # Store additional metadata as JSON string
                if self.config.save_metadata:
                    iq_group.attrs['metadata'] = json.dumps(sample.metadata)
    
    def _save_as_sigmf(self, samples: List[SignalSample], output_file: Path):
        """Save samples in SigMF format"""
        # For simplicity, save as single SigMF file with multiple annotations
        if not samples:
            return
        
        # Concatenate all IQ data
        all_iq_data = np.concatenate([sample.iq_data for sample in samples])
        
        # Create SigMF metadata
        metadata = {
            "global": {
                "core:datatype": "cf32_le",
                "core:sample_rate": samples[0].sample_rate,
                "core:hw": "real_world_data_collector",
                "core:author": "federated_pipeline",
                "core:version": "1.0.0"
            },
            "captures": [
                {
                    "core:sample_start": 0,
                    "core:frequency": samples[0].frequency,
                    "core:datetime": datetime.now().isoformat()
                }
            ],
            "annotations": []
        }
        
        # Add annotations for each sample
        sample_start = 0
        for sample in samples:
            annotation = {
                "core:sample_start": sample_start,
                "core:sample_count": len(sample.iq_data),
                "core:frequency": sample.frequency,
                "core:label": sample.modulation_type
            }
            
            if sample.timestamp:
                annotation["core:datetime"] = sample.timestamp.isoformat()
            
            if sample.location:
                annotation["core:latitude"] = sample.location.get('latitude')
                annotation["core:longitude"] = sample.location.get('longitude')
            
            metadata["annotations"].append(annotation)
            sample_start += len(sample.iq_data)
        
        # Save metadata file
        meta_file = output_file.with_suffix('.sigmf-meta')
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save data file
        data_file = output_file.with_suffix('.sigmf-data')
        all_iq_data.astype(np.complex64).tofile(data_file)
    
    def _save_as_numpy(self, samples: List[SignalSample], output_file: Path):
        """Save samples in NumPy format"""
        # Prepare data arrays
        iq_data_list = [sample.iq_data for sample in samples]
        
        # Create metadata arrays
        frequencies = np.array([sample.frequency for sample in samples])
        sample_rates = np.array([sample.sample_rate for sample in samples])
        modulation_types = np.array([sample.modulation_type for sample in samples])
        snr_values = np.array([sample.snr for sample in samples])
        device_ids = np.array([sample.device_id for sample in samples])
        
        # Save with compression
        np.savez_compressed(
            output_file,
            iq_data=iq_data_list,
            frequencies=frequencies,
            sample_rates=sample_rates,
            modulation_types=modulation_types,
            snr_values=snr_values,
            device_ids=device_ids
        )
    
    def _save_collection_metadata(self, data: Dict[str, List[SignalSample]], metadata_file: Path):
        """Save collection metadata"""
        metadata = {
            'collection_timestamp': datetime.now().isoformat(),
            'config': asdict(self.config),
            'datasets': {}
        }
        
        for dataset_name, samples in data.items():
            if samples:
                # Calculate statistics
                modulation_counts = {}
                snr_values = []
                frequencies = []
                
                for sample in samples:
                    mod_type = sample.modulation_type
                    modulation_counts[mod_type] = modulation_counts.get(mod_type, 0) + 1
                    snr_values.append(sample.snr)
                    frequencies.append(sample.frequency)
                
                metadata['datasets'][dataset_name] = {
                    'total_samples': len(samples),
                    'modulation_distribution': modulation_counts,
                    'snr_range': [float(np.min(snr_values)), float(np.max(snr_values))],
                    'frequency_range': [float(np.min(frequencies)), float(np.max(frequencies))],
                    'avg_snr': float(np.mean(snr_values))
                }
        
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def get_collection_summary(self, data: Dict[str, List[SignalSample]]) -> Dict[str, Any]:
        """Get summary statistics of collected data"""
        summary = {
            'total_samples': 0,
            'datasets': {},
            'overall_stats': {
                'modulation_types': set(),
                'snr_range': [float('inf'), float('-inf')],
                'frequency_range': [float('inf'), float('-inf')],
                'sample_lengths': []
            }
        }
        
        for dataset_name, samples in data.items():
            if not samples:
                continue
            
            dataset_stats = {
                'sample_count': len(samples),
                'modulation_types': set(),
                'snr_values': [],
                'frequencies': [],
                'sample_lengths': []
            }
            
            for sample in samples:
                dataset_stats['modulation_types'].add(sample.modulation_type)
                dataset_stats['snr_values'].append(sample.snr)
                dataset_stats['frequencies'].append(sample.frequency)
                dataset_stats['sample_lengths'].append(len(sample.iq_data))
                
                # Update overall stats
                summary['overall_stats']['modulation_types'].add(sample.modulation_type)
                summary['overall_stats']['snr_range'][0] = min(summary['overall_stats']['snr_range'][0], sample.snr)
                summary['overall_stats']['snr_range'][1] = max(summary['overall_stats']['snr_range'][1], sample.snr)
                summary['overall_stats']['frequency_range'][0] = min(summary['overall_stats']['frequency_range'][0], sample.frequency)
                summary['overall_stats']['frequency_range'][1] = max(summary['overall_stats']['frequency_range'][1], sample.frequency)
                summary['overall_stats']['sample_lengths'].append(len(sample.iq_data))
            
            # Calculate dataset statistics
            dataset_stats['modulation_types'] = list(dataset_stats['modulation_types'])
            dataset_stats['snr_range'] = [min(dataset_stats['snr_values']), max(dataset_stats['snr_values'])]
            dataset_stats['frequency_range'] = [min(dataset_stats['frequencies']), max(dataset_stats['frequencies'])]
            dataset_stats['avg_sample_length'] = np.mean(dataset_stats['sample_lengths'])
            
            summary['datasets'][dataset_name] = dataset_stats
            summary['total_samples'] += len(samples)
        
        # Finalize overall stats
        summary['overall_stats']['modulation_types'] = list(summary['overall_stats']['modulation_types'])
        if summary['overall_stats']['sample_lengths']:
            summary['overall_stats']['avg_sample_length'] = np.mean(summary['overall_stats']['sample_lengths'])
        
        return summary