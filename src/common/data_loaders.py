"""
Data loaders for SigMF format and common SDR file formats
"""
import json
import numpy as np
import h5py
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, Tuple
from datetime import datetime
import struct
import gzip
from dataclasses import dataclass

from .interfaces import SignalSample, BaseDataLoader


@dataclass
class DatasetInfo:
    """Dataset information and metadata"""
    name: str
    version: str
    description: str
    total_samples: int
    modulation_types: List[str]
    snr_range: Tuple[float, float]
    sample_rate: float
    center_frequency: float
    file_format: str


class SigMFLoader(BaseDataLoader):
    """Loader for SigMF (Signal Metadata Format) files"""
    
    def __init__(self):
        self.supported_datatypes = {
            'cf32_le': ('f', 4),  # Complex float32 little endian
            'ci16_le': ('h', 2),  # Complex int16 little endian
            'ci8': ('b', 1),      # Complex int8
        }
    
    def load_signal_data(self, file_path: str) -> List[SignalSample]:
        """Load SigMF format signal data"""
        sigmf_path = Path(file_path)
        
        # Load metadata file (.sigmf-meta)
        meta_path = sigmf_path.with_suffix('.sigmf-meta')
        if not meta_path.exists():
            raise FileNotFoundError(f"SigMF metadata file not found: {meta_path}")
        
        with open(meta_path, 'r') as f:
            metadata = json.load(f)
        
        # Load data file (.sigmf-data)
        data_path = sigmf_path.with_suffix('.sigmf-data')
        if not data_path.exists():
            raise FileNotFoundError(f"SigMF data file not found: {data_path}")
        
        # Parse global metadata
        global_meta = metadata.get('global', {})
        sample_rate = global_meta.get('core:sample_rate', 1e6)
        datatype = global_meta.get('core:datatype', 'cf32_le')
        
        # Load binary data
        iq_data = self._load_binary_data(data_path, datatype)
        
        # Parse annotations to create signal samples
        samples = []
        annotations = metadata.get('annotations', [])
        
        if not annotations:
            # No annotations, create single sample
            sample = SignalSample(
                timestamp=datetime.now(),
                frequency=global_meta.get('core:frequency', 0),
                sample_rate=sample_rate,
                iq_data=iq_data,
                modulation_type='unknown',
                snr=0.0,
                location=None,
                device_id=global_meta.get('core:recorder', 'unknown'),
                metadata=global_meta
            )
            samples.append(sample)
        else:
            # Process each annotation
            for annotation in annotations:
                start_idx = annotation.get('core:sample_start', 0)
                count = annotation.get('core:sample_count', len(iq_data))
                end_idx = start_idx + count
                
                sample = SignalSample(
                    timestamp=datetime.fromtimestamp(annotation.get('core:datetime', 0)),
                    frequency=annotation.get('core:frequency', global_meta.get('core:frequency', 0)),
                    sample_rate=sample_rate,
                    iq_data=iq_data[start_idx:end_idx],
                    modulation_type=annotation.get('core:label', 'unknown'),
                    snr=annotation.get('snr', 0.0),
                    location=self._parse_location(annotation),
                    device_id=global_meta.get('core:recorder', 'unknown'),
                    metadata={**global_meta, **annotation}
                )
                samples.append(sample)
        
        return samples
    
    def _load_binary_data(self, data_path: Path, datatype: str) -> np.ndarray:
        """Load binary IQ data based on datatype"""
        if datatype not in self.supported_datatypes:
            raise ValueError(f"Unsupported datatype: {datatype}")
        
        format_char, bytes_per_sample = self.supported_datatypes[datatype]
        
        with open(data_path, 'rb') as f:
            data = f.read()
        
        # Unpack binary data
        if datatype == 'cf32_le':
            # Complex float32
            values = struct.unpack(f'<{len(data)//4}f', data)
            # Convert to complex numbers (I, Q pairs)
            iq_data = np.array([complex(values[i], values[i+1]) 
                               for i in range(0, len(values), 2)])
        elif datatype == 'ci16_le':
            # Complex int16
            values = struct.unpack(f'<{len(data)//2}h', data)
            iq_data = np.array([complex(values[i], values[i+1]) 
                               for i in range(0, len(values), 2)]) / 32768.0
        elif datatype == 'ci8':
            # Complex int8
            values = struct.unpack(f'{len(data)}b', data)
            iq_data = np.array([complex(values[i], values[i+1]) 
                               for i in range(0, len(values), 2)]) / 128.0
        
        return iq_data
    
    def _parse_location(self, annotation: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Parse GPS location from annotation"""
        if 'core:latitude' in annotation and 'core:longitude' in annotation:
            return {
                'latitude': annotation['core:latitude'],
                'longitude': annotation['core:longitude'],
                'altitude': annotation.get('core:altitude', 0.0)
            }
        return None
    
    def preprocess_data(self, data: List[SignalSample]) -> Dict[str, Any]:
        """Preprocess signal data for training"""
        processed_data = {
            'iq_samples': [],
            'labels': [],
            'metadata': []
        }
        
        for sample in data:
            # Normalize IQ data
            normalized_iq = self._normalize_iq(sample.iq_data)
            processed_data['iq_samples'].append(normalized_iq)
            processed_data['labels'].append(sample.modulation_type)
            processed_data['metadata'].append({
                'snr': sample.snr,
                'frequency': sample.frequency,
                'sample_rate': sample.sample_rate
            })
        
        return processed_data
    
    def _normalize_iq(self, iq_data: np.ndarray) -> np.ndarray:
        """Normalize IQ data"""
        # Power normalization
        power = np.mean(np.abs(iq_data) ** 2)
        if power > 0:
            return iq_data / np.sqrt(power)
        return iq_data
    
    def validate_data_quality(self, data: List[SignalSample]) -> Dict[str, Any]:
        """Validate data quality and return metrics"""
        if not data:
            return {'valid': False, 'error': 'No data provided'}
        
        metrics = {
            'valid': True,
            'total_samples': len(data),
            'sample_rates': [],
            'snr_values': [],
            'modulation_types': set(),
            'frequency_range': [float('inf'), float('-inf')],
            'quality_issues': []
        }
        
        for sample in data:
            metrics['sample_rates'].append(sample.sample_rate)
            metrics['snr_values'].append(sample.snr)
            metrics['modulation_types'].add(sample.modulation_type)
            
            # Update frequency range
            metrics['frequency_range'][0] = min(metrics['frequency_range'][0], sample.frequency)
            metrics['frequency_range'][1] = max(metrics['frequency_range'][1], sample.frequency)
            
            # Check for quality issues
            if len(sample.iq_data) == 0:
                metrics['quality_issues'].append(f"Empty IQ data in sample from {sample.device_id}")
            
            if np.any(np.isnan(sample.iq_data)) or np.any(np.isinf(sample.iq_data)):
                metrics['quality_issues'].append(f"Invalid IQ values in sample from {sample.device_id}")
        
        metrics['modulation_types'] = list(metrics['modulation_types'])
        metrics['avg_snr'] = np.mean(metrics['snr_values']) if metrics['snr_values'] else 0
        metrics['unique_sample_rates'] = len(set(metrics['sample_rates']))
        
        return metrics


class RadioMLLoader(BaseDataLoader):
    """Loader for RadioML datasets (HDF5 format)"""
    
    def __init__(self):
        self.modulation_mapping = {
            0: '8PSK', 1: 'AM-DSB', 2: 'AM-SSB', 3: 'BPSK', 4: 'CPFSK',
            5: 'GFSK', 6: 'PAM4', 7: 'QAM16', 8: 'QAM64', 9: 'QPSK', 10: 'WBFM'
        }
    
    def load_signal_data(self, file_path: str) -> List[SignalSample]:
        """Load RadioML dataset"""
        samples = []
        
        with h5py.File(file_path, 'r') as f:
            # RadioML 2016.10 format
            if 'X' in f and 'Y' in f and 'Z' in f:
                X = f['X'][:]  # IQ data
                Y = f['Y'][:]  # Modulation labels
                Z = f['Z'][:]  # SNR values
                
                for i in range(len(X)):
                    # Convert to complex IQ data
                    iq_data = X[i, 0, :] + 1j * X[i, 1, :]
                    
                    # Get modulation type
                    mod_idx = np.argmax(Y[i])
                    modulation = self.modulation_mapping.get(mod_idx, 'unknown')
                    
                    sample = SignalSample(
                        timestamp=datetime.now(),
                        frequency=915e6,  # Default frequency for RadioML
                        sample_rate=200e3,  # Default sample rate
                        iq_data=iq_data,
                        modulation_type=modulation,
                        snr=float(Z[i]),
                        location=None,
                        device_id='radioml_dataset',
                        metadata={'dataset': 'RadioML2016.10', 'sample_index': i}
                    )
                    samples.append(sample)
            
            # RadioML 2018.01 format
            elif 'X' in f and 'Y' in f:
                X = f['X'][:]
                Y = f['Y'][:]
                
                for i in range(len(X)):
                    iq_data = X[i, :, 0] + 1j * X[i, :, 1]
                    
                    # Y contains (modulation, snr) tuples
                    modulation = Y[i][0].decode('utf-8') if isinstance(Y[i][0], bytes) else str(Y[i][0])
                    snr = float(Y[i][1])
                    
                    sample = SignalSample(
                        timestamp=datetime.now(),
                        frequency=915e6,
                        sample_rate=200e3,
                        iq_data=iq_data,
                        modulation_type=modulation,
                        snr=snr,
                        location=None,
                        device_id='radioml_dataset',
                        metadata={'dataset': 'RadioML2018.01', 'sample_index': i}
                    )
                    samples.append(sample)
        
        return samples
    
    def preprocess_data(self, data: List[SignalSample]) -> Dict[str, Any]:
        """Preprocess RadioML data"""
        processed_data = {
            'iq_samples': [],
            'labels': [],
            'snr_values': [],
            'modulation_types': []
        }
        
        for sample in data:
            # Normalize and reshape IQ data
            iq_normalized = self._normalize_power(sample.iq_data)
            
            # Convert to real/imaginary representation
            iq_real_imag = np.stack([iq_normalized.real, iq_normalized.imag])
            
            processed_data['iq_samples'].append(iq_real_imag)
            processed_data['labels'].append(sample.modulation_type)
            processed_data['snr_values'].append(sample.snr)
            processed_data['modulation_types'].append(sample.modulation_type)
        
        return processed_data
    
    def _normalize_power(self, iq_data: np.ndarray) -> np.ndarray:
        """Normalize signal power"""
        power = np.mean(np.abs(iq_data) ** 2)
        return iq_data / np.sqrt(power) if power > 0 else iq_data
    
    def validate_data_quality(self, data: List[SignalSample]) -> Dict[str, Any]:
        """Validate RadioML data quality"""
        if not data:
            return {'valid': False, 'error': 'No data provided'}
        
        modulation_counts = {}
        snr_distribution = []
        
        for sample in data:
            mod_type = sample.modulation_type
            modulation_counts[mod_type] = modulation_counts.get(mod_type, 0) + 1
            snr_distribution.append(sample.snr)
        
        return {
            'valid': True,
            'total_samples': len(data),
            'modulation_distribution': modulation_counts,
            'snr_range': [min(snr_distribution), max(snr_distribution)],
            'avg_snr': np.mean(snr_distribution),
            'unique_modulations': len(modulation_counts)
        }


class HDF5Loader(BaseDataLoader):
    """Loader for HDF5 signal datasets"""
    
    def __init__(self):
        self.supported_formats = ['radioml', 'custom', 'matlab']
    
    def load_signal_data(self, file_path: str) -> List[SignalSample]:
        """Load HDF5 format signal data"""
        samples = []
        
        with h5py.File(file_path, 'r') as f:
            # Try to detect format
            if self._is_radioml_format(f):
                samples = self._load_radioml_hdf5(f)
            elif self._is_custom_format(f):
                samples = self._load_custom_hdf5(f)
            else:
                samples = self._load_generic_hdf5(f)
        
        return samples
    
    def _is_radioml_format(self, f: h5py.File) -> bool:
        """Check if HDF5 file is RadioML format"""
        return 'X' in f and 'Y' in f
    
    def _is_custom_format(self, f: h5py.File) -> bool:
        """Check if HDF5 file is custom format"""
        return 'samples' in f or any(key.startswith('sample_') for key in f.keys())
    
    def _load_radioml_hdf5(self, f: h5py.File) -> List[SignalSample]:
        """Load RadioML format HDF5"""
        samples = []
        
        if 'X' in f and 'Y' in f:
            X = f['X'][:]
            Y = f['Y'][:]
            Z = f['Z'][:] if 'Z' in f else None
            
            for i in range(len(X)):
                if len(X[i].shape) == 2 and X[i].shape[0] == 2:
                    # Format: [2, N] where first row is I, second is Q
                    iq_data = X[i][0] + 1j * X[i][1]
                elif len(X[i].shape) == 2 and X[i].shape[1] == 2:
                    # Format: [N, 2] where columns are I, Q
                    iq_data = X[i][:, 0] + 1j * X[i][:, 1]
                else:
                    continue
                
                # Handle labels
                if isinstance(Y[i], (list, tuple, np.ndarray)) and len(Y[i]) >= 2:
                    modulation = str(Y[i][0])
                    snr = float(Y[i][1])
                else:
                    modulation = str(Y[i])
                    snr = float(Z[i]) if Z is not None else 0.0
                
                sample = SignalSample(
                    timestamp=datetime.now(),
                    frequency=915e6,
                    sample_rate=200e3,
                    iq_data=iq_data,
                    modulation_type=modulation,
                    snr=snr,
                    location=None,
                    device_id='hdf5_loader',
                    metadata={'dataset': 'hdf5', 'sample_index': i}
                )
                samples.append(sample)
        
        return samples
    
    def _load_custom_hdf5(self, f: h5py.File) -> List[SignalSample]:
        """Load custom format HDF5"""
        samples = []
        
        # Look for sample groups
        for key in f.keys():
            if key.startswith('sample_') or key == 'samples':
                group = f[key]
                
                if isinstance(group, h5py.Group):
                    # Load from group
                    if 'iq_data' in group:
                        iq_data = group['iq_data'][:]
                        
                        sample = SignalSample(
                            timestamp=datetime.fromisoformat(group.attrs.get('timestamp', datetime.now().isoformat())),
                            frequency=float(group.attrs.get('frequency', 915e6)),
                            sample_rate=float(group.attrs.get('sample_rate', 1e6)),
                            iq_data=iq_data,
                            modulation_type=str(group.attrs.get('modulation_type', 'unknown')),
                            snr=float(group.attrs.get('snr', 0.0)),
                            location=json.loads(group.attrs.get('location', 'null')),
                            device_id=str(group.attrs.get('device_id', 'hdf5_loader')),
                            metadata=json.loads(group.attrs.get('metadata', '{}'))
                        )
                        samples.append(sample)
        
        return samples
    
    def _load_generic_hdf5(self, f: h5py.File) -> List[SignalSample]:
        """Load generic HDF5 format"""
        samples = []
        
        # Try to find IQ data arrays
        for key in f.keys():
            dataset = f[key]
            if isinstance(dataset, h5py.Dataset) and len(dataset.shape) >= 1:
                data = dataset[:]
                
                # Try to interpret as complex data
                if np.iscomplexobj(data):
                    iq_data = data
                elif len(data.shape) == 2 and data.shape[1] == 2:
                    iq_data = data[:, 0] + 1j * data[:, 1]
                elif len(data.shape) == 2 and data.shape[0] == 2:
                    iq_data = data[0] + 1j * data[1]
                else:
                    continue
                
                sample = SignalSample(
                    timestamp=datetime.now(),
                    frequency=float(dataset.attrs.get('frequency', 915e6)),
                    sample_rate=float(dataset.attrs.get('sample_rate', 1e6)),
                    iq_data=iq_data,
                    modulation_type=str(dataset.attrs.get('modulation_type', 'unknown')),
                    snr=float(dataset.attrs.get('snr', 0.0)),
                    location=None,
                    device_id='hdf5_loader',
                    metadata={'dataset_name': key}
                )
                samples.append(sample)
        
        return samples
    
    def preprocess_data(self, data: List[SignalSample]) -> Dict[str, Any]:
        """Preprocess HDF5 data"""
        return {
            'iq_samples': [sample.iq_data for sample in data],
            'labels': [sample.modulation_type for sample in data],
            'snr_values': [sample.snr for sample in data],
            'metadata': [sample.metadata for sample in data]
        }
    
    def validate_data_quality(self, data: List[SignalSample]) -> Dict[str, Any]:
        """Validate HDF5 data quality"""
        if not data:
            return {'valid': False, 'error': 'No data provided'}
        
        return {
            'valid': True,
            'total_samples': len(data),
            'modulation_types': list(set(sample.modulation_type for sample in data)),
            'snr_range': [min(sample.snr for sample in data), max(sample.snr for sample in data)],
            'sample_lengths': [len(sample.iq_data) for sample in data]
        }


class BinaryIQLoader(BaseDataLoader):
    """Loader for binary IQ files (common SDR format)"""
    
    def __init__(self, sample_rate: float = 1e6, center_freq: float = 915e6):
        self.sample_rate = sample_rate
        self.center_freq = center_freq
    
    def load_signal_data(self, file_path: str) -> List[SignalSample]:
        """Load binary IQ file"""
        file_path = Path(file_path)
        
        # Determine data type from filename or assume complex float32
        if 'int16' in file_path.name.lower() or 'i16' in file_path.name.lower():
            dtype = np.int16
            scale_factor = 1.0 / 32768.0
        elif 'int8' in file_path.name.lower() or 'i8' in file_path.name.lower():
            dtype = np.int8
            scale_factor = 1.0 / 128.0
        else:
            dtype = np.float32
            scale_factor = 1.0
        
        # Load binary data
        if file_path.suffix == '.gz':
            with gzip.open(file_path, 'rb') as f:
                data = np.frombuffer(f.read(), dtype=dtype)
        else:
            data = np.fromfile(file_path, dtype=dtype)
        
        # Convert to complex IQ
        if len(data) % 2 != 0:
            data = data[:-1]  # Remove last sample if odd number
        
        iq_data = (data[::2] + 1j * data[1::2]) * scale_factor
        
        # Create signal sample
        sample = SignalSample(
            timestamp=datetime.fromtimestamp(file_path.stat().st_mtime),
            frequency=self.center_freq,
            sample_rate=self.sample_rate,
            iq_data=iq_data,
            modulation_type='unknown',
            snr=self._estimate_snr(iq_data),
            location=None,
            device_id='binary_file_loader',
            metadata={
                'filename': file_path.name,
                'file_size': len(data),
                'data_type': str(dtype)
            }
        )
        
        return [sample]
    
    def _estimate_snr(self, iq_data: np.ndarray) -> float:
        """Estimate SNR from IQ data"""
        # Simple SNR estimation based on signal power variation
        power = np.abs(iq_data) ** 2
        signal_power = np.mean(power)
        noise_power = np.var(power)
        
        if noise_power > 0:
            snr_linear = signal_power / noise_power
            return 10 * np.log10(snr_linear)
        return 0.0
    
    def preprocess_data(self, data: List[SignalSample]) -> Dict[str, Any]:
        """Preprocess binary IQ data"""
        processed_data = {
            'iq_samples': [],
            'estimated_snr': [],
            'metadata': []
        }
        
        for sample in data:
            # Apply windowing and normalization
            windowed_iq = self._apply_window(sample.iq_data)
            normalized_iq = self._normalize_power(windowed_iq)
            
            processed_data['iq_samples'].append(normalized_iq)
            processed_data['estimated_snr'].append(sample.snr)
            processed_data['metadata'].append(sample.metadata)
        
        return processed_data
    
    def _apply_window(self, iq_data: np.ndarray) -> np.ndarray:
        """Apply Hamming window to reduce spectral leakage"""
        window = np.hamming(len(iq_data))
        return iq_data * window
    
    def _normalize_power(self, iq_data: np.ndarray) -> np.ndarray:
        """Normalize signal power"""
        power = np.mean(np.abs(iq_data) ** 2)
        return iq_data / np.sqrt(power) if power > 0 else iq_data
    
    def validate_data_quality(self, data: List[SignalSample]) -> Dict[str, Any]:
        """Validate binary IQ data quality"""
        if not data:
            return {'valid': False, 'error': 'No data provided'}
        
        sample = data[0]  # Single sample for binary files
        
        # Check for common issues
        issues = []
        if len(sample.iq_data) == 0:
            issues.append("Empty IQ data")
        
        if np.any(np.isnan(sample.iq_data)) or np.any(np.isinf(sample.iq_data)):
            issues.append("Invalid IQ values (NaN or Inf)")
        
        # Check dynamic range
        max_amplitude = np.max(np.abs(sample.iq_data))
        if max_amplitude > 10:
            issues.append("Unusually high amplitude values")
        elif max_amplitude < 1e-6:
            issues.append("Unusually low amplitude values")
        
        return {
            'valid': len(issues) == 0,
            'total_samples': len(sample.iq_data),
            'estimated_snr': sample.snr,
            'max_amplitude': float(max_amplitude),
            'sample_rate': sample.sample_rate,
            'quality_issues': issues
        }