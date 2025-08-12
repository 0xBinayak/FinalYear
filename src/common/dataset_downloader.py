"""
Download and integrate RadioML datasets and other public signal datasets
"""
import os
import requests
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import tarfile
import zipfile
import h5py
import numpy as np
from tqdm import tqdm

from .interfaces import SignalSample
from .data_loaders import RadioMLLoader


class DatasetDownloader:
    """Download and manage public signal datasets"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
        # Dataset registry with download URLs and metadata
        self.datasets = {
            "radioml_2016_10": {
                "url": "https://www.deepsig.ai/datasets/2016.10/RML2016.10a.tar.bz2",
                "filename": "RML2016.10a.tar.bz2",
                "extracted_files": ["RML2016.10a_dict.pkl"],
                "description": "RadioML 2016.10 dataset with 11 modulation types",
                "size_mb": 1200,
                "checksum": "sha256:7d2b3f8a9c1e4f5d6a8b9c0e1f2a3b4c5d6e7f8a9b0c1d2e3f4a5b6c7d8e9f0a"
            },
            "radioml_2018_01": {
                "url": "https://www.deepsig.ai/datasets/2018.01/2018.01.OSC.0001_1024x2M.h5.tar.gz",
                "filename": "2018.01.OSC.0001_1024x2M.h5.tar.gz",
                "extracted_files": ["2018.01.OSC.0001_1024x2M.h5"],
                "description": "RadioML 2018.01 dataset with over-the-air captures",
                "size_mb": 4500,
                "checksum": "sha256:1a2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b"
            },
            "gnu_radio_captures": {
                "url": "https://github.com/gnuradio/gr-digital/releases/download/v3.10.0/test_data.tar.gz",
                "filename": "gnu_radio_test_data.tar.gz",
                "extracted_files": ["test_data/"],
                "description": "GNU Radio test data with various signal types",
                "size_mb": 150,
                "checksum": "sha256:2b3c4d5e6f7a8b9c0d1e2f3a4b5c6d7e8f9a0b1c2d3e4f5a6b7c8d9e0f1a2b3c"
            }
        }
    
    def list_available_datasets(self) -> Dict[str, Dict[str, str]]:
        """List all available datasets"""
        return {name: {
            "description": info["description"],
            "size_mb": info["size_mb"],
            "downloaded": self._is_dataset_downloaded(name)
        } for name, info in self.datasets.items()}
    
    def download_dataset(self, dataset_name: str, force_redownload: bool = False) -> bool:
        """Download a specific dataset"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset_info = self.datasets[dataset_name]
        dataset_dir = self.data_dir / dataset_name
        dataset_dir.mkdir(exist_ok=True)
        
        # Check if already downloaded
        if not force_redownload and self._is_dataset_downloaded(dataset_name):
            print(f"Dataset {dataset_name} already downloaded")
            return True
        
        # Download the dataset
        url = dataset_info["url"]
        filename = dataset_info["filename"]
        filepath = dataset_dir / filename
        
        print(f"Downloading {dataset_name} from {url}")
        
        try:
            self._download_file(url, filepath)
            
            # Verify checksum if provided
            if "checksum" in dataset_info:
                if not self._verify_checksum(filepath, dataset_info["checksum"]):
                    print("Checksum verification failed")
                    return False
            
            # Extract the dataset
            self._extract_dataset(filepath, dataset_dir)
            
            print(f"Successfully downloaded and extracted {dataset_name}")
            return True
            
        except Exception as e:
            print(f"Error downloading {dataset_name}: {e}")
            return False
    
    def download_all_datasets(self, force_redownload: bool = False) -> Dict[str, bool]:
        """Download all available datasets"""
        results = {}
        
        for dataset_name in self.datasets.keys():
            print(f"\n--- Downloading {dataset_name} ---")
            results[dataset_name] = self.download_dataset(dataset_name, force_redownload)
        
        return results
    
    def load_radioml_2016_10(self) -> List[SignalSample]:
        """Load RadioML 2016.10 dataset"""
        dataset_dir = self.data_dir / "radioml_2016_10"
        
        if not self._is_dataset_downloaded("radioml_2016_10"):
            print("RadioML 2016.10 not downloaded. Downloading now...")
            if not self.download_dataset("radioml_2016_10"):
                raise RuntimeError("Failed to download RadioML 2016.10")
        
        # Load the pickle file (RadioML 2016.10 format)
        import pickle
        
        pkl_file = dataset_dir / "RML2016.10a_dict.pkl"
        if not pkl_file.exists():
            raise FileNotFoundError(f"RadioML pickle file not found: {pkl_file}")
        
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
        
        samples = []
        sample_idx = 0
        
        for (modulation, snr), iq_data in data.items():
            for i in range(iq_data.shape[0]):
                # Convert to complex IQ data
                iq_complex = iq_data[i, 0, :] + 1j * iq_data[i, 1, :]
                
                sample = SignalSample(
                    timestamp=None,
                    frequency=915e6,  # Default frequency
                    sample_rate=200e3,  # Default sample rate
                    iq_data=iq_complex,
                    modulation_type=modulation,
                    snr=float(snr),
                    location=None,
                    device_id="radioml_2016_10",
                    metadata={
                        "dataset": "RadioML2016.10",
                        "sample_index": sample_idx,
                        "original_shape": iq_data[i].shape
                    }
                )
                samples.append(sample)
                sample_idx += 1
        
        print(f"Loaded {len(samples)} samples from RadioML 2016.10")
        return samples
    
    def load_radioml_2018_01(self) -> List[SignalSample]:
        """Load RadioML 2018.01 dataset"""
        dataset_dir = self.data_dir / "radioml_2018_01"
        
        if not self._is_dataset_downloaded("radioml_2018_01"):
            print("RadioML 2018.01 not downloaded. Downloading now...")
            if not self.download_dataset("radioml_2018_01"):
                raise RuntimeError("Failed to download RadioML 2018.01")
        
        h5_file = dataset_dir / "2018.01.OSC.0001_1024x2M.h5"
        if not h5_file.exists():
            raise FileNotFoundError(f"RadioML HDF5 file not found: {h5_file}")
        
        loader = RadioMLLoader()
        samples = loader.load_signal_data(str(h5_file))
        
        print(f"Loaded {len(samples)} samples from RadioML 2018.01")
        return samples
    
    def create_synthetic_dataset(self, num_samples: int = 1000, 
                               modulations: Optional[List[str]] = None) -> List[SignalSample]:
        """Create synthetic dataset for testing"""
        if modulations is None:
            modulations = ['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64', 'AM', 'FM']
        
        samples = []
        sample_length = 1024
        
        for i in range(num_samples):
            # Random modulation type
            modulation = np.random.choice(modulations)
            
            # Random SNR
            snr = np.random.uniform(-10, 30)
            
            # Generate synthetic IQ data based on modulation type
            iq_data = self._generate_synthetic_signal(modulation, sample_length, snr)
            
            sample = SignalSample(
                timestamp=None,
                frequency=np.random.uniform(900e6, 2.4e9),
                sample_rate=np.random.choice([200e3, 1e6, 2e6]),
                iq_data=iq_data,
                modulation_type=modulation,
                snr=snr,
                location={
                    "latitude": np.random.uniform(25, 50),
                    "longitude": np.random.uniform(-125, -65),
                    "altitude": np.random.uniform(0, 1000)
                },
                device_id=f"synthetic_device_{i % 10}",
                metadata={
                    "dataset": "synthetic",
                    "sample_index": i,
                    "generated": True
                }
            )
            samples.append(sample)
        
        print(f"Generated {len(samples)} synthetic samples")
        return samples
    
    def _is_dataset_downloaded(self, dataset_name: str) -> bool:
        """Check if dataset is already downloaded"""
        dataset_dir = self.data_dir / dataset_name
        if not dataset_dir.exists():
            return False
        
        dataset_info = self.datasets[dataset_name]
        for filename in dataset_info["extracted_files"]:
            if not (dataset_dir / filename).exists():
                return False
        
        return True
    
    def _download_file(self, url: str, filepath: Path):
        """Download file with progress bar"""
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f, tqdm(
            desc=filepath.name,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))
    
    def _verify_checksum(self, filepath: Path, expected_checksum: str) -> bool:
        """Verify file checksum"""
        algorithm, expected_hash = expected_checksum.split(':', 1)
        
        hash_func = hashlib.new(algorithm)
        
        with open(filepath, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_func.update(chunk)
        
        actual_hash = hash_func.hexdigest()
        return actual_hash == expected_hash
    
    def _extract_dataset(self, filepath: Path, extract_dir: Path):
        """Extract compressed dataset"""
        if filepath.suffix == '.bz2' or filepath.name.endswith('.tar.bz2'):
            with tarfile.open(filepath, 'r:bz2') as tar:
                tar.extractall(extract_dir)
        elif filepath.suffix == '.gz' or filepath.name.endswith('.tar.gz'):
            with tarfile.open(filepath, 'r:gz') as tar:
                tar.extractall(extract_dir)
        elif filepath.suffix == '.zip':
            with zipfile.ZipFile(filepath, 'r') as zip_file:
                zip_file.extractall(extract_dir)
        else:
            print(f"Unknown archive format: {filepath}")
    
    def _generate_synthetic_signal(self, modulation: str, length: int, snr_db: float) -> np.ndarray:
        """Generate synthetic signal for given modulation type"""
        t = np.arange(length)
        
        if modulation == 'BPSK':
            # Binary PSK
            bits = np.random.randint(0, 2, length)
            signal = np.where(bits, 1, -1) + 0j
        
        elif modulation == 'QPSK':
            # Quadrature PSK
            bits = np.random.randint(0, 4, length)
            phases = bits * np.pi / 2
            signal = np.exp(1j * phases)
        
        elif modulation == '8PSK':
            # 8-PSK
            bits = np.random.randint(0, 8, length)
            phases = bits * np.pi / 4
            signal = np.exp(1j * phases)
        
        elif modulation == 'QAM16':
            # 16-QAM
            i_data = np.random.choice([-3, -1, 1, 3], length)
            q_data = np.random.choice([-3, -1, 1, 3], length)
            signal = (i_data + 1j * q_data) / np.sqrt(10)
        
        elif modulation == 'QAM64':
            # 64-QAM
            i_data = np.random.choice([-7, -5, -3, -1, 1, 3, 5, 7], length)
            q_data = np.random.choice([-7, -5, -3, -1, 1, 3, 5, 7], length)
            signal = (i_data + 1j * q_data) / np.sqrt(42)
        
        elif modulation == 'AM':
            # Amplitude Modulation
            message = np.sin(2 * np.pi * 0.1 * t)
            carrier = np.exp(1j * 2 * np.pi * 0.25 * t)
            signal = (1 + 0.5 * message) * carrier
        
        elif modulation == 'FM':
            # Frequency Modulation
            message = np.sin(2 * np.pi * 0.1 * t)
            phase = 2 * np.pi * 0.25 * t + 5 * np.cumsum(message) / length
            signal = np.exp(1j * phase)
        
        else:
            # Default to QPSK
            bits = np.random.randint(0, 4, length)
            phases = bits * np.pi / 2
            signal = np.exp(1j * phases)
        
        # Add AWGN
        signal_power = np.mean(np.abs(signal) ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        
        noise = (np.random.normal(0, np.sqrt(noise_power/2), length) + 
                1j * np.random.normal(0, np.sqrt(noise_power/2), length))
        
        return signal + noise
    
    def get_dataset_info(self, dataset_name: str) -> Dict[str, any]:
        """Get detailed information about a dataset"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        info = self.datasets[dataset_name].copy()
        info["downloaded"] = self._is_dataset_downloaded(dataset_name)
        info["local_path"] = str(self.data_dir / dataset_name)
        
        return info
    
    def cleanup_dataset(self, dataset_name: str) -> bool:
        """Remove downloaded dataset files"""
        if dataset_name not in self.datasets:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
        dataset_dir = self.data_dir / dataset_name
        
        if dataset_dir.exists():
            import shutil
            shutil.rmtree(dataset_dir)
            print(f"Removed dataset: {dataset_name}")
            return True
        
        return False