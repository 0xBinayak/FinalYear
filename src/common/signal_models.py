"""
Enhanced signal processing data models with real-world metadata and validation
"""
import numpy as np
import json
import pickle
import gzip
from typing import Dict, Any, Optional, List, Tuple, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import hashlib
import struct

from .interfaces import SignalSample


class ModulationType(Enum):
    """Enumeration of supported modulation types"""
    # Digital modulations
    BPSK = "BPSK"
    QPSK = "QPSK"
    PSK8 = "8PSK"
    QAM16 = "QAM16"
    QAM64 = "QAM64"
    QAM256 = "QAM256"
    
    # Analog modulations
    AM_DSB = "AM-DSB"
    AM_SSB = "AM-SSB"
    FM = "FM"
    WBFM = "WBFM"
    
    # Frequency shift keying
    FSK = "FSK"
    GFSK = "GFSK"
    CPFSK = "CPFSK"
    
    # Pulse amplitude modulation
    PAM4 = "PAM4"
    PAM8 = "PAM8"
    
    # Other
    OFDM = "OFDM"
    UNKNOWN = "UNKNOWN"


class HardwareType(Enum):
    """Enumeration of SDR hardware types"""
    RTL_SDR = "RTL-SDR"
    HACKRF = "HackRF"
    USRP_B200 = "USRP-B200"
    USRP_B210 = "USRP-B210"
    USRP_N210 = "USRP-N210"
    USRP_X310 = "USRP-X310"
    BLADERF = "BladeRF"
    LIMESDR = "LimeSDR"
    PLUTO_SDR = "PlutoSDR"
    AIRSPY = "Airspy"
    SIMULATED = "Simulated"
    UNKNOWN = "Unknown"


@dataclass
class GPSCoordinate:
    """GPS coordinate with validation"""
    latitude: float
    longitude: float
    altitude: float = 0.0
    accuracy: float = 0.0  # meters
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        self.validate()
    
    def validate(self):
        """Validate GPS coordinates"""
        if not (-90 <= self.latitude <= 90):
            raise ValueError(f"Invalid latitude: {self.latitude}")
        if not (-180 <= self.longitude <= 180):
            raise ValueError(f"Invalid longitude: {self.longitude}")
        if self.altitude < -1000 or self.altitude > 50000:  # Reasonable altitude range
            raise ValueError(f"Invalid altitude: {self.altitude}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'latitude': self.latitude,
            'longitude': self.longitude,
            'altitude': self.altitude,
            'accuracy': self.accuracy,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None
        }


@dataclass
class HardwareInfo:
    """Hardware information for SDR devices"""
    hardware_type: HardwareType
    serial_number: Optional[str] = None
    firmware_version: Optional[str] = None
    driver_version: Optional[str] = None
    calibration_date: Optional[datetime] = None
    frequency_range: Tuple[float, float] = (0.0, 6e9)  # Hz
    max_sample_rate: float = 20e6  # Hz
    resolution_bits: int = 12
    gain_range: Tuple[float, float] = (0.0, 50.0)  # dB
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'hardware_type': self.hardware_type.value,
            'serial_number': self.serial_number,
            'firmware_version': self.firmware_version,
            'driver_version': self.driver_version,
            'calibration_date': self.calibration_date.isoformat() if self.calibration_date else None,
            'frequency_range': list(self.frequency_range),
            'max_sample_rate': self.max_sample_rate,
            'resolution_bits': self.resolution_bits,
            'gain_range': list(self.gain_range)
        }


@dataclass
class RFParameters:
    """RF parameters with validation"""
    center_frequency: float  # Hz
    sample_rate: float  # Hz
    gain: float = 0.0  # dB
    bandwidth: Optional[float] = None  # Hz
    antenna: str = "default"
    
    def __post_init__(self):
        self.validate()
    
    def validate(self):
        """Validate RF parameters"""
        if self.center_frequency <= 0:
            raise ValueError(f"Invalid center frequency: {self.center_frequency}")
        if self.sample_rate <= 0:
            raise ValueError(f"Invalid sample rate: {self.sample_rate}")
        if self.bandwidth and self.bandwidth > self.sample_rate:
            raise ValueError(f"Bandwidth ({self.bandwidth}) cannot exceed sample rate ({self.sample_rate})")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'center_frequency': self.center_frequency,
            'sample_rate': self.sample_rate,
            'gain': self.gain,
            'bandwidth': self.bandwidth,
            'antenna': self.antenna
        }


@dataclass
class SignalQualityMetrics:
    """Signal quality assessment metrics"""
    snr_db: float
    evm_percent: Optional[float] = None  # Error Vector Magnitude
    ber: Optional[float] = None  # Bit Error Rate
    papr_db: Optional[float] = None  # Peak-to-Average Power Ratio
    spectral_efficiency: Optional[float] = None  # bits/s/Hz
    
    # Statistical metrics
    mean_power: Optional[float] = None
    peak_power: Optional[float] = None
    rms_power: Optional[float] = None
    
    # Frequency domain metrics
    occupied_bandwidth: Optional[float] = None
    spectral_flatness: Optional[float] = None
    
    def calculate_derived_metrics(self, iq_data: np.ndarray):
        """Calculate derived metrics from IQ data"""
        # Power metrics
        power = np.abs(iq_data) ** 2
        self.mean_power = float(np.mean(power))
        self.peak_power = float(np.max(power))
        self.rms_power = float(np.sqrt(np.mean(power)))
        
        # PAPR
        if self.mean_power > 0:
            self.papr_db = 10 * np.log10(self.peak_power / self.mean_power)
        
        # Spectral flatness (geometric mean / arithmetic mean of power spectrum)
        fft_data = np.fft.fft(iq_data)
        power_spectrum = np.abs(fft_data) ** 2
        power_spectrum = power_spectrum[power_spectrum > 0]  # Remove zeros
        
        if len(power_spectrum) > 0:
            geometric_mean = np.exp(np.mean(np.log(power_spectrum)))
            arithmetic_mean = np.mean(power_spectrum)
            self.spectral_flatness = geometric_mean / arithmetic_mean
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


@dataclass
class EnhancedSignalSample:
    """Enhanced signal sample with comprehensive metadata and validation"""
    # Core signal data
    iq_data: np.ndarray
    
    # Timing information
    timestamp: datetime
    duration: float  # seconds
    
    # RF parameters
    rf_params: RFParameters
    
    # Signal characteristics
    modulation_type: ModulationType
    symbol_rate: Optional[float] = None  # symbols/second
    carrier_offset: float = 0.0  # Hz
    
    # Quality metrics
    quality_metrics: SignalQualityMetrics = field(default_factory=lambda: SignalQualityMetrics(0.0))
    
    # Location and hardware
    location: Optional[GPSCoordinate] = None
    hardware_info: Optional[HardwareInfo] = None
    
    # Identification
    device_id: str = "unknown"
    session_id: Optional[str] = None
    sample_id: Optional[str] = None
    
    # Additional metadata
    environment: str = "unknown"  # indoor, outdoor, urban, rural
    weather_condition: str = "unknown"  # clear, rain, fog, etc.
    interference_level: str = "low"  # low, medium, high
    
    # Processing metadata
    preprocessing_applied: List[str] = field(default_factory=list)
    augmentation_applied: List[str] = field(default_factory=list)
    
    # Custom metadata
    custom_metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization validation and processing"""
        self.validate()
        self._calculate_derived_fields()
        if self.sample_id is None:
            self.sample_id = self._generate_sample_id()
    
    def validate(self):
        """Validate signal sample data"""
        # Validate IQ data
        if self.iq_data is None or len(self.iq_data) == 0:
            raise ValueError("IQ data cannot be empty")
        
        if not np.iscomplexobj(self.iq_data):
            raise ValueError("IQ data must be complex")
        
        if np.any(np.isnan(self.iq_data)) or np.any(np.isinf(self.iq_data)):
            raise ValueError("IQ data contains invalid values (NaN or Inf)")
        
        # Validate duration
        if self.duration <= 0:
            raise ValueError(f"Invalid duration: {self.duration}")
        
        # Validate RF parameters
        self.rf_params.validate()
        
        # Check consistency between sample rate and data length
        expected_samples = int(self.duration * self.rf_params.sample_rate)
        actual_samples = len(self.iq_data)
        
        # Allow 10% tolerance
        if abs(expected_samples - actual_samples) > 0.1 * expected_samples:
            raise ValueError(f"Inconsistent data length: expected ~{expected_samples}, got {actual_samples}")
    
    def _calculate_derived_fields(self):
        """Calculate derived fields from core data"""
        # Update quality metrics
        self.quality_metrics.calculate_derived_metrics(self.iq_data)
        
        # Calculate actual duration from data length
        self.duration = len(self.iq_data) / self.rf_params.sample_rate
    
    def _generate_sample_id(self) -> str:
        """Generate unique sample ID"""
        # Create hash from key properties
        hash_input = f"{self.timestamp.isoformat()}{self.device_id}{len(self.iq_data)}{self.rf_params.center_frequency}"
        return hashlib.md5(hash_input.encode()).hexdigest()[:16]
    
    def to_legacy_format(self) -> SignalSample:
        """Convert to legacy SignalSample format for compatibility"""
        location_dict = self.location.to_dict() if self.location else None
        
        metadata = {
            'rf_params': self.rf_params.to_dict(),
            'quality_metrics': self.quality_metrics.to_dict(),
            'hardware_info': self.hardware_info.to_dict() if self.hardware_info else None,
            'symbol_rate': self.symbol_rate,
            'carrier_offset': self.carrier_offset,
            'environment': self.environment,
            'weather_condition': self.weather_condition,
            'interference_level': self.interference_level,
            'preprocessing_applied': self.preprocessing_applied,
            'augmentation_applied': self.augmentation_applied,
            'session_id': self.session_id,
            'sample_id': self.sample_id,
            **self.custom_metadata
        }
        
        return SignalSample(
            timestamp=self.timestamp,
            frequency=self.rf_params.center_frequency,
            sample_rate=self.rf_params.sample_rate,
            iq_data=self.iq_data,
            modulation_type=self.modulation_type.value,
            snr=self.quality_metrics.snr_db,
            location=location_dict,
            device_id=self.device_id,
            metadata=metadata
        )
    
    @classmethod
    def from_legacy_format(cls, legacy_sample: SignalSample) -> 'EnhancedSignalSample':
        """Create from legacy SignalSample format"""
        # Extract RF parameters
        rf_params_dict = legacy_sample.metadata.get('rf_params', {})
        rf_params = RFParameters(
            center_frequency=legacy_sample.frequency,
            sample_rate=legacy_sample.sample_rate,
            gain=rf_params_dict.get('gain', 0.0),
            bandwidth=rf_params_dict.get('bandwidth'),
            antenna=rf_params_dict.get('antenna', 'default')
        )
        
        # Extract quality metrics
        quality_dict = legacy_sample.metadata.get('quality_metrics', {})
        quality_metrics = SignalQualityMetrics(
            snr_db=legacy_sample.snr,
            evm_percent=quality_dict.get('evm_percent'),
            ber=quality_dict.get('ber'),
            papr_db=quality_dict.get('papr_db'),
            spectral_efficiency=quality_dict.get('spectral_efficiency')
        )
        
        # Extract location
        location = None
        if legacy_sample.location:
            location = GPSCoordinate(
                latitude=legacy_sample.location['latitude'],
                longitude=legacy_sample.location['longitude'],
                altitude=legacy_sample.location.get('altitude', 0.0)
            )
        
        # Extract hardware info
        hardware_info = None
        hardware_dict = legacy_sample.metadata.get('hardware_info')
        if hardware_dict:
            hardware_info = HardwareInfo(
                hardware_type=HardwareType(hardware_dict.get('hardware_type', 'Unknown')),
                serial_number=hardware_dict.get('serial_number'),
                firmware_version=hardware_dict.get('firmware_version'),
                driver_version=hardware_dict.get('driver_version')
            )
        
        # Calculate duration
        duration = len(legacy_sample.iq_data) / legacy_sample.sample_rate
        
        return cls(
            iq_data=legacy_sample.iq_data,
            timestamp=legacy_sample.timestamp,
            duration=duration,
            rf_params=rf_params,
            modulation_type=ModulationType(legacy_sample.modulation_type),
            symbol_rate=legacy_sample.metadata.get('symbol_rate'),
            carrier_offset=legacy_sample.metadata.get('carrier_offset', 0.0),
            quality_metrics=quality_metrics,
            location=location,
            hardware_info=hardware_info,
            device_id=legacy_sample.device_id,
            session_id=legacy_sample.metadata.get('session_id'),
            sample_id=legacy_sample.metadata.get('sample_id'),
            environment=legacy_sample.metadata.get('environment', 'unknown'),
            weather_condition=legacy_sample.metadata.get('weather_condition', 'unknown'),
            interference_level=legacy_sample.metadata.get('interference_level', 'low'),
            preprocessing_applied=legacy_sample.metadata.get('preprocessing_applied', []),
            augmentation_applied=legacy_sample.metadata.get('augmentation_applied', []),
            custom_metadata={k: v for k, v in legacy_sample.metadata.items() 
                           if k not in ['rf_params', 'quality_metrics', 'hardware_info', 
                                      'symbol_rate', 'carrier_offset', 'session_id', 'sample_id',
                                      'environment', 'weather_condition', 'interference_level',
                                      'preprocessing_applied', 'augmentation_applied']}
        )
    
    def serialize(self, format: str = 'pickle', compress: bool = True) -> bytes:
        """Serialize signal sample to bytes"""
        if format == 'pickle':
            data = pickle.dumps(self)
        elif format == 'json':
            # Convert to JSON-serializable format
            json_data = {
                'iq_data': {'real': self.iq_data.real.tolist(), 'imag': self.iq_data.imag.tolist()},
                'timestamp': self.timestamp.isoformat(),
                'duration': self.duration,
                'rf_params': self.rf_params.to_dict(),
                'modulation_type': self.modulation_type.value,
                'symbol_rate': self.symbol_rate,
                'carrier_offset': self.carrier_offset,
                'quality_metrics': self.quality_metrics.to_dict(),
                'location': self.location.to_dict() if self.location else None,
                'hardware_info': self.hardware_info.to_dict() if self.hardware_info else None,
                'device_id': self.device_id,
                'session_id': self.session_id,
                'sample_id': self.sample_id,
                'environment': self.environment,
                'weather_condition': self.weather_condition,
                'interference_level': self.interference_level,
                'preprocessing_applied': self.preprocessing_applied,
                'augmentation_applied': self.augmentation_applied,
                'custom_metadata': self.custom_metadata
            }
            data = json.dumps(json_data).encode('utf-8')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        if compress:
            data = gzip.compress(data)
        
        return data
    
    @classmethod
    def deserialize(cls, data: bytes, format: str = 'pickle', compressed: bool = True) -> 'EnhancedSignalSample':
        """Deserialize signal sample from bytes"""
        if compressed:
            data = gzip.decompress(data)
        
        if format == 'pickle':
            return pickle.loads(data)
        elif format == 'json':
            json_data = json.loads(data.decode('utf-8'))
            
            # Reconstruct IQ data
            iq_real = np.array(json_data['iq_data']['real'])
            iq_imag = np.array(json_data['iq_data']['imag'])
            iq_data = iq_real + 1j * iq_imag
            
            # Reconstruct objects
            rf_params = RFParameters(**json_data['rf_params'])
            quality_metrics = SignalQualityMetrics(**json_data['quality_metrics'])
            
            location = None
            if json_data['location']:
                location_data = json_data['location'].copy()
                if location_data['timestamp']:
                    location_data['timestamp'] = datetime.fromisoformat(location_data['timestamp'])
                location = GPSCoordinate(**location_data)
            
            hardware_info = None
            if json_data['hardware_info']:
                hardware_data = json_data['hardware_info'].copy()
                hardware_data['hardware_type'] = HardwareType(hardware_data['hardware_type'])
                if hardware_data['calibration_date']:
                    hardware_data['calibration_date'] = datetime.fromisoformat(hardware_data['calibration_date'])
                hardware_data['frequency_range'] = tuple(hardware_data['frequency_range'])
                hardware_data['gain_range'] = tuple(hardware_data['gain_range'])
                hardware_info = HardwareInfo(**hardware_data)
            
            return cls(
                iq_data=iq_data,
                timestamp=datetime.fromisoformat(json_data['timestamp']),
                duration=json_data['duration'],
                rf_params=rf_params,
                modulation_type=ModulationType(json_data['modulation_type']),
                symbol_rate=json_data['symbol_rate'],
                carrier_offset=json_data['carrier_offset'],
                quality_metrics=quality_metrics,
                location=location,
                hardware_info=hardware_info,
                device_id=json_data['device_id'],
                session_id=json_data['session_id'],
                sample_id=json_data['sample_id'],
                environment=json_data['environment'],
                weather_condition=json_data['weather_condition'],
                interference_level=json_data['interference_level'],
                preprocessing_applied=json_data['preprocessing_applied'],
                augmentation_applied=json_data['augmentation_applied'],
                custom_metadata=json_data['custom_metadata']
            )
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def get_size_bytes(self) -> int:
        """Get approximate size in bytes"""
        # IQ data size (complex64)
        iq_size = self.iq_data.nbytes
        
        # Metadata size (rough estimate)
        metadata_size = 1024  # Conservative estimate for all metadata
        
        return iq_size + metadata_size
    
    def extract_features(self) -> Dict[str, np.ndarray]:
        """Extract features for machine learning"""
        features = {}
        
        # Time domain features
        features['iq_real_imag'] = np.stack([self.iq_data.real, self.iq_data.imag])
        features['amplitude'] = np.abs(self.iq_data)
        features['phase'] = np.angle(self.iq_data)
        features['instantaneous_frequency'] = np.diff(np.unwrap(np.angle(self.iq_data)))
        
        # Frequency domain features
        fft_data = np.fft.fft(self.iq_data)
        features['magnitude_spectrum'] = np.abs(fft_data)
        features['phase_spectrum'] = np.angle(fft_data)
        features['power_spectrum'] = np.abs(fft_data) ** 2
        
        # Statistical features
        features['statistical'] = np.array([
            np.mean(self.iq_data.real), np.std(self.iq_data.real),
            np.mean(self.iq_data.imag), np.std(self.iq_data.imag),
            np.mean(features['amplitude']), np.std(features['amplitude']),
            np.var(features['amplitude']), np.max(features['amplitude']),
            self.quality_metrics.snr_db, self.quality_metrics.papr_db or 0.0
        ])
        
        return features