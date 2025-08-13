"""
Integration with public signal datasets for real-world demonstration.
Handles RadioML 2016.10, 2018.01, and GNU Radio captures.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import logging
from datetime import datetime, timedelta
import random

from src.common.dataset_downloader import DatasetDownloader
from src.common.interfaces import SignalSample
from src.common.data_distribution import GeographicDataDistributor


@dataclass
class LocationProfile:
    """RF environment profile for a specific location."""
    name: str
    latitude: float
    longitude: float
    environment_type: str  # urban, rural, indoor, maritime
    noise_floor_db: float
    interference_sources: List[str]
    propagation_model: str
    weather_effects: bool = True


@dataclass
class MultiLocationScenario:
    """Multi-location federated learning scenario configuration."""
    scenario_name: str
    locations: List[LocationProfile]
    client_distribution: Dict[str, int]  # location_name -> num_clients
    data_distribution_strategy: str
    temporal_variation: bool = True
    concept_drift_enabled: bool = False


class DatasetIntegrator:
    """Integrates public datasets into realistic multi-location scenarios."""
    
    def __init__(self, data_dir: str = "data"):
        self.downloader = DatasetDownloader(data_dir)
        self.distributor = GeographicDataDistributor()
        self.logger = logging.getLogger(__name__)
        
        # Define realistic RF environments
        self.location_profiles = {
            "urban_downtown": LocationProfile(
                name="Urban Downtown",
                latitude=40.7589, longitude=-73.9851,  # NYC Times Square
                environment_type="urban",
                noise_floor_db=-95,
                interference_sources=["cellular", "wifi", "bluetooth", "radar"],
                propagation_model="urban_canyon"
            ),
            "suburban_residential": LocationProfile(
                name="Suburban Residential", 
                latitude=37.4419, longitude=-122.1430,  # Palo Alto
                environment_type="suburban",
                noise_floor_db=-105,
                interference_sources=["wifi", "bluetooth", "microwave"],
                propagation_model="suburban"
            ),
            "rural_farmland": LocationProfile(
                name="Rural Farmland",
                latitude=41.8781, longitude=-87.6298,  # Rural Illinois
                environment_type="rural",
                noise_floor_db=-115,
                interference_sources=["amateur_radio", "agricultural"],
                propagation_model="free_space"
            ),
            "indoor_office": LocationProfile(
                name="Indoor Office Complex",
                latitude=47.6062, longitude=-122.3321,  # Seattle
                environment_type="indoor",
                noise_floor_db=-85,
                interference_sources=["wifi", "bluetooth", "fluorescent", "hvac"],
                propagation_model="indoor"
            ),
            "maritime_coastal": LocationProfile(
                name="Maritime Coastal",
                latitude=36.8508, longitude=-75.9776,  # Virginia Beach
                environment_type="maritime",
                noise_floor_db=-110,
                interference_sources=["radar", "maritime_radio", "atmospheric"],
                propagation_model="two_ray"
            )
        }
    
    def create_realistic_scenario(self, scenario_name: str = "multi_environment_demo") -> MultiLocationScenario:
        """Create a realistic multi-location federated learning scenario."""
        
        # Select diverse locations for demonstration
        selected_locations = [
            self.location_profiles["urban_downtown"],
            self.location_profiles["suburban_residential"], 
            self.location_profiles["rural_farmland"],
            self.location_profiles["indoor_office"]
        ]
        
        # Distribute clients realistically (more in urban areas)
        client_distribution = {
            "urban_downtown": 15,
            "suburban_residential": 10,
            "rural_farmland": 5,
            "indoor_office": 8
        }
        
        scenario = MultiLocationScenario(
            scenario_name=scenario_name,
            locations=selected_locations,
            client_distribution=client_distribution,
            data_distribution_strategy="geographic_iid",
            temporal_variation=True,
            concept_drift_enabled=True
        )
        
        return scenario
    
    def integrate_radioml_datasets(self) -> Dict[str, List[SignalSample]]:
        """Integrate RadioML datasets with location-specific characteristics."""
        
        # Download datasets if needed
        datasets = {}
        
        try:
            # Load RadioML 2016.10
            radioml_2016 = self.downloader.load_radioml_2016_10()
            datasets["radioml_2016_10"] = radioml_2016
            self.logger.info(f"Loaded {len(radioml_2016)} samples from RadioML 2016.10")
            
        except Exception as e:
            self.logger.warning(f"Could not load RadioML 2016.10: {e}")
            # Create synthetic data as fallback
            datasets["radioml_2016_10"] = self.downloader.create_synthetic_dataset(
                num_samples=10000, 
                modulations=['BPSK', 'QPSK', '8PSK', 'QAM16', 'QAM64']
            )
        
        try:
            # Load RadioML 2018.01
            radioml_2018 = self.downloader.load_radioml_2018_01()
            datasets["radioml_2018_01"] = radioml_2018
            self.logger.info(f"Loaded {len(radioml_2018)} samples from RadioML 2018.01")
            
        except Exception as e:
            self.logger.warning(f"Could not load RadioML 2018.01: {e}")
            # Create synthetic data as fallback
            datasets["radioml_2018_01"] = self.downloader.create_synthetic_dataset(
                num_samples=5000,
                modulations=['BPSK', 'QPSK', '8PSK', 'QAM16', 'AM', 'FM']
            )
        
        return datasets
    
    def apply_location_characteristics(self, samples: List[SignalSample], 
                                    location: LocationProfile) -> List[SignalSample]:
        """Apply location-specific RF characteristics to signal samples."""
        
        modified_samples = []
        
        for sample in samples:
            # Create a copy of the sample
            modified_sample = SignalSample(
                timestamp=sample.timestamp,
                frequency=sample.frequency,
                sample_rate=sample.sample_rate,
                iq_data=sample.iq_data.copy(),
                modulation_type=sample.modulation_type,
                snr=sample.snr,
                location={
                    "latitude": location.latitude,
                    "longitude": location.longitude,
                    "environment": location.environment_type
                },
                device_id=sample.device_id,
                metadata=sample.metadata.copy()
            )
            
            # Apply location-specific modifications
            modified_sample.iq_data = self._apply_environment_effects(
                modified_sample.iq_data, location, sample.snr
            )
            
            # Adjust SNR based on environment
            snr_adjustment = self._calculate_snr_adjustment(location)
            modified_sample.snr = max(-20, sample.snr + snr_adjustment)
            
            # Add location metadata
            modified_sample.metadata.update({
                "location_profile": location.name,
                "environment_type": location.environment_type,
                "noise_floor_db": location.noise_floor_db,
                "interference_sources": location.interference_sources,
                "snr_adjustment_db": snr_adjustment
            })
            
            modified_samples.append(modified_sample)
        
        return modified_samples
    
    def distribute_data_by_location(self, datasets: Dict[str, List[SignalSample]], 
                                  scenario: MultiLocationScenario) -> Dict[str, Dict[str, List[SignalSample]]]:
        """Distribute dataset samples across locations according to scenario."""
        
        location_datasets = {}
        
        for location in scenario.locations:
            location_datasets[location.name] = {}
            
            for dataset_name, samples in datasets.items():
                # Calculate number of samples for this location
                total_clients = sum(scenario.client_distribution.values())
                location_clients = scenario.client_distribution.get(location.name, 0)
                location_fraction = location_clients / total_clients
                
                # Select samples for this location
                num_samples = int(len(samples) * location_fraction)
                location_samples = random.sample(samples, min(num_samples, len(samples)))
                
                # Apply location characteristics
                location_samples = self.apply_location_characteristics(location_samples, location)
                
                location_datasets[location.name][dataset_name] = location_samples
                
                self.logger.info(f"Assigned {len(location_samples)} {dataset_name} samples to {location.name}")
        
        return location_datasets
    
    def create_temporal_variations(self, samples: List[SignalSample], 
                                 time_span_hours: int = 24) -> List[SignalSample]:
        """Create temporal variations in signal characteristics."""
        
        varied_samples = []
        start_time = datetime.now()
        
        for i, sample in enumerate(samples):
            # Assign timestamp within time span
            time_offset = timedelta(hours=random.uniform(0, time_span_hours))
            sample_time = start_time + time_offset
            
            # Create time-varying effects
            hour_of_day = sample_time.hour
            
            # Simulate daily patterns (more interference during business hours)
            if 8 <= hour_of_day <= 18:  # Business hours
                interference_factor = 1.5
                snr_penalty = -3
            elif 22 <= hour_of_day or hour_of_day <= 6:  # Night hours
                interference_factor = 0.7
                snr_penalty = 2
            else:  # Evening hours
                interference_factor = 1.0
                snr_penalty = 0
            
            # Apply temporal effects
            modified_sample = SignalSample(
                timestamp=sample_time,
                frequency=sample.frequency,
                sample_rate=sample.sample_rate,
                iq_data=self._apply_temporal_effects(sample.iq_data, interference_factor),
                modulation_type=sample.modulation_type,
                snr=max(-20, sample.snr + snr_penalty),
                location=sample.location,
                device_id=sample.device_id,
                metadata={
                    **sample.metadata,
                    "hour_of_day": hour_of_day,
                    "interference_factor": interference_factor,
                    "temporal_snr_penalty": snr_penalty
                }
            )
            
            varied_samples.append(modified_sample)
        
        return varied_samples
    
    def _apply_environment_effects(self, iq_data: np.ndarray, location: LocationProfile, 
                                 original_snr: float) -> np.ndarray:
        """Apply environment-specific effects to IQ data."""
        
        modified_data = iq_data.copy()
        
        # Add environment-specific noise
        noise_power = self._calculate_noise_power(location, original_snr)
        noise = (np.random.normal(0, np.sqrt(noise_power/2), len(iq_data)) + 
                1j * np.random.normal(0, np.sqrt(noise_power/2), len(iq_data)))
        
        modified_data += noise
        
        # Apply propagation effects
        if location.propagation_model == "urban_canyon":
            # Multipath fading
            modified_data = self._apply_multipath_fading(modified_data, num_paths=3)
        elif location.propagation_model == "indoor":
            # Indoor fading and reflections
            modified_data = self._apply_indoor_effects(modified_data)
        elif location.propagation_model == "two_ray":
            # Two-ray ground reflection model
            modified_data = self._apply_two_ray_effects(modified_data)
        
        # Add interference based on sources
        for interference_source in location.interference_sources:
            modified_data = self._add_interference(modified_data, interference_source)
        
        return modified_data
    
    def _calculate_snr_adjustment(self, location: LocationProfile) -> float:
        """Calculate SNR adjustment based on location characteristics."""
        
        base_adjustment = 0
        
        # Environment-based adjustments
        if location.environment_type == "urban":
            base_adjustment -= 5  # More interference
        elif location.environment_type == "rural":
            base_adjustment += 3  # Less interference
        elif location.environment_type == "indoor":
            base_adjustment -= 8  # Significant attenuation
        elif location.environment_type == "maritime":
            base_adjustment += 2  # Less terrestrial interference
        
        # Interference source penalties
        interference_penalty = len(location.interference_sources) * -1.5
        
        return base_adjustment + interference_penalty
    
    def _calculate_noise_power(self, location: LocationProfile, original_snr: float) -> float:
        """Calculate noise power based on location and SNR."""
        
        # Assume unit signal power for simplicity
        signal_power = 1.0
        
        # Calculate noise power from SNR
        snr_linear = 10 ** (original_snr / 10)
        base_noise_power = signal_power / snr_linear
        
        # Adjust based on location noise floor
        noise_floor_linear = 10 ** (location.noise_floor_db / 10)
        
        return max(base_noise_power, noise_floor_linear * 0.1)
    
    def _apply_multipath_fading(self, iq_data: np.ndarray, num_paths: int = 3) -> np.ndarray:
        """Apply multipath fading effects."""
        
        result = np.zeros_like(iq_data, dtype=complex)
        
        for path in range(num_paths):
            # Random delay (in samples)
            delay = random.randint(0, min(10, len(iq_data) // 4))
            
            # Random attenuation
            attenuation = random.uniform(0.3, 1.0) if path == 0 else random.uniform(0.1, 0.5)
            
            # Random phase shift
            phase_shift = random.uniform(0, 2 * np.pi)
            
            # Apply path effects
            delayed_signal = np.roll(iq_data, delay) * attenuation * np.exp(1j * phase_shift)
            result += delayed_signal
        
        return result / num_paths
    
    def _apply_indoor_effects(self, iq_data: np.ndarray) -> np.ndarray:
        """Apply indoor propagation effects."""
        
        # Simulate wall reflections and attenuation
        attenuated = iq_data * random.uniform(0.3, 0.7)
        
        # Add reflection with delay
        reflection_delay = random.randint(1, 5)
        reflection_strength = random.uniform(0.1, 0.3)
        reflection = np.roll(iq_data, reflection_delay) * reflection_strength
        
        return attenuated + reflection
    
    def _apply_two_ray_effects(self, iq_data: np.ndarray) -> np.ndarray:
        """Apply two-ray ground reflection model."""
        
        # Direct path
        direct = iq_data
        
        # Ground reflection path with delay and phase inversion
        reflection_delay = random.randint(1, 3)
        reflected = -np.roll(iq_data, reflection_delay) * random.uniform(0.5, 0.8)
        
        return direct + reflected
    
    def _add_interference(self, iq_data: np.ndarray, interference_type: str) -> np.ndarray:
        """Add specific types of interference."""
        
        interference_power = random.uniform(0.05, 0.2)
        
        if interference_type == "wifi":
            # OFDM-like interference
            interference = self._generate_ofdm_interference(len(iq_data), interference_power)
        elif interference_type == "bluetooth":
            # Frequency hopping interference
            interference = self._generate_fh_interference(len(iq_data), interference_power)
        elif interference_type == "cellular":
            # Cellular-like interference
            interference = self._generate_cellular_interference(len(iq_data), interference_power)
        elif interference_type == "radar":
            # Pulsed radar interference
            interference = self._generate_radar_interference(len(iq_data), interference_power)
        else:
            # Generic broadband interference
            interference = (np.random.normal(0, np.sqrt(interference_power/2), len(iq_data)) + 
                          1j * np.random.normal(0, np.sqrt(interference_power/2), len(iq_data)))
        
        return iq_data + interference
    
    def _generate_ofdm_interference(self, length: int, power: float) -> np.ndarray:
        """Generate OFDM-like interference."""
        
        # Create OFDM subcarriers
        num_subcarriers = 64
        subcarrier_data = np.random.choice([-1, 1], num_subcarriers) + 1j * np.random.choice([-1, 1], num_subcarriers)
        
        # IFFT to create time domain signal
        time_signal = np.fft.ifft(subcarrier_data)
        
        # Repeat and truncate to desired length
        repeated = np.tile(time_signal, (length // len(time_signal)) + 1)[:length]
        
        # Scale to desired power
        return repeated * np.sqrt(power / np.mean(np.abs(repeated)**2))
    
    def _generate_fh_interference(self, length: int, power: float) -> np.ndarray:
        """Generate frequency hopping interference."""
        
        # Simulate frequency hops
        hop_duration = length // 10  # 10 hops
        signal = np.zeros(length, dtype=complex)
        
        for hop in range(10):
            start_idx = hop * hop_duration
            end_idx = min((hop + 1) * hop_duration, length)
            
            # Random frequency for this hop
            freq = random.uniform(-0.4, 0.4)  # Normalized frequency
            t = np.arange(end_idx - start_idx)
            hop_signal = np.exp(1j * 2 * np.pi * freq * t)
            
            signal[start_idx:end_idx] = hop_signal
        
        return signal * np.sqrt(power)
    
    def _generate_cellular_interference(self, length: int, power: float) -> np.ndarray:
        """Generate cellular-like interference."""
        
        # Simulate CDMA-like spreading
        spreading_code = np.random.choice([-1, 1], length)
        data_bits = np.random.choice([-1, 1], length // 8)
        
        # Repeat data bits to match length
        repeated_bits = np.repeat(data_bits, 8)[:length]
        
        # Apply spreading
        spread_signal = repeated_bits * spreading_code
        
        return spread_signal * np.sqrt(power) * (1 + 1j) / np.sqrt(2)
    
    def _generate_radar_interference(self, length: int, power: float) -> np.ndarray:
        """Generate pulsed radar interference."""
        
        # Create pulsed signal
        pulse_width = length // 20  # 5% duty cycle
        pulse_period = length // 4   # 4 pulses
        
        signal = np.zeros(length, dtype=complex)
        
        for pulse in range(4):
            start_idx = pulse * pulse_period
            end_idx = min(start_idx + pulse_width, length)
            
            # Linear chirp for radar pulse
            t = np.arange(end_idx - start_idx)
            chirp_rate = 0.1
            pulse_signal = np.exp(1j * np.pi * chirp_rate * t**2)
            
            signal[start_idx:end_idx] = pulse_signal
        
        return signal * np.sqrt(power)
    
    def _apply_temporal_effects(self, iq_data: np.ndarray, interference_factor: float) -> np.ndarray:
        """Apply temporal variations to signal."""
        
        # Scale existing interference
        noise_power = np.var(iq_data) * 0.1 * interference_factor
        additional_noise = (np.random.normal(0, np.sqrt(noise_power/2), len(iq_data)) + 
                          1j * np.random.normal(0, np.sqrt(noise_power/2), len(iq_data)))
        
        return iq_data + additional_noise
    
    def get_scenario_summary(self, scenario: MultiLocationScenario) -> Dict[str, Any]:
        """Get summary statistics for a scenario."""
        
        total_clients = sum(scenario.client_distribution.values())
        
        summary = {
            "scenario_name": scenario.scenario_name,
            "total_locations": len(scenario.locations),
            "total_clients": total_clients,
            "location_details": [],
            "environment_distribution": {},
            "data_distribution_strategy": scenario.data_distribution_strategy
        }
        
        for location in scenario.locations:
            num_clients = scenario.client_distribution.get(location.name, 0)
            
            summary["location_details"].append({
                "name": location.name,
                "coordinates": (location.latitude, location.longitude),
                "environment": location.environment_type,
                "clients": num_clients,
                "client_percentage": (num_clients / total_clients) * 100,
                "noise_floor_db": location.noise_floor_db,
                "interference_sources": len(location.interference_sources)
            })
            
            # Count environment types
            env_type = location.environment_type
            summary["environment_distribution"][env_type] = summary["environment_distribution"].get(env_type, 0) + num_clients
        
        return summary