"""
Data distribution system to simulate geographically distributed clients
"""
import numpy as np
import random
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path
import json
from datetime import datetime, timedelta

from .interfaces import SignalSample, ClientInfo


@dataclass
class GeographicRegion:
    """Geographic region definition"""
    name: str
    center_lat: float
    center_lon: float
    radius_km: float
    signal_characteristics: Dict[str, Any]
    client_density: float  # clients per km²


@dataclass
class ClientProfile:
    """Client profile for data distribution"""
    client_id: str
    region: str
    location: Dict[str, float]
    device_type: str
    data_preferences: Dict[str, Any]
    quality_factor: float  # 0.0 to 1.0
    availability_pattern: Dict[str, Any]


class GeographicDataDistributor:
    """Distributes data across geographically distributed clients"""
    
    def __init__(self):
        self.regions = self._initialize_default_regions()
        self.clients = []
        self.data_assignments = {}
    
    def _initialize_default_regions(self) -> List[GeographicRegion]:
        """Initialize default geographic regions"""
        return [
            GeographicRegion(
                name="urban_east",
                center_lat=40.7128,
                center_lon=-74.0060,  # New York
                radius_km=50,
                signal_characteristics={
                    "interference_level": "high",
                    "multipath": "severe",
                    "dominant_modulations": ["QAM64", "QAM16", "OFDM"],
                    "frequency_bands": ["2.4GHz", "5GHz", "LTE"],
                    "snr_range": [-5, 15]
                },
                client_density=100.0
            ),
            GeographicRegion(
                name="suburban_west",
                center_lat=37.7749,
                center_lon=-122.4194,  # San Francisco
                radius_km=75,
                signal_characteristics={
                    "interference_level": "medium",
                    "multipath": "moderate",
                    "dominant_modulations": ["QPSK", "QAM16", "8PSK"],
                    "frequency_bands": ["900MHz", "1.8GHz", "2.4GHz"],
                    "snr_range": [0, 20]
                },
                client_density=50.0
            ),
            GeographicRegion(
                name="rural_central",
                center_lat=39.8283,
                center_lon=-98.5795,  # Geographic center of US
                radius_km=100,
                signal_characteristics={
                    "interference_level": "low",
                    "multipath": "minimal",
                    "dominant_modulations": ["BPSK", "QPSK", "AM"],
                    "frequency_bands": ["VHF", "UHF", "900MHz"],
                    "snr_range": [10, 30]
                },
                client_density=10.0
            ),
            GeographicRegion(
                name="industrial_south",
                center_lat=29.7604,
                center_lon=-95.3698,  # Houston
                radius_km=60,
                signal_characteristics={
                    "interference_level": "very_high",
                    "multipath": "high",
                    "dominant_modulations": ["FSK", "PSK", "QAM"],
                    "frequency_bands": ["ISM", "Industrial", "Cellular"],
                    "snr_range": [-10, 10]
                },
                client_density=75.0
            ),
            GeographicRegion(
                name="coastal_northwest",
                center_lat=47.6062,
                center_lon=-122.3321,  # Seattle
                radius_km=80,
                signal_characteristics={
                    "interference_level": "medium",
                    "multipath": "high",  # Due to terrain
                    "dominant_modulations": ["OFDM", "QAM16", "QPSK"],
                    "frequency_bands": ["2.4GHz", "5GHz", "Maritime"],
                    "snr_range": [5, 25]
                },
                client_density=60.0
            )
        ]
    
    def generate_client_profiles(self, num_clients: int) -> List[ClientProfile]:
        """Generate client profiles distributed across regions"""
        profiles = []
        
        # Calculate clients per region based on density
        region_weights = [r.client_density * (r.radius_km ** 2) for r in self.regions]
        total_weight = sum(region_weights)
        region_probabilities = [w / total_weight for w in region_weights]
        
        for i in range(num_clients):
            # Select region based on density
            region = np.random.choice(self.regions, p=region_probabilities)
            
            # Generate random location within region
            location = self._generate_location_in_region(region)
            
            # Generate client profile
            profile = ClientProfile(
                client_id=f"client_{i:04d}",
                region=region.name,
                location=location,
                device_type=self._select_device_type(region),
                data_preferences=self._generate_data_preferences(region),
                quality_factor=self._generate_quality_factor(region),
                availability_pattern=self._generate_availability_pattern()
            )
            
            profiles.append(profile)
        
        self.clients = profiles
        return profiles
    
    def distribute_data(self, samples: List[SignalSample], 
                       client_profiles: List[ClientProfile],
                       distribution_strategy: str = "iid") -> Dict[str, List[SignalSample]]:
        """Distribute data samples to clients"""
        
        if distribution_strategy == "iid":
            return self._distribute_iid(samples, client_profiles)
        elif distribution_strategy == "non_iid_geographic":
            return self._distribute_non_iid_geographic(samples, client_profiles)
        elif distribution_strategy == "non_iid_modulation":
            return self._distribute_non_iid_modulation(samples, client_profiles)
        elif distribution_strategy == "realistic_mixed":
            return self._distribute_realistic_mixed(samples, client_profiles)
        else:
            raise ValueError(f"Unknown distribution strategy: {distribution_strategy}")
    
    def _generate_location_in_region(self, region: GeographicRegion) -> Dict[str, float]:
        """Generate random location within a geographic region"""
        # Generate random point within circle
        angle = np.random.uniform(0, 2 * np.pi)
        radius = np.random.uniform(0, region.radius_km)
        
        # Convert to lat/lon offset (approximate)
        lat_offset = (radius * np.cos(angle)) / 111.0  # ~111 km per degree
        lon_offset = (radius * np.sin(angle)) / (111.0 * np.cos(np.radians(region.center_lat)))
        
        return {
            "latitude": region.center_lat + lat_offset,
            "longitude": region.center_lon + lon_offset,
            "altitude": np.random.uniform(0, 500)  # meters
        }
    
    def _select_device_type(self, region: GeographicRegion) -> str:
        """Select device type based on region characteristics"""
        if region.name.startswith("urban"):
            return np.random.choice(["smartphone", "tablet", "laptop"], p=[0.6, 0.2, 0.2])
        elif region.name.startswith("suburban"):
            return np.random.choice(["smartphone", "iot_sensor", "base_station"], p=[0.5, 0.3, 0.2])
        elif region.name.startswith("rural"):
            return np.random.choice(["sdr_device", "base_station", "iot_sensor"], p=[0.4, 0.3, 0.3])
        elif region.name.startswith("industrial"):
            return np.random.choice(["iot_sensor", "industrial_gateway", "sdr_device"], p=[0.5, 0.3, 0.2])
        else:
            return np.random.choice(["smartphone", "sdr_device", "iot_sensor"], p=[0.4, 0.3, 0.3])
    
    def _generate_data_preferences(self, region: GeographicRegion) -> Dict[str, Any]:
        """Generate data preferences based on region"""
        preferences = {
            "preferred_modulations": region.signal_characteristics["dominant_modulations"],
            "frequency_bands": region.signal_characteristics["frequency_bands"],
            "min_snr": region.signal_characteristics["snr_range"][0],
            "max_samples_per_round": np.random.randint(100, 1000),
            "quality_threshold": np.random.uniform(0.3, 0.8)
        }
        return preferences
    
    def _generate_quality_factor(self, region: GeographicRegion) -> float:
        """Generate quality factor based on region characteristics"""
        base_quality = {
            "urban_east": 0.7,
            "suburban_west": 0.8,
            "rural_central": 0.9,
            "industrial_south": 0.6,
            "coastal_northwest": 0.75
        }.get(region.name, 0.7)
        
        # Add some randomness
        return np.clip(np.random.normal(base_quality, 0.1), 0.1, 1.0)
    
    def _generate_availability_pattern(self) -> Dict[str, Any]:
        """Generate availability pattern for client"""
        patterns = ["always_on", "business_hours", "evening_weekend", "random"]
        pattern = np.random.choice(patterns, p=[0.2, 0.3, 0.3, 0.2])
        
        return {
            "pattern": pattern,
            "availability_probability": np.random.uniform(0.6, 0.95),
            "session_duration_minutes": np.random.randint(10, 120),
            "preferred_hours": self._get_preferred_hours(pattern)
        }
    
    def _get_preferred_hours(self, pattern: str) -> List[int]:
        """Get preferred hours based on availability pattern"""
        if pattern == "always_on":
            return list(range(24))
        elif pattern == "business_hours":
            return list(range(9, 17))
        elif pattern == "evening_weekend":
            return list(range(18, 23)) + [0, 1] + list(range(6, 23))  # Weekends approximation
        else:  # random
            return sorted(np.random.choice(range(24), size=np.random.randint(8, 16), replace=False))
    
    def _distribute_iid(self, samples: List[SignalSample], 
                       client_profiles: List[ClientProfile]) -> Dict[str, List[SignalSample]]:
        """Distribute data in IID manner"""
        # Shuffle samples
        shuffled_samples = samples.copy()
        random.shuffle(shuffled_samples)
        
        # Distribute evenly
        samples_per_client = len(shuffled_samples) // len(client_profiles)
        distribution = {}
        
        for i, profile in enumerate(client_profiles):
            start_idx = i * samples_per_client
            end_idx = start_idx + samples_per_client
            if i == len(client_profiles) - 1:  # Last client gets remaining samples
                end_idx = len(shuffled_samples)
            
            distribution[profile.client_id] = shuffled_samples[start_idx:end_idx]
        
        return distribution
    
    def _distribute_non_iid_geographic(self, samples: List[SignalSample], 
                                     client_profiles: List[ClientProfile]) -> Dict[str, List[SignalSample]]:
        """Distribute data based on geographic characteristics"""
        distribution = {profile.client_id: [] for profile in client_profiles}
        
        # Group samples by characteristics that match regions
        for sample in samples:
            # Find best matching clients based on signal characteristics
            matching_clients = []
            
            for profile in client_profiles:
                region = next(r for r in self.regions if r.name == profile.region)
                
                # Check if sample characteristics match region
                match_score = self._calculate_match_score(sample, region, profile)
                
                if match_score > 0.3:  # Threshold for assignment
                    matching_clients.append((profile, match_score))
            
            if matching_clients:
                # Sort by match score and assign to best matches
                matching_clients.sort(key=lambda x: x[1], reverse=True)
                
                # Assign to top matching clients (with some randomness)
                num_assignments = min(3, len(matching_clients))
                selected_clients = matching_clients[:num_assignments]
                
                for client_profile, _ in selected_clients:
                    if len(distribution[client_profile.client_id]) < profile.data_preferences["max_samples_per_round"]:
                        distribution[client_profile.client_id].append(sample)
        
        return distribution
    
    def _distribute_non_iid_modulation(self, samples: List[SignalSample], 
                                     client_profiles: List[ClientProfile]) -> Dict[str, List[SignalSample]]:
        """Distribute data based on modulation type preferences"""
        distribution = {profile.client_id: [] for profile in client_profiles}
        
        # Group samples by modulation type
        modulation_groups = {}
        for sample in samples:
            mod_type = sample.modulation_type
            if mod_type not in modulation_groups:
                modulation_groups[mod_type] = []
            modulation_groups[mod_type].append(sample)
        
        # Assign modulation groups to clients based on preferences
        for profile in client_profiles:
            preferred_mods = profile.data_preferences["preferred_modulations"]
            max_samples = profile.data_preferences["max_samples_per_round"]
            
            assigned_samples = []
            
            # Prioritize preferred modulations
            for mod_type in preferred_mods:
                if mod_type in modulation_groups and len(assigned_samples) < max_samples:
                    available_samples = modulation_groups[mod_type]
                    num_to_assign = min(len(available_samples), max_samples - len(assigned_samples))
                    
                    # Randomly select samples from this modulation type
                    selected_samples = random.sample(available_samples, num_to_assign)
                    assigned_samples.extend(selected_samples)
                    
                    # Remove assigned samples from pool
                    for sample in selected_samples:
                        modulation_groups[mod_type].remove(sample)
            
            distribution[profile.client_id] = assigned_samples
        
        return distribution
    
    def _distribute_realistic_mixed(self, samples: List[SignalSample], 
                                  client_profiles: List[ClientProfile]) -> Dict[str, List[SignalSample]]:
        """Distribute data using realistic mixed strategy"""
        distribution = {profile.client_id: [] for profile in client_profiles}
        
        # Apply geographic filtering first
        geo_filtered = {}
        for profile in client_profiles:
            region = next(r for r in self.regions if r.name == profile.region)
            geo_filtered[profile.client_id] = []
            
            for sample in samples:
                match_score = self._calculate_match_score(sample, region, profile)
                
                # Apply quality factor and availability
                if (match_score > 0.2 and 
                    random.random() < profile.quality_factor and
                    random.random() < profile.availability_pattern["availability_probability"]):
                    geo_filtered[profile.client_id].append(sample)
        
        # Then apply modulation preferences
        for profile in client_profiles:
            available_samples = geo_filtered[profile.client_id]
            preferred_mods = profile.data_preferences["preferred_modulations"]
            max_samples = profile.data_preferences["max_samples_per_round"]
            
            # Score samples based on preferences
            scored_samples = []
            for sample in available_samples:
                score = 1.0 if sample.modulation_type in preferred_mods else 0.3
                
                # Adjust score based on SNR preference
                snr_min = profile.data_preferences["min_snr"]
                if sample.snr >= snr_min:
                    score *= 1.2
                else:
                    score *= 0.5
                
                scored_samples.append((sample, score))
            
            # Sort by score and select top samples
            scored_samples.sort(key=lambda x: x[1], reverse=True)
            selected_samples = [s[0] for s in scored_samples[:max_samples]]
            
            distribution[profile.client_id] = selected_samples
        
        return distribution
    
    def _calculate_match_score(self, sample: SignalSample, 
                             region: GeographicRegion, 
                             profile: ClientProfile) -> float:
        """Calculate how well a sample matches a region and client profile"""
        score = 0.0
        
        # Modulation type match
        if sample.modulation_type in region.signal_characteristics["dominant_modulations"]:
            score += 0.4
        
        # SNR range match
        snr_range = region.signal_characteristics["snr_range"]
        if snr_range[0] <= sample.snr <= snr_range[1]:
            score += 0.3
        
        # Device type compatibility
        device_compatibility = {
            "smartphone": ["QAM64", "QAM16", "OFDM", "LTE"],
            "sdr_device": ["BPSK", "QPSK", "8PSK", "AM", "FM"],
            "iot_sensor": ["FSK", "BPSK", "LoRa"],
            "base_station": ["QAM64", "QAM16", "OFDM", "LTE", "5G"],
            "industrial_gateway": ["FSK", "PSK", "Industrial"]
        }
        
        compatible_mods = device_compatibility.get(profile.device_type, [])
        if sample.modulation_type in compatible_mods:
            score += 0.2
        
        # Quality threshold
        if sample.snr >= profile.data_preferences.get("quality_threshold", 0) * 30 - 10:
            score += 0.1
        
        return min(score, 1.0)
    
    def simulate_temporal_distribution(self, samples: List[SignalSample], 
                                     client_profiles: List[ClientProfile],
                                     simulation_hours: int = 24) -> Dict[str, Dict[int, List[SignalSample]]]:
        """Simulate temporal distribution of data over time"""
        temporal_distribution = {}
        
        for profile in client_profiles:
            temporal_distribution[profile.client_id] = {}
            preferred_hours = profile.availability_pattern["preferred_hours"]
            availability_prob = profile.availability_pattern["availability_probability"]
            
            # Distribute samples across preferred hours
            client_samples = self.data_assignments.get(profile.client_id, [])
            
            for hour in range(simulation_hours):
                temporal_distribution[profile.client_id][hour] = []
                
                if hour % 24 in preferred_hours and random.random() < availability_prob:
                    # Assign some samples for this hour
                    num_samples = min(
                        len(client_samples) // len(preferred_hours) + 1,
                        random.randint(1, 10)
                    )
                    
                    if client_samples:
                        hour_samples = random.sample(
                            client_samples, 
                            min(num_samples, len(client_samples))
                        )
                        temporal_distribution[profile.client_id][hour] = hour_samples
        
        return temporal_distribution
    
    def get_distribution_statistics(self, distribution: Dict[str, List[SignalSample]]) -> Dict[str, Any]:
        """Get statistics about data distribution"""
        stats = {
            "total_clients": len(distribution),
            "total_samples": sum(len(samples) for samples in distribution.values()),
            "samples_per_client": {},
            "modulation_distribution": {},
            "snr_distribution": {},
            "regional_distribution": {}
        }
        
        # Per-client statistics
        for client_id, samples in distribution.items():
            stats["samples_per_client"][client_id] = len(samples)
            
            # Find client profile
            profile = next((p for p in self.clients if p.client_id == client_id), None)
            if profile:
                region = profile.region
                if region not in stats["regional_distribution"]:
                    stats["regional_distribution"][region] = 0
                stats["regional_distribution"][region] += len(samples)
        
        # Modulation and SNR distribution
        all_samples = [sample for samples in distribution.values() for sample in samples]
        
        for sample in all_samples:
            mod_type = sample.modulation_type
            stats["modulation_distribution"][mod_type] = stats["modulation_distribution"].get(mod_type, 0) + 1
            
            snr_bin = f"{int(sample.snr//5)*5}-{int(sample.snr//5)*5+5}dB"
            stats["snr_distribution"][snr_bin] = stats["snr_distribution"].get(snr_bin, 0) + 1
        
        return stats
    
    def save_distribution_config(self, filepath: str):
        """Save distribution configuration to file"""
        config = {
            "regions": [
                {
                    "name": r.name,
                    "center_lat": r.center_lat,
                    "center_lon": r.center_lon,
                    "radius_km": r.radius_km,
                    "signal_characteristics": r.signal_characteristics,
                    "client_density": r.client_density
                }
                for r in self.regions
            ],
            "clients": [
                {
                    "client_id": c.client_id,
                    "region": c.region,
                    "location": c.location,
                    "device_type": c.device_type,
                    "data_preferences": c.data_preferences,
                    "quality_factor": c.quality_factor,
                    "availability_pattern": c.availability_pattern
                }
                for c in self.clients
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2, default=str)
    
    def load_distribution_config(self, filepath: str):
        """Load distribution configuration from file"""
        with open(filepath, 'r') as f:
            config = json.load(f)
        
        # Load regions
        self.regions = []
        for r_config in config["regions"]:
            region = GeographicRegion(**r_config)
            self.regions.append(region)
        
        # Load clients
        self.clients = []
        for c_config in config["clients"]:
            client = ClientProfile(**c_config)
            self.clients.append(client)