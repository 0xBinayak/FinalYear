"""
Resource Management and Optimization for Edge Coordinator
"""
import asyncio
import logging
import time
import json
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import threading
import psutil
import platform

from ..common.federated_data_structures import ComputeResources, NetworkConditions
from ..common.interfaces import ClientInfo


class ResourceType(Enum):
    """Types of resources to manage"""
    CPU = "cpu"
    MEMORY = "memory"
    NETWORK = "network"
    STORAGE = "storage"
    BATTERY = "battery"
    THERMAL = "thermal"


class SchedulingStrategy(Enum):
    """Training scheduling strategies"""
    ROUND_ROBIN = "round_robin"
    RESOURCE_AWARE = "resource_aware"
    PRIORITY_BASED = "priority_based"
    LOAD_BALANCED = "load_balanced"
    ENERGY_EFFICIENT = "energy_efficient"


@dataclass
class ResourceProfile:
    """Resource profile for a client device"""
    client_id: str
    cpu_cores: int
    cpu_frequency_ghz: float
    memory_total_gb: float
    memory_available_gb: float
    storage_total_gb: float
    storage_available_gb: float
    network_bandwidth_mbps: float
    battery_level: Optional[float] = None  # 0.0 to 1.0
    thermal_state: str = "normal"  # normal, warm, hot, critical
    power_source: str = "unknown"  # battery, plugged, unknown
    
    # Performance characteristics
    training_speed_samples_per_sec: float = 0.0
    communication_latency_ms: float = 0.0
    reliability_score: float = 1.0  # 0.0 to 1.0
    
    # Resource utilization history
    cpu_utilization_history: List[float] = field(default_factory=list)
    memory_utilization_history: List[float] = field(default_factory=list)
    network_utilization_history: List[float] = field(default_factory=list)
    
    # Timestamps
    last_updated: datetime = field(default_factory=datetime.now)
    created_at: datetime = field(default_factory=datetime.now)
    
    def get_cpu_utilization(self) -> float:
        """Get current CPU utilization"""
        if self.cpu_utilization_history:
            return self.cpu_utilization_history[-1]
        return 0.0
    
    def get_memory_utilization(self) -> float:
        """Get current memory utilization"""
        if self.memory_total_gb > 0:
            return (self.memory_total_gb - self.memory_available_gb) / self.memory_total_gb
        return 0.0
    
    def get_resource_score(self) -> float:
        """Calculate overall resource availability score"""
        cpu_score = min(1.0, self.cpu_cores / 8.0)  # Normalize to 8 cores
        memory_score = min(1.0, self.memory_available_gb / 16.0)  # Normalize to 16GB
        network_score = min(1.0, self.network_bandwidth_mbps / 100.0)  # Normalize to 100Mbps
        
        # Battery penalty
        battery_score = 1.0
        if self.battery_level is not None:
            battery_score = max(0.1, self.battery_level)
        
        # Thermal penalty
        thermal_score = 1.0
        if self.thermal_state == "warm":
            thermal_score = 0.8
        elif self.thermal_state == "hot":
            thermal_score = 0.5
        elif self.thermal_state == "critical":
            thermal_score = 0.1
        
        # Weighted average
        weights = [0.3, 0.3, 0.2, 0.1, 0.1]  # CPU, Memory, Network, Battery, Thermal
        scores = [cpu_score, memory_score, network_score, battery_score, thermal_score]
        
        return sum(w * s for w, s in zip(weights, scores))
    
    def can_handle_training(self, model_size_mb: float, batch_size: int, 
                           estimated_duration_minutes: float) -> bool:
        """Check if device can handle training task"""
        # Memory check
        memory_needed_gb = model_size_mb / 1024 + batch_size * 0.01  # Rough estimate
        if memory_needed_gb > self.memory_available_gb:
            return False
        
        # Battery check for mobile devices
        if self.battery_level is not None:
            if self.power_source == "battery" and self.battery_level < 0.3:
                return False
            
            # Estimate battery consumption
            estimated_battery_drain = estimated_duration_minutes * 0.01  # 1% per minute rough estimate
            if self.battery_level < estimated_battery_drain + 0.1:  # Keep 10% buffer
                return False
        
        # Thermal check
        if self.thermal_state in ["hot", "critical"]:
            return False
        
        # CPU utilization check
        if self.get_cpu_utilization() > 0.9:
            return False
        
        return True
    
    def update_utilization(self, cpu_util: float, memory_util: float, network_util: float):
        """Update resource utilization history"""
        max_history = 100  # Keep last 100 measurements
        
        self.cpu_utilization_history.append(cpu_util)
        if len(self.cpu_utilization_history) > max_history:
            self.cpu_utilization_history.pop(0)
        
        self.memory_utilization_history.append(memory_util)
        if len(self.memory_utilization_history) > max_history:
            self.memory_utilization_history.pop(0)
        
        self.network_utilization_history.append(network_util)
        if len(self.network_utilization_history) > max_history:
            self.network_utilization_history.pop(0)
        
        self.last_updated = datetime.now()


@dataclass
class TrainingTask:
    """Training task to be scheduled"""
    task_id: str
    client_id: str
    model_size_mb: float
    batch_size: int
    estimated_epochs: int
    estimated_duration_minutes: float
    priority: float = 1.0
    deadline: Optional[datetime] = None
    resource_requirements: Dict[str, float] = field(default_factory=dict)
    
    # Scheduling metadata
    created_at: datetime = field(default_factory=datetime.now)
    scheduled_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def is_overdue(self) -> bool:
        """Check if task is overdue"""
        if self.deadline is None:
            return False
        return datetime.now() > self.deadline
    
    def get_age_minutes(self) -> float:
        """Get task age in minutes"""
        return (datetime.now() - self.created_at).total_seconds() / 60


class ResourceManager:
    """
    Resource Manager for Edge Coordinator
    
    Handles:
    - Device capability assessment and profiling
    - Adaptive training scheduling based on resource constraints
    - Local data quality validation and preprocessing
    - Load balancing across edge clients
    """
    
    def __init__(self, coordinator_id: str, config: Dict[str, Any]):
        self.coordinator_id = coordinator_id
        self.config = config
        self.logger = logging.getLogger(f"ResourceManager-{coordinator_id}")
        
        # Resource tracking
        self.client_profiles: Dict[str, ResourceProfile] = {}
        self.training_tasks: Dict[str, TrainingTask] = {}
        self.active_training: Dict[str, str] = {}  # client_id -> task_id
        
        # Configuration
        self.scheduling_strategy = SchedulingStrategy(config.get('scheduling_strategy', 'resource_aware'))
        self.max_concurrent_training = config.get('max_concurrent_training', 5)
        self.resource_update_interval = config.get('resource_update_interval', 60)  # seconds
        self.load_balancing_enabled = config.get('load_balancing_enabled', True)
        
        # Performance tracking
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Background tasks
        self.running = False
        self.background_tasks: List[asyncio.Task] = []
    
    async def start(self):
        """Start resource manager"""
        self.running = True
        
        self.background_tasks = [
            asyncio.create_task(self._resource_monitoring_loop()),
            asyncio.create_task(self._task_scheduling_loop()),
            asyncio.create_task(self._performance_analysis_loop()),
            asyncio.create_task(self._load_balancing_loop())
        ]
        
        self.logger.info("Resource manager started")
    
    async def stop(self):
        """Stop resource manager"""
        self.running = False
        
        for task in self.background_tasks:
            task.cancel()
        
        await asyncio.gather(*self.background_tasks, return_exceptions=True)
        self.logger.info("Resource manager stopped")
    
    def register_client(self, client_info: ClientInfo) -> ResourceProfile:
        """Register client and create resource profile"""
        # Extract resource information from client info
        capabilities = client_info.capabilities
        hardware_specs = client_info.hardware_specs
        
        profile = ResourceProfile(
            client_id=client_info.client_id,
            cpu_cores=capabilities.get('cpu_cores', 1),
            cpu_frequency_ghz=capabilities.get('cpu_frequency_ghz', 1.0),
            memory_total_gb=capabilities.get('memory_gb', 1.0),
            memory_available_gb=capabilities.get('available_memory_gb', 0.8),
            storage_total_gb=capabilities.get('storage_gb', 10.0),
            storage_available_gb=capabilities.get('available_storage_gb', 8.0),
            network_bandwidth_mbps=client_info.network_info.get('bandwidth_mbps', 10.0),
            battery_level=capabilities.get('battery_level'),
            thermal_state=capabilities.get('thermal_state', 'normal'),
            power_source=capabilities.get('power_source', 'unknown')
        )
        
        self.client_profiles[client_info.client_id] = profile
        self.performance_history[client_info.client_id] = []
        
        self.logger.info(f"Registered client {client_info.client_id} with resource score: {profile.get_resource_score():.2f}")
        return profile
    
    def update_client_resources(self, client_id: str, resource_update: Dict[str, Any]) -> bool:
        """Update client resource information"""
        if client_id not in self.client_profiles:
            self.logger.warning(f"Attempted to update unknown client {client_id}")
            return False
        
        profile = self.client_profiles[client_id]
        
        # Update resource values
        if 'memory_available_gb' in resource_update:
            profile.memory_available_gb = resource_update['memory_available_gb']
        
        if 'battery_level' in resource_update:
            profile.battery_level = resource_update['battery_level']
        
        if 'thermal_state' in resource_update:
            profile.thermal_state = resource_update['thermal_state']
        
        if 'cpu_utilization' in resource_update:
            cpu_util = resource_update['cpu_utilization']
            memory_util = resource_update.get('memory_utilization', profile.get_memory_utilization())
            network_util = resource_update.get('network_utilization', 0.0)
            profile.update_utilization(cpu_util, memory_util, network_util)
        
        profile.last_updated = datetime.now()
        
        self.logger.debug(f"Updated resources for client {client_id}")
        return True
    
    def assess_device_capability(self, client_id: str) -> Dict[str, Any]:
        """Assess device capability for training tasks"""
        if client_id not in self.client_profiles:
            return {'error': 'Client not found'}
        
        profile = self.client_profiles[client_id]
        
        # Calculate capability metrics
        resource_score = profile.get_resource_score()
        
        # Estimate training capacity
        estimated_samples_per_hour = profile.cpu_cores * profile.cpu_frequency_ghz * 1000  # Rough estimate
        
        # Memory capacity for different model sizes
        memory_capacity = {
            'small_model_mb': profile.memory_available_gb * 1024 * 0.5,  # 50% of available memory
            'medium_model_mb': profile.memory_available_gb * 1024 * 0.3,  # 30% of available memory
            'large_model_mb': profile.memory_available_gb * 1024 * 0.1   # 10% of available memory
        }
        
        # Network capacity
        network_capacity = {
            'upload_mbps': profile.network_bandwidth_mbps * 0.8,  # 80% for uploads
            'download_mbps': profile.network_bandwidth_mbps,
            'latency_category': self._categorize_latency(profile.communication_latency_ms)
        }
        
        # Battery constraints
        battery_constraints = {}
        if profile.battery_level is not None:
            battery_constraints = {
                'can_train_on_battery': profile.battery_level > 0.3,
                'max_training_minutes_on_battery': profile.battery_level * 100,  # Rough estimate
                'requires_charging': profile.battery_level < 0.2
            }
        
        return {
            'client_id': client_id,
            'resource_score': resource_score,
            'estimated_samples_per_hour': estimated_samples_per_hour,
            'memory_capacity': memory_capacity,
            'network_capacity': network_capacity,
            'battery_constraints': battery_constraints,
            'thermal_state': profile.thermal_state,
            'reliability_score': profile.reliability_score,
            'last_updated': profile.last_updated.isoformat()
        }
    
    def schedule_training_task(self, task: TrainingTask) -> bool:
        """Schedule a training task"""
        # Find suitable client
        suitable_clients = self._find_suitable_clients(task)
        
        if not suitable_clients:
            self.logger.warning(f"No suitable clients found for task {task.task_id}")
            return False
        
        # Select best client based on strategy
        selected_client = self._select_client_for_task(task, suitable_clients)
        
        if selected_client is None:
            self.logger.warning(f"No client selected for task {task.task_id}")
            return False
        
        # Assign task
        task.client_id = selected_client
        task.scheduled_at = datetime.now()
        self.training_tasks[task.task_id] = task
        self.active_training[selected_client] = task.task_id
        
        self.logger.info(f"Scheduled task {task.task_id} on client {selected_client}")
        return True
    
    def _find_suitable_clients(self, task: TrainingTask) -> List[str]:
        """Find clients suitable for a training task"""
        suitable_clients = []
        
        for client_id, profile in self.client_profiles.items():
            # Skip if client is already training
            if client_id in self.active_training:
                continue
            
            # Check if client can handle the task
            if profile.can_handle_training(
                task.model_size_mb, 
                task.batch_size, 
                task.estimated_duration_minutes
            ):
                suitable_clients.append(client_id)
        
        return suitable_clients
    
    def _select_client_for_task(self, task: TrainingTask, candidates: List[str]) -> Optional[str]:
        """Select best client for task based on scheduling strategy"""
        if not candidates:
            return None
        
        if self.scheduling_strategy == SchedulingStrategy.ROUND_ROBIN:
            # Simple round-robin selection
            return candidates[len(self.active_training) % len(candidates)]
        
        elif self.scheduling_strategy == SchedulingStrategy.RESOURCE_AWARE:
            # Select client with highest resource score
            best_client = None
            best_score = -1
            
            for client_id in candidates:
                profile = self.client_profiles[client_id]
                score = profile.get_resource_score()
                
                if score > best_score:
                    best_score = score
                    best_client = client_id
            
            return best_client
        
        elif self.scheduling_strategy == SchedulingStrategy.PRIORITY_BASED:
            # Consider task priority and client capability
            best_client = None
            best_score = -1
            
            for client_id in candidates:
                profile = self.client_profiles[client_id]
                resource_score = profile.get_resource_score()
                
                # Combine task priority with resource score
                combined_score = task.priority * 0.6 + resource_score * 0.4
                
                if combined_score > best_score:
                    best_score = combined_score
                    best_client = client_id
            
            return best_client
        
        elif self.scheduling_strategy == SchedulingStrategy.LOAD_BALANCED:
            # Select client with lowest current load
            best_client = None
            lowest_load = float('inf')
            
            for client_id in candidates:
                profile = self.client_profiles[client_id]
                current_load = profile.get_cpu_utilization() + profile.get_memory_utilization()
                
                if current_load < lowest_load:
                    lowest_load = current_load
                    best_client = client_id
            
            return best_client
        
        elif self.scheduling_strategy == SchedulingStrategy.ENERGY_EFFICIENT:
            # Prefer clients with good battery or plugged in
            plugged_clients = []
            battery_clients = []
            
            for client_id in candidates:
                profile = self.client_profiles[client_id]
                
                if profile.power_source == "plugged":
                    plugged_clients.append(client_id)
                elif profile.battery_level is not None and profile.battery_level > 0.5:
                    battery_clients.append(client_id)
            
            # Prefer plugged clients, then high battery clients
            preferred_clients = plugged_clients if plugged_clients else battery_clients
            if preferred_clients:
                # Among preferred, select by resource score
                return self._select_client_for_task(
                    task, preferred_clients
                ) if len(preferred_clients) > 1 else preferred_clients[0]
            
            # Fallback to resource-aware selection
            return self._select_client_for_task(task, candidates)
        
        else:
            # Default to first available
            return candidates[0]
    
    def complete_training_task(self, task_id: str, performance_metrics: Dict[str, Any]) -> bool:
        """Mark training task as completed and update performance metrics"""
        if task_id not in self.training_tasks:
            return False
        
        task = self.training_tasks[task_id]
        task.completed_at = datetime.now()
        
        # Remove from active training
        if task.client_id in self.active_training:
            del self.active_training[task.client_id]
        
        # Update client performance history
        if task.client_id in self.performance_history:
            performance_record = {
                'task_id': task_id,
                'completed_at': task.completed_at.isoformat(),
                'duration_minutes': (task.completed_at - task.started_at).total_seconds() / 60 if task.started_at else 0,
                'metrics': performance_metrics
            }
            
            self.performance_history[task.client_id].append(performance_record)
            
            # Keep only recent history
            max_history = 50
            if len(self.performance_history[task.client_id]) > max_history:
                self.performance_history[task.client_id] = self.performance_history[task.client_id][-max_history:]
        
        # Update client reliability score
        if task.client_id in self.client_profiles:
            profile = self.client_profiles[task.client_id]
            
            # Simple reliability update based on task completion
            success_rate = performance_metrics.get('success', True)
            if success_rate:
                profile.reliability_score = min(1.0, profile.reliability_score + 0.05)
            else:
                profile.reliability_score = max(0.0, profile.reliability_score - 0.1)
        
        self.logger.info(f"Completed training task {task_id} on client {task.client_id}")
        return True
    
    def get_load_balancing_recommendations(self) -> Dict[str, Any]:
        """Get load balancing recommendations"""
        if not self.client_profiles:
            return {'recommendations': []}
        
        # Calculate load distribution
        client_loads = {}
        for client_id, profile in self.client_profiles.items():
            cpu_load = profile.get_cpu_utilization()
            memory_load = profile.get_memory_utilization()
            combined_load = (cpu_load + memory_load) / 2
            client_loads[client_id] = combined_load
        
        # Find overloaded and underloaded clients
        avg_load = sum(client_loads.values()) / len(client_loads)
        overloaded_threshold = avg_load * 1.5
        underloaded_threshold = avg_load * 0.5
        
        overloaded_clients = [
            client_id for client_id, load in client_loads.items()
            if load > overloaded_threshold
        ]
        
        underloaded_clients = [
            client_id for client_id, load in client_loads.items()
            if load < underloaded_threshold
        ]
        
        recommendations = []
        
        # Recommend task migration from overloaded to underloaded clients
        for overloaded_client in overloaded_clients:
            if overloaded_client in self.active_training:
                task_id = self.active_training[overloaded_client]
                
                for underloaded_client in underloaded_clients:
                    if underloaded_client not in self.active_training:
                        recommendations.append({
                            'action': 'migrate_task',
                            'task_id': task_id,
                            'from_client': overloaded_client,
                            'to_client': underloaded_client,
                            'reason': 'load_balancing'
                        })
                        break
        
        # Recommend scaling adjustments
        if len(overloaded_clients) > len(underloaded_clients):
            recommendations.append({
                'action': 'scale_up',
                'reason': 'high_overall_load',
                'overloaded_clients': len(overloaded_clients)
            })
        elif len(underloaded_clients) > len(overloaded_clients) * 2:
            recommendations.append({
                'action': 'scale_down',
                'reason': 'low_overall_load',
                'underloaded_clients': len(underloaded_clients)
            })
        
        return {
            'average_load': avg_load,
            'overloaded_clients': overloaded_clients,
            'underloaded_clients': underloaded_clients,
            'recommendations': recommendations
        }
    
    def _categorize_latency(self, latency_ms: float) -> str:
        """Categorize network latency"""
        if latency_ms < 10:
            return "excellent"
        elif latency_ms < 50:
            return "good"
        elif latency_ms < 100:
            return "fair"
        else:
            return "poor"
    
    async def _resource_monitoring_loop(self):
        """Background task for resource monitoring"""
        while self.running:
            try:
                # Update resource information for all clients
                for client_id in list(self.client_profiles.keys()):
                    await self._update_client_resources(client_id)
                
                await asyncio.sleep(self.resource_update_interval)
                
            except Exception as e:
                self.logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(self.resource_update_interval)
    
    async def _update_client_resources(self, client_id: str):
        """Update resource information for a specific client"""
        # This would typically query the client for current resource usage
        # For now, simulate some resource updates
        
        if client_id not in self.client_profiles:
            return
        
        profile = self.client_profiles[client_id]
        
        # Simulate resource usage fluctuations
        import random
        
        # Simulate CPU utilization
        base_cpu = 0.3 if client_id in self.active_training else 0.1
        cpu_util = base_cpu + random.uniform(-0.1, 0.2)
        cpu_util = max(0.0, min(1.0, cpu_util))
        
        # Simulate memory utilization
        memory_util = profile.get_memory_utilization() + random.uniform(-0.05, 0.05)
        memory_util = max(0.0, min(1.0, memory_util))
        
        # Update utilization
        profile.update_utilization(cpu_util, memory_util, 0.0)
        
        # Simulate battery drain for mobile devices
        if profile.battery_level is not None and profile.power_source == "battery":
            drain_rate = 0.001 if client_id not in self.active_training else 0.005  # Per minute
            profile.battery_level = max(0.0, profile.battery_level - drain_rate)
    
    async def _task_scheduling_loop(self):
        """Background task for training task scheduling"""
        while self.running:
            try:
                # Check for pending tasks that need scheduling
                pending_tasks = [
                    task for task in self.training_tasks.values()
                    if task.scheduled_at is None
                ]
                
                for task in pending_tasks:
                    if len(self.active_training) < self.max_concurrent_training:
                        self.schedule_training_task(task)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Task scheduling error: {e}")
                await asyncio.sleep(10)
    
    async def _performance_analysis_loop(self):
        """Background task for performance analysis"""
        while self.running:
            try:
                # Analyze client performance and update reliability scores
                for client_id, history in self.performance_history.items():
                    if len(history) >= 5:  # Need at least 5 records for analysis
                        self._analyze_client_performance(client_id, history)
                
                await asyncio.sleep(300)  # Analyze every 5 minutes
                
            except Exception as e:
                self.logger.error(f"Performance analysis error: {e}")
                await asyncio.sleep(300)
    
    def _analyze_client_performance(self, client_id: str, history: List[Dict[str, Any]]):
        """Analyze client performance and update profile"""
        if client_id not in self.client_profiles:
            return
        
        profile = self.client_profiles[client_id]
        
        # Calculate average training speed
        recent_history = history[-10:]  # Last 10 tasks
        durations = [record['duration_minutes'] for record in recent_history if record['duration_minutes'] > 0]
        
        if durations:
            avg_duration = sum(durations) / len(durations)
            # Update training speed estimate (samples per second)
            # This is a rough estimate - in practice you'd have more detailed metrics
            profile.training_speed_samples_per_sec = 1000 / (avg_duration * 60)  # Rough estimate
        
        # Calculate success rate
        successes = sum(1 for record in recent_history if record['metrics'].get('success', True))
        success_rate = successes / len(recent_history)
        
        # Update reliability score
        profile.reliability_score = success_rate
    
    async def _load_balancing_loop(self):
        """Background task for load balancing"""
        while self.running:
            try:
                if self.load_balancing_enabled:
                    recommendations = self.get_load_balancing_recommendations()
                    
                    # Log recommendations (in practice, you might act on them)
                    if recommendations['recommendations']:
                        self.logger.info(f"Load balancing recommendations: {len(recommendations['recommendations'])} actions suggested")
                
                await asyncio.sleep(120)  # Check every 2 minutes
                
            except Exception as e:
                self.logger.error(f"Load balancing error: {e}")
                await asyncio.sleep(120)
    
    def get_resource_status(self) -> Dict[str, Any]:
        """Get comprehensive resource management status"""
        total_clients = len(self.client_profiles)
        active_training = len(self.active_training)
        pending_tasks = len([t for t in self.training_tasks.values() if t.scheduled_at is None])
        
        # Calculate resource utilization statistics
        if self.client_profiles:
            resource_scores = [profile.get_resource_score() for profile in self.client_profiles.values()]
            avg_resource_score = sum(resource_scores) / len(resource_scores)
            
            cpu_utilizations = [profile.get_cpu_utilization() for profile in self.client_profiles.values()]
            avg_cpu_utilization = sum(cpu_utilizations) / len(cpu_utilizations)
        else:
            avg_resource_score = 0.0
            avg_cpu_utilization = 0.0
        
        return {
            'total_clients': total_clients,
            'active_training_tasks': active_training,
            'pending_tasks': pending_tasks,
            'max_concurrent_training': self.max_concurrent_training,
            'scheduling_strategy': self.scheduling_strategy.value,
            'load_balancing_enabled': self.load_balancing_enabled,
            'average_resource_score': avg_resource_score,
            'average_cpu_utilization': avg_cpu_utilization,
            'client_profiles': {
                client_id: {
                    'resource_score': profile.get_resource_score(),
                    'cpu_utilization': profile.get_cpu_utilization(),
                    'memory_utilization': profile.get_memory_utilization(),
                    'battery_level': profile.battery_level,
                    'thermal_state': profile.thermal_state,
                    'reliability_score': profile.reliability_score
                }
                for client_id, profile in self.client_profiles.items()
            }
        }