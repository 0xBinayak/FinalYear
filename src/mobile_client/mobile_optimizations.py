"""
Mobile-specific optimizations for federated learning
"""
import asyncio
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import json
import numpy as np

from ..common.federated_data_structures import NetworkConditions, ComputeResources


class BackgroundTrainingState(Enum):
    """Background training states"""
    IDLE = "idle"
    SCHEDULED = "scheduled"
    ACTIVE = "active"
    PAUSED = "paused"
    SUSPENDED = "suspended"
    COMPLETED = "completed"
    FAILED = "failed"


class NetworkHandoffState(Enum):
    """Network handoff states"""
    STABLE = "stable"
    HANDOFF_DETECTED = "handoff_detected"
    HANDOFF_IN_PROGRESS = "handoff_in_progress"
    HANDOFF_COMPLETED = "handoff_completed"
    CONNECTION_LOST = "connection_lost"
    RECONNECTING = "reconnecting"


@dataclass
class BackgroundTrainingConfig:
    """Configuration for background training"""
    enabled: bool = True
    max_background_time_minutes: int = 60
    min_idle_time_seconds: int = 300  # 5 minutes
    battery_threshold: float = 0.3
    thermal_threshold: str = "warm"
    network_quality_threshold: float = 0.7
    pause_on_user_activity: bool = True
    resume_delay_seconds: int = 30
    max_retries: int = 3
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelComplexityProfile:
    """Model complexity profile for adaptive training"""
    profile_name: str
    max_parameters: int
    max_layers: int
    batch_size: int
    learning_rate: float
    epochs: int
    memory_requirement_mb: float
    compute_requirement_flops: int
    training_time_estimate_minutes: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class IncentiveMetrics:
    """Metrics for incentive and reputation system"""
    participation_count: int = 0
    successful_trainings: int = 0
    failed_trainings: int = 0
    total_samples_contributed: int = 0
    total_training_time_minutes: float = 0.0
    average_model_quality: float = 0.0
    network_reliability_score: float = 1.0
    battery_efficiency_score: float = 1.0
    data_quality_score: float = 1.0
    reputation_score: float = 1.0
    
    # Rewards and penalties
    total_rewards_earned: float = 0.0
    total_penalties: float = 0.0
    current_streak: int = 0
    best_streak: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BackgroundTrainingManager:
    """Manager for background training with system integration"""
    
    def __init__(self, config: BackgroundTrainingConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # State management
        self.state = BackgroundTrainingState.IDLE
        self.training_task = None
        self.last_activity_time = datetime.now()
        self.pause_requested = False
        self.suspend_requested = False
        
        # Monitoring
        self.system_monitor_active = False
        self._monitor_thread = None
        
        # Callbacks
        self.training_callback = None
        self.progress_callback = None
        self.completion_callback = None
        
        # Statistics
        self.training_sessions = []
        self.total_background_time = 0.0
    
    def set_training_callback(self, callback: Callable[[Dict[str, Any]], Any]):
        """Set callback for training execution"""
        self.training_callback = callback
    
    def set_progress_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set callback for training progress updates"""
        self.progress_callback = callback
    
    def set_completion_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set callback for training completion"""
        self.completion_callback = callback
    
    def start_system_monitoring(self):
        """Start system monitoring for background training opportunities"""
        if self.system_monitor_active:
            return
        
        self.system_monitor_active = True
        self._monitor_thread = threading.Thread(target=self._system_monitor_worker)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        
        self.logger.info("Background training system monitoring started")
    
    def stop_system_monitoring(self):
        """Stop system monitoring"""
        self.system_monitor_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        
        self.logger.info("Background training system monitoring stopped")
    
    def _system_monitor_worker(self):
        """Worker thread for system monitoring"""
        while self.system_monitor_active:
            try:
                # Check if background training should be triggered
                if self._should_start_background_training():
                    self._schedule_background_training()
                
                # Check if active training should be paused/resumed
                if self.state == BackgroundTrainingState.ACTIVE:
                    if self._should_pause_training():
                        self.pause_training()
                elif self.state == BackgroundTrainingState.PAUSED:
                    if self._should_resume_training():
                        self.resume_training()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"System monitoring error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _should_start_background_training(self) -> bool:
        """Check if background training should start"""
        if not self.config.enabled:
            return False
        
        if self.state != BackgroundTrainingState.IDLE:
            return False
        
        # Check idle time
        idle_time = (datetime.now() - self.last_activity_time).total_seconds()
        if idle_time < self.config.min_idle_time_seconds:
            return False
        
        # Check system conditions (would need actual system integration)
        # For now, simulate conditions
        battery_ok = True  # Would check actual battery level
        thermal_ok = True  # Would check thermal state
        network_ok = True  # Would check network quality
        
        return battery_ok and thermal_ok and network_ok
    
    def _should_pause_training(self) -> bool:
        """Check if training should be paused"""
        if self.pause_requested or self.suspend_requested:
            return True
        
        # Check for user activity (would need actual system integration)
        if self.config.pause_on_user_activity:
            # Simulate user activity detection
            return False  # Would check actual user activity
        
        # Check system conditions
        return False  # Would check battery, thermal, network conditions
    
    def _should_resume_training(self) -> bool:
        """Check if training should be resumed"""
        if self.suspend_requested:
            return False
        
        if self.pause_requested:
            return False
        
        # Check if enough time has passed since pause
        # Would implement actual resume logic
        return True
    
    def _schedule_background_training(self):
        """Schedule background training"""
        if self.state != BackgroundTrainingState.IDLE:
            return
        
        self.state = BackgroundTrainingState.SCHEDULED
        self.logger.info("Background training scheduled")
        
        # Start training in background thread
        self.training_task = threading.Thread(target=self._background_training_worker)
        self.training_task.daemon = True
        self.training_task.start()
    
    def _background_training_worker(self):
        """Worker thread for background training"""
        try:
            self.state = BackgroundTrainingState.ACTIVE
            start_time = datetime.now()
            
            self.logger.info("Background training started")
            
            # Execute training if callback is set
            if self.training_callback:
                training_params = {
                    'background_mode': True,
                    'max_time_minutes': self.config.max_background_time_minutes,
                    'priority': 'low'
                }
                
                result = self.training_callback(training_params)
                
                # Record session
                session = {
                    'start_time': start_time,
                    'end_time': datetime.now(),
                    'duration_minutes': (datetime.now() - start_time).total_seconds() / 60,
                    'result': result,
                    'state_changes': []
                }
                
                self.training_sessions.append(session)
                self.total_background_time += session['duration_minutes']
                
                if self.completion_callback:
                    self.completion_callback(session)
                
                self.state = BackgroundTrainingState.COMPLETED
                self.logger.info(f"Background training completed in {session['duration_minutes']:.1f} minutes")
            
            else:
                self.logger.warning("No training callback set for background training")
                self.state = BackgroundTrainingState.FAILED
            
        except Exception as e:
            self.logger.error(f"Background training error: {e}")
            self.state = BackgroundTrainingState.FAILED
        
        finally:
            # Reset to idle after delay
            time.sleep(self.config.resume_delay_seconds)
            if self.state in [BackgroundTrainingState.COMPLETED, BackgroundTrainingState.FAILED]:
                self.state = BackgroundTrainingState.IDLE
    
    def pause_training(self):
        """Pause active background training"""
        if self.state == BackgroundTrainingState.ACTIVE:
            self.pause_requested = True
            self.state = BackgroundTrainingState.PAUSED
            self.logger.info("Background training paused")
    
    def resume_training(self):
        """Resume paused background training"""
        if self.state == BackgroundTrainingState.PAUSED:
            self.pause_requested = False
            self.state = BackgroundTrainingState.ACTIVE
            self.logger.info("Background training resumed")
    
    def suspend_training(self):
        """Suspend background training (stronger than pause)"""
        self.suspend_requested = True
        if self.state in [BackgroundTrainingState.ACTIVE, BackgroundTrainingState.PAUSED]:
            self.state = BackgroundTrainingState.SUSPENDED
            self.logger.info("Background training suspended")
    
    def resume_from_suspension(self):
        """Resume from suspension"""
        self.suspend_requested = False
        if self.state == BackgroundTrainingState.SUSPENDED:
            self.state = BackgroundTrainingState.IDLE
            self.logger.info("Background training resumed from suspension")
    
    def notify_user_activity(self):
        """Notify manager of user activity"""
        self.last_activity_time = datetime.now()
        
        if self.config.pause_on_user_activity and self.state == BackgroundTrainingState.ACTIVE:
            self.pause_training()
    
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get background training statistics"""
        return {
            'total_sessions': len(self.training_sessions),
            'total_background_time_minutes': self.total_background_time,
            'average_session_time_minutes': self.total_background_time / max(1, len(self.training_sessions)),
            'current_state': self.state.value,
            'last_activity_time': self.last_activity_time.isoformat(),
            'config': self.config.to_dict()
        }


class NetworkHandoffManager:
    """Manager for network handoff handling in mobile scenarios"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # State management
        self.current_state = NetworkHandoffState.STABLE
        self.previous_network = None
        self.current_network = None
        self.handoff_start_time = None
        
        # Network history
        self.network_history = []
        self.handoff_events = []
        
        # Callbacks
        self.handoff_callback = None
        self.reconnection_callback = None
        
        # Monitoring
        self.monitoring_active = False
        self._monitor_thread = None
    
    def set_handoff_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set callback for handoff events"""
        self.handoff_callback = callback
    
    def set_reconnection_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Set callback for reconnection events"""
        self.reconnection_callback = callback
    
    def start_monitoring(self):
        """Start network monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self._monitor_thread = threading.Thread(target=self._network_monitor_worker)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
        
        self.logger.info("Network handoff monitoring started")
    
    def stop_monitoring(self):
        """Stop network monitoring"""
        self.monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        
        self.logger.info("Network handoff monitoring stopped")
    
    def _network_monitor_worker(self):
        """Worker thread for network monitoring"""
        while self.monitoring_active:
            try:
                # Get current network conditions
                current_network = self._get_current_network_info()
                
                # Check for network changes
                if self._detect_network_change(current_network):
                    self._handle_network_change(current_network)
                
                # Update network history
                self.network_history.append({
                    'timestamp': datetime.now(),
                    'network_info': current_network,
                    'state': self.current_state.value
                })
                
                # Keep only last hour of history
                cutoff_time = datetime.now() - timedelta(hours=1)
                self.network_history = [
                    entry for entry in self.network_history
                    if entry['timestamp'] > cutoff_time
                ]
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Network monitoring error: {e}")
                time.sleep(30)
    
    def _get_current_network_info(self) -> Dict[str, Any]:
        """Get current network information"""
        # In a real implementation, this would query actual network interfaces
        # For now, simulate network detection
        
        # Simulate different network types
        network_types = ['wifi', 'cellular_4g', 'cellular_5g', 'ethernet']
        current_type = np.random.choice(network_types, p=[0.6, 0.2, 0.15, 0.05])
        
        # Simulate network quality based on type
        if current_type == 'wifi':
            bandwidth = 20 + np.random.random() * 80  # 20-100 Mbps
            latency = 10 + np.random.random() * 40    # 10-50 ms
            stability = 0.8 + np.random.random() * 0.2
        elif current_type == 'cellular_5g':
            bandwidth = 50 + np.random.random() * 150  # 50-200 Mbps
            latency = 5 + np.random.random() * 20      # 5-25 ms
            stability = 0.7 + np.random.random() * 0.3
        elif current_type == 'cellular_4g':
            bandwidth = 5 + np.random.random() * 45    # 5-50 Mbps
            latency = 20 + np.random.random() * 80     # 20-100 ms
            stability = 0.6 + np.random.random() * 0.3
        else:  # ethernet
            bandwidth = 100 + np.random.random() * 900  # 100-1000 Mbps
            latency = 1 + np.random.random() * 9        # 1-10 ms
            stability = 0.95 + np.random.random() * 0.05
        
        return {
            'type': current_type,
            'bandwidth_mbps': bandwidth,
            'latency_ms': latency,
            'stability': stability,
            'is_metered': current_type.startswith('cellular'),
            'signal_strength': 0.5 + np.random.random() * 0.5,
            'interface_id': f"{current_type}_interface_1"
        }
    
    def _detect_network_change(self, current_network: Dict[str, Any]) -> bool:
        """Detect if network has changed"""
        if self.current_network is None:
            return True
        
        # Check for interface change
        if current_network['interface_id'] != self.current_network['interface_id']:
            return True
        
        # Check for significant quality change
        bandwidth_change = abs(current_network['bandwidth_mbps'] - self.current_network['bandwidth_mbps'])
        if bandwidth_change > 20:  # 20 Mbps threshold
            return True
        
        latency_change = abs(current_network['latency_ms'] - self.current_network['latency_ms'])
        if latency_change > 50:  # 50 ms threshold
            return True
        
        return False
    
    def _handle_network_change(self, new_network: Dict[str, Any]):
        """Handle detected network change"""
        self.logger.info(f"Network change detected: {self.current_network['type'] if self.current_network else 'None'} -> {new_network['type']}")
        
        # Update state
        self.previous_network = self.current_network
        self.current_network = new_network
        
        # Determine handoff type
        if self.previous_network is None:
            # Initial connection
            self.current_state = NetworkHandoffState.STABLE
        elif new_network['interface_id'] != self.previous_network['interface_id']:
            # Interface change - handoff
            self.current_state = NetworkHandoffState.HANDOFF_DETECTED
            self.handoff_start_time = datetime.now()
            self._process_handoff()
        else:
            # Quality change on same interface
            self.current_state = NetworkHandoffState.STABLE
    
    def _process_handoff(self):
        """Process network handoff"""
        self.current_state = NetworkHandoffState.HANDOFF_IN_PROGRESS
        
        # Record handoff event
        handoff_event = {
            'timestamp': datetime.now(),
            'from_network': self.previous_network,
            'to_network': self.current_network,
            'handoff_duration_ms': 0,
            'success': False
        }
        
        try:
            # Simulate handoff processing time
            time.sleep(0.5)  # 500ms handoff time
            
            # Check if handoff was successful
            if self.current_network['stability'] > 0.5:
                self.current_state = NetworkHandoffState.HANDOFF_COMPLETED
                handoff_event['success'] = True
                self.logger.info("Network handoff completed successfully")
            else:
                self.current_state = NetworkHandoffState.CONNECTION_LOST
                self.logger.warning("Network handoff failed - connection lost")
            
            handoff_event['handoff_duration_ms'] = (datetime.now() - self.handoff_start_time).total_seconds() * 1000
            
            # Notify callback
            if self.handoff_callback:
                self.handoff_callback(handoff_event)
            
        except Exception as e:
            self.logger.error(f"Handoff processing error: {e}")
            self.current_state = NetworkHandoffState.CONNECTION_LOST
            handoff_event['error'] = str(e)
        
        finally:
            self.handoff_events.append(handoff_event)
            
            # Transition to stable or attempt reconnection
            if self.current_state == NetworkHandoffState.HANDOFF_COMPLETED:
                self.current_state = NetworkHandoffState.STABLE
            elif self.current_state == NetworkHandoffState.CONNECTION_LOST:
                self._attempt_reconnection()
    
    def _attempt_reconnection(self):
        """Attempt to reconnect after connection loss"""
        self.current_state = NetworkHandoffState.RECONNECTING
        self.logger.info("Attempting network reconnection")
        
        # Simulate reconnection attempts
        for attempt in range(3):
            time.sleep(1.0)  # Wait between attempts
            
            # Simulate reconnection success/failure
            if np.random.random() > 0.3:  # 70% success rate
                self.current_state = NetworkHandoffState.STABLE
                self.logger.info(f"Reconnection successful after {attempt + 1} attempts")
                
                if self.reconnection_callback:
                    self.reconnection_callback({
                        'success': True,
                        'attempts': attempt + 1,
                        'network': self.current_network
                    })
                return
        
        # All attempts failed
        self.logger.error("Reconnection failed after 3 attempts")
        if self.reconnection_callback:
            self.reconnection_callback({
                'success': False,
                'attempts': 3,
                'network': self.current_network
            })
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get current network status"""
        return {
            'current_state': self.current_state.value,
            'current_network': self.current_network,
            'previous_network': self.previous_network,
            'handoff_events_count': len(self.handoff_events),
            'network_history_count': len(self.network_history),
            'is_stable': self.current_state == NetworkHandoffState.STABLE
        }


class AdaptiveModelComplexityManager:
    """Manager for adaptive model complexity based on device capabilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Predefined complexity profiles
        self.complexity_profiles = {
            'minimal': ModelComplexityProfile(
                profile_name='minimal',
                max_parameters=10000,
                max_layers=3,
                batch_size=8,
                learning_rate=0.01,
                epochs=3,
                memory_requirement_mb=50,
                compute_requirement_flops=1e6,
                training_time_estimate_minutes=2
            ),
            'low': ModelComplexityProfile(
                profile_name='low',
                max_parameters=50000,
                max_layers=5,
                batch_size=16,
                learning_rate=0.005,
                epochs=5,
                memory_requirement_mb=100,
                compute_requirement_flops=5e6,
                training_time_estimate_minutes=5
            ),
            'medium': ModelComplexityProfile(
                profile_name='medium',
                max_parameters=200000,
                max_layers=8,
                batch_size=32,
                learning_rate=0.001,
                epochs=8,
                memory_requirement_mb=300,
                compute_requirement_flops=2e7,
                training_time_estimate_minutes=15
            ),
            'high': ModelComplexityProfile(
                profile_name='high',
                max_parameters=1000000,
                max_layers=12,
                batch_size=64,
                learning_rate=0.0005,
                epochs=12,
                memory_requirement_mb=800,
                compute_requirement_flops=1e8,
                training_time_estimate_minutes=45
            ),
            'maximum': ModelComplexityProfile(
                profile_name='maximum',
                max_parameters=5000000,
                max_layers=20,
                batch_size=128,
                learning_rate=0.0001,
                epochs=20,
                memory_requirement_mb=2000,
                compute_requirement_flops=5e8,
                training_time_estimate_minutes=120
            )
        }
        
        # Current selection
        self.current_profile = None
        self.adaptation_history = []
    
    def select_optimal_profile(self, compute_resources: ComputeResources,
                             network_conditions: NetworkConditions,
                             constraints: Dict[str, Any] = None) -> ModelComplexityProfile:
        """Select optimal complexity profile based on device capabilities"""
        
        constraints = constraints or {}
        
        # Score each profile based on device capabilities
        profile_scores = {}
        
        for profile_name, profile in self.complexity_profiles.items():
            score = self._score_profile(profile, compute_resources, network_conditions, constraints)
            profile_scores[profile_name] = score
        
        # Select profile with highest score
        best_profile_name = max(profile_scores, key=profile_scores.get)
        best_profile = self.complexity_profiles[best_profile_name]
        
        # Record adaptation decision
        adaptation_record = {
            'timestamp': datetime.now(),
            'selected_profile': best_profile_name,
            'profile_scores': profile_scores,
            'compute_resources': compute_resources.to_dict(),
            'network_conditions': network_conditions.to_dict(),
            'constraints': constraints
        }
        
        self.adaptation_history.append(adaptation_record)
        self.current_profile = best_profile
        
        self.logger.info(f"Selected complexity profile: {best_profile_name} (score: {profile_scores[best_profile_name]:.2f})")
        
        return best_profile
    
    def _score_profile(self, profile: ModelComplexityProfile,
                      compute_resources: ComputeResources,
                      network_conditions: NetworkConditions,
                      constraints: Dict[str, Any]) -> float:
        """Score a complexity profile based on device capabilities"""
        
        # Start with base score that favors higher complexity when resources allow
        base_complexity_score = min(1.0, profile.max_parameters / 1000000)  # Normalize to 0-1
        score = 0.5 + 0.5 * base_complexity_score  # Base score 0.5-1.0
        
        # Memory constraint
        available_memory_mb = compute_resources.available_memory_gb * 1024
        if profile.memory_requirement_mb > available_memory_mb:
            score *= 0.1  # Heavy penalty for insufficient memory
        elif profile.memory_requirement_mb > available_memory_mb * 0.8:
            score *= 0.5  # Moderate penalty for tight memory
        else:
            # Bonus for efficient memory usage when plenty available
            memory_efficiency = 1.0 - (profile.memory_requirement_mb / available_memory_mb)
            score *= (1.0 + 0.2 * memory_efficiency)
        
        # Battery constraint
        if compute_resources.battery_level is not None:
            if compute_resources.battery_level < 0.3:
                # Low battery - prefer simpler models
                complexity_penalty = (profile.max_parameters / 1000000) * 0.5
                score *= max(0.1, 1.0 - complexity_penalty)
            elif compute_resources.battery_level < 0.5:
                # Medium battery - moderate preference for simpler models
                complexity_penalty = (profile.max_parameters / 1000000) * 0.2
                score *= max(0.3, 1.0 - complexity_penalty)
        
        # Thermal constraint
        if compute_resources.thermal_state in ['hot', 'critical']:
            score *= 0.2  # Heavy penalty for thermal issues
        elif compute_resources.thermal_state == 'warm':
            score *= 0.6  # Moderate penalty for warm state
        
        # Network constraint
        if network_conditions.is_metered:
            # Prefer smaller models on metered connections
            size_penalty = (profile.max_parameters / 1000000) * 0.3
            score *= max(0.2, 1.0 - size_penalty)
        
        if network_conditions.bandwidth_mbps < 5.0:
            # Low bandwidth - prefer smaller models
            size_penalty = (profile.max_parameters / 1000000) * 0.4
            score *= max(0.1, 1.0 - size_penalty)
        
        # Time constraints
        max_time_minutes = constraints.get('max_training_time_minutes', 60)
        if profile.training_time_estimate_minutes > max_time_minutes:
            time_penalty = (profile.training_time_estimate_minutes - max_time_minutes) / max_time_minutes
            score *= max(0.1, 1.0 - time_penalty)
        
        # CPU constraint
        if compute_resources.cpu_cores < 4:
            # Fewer cores - prefer simpler models
            complexity_penalty = (profile.max_layers / 20) * 0.3
            score *= max(0.3, 1.0 - complexity_penalty)
        else:
            # More cores - can handle more complex models
            cpu_bonus = min(0.3, (compute_resources.cpu_cores - 4) * 0.05)
            score *= (1.0 + cpu_bonus)
        
        # Power source constraint
        if compute_resources.power_source == 'battery':
            # On battery - slight preference for efficiency
            efficiency_bonus = 1.0 - (profile.compute_requirement_flops / 1e9) * 0.1
            score *= max(0.5, efficiency_bonus)
        
        return score
    
    def get_current_profile(self) -> Optional[ModelComplexityProfile]:
        """Get currently selected profile"""
        return self.current_profile
    
    def get_adaptation_history(self) -> List[Dict[str, Any]]:
        """Get adaptation history"""
        return self.adaptation_history.copy()
    
    def add_custom_profile(self, profile: ModelComplexityProfile):
        """Add custom complexity profile"""
        self.complexity_profiles[profile.profile_name] = profile
        self.logger.info(f"Added custom complexity profile: {profile.profile_name}")


class IncentiveReputationManager:
    """Manager for incentive mechanisms and reputation tracking"""
    
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.logger = logging.getLogger(__name__)
        
        # Metrics tracking
        self.metrics = IncentiveMetrics()
        
        # Reward system parameters
        self.reward_rates = {
            'participation_base': 1.0,
            'quality_bonus': 0.5,
            'consistency_bonus': 0.3,
            'efficiency_bonus': 0.2,
            'data_quality_bonus': 0.4
        }
        
        self.penalty_rates = {
            'training_failure': -0.5,
            'poor_quality': -0.3,
            'unreliability': -0.2,
            'resource_waste': -0.1
        }
        
        # Reputation calculation weights
        self.reputation_weights = {
            'success_rate': 0.3,
            'quality_score': 0.25,
            'reliability_score': 0.2,
            'efficiency_score': 0.15,
            'consistency_score': 0.1
        }
        
        # History tracking
        self.participation_history = []
        self.reward_history = []
    
    def record_training_participation(self, training_result: Dict[str, Any]):
        """Record training participation and calculate rewards"""
        
        participation_record = {
            'timestamp': datetime.now(),
            'training_result': training_result,
            'rewards_earned': 0.0,
            'penalties_applied': 0.0
        }
        
        # Update basic metrics
        self.metrics.participation_count += 1
        
        if training_result.get('success', False):
            self.metrics.successful_trainings += 1
            self.metrics.current_streak += 1
            self.metrics.best_streak = max(self.metrics.best_streak, self.metrics.current_streak)
        else:
            self.metrics.failed_trainings += 1
            self.metrics.current_streak = 0
        
        # Update contribution metrics
        samples_used = training_result.get('samples_used', 0)
        training_time = training_result.get('training_time_minutes', 0)
        
        self.metrics.total_samples_contributed += samples_used
        self.metrics.total_training_time_minutes += training_time
        
        # Calculate rewards and penalties
        rewards = self._calculate_rewards(training_result)
        penalties = self._calculate_penalties(training_result)
        
        participation_record['rewards_earned'] = rewards
        participation_record['penalties_applied'] = penalties
        
        self.metrics.total_rewards_earned += rewards
        self.metrics.total_penalties += penalties
        
        # Update quality scores
        self._update_quality_scores(training_result)
        
        # Recalculate reputation
        self._update_reputation_score()
        
        # Store records
        self.participation_history.append(participation_record)
        self.reward_history.append({
            'timestamp': datetime.now(),
            'type': 'training_participation',
            'rewards': rewards,
            'penalties': penalties,
            'net_change': rewards - penalties,
            'new_reputation': self.metrics.reputation_score
        })
        
        self.logger.info(f"Training participation recorded: +{rewards:.2f} rewards, -{penalties:.2f} penalties")
        
        return participation_record
    
    def _calculate_rewards(self, training_result: Dict[str, Any]) -> float:
        """Calculate rewards for training participation"""
        total_rewards = 0.0
        
        # Base participation reward
        if training_result.get('success', False):
            total_rewards += self.reward_rates['participation_base']
        
        # Quality bonus
        model_quality = training_result.get('model_quality_score', 0.5)
        if model_quality > 0.7:
            quality_bonus = (model_quality - 0.7) * self.reward_rates['quality_bonus']
            total_rewards += quality_bonus
        
        # Consistency bonus (based on streak)
        if self.metrics.current_streak >= 5:
            consistency_bonus = min(self.metrics.current_streak / 10, 1.0) * self.reward_rates['consistency_bonus']
            total_rewards += consistency_bonus
        
        # Efficiency bonus (based on resource usage)
        efficiency_score = training_result.get('efficiency_score', 0.5)
        if efficiency_score > 0.6:
            efficiency_bonus = (efficiency_score - 0.6) * self.reward_rates['efficiency_bonus']
            total_rewards += efficiency_bonus
        
        # Data quality bonus
        data_quality = training_result.get('data_quality_score', 0.5)
        if data_quality > 0.8:
            data_bonus = (data_quality - 0.8) * self.reward_rates['data_quality_bonus']
            total_rewards += data_bonus
        
        return total_rewards
    
    def _calculate_penalties(self, training_result: Dict[str, Any]) -> float:
        """Calculate penalties for training issues"""
        total_penalties = 0.0
        
        # Training failure penalty
        if not training_result.get('success', False):
            total_penalties += abs(self.penalty_rates['training_failure'])
        
        # Poor quality penalty
        model_quality = training_result.get('model_quality_score', 0.5)
        if model_quality < 0.3:
            quality_penalty = (0.3 - model_quality) * abs(self.penalty_rates['poor_quality'])
            total_penalties += quality_penalty
        
        # Unreliability penalty (based on network issues)
        if training_result.get('network_issues', 0) > 3:
            reliability_penalty = abs(self.penalty_rates['unreliability'])
            total_penalties += reliability_penalty
        
        # Resource waste penalty
        efficiency_score = training_result.get('efficiency_score', 0.5)
        if efficiency_score < 0.3:
            waste_penalty = (0.3 - efficiency_score) * abs(self.penalty_rates['resource_waste'])
            total_penalties += waste_penalty
        
        return total_penalties
    
    def _update_quality_scores(self, training_result: Dict[str, Any]):
        """Update quality scores based on training result"""
        
        # Update average model quality
        model_quality = training_result.get('model_quality_score', 0.5)
        current_avg = self.metrics.average_model_quality
        participation_count = self.metrics.participation_count
        
        # Exponential moving average
        alpha = 0.1
        self.metrics.average_model_quality = alpha * model_quality + (1 - alpha) * current_avg
        
        # Update network reliability score
        network_issues = training_result.get('network_issues', 0)
        if network_issues == 0:
            self.metrics.network_reliability_score = min(1.0, self.metrics.network_reliability_score + 0.05)
        else:
            self.metrics.network_reliability_score = max(0.0, self.metrics.network_reliability_score - 0.1)
        
        # Update battery efficiency score
        battery_usage = training_result.get('battery_usage_percent', 10)
        if battery_usage < 5:
            self.metrics.battery_efficiency_score = min(1.0, self.metrics.battery_efficiency_score + 0.03)
        elif battery_usage > 20:
            self.metrics.battery_efficiency_score = max(0.0, self.metrics.battery_efficiency_score - 0.05)
        
        # Update data quality score
        data_quality = training_result.get('data_quality_score', 0.5)
        self.metrics.data_quality_score = alpha * data_quality + (1 - alpha) * self.metrics.data_quality_score
    
    def _update_reputation_score(self):
        """Update overall reputation score"""
        
        # Calculate component scores
        success_rate = self.metrics.successful_trainings / max(1, self.metrics.participation_count)
        quality_score = self.metrics.average_model_quality
        reliability_score = self.metrics.network_reliability_score
        efficiency_score = self.metrics.battery_efficiency_score
        consistency_score = min(1.0, self.metrics.current_streak / 10)
        
        # Weighted combination
        reputation = (
            self.reputation_weights['success_rate'] * success_rate +
            self.reputation_weights['quality_score'] * quality_score +
            self.reputation_weights['reliability_score'] * reliability_score +
            self.reputation_weights['efficiency_score'] * efficiency_score +
            self.reputation_weights['consistency_score'] * consistency_score
        )
        
        self.metrics.reputation_score = max(0.0, min(1.0, reputation))
    
    def get_current_metrics(self) -> IncentiveMetrics:
        """Get current incentive metrics"""
        return self.metrics
    
    def get_reputation_breakdown(self) -> Dict[str, float]:
        """Get detailed reputation score breakdown"""
        success_rate = self.metrics.successful_trainings / max(1, self.metrics.participation_count)
        
        return {
            'overall_reputation': self.metrics.reputation_score,
            'success_rate': success_rate,
            'quality_score': self.metrics.average_model_quality,
            'reliability_score': self.metrics.network_reliability_score,
            'efficiency_score': self.metrics.battery_efficiency_score,
            'consistency_score': min(1.0, self.metrics.current_streak / 10),
            'current_streak': self.metrics.current_streak,
            'best_streak': self.metrics.best_streak
        }
    
    def get_reward_summary(self) -> Dict[str, Any]:
        """Get reward and penalty summary"""
        return {
            'total_rewards': self.metrics.total_rewards_earned,
            'total_penalties': self.metrics.total_penalties,
            'net_rewards': self.metrics.total_rewards_earned - self.metrics.total_penalties,
            'average_reward_per_participation': self.metrics.total_rewards_earned / max(1, self.metrics.participation_count),
            'recent_rewards': [r for r in self.reward_history[-10:]],  # Last 10 rewards
            'participation_count': self.metrics.participation_count
        }
    
    def export_metrics(self) -> Dict[str, Any]:
        """Export all metrics for external systems"""
        return {
            'client_id': self.client_id,
            'metrics': self.metrics.to_dict(),
            'reputation_breakdown': self.get_reputation_breakdown(),
            'reward_summary': self.get_reward_summary(),
            'participation_history_count': len(self.participation_history),
            'last_updated': datetime.now().isoformat()
        }