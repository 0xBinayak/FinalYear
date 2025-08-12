"""
Error Handling and Recovery for SDR Operations

Provides robust error handling, recovery mechanisms, and optimization
for SDR hardware operations.
"""
import logging
import time
import threading
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorType(Enum):
    """Types of SDR errors"""
    HARDWARE_FAILURE = "hardware_failure"
    COMMUNICATION_ERROR = "communication_error"
    CONFIGURATION_ERROR = "configuration_error"
    BUFFER_OVERFLOW = "buffer_overflow"
    SIGNAL_QUALITY = "signal_quality"
    RESOURCE_EXHAUSTION = "resource_exhaustion"
    THERMAL_ISSUE = "thermal_issue"


@dataclass
class SDRError:
    """SDR error information"""
    error_type: ErrorType
    severity: ErrorSeverity
    message: str
    timestamp: datetime
    device_id: str
    context: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_successful: bool = False


@dataclass
class RecoveryAction:
    """Recovery action definition"""
    name: str
    action: Callable[[str, Dict[str, Any]], bool]  # device_id, context -> success
    applicable_errors: List[ErrorType]
    max_attempts: int = 3
    delay_between_attempts: float = 1.0


class SDRErrorHandler:
    """Handles SDR errors and implements recovery strategies"""
    
    def __init__(self):
        self.error_history: List[SDRError] = []
        self.recovery_actions: List[RecoveryAction] = []
        self.error_callbacks: List[Callable[[SDRError], None]] = []
        self.max_history_size = 1000
        self.lock = threading.Lock()
        
        # Initialize default recovery actions
        self._setup_default_recovery_actions()
    
    def register_error_callback(self, callback: Callable[[SDRError], None]):
        """Register callback for error notifications"""
        self.error_callbacks.append(callback)
    
    def register_recovery_action(self, action: RecoveryAction):
        """Register custom recovery action"""
        self.recovery_actions.append(action)
    
    def handle_error(self, error: SDRError) -> bool:
        """Handle an SDR error and attempt recovery"""
        with self.lock:
            # Add to history
            self.error_history.append(error)
            
            # Trim history if needed
            if len(self.error_history) > self.max_history_size:
                self.error_history = self.error_history[-self.max_history_size:]
            
            # Notify callbacks
            for callback in self.error_callbacks:
                try:
                    callback(error)
                except Exception as e:
                    logger.error(f"Error in error callback: {e}")
            
            # Attempt recovery for high/critical errors
            if error.severity in [ErrorSeverity.HIGH, ErrorSeverity.CRITICAL]:
                return self._attempt_recovery(error)
            
            return True
    
    def get_error_history(self, device_id: Optional[str] = None, 
                         hours: Optional[int] = None) -> List[SDRError]:
        """Get error history with optional filtering"""
        with self.lock:
            errors = self.error_history.copy()
        
        # Filter by device
        if device_id:
            errors = [e for e in errors if e.device_id == device_id]
        
        # Filter by time
        if hours:
            cutoff = datetime.now() - timedelta(hours=hours)
            errors = [e for e in errors if e.timestamp >= cutoff]
        
        return errors
    
    def get_error_stats(self, device_id: Optional[str] = None) -> Dict[str, Any]:
        """Get error statistics"""
        errors = self.get_error_history(device_id)
        
        if not errors:
            return {
                'total_errors': 0,
                'error_rate': 0.0,
                'most_common_error': None,
                'recovery_success_rate': 0.0
            }
        
        # Count by type
        error_counts = {}
        recovery_attempts = 0
        recovery_successes = 0
        
        for error in errors:
            error_type = error.error_type.value
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
            
            if error.recovery_attempted:
                recovery_attempts += 1
                if error.recovery_successful:
                    recovery_successes += 1
        
        most_common = max(error_counts.items(), key=lambda x: x[1]) if error_counts else None
        
        # Calculate error rate (errors per hour)
        if len(errors) > 1:
            time_span = (errors[-1].timestamp - errors[0].timestamp).total_seconds() / 3600
            error_rate = len(errors) / max(time_span, 1.0)
        else:
            error_rate = 0.0
        
        return {
            'total_errors': len(errors),
            'error_rate': error_rate,
            'most_common_error': most_common[0] if most_common else None,
            'error_counts': error_counts,
            'recovery_success_rate': recovery_successes / max(recovery_attempts, 1)
        }
    
    def _attempt_recovery(self, error: SDRError) -> bool:
        """Attempt to recover from an error"""
        applicable_actions = [
            action for action in self.recovery_actions
            if error.error_type in action.applicable_errors
        ]
        
        if not applicable_actions:
            logger.warning(f"No recovery actions available for {error.error_type}")
            return False
        
        error.recovery_attempted = True
        
        for action in applicable_actions:
            logger.info(f"Attempting recovery action: {action.name}")
            
            for attempt in range(action.max_attempts):
                try:
                    if action.action(error.device_id, error.context):
                        logger.info(f"Recovery successful: {action.name}")
                        error.recovery_successful = True
                        return True
                    
                    if attempt < action.max_attempts - 1:
                        time.sleep(action.delay_between_attempts)
                        
                except Exception as e:
                    logger.error(f"Recovery action {action.name} failed: {e}")
            
            logger.warning(f"Recovery action {action.name} failed after {action.max_attempts} attempts")
        
        logger.error(f"All recovery attempts failed for error: {error.message}")
        return False
    
    def _setup_default_recovery_actions(self):
        """Setup default recovery actions"""
        
        def restart_device(device_id: str, context: Dict[str, Any]) -> bool:
            """Restart SDR device"""
            try:
                from .device_manager import SDRDeviceManager
                
                # This would need access to the device manager
                # In practice, this would be injected or passed as context
                logger.info(f"Restarting device {device_id}")
                
                # Simulate restart logic
                time.sleep(2.0)
                return True
                
            except Exception as e:
                logger.error(f"Failed to restart device {device_id}: {e}")
                return False
        
        def reset_configuration(device_id: str, context: Dict[str, Any]) -> bool:
            """Reset device configuration to safe defaults"""
            try:
                logger.info(f"Resetting configuration for device {device_id}")
                
                # Reset to safe defaults
                # This would interact with the device manager
                return True
                
            except Exception as e:
                logger.error(f"Failed to reset configuration for {device_id}: {e}")
                return False
        
        def reduce_sample_rate(device_id: str, context: Dict[str, Any]) -> bool:
            """Reduce sample rate to handle buffer overflow"""
            try:
                current_rate = context.get('sample_rate', 2e6)
                new_rate = current_rate * 0.5
                
                logger.info(f"Reducing sample rate from {current_rate} to {new_rate}")
                
                # This would interact with the device manager
                return True
                
            except Exception as e:
                logger.error(f"Failed to reduce sample rate for {device_id}: {e}")
                return False
        
        def increase_buffer_size(device_id: str, context: Dict[str, Any]) -> bool:
            """Increase buffer size to handle overflow"""
            try:
                current_size = context.get('buffer_size', 8192)
                new_size = min(current_size * 2, 65536)
                
                logger.info(f"Increasing buffer size from {current_size} to {new_size}")
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to increase buffer size for {device_id}: {e}")
                return False
        
        # Register default recovery actions
        self.recovery_actions.extend([
            RecoveryAction(
                name="restart_device",
                action=restart_device,
                applicable_errors=[
                    ErrorType.HARDWARE_FAILURE,
                    ErrorType.COMMUNICATION_ERROR
                ],
                max_attempts=2,
                delay_between_attempts=5.0
            ),
            RecoveryAction(
                name="reset_configuration",
                action=reset_configuration,
                applicable_errors=[
                    ErrorType.CONFIGURATION_ERROR,
                    ErrorType.SIGNAL_QUALITY
                ],
                max_attempts=1
            ),
            RecoveryAction(
                name="reduce_sample_rate",
                action=reduce_sample_rate,
                applicable_errors=[
                    ErrorType.BUFFER_OVERFLOW,
                    ErrorType.RESOURCE_EXHAUSTION
                ],
                max_attempts=3
            ),
            RecoveryAction(
                name="increase_buffer_size",
                action=increase_buffer_size,
                applicable_errors=[ErrorType.BUFFER_OVERFLOW],
                max_attempts=2
            )
        ])


class SDROptimizer:
    """Optimizes SDR performance based on conditions and requirements"""
    
    def __init__(self):
        self.optimization_history = []
        self.performance_metrics = {}
        
    def optimize_for_signal_type(self, device_id: str, signal_type: str) -> Dict[str, Any]:
        """Optimize configuration for specific signal type"""
        optimizations = {
            'narrowband': {
                'sample_rate': 1e6,
                'bandwidth': 200e3,
                'gain': 30,
                'buffer_size': 4096
            },
            'wideband': {
                'sample_rate': 10e6,
                'bandwidth': 8e6,
                'gain': 20,
                'buffer_size': 16384
            },
            'weak_signal': {
                'sample_rate': 2e6,
                'bandwidth': 1e6,
                'gain': 45,
                'buffer_size': 8192
            },
            'strong_signal': {
                'sample_rate': 5e6,
                'bandwidth': 4e6,
                'gain': 10,
                'buffer_size': 8192
            }
        }
        
        return optimizations.get(signal_type, optimizations['narrowband'])
    
    def optimize_for_environment(self, device_id: str, environment: str) -> Dict[str, Any]:
        """Optimize configuration for specific environment"""
        optimizations = {
            'urban': {
                'gain': 25,  # Lower gain due to interference
                'bandwidth': 2e6,
                'adaptive_gain': True
            },
            'rural': {
                'gain': 40,  # Higher gain for weak signals
                'bandwidth': 5e6,
                'adaptive_gain': False
            },
            'indoor': {
                'gain': 35,
                'bandwidth': 1e6,
                'adaptive_gain': True
            },
            'mobile': {
                'gain': 30,
                'bandwidth': 2e6,
                'adaptive_gain': True,
                'fast_agc': True
            }
        }
        
        return optimizations.get(environment, optimizations['urban'])
    
    def optimize_for_power_consumption(self, device_id: str, 
                                     power_budget: str) -> Dict[str, Any]:
        """Optimize for power consumption"""
        optimizations = {
            'low_power': {
                'sample_rate': 1e6,
                'buffer_size': 2048,
                'processing_interval': 0.1,
                'duty_cycle': 0.5
            },
            'balanced': {
                'sample_rate': 2e6,
                'buffer_size': 4096,
                'processing_interval': 0.05,
                'duty_cycle': 0.8
            },
            'high_performance': {
                'sample_rate': 10e6,
                'buffer_size': 16384,
                'processing_interval': 0.01,
                'duty_cycle': 1.0
            }
        }
        
        return optimizations.get(power_budget, optimizations['balanced'])
    
    def auto_optimize(self, device_id: str, requirements: Dict[str, Any],
                     current_performance: Dict[str, Any]) -> Dict[str, Any]:
        """Automatically optimize based on requirements and current performance"""
        optimizations = {}
        
        # Optimize based on SNR
        current_snr = current_performance.get('snr', 0)
        target_snr = requirements.get('min_snr', 10)
        
        if current_snr < target_snr:
            # Increase gain or reduce bandwidth
            optimizations['gain_adjustment'] = min(5, target_snr - current_snr)
            optimizations['bandwidth_reduction'] = 0.8
        
        # Optimize based on processing load
        cpu_usage = current_performance.get('cpu_usage', 0)
        if cpu_usage > 80:
            # Reduce sample rate or increase buffer size
            optimizations['sample_rate_reduction'] = 0.8
            optimizations['buffer_size_increase'] = 1.5
        
        # Optimize based on error rate
        error_rate = current_performance.get('error_rate', 0)
        if error_rate > 0.1:  # >10% error rate
            # Increase buffer size and reduce sample rate
            optimizations['buffer_size_increase'] = 2.0
            optimizations['sample_rate_reduction'] = 0.7
        
        return optimizations


class ThermalManager:
    """Manages thermal conditions for SDR devices"""
    
    def __init__(self):
        self.temperature_history = {}
        self.thermal_limits = {
            'warning': 70.0,  # Celsius
            'critical': 85.0,
            'shutdown': 95.0
        }
        
    def check_temperature(self, device_id: str) -> Optional[float]:
        """Check device temperature (simulated)"""
        # In practice, this would read from actual temperature sensors
        import random
        
        # Simulate temperature reading
        base_temp = 45.0
        variation = random.uniform(-10, 25)
        temperature = base_temp + variation
        
        # Store in history
        if device_id not in self.temperature_history:
            self.temperature_history[device_id] = []
        
        self.temperature_history[device_id].append({
            'timestamp': datetime.now(),
            'temperature': temperature
        })
        
        # Keep only recent history
        cutoff = datetime.now() - timedelta(hours=1)
        self.temperature_history[device_id] = [
            entry for entry in self.temperature_history[device_id]
            if entry['timestamp'] >= cutoff
        ]
        
        return temperature
    
    def get_thermal_status(self, device_id: str) -> Dict[str, Any]:
        """Get thermal status for device"""
        temperature = self.check_temperature(device_id)
        
        if temperature is None:
            return {'status': 'unknown', 'temperature': None}
        
        if temperature >= self.thermal_limits['shutdown']:
            status = 'shutdown'
            action = 'immediate_shutdown'
        elif temperature >= self.thermal_limits['critical']:
            status = 'critical'
            action = 'reduce_performance'
        elif temperature >= self.thermal_limits['warning']:
            status = 'warning'
            action = 'monitor_closely'
        else:
            status = 'normal'
            action = 'none'
        
        return {
            'status': status,
            'temperature': temperature,
            'recommended_action': action,
            'history': self.temperature_history.get(device_id, [])
        }
    
    def apply_thermal_throttling(self, device_id: str, 
                               current_config: Dict[str, Any]) -> Dict[str, Any]:
        """Apply thermal throttling to configuration"""
        thermal_status = self.get_thermal_status(device_id)
        
        if thermal_status['status'] == 'critical':
            # Aggressive throttling
            return {
                'sample_rate_factor': 0.5,
                'gain_reduction': 10,
                'duty_cycle': 0.3,
                'buffer_size_factor': 0.5
            }
        elif thermal_status['status'] == 'warning':
            # Moderate throttling
            return {
                'sample_rate_factor': 0.8,
                'gain_reduction': 5,
                'duty_cycle': 0.7,
                'buffer_size_factor': 0.8
            }
        else:
            # No throttling needed
            return {
                'sample_rate_factor': 1.0,
                'gain_reduction': 0,
                'duty_cycle': 1.0,
                'buffer_size_factor': 1.0
            }