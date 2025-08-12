"""
SDR Client Module

Provides SDR hardware abstraction, signal collection, and federated learning
client functionality for Software Defined Radio devices.
"""

from .device_manager import SDRDeviceManager, SDRHealthMonitor
from .hardware_abstraction import (
    BaseSDRHardware, SDRType, SDRConfig, SDRCapabilities, SignalBuffer,
    RTLSDRHardware, HackRFHardware, USRPHardware, SimulatedSDRHardware
)
from .signal_collector import (
    SignalCollector, StreamingCollector, AdaptiveCollector,
    CollectionConfig, CollectionStats
)
from .error_handling import (
    SDRErrorHandler, SDRError, ErrorType, ErrorSeverity,
    SDROptimizer, ThermalManager
)

__all__ = [
    'SDRDeviceManager',
    'SDRHealthMonitor',
    'BaseSDRHardware',
    'SDRType',
    'SDRConfig',
    'SDRCapabilities',
    'SignalBuffer',
    'RTLSDRHardware',
    'HackRFHardware',
    'USRPHardware',
    'SimulatedSDRHardware',
    'SignalCollector',
    'StreamingCollector',
    'AdaptiveCollector',
    'CollectionConfig',
    'CollectionStats',
    'SDRErrorHandler',
    'SDRError',
    'ErrorType',
    'ErrorSeverity',
    'SDROptimizer',
    'ThermalManager'
]