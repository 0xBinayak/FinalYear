"""
Mobile client implementation for federated learning
"""

from .mobile_client import (
    MobileClient,
    MobileDeviceCapabilities,
    MobileTrainingConfig,
    BatteryManager,
    NetworkManager,
    DatasetCache
)

from .mobile_sdr import (
    MobileSDRManager,
    MobileSDRType,
    MobileSDRCapabilities
)

from .auth import (
    MobileAuthenticator,
    MobileAuthConfig,
    generate_client_id,
    create_mobile_auth_config
)

from .mobile_optimizations import (
    BackgroundTrainingManager,
    BackgroundTrainingConfig,
    NetworkHandoffManager,
    AdaptiveModelComplexityManager,
    IncentiveReputationManager,
    ModelComplexityProfile,
    IncentiveMetrics
)

__all__ = [
    'MobileClient',
    'MobileDeviceCapabilities', 
    'MobileTrainingConfig',
    'BatteryManager',
    'NetworkManager',
    'DatasetCache',
    'MobileSDRManager',
    'MobileSDRType',
    'MobileSDRCapabilities',
    'MobileAuthenticator',
    'MobileAuthConfig',
    'generate_client_id',
    'create_mobile_auth_config',
    'BackgroundTrainingManager',
    'BackgroundTrainingConfig',
    'NetworkHandoffManager',
    'AdaptiveModelComplexityManager',
    'IncentiveReputationManager',
    'ModelComplexityProfile',
    'IncentiveMetrics'
]