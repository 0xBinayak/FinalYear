"""
Demo script for mobile client functionality
"""
import asyncio
import logging
import time
from pathlib import Path
import numpy as np

from .mobile_client import MobileClient, MobileTrainingConfig
from .mobile_sdr import MobileSDRManager
from .auth import create_mobile_auth_config
from ..common.signal_models import EnhancedSignalSample, ModulationType, RFParameters


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def demo_device_capabilities():
    """Demonstrate device capabilities detection"""
    logger.info("=== Mobile Device Capabilities Demo ===")
    
    client = MobileClient("demo_client", "http://localhost:8000")
    
    capabilities = client.capabilities
    logger.info(f"Platform: {capabilities.platform}")
    logger.info(f"CPU Cores: {capabilities.cpu_cores}")
    logger.info(f"CPU Frequency: {capabilities.cpu_frequency_ghz:.1f} GHz")
    logger.info(f"Total Memory: {capabilities.total_memory_gb:.1f} GB")
    logger.info(f"Available Memory: {capabilities.available_memory_gb:.1f} GB")
    logger.info(f"Battery Level: {capabilities.battery_level:.1%}" if capabilities.battery_level else "N/A")
    logger.info(f"Is Charging: {capabilities.is_charging}")
    logger.info(f"Storage Available: {capabilities.storage_available_gb:.1f} GB")
    
    client.cleanup()


def demo_battery_management():
    """Demonstrate battery management"""
    logger.info("\n=== Battery Management Demo ===")
    
    client = MobileClient("demo_client", "http://localhost:8000")
    
    # Get battery info
    battery_info = client.battery_manager.get_battery_info()
    logger.info(f"Battery Level: {battery_info['level']:.1%}")
    logger.info(f"Is Charging: {battery_info['is_charging']}")
    logger.info(f"Time Left: {battery_info['time_left_hours']:.1f} hours" if battery_info['time_left_hours'] else "N/A")
    
    # Check training constraints
    can_train, reason = client.battery_manager.can_start_training(client.training_config)
    logger.info(f"Can Start Training: {can_train} - {reason}")
    
    client.cleanup()


def demo_network_conditions():
    """Demonstrate network condition monitoring"""
    logger.info("\n=== Network Conditions Demo ===")
    
    client = MobileClient("demo_client", "http://localhost:8000")
    
    conditions = client.network_manager.get_network_conditions()
    logger.info(f"Bandwidth: {conditions.bandwidth_mbps:.1f} Mbps")
    logger.info(f"Latency: {conditions.latency_ms:.1f} ms")
    logger.info(f"Packet Loss: {conditions.packet_loss_rate:.1%}")
    logger.info(f"Connection Stability: {conditions.connection_stability:.1%}")
    logger.info(f"Is Mobile: {conditions.is_mobile}")
    logger.info(f"Is Metered: {conditions.is_metered}")
    
    # Check if training should be deferred
    should_defer, reason = client.network_manager.should_defer_training()
    logger.info(f"Should Defer Training: {should_defer} - {reason}")
    
    # Estimate transfer time for 1MB model
    transfer_time = client.network_manager.estimate_transfer_time(1024 * 1024)
    logger.info(f"Estimated transfer time for 1MB: {transfer_time:.1f} seconds")
    
    client.cleanup()


def demo_mobile_sdr():
    """Demonstrate mobile SDR functionality"""
    logger.info("\n=== Mobile SDR Demo ===")
    
    sdr_manager = MobileSDRManager()
    
    # Show available SDRs
    available_sdrs = sdr_manager.get_available_sdrs()
    logger.info(f"Available SDR devices: {len(available_sdrs)}")
    
    for sdr_id, capabilities in available_sdrs.items():
        logger.info(f"  {sdr_id}: {capabilities.sdr_type.value}")
        logger.info(f"    Frequency Range: {capabilities.frequency_range[0]/1e6:.1f} - {capabilities.frequency_range[1]/1e6:.1f} MHz")
        logger.info(f"    Sample Rate Range: {capabilities.sample_rate_range[0]/1e6:.1f} - {capabilities.sample_rate_range[1]/1e6:.1f} Msps")
        logger.info(f"    Power Consumption: {capabilities.power_consumption_mw:.0f} mW")
    
    # Select and configure simulated SDR
    sdr_manager.select_sdr('simulated')
    rf_params = RFParameters(915e6, 2e6, 20.0)
    
    if sdr_manager.configure_sdr(rf_params):
        logger.info("SDR configured successfully")
        
        # Collect some samples
        samples_collected = []
        
        def sample_callback(sample):
            samples_collected.append(sample)
            logger.info(f"Collected sample: {sample.modulation_type.value}, SNR: {sample.quality_metrics.snr_db:.1f} dB")
        
        logger.info("Starting signal collection for 2 seconds...")
        sdr_manager.start_collection(rf_params, 2.0, sample_callback)
        
        # Wait for collection to complete
        time.sleep(2.5)
        
        logger.info(f"Collection completed: {len(samples_collected)} samples")
    
    sdr_manager.cleanup()


def demo_dataset_caching():
    """Demonstrate dataset caching"""
    logger.info("\n=== Dataset Caching Demo ===")
    
    # Create temporary cache directory
    cache_dir = Path.home() / ".federated_mobile_demo_cache"
    client = MobileClient("demo_client", "http://localhost:8000", cache_dir)
    
    # Generate synthetic dataset
    logger.info("Generating synthetic dataset...")
    samples = []
    for i in range(50):
        modulation = [ModulationType.QPSK, ModulationType.QAM16, ModulationType.BPSK][i % 3]
        iq_data = np.random.randn(256) + 1j * np.random.randn(256)
        
        sample = EnhancedSignalSample(
            iq_data=iq_data,
            timestamp=time.time(),
            duration=0.001,
            rf_params=RFParameters(915e6, 200e3),
            modulation_type=modulation,
            device_id="demo_device"
        )
        samples.append(sample)
    
    # Cache dataset
    logger.info("Caching dataset...")
    success = client.dataset_cache.cache_dataset("demo_dataset", "demo://synthetic", samples)
    logger.info(f"Dataset cached: {success}")
    
    # Load from cache
    logger.info("Loading from cache...")
    cached_samples = client.dataset_cache.load_cached_dataset("demo_dataset")
    logger.info(f"Loaded {len(cached_samples)} samples from cache")
    
    # Show cache status
    cache_info = client.dataset_cache.metadata
    logger.info(f"Cache size: {cache_info['total_size_bytes'] / (1024*1024):.1f} MB")
    logger.info(f"Cached datasets: {list(cache_info['datasets'].keys())}")
    
    client.cleanup()


def demo_adaptive_training():
    """Demonstrate adaptive training"""
    logger.info("\n=== Adaptive Training Demo ===")
    
    client = MobileClient("demo_client", "http://localhost:8000")
    
    # Show current training parameters
    params = client._get_adaptive_training_params()
    logger.info("Adaptive training parameters:")
    for key, value in params.items():
        logger.info(f"  {key}: {value}")
    
    # Simulate different device conditions
    logger.info("\nSimulating low memory condition...")
    original_memory = client.capabilities.available_memory_gb
    client.capabilities.available_memory_gb = 1.0
    
    low_memory_params = client._get_adaptive_training_params()
    logger.info("Low memory parameters:")
    for key, value in low_memory_params.items():
        logger.info(f"  {key}: {value}")
    
    # Restore original memory
    client.capabilities.available_memory_gb = original_memory
    
    # Simulate low battery
    logger.info("\nSimulating low battery condition...")
    original_battery = client.capabilities.battery_level
    client.capabilities.battery_level = 0.3
    
    low_battery_params = client._get_adaptive_training_params()
    logger.info("Low battery parameters:")
    for key, value in low_battery_params.items():
        logger.info(f"  {key}: {value}")
    
    # Restore original battery
    client.capabilities.battery_level = original_battery
    
    client.cleanup()


def demo_training_workflow():
    """Demonstrate complete training workflow"""
    logger.info("\n=== Training Workflow Demo ===")
    
    client = MobileClient("demo_client", "http://localhost:8000")
    
    # Initialize client
    logger.info("Initializing mobile client...")
    config = {'training_config': {'max_training_time_minutes': 5}}
    
    # Mock successful initialization
    client.is_registered = True
    client.auth_token = "demo_token"
    
    success = client.initialize(config)
    logger.info(f"Client initialized: {success}")
    
    # Check training status
    status = client.get_training_status()
    logger.info("Training status:")
    for key, value in status.items():
        if key not in ['network_conditions', 'optimizations']:  # Skip detailed info
            logger.info(f"  {key}: {value}")
    
    # Generate training data
    logger.info("Generating training data...")
    training_data = client._download_dataset("demo_synthetic_dataset")
    logger.info(f"Generated {len(training_data)} training samples")
    
    # Check if training can start
    can_train, reason = client._can_start_training()
    logger.info(f"Can start training: {can_train} - {reason}")
    
    if can_train:
        logger.info("Starting training simulation...")
        
        # Mock training (don't actually train to avoid long delays)
        training_result = {
            'model_weights': np.random.randn(1000).astype(np.float32).tobytes(),
            'training_time': 2.5,
            'training_loss': 0.8,
            'training_accuracy': 0.75,
            'samples_used': len(training_data),
            'epochs_completed': 3,
            'batch_size': 32
        }
        
        # Create model update
        training_params = client._get_adaptive_training_params()
        model_update = client._create_model_update(training_result, training_params)
        
        logger.info("Training completed!")
        logger.info(f"  Training time: {training_result['training_time']:.1f}s")
        logger.info(f"  Training loss: {training_result['training_loss']:.3f}")
        logger.info(f"  Training accuracy: {training_result['training_accuracy']:.1%}")
        logger.info(f"  Model size: {len(training_result['model_weights'])} bytes")
        logger.info(f"  Samples used: {training_result['samples_used']}")
    
    client.cleanup()


def demo_mobile_optimizations():
    """Demonstrate mobile-specific optimizations"""
    logger.info("\n=== Mobile Optimizations Demo ===")
    
    client = MobileClient("demo_client", "http://localhost:8000")
    
    # Mock initialization
    client.is_registered = True
    client.auth_token = "demo_token"
    client.initialize({})
    
    # Demo adaptive model complexity
    logger.info("Adaptive Model Complexity:")
    adaptation_result = client.force_model_complexity_adaptation()
    logger.info(f"  Selected profile: {adaptation_result['selected_profile']}")
    logger.info(f"  Max parameters: {adaptation_result['profile_details']['max_parameters']:,}")
    logger.info(f"  Batch size: {adaptation_result['profile_details']['batch_size']}")
    logger.info(f"  Estimated training time: {adaptation_result['profile_details']['training_time_estimate_minutes']:.1f} min")
    
    # Demo background training
    logger.info("\nBackground Training:")
    bg_stats = client.background_training_manager.get_training_statistics()
    logger.info(f"  Current state: {bg_stats['current_state']}")
    logger.info(f"  Total sessions: {bg_stats['total_sessions']}")
    logger.info(f"  Background training enabled: {client.background_training_enabled}")
    
    # Demo network handoff
    logger.info("\nNetwork Handoff Status:")
    network_status = client.network_handoff_manager.get_network_status()
    logger.info(f"  Current state: {network_status['current_state']}")
    logger.info(f"  Is stable: {network_status['is_stable']}")
    logger.info(f"  Handoff events: {network_status['handoff_events_count']}")
    
    # Demo incentive system
    logger.info("\nIncentive System:")
    reputation_score = client.get_reputation_score()
    logger.info(f"  Reputation score: {reputation_score:.2f}")
    
    reputation_breakdown = client.incentive_manager.get_reputation_breakdown()
    logger.info(f"  Success rate: {reputation_breakdown['success_rate']:.1%}")
    logger.info(f"  Current streak: {reputation_breakdown['current_streak']}")
    logger.info(f"  Best streak: {reputation_breakdown['best_streak']}")
    
    reward_summary = client.incentive_manager.get_reward_summary()
    logger.info(f"  Total rewards: {reward_summary['total_rewards']:.2f}")
    logger.info(f"  Participation count: {reward_summary['participation_count']}")
    
    # Demo optimization status
    logger.info("\nOptimization Status:")
    opt_status = client.get_optimization_status()
    logger.info(f"  Background training sessions: {opt_status['background_training']['total_sessions']}")
    logger.info(f"  Network stability: {opt_status['network_handoff']['is_stable']}")
    logger.info(f"  Current complexity profile: {opt_status['adaptive_complexity']['current_profile']}")
    
    client.cleanup()


def demo_background_training():
    """Demonstrate background training functionality"""
    logger.info("\n=== Background Training Demo ===")
    
    from .mobile_optimizations import BackgroundTrainingManager, BackgroundTrainingConfig
    
    config = BackgroundTrainingConfig(
        enabled=True,
        max_background_time_minutes=2,  # Short for demo
        min_idle_time_seconds=1
    )
    
    manager = BackgroundTrainingManager(config)
    
    # Set up callback
    training_results = []
    
    def training_callback(params):
        logger.info(f"Background training triggered with params: {params}")
        result = {
            'success': True,
            'training_time': 1.5,
            'model_quality': 0.8,
            'background_mode': params.get('background_mode', False)
        }
        training_results.append(result)
        return result
    
    manager.set_training_callback(training_callback)
    
    # Start monitoring
    manager.start_system_monitoring()
    logger.info("Background training monitoring started")
    
    # Simulate idle time
    logger.info("Simulating idle time...")
    time.sleep(2)
    
    # Check results
    stats = manager.get_training_statistics()
    logger.info(f"Training sessions: {stats['total_sessions']}")
    logger.info(f"Total background time: {stats['total_background_time_minutes']:.1f} minutes")
    logger.info(f"Current state: {stats['current_state']}")
    
    # Cleanup
    manager.stop_system_monitoring()
    logger.info("Background training demo completed")


def demo_authentication():
    """Demonstrate authentication"""
    logger.info("\n=== Authentication Demo ===")
    
    # Create auth configuration
    auth_config = create_mobile_auth_config("http://localhost:8000")
    logger.info(f"Client ID: {auth_config.client_id}")
    logger.info(f"Device Fingerprint: {auth_config.device_fingerprint}")
    
    from .auth import MobileAuthenticator
    authenticator = MobileAuthenticator(auth_config)
    
    # Show device info
    logger.info("Device information:")
    for key, value in authenticator.device_info.items():
        logger.info(f"  {key}: {value}")
    
    # Show authentication status
    status = authenticator.get_authentication_status()
    logger.info("Authentication status:")
    for key, value in status.items():
        logger.info(f"  {key}: {value}")
    
    authenticator.cleanup()


def main():
    """Run all demos"""
    logger.info("Starting Mobile Client Demo")
    logger.info("=" * 50)
    
    try:
        demo_device_capabilities()
        demo_battery_management()
        demo_network_conditions()
        demo_mobile_sdr()
        demo_dataset_caching()
        demo_adaptive_training()
        demo_training_workflow()
        demo_mobile_optimizations()
        demo_background_training()
        demo_authentication()
        
        logger.info("\n" + "=" * 50)
        logger.info("Mobile Client Demo Completed Successfully!")
        
    except Exception as e:
        logger.error(f"Demo failed: {e}")
        raise


if __name__ == "__main__":
    main()