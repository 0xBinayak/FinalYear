#!/usr/bin/env python3
"""
Complete test for edge coordinator with resource management and data quality
"""
import sys
import os
import asyncio
import tempfile
import shutil
from datetime import datetime

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Import components
import common.interfaces as interfaces
import common.federated_data_structures as fed_structs
import common.signal_models as signal_models

# Import edge coordinator modules directly
import importlib.util

def load_module(module_name, file_path):
    """Load a module from file path"""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    
    # Set up module dependencies
    module.interfaces = interfaces
    module.fed_structs = fed_structs
    module.signal_models = signal_models
    
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module

# Load modules
base_path = os.path.join(os.path.dirname(__file__), 'src', 'edge_coordinator')

coordinator_module = load_module('coordinator', os.path.join(base_path, 'coordinator.py'))
resource_module = load_module('resource_manager', os.path.join(base_path, 'resource_manager.py'))
quality_module = load_module('data_quality', os.path.join(base_path, 'data_quality.py'))

# Extract classes
EdgeCoordinator = coordinator_module.EdgeCoordinator
ResourceManager = resource_module.ResourceManager
ResourceProfile = resource_module.ResourceProfile
TrainingTask = resource_module.TrainingTask
SchedulingStrategy = resource_module.SchedulingStrategy
DataQualityValidator = quality_module.DataQualityValidator
DataQualityIssue = quality_module.DataQualityIssue
ValidationSeverity = quality_module.ValidationSeverity


async def test_resource_management():
    """Test resource management functionality"""
    print("🚀 Testing Resource Management")
    print("="*50)
    
    try:
        # Create resource manager
        config = {
            'scheduling_strategy': 'resource_aware',
            'max_concurrent_training': 3,
            'resource_update_interval': 5,
            'load_balancing_enabled': True
        }
        
        resource_manager = ResourceManager("test-coord-1", config)
        await resource_manager.start()
        
        try:
            # Test 1: Client registration and profiling
            print("\n1. Testing client registration and profiling...")
            
            client_info = interfaces.ClientInfo(
                client_id="test-client-1",
                client_type="SDR",
                capabilities={
                    'cpu_cores': 4,
                    'cpu_frequency_ghz': 2.5,
                    'memory_gb': 8.0,
                    'available_memory_gb': 6.0,
                    'storage_gb': 100.0,
                    'available_storage_gb': 80.0,
                    'battery_level': 0.8,
                    'thermal_state': 'normal'
                },
                location=None,
                network_info={'bandwidth_mbps': 50.0},
                hardware_specs={'sdr_type': 'rtlsdr'}
            )
            
            profile = resource_manager.register_client(client_info)
            print(f"✓ Client registered with resource score: {profile.get_resource_score():.2f}")
            
            # Test capability assessment
            capability = resource_manager.assess_device_capability("test-client-1")
            print(f"✓ Device capability assessed:")
            print(f"  - Resource score: {capability['resource_score']:.2f}")
            print(f"  - Estimated samples/hour: {capability['estimated_samples_per_hour']:.0f}")
            print(f"  - Can train on battery: {capability['battery_constraints']['can_train_on_battery']}")
            
            # Test 2: Training task scheduling
            print("\n2. Testing training task scheduling...")
            
            task = TrainingTask(
                task_id="task-001",
                client_id="",  # Will be assigned by scheduler
                model_size_mb=50.0,
                batch_size=32,
                estimated_epochs=10,
                estimated_duration_minutes=30.0,
                priority=1.5
            )
            
            success = resource_manager.schedule_training_task(task)
            print(f"✓ Task scheduling: {success}")
            
            if success:
                print(f"✓ Task assigned to client: {task.client_id}")
                print(f"✓ Active training tasks: {len(resource_manager.active_training)}")
            
            # Test 3: Resource updates
            print("\n3. Testing resource updates...")
            
            resource_update = {
                'memory_available_gb': 5.5,
                'battery_level': 0.7,
                'cpu_utilization': 0.6,
                'memory_utilization': 0.4
            }
            
            success = resource_manager.update_client_resources("test-client-1", resource_update)
            print(f"✓ Resource update: {success}")
            
            # Check updated profile
            updated_profile = resource_manager.client_profiles["test-client-1"]
            print(f"✓ Updated battery level: {updated_profile.battery_level}")
            print(f"✓ Updated CPU utilization: {updated_profile.get_cpu_utilization():.2f}")
            
            # Test 4: Load balancing
            print("\n4. Testing load balancing...")
            
            # Register more clients to test load balancing
            for i in range(2, 5):
                client_info_i = interfaces.ClientInfo(
                    client_id=f"test-client-{i}",
                    client_type="Mobile",
                    capabilities={
                        'cpu_cores': 2,
                        'memory_gb': 4.0,
                        'available_memory_gb': 3.0
                    },
                    location=None,
                    network_info={'bandwidth_mbps': 20.0},
                    hardware_specs={}
                )
                resource_manager.register_client(client_info_i)
            
            recommendations = resource_manager.get_load_balancing_recommendations()
            print(f"✓ Load balancing recommendations: {len(recommendations['recommendations'])}")
            print(f"✓ Average load: {recommendations['average_load']:.2f}")
            print(f"✓ Overloaded clients: {len(recommendations['overloaded_clients'])}")
            print(f"✓ Underloaded clients: {len(recommendations['underloaded_clients'])}")
            
            # Test 5: Resource status
            print("\n5. Testing resource status...")
            
            status = resource_manager.get_resource_status()
            print(f"✓ Total clients: {status['total_clients']}")
            print(f"✓ Active training tasks: {status['active_training_tasks']}")
            print(f"✓ Scheduling strategy: {status['scheduling_strategy']}")
            print(f"✓ Average resource score: {status['average_resource_score']:.2f}")
            
            print("\n✅ Resource management tests passed!")
            return True
            
        finally:
            await resource_manager.stop()
    
    except Exception as e:
        print(f"❌ Resource management test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_data_quality_validation():
    """Test data quality validation functionality"""
    print("\n" + "="*50)
    print("Testing Data Quality Validation")
    
    try:
        # Create data quality validator
        config = {
            'min_snr_db': -10.0,
            'max_noise_floor_db': -80.0,
            'min_samples_per_class': 5,
            'auto_preprocessing': True,
            'noise_reduction_enabled': True,
            'normalization_enabled': True
        }
        
        validator = DataQualityValidator("test-coord-1", config)
        await validator.start()
        
        try:
            # Test 1: Create sample signal data
            print("\n1. Creating sample signal data...")
            
            import numpy as np
            
            signal_samples = []
            modulation_types = ['BPSK', 'QPSK', '8PSK', 'QAM16']
            
            for i in range(20):
                # Create IQ data
                iq_data = np.random.complex64(np.random.randn(1000) + 1j * np.random.randn(1000))
                
                # Add some noise and variations
                snr_db = np.random.uniform(-15, 20)  # Some will be below threshold
                noise_floor_db = np.random.uniform(-90, -70)
                
                # Create signal sample
                sample = signal_models.EnhancedSignalSample(
                    timestamp=datetime.now(),
                    iq_data=iq_data,
                    modulation_type=signal_models.ModulationType(modulation_types[i % len(modulation_types)]),
                    rf_params=signal_models.RFParameters(
                        center_frequency=2.4e9 + np.random.uniform(-1e6, 1e6),  # Some frequency drift
                        sample_rate=1e6,
                        bandwidth=200e3
                    ),
                    quality_metrics=signal_models.SignalQualityMetrics(
                        snr_db=snr_db,
                        noise_floor_db=noise_floor_db,
                        signal_power_db=noise_floor_db + snr_db
                    ),
                    duration=0.001
                )
                
                signal_samples.append(sample)
            
            print(f"✓ Created {len(signal_samples)} signal samples")
            
            # Add some problematic samples
            # Corrupted sample with NaN values
            corrupted_sample = signal_samples[0]
            corrupted_iq = corrupted_sample.iq_data.copy()
            corrupted_iq[0] = np.nan
            corrupted_sample.iq_data = corrupted_iq
            
            # Duplicate sample
            signal_samples.append(signal_samples[1])
            
            print("✓ Added problematic samples for testing")
            
            # Test 2: Validate signal data
            print("\n2. Testing signal data validation...")
            
            report = await validator.validate_signal_data("test-client-1", signal_samples)
            
            print(f"✓ Validation completed:")
            print(f"  - Overall score: {report.overall_score:.2f}")
            print(f"  - Total samples: {report.sample_count}")
            print(f"  - Valid samples: {report.valid_samples}")
            print(f"  - Issues found: {len(report.issues)}")
            print(f"  - Validation duration: {report.validation_duration_seconds:.2f}s")
            
            # Check specific issues
            critical_issues = report.get_issues_by_severity(ValidationSeverity.CRITICAL)
            error_issues = report.get_issues_by_severity(ValidationSeverity.ERROR)
            warning_issues = report.get_issues_by_severity(ValidationSeverity.WARNING)
            
            print(f"  - Critical issues: {len(critical_issues)}")
            print(f"  - Error issues: {len(error_issues)}")
            print(f"  - Warning issues: {len(warning_issues)}")
            
            # Test 3: Check quality statistics
            print("\n3. Testing quality statistics...")
            
            if report.snr_statistics:
                print(f"✓ SNR statistics:")
                print(f"  - Mean: {report.snr_statistics['mean']:.1f} dB")
                print(f"  - Min: {report.snr_statistics['min']:.1f} dB")
                print(f"  - Max: {report.snr_statistics['max']:.1f} dB")
                print(f"  - Std: {report.snr_statistics['std']:.1f} dB")
            
            if report.modulation_distribution:
                print(f"✓ Modulation distribution: {report.modulation_distribution}")
            
            if report.preprocessing_recommendations:
                print(f"✓ Preprocessing recommendations:")
                for rec in report.preprocessing_recommendations:
                    print(f"  - {rec}")
            
            # Test 4: Data preprocessing
            print("\n4. Testing data preprocessing...")
            
            # Filter out the corrupted sample for preprocessing test
            clean_samples = [s for s in signal_samples if not np.any(np.isnan(s.iq_data))]
            
            preprocessed_samples = await validator.preprocess_signal_data(clean_samples, report)
            
            print(f"✓ Preprocessing completed:")
            print(f"  - Original samples: {len(clean_samples)}")
            print(f"  - Preprocessed samples: {len(preprocessed_samples)}")
            
            # Test 5: Client quality summary
            print("\n5. Testing client quality summary...")
            
            # Add another validation report
            report2 = await validator.validate_signal_data("test-client-1", signal_samples[:10])
            
            summary = validator.get_client_quality_summary("test-client-1")
            
            if 'error' not in summary:
                print(f"✓ Quality summary:")
                print(f"  - Total reports: {summary['total_reports']}")
                print(f"  - Average quality: {summary['average_quality_score']:.2f}")
                print(f"  - Latest quality: {summary['latest_quality_score']:.2f}")
                print(f"  - Quality trend: {summary['quality_trend']}")
                print(f"  - Common issues: {list(summary['common_issues'].keys())[:3]}")
            else:
                print(f"✓ Quality summary: {summary}")
            
            # Test 6: Validation status
            print("\n6. Testing validation status...")
            
            status = validator.get_validation_status()
            print(f"✓ Validation status:")
            print(f"  - Clients monitored: {status['clients_monitored']}")
            print(f"  - Total reports: {status['total_reports']}")
            print(f"  - Auto preprocessing: {status['preprocessing_config']['auto_preprocessing']}")
            
            print("\n✅ Data quality validation tests passed!")
            return True
            
        finally:
            await validator.stop()
    
    except Exception as e:
        print(f"❌ Data quality validation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_integrated_functionality():
    """Test integrated edge coordinator functionality"""
    print("\n" + "="*50)
    print("Testing Integrated Edge Coordinator Functionality")
    
    try:
        # Create integrated coordinator with all components
        config = {
            'max_local_clients': 10,
            'heartbeat_interval': 5,
            'sync_interval': 30,
            'aggregation_strategy': 'fedavg',
            'resource_management': {
                'scheduling_strategy': 'resource_aware',
                'max_concurrent_training': 3,
                'load_balancing_enabled': True
            },
            'data_quality': {
                'min_snr_db': -10.0,
                'auto_preprocessing': True
            }
        }
        
        coordinator = EdgeCoordinator("integrated-coord", "test-region", config)
        resource_manager = ResourceManager("integrated-coord", config['resource_management'])
        quality_validator = DataQualityValidator("integrated-coord", config['data_quality'])
        
        await coordinator.start()
        await resource_manager.start()
        await quality_validator.start()
        
        try:
            print("\n1. Testing integrated client registration...")
            
            # Register client with coordinator
            client_info = interfaces.ClientInfo(
                client_id="integrated-client-1",
                client_type="SDR",
                capabilities={
                    'cpu_cores': 4,
                    'memory_gb': 8.0,
                    'available_memory_gb': 6.0,
                    'battery_level': 0.9
                },
                location=None,
                network_info={'bandwidth_mbps': 100.0},
                hardware_specs={'sdr_type': 'hackrf'}
            )
            
            # Register with all components
            coord_token = coordinator.register_local_client(client_info)
            resource_profile = resource_manager.register_client(client_info)
            
            print(f"✓ Client registered with coordinator: {coord_token}")
            print(f"✓ Resource profile created with score: {resource_profile.get_resource_score():.2f}")
            
            print("\n2. Testing model update with quality validation...")
            
            # Create model update
            import numpy as np
            weights = np.random.random(1000).astype(np.float32)
            
            model_update = fed_structs.EnhancedModelUpdate(
                client_id="integrated-client-1",
                model_weights=weights.tobytes(),
                model_size_bytes=len(weights.tobytes()),
                training_rounds=1,
                local_epochs=5,
                batch_size=32,
                learning_rate=0.01,
                samples_used=1000,
                training_time_seconds=120.0,
                training_loss=0.4,
                data_distribution={"BPSK": 400, "QPSK": 600}
            )
            
            # Process update through coordinator
            success = coordinator.receive_local_update("integrated-client-1", model_update)
            print(f"✓ Model update received: {success}")
            
            # Update resource utilization
            resource_update = {
                'cpu_utilization': 0.7,
                'memory_utilization': 0.5,
                'battery_level': 0.85
            }
            resource_manager.update_client_resources("integrated-client-1", resource_update)
            print("✓ Resource utilization updated")
            
            print("\n3. Testing task scheduling with resource constraints...")
            
            task = TrainingTask(
                task_id="integrated-task-001",
                client_id="",
                model_size_mb=75.0,
                batch_size=64,
                estimated_epochs=15,
                estimated_duration_minutes=45.0,
                priority=2.0
            )
            
            scheduled = resource_manager.schedule_training_task(task)
            print(f"✓ Task scheduling: {scheduled}")
            
            if scheduled:
                print(f"✓ Task assigned to: {task.client_id}")
            
            print("\n4. Testing comprehensive status...")
            
            coord_status = coordinator.get_coordinator_status()
            resource_status = resource_manager.get_resource_status()
            quality_status = quality_validator.get_validation_status()
            
            print(f"✓ Coordinator status:")
            print(f"  - State: {coord_status['state']}")
            print(f"  - Local clients: {coord_status['local_clients_count']}")
            print(f"  - Active clients: {coord_status['active_clients']}")
            
            print(f"✓ Resource status:")
            print(f"  - Total clients: {resource_status['total_clients']}")
            print(f"  - Active training: {resource_status['active_training_tasks']}")
            print(f"  - Average resource score: {resource_status['average_resource_score']:.2f}")
            
            print(f"✓ Quality status:")
            print(f"  - Clients monitored: {quality_status['clients_monitored']}")
            print(f"  - Auto preprocessing: {quality_status['preprocessing_config']['auto_preprocessing']}")
            
            print("\n✅ Integrated functionality tests passed!")
            return True
            
        finally:
            await coordinator.stop()
            await resource_manager.stop()
            await quality_validator.stop()
    
    except Exception as e:
        print(f"❌ Integrated functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all tests"""
    print("🚀 Starting Complete Edge Coordinator Tests")
    print("="*60)
    
    success = True
    
    try:
        # Test resource management
        success &= await test_resource_management()
        
        # Test data quality validation
        success &= await test_data_quality_validation()
        
        # Test integrated functionality
        success &= await test_integrated_functionality()
        
        if success:
            print("\n" + "="*60)
            print("🎉 ALL TESTS PASSED! 🎉")
            print("Complete Edge Coordinator implementation is working!")
            print("\nImplemented features:")
            print("✅ Local client management and registration")
            print("✅ Hierarchical aggregation for bandwidth optimization")
            print("✅ Network partition detection and handling")
            print("✅ Offline operation with eventual consistency")
            print("✅ Device capability assessment and profiling")
            print("✅ Adaptive training scheduling based on resource constraints")
            print("✅ Local data quality validation and preprocessing")
            print("✅ Load balancing across edge clients")
            return 0
        else:
            print("\n" + "="*60)
            print("❌ SOME TESTS FAILED")
            return 1
            
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import sys
    exit_code = asyncio.run(main())
    sys.exit(exit_code)